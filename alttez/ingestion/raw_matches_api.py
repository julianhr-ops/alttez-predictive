import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from alttez.api.db import get_engine


@dataclass(slots=True)
class RawMatchRecord:
    provider: str
    external_match_id: str
    payload: dict[str, Any]
    fetched_at: datetime | None = None


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reflect_raw_table(engine: Engine, schema: str) -> Table:
    metadata = MetaData(schema=schema)
    metadata.reflect(bind=engine, schema=schema, only=["raw_matches_api"])
    return metadata.tables[f"{schema}.raw_matches_api"]


def upsert_raw_matches(
    engine: Engine,
    records: list[RawMatchRecord],
    schema: str = "public",
) -> int:
    if not records:
        return 0

    raw_table = reflect_raw_table(engine, schema)
    rows = []
    for record in records:
        rows.append(
            {
                "provider": record.provider,
                "external_match_id": record.external_match_id,
                "fetched_at": record.fetched_at or datetime.now(timezone.utc),
                "payload": record.payload,
                "payload_hash": _hash_payload(record.payload),
                "processed": False,
                "processed_at": None,
            }
        )

    with engine.begin() as connection:
        stmt = insert(raw_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "external_match_id"],
            set_={
                "fetched_at": stmt.excluded.fetched_at,
                "payload": stmt.excluded.payload,
                "payload_hash": stmt.excluded.payload_hash,
                "processed": False,
                "processed_at": None,
            },
        )
        connection.execute(stmt)

    return len(rows)


def _extract_items(payload: Any, records_key: str | None) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        if records_key:
            items = payload.get(records_key, [])
            if not isinstance(items, list):
                raise ValueError(f'The key "{records_key}" does not contain a list of records.')
            return [item for item in items if isinstance(item, dict)]

        for candidate_key in ("response", "matches", "data", "fixtures"):
            items = payload.get(candidate_key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]

    raise ValueError("Could not find a list of match-like records in the provided payload.")


def build_records_from_json(
    provider: str,
    payload: Any,
    id_field: str = "id",
    records_key: str | None = None,
) -> list[RawMatchRecord]:
    records = []
    for item in _extract_items(payload, records_key):
        external_id = item.get(id_field)
        if external_id is None:
            raise ValueError(f'Missing id field "{id_field}" in one of the records.')

        records.append(
            RawMatchRecord(
                provider=provider,
                external_match_id=str(external_id),
                payload=item,
            )
        )

    return records


def load_json_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_ingestion(
    provider: str,
    input_json: Path,
    database_url: str | None,
    schema: str,
    id_field: str,
    records_key: str | None,
    dry_run: bool,
) -> None:
    payload = load_json_payload(input_json)
    records = build_records_from_json(provider, payload, id_field=id_field, records_key=records_key)

    print(f"Provider: {provider}")
    print(f"Input file: {input_json}")
    print(f"Records detected: {len(records)}")

    if dry_run:
        return

    engine = get_engine(database_url)
    affected = upsert_raw_matches(engine, records, schema=schema)
    print(f"Raw rows upserted: {affected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load raw match payloads into public.raw_matches_api.")
    parser.add_argument("--provider", required=True, help="Provider name, e.g. api_football or sportmonks.")
    parser.add_argument("--input-json", required=True, help="Path to a local JSON payload file.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--schema", default="public", help="Target PostgreSQL schema.")
    parser.add_argument("--id-field", default="id", help="Field used as the external match identifier.")
    parser.add_argument(
        "--records-key",
        default=None,
        help='Optional top-level key containing the list of records, e.g. "response" or "data".',
    )
    parser.add_argument("--dry-run", action="store_true", help="Only parse and count the payload.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(
        provider=args.provider,
        input_json=Path(args.input_json),
        database_url=args.database_url,
        schema=args.schema,
        id_field=args.id_field,
        records_key=args.records_key,
        dry_run=args.dry_run,
    )
