from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterable
from datetime import date, datetime

from alttez.api.db import get_engine
from alttez.api.settings import load_local_env
from alttez.ingestion.providers import (
    ApiFootballProvider,
    FootballDataOrgProvider,
    LocalJsonProvider,
    MatchProvider,
    ProviderFetchRequest,
    StatsBombOpenProvider,
)
from alttez.ingestion.raw_matches_api import RawMatchRecord, upsert_raw_matches


logging.basicConfig(level=logging.INFO)


DEFAULT_PROVIDER_ORDER = (
    "api_football",
    "football_data_org",
    "statsbomb_open",
    "local_json",
)


def _provider_registry() -> dict[str, MatchProvider]:
    return {
        "api_football": ApiFootballProvider(),
        "football_data_org": FootballDataOrgProvider(),
        "statsbomb_open": StatsBombOpenProvider(),
        "local_json": LocalJsonProvider(),
    }


def resolve_provider_order(explicit_order: str | None = None) -> list[str]:
    configured = explicit_order or os.getenv("ALTTEZ_PROVIDER_STACK", "")
    if not configured.strip():
        return list(DEFAULT_PROVIDER_ORDER)
    return [item.strip() for item in configured.split(",") if item.strip()]


def build_stack(explicit_order: str | None = None) -> list[MatchProvider]:
    load_local_env()
    registry = _provider_registry()
    stack: list[MatchProvider] = []
    for provider_name in resolve_provider_order(explicit_order):
        provider = registry.get(provider_name)
        if provider is None:
            logging.warning("Skipping unknown provider in stack: %s", provider_name)
            continue
        stack.append(provider)
    return stack


def fetch_from_stack(
    request: ProviderFetchRequest,
    explicit_order: str | None = None,
) -> tuple[list[RawMatchRecord], list[dict[str, str]]]:
    records: list[RawMatchRecord] = []
    audit: list[dict[str, str]] = []

    for provider in build_stack(explicit_order):
        if not provider.is_configured():
            audit.append({"provider": provider.provider_name, "status": "skipped", "detail": "not configured"})
            continue

        try:
            result = provider.fetch_matches(request)
            provider_records = [
                RawMatchRecord(
                    provider=result.provider,
                    external_match_id=_extract_external_id(result.provider, record),
                    payload=record,
                    fetched_at=result.fetched_at,
                )
                for record in result.records
            ]
            records.extend(provider_records)
            audit.append(
                {
                    "provider": provider.provider_name,
                    "status": "ok",
                    "detail": f"{len(provider_records)} records",
                }
            )
        except Exception as exc:  # noqa: BLE001 - provider isolation is intentional here.
            audit.append({"provider": provider.provider_name, "status": "error", "detail": str(exc)})

    return deduplicate_records(records), audit


def deduplicate_records(records: Iterable[RawMatchRecord]) -> list[RawMatchRecord]:
    deduped: dict[tuple[str, str], RawMatchRecord] = {}
    for record in records:
        deduped[(record.provider, record.external_match_id)] = record
    return list(deduped.values())


def _extract_external_id(provider_name: str, payload: dict) -> str:
    if provider_name == "api_football":
        fixture = payload.get("fixture", {})
        if fixture.get("id") is not None:
            return str(fixture["id"])

    if provider_name == "football_data_org" and payload.get("id") is not None:
        return str(payload["id"])

    if provider_name == "statsbomb_open" and payload.get("match_id") is not None:
        return str(payload["match_id"])

    if provider_name == "local_json":
        for key in ("id", "match_id", "fixture_id"):
            if payload.get(key) is not None:
                return str(payload[key])

    for key in ("id", "match_id", "fixture_id"):
        if payload.get(key) is not None:
            return str(payload[key])

    raise ValueError(f"Could not determine external match id for provider {provider_name}.")


def run_provider_stack(
    request: ProviderFetchRequest,
    database_url: str | None,
    schema: str,
    explicit_order: str | None = None,
    dry_run: bool = False,
) -> None:
    records, audit = fetch_from_stack(request, explicit_order=explicit_order)

    for item in audit:
        logging.info("[%s] %s - %s", item["provider"], item["status"], item["detail"])

    logging.info("Total deduplicated raw records: %s", len(records))
    if dry_run:
        return

    engine = get_engine(database_url)
    affected = upsert_raw_matches(engine, records, schema=schema)
    logging.info("Raw rows upserted: %s", affected)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch match payloads from a configured provider stack.")
    parser.add_argument("--date-from", default=None, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--date-to", default=None, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--competition", default=None, help="Competition/league code or provider-specific identifier.")
    parser.add_argument("--season", default=None, help="Season code or year if supported by the provider.")
    parser.add_argument("--provider-order", default=None, help="Comma-separated provider priority override.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--schema", default="public", help="Target PostgreSQL schema.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and audit without writing to the database.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    request = ProviderFetchRequest(
        date_from=_parse_date(args.date_from),
        date_to=_parse_date(args.date_to),
        competition=args.competition,
        season=args.season,
    )
    run_provider_stack(
        request=request,
        database_url=args.database_url,
        schema=args.schema,
        explicit_order=args.provider_order,
        dry_run=args.dry_run,
    )
