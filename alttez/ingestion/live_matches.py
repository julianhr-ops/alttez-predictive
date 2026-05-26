from __future__ import annotations

import argparse
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, inspect as sqlalchemy_inspect, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from alttez.api.db import get_engine
from alttez.api.settings import load_local_env
from alttez.ingestion.certification import build_daily_certification_report, build_raw_match_payload, write_certification_report
from alttez.ingestion.contracts import RawMatchPayload
from alttez.ingestion.provider_stack import build_stack
from alttez.ingestion.providers.base import ProviderFetchRequest
from alttez.ingestion.raw_matches_api import RawMatchRecord, upsert_raw_matches


DEFAULT_SCHEMA = "public"
DEFAULT_LIVE_PROVIDER_ORDER = (
    "api_football",
    "football_data_org",
    "local_json",
)
DEFAULT_FOOTBALL_DATA_COMPETITIONS = (
    "WC",
    "EC",
)
WORLD_CUP_KEYWORDS = (
    "fifa world cup",
    "world cup",
)
FRIENDLY_KEYWORDS = (
    "friendly",
    "friendlies",
    "international friendly",
)
QUALIFIER_KEYWORDS = (
    "qualification",
    "qualifying",
    "qualifier",
)
CONTINENTAL_KEYWORDS = (
    "uefa euro",
    "euro championship",
    "copa america",
    "copa amÃ©rica",
    "nations league",
    "gold cup",
    "africa cup",
    "nations cup",
    "asian cup",
    "aff championship",
    "olympic",
    "confederations",
)
INTERNATIONAL_KEYWORDS = (
    "friendly",
    "friendlies",
    "world cup",
    "qualification",
    "qualifying",
    "uefa euro",
    "euro championship",
    "copa america",
    "copa américa",
    "nations league",
    "gold cup",
    "africa cup",
    "nations cup",
    "asian cup",
    "aff championship",
    "olympic",
    "confederations",
)


logging.basicConfig(level=logging.INFO)


CERTIFICATION_REPORT_DIR = Path("artifacts/certification")


@dataclass(slots=True)
class ScheduledMatch:
    provider: str
    external_match_id: str
    kickoff_at: str | None
    match_date: date
    season_label: str | None
    stage: str | None
    group_label: str | None
    tournament_name: str
    competition_normalized_name: str
    competition_class: str
    tournament_region: str | None
    home_team: str
    away_team: str
    venue_name: str | None
    city: str | None
    venue_country_name: str | None
    referee_name: str | None
    is_neutral: bool
    status: str
    home_score: int
    away_score: int
    payload: dict[str, Any]


def normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_competition_name(value: str | None) -> str:
    return normalize_name(value or "unknown tournament")


def infer_group_label(stage: str | None, payload: dict[str, Any] | None = None) -> str | None:
    payload = payload or {}
    direct_candidates = [
        payload.get("group"),
        ((payload.get("league") or {}).get("group") if isinstance(payload.get("league"), dict) else None),
        ((payload.get("fixture") or {}).get("group") if isinstance(payload.get("fixture"), dict) else None),
    ]
    for candidate in direct_candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    text = (stage or "").strip()
    if not text:
        return None

    patterns = (
        r"(group\s+[a-z0-9]+)",
        r"(grupo\s+[a-z0-9]+)",
    )
    lowered = text.casefold()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1).title()

    return None


def classify_competition(tournament_name: str | None, stage: str | None) -> str:
    haystack = f"{tournament_name or ''} {stage or ''}".casefold()
    if any(keyword in haystack for keyword in FRIENDLY_KEYWORDS):
        return "friendly"
    if any(keyword in haystack for keyword in WORLD_CUP_KEYWORDS):
        return "world_cup"
    if any(keyword in haystack for keyword in QUALIFIER_KEYWORDS):
        return "qualifiers"
    if any(keyword in haystack for keyword in CONTINENTAL_KEYWORDS):
        return "continental"
    return "other_international"


def classify_provider_error(exc: Exception) -> str:
    detail = str(exc).casefold()
    if "not configured" in detail or "requires a competition code" in detail:
        return "config_missing"
    if "401" in detail or "403" in detail or "forbidden" in detail or "unauthorized" in detail:
        return "provider_http_error"
    if "payload" in detail or "unexpected" in detail or "returned errors" in detail:
        return "provider_payload_error"
    if "connection" in detail or "timeout" in detail or "temporarily unavailable" in detail:
        return "provider_http_error"
    return "provider_runtime_error"


def classify_sync_error(exc: Exception) -> str:
    detail = str(exc).casefold()
    if "database url" in detail or "supabase" in detail or "postgres" in detail or "sqlalchemy" in detail:
        return "db_write_error"
    return classify_provider_error(exc)


def reflect_tables(engine: Engine, schema: str = DEFAULT_SCHEMA) -> dict[str, Table]:
    metadata = MetaData(schema=schema)
    inspector = sqlalchemy_inspect(engine)
    existing_tables = set(inspector.get_table_names(schema=schema))
    requested_tables = [table for table in ("teams", "tournaments", "matches", "competition_groups") if table in existing_tables]
    metadata.reflect(bind=engine, schema=schema, only=requested_tables)
    return {table: metadata.tables[f"{schema}.{table}"] for table in requested_tables}


def get_live_provider_order(explicit_order: str | None = None) -> list[str]:
    configured = explicit_order or os.getenv("ALTTEZ_LIVE_PROVIDER_STACK", "")
    if configured.strip():
        return [item.strip() for item in configured.split(",") if item.strip()]
    return list(DEFAULT_LIVE_PROVIDER_ORDER)


def get_football_data_competitions() -> list[str]:
    configured = os.getenv("ALTTEZ_FOOTBALL_DATA_LIVE_COMPETITIONS", "")
    if configured.strip():
        return [item.strip() for item in configured.split(",") if item.strip()]
    return list(DEFAULT_FOOTBALL_DATA_COMPETITIONS)


def build_live_provider_stack(explicit_order: str | None = None):
    return build_stack(",".join(get_live_provider_order(explicit_order)))


def fetch_live_fixtures_from_stack(
    target_date: date,
    explicit_order: str | None = None,
) -> tuple[list[ScheduledMatch], list[RawMatchRecord], str | None, list[dict[str, Any]]]:
    load_local_env(override=True)

    request = ProviderFetchRequest(date_from=target_date, date_to=target_date)
    attempts: list[dict[str, Any]] = []
    configured_empty_provider: str | None = None

    for provider in build_live_provider_stack(explicit_order):
        if provider.provider_name == "statsbomb_open":
            attempts.append(
                {
                    "provider": provider.provider_name,
                    "status": "skipped",
                    "detail": "not used for live agenda",
                    "error_category": None,
                }
            )
            continue

        if not provider.is_configured():
            attempts.append(
                {
                    "provider": provider.provider_name,
                    "status": "skipped",
                    "detail": "not configured",
                    "error_category": "config_missing",
                }
            )
            continue

        try:
            if provider.provider_name == "football_data_org":
                scheduled_matches, raw_records = _fetch_football_data_daily(provider, target_date)
            else:
                result = provider.fetch_matches(request)
                scheduled_matches, raw_records = parse_provider_records(
                    provider.provider_name,
                    result.records,
                    result.fetched_at,
                    target_date,
                )
            scheduled_matches, raw_records = filter_international_matches(scheduled_matches, raw_records)
        except Exception as exc:  # noqa: BLE001 - provider isolation is intentional.
            attempts.append(
                {
                    "provider": provider.provider_name,
                    "status": "error",
                    "detail": str(exc),
                    "error_category": classify_provider_error(exc),
                }
            )
            continue

        attempts.append(
            {
                "provider": provider.provider_name,
                "status": "ok",
                "detail": f"{len(scheduled_matches)} fixtures",
                "error_category": "no_fixtures_found" if not scheduled_matches else None,
            }
        )

        if scheduled_matches:
            return scheduled_matches, raw_records, provider.provider_name, attempts

        configured_empty_provider = provider.provider_name

    return [], [], configured_empty_provider, attempts


def parse_provider_records(
    provider_name: str,
    records: Iterable[dict[str, Any]],
    fetched_at,
    target_date: date,
) -> tuple[list[ScheduledMatch], list[RawMatchRecord]]:
    if provider_name == "api_football":
        return parse_api_football_records(records, fetched_at, target_date, provider_name)
    if provider_name == "football_data_org":
        return parse_football_data_records(records, fetched_at, target_date, provider_name)
    if provider_name == "local_json":
        return parse_local_json_records(records, fetched_at, target_date, provider_name)
    raise ValueError(f"Unsupported live provider for agenda sync: {provider_name}")


def parse_api_football_records(
    records: Iterable[dict[str, Any]],
    fetched_at,
    target_date: date,
    provider_name: str = "api_football",
) -> tuple[list[ScheduledMatch], list[RawMatchRecord]]:
    scheduled_matches: list[ScheduledMatch] = []
    raw_records: list[RawMatchRecord] = []

    for record in records:
        fixture = record.get("fixture", {})
        league = record.get("league", {})
        teams = record.get("teams", {})
        goals = record.get("goals", {})
        status_info = fixture.get("status", {})

        external_id = str(fixture.get("id"))
        kickoff_at = fixture.get("date")
        match_day = pd.to_datetime(kickoff_at).date() if kickoff_at else target_date
        stage = league.get("round")
        group_label = infer_group_label(stage, league)
        tournament_name = league.get("name") or "Unknown Tournament"
        competition_normalized_name = normalize_competition_name(tournament_name)
        competition_class = classify_competition(tournament_name, stage)
        tournament_region = league.get("country")
        home_team = (teams.get("home") or {}).get("name")
        away_team = (teams.get("away") or {}).get("name")

        if not home_team or not away_team:
            continue

        status = map_api_football_status(status_info.get("short"), goals)
        scheduled_matches.append(
            ScheduledMatch(
                provider=provider_name,
                external_match_id=external_id,
                kickoff_at=kickoff_at,
                match_date=match_day,
                season_label=str(league.get("season")) if league.get("season") is not None else None,
                stage=stage,
                group_label=group_label,
                tournament_name=tournament_name,
                competition_normalized_name=competition_normalized_name,
                competition_class=competition_class,
                tournament_region=tournament_region,
                home_team=home_team,
                away_team=away_team,
                venue_name=(fixture.get("venue") or {}).get("name"),
                city=(fixture.get("venue") or {}).get("city"),
                venue_country_name=tournament_region,
                referee_name=fixture.get("referee"),
                is_neutral=False,
                status=status,
                home_score=int(goals.get("home") or 0),
                away_score=int(goals.get("away") or 0),
                payload=record,
            )
        )

        raw_records.append(
            RawMatchRecord(
                provider=provider_name,
                external_match_id=external_id,
                payload=record,
                fetched_at=fetched_at,
            )
        )

    return scheduled_matches, raw_records


def parse_football_data_records(
    records: Iterable[dict[str, Any]],
    fetched_at,
    target_date: date,
    provider_name: str = "football_data_org",
) -> tuple[list[ScheduledMatch], list[RawMatchRecord]]:
    scheduled_matches: list[ScheduledMatch] = []
    raw_records: list[RawMatchRecord] = []

    for record in records:
        external_id = record.get("id")
        if external_id is None:
            continue

        kickoff_at = record.get("utcDate")
        match_day = pd.to_datetime(kickoff_at).date() if kickoff_at else target_date
        home_team = (record.get("homeTeam") or {}).get("name")
        away_team = (record.get("awayTeam") or {}).get("name")
        competition = record.get("competition") or {}
        area = competition.get("area") or {}
        full_time = ((record.get("score") or {}).get("fullTime") or {})

        stage = record.get("stage") or record.get("group")
        group_label = infer_group_label(record.get("group") or stage, record)
        tournament_name = competition.get("name") or "Unknown Tournament"

        if not home_team or not away_team:
            continue

        scheduled_matches.append(
            ScheduledMatch(
                provider=provider_name,
                external_match_id=str(external_id),
                kickoff_at=kickoff_at,
                match_date=match_day,
                season_label=(
                    str((record.get("season") or {}).get("id"))
                    if (record.get("season") or {}).get("id") is not None
                    else None
                ),
                stage=stage,
                group_label=group_label,
                tournament_name=tournament_name,
                competition_normalized_name=normalize_competition_name(tournament_name),
                competition_class=classify_competition(tournament_name, stage),
                tournament_region=area.get("name"),
                home_team=home_team,
                away_team=away_team,
                venue_name=record.get("venue"),
                city=None,
                venue_country_name=area.get("name"),
                referee_name=None,
                is_neutral=False,
                status=map_football_data_status(record.get("status")),
                home_score=int(full_time.get("home") or 0),
                away_score=int(full_time.get("away") or 0),
                payload=record,
            )
        )
        raw_records.append(
            RawMatchRecord(
                provider=provider_name,
                external_match_id=str(external_id),
                payload=record,
                fetched_at=fetched_at,
            )
        )

    return scheduled_matches, raw_records


def parse_local_json_records(
    records: Iterable[dict[str, Any]],
    fetched_at,
    target_date: date,
    provider_name: str = "local_json",
) -> tuple[list[ScheduledMatch], list[RawMatchRecord]]:
    records_list = [record for record in records if isinstance(record, dict)]
    if not records_list:
        return [], []

    sample = records_list[0]
    if "fixture" in sample and "teams" in sample:
        return parse_api_football_records(records_list, fetched_at, target_date, provider_name)
    if "homeTeam" in sample and "awayTeam" in sample:
        return parse_football_data_records(records_list, fetched_at, target_date, provider_name)

    scheduled_matches: list[ScheduledMatch] = []
    raw_records: list[RawMatchRecord] = []
    for record in records_list:
        external_id = record.get("id") or record.get("match_id") or record.get("fixture_id")
        home_team = record.get("home_team")
        away_team = record.get("away_team")
        if external_id is None or not home_team or not away_team:
            continue

        kickoff_at = record.get("kickoff_at") or record.get("utcDate")
        match_day = pd.to_datetime(kickoff_at).date() if kickoff_at else target_date
        tournament_name = record.get("tournament") or "Unknown Tournament"
        stage = record.get("stage")
        group_label = infer_group_label(record.get("group_label") or stage, record)
        scheduled_matches.append(
            ScheduledMatch(
                provider=provider_name,
                external_match_id=str(external_id),
                kickoff_at=kickoff_at,
                match_date=match_day,
                season_label=record.get("season_label"),
                stage=stage,
                group_label=group_label,
                tournament_name=tournament_name,
                competition_normalized_name=normalize_competition_name(tournament_name),
                competition_class=classify_competition(tournament_name, stage),
                tournament_region=record.get("country"),
                home_team=home_team,
                away_team=away_team,
                venue_name=record.get("venue_name"),
                city=record.get("city"),
                venue_country_name=record.get("country"),
                referee_name=record.get("referee_name"),
                is_neutral=bool(record.get("is_neutral", False)),
                status=record.get("status") or "scheduled",
                home_score=int(record.get("home_score") or 0),
                away_score=int(record.get("away_score") or 0),
                payload=record,
            )
        )
        raw_records.append(
            RawMatchRecord(
                provider=provider_name,
                external_match_id=str(external_id),
                payload=record,
                fetched_at=fetched_at,
            )
        )
    return scheduled_matches, raw_records


def _build_raw_payloads(
    scheduled_matches: Iterable[ScheduledMatch],
    raw_records: Iterable[RawMatchRecord],
) -> list[RawMatchPayload]:
    raw_record_map = {
        (record.provider, record.external_match_id): record
        for record in raw_records
    }
    raw_payloads: list[RawMatchPayload] = []

    for match in scheduled_matches:
        raw_record = raw_record_map.get((match.provider, match.external_match_id))
        if raw_record is None:
            raw_record = RawMatchRecord(
                provider=match.provider,
                external_match_id=match.external_match_id,
                payload=match.payload,
                fetched_at=datetime.now(timezone.utc),
            )

        try:
            raw_payloads.append(
                build_raw_match_payload(
                    match,
                    raw_record,
                    source_payload=raw_record.payload,
                )
            )
        except Exception as exc:  # noqa: BLE001 - we preserve the rest of the batch.
            logging.warning(
                "Skipping raw certification payload for provider=%s match_id=%s detail=%s",
                match.provider,
                match.external_match_id,
                exc,
            )

    return raw_payloads


def _certify_finished_batch(
    scheduled_matches: Iterable[ScheduledMatch],
    raw_records: Iterable[RawMatchRecord],
    target_date: date,
    provider_used: str | None,
) -> dict[str, Any]:
    raw_payloads = _build_raw_payloads(scheduled_matches, raw_records)
    finished_payloads = [payload for payload in raw_payloads if payload.status == "finished"]
    batch_id = f"{target_date.isoformat()}_{provider_used or 'none'}"

    if not finished_payloads:
        return {
            "status": "skipped",
            "batch_id": batch_id,
            "report_path": None,
            "total_records": 0,
            "certified_records": 0,
            "rejected_records": 0,
            "coverage_pct": 0.0,
            "quarantine_reason": "No finished matches were available for certification.",
        }

    report = build_daily_certification_report(
        finished_payloads,
        batch_id=batch_id,
        target_date=target_date,
        provider_used=provider_used,
    )
    report_path = write_certification_report(report, CERTIFICATION_REPORT_DIR)

    return {
        "status": report.certification_status,
        "batch_id": report.batch_id,
        "report_path": str(report_path),
        "total_records": report.total_records,
        "certified_records": report.certified_records,
        "rejected_records": report.rejected_records,
        "coverage_pct": report.coverage_pct,
        "quarantine_reason": report.quarantine_reason,
    }


def map_api_football_status(short_status: str | None, goals: dict[str, Any]) -> str:
    if short_status in {"FT", "AET", "PEN"}:
        return "finished"
    if short_status in {"PST", "CANC", "ABD", "AWD", "WO"}:
        return "cancelled"
    if short_status in {"1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"}:
        return "in_progress"
    if short_status in {"NS", "TBD"}:
        return "scheduled"

    if goals and goals.get("home") is not None and goals.get("away") is not None:
        return "finished"
    return "scheduled"


def map_football_data_status(status: str | None) -> str:
    if status in {"FINISHED"}:
        return "finished"
    if status in {"POSTPONED", "CANCELLED", "SUSPENDED"}:
        return "cancelled"
    if status in {"IN_PLAY", "PAUSED"}:
        return "in_progress"
    return "scheduled"


def filter_international_matches(
    scheduled_matches: list[ScheduledMatch],
    raw_records: list[RawMatchRecord],
) -> tuple[list[ScheduledMatch], list[RawMatchRecord]]:
    match_ids = {
        match.external_match_id
        for match in scheduled_matches
        if is_international_match(match)
    }
    filtered_matches = [match for match in scheduled_matches if match.external_match_id in match_ids]
    filtered_raw = [record for record in raw_records if record.external_match_id in match_ids]
    return filtered_matches, filtered_raw


def is_international_match(match: ScheduledMatch) -> bool:
    if match.provider in {"football_data_org", "local_json"}:
        return True

    tournament_name = (match.tournament_name or "").casefold()
    stage = (match.stage or "").casefold()
    haystack = f"{tournament_name} {stage}"
    return any(keyword in haystack for keyword in INTERNATIONAL_KEYWORDS)


def _fetch_football_data_daily(provider, target_date: date) -> tuple[list[ScheduledMatch], list[RawMatchRecord]]:
    all_matches: list[ScheduledMatch] = []
    all_raw_records: list[RawMatchRecord] = []
    fetched_ids: set[str] = set()

    for competition in get_football_data_competitions():
        result = provider.fetch_matches(
            ProviderFetchRequest(
                date_from=target_date,
                date_to=target_date,
                competition=competition,
            )
        )
        scheduled_matches, raw_records = parse_football_data_records(
            result.records,
            result.fetched_at,
            target_date,
            provider.provider_name,
        )
        for match in scheduled_matches:
            if match.external_match_id in fetched_ids:
                continue
            fetched_ids.add(match.external_match_id)
            all_matches.append(match)
        for raw_record in raw_records:
            if raw_record.external_match_id in fetched_ids:
                # If the match was already included above, we still want one raw payload entry.
                pass
            all_raw_records.append(raw_record)

    deduped_raw = {
        (record.provider, record.external_match_id): record for record in all_raw_records
    }
    return all_matches, list(deduped_raw.values())


def _optional_match_columns(matches_table: Table) -> set[str]:
    return set(matches_table.c.keys())


def upsert_catalogs(
    engine: Engine,
    teams_table: Table,
    tournaments_table: Table,
    matches: Iterable[ScheduledMatch],
) -> tuple[dict[str, Any], dict[str, Any]]:
    team_rows = []
    tournament_rows = []

    seen_teams = set()
    seen_tournaments = set()

    for match in matches:
        for team_name in (match.home_team, match.away_team):
            normalized = normalize_name(team_name)
            if normalized in seen_teams:
                continue
            team_rows.append(
                {
                    "name": team_name,
                    "normalized_name": normalized,
                    "is_national_team": True,
                }
            )
            seen_teams.add(normalized)

        normalized_tournament = normalize_name(match.tournament_name)
        if normalized_tournament not in seen_tournaments:
            tournament_rows.append(
                {
                    "name": match.tournament_name,
                    "normalized_name": normalized_tournament,
                    "competition_type": "international",
                    "region": match.tournament_region,
                }
            )
            seen_tournaments.add(normalized_tournament)

    with engine.begin() as connection:
        if team_rows:
            stmt = insert(teams_table).values(team_rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["normalized_name"],
                set_={"name": stmt.excluded.name, "is_national_team": stmt.excluded.is_national_team},
            )
            connection.execute(stmt)

        if tournament_rows:
            stmt = insert(tournaments_table).values(tournament_rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["normalized_name"],
                set_={
                    "name": stmt.excluded.name,
                    "competition_type": stmt.excluded.competition_type,
                    "region": stmt.excluded.region,
                },
            )
            connection.execute(stmt)

        teams_map = {
            row.normalized_name: row.id
            for row in connection.execute(select(teams_table.c.id, teams_table.c.normalized_name)).all()
        }
        tournaments_map = {
            row.normalized_name: row.id
            for row in connection.execute(select(tournaments_table.c.id, tournaments_table.c.normalized_name)).all()
        }

    return teams_map, tournaments_map


def upsert_competition_groups(
    engine: Engine,
    competition_groups_table: Table | None,
    matches: Iterable[ScheduledMatch],
    tournaments_map: dict[str, Any],
) -> dict[tuple[Any, str | None, str | None, str], Any]:
    if competition_groups_table is None:
        return {}

    rows = []
    seen = set()
    for match in matches:
        if not match.group_label:
            continue
        group_key = (
            tournaments_map[normalize_name(match.tournament_name)],
            match.season_label,
            match.stage,
            match.group_label,
        )
        if group_key in seen:
            continue
        seen.add(group_key)
        rows.append(
            {
                "tournament_id": group_key[0],
                "season_label": group_key[1],
                "stage": group_key[2],
                "group_label": group_key[3],
                "competition_class": match.competition_class,
                "competition_normalized_name": match.competition_normalized_name,
            }
        )

    if not rows:
        return {}

    with engine.begin() as connection:
        stmt = insert(competition_groups_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["tournament_id", "season_label", "stage", "group_label"],
            set_={
                "competition_class": stmt.excluded.competition_class,
                "competition_normalized_name": stmt.excluded.competition_normalized_name,
            },
        )
        connection.execute(stmt)

        group_rows = connection.execute(
            select(
                competition_groups_table.c.id,
                competition_groups_table.c.tournament_id,
                competition_groups_table.c.season_label,
                competition_groups_table.c.stage,
                competition_groups_table.c.group_label,
            )
        ).all()

    return {
        (row.tournament_id, row.season_label, row.stage, row.group_label): row.id
        for row in group_rows
    }


def upsert_scheduled_matches(
    engine: Engine,
    matches_table: Table,
    matches: Iterable[ScheduledMatch],
    teams_map: dict[str, Any],
    tournaments_map: dict[str, Any],
    competition_group_map: dict[tuple[Any, str | None, str | None, str], Any] | None = None,
) -> int:
    competition_group_map = competition_group_map or {}
    match_columns = _optional_match_columns(matches_table)
    rows = []
    for match in matches:
        tournament_id = tournaments_map[normalize_name(match.tournament_name)]
        row = {
            "provider": match.provider,
            "external_match_id": match.external_match_id,
            "match_date": match.match_date,
            "kickoff_at": pd.to_datetime(match.kickoff_at) if match.kickoff_at else None,
            "season_label": match.season_label,
            "stage": match.stage,
            "status": match.status,
            "tournament_id": tournament_id,
            "home_team_id": teams_map[normalize_name(match.home_team)],
            "away_team_id": teams_map[normalize_name(match.away_team)],
            "venue_name": match.venue_name,
            "venue_country_name": match.venue_country_name,
            "city": match.city,
            "referee_name": match.referee_name,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "is_neutral": match.is_neutral,
            "source_quality": "api_raw",
        }
        if "competition_class" in match_columns:
            row["competition_class"] = match.competition_class
        if "competition_normalized_name" in match_columns:
            row["competition_normalized_name"] = match.competition_normalized_name
        if "group_label" in match_columns:
            row["group_label"] = match.group_label
        if "competition_group_id" in match_columns and match.group_label:
            row["competition_group_id"] = competition_group_map.get(
                (tournament_id, match.season_label, match.stage, match.group_label)
            )
        rows.append(row)

    if not rows:
        return 0

    with engine.begin() as connection:
        stmt = insert(matches_table).values(rows)
        update_map = {
            "match_date": stmt.excluded.match_date,
            "kickoff_at": stmt.excluded.kickoff_at,
            "season_label": stmt.excluded.season_label,
            "stage": stmt.excluded.stage,
            "status": stmt.excluded.status,
            "tournament_id": stmt.excluded.tournament_id,
            "home_team_id": stmt.excluded.home_team_id,
            "away_team_id": stmt.excluded.away_team_id,
            "venue_name": stmt.excluded.venue_name,
            "venue_country_name": stmt.excluded.venue_country_name,
            "city": stmt.excluded.city,
            "referee_name": stmt.excluded.referee_name,
            "is_neutral": stmt.excluded.is_neutral,
            "source_quality": stmt.excluded.source_quality,
        }
        if "competition_class" in match_columns:
            update_map["competition_class"] = stmt.excluded.competition_class
        if "competition_normalized_name" in match_columns:
            update_map["competition_normalized_name"] = stmt.excluded.competition_normalized_name
        if "group_label" in match_columns:
            update_map["group_label"] = stmt.excluded.group_label
        if "competition_group_id" in match_columns:
            update_map["competition_group_id"] = stmt.excluded.competition_group_id
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "external_match_id"],
            set_=update_map,
        )
        connection.execute(stmt)

    return len(rows)


def ingest_live_matches_for_date(
    target_date: date,
    database_url: str | None = None,
    schema: str = DEFAULT_SCHEMA,
    write_raw: bool = True,
) -> dict[str, Any]:
    load_local_env(override=True)

    scheduled_matches, raw_records, provider_used, attempts = fetch_live_fixtures_from_stack(target_date)

    if not scheduled_matches:
        any_provider_ok = any(item["status"] == "ok" for item in attempts)
        status = "empty" if any_provider_ok else "error"
        error_detail = None
        error_category = "no_fixtures_found" if status == "empty" else "config_missing"
        if status == "error":
            categories = [item.get("error_category") for item in attempts if item.get("error_category")]
            if categories:
                error_category = categories[0]
            error_detail = "; ".join(
                f"{item['provider']}: {item['detail']}"
                for item in attempts
                if item["status"] in {"error", "skipped"}
            ) or "No live providers are configured for agenda sync."
        return {
            "status": status,
            "provider_used": provider_used,
            "attempted_providers": attempts,
            "scheduled_matches": 0,
            "raw_records": 0,
            "teams_catalog": 0,
            "tournaments_catalog": 0,
            "competition_groups_catalog": 0,
            "certification_status": "skipped",
            "certification_report_path": None,
            "certification_batch_id": None,
            "certification_total_records": 0,
            "certification_certified_records": 0,
            "certification_rejected_records": 0,
            "certification_coverage_pct": 0.0,
            "certification_quarantine_reason": "No scheduled matches were available for certification.",
            "error_category": error_category,
            "error": error_detail,
        }

    try:
        engine = get_engine(database_url)
        tables = reflect_tables(engine, schema)
        teams_map, tournaments_map = upsert_catalogs(engine, tables["teams"], tables["tournaments"], scheduled_matches)

        inserted_raw = 0
        if write_raw and raw_records:
            inserted_raw = upsert_raw_matches(engine, raw_records, schema=schema)

        competition_group_map = upsert_competition_groups(
            engine,
            tables.get("competition_groups"),
            scheduled_matches,
            tournaments_map,
        )
        inserted_matches = upsert_scheduled_matches(
            engine,
            tables["matches"],
            scheduled_matches,
            teams_map,
            tournaments_map,
            competition_group_map=competition_group_map,
        )
        certification_result = _certify_finished_batch(
            scheduled_matches,
            raw_records,
            target_date,
            provider_used,
        )
    except Exception as exc:  # noqa: BLE001 - reporting should preserve dashboard uptime.
        return {
            "status": "error",
            "provider_used": provider_used,
            "attempted_providers": attempts,
            "scheduled_matches": 0,
            "raw_records": 0,
            "teams_catalog": 0,
            "tournaments_catalog": 0,
            "competition_groups_catalog": 0,
            "certification_status": "error",
            "certification_report_path": None,
            "certification_batch_id": None,
            "certification_total_records": 0,
            "certification_certified_records": 0,
            "certification_rejected_records": 0,
            "certification_coverage_pct": 0.0,
            "certification_quarantine_reason": None,
            "error_category": classify_sync_error(exc),
            "error": str(exc),
        }
    return {
        "status": "success",
        "provider_used": provider_used,
        "attempted_providers": attempts,
        "raw_records": inserted_raw,
        "scheduled_matches": inserted_matches,
        "teams_catalog": len(teams_map),
        "tournaments_catalog": len(tournaments_map),
        "competition_groups_catalog": len(competition_group_map),
        "certification_status": certification_result["status"],
        "certification_report_path": certification_result["report_path"],
        "certification_batch_id": certification_result["batch_id"],
        "certification_total_records": certification_result["total_records"],
        "certification_certified_records": certification_result["certified_records"],
        "certification_rejected_records": certification_result["rejected_records"],
        "certification_coverage_pct": certification_result["coverage_pct"],
        "certification_quarantine_reason": certification_result["quarantine_reason"],
        "error_category": None,
        "error": None,
    }


def list_matches_for_date(
    target_date: date,
    include_statuses: tuple[str, ...] = ("scheduled", "in_progress", "finished"),
    database_url: str | None = None,
) -> pd.DataFrame:
    engine = get_engine(database_url)
    inspector = sqlalchemy_inspect(engine)
    match_columns = {column["name"] for column in inspector.get_columns("matches", schema=DEFAULT_SCHEMA)}
    group_columns = (
        {column["name"] for column in inspector.get_columns("competition_groups", schema=DEFAULT_SCHEMA)}
        if "competition_groups" in inspector.get_table_names(schema=DEFAULT_SCHEMA)
        else set()
    )
    placeholders = ", ".join(f":status_{index}" for index, _ in enumerate(include_statuses))
    params = {f"status_{index}": status for index, status in enumerate(include_statuses)}
    params["target_date"] = target_date.isoformat()

    select_fields = [
        "m.id",
        "m.provider",
        "m.external_match_id",
        "m.match_date",
        "m.kickoff_at",
        "m.stage",
        "m.status",
        "m.is_neutral",
        "m.venue_name",
        "m.city",
        "m.venue_country_name",
        "tr.name as tournament",
        "ht.name as home_team",
        "at.name as away_team",
    ]
    if "season_label" in match_columns:
        select_fields.append("m.season_label")
    else:
        select_fields.append("null::varchar as season_label")
    if "competition_class" in match_columns:
        select_fields.append("m.competition_class")
    else:
        select_fields.append("null::varchar as competition_class")
    if "competition_normalized_name" in match_columns:
        select_fields.append("m.competition_normalized_name")
    else:
        select_fields.append("tr.normalized_name as competition_normalized_name")
    if "group_label" in match_columns:
        select_fields.append("m.group_label")
    elif "group_label" in group_columns:
        select_fields.append("cg.group_label")
    else:
        select_fields.append("null::varchar as group_label")

    join_group_sql = ""
    if "competition_group_id" in match_columns and group_columns:
        join_group_sql = """
        left join public.competition_groups as cg
            on cg.id = m.competition_group_id
        """

    sql = f"""
        select
            {", ".join(select_fields)}
        from public.matches as m
        join public.teams as ht
            on ht.id = m.home_team_id
        join public.teams as at
            on at.id = m.away_team_id
        left join public.tournaments as tr
            on tr.id = m.tournament_id
        {join_group_sql}
        where m.match_date = :target_date
          and m.status in ({placeholders})
        order by m.kickoff_at nulls last, ht.name, at.name
    """
    with engine.connect() as connection:
        return pd.read_sql(text(sql), con=connection, params=params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and upsert live international matches for a given date.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Target PostgreSQL schema.")
    parser.add_argument("--skip-raw", action="store_true", help="Do not write provider payloads into raw_matches_api.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = ingest_live_matches_for_date(
        target_date=pd.to_datetime(args.date).date(),
        database_url=args.database_url,
        schema=args.schema,
        write_raw=not args.skip_raw,
    )
    print(result)
