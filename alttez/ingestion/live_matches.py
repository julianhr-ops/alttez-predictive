from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from alttez.api.db import get_engine, read_dataframe
from alttez.ingestion.providers.api_football import ApiFootballProvider
from alttez.ingestion.providers.base import ProviderFetchRequest
from alttez.ingestion.raw_matches_api import RawMatchRecord, upsert_raw_matches


DEFAULT_SCHEMA = "public"


@dataclass(slots=True)
class ScheduledMatch:
    provider: str
    external_match_id: str
    kickoff_at: str | None
    match_date: date
    season_label: str | None
    stage: str | None
    tournament_name: str
    tournament_region: str | None
    home_team: str
    away_team: str
    venue_name: str | None
    city: str | None
    venue_country_name: str | None
    referee_name: str | None
    is_neutral: bool
    status: str
    payload: dict[str, Any]


def normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def reflect_tables(engine: Engine, schema: str = DEFAULT_SCHEMA) -> dict[str, Table]:
    metadata = MetaData(schema=schema)
    metadata.reflect(bind=engine, schema=schema, only=["teams", "tournaments", "matches"])
    return {
        "teams": metadata.tables[f"{schema}.teams"],
        "tournaments": metadata.tables[f"{schema}.tournaments"],
        "matches": metadata.tables[f"{schema}.matches"],
    }


def fetch_api_football_fixtures(target_date: date) -> tuple[list[ScheduledMatch], list[RawMatchRecord]]:
    provider = ApiFootballProvider()
    result = provider.fetch_matches(ProviderFetchRequest(date_from=target_date, date_to=target_date))
    return parse_api_football_records(result.records, result.fetched_at, target_date, provider.provider_name)


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
        tournament_name = league.get("name") or "Unknown Tournament"
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
                tournament_name=tournament_name,
                tournament_region=tournament_region,
                home_team=home_team,
                away_team=away_team,
                venue_name=(fixture.get("venue") or {}).get("name"),
                city=(fixture.get("venue") or {}).get("city"),
                venue_country_name=tournament_region,
                referee_name=fixture.get("referee"),
                is_neutral=False,
                status=status,
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


def upsert_scheduled_matches(
    engine: Engine,
    matches_table: Table,
    matches: Iterable[ScheduledMatch],
    teams_map: dict[str, Any],
    tournaments_map: dict[str, Any],
) -> int:
    rows = []
    for match in matches:
        rows.append(
            {
                "provider": match.provider,
                "external_match_id": match.external_match_id,
                "match_date": match.match_date,
                "kickoff_at": pd.to_datetime(match.kickoff_at) if match.kickoff_at else None,
                "season_label": match.season_label,
                "stage": match.stage,
                "status": match.status,
                "tournament_id": tournaments_map[normalize_name(match.tournament_name)],
                "home_team_id": teams_map[normalize_name(match.home_team)],
                "away_team_id": teams_map[normalize_name(match.away_team)],
                "venue_name": match.venue_name,
                "venue_country_name": match.venue_country_name,
                "city": match.city,
                "referee_name": match.referee_name,
                "home_score": 0,
                "away_score": 0,
                "is_neutral": match.is_neutral,
                "source_quality": "api_raw",
            }
        )

    if not rows:
        return 0

    with engine.begin() as connection:
        stmt = insert(matches_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "external_match_id"],
            set_={
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
            },
        )
        connection.execute(stmt)

    return len(rows)


def ingest_live_matches_for_date(
    target_date: date,
    database_url: str | None = None,
    schema: str = DEFAULT_SCHEMA,
    write_raw: bool = True,
) -> dict[str, int]:
    scheduled_matches, raw_records = fetch_api_football_fixtures(target_date)
    engine = get_engine(database_url)
    tables = reflect_tables(engine, schema)
    teams_map, tournaments_map = upsert_catalogs(engine, tables["teams"], tables["tournaments"], scheduled_matches)

    inserted_raw = 0
    if write_raw and raw_records:
        inserted_raw = upsert_raw_matches(engine, raw_records, schema=schema)

    inserted_matches = upsert_scheduled_matches(
        engine,
        tables["matches"],
        scheduled_matches,
        teams_map,
        tournaments_map,
    )
    return {
        "raw_records": inserted_raw,
        "scheduled_matches": inserted_matches,
        "teams_catalog": len(teams_map),
        "tournaments_catalog": len(tournaments_map),
    }


def list_matches_for_date(
    target_date: date,
    include_statuses: tuple[str, ...] = ("scheduled", "in_progress", "finished"),
    database_url: str | None = None,
) -> pd.DataFrame:
    placeholders = ", ".join(f":status_{index}" for index, _ in enumerate(include_statuses))
    params = {f"status_{index}": status for index, status in enumerate(include_statuses)}
    params["target_date"] = target_date.isoformat()

    sql = f"""
        select
            m.id,
            m.provider,
            m.external_match_id,
            m.match_date,
            m.kickoff_at,
            m.stage,
            m.status,
            m.is_neutral,
            m.venue_name,
            m.city,
            m.venue_country_name,
            tr.name as tournament,
            ht.name as home_team,
            at.name as away_team
        from public.matches as m
        join public.teams as ht
            on ht.id = m.home_team_id
        join public.teams as at
            on at.id = m.away_team_id
        left join public.tournaments as tr
            on tr.id = m.tournament_id
        where m.match_date = :target_date
          and m.status in ({placeholders})
        order by m.kickoff_at nulls last, ht.name, at.name
    """
    return read_dataframe(sql, params=params, database_url=database_url)


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
