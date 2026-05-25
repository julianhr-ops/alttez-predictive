import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy import MetaData, Table

from alttez.api.db import get_engine


DEFAULT_CSV_PATH = Path("storage/network/international_results.csv")


def normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def compute_historical_fingerprint(row: pd.Series) -> str:
    payload = "|".join(
        [
            str(row["date"]),
            normalize_name(row["home_team"]),
            normalize_name(row["away_team"]),
            str(int(row["home_score"])),
            str(int(row["away_score"])),
            normalize_name(row["tournament"]),
            normalize_name(row["city"]) if pd.notna(row["city"]) else "",
            normalize_name(row["country"]) if pd.notna(row["country"]) else "",
            str(bool(row["neutral"])),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_historical_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    df["historical_fingerprint"] = df.apply(compute_historical_fingerprint, axis=1)
    return df


def reflect_tables(engine: Engine, schema: str) -> dict[str, Table]:
    metadata = MetaData(schema=schema)
    metadata.reflect(bind=engine, schema=schema, only=["teams", "tournaments", "matches", "match_team_stats"])
    return {
        "teams": metadata.tables[f"{schema}.teams"],
        "tournaments": metadata.tables[f"{schema}.tournaments"],
        "matches": metadata.tables[f"{schema}.matches"],
        "match_team_stats": metadata.tables[f"{schema}.match_team_stats"],
    }


def upsert_catalogs(engine: Engine, teams_table: Table, tournaments_table: Table, df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    team_rows = [
        {"name": name, "normalized_name": normalize_name(name), "is_national_team": True}
        for name in sorted(set(df["home_team"]).union(set(df["away_team"])))
    ]
    tournament_rows = [
        {"name": name, "normalized_name": normalize_name(name)}
        for name in sorted(set(df["tournament"]))
    ]

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
                set_={"name": stmt.excluded.name},
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


def upsert_matches(engine: Engine, matches_table: Table, df: pd.DataFrame, teams_map: dict[str, Any], tournaments_map: dict[str, Any]) -> dict[str, Any]:
    match_rows = []
    for row in df.itertuples(index=False):
        match_rows.append(
            {
                "historical_fingerprint": row.historical_fingerprint,
                "match_date": row.date,
                "season_label": str(row.date.year),
                "status": "finished",
                "tournament_id": tournaments_map[normalize_name(row.tournament)],
                "home_team_id": teams_map[normalize_name(row.home_team)],
                "away_team_id": teams_map[normalize_name(row.away_team)],
                "city": row.city,
                "venue_country_name": row.country,
                "home_score": row.home_score,
                "away_score": row.away_score,
                "is_neutral": bool(row.neutral),
                "source_quality": "historical_csv",
            }
        )

    with engine.begin() as connection:
        if match_rows:
            stmt = insert(matches_table).values(match_rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["historical_fingerprint"],
                set_={
                    "match_date": stmt.excluded.match_date,
                    "season_label": stmt.excluded.season_label,
                    "status": stmt.excluded.status,
                    "tournament_id": stmt.excluded.tournament_id,
                    "home_team_id": stmt.excluded.home_team_id,
                    "away_team_id": stmt.excluded.away_team_id,
                    "city": stmt.excluded.city,
                    "venue_country_name": stmt.excluded.venue_country_name,
                    "home_score": stmt.excluded.home_score,
                    "away_score": stmt.excluded.away_score,
                    "is_neutral": stmt.excluded.is_neutral,
                    "source_quality": stmt.excluded.source_quality,
                },
            )
            connection.execute(stmt)

        match_id_map = {
            row.historical_fingerprint: row.id
            for row in connection.execute(select(matches_table.c.id, matches_table.c.historical_fingerprint)).all()
        }

    return match_id_map


def upsert_match_team_stats(
    engine: Engine,
    stats_table: Table,
    df: pd.DataFrame,
    teams_map: dict[str, Any],
    match_id_map: dict[str, Any],
) -> None:
    stats_rows = []
    for row in df.itertuples(index=False):
        match_id = match_id_map[row.historical_fingerprint]
        home_team_id = teams_map[normalize_name(row.home_team)]
        away_team_id = teams_map[normalize_name(row.away_team)]

        stats_rows.append(
            {
                "match_id": match_id,
                "team_id": home_team_id,
                "opponent_team_id": away_team_id,
                "is_home": True,
                "goals_scored": row.home_score,
                "goals_conceded": row.away_score,
                "source_quality": "historical_csv",
            }
        )
        stats_rows.append(
            {
                "match_id": match_id,
                "team_id": away_team_id,
                "opponent_team_id": home_team_id,
                "is_home": False,
                "goals_scored": row.away_score,
                "goals_conceded": row.home_score,
                "source_quality": "historical_csv",
            }
        )

    with engine.begin() as connection:
        if stats_rows:
            stmt = insert(stats_table).values(stats_rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["match_id", "team_id"],
                set_={
                    "opponent_team_id": stmt.excluded.opponent_team_id,
                    "is_home": stmt.excluded.is_home,
                    "goals_scored": stmt.excluded.goals_scored,
                    "goals_conceded": stmt.excluded.goals_conceded,
                    "source_quality": stmt.excluded.source_quality,
                },
            )
            connection.execute(stmt)


def run_seed(csv_path: Path, database_url: str | None, schema: str, dry_run: bool) -> None:
    df = load_historical_dataframe(csv_path)
    print(f"Historical rows loaded: {len(df)}")
    print(f"Unique teams: {len(set(df['home_team']).union(set(df['away_team'])))}")
    print(f"Unique tournaments: {df['tournament'].nunique()}")

    if dry_run:
        return

    engine = get_engine(database_url)
    tables = reflect_tables(engine, schema)
    teams_map, tournaments_map = upsert_catalogs(engine, tables["teams"], tables["tournaments"], df)
    match_id_map = upsert_matches(engine, tables["matches"], df, teams_map, tournaments_map)
    upsert_match_team_stats(engine, tables["match_team_stats"], df, teams_map, match_id_map)

    print(f"Teams upserted: {len(teams_map)}")
    print(f"Tournaments upserted: {len(tournaments_map)}")
    print(f"Matches upserted: {len(match_id_map)}")
    print(f"Match-team stats upserted: {len(df) * 2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed historical international matches into Supabase/PostgreSQL.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Path to the historical CSV file.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--schema", default="public", help="Target PostgreSQL schema.")
    parser.add_argument("--dry-run", action="store_true", help="Only parse the CSV and print counts.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_seed(
        csv_path=Path(args.csv_path),
        database_url=args.database_url,
        schema=args.schema,
        dry_run=args.dry_run,
    )
