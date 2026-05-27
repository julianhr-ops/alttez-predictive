from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, bindparam, select, update
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.db import get_engine
from alttez.ingestion.seed_historical import normalize_name


DEFAULT_CSV_PATH = Path("storage/network/klement_macro.csv")


def load_macro_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"team_name", "gdp_per_capita", "confederation_code"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"El CSV macro no tiene las columnas requeridas: {sorted(missing)}")

    working_df = df.copy()
    if "normalized_name" not in working_df.columns:
        working_df["normalized_name"] = working_df["team_name"].map(normalize_name)
    else:
        working_df["normalized_name"] = working_df["normalized_name"].fillna("").map(
            lambda value: normalize_name(str(value)) if str(value).strip() else ""
        )
        missing_mask = working_df["normalized_name"].eq("")
        working_df.loc[missing_mask, "normalized_name"] = working_df.loc[missing_mask, "team_name"].map(normalize_name)

    working_df["gdp_per_capita"] = pd.to_numeric(working_df["gdp_per_capita"], errors="coerce")
    working_df["confederation_code"] = (
        working_df["confederation_code"].fillna("").astype(str).str.strip().str.upper()
    )
    return working_df


def reflect_teams_table(engine: Engine, schema: str) -> Table:
    metadata = MetaData(schema=schema)
    metadata.reflect(bind=engine, schema=schema, only=["teams"])
    return metadata.tables[f"{schema}.teams"]


def build_seed_summary(df: pd.DataFrame, known_teams: set[str]) -> dict[str, Any]:
    updated_df = df.loc[df["normalized_name"].isin(known_teams)].copy()
    unmatched_df = df.loc[~df["normalized_name"].isin(known_teams)].copy()
    null_value_df = df.loc[df["gdp_per_capita"].isna() | df["confederation_code"].eq("")].copy()
    return {
        "rows_read": int(len(df)),
        "teams_to_update": int(updated_df["normalized_name"].nunique()),
        "unmatched_teams": sorted(unmatched_df["team_name"].astype(str).unique().tolist()),
        "rows_with_null_values": int(len(null_value_df)),
    }


def apply_seed(df: pd.DataFrame, teams_table: Table, engine: Engine) -> dict[str, Any]:
    with engine.begin() as connection:
        known_teams = {
            str(row.normalized_name)
            for row in connection.execute(select(teams_table.c.normalized_name)).all()
        }
        summary = build_seed_summary(df, known_teams)

        update_rows = []
        for row in df.itertuples(index=False):
            if row.normalized_name not in known_teams:
                continue
            update_rows.append(
                {
                    "where_normalized_name": row.normalized_name,
                    "gdp_per_capita": None if pd.isna(row.gdp_per_capita) else float(row.gdp_per_capita),
                    "confederation_code": str(row.confederation_code or "").strip().upper() or None,
                    "macro_source": "manual_csv",
                }
            )

        if update_rows:
            stmt = (
                update(teams_table)
                .where(teams_table.c.normalized_name == bindparam("where_normalized_name"))
                .values(
                    gdp_per_capita=bindparam("gdp_per_capita"),
                    confederation_code=bindparam("confederation_code"),
                    macro_source=bindparam("macro_source"),
                    macro_updated_at=pd.Timestamp.now("UTC").to_pydatetime(),
                )
            )
            connection.execute(stmt, update_rows)

    summary["teams_updated"] = len(update_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Klement macro context into public.teams.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Path to the macro CSV file.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--schema", default="public", help="Target PostgreSQL schema.")
    parser.add_argument("--dry-run", action="store_true", help="Only parse the CSV and print the summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_macro_dataframe(Path(args.csv_path))
    engine = get_engine(args.database_url)
    teams_table = reflect_teams_table(engine, args.schema)

    with engine.begin() as connection:
        known_teams = {
            str(row.normalized_name)
            for row in connection.execute(select(teams_table.c.normalized_name)).all()
        }
    summary = build_seed_summary(df, known_teams)

    print(f"rows_read={summary['rows_read']}")
    print(f"teams_to_update={summary['teams_to_update']}")
    print(f"rows_with_null_values={summary['rows_with_null_values']}")
    print(f"unmatched_teams={len(summary['unmatched_teams'])}")
    if summary["unmatched_teams"]:
        for team_name in summary["unmatched_teams"]:
            print(f"unmatched_team={team_name}")

    if args.dry_run:
        print("mode=dry-run")
        return 0

    applied_summary = apply_seed(df, teams_table, engine)
    print(f"teams_updated={applied_summary['teams_updated']}")
    print("mode=apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
