"""Canonical JSONL representation of NBA predictions."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import text

from alttez.nba import store

DDL = """
create table if not exists public.nba_predictions (
    game_id integer not null,
    game_date date not null,
    tipoff_utc timestamp not null,
    home_team_id integer not null,
    away_team_id integer not null,
    model_version text not null,
    served_model text not null check (served_model in ('blend', 'elo_platt')),
    p_home_elo real not null,
    p_home_market real,
    p_home_served real not null,
    expected_margin real not null,
    market_spread real,
    market_total real,
    market_source text,
    created_at timestamp not null,
    status text not null default 'scheduled' check (status in ('scheduled', 'graded')),
    home_score integer,
    away_score integer,
    home_win boolean,
    brier real,
    graded_at timestamp,
    primary key (game_id, model_version, served_model)
)
"""

TEAMS = json.loads((Path(__file__).resolve().parents[2] / "site/results/nba_teams.json").read_text(encoding="utf-8"))


def timestamp(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def row_to_record(row, abbreviations=None):
    result = store.row_to_params(row)
    for key, value in result.items():
        if isinstance(value, datetime):
            result[key] = timestamp(value)
        elif isinstance(value, date):
            result[key] = value.isoformat()
    result["home_abbr"], result["away_abbr"] = abbreviations or (
        TEAMS[str(row.home_team_id)], TEAMS[str(row.away_team_id)]
    )
    return result


def row_to_line(row, abbreviations=None):
    return json.dumps(row_to_record(row, abbreviations), sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_records(records):
    seen = set()
    for index, record in enumerate(records, 1):
        key = (record["game_id"], record["model_version"], record["served_model"])
        if key in seen:
            raise ValueError(f"line {index}: duplicate key {key}")
        seen.add(key)
        if datetime.fromisoformat(record["created_at"]) > datetime.fromisoformat(record["tipoff_utc"]):
            raise ValueError(f"line {index}: created_at after tipoff_utc")
        if record["status"] == "graded" and (record["home_score"] is None or record["away_score"] is None):
            raise ValueError(f"line {index}: graded row without scores")
        if not 0 <= record["p_home_served"] <= 1:
            raise ValueError(f"line {index}: p_home_served outside [0,1]")
        if record["served_model"] == "blend" and record["p_home_market"] is None:
            raise ValueError(f"line {index}: blend row without market probability")
        store.row_from_mapping(record)


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows(engine):
    with engine.connect() as connection:
        records = connection.execute(text(f"select {', '.join(store.COLUMNS)} from {store.TABLE}")).mappings()
        return [store.row_from_mapping(record) for record in records]


def import_records(engine, records):
    verify_records(records)
    with engine.begin() as connection:
        connection.execute(text(DDL))
        connection.execute(text("""
            create table if not exists public.nba_site_abbreviations (
                game_id integer not null, model_version text not null, served_model text not null,
                home_abbr text not null, away_abbr text not null,
                primary key (game_id, model_version, served_model)
            )
        """))
        for record in records:
            connection.execute(store._INSERT_SQL, store.row_to_params(store.row_from_mapping(record)))
            connection.execute(text("""
                insert into public.nba_site_abbreviations
                (game_id, model_version, served_model, home_abbr, away_abbr)
                values (:game_id, :model_version, :served_model, :home_abbr, :away_abbr)
                on conflict do nothing
            """), record)


def export_rows(engine, path):
    rows = sorted(load_rows(engine), key=lambda r: (r.tipoff_utc, r.game_id, r.model_version, r.served_model))
    with engine.connect() as connection:
        records = connection.execute(text("select * from public.nba_site_abbreviations")).mappings()
        names = {(r["game_id"], r["model_version"], r["served_model"]): (r["home_abbr"], r["away_abbr"])
                 for r in records}
    Path(path).write_bytes("".join(
        row_to_line(row, names.get((row.game_id, row.model_version, row.served_model.value))) + "\n"
        for row in rows
    ).encode("utf-8"))
