"""In-memory SQLite engine with the ``nba_predictions`` table under a ``public`` schema (no network)."""

from unittest import TestCase

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

DDL = """
create table public.nba_predictions (
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


def make_engine(case: TestCase) -> Engine:
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _attach(dbapi_connection, _record):
        dbapi_connection.execute("attach database ':memory:' as public")

    with engine.begin() as connection:
        connection.execute(text(DDL))
    case.addCleanup(engine.dispose)
    return engine
