"""In-memory SQLite engine with the ``nba_predictions`` table under a ``public`` schema (no network)."""

from unittest import TestCase

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from alttez.site.ledger import DDL



def make_engine(case: TestCase) -> Engine:
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _attach(dbapi_connection, _record):
        dbapi_connection.execute("attach database ':memory:' as public")

    with engine.begin() as connection:
        connection.execute(text(DDL))
    case.addCleanup(engine.dispose)
    return engine
