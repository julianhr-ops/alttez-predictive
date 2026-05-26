import os
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from alttez.api.settings import load_local_env

DEFAULT_DB_ENV_VARS = ("SUPABASE_DB_URL", "DATABASE_URL")


def get_database_url(explicit_url: Optional[str] = None) -> str:
    """Resolves the database URL from an explicit value or environment."""

    load_local_env()

    if explicit_url:
        return _ensure_sslmode(explicit_url)

    for env_var in DEFAULT_DB_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return _ensure_sslmode(value)

    raise RuntimeError(
        "No database URL configured. Set SUPABASE_DB_URL or DATABASE_URL to connect ALTTEZ to Supabase."
    )


def get_engine(database_url: Optional[str] = None) -> Engine:
    """Builds a SQLAlchemy engine for PostgreSQL/Supabase."""

    return create_engine(
        get_database_url(database_url),
        pool_pre_ping=True,
        future=True,
    )


def read_dataframe(sql: str, params: Optional[dict[str, Any]] = None, database_url: Optional[str] = None) -> pd.DataFrame:
    """Reads a SQL query into a DataFrame using the configured engine."""

    engine = get_engine(database_url)
    with engine.connect() as connection:
        return pd.read_sql(text(sql), con=connection, params=params or {})


def _ensure_sslmode(database_url: str) -> str:
    """Appends sslmode=require when the URL does not declare it explicitly."""

    normalized_url = _normalize_postgres_driver(database_url)
    if "sslmode=" in normalized_url or not normalized_url.startswith(("postgresql://", "postgresql+psycopg://")):
        return normalized_url

    separator = "&" if "?" in normalized_url else "?"
    return f"{normalized_url}{separator}sslmode=require"


def _normalize_postgres_driver(database_url: str) -> str:
    """Forces SQLAlchemy to use the installed psycopg v3 driver."""

    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url
