from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine


DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "sql" / "supabase" / "migrations"
MIGRATION_HISTORY_TABLE = "public.schema_migration_history"
MIGRATION_PROBES: dict[str, str] = {
    "001_base_catalogs_and_match_tables.sql": """
        select exists (
            select 1
            from information_schema.tables
            where table_schema = 'public'
              and table_name = 'match_team_stats'
        )
    """,
    "002_match_enrichment.sql": """
        select exists (
            select 1
            from information_schema.columns
            where table_schema = 'public'
              and table_name = 'matches'
              and column_name = 'competition_class'
        )
    """,
    "003_indexes.sql": """
        select exists (
            select 1
            from pg_indexes
            where schemaname = 'public'
              and indexname = 'idx_matches_competition_class'
        )
    """,
    "004_updated_at_triggers.sql": """
        select exists (
            select 1
            from information_schema.triggers
            where event_object_schema = 'public'
              and trigger_name = 'trg_matches_updated_at'
        )
    """,
    "005_international_results_compat_view.sql": """
        select exists (
            select 1
            from information_schema.views
            where table_schema = 'public'
              and table_name = 'international_results_compat'
        )
    """,
    "006_certification_batches.sql": """
        select exists (
            select 1
            from information_schema.tables
            where table_schema = 'public'
              and table_name = 'certification_batches'
        )
    """,
    "007_alert_events.sql": """
        select exists (
            select 1
            from information_schema.tables
            where table_schema = 'public'
              and table_name = 'alert_events'
        )
    """,
    "008_prediction_memory.sql": """
        select exists (
            select 1
            from information_schema.tables
            where table_schema = 'public'
              and table_name = 'prediction_runs'
        )
    """,
    "009_model_governance.sql": """
        select exists (
            select 1
            from information_schema.columns
            where table_schema = 'public'
              and table_name = 'model_versions'
              and column_name = 'production_status'
        )
    """,
    "010_klement_macro_features.sql": """
        select exists (
            select 1
            from information_schema.columns
            where table_schema = 'public'
              and table_name = 'teams'
              and column_name = 'gdp_per_capita'
        )
    """,
}


def resolve_migrations_dir(migrations_dir: str | Path | None = None) -> Path:
    path = Path(migrations_dir) if migrations_dir is not None else DEFAULT_MIGRATIONS_DIR
    if not path.is_absolute():
        candidate = Path.cwd() / path
        if candidate.exists():
            path = candidate
        else:
            path = DEFAULT_MIGRATIONS_DIR if path == Path("sql/supabase/migrations") else path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Migration directory does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Migration path is not a directory: {path}")

    return path


def discover_migration_files(migrations_dir: str | Path | None = None) -> list[Path]:
    path = resolve_migrations_dir(migrations_dir)
    return sorted(
        [item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".sql"],
        key=lambda item: item.name,
    )


def render_migration_plan(migration_files: list[Path]) -> list[str]:
    return [f"{index:02d}. {path.name}" for index, path in enumerate(migration_files, start=1)]


def ensure_migration_history_table(engine: Engine) -> None:
    create_sql = f"""
    create table if not exists {MIGRATION_HISTORY_TABLE} (
        id bigint generated always as identity primary key,
        migration_name varchar(255) not null unique,
        applied_at timestamptz not null default timezone('utc', now())
    );
    """
    with engine.begin() as connection:
        connection.exec_driver_sql(create_sql)


def load_applied_migration_names(engine: Engine) -> set[str]:
    ensure_migration_history_table(engine)
    with engine.begin() as connection:
        rows = connection.execute(text(f"select migration_name from {MIGRATION_HISTORY_TABLE}")).scalars().all()
    return {str(row) for row in rows}


def reconcile_existing_schema(engine: Engine, migration_files: list[Path]) -> set[str]:
    ensure_migration_history_table(engine)
    reconciled: set[str] = set()
    with engine.begin() as connection:
        for migration_file in migration_files:
            probe_sql = MIGRATION_PROBES.get(migration_file.name)
            if not probe_sql:
                continue
            probe_passed = bool(connection.execute(text(probe_sql)).scalar())
            if not probe_passed:
                continue
            connection.execute(
                text(
                    f"""
                    insert into {MIGRATION_HISTORY_TABLE} (migration_name, applied_at)
                    values (:migration_name, :applied_at)
                    on conflict (migration_name) do nothing
                    """
                ),
                {
                    "migration_name": migration_file.name,
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            reconciled.add(migration_file.name)
    return reconciled


def discover_pending_migration_files(engine: Engine, migrations_dir: str | Path | None = None) -> tuple[list[Path], set[str]]:
    migration_files = discover_migration_files(migrations_dir)
    reconcile_existing_schema(engine, migration_files)
    applied_names = load_applied_migration_names(engine)
    pending_files = [migration_file for migration_file in migration_files if migration_file.name not in applied_names]
    return pending_files, applied_names


def apply_migrations(engine: Engine, migrations_dir: str | Path | None = None) -> list[Path]:
    migration_files, _ = discover_pending_migration_files(engine, migrations_dir)

    with engine.begin() as connection:
        for migration_file in migration_files:
            sql = migration_file.read_text(encoding="utf-8")
            connection.exec_driver_sql(sql)
            connection.execute(
                text(
                    f"""
                    insert into {MIGRATION_HISTORY_TABLE} (migration_name, applied_at)
                    values (:migration_name, :applied_at)
                    on conflict (migration_name) do nothing
                    """
                ),
                {
                    "migration_name": migration_file.name,
                    "applied_at": datetime.now(timezone.utc),
                },
            )

    return migration_files
