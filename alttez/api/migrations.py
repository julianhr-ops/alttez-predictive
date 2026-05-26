from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine


DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "sql" / "supabase" / "migrations"


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


def apply_migrations(engine: Engine, migrations_dir: str | Path | None = None) -> list[Path]:
    migration_files = discover_migration_files(migrations_dir)

    with engine.begin() as connection:
        for migration_file in migration_files:
            sql = migration_file.read_text(encoding="utf-8")
            connection.exec_driver_sql(sql)

    return migration_files
