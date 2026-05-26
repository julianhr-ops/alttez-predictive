from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.db import get_engine
from alttez.api.migrations import apply_migrations, discover_migration_files, render_migration_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply ALTTEZ Supabase migrations in lexical order.")
    parser.add_argument(
        "--migrations-dir",
        default="sql/supabase/migrations",
        help="Directory containing versioned SQL migrations.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Explicit database URL. Falls back to environment if omitted.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the migrations instead of only printing the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    migration_files = discover_migration_files(args.migrations_dir)

    print("[plan]")
    for item in render_migration_plan(migration_files):
        print(item)

    if not args.apply:
        print("mode=dry-run")
        return 0

    engine = get_engine(args.database_url)
    applied = apply_migrations(engine, args.migrations_dir)

    print("\n[apply]")
    for item in render_migration_plan(applied):
        print(item)
    print("mode=apply")
    print(f"applied={len(applied)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
