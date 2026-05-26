from __future__ import annotations

import unittest
from pathlib import Path

from alttez.api.migrations import discover_migration_files, render_migration_plan


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "migration_runner"


class MigrationRunnerTest(unittest.TestCase):
    def test_discovers_sql_files_in_lexical_order(self) -> None:
        files = discover_migration_files(FIXTURE_DIR)

        self.assertEqual([item.name for item in files], ["001_base.sql", "002_enrichment.sql", "003_indexes.sql"])

    def test_render_migration_plan_numbers_entries(self) -> None:
        files = [Path("001_base.sql"), Path("002_enrichment.sql")]

        self.assertEqual(render_migration_plan(files), ["01. 001_base.sql", "02. 002_enrichment.sql"])


if __name__ == "__main__":
    unittest.main()
