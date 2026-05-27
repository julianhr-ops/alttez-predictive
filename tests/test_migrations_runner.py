from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from alttez.api.migrations import discover_migration_files, discover_pending_migration_files, render_migration_plan


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "migration_runner"


class MigrationRunnerTest(unittest.TestCase):
    def test_discovers_sql_files_in_lexical_order(self) -> None:
        files = discover_migration_files(FIXTURE_DIR)

        self.assertEqual([item.name for item in files], ["001_base.sql", "002_enrichment.sql", "003_indexes.sql"])

    def test_render_migration_plan_numbers_entries(self) -> None:
        files = [Path("001_base.sql"), Path("002_enrichment.sql")]

        self.assertEqual(render_migration_plan(files), ["01. 001_base.sql", "02. 002_enrichment.sql"])

    @patch("alttez.api.migrations.reconcile_existing_schema", return_value=set())
    @patch("alttez.api.migrations.load_applied_migration_names", return_value={"001_base.sql"})
    def test_discovers_only_pending_files(self, _mock_applied: object, _mock_reconcile: object) -> None:
        pending_files, applied_names = discover_pending_migration_files(object(), FIXTURE_DIR)

        self.assertEqual([item.name for item in pending_files], ["002_enrichment.sql", "003_indexes.sql"])
        self.assertEqual(applied_names, {"001_base.sql"})


if __name__ == "__main__":
    unittest.main()
