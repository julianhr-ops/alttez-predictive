from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
from sqlalchemy import create_engine, text

from alttez.api.migrations import MIGRATION_PROBES

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

    def test_nba_probes_distinguish_old_and_new_primary_keys(self):
        engine = create_engine("sqlite://")
        self.addCleanup(engine.dispose)
        with engine.begin() as connection:
            connection.exec_driver_sql("attach database ':memory:' as information_schema")
            connection.exec_driver_sql("create table information_schema.tables (table_schema text, table_name text)")
            connection.exec_driver_sql(
                "create table information_schema.table_constraints "
                "(constraint_schema text, constraint_name text, table_schema text, table_name text, constraint_type text)"
            )
            connection.exec_driver_sql(
                "create table information_schema.key_column_usage "
                "(constraint_schema text, constraint_name text, table_schema text, table_name text, "
                "column_name text, ordinal_position integer)"
            )
            probe15 = text(MIGRATION_PROBES["015_nba_predictions.sql"])
            probe16 = text(MIGRATION_PROBES["016_nba_predictions_key_served_model.sql"])
            self.assertFalse(connection.execute(probe15).scalar())
            connection.exec_driver_sql("insert into information_schema.tables values ('public', 'nba_predictions')")
            connection.exec_driver_sql(
                "insert into information_schema.table_constraints values "
                "('public', 'pk_nba_predictions', 'public', 'nba_predictions', 'PRIMARY KEY')"
            )
            for position, name in enumerate(("game_id", "model_version"), 1):
                connection.execute(text(
                    "insert into information_schema.key_column_usage values "
                    "('public', 'pk_nba_predictions', 'public', 'nba_predictions', :name, :position)"
                ), {"name": name, "position": position})
            self.assertTrue(connection.execute(probe15).scalar())
            self.assertFalse(connection.execute(probe16).scalar())
            connection.exec_driver_sql(
                "insert into information_schema.key_column_usage values "
                "('public', 'pk_nba_predictions', 'public', 'nba_predictions', 'served_model', 3)"
            )
            self.assertTrue(connection.execute(probe16).scalar())


if __name__ == "__main__":
    unittest.main()
