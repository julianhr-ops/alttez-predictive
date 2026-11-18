import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from sqlalchemy import text

from alttez.api.db import get_engine
from alttez.nba import store
from alttez.site import ledger
from tests.test_nba_store import make_row


class LedgerInvariantTests(unittest.TestCase):
    def setUp(self):
        self.row = ledger.row_to_record(make_row(1))

    def test_round_trip_record(self):
        line = ledger.row_to_line(make_row(1))
        self.assertEqual(ledger.row_to_line(ledger.store.row_from_mapping(json.loads(line))), line)

    def test_duplicate_key(self):
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            ledger.verify_records([self.row, self.row])

    def test_late_prediction(self):
        self.row["created_at"] = ledger.timestamp(make_row(1).tipoff_utc + timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError, "after tipoff"):
            ledger.verify_records([self.row])

    def test_graded_without_scores(self):
        self.row["status"] = "graded"
        with self.assertRaisesRegex(ValueError, "without scores"):
            ledger.verify_records([self.row])

    def test_probability_out_of_range(self):
        self.row["p_home_served"] = 1.01
        with self.assertRaisesRegex(ValueError, "outside"):
            ledger.verify_records([self.row])

    def test_blend_without_market(self):
        self.row["served_model"] = "blend"
        self.row["p_home_market"] = None
        with self.assertRaisesRegex(ValueError, "without market"):
            ledger.verify_records([self.row])


class SQLiteEngineTests(unittest.TestCase):
    def test_file_database_visible_to_second_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nba.db"
            url = f"sqlite:///{path.as_posix()}"
            first = get_engine(url)
            second = get_engine(url)
            try:
                with first.begin() as connection:
                    connection.execute(text(ledger.DDL))
                row = make_row()
                store.insert_prediction_if_absent(first, row, now=row.created_at)
                self.assertEqual(ledger.load_rows(second), [row])
            finally:
                first.dispose()
                second.dispose()
