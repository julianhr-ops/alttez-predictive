import tempfile
import unittest
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

from alttez.dashboard.nba_view import graded_history, scoreboard, upcoming_games
from alttez.nba import store
from scripts import nba_daily
from tests.nba_sqlite import make_engine
from tests.test_nba_store import NOW, TIPOFF, make_row


class NbaViewTests(unittest.TestCase):
    def test_sqlite_predictions_and_paired_scoreboard(self):
        engine = make_engine(self)
        for row in (make_row(1, p_served=0.8), make_row(2, p_served=0.6, market=None), make_row(3)):
            store.insert_prediction_if_absent(engine, row, now=NOW)
        store.grade_prediction(engine, 1, "nba-elo-v1", 110, 100, 0.8, now=TIPOFF)
        store.grade_prediction(engine, 2, "nba-elo-v1", 100, 110, 0.6, now=TIPOFF)
        window = store.DateWindow(date(2026, 11, 1), date(2026, 11, 30))
        pending = pd.DataFrame([asdict(r) for r in store.load_ungraded(engine, window)])
        upcoming = upcoming_games(pending, NOW)
        self.assertEqual(upcoming.iloc[0]["source"], "Market blend")
        self.assertEqual(upcoming.iloc[0]["home_line"], 3.5)
        self.assertTrue(upcoming_games(pending, TIPOFF + timedelta(seconds=1)).empty)
        rows = store.load_graded_summary(engine, window).rows
        history = graded_history(pd.DataFrame([asdict(r) for r in rows]))
        result = scoreboard(history)
        self.assertEqual((result["n"], result["market_n"]), (2, 1))
        self.assertAlmostEqual(result["hit_rate"], 0.5)
        self.assertAlmostEqual(result["brier"], 0.2)
        self.assertAlmostEqual(result["paired_brier"], 0.04)
        self.assertAlmostEqual(result["market_brier"], 0.35 ** 2)
        self.assertEqual(set(history["outcome"]), {"Home", "Away"})

    def test_empty_and_no_market(self):
        self.assertEqual(scoreboard(pd.DataFrame())["n"], 0)
        frame = pd.DataFrame([{"p_home_served": 0.4, "home_win": False, "p_home_market": None,
                               "market_spread": None, "brier": 0.16, "tipoff_utc": TIPOFF}])
        result = scoreboard(graded_history(frame))
        self.assertEqual(result["market_n"], 0)
        self.assertIsNone(result["market_brier"])
        upcoming = upcoming_games(pd.DataFrame([asdict(make_row(4, market=None))]), NOW)
        self.assertEqual(upcoming.iloc[0]["source"], "Elo + Platt")
        self.assertTrue(pd.isna(upcoming.iloc[0]["home_line"]))

    def test_html_escapes_digest_and_has_no_external_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "digest.html"
            with mock.patch.object(nba_daily, "run", return_value=(0, '<script>alert("x")</script>')):
                self.assertEqual(nba_daily.main(["--dry-run", "--html", str(path)]), 0)
            page = path.read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>", page)
            self.assertIn("<style>", page)
            self.assertNotIn("https://", page)
