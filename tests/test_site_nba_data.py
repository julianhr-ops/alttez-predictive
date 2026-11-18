import json
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from alttez.nba.store import PredictionStatus
from alttez.site.nba_data import build_data
from tests.test_nba_store import TIPOFF, make_row


ROOT = Path(__file__).resolve().parents[1]


class ValidationSnapshotTests(unittest.TestCase):
    def test_snapshot_matches_report_table(self):
        snapshot = json.loads((ROOT / "site/results/nba_validation.json").read_text(encoding="utf-8"))
        report = (ROOT / "docs/nba.md").read_text(encoding="utf-8")
        table = report.split("| model | Brier | log loss | accuracy | ECE |", 1)[1]
        lines = table.strip().splitlines()[1:]
        expected = []
        for line in lines:
            if not line.startswith("|"):
                break
            name, brier, log_loss, accuracy, ece = (cell.strip() for cell in line.strip("|").split("|"))
            expected.append({"model": name, "brier": float(brier), "log_loss": float(log_loss),
                             "accuracy": float(accuracy), "ece": float(ece)})
        self.assertEqual(snapshot["rows"], expected)


class ScoreboardTests(unittest.TestCase):
    def test_graded_game_scores_and_recent_result(self):
        row = replace(make_row(p_served=0.8, market=0.6), status=PredictionStatus.GRADED,
                      home_score=110, away_score=100, home_win=True, brier=0.04,
                      graded_at=TIPOFF + timedelta(hours=3))
        data = build_data([row], TIPOFF + timedelta(hours=4))
        scoreboard = data["nba_scoreboard.json"]
        self.assertEqual(scoreboard["served"], {"brier": 0.04, "accuracy": 1.0})
        self.assertEqual(scoreboard["market"], {"brier": 0.16, "accuracy": 1.0})
        self.assertIsNone(scoreboard["paired_brier_diff"])
        self.assertEqual(scoreboard["predicted_before_tipoff"], {"n": 1, "of": 1})
        game = data["nba_recent.json"]["games"][0]
        self.assertEqual((game["home_score"], game["away_score"], game["brier"]), (110, 100, 0.04))

    def test_paired_interval_starts_at_30_lined_games(self):
        rows = [
            replace(make_row(game_id=i, p_served=0.8, market=0.6), status=PredictionStatus.GRADED,
                    home_score=110, away_score=100, home_win=True, brier=0.04,
                    graded_at=TIPOFF + timedelta(hours=3))
            for i in range(30)
        ]
        below = build_data(rows[:29], TIPOFF)["nba_scoreboard.json"]
        at = build_data(rows, TIPOFF)["nba_scoreboard.json"]
        self.assertIsNone(below["paired_brier_diff"])
        self.assertEqual(at["paired_brier_diff"], {"mean": -0.12, "lo": -0.12, "hi": -0.12})

    def test_upcoming_uses_one_served_row_and_omits_past_games(self):
        elo = make_row(1, market=None)
        blend = make_row(1)
        past = make_row(2, tipoff_utc=TIPOFF - timedelta(days=1))
        games = build_data([elo, blend, past], TIPOFF - timedelta(hours=1))["nba_upcoming.json"]["games"]
        self.assertEqual([(game["game_id"], game["served_model"]) for game in games], [(1, "blend")])