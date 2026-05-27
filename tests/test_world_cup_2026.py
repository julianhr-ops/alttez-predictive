from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from alttez.inference.router import build_active_predictor
from alttez.inference.world_cup_2026 import load_world_cup_2026_config, simulate_world_cup_2026


def _historical_df() -> pd.DataFrame:
    rows = [
        ("2025-01-10", "Mexico", "USA", 1, 0, True),
        ("2025-02-15", "USA", "Canada", 2, 1, False),
        ("2025-03-12", "Canada", "Mexico", 0, 1, False),
        ("2025-04-11", "Brazil", "Argentina", 2, 2, True),
        ("2025-05-19", "Argentina", "Colombia", 1, 0, False),
        ("2025-06-21", "Colombia", "Brazil", 0, 1, False),
        ("2025-07-18", "Mexico", "Canada", 2, 0, False),
        ("2025-08-14", "USA", "Mexico", 1, 1, False),
        ("2025-09-09", "Canada", "USA", 0, 1, False),
        ("2025-10-04", "Brazil", "Colombia", 3, 1, False),
        ("2025-11-17", "Argentina", "Brazil", 0, 1, False),
        ("2025-12-01", "Colombia", "Argentina", 1, 1, False),
        ("2026-01-15", "Mexico", "Colombia", 1, 0, True),
        ("2026-02-12", "USA", "Brazil", 0, 2, True),
        ("2026-03-08", "Canada", "Argentina", 1, 2, True),
        ("2026-04-19", "Mexico", "Brazil", 1, 1, True),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


class TestWorldCup2026Simulation(unittest.TestCase):
    def test_simulation_runs_with_official_config(self) -> None:
        with patch("alttez.inference.router.load_active_model_record", return_value=None):
            predictor = build_active_predictor(_historical_df(), "2026-05-27")

        report = simulate_world_cup_2026(
            predictor,
            config=load_world_cup_2026_config(),
            iterations=5,
            random_seed=11,
        )

        probabilities = report["team_probabilities"]
        champion_sum = sum(float(row["champion_probability"]) for row in probabilities)

        self.assertEqual(len(probabilities), 48)
        self.assertAlmostEqual(champion_sum, 1.0, places=6)
        self.assertEqual(report["model_family"], predictor.model_family)


if __name__ == "__main__":
    unittest.main()
