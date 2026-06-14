from __future__ import annotations

import unittest

import numpy as np

from alttez.evaluation.regulation_points import (
    expected_regulation_points,
    points_optimal_prediction,
    rank_regulation_predictions,
    regulation_points,
    scoreline_candidate_profiles,
    scoreline_scenarios_for_prediction,
    summarize_regulation_points,
)


class TestRegulationPoints(unittest.TestCase):
    def test_examples_match_pool_rules(self) -> None:
        self.assertEqual(regulation_points(1, 0, 1, 0), 10)
        self.assertEqual(regulation_points(2, 0, 3, 1), 6)
        self.assertEqual(regulation_points(2, 1, 1, 0), 6)
        self.assertEqual(regulation_points(0, 1, 2, 1), 2)

    def test_knockout_doubles_points(self) -> None:
        group_points = regulation_points(2, 0, 3, 1, is_knockout=False)
        knockout_points = regulation_points(2, 0, 3, 1, is_knockout=True)

        self.assertEqual(knockout_points, group_points * 2)

    def test_points_optimal_prediction_uses_expected_regulation_value(self) -> None:
        matrix = np.zeros((3, 3), dtype=float)
        matrix[1, 0] = 0.45
        matrix[2, 0] = 0.30
        matrix[0, 1] = 0.25

        pick = points_optimal_prediction(matrix)
        ranking = rank_regulation_predictions(matrix)

        self.assertEqual(pick["scoreline"], ranking.iloc[0]["Pronostico"])
        self.assertAlmostEqual(
            pick["ev"],
            expected_regulation_points(matrix, pick["home_goals"], pick["away_goals"]),
        )

    def test_summarize_regulation_points_reports_distribution(self) -> None:
        summary = summarize_regulation_points(
            [
                {"regulation_points": 10, "regulation_max_points": 10},
                {"regulation_points": 6, "regulation_max_points": 10},
                {"regulation_points": 0, "regulation_max_points": 10},
            ]
        )

        self.assertEqual(summary["regulation_points_total"], 16.0)
        self.assertAlmostEqual(summary["regulation_points_avg"], 16 / 3)
        self.assertAlmostEqual(summary["regulation_points_pct_of_max"], 16 / 30)
        self.assertAlmostEqual(summary["regulation_full_hit_rate"], 1 / 3)
        self.assertAlmostEqual(summary["regulation_zero_rate"], 1 / 3)
        self.assertEqual(summary["regulation_points_distribution"], {0: 1, 6: 1, 10: 1})

    def test_scoreline_scenarios_show_points_for_primary_pick(self) -> None:
        matrix = np.zeros((3, 3), dtype=float)
        matrix[1, 1] = 0.40
        matrix[1, 0] = 0.35
        matrix[2, 0] = 0.25

        scenarios = scoreline_scenarios_for_prediction(matrix, 1, 0, top_n=3)

        self.assertEqual([item["actual_score"] for item in scenarios], ["1 - 1", "1 - 0", "2 - 0"])
        self.assertEqual([item["points_if_actual"] for item in scenarios], [2, 10, 7])

    def test_scoreline_candidate_profiles_compare_different_predictions(self) -> None:
        matrix = np.zeros((3, 3), dtype=float)
        matrix[1, 1] = 0.40
        matrix[1, 0] = 0.35
        matrix[2, 0] = 0.25

        candidates = scoreline_candidate_profiles(matrix, top_probability=2, top_ev=2)

        scores = [candidate["score"] for candidate in candidates]
        self.assertIn("1 - 1", scores)
        self.assertIn("1 - 0", scores)
        for candidate in candidates:
            self.assertIn("full_points_probability", candidate)
            self.assertIn("expected_points", candidate)
            self.assertIn("points_probability", candidate)


if __name__ == "__main__":
    unittest.main()
