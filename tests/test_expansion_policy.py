from __future__ import annotations

import unittest

from alttez.operations.expansion_policy import ExpansionThresholds, classify_competition_readiness


class TestExpansionPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = ExpansionThresholds(
            min_matches=60,
            min_teams=10,
            min_evaluated_predictions=20,
            min_analyzable_rate=0.55,
            max_brier_score=0.66,
        )

    def test_classify_competition_readiness_approves_strong_candidate(self) -> None:
        readiness, reason = classify_competition_readiness(
            {
                "competition_class": "continental",
                "matches_total": 120,
                "teams_total": 18,
                "evaluated_predictions": 45,
                "analyzable_rate": 0.72,
                "brier_score": 0.58,
            },
            self.thresholds,
        )
        self.assertEqual(readiness, "approved")
        self.assertIn("apta", reason.lower())

    def test_classify_competition_readiness_blocks_low_history(self) -> None:
        readiness, _ = classify_competition_readiness(
            {
                "competition_class": "friendly",
                "matches_total": 24,
                "teams_total": 8,
                "evaluated_predictions": 0,
                "analyzable_rate": 0.2,
                "brier_score": None,
            },
            self.thresholds,
        )
        self.assertEqual(readiness, "blocked")


if __name__ == "__main__":
    unittest.main()
