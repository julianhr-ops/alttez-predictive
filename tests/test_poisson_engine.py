from __future__ import annotations

import unittest

from alttez.features.poisson import PoissonEngine


class TestPoissonEngine(unittest.TestCase):
    def test_predict_xg_preserves_precision_and_uses_recent_form(self) -> None:
        engine = PoissonEngine()
        engine.global_avg_goals = 1.25
        engine.global_home_avg_goals = 1.4
        engine.global_away_avg_goals = 1.1
        engine.team_stats = {
            "A": {
                "home_attack": 1.10,
                "away_attack": 1.00,
                "home_defense": 0.95,
                "away_defense": 1.00,
                "attack": 1.05,
                "defense": 0.90,
                "matches": 20.0,
                "effective_matches": 12.0,
                "confidence": 0.8,
                "recent_points_per_match": 2.5,
                "recent_goal_diff_per_match": 1.2,
            },
            "B": {
                "home_attack": 1.00,
                "away_attack": 0.92,
                "home_defense": 1.02,
                "away_defense": 1.08,
                "attack": 0.94,
                "defense": 1.04,
                "matches": 20.0,
                "effective_matches": 12.0,
                "confidence": 0.8,
                "recent_points_per_match": 0.8,
                "recent_goal_diff_per_match": -0.6,
            },
        }

        lambda_home, lambda_away = engine.predict_xg("A", "B", is_neutral=False)

        self.assertGreater(lambda_home, 1.4 * 1.10 * 1.08)
        self.assertLess(lambda_away, 1.1 * 0.92 * 0.95)
        self.assertNotEqual(round(lambda_home, 2), lambda_home)

    def test_confidence_no_longer_saturates_too_early(self) -> None:
        engine = PoissonEngine(min_history_matches=5)

        confidence = engine._estimate_confidence(raw_matches=35, weighted_matches=8.68)

        self.assertGreater(confidence, 0.5)
        self.assertLess(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
