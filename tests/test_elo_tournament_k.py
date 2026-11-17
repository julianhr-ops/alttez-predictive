import unittest
from unittest.mock import patch

import pandas as pd

from alttez.features.elo import EloEngine, tournament_k_factor
from alttez.features.klement import build_klement_feature_state

BASE_K = 30.0

TOURNAMENT_K_CASES = [
    ("FIFA World Cup", 60.0),
    ("FIFA World Cup qualification", 40.0),
    ("UEFA Euro", 50.0),
    ("UEFA Euro qualification", 40.0),
    ("Copa America", 50.0),
    ("UEFA Nations League", BASE_K),
    ("Friendly", BASE_K),
    ("African Cup of Nations", BASE_K),
    ("", BASE_K),
    (None, BASE_K),
]


def _k_applied_by_engine(tournament: str | None) -> float:
    """Recover the K the engine used from the first-match rating update.

    Two equal-rated teams on neutral ground: expected score is 0.5, so a home win
    moves the home rating by exactly K * 0.5.
    """

    engine = EloEngine(k_factor=BASE_K, mov_enabled=False)
    match = pd.DataFrame(
        [
            {
                "home_team": "A",
                "away_team": "B",
                "neutral": True,
                "tournament": tournament,
                "home_score": 1,
                "away_score": 0,
                "home_result": 1.0,
            }
        ]
    )
    engine.compute_elo_features(match)
    return (engine.ratings["A"] - engine.base_rating) / 0.5


class EloTournamentKTest(unittest.TestCase):
    def test_engine_k_depends_on_tournament_name(self) -> None:
        for tournament, expected_k in TOURNAMENT_K_CASES:
            with self.subTest(tournament=tournament):
                self.assertAlmostEqual(_k_applied_by_engine(tournament), expected_k)

    def test_tournament_k_factor_table(self) -> None:
        for tournament, expected_k in TOURNAMENT_K_CASES:
            with self.subTest(tournament=tournament):
                self.assertEqual(tournament_k_factor(tournament, BASE_K), expected_k)

    def test_qualification_is_case_insensitive(self) -> None:
        self.assertEqual(tournament_k_factor("FIFA World Cup Qualification", BASE_K), 40)

    def test_klement_feature_state_applies_qualifier_k(self) -> None:
        for tournament, expected_k in TOURNAMENT_K_CASES[:5]:
            with self.subTest(tournament=tournament):
                match = pd.DataFrame(
                    [
                        {
                            "date": "2025-01-10",
                            "home_team": "A",
                            "away_team": "B",
                            "home_score": 1,
                            "away_score": 0,
                            "neutral": True,
                            "tournament": tournament,
                        }
                    ]
                )
                with patch("alttez.features.klement.load_team_macro_context", return_value={}):
                    _, state = build_klement_feature_state(match)
                self.assertAlmostEqual((state.final_ratings["A"] - 1500) / 0.5, expected_k)


if __name__ == "__main__":
    unittest.main()
