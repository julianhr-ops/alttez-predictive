from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from alttez.inference.router import (
    ActiveModelSelection,
    ActiveRecordLookup,
    BaseMatchPredictor,
    build_active_predictor,
)
from alttez.inference.world_cup_2026 import (
    _sample_match,
    load_world_cup_2026_config,
    predict_world_cup_2026_bracket,
    simulate_world_cup_2026,
)


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
        with patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(None)):
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

    def test_simulation_passes_world_cup_context_to_predictor(self) -> None:
        class CapturingPredictor(BaseMatchPredictor):
            def __init__(self) -> None:
                super().__init__(
                    selection=ActiveModelSelection(
                        version_key=None,
                        model_version="test-model",
                        model_family="test-family",
                        artifact_path=None,
                        requested_production_status="active",
                    )
                )
                self.contexts: list[dict[str, object]] = []

            def get_team_profile(self, team: str) -> dict[str, float]:
                return {"matches": 10.0, "effective_matches": 10.0}

            def predict_match(
                self,
                home_team: str,
                away_team: str,
                *,
                is_neutral: bool = False,
                is_knockout: bool = False,
                max_goals: int = 7,
                tournament_context: dict[str, object] | None = None,
            ) -> dict[str, object]:
                self.contexts.append(tournament_context or {})
                return {
                    "prob_home": 0.5,
                    "prob_draw": 0.2,
                    "prob_away": 0.3,
                    "prob_matrix": None,
                }

        predictor = CapturingPredictor()
        simulate_world_cup_2026(
            predictor,
            config=load_world_cup_2026_config(),
            iterations=1,
            random_seed=3,
        )

        self.assertTrue(predictor.contexts)
        self.assertTrue(all(item.get("tournament") == "FIFA World Cup 2026" for item in predictor.contexts))
        self.assertTrue(all(item.get("edition") == 2026 for item in predictor.contexts))

    def test_simulation_prefers_probability_matrix_when_available(self) -> None:
        class XGBoostLikePredictor(BaseMatchPredictor):
            def __init__(self) -> None:
                super().__init__(
                    selection=ActiveModelSelection(
                        version_key=None,
                        model_version="xgboost-test",
                        model_family="xgboost",
                        artifact_path=None,
                        requested_production_status="active",
                    )
                )

            def get_team_profile(self, team: str) -> dict[str, float]:
                return {"matches": 10.0, "effective_matches": 10.0}

            def predict_match(
                self,
                home_team: str,
                away_team: str,
                *,
                is_neutral: bool = False,
                is_knockout: bool = False,
                max_goals: int = 7,
                tournament_context: dict[str, object] | None = None,
            ) -> dict[str, object]:
                return {
                    "model_family": "xgboost",
                    "prob_home": 0.0,
                    "prob_draw": 0.0,
                    "prob_away": 1.0,
                    "prob_matrix": np.array([[1.0, 0.0], [0.0, 0.0]]),
                }

        predictor = XGBoostLikePredictor()
        match = _sample_match(
            predictor,
            "Home",
            "Away",
            rng=np.random.default_rng(3),
            is_neutral=True,
            is_knockout=False,
            tournament_context={"tournament": "FIFA World Cup 2026", "edition": 2026},
        )

        self.assertEqual(match.home_score, 0)
        self.assertEqual(match.away_score, 0)
        self.assertIsNone(match.winner)

    def test_sample_match_deterministic_picks_argmax_scoreline(self) -> None:
        class FixedPredictor(BaseMatchPredictor):
            def __init__(self) -> None:
                super().__init__(
                    selection=ActiveModelSelection(
                        version_key=None,
                        model_version="fixed",
                        model_family="poisson",
                        artifact_path=None,
                        requested_production_status="active",
                    )
                )

            def get_team_profile(self, team: str) -> dict[str, float]:
                return {"matches": 10.0, "effective_matches": 10.0}

            def predict_match(self, home_team, away_team, *, is_neutral=False, is_knockout=False, max_goals=7, tournament_context=None):
                # Most-likely scoreline is 2-1 (index [2,1]).
                matrix = np.zeros((3, 3))
                matrix[2, 1] = 0.6
                matrix[0, 0] = 0.4
                return {"prob_home": 0.7, "prob_draw": 0.1, "prob_away": 0.2, "prob_matrix": matrix}

        match = _sample_match(
            FixedPredictor(), "Home", "Away",
            rng=np.random.default_rng(0), is_neutral=True, is_knockout=False, deterministic=True,
        )
        self.assertEqual((match.home_score, match.away_score), (2, 1))
        self.assertEqual(match.winner, "Home")

    def test_deterministic_knockout_avoids_penalty_draw(self) -> None:
        class DrawArgmaxPredictor(BaseMatchPredictor):
            def __init__(self) -> None:
                super().__init__(
                    selection=ActiveModelSelection(
                        version_key=None, model_version="fixed", model_family="poisson",
                        artifact_path=None, requested_production_status="active",
                    )
                )

            def get_team_profile(self, team: str) -> dict[str, float]:
                return {"matches": 10.0, "effective_matches": 10.0}

            def predict_match(self, home_team, away_team, *, is_neutral=False, is_knockout=False, max_goals=7, tournament_context=None):
                # Argmax cell is the 1-1 draw, but the home win-region carries more mass.
                matrix = np.zeros((3, 3))
                matrix[1, 1] = 0.50  # draw is the single most-likely scoreline
                matrix[2, 1] = 0.30  # home decisive
                matrix[1, 2] = 0.10  # away decisive
                matrix[0, 0] = 0.10
                return {"prob_home": 0.6, "prob_draw": 0.25, "prob_away": 0.15, "prob_matrix": matrix}

        match = _sample_match(
            DrawArgmaxPredictor(), "Home", "Away",
            rng=np.random.default_rng(0), is_neutral=True, is_knockout=True, deterministic=True,
        )
        # No penalty draw: a decisive home scoreline from the winning region.
        self.assertGreater(match.home_score, match.away_score)
        self.assertEqual(match.winner, "Home")
        self.assertEqual(match.decided_by, "regular_time")


class TestWorldCup2026Bracket(unittest.TestCase):
    def _bracket(self):
        with patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(None)):
            predictor = build_active_predictor(_historical_df(), "2026-05-27")
        return predict_world_cup_2026_bracket(predictor, config=load_world_cup_2026_config())

    def test_bracket_covers_all_phases_and_has_single_champion(self) -> None:
        bracket = self._bracket()
        self.assertEqual(len(bracket["groups"]), 12)
        for stage in ("round_of_32", "round_of_16", "quarter_final", "semi_final", "final"):
            self.assertIn(stage, bracket["knockout"])
            self.assertTrue(bracket["knockout"][stage])
        self.assertEqual(len(bracket["knockout"]["final"]), 1)
        self.assertIsNotNone(bracket["champion"])
        self.assertIsNotNone(bracket["runner_up"])
        self.assertNotEqual(bracket["champion"], bracket["runner_up"])
        # Champion is the winner of the final.
        final_match = bracket["final"]
        self.assertEqual(bracket["champion"], final_match["winner"])

    def test_bracket_is_reproducible(self) -> None:
        first = self._bracket()
        second = self._bracket()
        self.assertEqual(first["champion"], second["champion"])
        self.assertEqual(first["knockout"]["final"], second["knockout"]["final"])


if __name__ == "__main__":
    unittest.main()
