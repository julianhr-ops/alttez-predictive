from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alttez.inference.prediction_committee import CommitteeMember, PredictionCommittee
from alttez.inference.router import ActiveModelSelection, BaseMatchPredictor, most_likely_score_from_matrix


class DummyPredictor(BaseMatchPredictor):
    def __init__(self, *, score: tuple[int, int], model_version: str) -> None:
        super().__init__(
            selection=ActiveModelSelection(
                version_key=None,
                model_version=model_version,
                model_family="dummy",
                artifact_path=None,
                requested_production_status="test",
            )
        )
        self.score = score

    def get_team_profile(self, team: str) -> dict[str, float]:
        return {
            "matches": 20.0,
            "effective_matches": 12.0,
            "confidence": 0.8,
            "recent_points_per_match": 1.5,
            "recent_goal_diff_per_match": 0.2,
        }

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        is_neutral: bool = False,
        is_knockout: bool = False,
        max_goals: int = 7,
        tournament_context: dict | None = None,
    ) -> dict:
        matrix = np.full((max_goals + 1, max_goals + 1), 0.001, dtype=float)
        matrix[self.score[0], self.score[1]] = 0.8
        matrix = matrix / matrix.sum()
        primary_score, primary_probability = most_likely_score_from_matrix(matrix)
        return {
            "model_version": self.model_version,
            "model_family": self.model_family,
            "profile_local": self.get_team_profile(home_team),
            "profile_visitante": self.get_team_profile(away_team),
            "prediction_supported": True,
            "support_message": None,
            "support_label": "Analizable",
            "lambda_h": float(self.score[0] + 0.5),
            "lambda_a": float(self.score[1] + 0.5),
            "prob_matrix": matrix,
            "prob_home": 0.5,
            "prob_draw": 0.2,
            "prob_away": 0.3,
            "result_summary": pd.DataFrame(),
            "recommendations": pd.DataFrame([{"Pronostico": primary_score, "EV": 4.0}]),
            "primary_score": primary_score,
            "primary_score_probability": primary_probability,
            "primary_ev": 4.0,
            "ev_optimal_score": primary_score,
            "ev_optimal_value": 4.0,
            "score_selection_policy": "dummy",
            "best_market": f"Victoria {home_team}",
            "best_market_probability": 0.5,
        }


class TestPredictionCommittee(unittest.TestCase):
    def test_committee_builds_weighted_consensus_matrix(self) -> None:
        committee = PredictionCommittee(
            [
                CommitteeMember("champion", DummyPredictor(score=(1, 0), model_version="champion"), 0.5),
                CommitteeMember("challenger", DummyPredictor(score=(2, 0), model_version="challenger"), 0.3),
                CommitteeMember("poisson_baseline", DummyPredictor(score=(1, 0), model_version="poisson"), 0.2),
            ]
        )

        prediction = committee.predict_match("United States", "Mexico")

        self.assertTrue(prediction["committee_consensus_used"])
        self.assertEqual(prediction["primary_score"], "1 - 0")
        self.assertFalse(prediction["discordance_warning"])
        self.assertEqual(prediction["score_selection_policy"], "committee_consensus_matrix")
        self.assertEqual(len(prediction["committee_members"]), 3)

    def test_committee_flags_radical_score_discordance(self) -> None:
        committee = PredictionCommittee(
            [
                CommitteeMember("champion", DummyPredictor(score=(1, 0), model_version="champion"), 0.5),
                CommitteeMember("challenger", DummyPredictor(score=(0, 2), model_version="challenger"), 0.3),
                CommitteeMember("poisson_baseline", DummyPredictor(score=(1, 1), model_version="poisson"), 0.2),
            ]
        )

        prediction = committee.predict_match("USA", "Mexico")

        self.assertTrue(prediction["discordance_warning"])
        self.assertIn("0 - 2", prediction["member_primary_scores"])


if __name__ == "__main__":
    unittest.main()
