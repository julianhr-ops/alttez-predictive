from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from alttez.inference.calibration import matrix_1x2, temperature_scale
from alttez.inference.prediction_committee import (
    CommitteeMember,
    PredictionCommittee,
    _resolve_member_weights,
    load_latest_challenger_record,
)
from alttez.inference.router import ActiveModelSelection, BaseMatchPredictor, most_likely_score_from_matrix
from alttez.operations.prediction_memory import _filter_match_prediction_row, paired_member_losses


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
    def test_latest_challenger_record_skips_missing_local_artifact(self) -> None:
        registry_df = pd.DataFrame(
            [
                {
                    "version_key": "xgboost-newer",
                    "model_version": "xgboost-challenger-v2",
                    "model_family": "xgboost",
                    "status": "challenger",
                    "artifact_path": "artifacts/model-artifacts/missing.joblib",
                },
                {
                    "version_key": "xgboost-json",
                    "model_version": "xgboost-challenger-v2",
                    "model_family": "xgboost",
                    "status": "challenger",
                    "artifact_path": "artifacts/model-evals/report.json",
                },
                {
                    "version_key": "xgboost-valid",
                    "model_version": "xgboost-challenger-v2",
                    "model_family": "xgboost",
                    "status": "challenger",
                    "artifact_path": "artifacts/model-artifacts/xgboost-valid.joblib",
                },
            ]
        )

        def exists_side_effect(path: Path) -> bool:
            return path.name in {"report.json", "xgboost-valid.joblib"}

        with (
            patch("alttez.inference.prediction_committee.read_dataframe", return_value=registry_df),
            patch("alttez.inference.prediction_committee.Path.exists", exists_side_effect),
            patch("alttez.inference.prediction_committee.joblib.load", return_value={"model": object()}),
        ):
            record = load_latest_challenger_record()

        self.assertIsNotNone(record)
        self.assertEqual(record["version_key"], "xgboost-valid")

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
        self.assertEqual(prediction["score_selection_policy"], "committee_most_likely_outcome_exact_score")
        self.assertTrue(prediction["primary_score_scenarios"])
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

    def _predict_with_artifact(self, payload: dict | None) -> dict:
        import json
        import os
        import tempfile

        committee_members = [CommitteeMember("champion", DummyPredictor(score=(1, 0), model_version="champion"), 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            env = {}
            if payload is not None:
                artifact = Path(tmp) / "active.json"
                artifact.write_text(json.dumps(payload), encoding="utf-8")
                env["ALTTEZ_CALIBRATION_PATH"] = str(artifact)
            else:
                env["ALTTEZ_CALIBRATION_PATH"] = str(Path(tmp) / "missing.json")
            with patch.dict(os.environ, env):
                return PredictionCommittee(committee_members).predict_match("United States", "Mexico")

    def test_committee_applies_active_calibration_artifact_to_1x2(self) -> None:
        raw_prediction = self._predict_with_artifact(None)
        calibrated = self._predict_with_artifact({"temperature": 3.0, "class_floor": 0.0})

        self.assertFalse(raw_prediction["calibration_applied"])
        self.assertTrue(calibrated["calibration_applied"])
        self.assertEqual(calibrated["calibration_temperature"], 3.0)
        raw = np.array([calibrated[f"raw_prob_{side}"] for side in ("home", "draw", "away")])
        np.testing.assert_allclose(raw, matrix_1x2(raw_prediction["prob_matrix"]), atol=1e-9)
        np.testing.assert_allclose(
            matrix_1x2(calibrated["prob_matrix"]), temperature_scale(raw, 3.0), atol=1e-9
        )
        served = [calibrated["prob_home"], calibrated["prob_draw"], calibrated["prob_away"]]
        np.testing.assert_allclose(served, matrix_1x2(calibrated["prob_matrix"]), atol=1e-9)

    def test_uncalibrated_committee_reports_raw_equal_to_served(self) -> None:
        prediction = self._predict_with_artifact(None)

        for side in ("home", "draw", "away"):
            self.assertAlmostEqual(prediction[f"raw_prob_{side}"], prediction[f"prob_{side}"], places=6)

    def test_committee_propagates_member_fallback(self) -> None:
        champion = DummyPredictor(score=(1, 0), model_version="champion")
        champion.used_fallback = True
        champion.fallback_reason = "model registry unavailable: db down"
        committee = PredictionCommittee(
            [
                CommitteeMember("champion", champion, 0.6),
                CommitteeMember("poisson_baseline", DummyPredictor(score=(1, 0), model_version="poisson"), 0.4),
            ]
        )

        prediction = committee.predict_match("United States", "Mexico")

        self.assertTrue(committee.used_fallback)
        self.assertTrue(prediction["used_fallback"])
        self.assertIn("champion", prediction["fallback_reason"])
        self.assertIn("db down", prediction["fallback_reason"])

    def test_committee_without_fallback_members_reports_none(self) -> None:
        committee = PredictionCommittee(
            [CommitteeMember("champion", DummyPredictor(score=(1, 0), model_version="champion"), 1.0)]
        )

        prediction = committee.predict_match("United States", "Mexico")

        self.assertFalse(prediction["used_fallback"])
        self.assertIsNone(prediction["fallback_reason"])

    def test_member_probabilities_survive_persistence_and_pair_with_consensus(self) -> None:
        committee = PredictionCommittee(
            [
                CommitteeMember("champion", DummyPredictor(score=(1, 0), model_version="champion-v1"), 0.7),
                CommitteeMember("poisson_baseline", DummyPredictor(score=(0, 1), model_version="poisson"), 0.3),
            ]
        )
        prediction = committee.predict_match("United States", "Mexico")
        champion_entry = next(item for item in prediction["committee_members"] if item["role"] == "champion")
        np.testing.assert_allclose(
            [champion_entry["prob_home"], champion_entry["prob_draw"], champion_entry["prob_away"]],
            matrix_1x2(DummyPredictor(score=(1, 0), model_version="x").predict_match("a", "b")["prob_matrix"]),
            atol=1e-9,
        )

        row = _filter_match_prediction_row(
            {
                **prediction,
                "match_id": "m1",
                "home_team": "United States",
                "away_team": "Mexico",
                "kickoff_at": "2026-06-01T18:00:00+00:00",
                "target_date": "2026-06-01",
                "generated_at": "2026-06-01T08:00:00+00:00",
            },
            "run-row",
            "run-1",
            "committee-consensus-v1",
            "committee",
        )
        frame = pd.DataFrame([{**row, "prob_home": row["predicted_home_win_prob"], "prob_draw": row["predicted_draw_prob"], "prob_away": row["predicted_away_win_prob"], "actual_outcome": "H"}])

        consensus_losses, champion_losses = paired_member_losses(
            frame, member_role="champion", member_version="champion-v1"
        )

        self.assertEqual(len(consensus_losses), 1)
        self.assertEqual(len(champion_losses), 1)
        self.assertGreater(consensus_losses[0], champion_losses[0])

    def _members_with_versions(self) -> list[CommitteeMember]:
        return [
            CommitteeMember("champion", DummyPredictor(score=(1, 0), model_version="champion-v1"), 0.5),
            CommitteeMember("poisson_baseline", DummyPredictor(score=(1, 0), model_version="poisson"), 0.5),
        ]

    def _audit(self) -> dict:
        return {
            "models": [
                {"model_version": "champion-v1", "evaluated_predictions": 40, "brier_score": 0.55},
                {"model_version": "poisson", "evaluated_predictions": 40, "brier_score": 0.60},
            ]
        }

    def test_resolve_member_weights_defers_to_champion_when_paired_losses_show_consensus_worse(self) -> None:
        rng = np.random.default_rng(0)
        champion_losses = 0.4 + rng.uniform(-0.02, 0.02, 40)
        consensus_losses = champion_losses + 0.2 + rng.uniform(-0.02, 0.02, 40)
        with (
            patch("alttez.operations.prediction_memory.load_prediction_learning_audit", return_value=self._audit()),
            patch(
                "alttez.operations.prediction_memory.load_committee_paired_losses",
                return_value=(consensus_losses, champion_losses),
            ) as loader,
        ):
            decision = _resolve_member_weights(self._members_with_versions(), database_url=None)

        self.assertEqual(loader.call_args.kwargs["member_version"], "champion-v1")
        self.assertEqual(loader.call_args.kwargs["consensus_version"], "committee-consensus-v1")
        self.assertTrue(decision.deferred_to_champion)
        self.assertEqual(decision.strategy, "champion_guard")
        self.assertEqual(decision.weights, {"champion": 1.0, "poisson_baseline": 0.0})

    def test_resolve_member_weights_keeps_blend_when_consensus_is_not_worse(self) -> None:
        rng = np.random.default_rng(1)
        champion_losses = 0.4 + rng.uniform(-0.05, 0.05, 40)
        consensus_losses = champion_losses - 0.01 + rng.uniform(-0.05, 0.05, 40)
        with (
            patch("alttez.operations.prediction_memory.load_prediction_learning_audit", return_value=self._audit()),
            patch(
                "alttez.operations.prediction_memory.load_committee_paired_losses",
                return_value=(consensus_losses, champion_losses),
            ),
        ):
            decision = _resolve_member_weights(self._members_with_versions(), database_url=None)

        self.assertFalse(decision.deferred_to_champion)
        self.assertEqual(decision.strategy, "inverse_brier")
        self.assertNotIn("no evaluada", decision.reason)

    def test_unreadable_paired_losses_leave_guard_unevaluated_and_warn(self) -> None:
        with (
            patch("alttez.operations.prediction_memory.load_prediction_learning_audit", return_value=self._audit()),
            patch(
                "alttez.operations.prediction_memory.load_committee_paired_losses",
                side_effect=RuntimeError("db down"),
            ),
            self.assertLogs("alttez.inference.prediction_committee", level="WARNING") as logs,
        ):
            decision = _resolve_member_weights(self._members_with_versions(), database_url=None)

        self.assertIn("db down", logs.output[0])
        self.assertFalse(decision.deferred_to_champion)
        self.assertIn("no evaluada", decision.reason)




if __name__ == "__main__":
    unittest.main()
