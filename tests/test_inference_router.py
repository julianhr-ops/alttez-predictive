from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from alttez.features.klement import MODEL_FEATURE_COLUMNS
from alttez.inference.router import XGBOOST_FEATURE_COLUMNS, _is_prediction_supported, build_active_predictor, most_likely_score_from_matrix
from alttez.training.challenger_cycle import FEATURE_COLUMNS as CHALLENGER_FEATURE_COLUMNS


class TestFeatureColumnContract(unittest.TestCase):
    def test_single_source_of_feature_columns(self) -> None:
        # Trainer and router must share the exact same object so they cannot drift.
        self.assertIs(CHALLENGER_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS)
        self.assertIs(XGBOOST_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS)


class TestPredictionGating(unittest.TestCase):
    def test_requires_both_teams_to_clear_support_threshold(self) -> None:
        strong = {"matches": 10.0, "effective_matches": 8.0}
        weak = {"matches": 2.0, "effective_matches": 1.0}
        with patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "5"}):
            # One weak team is no longer carried by the strong one.
            self.assertFalse(_is_prediction_supported(strong, weak)[0])
            self.assertFalse(_is_prediction_supported(weak, strong)[0])
            self.assertTrue(_is_prediction_supported(strong, strong)[0])

    def test_zero_effective_support_is_rejected(self) -> None:
        no_recent = {"matches": 20.0, "effective_matches": 0.0}
        ok = {"matches": 20.0, "effective_matches": 5.0}
        with patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "5"}):
            self.assertFalse(_is_prediction_supported(ok, no_recent)[0])


class DummyProbabilityModel:
    def predict_proba(self, frame: pd.DataFrame):
        return [[0.58, 0.19, 0.23] for _ in range(len(frame))]


class DummyGoalModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame):
        return [self.value for _ in range(len(frame))]


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
    frame = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


class TestInferenceRouter(unittest.TestCase):
    def test_build_active_predictor_defaults_to_poisson_when_no_registry_record(self) -> None:
        with patch("alttez.inference.router._load_active_model_record_safe", return_value=None):
            predictor = build_active_predictor(_historical_df(), "2026-05-27")
            prediction = predictor.predict_match("Mexico", "USA", is_neutral=True)

        self.assertEqual(predictor.model_family, "poisson")
        self.assertFalse(predictor.used_fallback)
        self.assertTrue(prediction["prediction_supported"])
        self.assertAlmostEqual(
            float(prediction["prob_home"] or 0.0) + float(prediction["prob_draw"] or 0.0) + float(prediction["prob_away"] or 0.0),
            1.0,
            places=3,
        )

    def test_build_active_predictor_loads_schema_v2_xgboost_artifact_when_available(self) -> None:
        artifact_dir = Path("artifacts/test-router")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "xgboost-active.joblib"
        joblib.dump(
            {
                "artifact_schema_version": 2,
                "models": {
                    "home_goals": DummyGoalModel(1.8),
                    "away_goals": DummyGoalModel(0.9),
                },
                "feature_columns": [
                    "elo_difference",
                    "elo_home_before",
                    "elo_away_before",
                    "is_neutral",
                    "home_recent_points",
                    "away_recent_points",
                    "home_recent_goal_diff",
                    "away_recent_goal_diff",
                    "home_continent_advantage",
                    "away_continent_advantage",
                    "continent_advantage_diff",
                    "is_knockout",
                ],
                "model_version": "xgboost-active-v1",
                "model_family": "xgboost",
            },
            artifact_path,
        )
        active_record = {
            "version_key": "xgboost-active-v1:2026-05-27",
            "model_version": "xgboost-active-v1",
            "model_family": "xgboost",
            "artifact_path": str(artifact_path),
            "production_status": "active",
        }
        with patch("alttez.inference.router._load_active_model_record_safe", return_value=active_record):
            predictor = build_active_predictor(_historical_df(), "2026-05-27")
            prediction = predictor.predict_match("Mexico", "USA", is_neutral=True)

        self.assertEqual(predictor.model_family, "xgboost")
        self.assertFalse(predictor.used_fallback)
        self.assertEqual(prediction["model_version"], "xgboost-active-v1")
        self.assertEqual(prediction["artifact_schema_version"], 2)
        self.assertGreater(float(prediction["lambda_h"]), float(prediction["lambda_a"]))
        self.assertAlmostEqual(
            float(prediction["prob_home"] or 0.0) + float(prediction["prob_draw"] or 0.0) + float(prediction["prob_away"] or 0.0),
            1.0,
            places=6,
        )
        self.assertEqual(prediction["prob_matrix"].shape, (8, 8))
        expected_score, expected_probability = most_likely_score_from_matrix(prediction["prob_matrix"])
        self.assertEqual(prediction["primary_score"], expected_score)
        self.assertAlmostEqual(prediction["primary_score_probability"], expected_probability)
        self.assertEqual(prediction["ev_optimal_score"], prediction["recommendations"].iloc[0]["Pronostico"])

    def test_xgboost_feature_payload_includes_klement_context(self) -> None:
        artifact_dir = Path("artifacts/test-router")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "xgboost-klement.joblib"
        joblib.dump(
            {
                "artifact_schema_version": 2,
                "models": {
                    "home_goals": DummyGoalModel(1.4),
                    "away_goals": DummyGoalModel(0.6),
                },
                "feature_columns": [
                    "home_continent_advantage",
                    "away_continent_advantage",
                    "continent_advantage_diff",
                    "is_knockout",
                ],
                "model_version": "xgboost-klement-v1",
                "model_family": "xgboost",
            },
            artifact_path,
        )
        active_record = {
            "version_key": "xgboost-klement-v1:2026-05-27",
            "model_version": "xgboost-klement-v1",
            "model_family": "xgboost",
            "artifact_path": str(artifact_path),
            "production_status": "active",
        }
        macro_context = {
            "mexico": {"gdp_per_capita": 12000.0, "confederation_code": "CONCACAF"},
            "germany": {"gdp_per_capita": 50000.0, "confederation_code": "UEFA"},
        }
        with (
            patch("alttez.inference.router._load_active_model_record_safe", return_value=active_record),
            patch("alttez.features.klement.load_team_macro_context", return_value=macro_context),
            patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0"}),
        ):
            predictor = build_active_predictor(_historical_df(), "2026-05-27")
            prediction = predictor.predict_match(
                "Mexico",
                "Germany",
                is_neutral=True,
                is_knockout=True,
                tournament_context={"tournament": "FIFA World Cup 2026", "edition": 2026},
            )

        payload = prediction["xgboost_feature_payload"]
        self.assertEqual(payload["is_knockout"], 1)
        self.assertEqual(payload["home_continent_advantage"], 1.0)
        self.assertEqual(payload["away_continent_advantage"], 0.0)
        self.assertEqual(payload["continent_advantage_diff"], 1.0)
        self.assertNotIn("gdp_per_capita_diff", payload)


if __name__ == "__main__":
    unittest.main()
