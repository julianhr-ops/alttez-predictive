from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from alttez.inference.router import build_active_predictor


class DummyProbabilityModel:
    def predict_proba(self, frame: pd.DataFrame):
        return [[0.58, 0.19, 0.23] for _ in range(len(frame))]


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

    def test_build_active_predictor_loads_xgboost_artifact_when_available(self) -> None:
        artifact_dir = Path("artifacts/test-router")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "xgboost-active.joblib"
        joblib.dump(
            {
                "model": DummyProbabilityModel(),
                "feature_columns": [
                    "elo_difference",
                    "elo_home_before",
                    "elo_away_before",
                    "is_neutral",
                    "home_recent_points",
                    "away_recent_points",
                    "home_recent_goal_diff",
                    "away_recent_goal_diff",
                    "gdp_per_capita_diff",
                    "home_continent_advantage",
                    "away_continent_advantage",
                    "continent_advantage_diff",
                    "is_knockout",
                ],
                "label_order": ["H", "D", "A"],
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
        self.assertAlmostEqual(float(prediction["prob_home"]), 0.58, places=2)
        self.assertEqual(prediction["best_market"], "Victoria Mexico")

    def test_xgboost_feature_payload_includes_klement_context(self) -> None:
        artifact_dir = Path("artifacts/test-router")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "xgboost-klement.joblib"
        joblib.dump(
            {
                "model": DummyProbabilityModel(),
                "feature_columns": [
                    "gdp_per_capita_diff",
                    "home_continent_advantage",
                    "away_continent_advantage",
                    "continent_advantage_diff",
                    "is_knockout",
                ],
                "label_order": ["H", "D", "A"],
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
        self.assertEqual(payload["gdp_per_capita_diff"], -38000.0)


if __name__ == "__main__":
    unittest.main()
