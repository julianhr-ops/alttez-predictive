from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from alttez.experiments.tournament_oracle import optimize_prediction
from alttez.features.klement import MODEL_FEATURE_COLUMNS
from alttez.inference.calibration import temperature_scale
from alttez.inference.router import (
    XGBOOST_FEATURE_COLUMNS,
    ActiveRecordLookup,
    ArtifactPathError,
    _is_prediction_supported,
    _load_active_model_record_safe,
    build_active_predictor,
    exact_scoreline_decision,
    most_likely_score_from_matrix,
    outcome_aligned_score_from_matrix,
    resolve_artifact_path,
)
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
    frame["tournament"] = "Friendly"
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


class TestInferenceRouter(unittest.TestCase):
    def test_exact_scoreline_decision_aligns_with_most_likely_outcome(self) -> None:
        matrix = pd.DataFrame(
            [
                [0.10, 0.03, 0.01],
                [0.18, 0.20, 0.02],
                [0.17, 0.16, 0.13],
            ]
        ).to_numpy(dtype=float)
        matrix = matrix / matrix.sum()
        recommendations = optimize_prediction(matrix).head(5)

        decision = exact_scoreline_decision(matrix, recommendations)

        self.assertEqual(decision["most_likely_score"], "1 - 1")
        self.assertEqual(decision["primary_outcome"], "H")
        self.assertEqual(decision["primary_score"], "1 - 0")

    def test_build_active_predictor_defaults_to_poisson_when_no_registry_record(self) -> None:
        with (
            patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(None)),
            patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0"}),
        ):
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
        with (
            patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(active_record)),
            patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0"}),
        ):
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
        raw_score, raw_probability = most_likely_score_from_matrix(prediction["prob_matrix"])
        expected_score, expected_probability, _ = outcome_aligned_score_from_matrix(prediction["prob_matrix"])
        self.assertEqual(prediction["primary_score"], expected_score)
        self.assertAlmostEqual(prediction["primary_score_probability"], expected_probability)
        self.assertEqual(prediction["most_likely_score"], raw_score)
        self.assertAlmostEqual(prediction["most_likely_score_probability"], raw_probability)
        self.assertEqual(prediction["ev_optimal_score"], prediction["recommendations"].iloc[0]["Pronostico"])
        self.assertEqual(prediction["score_selection_policy"], "most_likely_outcome_exact_score")
        self.assertTrue(prediction["primary_score_scenarios"])

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
            patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(active_record)),
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


class TestSinglePredictorCalibration(unittest.TestCase):
    def _max_prob(self, prediction) -> float:
        return max(
            float(prediction["prob_home"] or 0.0),
            float(prediction["prob_draw"] or 0.0),
            float(prediction["prob_away"] or 0.0),
        )

    def test_calibration_flattens_single_predictor_when_enabled(self) -> None:
        import json
        import tempfile

        with patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(None)):
            # Baseline sin calibracion (compuerta apagada).
            with patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0", "ALTTEZ_SERVING_POLICY": "off"}, clear=False):
                os.environ.pop("ALTTEZ_CALIBRATE_SINGLE_PREDICTOR", None)
                base_pred = build_active_predictor(_historical_df(), "2026-05-27").predict_match(
                    "Mexico", "USA", is_neutral=True
                )
            base_max = self._max_prob(base_pred)

            # Con calibracion T>1 (aplana) y artefacto separado.
            with tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp) / "single_predictor.json"
                artifact.write_text(json.dumps({"temperature": 4.0, "class_floor": 0.0}), encoding="utf-8")
                with patch.dict(
                    os.environ,
                    {
                        "ALTTEZ_MIN_SUPPORTED_MATCHES": "0",
                        "ALTTEZ_SERVING_POLICY": "off",
                        "ALTTEZ_CALIBRATE_SINGLE_PREDICTOR": "true",
                        "ALTTEZ_SINGLE_PREDICTOR_CALIBRATION_PATH": str(artifact),
                    },
                    clear=False,
                ):
                    cal_pred = build_active_predictor(_historical_df(), "2026-05-27").predict_match(
                        "Mexico", "USA", is_neutral=True
                    )
            cal_max = self._max_prob(cal_pred)

        # T=4 aplana el 1X2 agregado: la clase mas probable pierde masa.
        self.assertLess(cal_max, base_max)
        sides = ("home", "draw", "away")
        raw = [float(cal_pred[f"raw_prob_{side}"]) for side in sides]
        for side, raw_value in zip(sides, raw, strict=False):
            self.assertAlmostEqual(raw_value, float(base_pred[f"prob_{side}"]), places=6)
            self.assertAlmostEqual(float(base_pred[f"raw_prob_{side}"]), float(base_pred[f"prob_{side}"]), places=6)
        calibrated = [float(cal_pred[f"prob_{side}"]) for side in sides]
        for expected, actual in zip(temperature_scale(np.array(raw), 4.0), calibrated, strict=False):
            self.assertAlmostEqual(float(expected), actual, places=6)
        # Sigue siendo distribucion valida.
        self.assertAlmostEqual(
            float(cal_pred["prob_home"]) + float(cal_pred["prob_draw"]) + float(cal_pred["prob_away"]),
            1.0,
            places=3,
        )

    def test_calibration_neutral_when_flag_off(self) -> None:
        with patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(None)):
            with patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0"}, clear=False):
                os.environ.pop("ALTTEZ_CALIBRATE_SINGLE_PREDICTOR", None)
                predictor = build_active_predictor(_historical_df(), "2026-05-27")
        self.assertTrue(predictor.calibration.is_neutral)


class TestArtifactPathContainment(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("artifacts").resolve()

    def test_path_inside_artifacts_is_accepted(self) -> None:
        resolved = resolve_artifact_path("artifacts/model-artifacts/champion.joblib")
        self.assertTrue(resolved.is_relative_to(self.base))

    def test_parent_traversal_is_rejected(self) -> None:
        with self.assertRaises(ArtifactPathError):
            resolve_artifact_path("artifacts/../../etc/passwd.joblib")
        with self.assertRaises(ArtifactPathError):
            resolve_artifact_path("../../etc/x.joblib")

    def test_absolute_path_outside_is_rejected(self) -> None:
        import tempfile

        with self.assertRaises(ArtifactPathError):
            resolve_artifact_path(Path(tempfile.gettempdir()) / "evil.joblib")

    def test_sibling_directory_with_shared_prefix_is_rejected(self) -> None:
        with self.assertRaises(ArtifactPathError):
            resolve_artifact_path("artifacts-evil/x.joblib")

    def test_custom_artifacts_dir_from_environment(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ALTTEZ_ARTIFACTS_DIR": tmp}):
            inside = resolve_artifact_path(Path(tmp) / "m.joblib")
            self.assertEqual(inside.parent, Path(tmp).resolve())
            with self.assertRaises(ArtifactPathError):
                resolve_artifact_path("artifacts/m.joblib")


class TestLoudFallback(unittest.TestCase):
    def _build(self, lookup: ActiveRecordLookup):
        with (
            patch("alttez.inference.router._load_active_model_record_safe", return_value=lookup),
            patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0"}),
        ):
            return build_active_predictor(_historical_df(), "2026-05-27")

    def test_registry_error_sets_fallback_flags_and_reaches_prediction(self) -> None:
        predictor = self._build(ActiveRecordLookup(None, error="OperationalError: db down"))
        self.assertTrue(predictor.used_fallback)
        self.assertIn("db down", predictor.fallback_reason)
        prediction = predictor.predict_match("Mexico", "USA", is_neutral=True)
        self.assertTrue(prediction["used_fallback"])
        self.assertIn("model registry unavailable", prediction["fallback_reason"])

    def test_healthy_registry_without_active_record_is_not_a_fallback(self) -> None:
        predictor = self._build(ActiveRecordLookup(None))
        self.assertFalse(predictor.used_fallback)
        self.assertIsNone(predictor.fallback_reason)

    def test_registry_lookup_logs_warning_and_reports_error(self) -> None:
        with (
            patch("alttez.operations.prediction_memory.read_dataframe", side_effect=RuntimeError("db down")),
            self.assertLogs("alttez.inference.router", level="WARNING") as logs,
        ):
            lookup = _load_active_model_record_safe()
        self.assertIsNone(lookup.record)
        self.assertIn("RuntimeError: db down", lookup.error)
        self.assertTrue(any("db down" in line for line in logs.output))

    def test_missing_artifact_sets_fallback_and_logs_warning(self) -> None:
        record = {
            "version_key": None,
            "model_version": "xgb-missing",
            "model_family": "xgboost",
            "artifact_path": "artifacts/test-router/does-not-exist.joblib",
            "production_status": "active",
        }
        with self.assertLogs("alttez.inference.router", level="WARNING") as logs:
            predictor = self._build(ActiveRecordLookup(record))
        self.assertTrue(predictor.used_fallback)
        self.assertIn("FileNotFoundError", predictor.fallback_reason)
        self.assertTrue(any("xgb-missing" in line for line in logs.output))

    def test_artifact_path_traversal_falls_back_without_loading(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside.joblib"
            joblib.dump({"artifact_schema_version": 1, "model": DummyProbabilityModel()}, outside)
            for artifact_path in (str(outside), "artifacts/../../escape.joblib"):
                record = {
                    "version_key": None,
                    "model_version": "xgb-evil",
                    "model_family": "xgboost",
                    "artifact_path": artifact_path,
                    "production_status": "active",
                }
                with patch("alttez.inference.router.joblib.load") as load, self.assertLogs(
                    "alttez.inference.router", level="WARNING"
                ):
                    predictor = self._build(ActiveRecordLookup(record))
                load.assert_not_called()
                self.assertTrue(predictor.used_fallback)
                self.assertIn("ArtifactPathError", predictor.fallback_reason)


if __name__ == "__main__":
    unittest.main()
