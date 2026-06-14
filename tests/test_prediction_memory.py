from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from alttez.operations.prediction_memory import (
    activate_model_version_by_key,
    brier_score_1x2,
    load_prediction_learning_audit,
    log_loss_1x2,
)


class TestPredictionMemory(unittest.TestCase):
    def test_brier_score_rewards_calibrated_prediction(self) -> None:
        confident_correct = brier_score_1x2(0.75, 0.15, 0.10, "H")
        flat_prediction = brier_score_1x2(0.34, 0.33, 0.33, "H")

        self.assertLess(confident_correct, flat_prediction)

    def test_log_loss_penalizes_wrong_overconfidence(self) -> None:
        calibrated = log_loss_1x2(0.60, 0.25, 0.15, "H")
        wrong_overconfident = log_loss_1x2(0.05, 0.10, 0.85, "H")

        self.assertLess(calibrated, wrong_overconfident)

    def test_learning_audit_summarizes_prediction_evaluations(self) -> None:
        df = pd.DataFrame(
            [
                {"model_version": "xgb-v1", "brier_1x2": 0.20, "log_loss_1x2": 0.5, "hit_1x2": True, "hit_exact_score": False},
                {"model_version": "xgb-v1", "brier_1x2": 0.40, "log_loss_1x2": 0.9, "hit_1x2": False, "hit_exact_score": True},
            ]
        )
        with patch("alttez.operations.prediction_memory.read_dataframe", return_value=df):
            audit = load_prediction_learning_audit()

        self.assertEqual(audit["evaluated_predictions"], 2)
        self.assertAlmostEqual(audit["brier_score"], 0.30)
        self.assertAlmostEqual(audit["accuracy_1x2"], 0.50)
        self.assertAlmostEqual(audit["exact_score_hit_rate"], 0.50)
        self.assertEqual(audit["models"][0]["model_version"], "xgb-v1")

    def test_activation_rejects_missing_non_poisson_artifact(self) -> None:
        record = {
            "model_family": "xgboost",
            "artifact_path": "artifacts/model-artifacts/missing-model.joblib",
        }
        with (
            patch("alttez.operations.prediction_memory.get_engine"),
            patch("alttez.operations.prediction_memory.reflect_prediction_tables", return_value={"model_versions": object()}),
            patch("alttez.operations.prediction_memory.load_model_version_record", return_value=record),
        ):
            result = activate_model_version_by_key("xgb-missing")

        self.assertFalse(result["activated"])
        self.assertIn("artifact", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
