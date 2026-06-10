from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from alttez.experiments.tournament_oracle import DEFAULT_DIXON_COLES_RHO, build_probability_matrix
from alttez.training.challenger_cycle import (
    DEFAULT_CHALLENGER_VERSION,
    DEFAULT_MAX_GOALS,
    FEATURE_COLUMNS,
    _build_goal_regressor,
    _evaluate_challenger,
    _evaluate_poisson_champion,
    _fit_goal_models,
    _promotion_decision,
    _split_training_and_validation,
    _train_challenger,
)


class FeatureDependentGoalModel:
    """Predicts lambdas that depend on a feature column so batch and per-row
    predictions are non-trivially equal."""

    def __init__(self, base: float, weight: float) -> None:
        self.base = base
        self.weight = weight

    def predict(self, frame):
        return [self.base + self.weight * float(value) for value in frame["elo_difference"].to_numpy()]


def _feature_df(n: int, start: str = "2025-03-01") -> pd.DataFrame:
    rows = []
    for index in range(n):
        rows.append(
            {
                "date": pd.Timestamp(start) + pd.Timedelta(days=index),
                "home_team": "Mexico",
                "away_team": "Brazil",
                "neutral": True,
                "home_score": index % 3,
                "away_score": (index + 1) % 2,
                "elo_difference": float(index - n // 2),
            }
        )
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def _holdout_with_features() -> pd.DataFrame:
    return _feature_df(40)


class FakePoissonEngine:
    fit_calls = 0

    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, df: pd.DataFrame, target_date: str) -> None:
        FakePoissonEngine.fit_calls += 1

    def predict_xg(self, home_team: str, away_team: str, is_neutral: bool = False) -> tuple[float, float]:
        return 1.2, 0.8


class TestChallengerCycle(unittest.TestCase):
    def test_promotion_requires_better_brier(self) -> None:
        champion = {"brier_score": 0.6200, "log_loss": 0.9800}
        challenger = {"brier_score": 0.6100, "log_loss": 0.9810}

        promote, reason = _promotion_decision(champion, challenger)

        self.assertTrue(promote)
        self.assertIn("Brier", reason)

    def test_promotion_rejects_material_log_loss_decay(self) -> None:
        champion = {"brier_score": 0.6200, "log_loss": 0.9800}
        challenger = {"brier_score": 0.6100, "log_loss": 1.0500}

        promote, reason = _promotion_decision(champion, challenger)

        self.assertFalse(promote)
        self.assertIn("Log Loss", reason)

    def test_feature_columns_exclude_gdp_macros(self) -> None:
        self.assertTrue(all(not column.startswith("gdp_per_capita") for column in FEATURE_COLUMNS))

    def test_goal_regressor_enables_early_stopping(self) -> None:
        model = _build_goal_regressor()
        self.assertEqual(model.get_params()["early_stopping_rounds"], 40)
        self.assertEqual(model.get_params()["objective"], "count:poisson")

    def test_train_challenger_emits_schema_v2_artifact(self) -> None:
        with patch(
            "alttez.training.challenger_cycle._fit_goal_models",
            return_value={
                "home_goals": object(),
                "away_goals": object(),
                "best_iteration": {"home_goals": 12, "away_goals": 14},
                "validation_samples": 8,
                "train_samples": 64,
            },
        ):
            artifact = _train_challenger(pd.DataFrame({"date": pd.to_datetime(["2026-01-01"])}))

        self.assertEqual(artifact["artifact_schema_version"], 2)
        self.assertEqual(artifact["model_version"], DEFAULT_CHALLENGER_VERSION)
        self.assertIn("home_goals", artifact["models"])
        self.assertIn("away_goals", artifact["models"])

    def test_poisson_champion_refits_across_holdout_blocks(self) -> None:
        FakePoissonEngine.fit_calls = 0
        train_df = pd.DataFrame(
            [
                ("2025-01-01", "Mexico", "USA", 1, 0, True),
                ("2025-01-05", "Canada", "Brazil", 0, 1, True),
            ],
            columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"],
        )
        train_df["date"] = pd.to_datetime(train_df["date"])

        holdout_rows = []
        for index in range(55):
            holdout_rows.append(
                (
                    pd.Timestamp("2025-02-01") + pd.Timedelta(days=index),
                    "Mexico",
                    "Brazil",
                    index % 3,
                    (index + 1) % 2,
                    True,
                )
            )
        holdout_df = pd.DataFrame(
            holdout_rows,
            columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"],
        )

        with (
            patch("alttez.training.challenger_cycle.PoissonEngine", FakePoissonEngine),
            patch("alttez.training.challenger_cycle.get_active_model_version", return_value="poisson-test"),
            patch("alttez.training.challenger_cycle.get_active_model_family", return_value="poisson"),
        ):
            metrics = _evaluate_poisson_champion(train_df, holdout_df, champion_refit_interval=25)

        self.assertEqual(metrics["refit_count"], 3)
        self.assertEqual(FakePoissonEngine.fit_calls, 3)
        self.assertGreaterEqual(metrics["accuracy_1x2"], 0.0)

    def test_evaluate_challenger_scores_holdout_in_batch(self) -> None:
        holdout_df = _holdout_with_features()
        artifact = {
            "models": {
                "home_goals": FeatureDependentGoalModel(base=1.4, weight=0.02),
                "away_goals": FeatureDependentGoalModel(base=0.9, weight=-0.01),
            },
            "feature_columns": FEATURE_COLUMNS,
            "model_version": "xgboost-challenger-v2",
            "model_family": "xgboost",
            "best_iteration": {"home_goals": 50, "away_goals": 60},
        }

        metrics = _evaluate_challenger(artifact, holdout_df)

        self.assertEqual(metrics["model_family"], "xgboost")
        self.assertIsInstance(metrics["exact_score_hit_rate"], float)
        self.assertGreaterEqual(metrics["accuracy_1x2"], 0.0)
        self.assertLessEqual(metrics["accuracy_1x2"], 1.0)
        # No 1X2 probability mass is lost: brier stays in a valid range.
        self.assertGreaterEqual(metrics["brier_score"], 0.0)
        # Calibration metric is reported and in range.
        self.assertIn("expected_calibration_error", metrics)
        self.assertGreaterEqual(metrics["expected_calibration_error"], 0.0)
        self.assertLessEqual(metrics["expected_calibration_error"], 1.0)

    def test_evaluate_challenger_matches_per_row_reference(self) -> None:
        holdout_df = _holdout_with_features()
        home_model = FeatureDependentGoalModel(base=1.6, weight=0.03)
        away_model = FeatureDependentGoalModel(base=0.7, weight=-0.02)
        artifact = {
            "models": {"home_goals": home_model, "away_goals": away_model},
            "feature_columns": FEATURE_COLUMNS,
            "model_version": "xgboost-challenger-v2",
            "model_family": "xgboost",
            "best_iteration": None,
        }

        batch_metrics = _evaluate_challenger(artifact, holdout_df)

        # Reference: predict each row independently and rebuild its matrix.
        ordered = holdout_df.sort_values("date").reset_index(drop=True)
        reference_brier = []
        for _, row in ordered.iterrows():
            single = pd.DataFrame([row[FEATURE_COLUMNS]])
            lambda_h = max(float(home_model.predict(single)[0]), 0.05)
            lambda_a = max(float(away_model.predict(single)[0]), 0.05)
            matrix = build_probability_matrix(
                lambda_h, lambda_a, max_goals=DEFAULT_MAX_GOALS, rho=DEFAULT_DIXON_COLES_RHO
            )
            import numpy as np

            prob_home = float(np.tril(matrix, -1).sum())
            prob_draw = float(matrix.diagonal().sum())
            prob_away = float(np.triu(matrix, 1).sum())
            actual = "H" if row["home_score"] > row["away_score"] else "A" if row["home_score"] < row["away_score"] else "D"
            target = {"H": prob_home, "D": prob_draw, "A": prob_away}
            reference_brier.append(
                sum((target[label] - (1.0 if label == actual else 0.0)) ** 2 for label in ("H", "D", "A"))
            )

        self.assertAlmostEqual(batch_metrics["brier_score"], sum(reference_brier) / len(reference_brier), places=9)

    def test_split_small_dataset_returns_no_validation(self) -> None:
        small = _feature_df(20)
        core_train, validation_df = _split_training_and_validation(small)
        self.assertIsNone(validation_df)
        self.assertEqual(len(core_train), len(small))

    def test_split_medium_dataset_is_disjoint(self) -> None:
        medium = _feature_df(120)
        core_train, validation_df = _split_training_and_validation(medium)
        self.assertIsNotNone(validation_df)
        # No date appears on both sides of the split (disjoint, date-aligned).
        train_dates = set(core_train["date"])
        validation_dates = set(validation_df["date"])
        self.assertEqual(train_dates & validation_dates, set())
        self.assertGreater(len(validation_df), 0)
        self.assertGreater(len(core_train), 0)

    def test_fit_goal_models_without_early_stopping_on_small_data(self) -> None:
        small = _feature_df(20)
        fitted = _fit_goal_models(small)
        self.assertEqual(fitted["validation_samples"], 0)
        # No early_stopping_rounds means the regressor used all estimators.
        self.assertIsNone(fitted["home_goals"].get_params().get("early_stopping_rounds"))
        self.assertIn("home_goals", fitted)
        self.assertIn("away_goals", fitted)

    def test_goal_regressor_without_early_stopping_drops_param(self) -> None:
        model = _build_goal_regressor(use_early_stopping=False)
        self.assertIsNone(model.get_params().get("early_stopping_rounds"))


if __name__ == "__main__":
    unittest.main()
