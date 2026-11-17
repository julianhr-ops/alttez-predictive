from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from alttez.experiments.tournament_oracle import DEFAULT_DIXON_COLES_RHO, build_probability_matrix
from alttez.training.challenger_cycle import (
    DEFAULT_CHALLENGER_VERSION,
    DEFAULT_MAX_GOALS,
    DEFAULT_XGB_TIME_DECAY_HALFLIFE_DAYS,
    FEATURE_COLUMNS,
    XGB_TIME_DECAY_HALFLIFE_ENV,
    _build_goal_regressor,
    _compute_time_decay_weights,
    _evaluate_challenger,
    _evaluate_poisson_champion,
    _fit_goal_models,
    _governance_decision,
    _promotion_decision,
    _resolve_time_decay_half_life,
    _split_training_and_validation,
    _train_challenger,
)
from alttez.training.production_champion import ProductionChampionConfig


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


def _noisy_losses(center: float, n: int) -> np.ndarray:
    return np.random.default_rng(1).normal(center, 0.2, n)


def _metrics(losses: np.ndarray, *, log_loss: float, rps: float | None = None) -> dict:
    return {
        "brier_losses": [float(value) for value in losses],
        "brier_score": float(np.mean(losses)),
        "log_loss": log_loss,
        "rps": rps,
    }


class FakePoissonEngine:
    fit_calls = 0

    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, df: pd.DataFrame, target_date: str) -> None:
        FakePoissonEngine.fit_calls += 1

    def predict_xg(self, home_team: str, away_team: str, is_neutral: bool = False) -> tuple[float, float]:
        return 1.2, 0.8


class TestChallengerCycle(unittest.TestCase):
    def test_promotion_requires_ci_entirely_below_zero(self) -> None:
        champion = _metrics(_noisy_losses(0.62, 300), log_loss=0.98)
        challenger = _metrics(_noisy_losses(0.62, 300) - 0.05, log_loss=0.98)

        promote, reason, detail = _promotion_decision(champion, challenger)

        self.assertTrue(promote)
        self.assertIn("Brier", reason)
        self.assertLess(detail["paired_brier"]["ci_high"], 0.0)
        self.assertEqual(detail["paired_brier"]["n"], 300)

    def test_promotion_rejects_better_mean_when_ci_crosses_zero(self) -> None:
        # The challenger is better on average, but by far less than the per-match noise.
        noise = np.random.default_rng(3).normal(0.0, 0.2, 300)
        champion_losses = _noisy_losses(0.62, 300)
        challenger_losses = champion_losses + (noise - noise.mean()) - 0.002
        self.assertLess(challenger_losses.mean(), champion_losses.mean())

        promote, reason, detail = _promotion_decision(
            _metrics(champion_losses, log_loss=0.98), _metrics(challenger_losses, log_loss=0.98)
        )

        self.assertFalse(promote)
        self.assertLess(detail["paired_brier"]["mean_diff"], 0.0)
        self.assertGreater(detail["paired_brier"]["ci_high"], 0.0)
        self.assertIn("significancia", reason)

    def test_promotion_rejects_small_sample_even_with_clear_win(self) -> None:
        champion = _metrics(_noisy_losses(0.62, 50), log_loss=0.98)
        challenger = _metrics(_noisy_losses(0.62, 50) - 0.05, log_loss=0.98)

        promote, reason, detail = _promotion_decision(champion, challenger)

        self.assertFalse(promote)
        self.assertTrue(detail["paired_brier"]["a_better"])
        self.assertIn("Muestra insuficiente", reason)

    def test_promotion_rejects_series_from_different_match_sets(self) -> None:
        champion = _metrics(_noisy_losses(0.62, 300), log_loss=0.98)
        challenger = _metrics(_noisy_losses(0.62, 250) - 0.05, log_loss=0.98)

        with self.assertRaises(ValueError):
            _promotion_decision(champion, challenger)

    def test_promotion_rejects_material_log_loss_decay(self) -> None:
        champion = _metrics(_noisy_losses(0.62, 300), log_loss=0.98)
        challenger = _metrics(_noisy_losses(0.62, 300) - 0.05, log_loss=1.05)

        promote, reason, _detail = _promotion_decision(champion, challenger)

        self.assertFalse(promote)
        self.assertIn("Log Loss", reason)

    def test_promotion_decision_carries_rps_fields(self) -> None:
        champion = _metrics(_noisy_losses(0.62, 300), log_loss=0.98, rps=0.22)
        challenger = _metrics(_noisy_losses(0.62, 300) - 0.05, log_loss=0.98, rps=0.21)

        promote, reason, detail = _promotion_decision(champion, challenger)

        self.assertTrue(promote)
        self.assertIn("RPS", reason)
        self.assertEqual(detail["challenger_rps"], 0.21)
        self.assertEqual(detail["champion_rps"], 0.22)

    def test_promotion_blocked_when_rps_worsens_despite_better_brier(self) -> None:
        champion = _metrics(_noisy_losses(0.62, 300), log_loss=0.98, rps=0.20)
        challenger = _metrics(_noisy_losses(0.62, 300) - 0.05, log_loss=0.979, rps=0.25)

        promote, reason, detail = _promotion_decision(champion, challenger)

        self.assertFalse(promote)
        self.assertIn("RPS", reason)
        self.assertEqual(detail["challenger_rps"], 0.25)
        self.assertEqual(detail["champion_rps"], 0.20)

    def test_governance_auto_path_activates_when_enabled_and_supported(self) -> None:
        with (
            patch("alttez.training.challenger_cycle.auto_promotion_enabled", return_value=True),
            patch(
                "alttez.training.challenger_cycle.production_supported_model_families",
                return_value={"xgboost"},
            ),
        ):
            status, mode, _reason = _governance_decision(promote=True, challenger_family="xgboost")
        self.assertEqual((status, mode), ("active", "auto"))

    def test_governance_manual_path_when_auto_promotion_off(self) -> None:
        with (
            patch("alttez.training.challenger_cycle.auto_promotion_enabled", return_value=False),
            patch(
                "alttez.training.challenger_cycle.production_supported_model_families",
                return_value={"xgboost"},
            ),
        ):
            status, mode, _reason = _governance_decision(promote=True, challenger_family="xgboost")
        self.assertEqual((status, mode), ("eligible", "manual"))

    def test_feature_columns_include_gdp_with_coverage_flag(self) -> None:
        # GDP restored as a static macro prior, paired with a coverage flag.
        self.assertIn("gdp_per_capita_diff", FEATURE_COLUMNS)
        self.assertIn("gdp_coverage", FEATURE_COLUMNS)

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

        config = ProductionChampionConfig(
            version_key=None,
            model_version="poisson-test",
            model_family="poisson",
            poisson_kwargs={"years_window": 4},
            rho=-0.1,
        )
        with patch("alttez.training.challenger_cycle.PoissonEngine", FakePoissonEngine):
            metrics = _evaluate_poisson_champion(train_df, holdout_df, config=config, champion_refit_interval=25)

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

        metrics = _evaluate_challenger(artifact, holdout_df, rho=DEFAULT_DIXON_COLES_RHO)

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
        self.assertIn("regulation_points_avg", metrics)
        self.assertIn("regulation_points_total", metrics)
        self.assertIn("regulation_points_distribution", metrics)
        self.assertGreaterEqual(metrics["regulation_points_avg"], 0.0)
        self.assertLessEqual(metrics["regulation_points_pct_of_max"], 1.0)

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

        batch_metrics = _evaluate_challenger(artifact, holdout_df, rho=DEFAULT_DIXON_COLES_RHO)

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

    def test_lambda_blend_is_no_worse_than_either_endpoint(self) -> None:
        from alttez.training.challenger_cycle import _score_lambda_stream, evaluate_lambda_blend

        holdout = _feature_df(40)
        n = len(holdout)
        champion = [(1.2, 1.0)] * n
        challenger = [(1.8, 0.6)] * n

        blend = evaluate_lambda_blend(champion, challenger, holdout, rho=DEFAULT_DIXON_COLES_RHO)
        champion_only = _score_lambda_stream(
            champion, holdout, rho=DEFAULT_DIXON_COLES_RHO, model_version="p", model_family="poisson"
        )
        challenger_only = _score_lambda_stream(
            challenger, holdout, rho=DEFAULT_DIXON_COLES_RHO, model_version="x", model_family="xgboost"
        )

        self.assertEqual(blend["model_family"], "ensemble")
        self.assertIn(blend["blend_weight"], [round(0.1 * i, 1) for i in range(11)])
        # Grid includes w=0 (champion) and w=1 (challenger); best blend cannot be worse.
        self.assertLessEqual(blend["brier_score"], champion_only["brier_score"] + 1e-9)
        self.assertLessEqual(blend["brier_score"], challenger_only["brier_score"] + 1e-9)

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

    def test_time_decay_weights_decrease_into_the_past(self) -> None:
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01"]))
        weights = _compute_time_decay_weights(dates, pd.Timestamp("2025-01-01"), 365.0)
        # Estrictamente crecientes hacia el presente.
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        # El partido en la fecha de corte pesa ~1.0.
        self.assertAlmostEqual(weights[2], 1.0)
        # Exactamente un half-life atras -> ~0.5.
        self.assertAlmostEqual(weights[0], 0.5, places=2)

    def test_time_decay_weights_neutral_when_half_life_non_positive(self) -> None:
        dates = pd.Series(pd.to_datetime(["2020-01-01", "2025-01-01"]))
        weights = _compute_time_decay_weights(dates, pd.Timestamp("2025-01-01"), 0.0)
        self.assertTrue((weights == 1.0).all())

    def test_resolve_half_life_default_and_override(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop(XGB_TIME_DECAY_HALFLIFE_ENV, None)
            self.assertEqual(
                _resolve_time_decay_half_life(), float(DEFAULT_XGB_TIME_DECAY_HALFLIFE_DAYS)
            )
        with patch.dict("os.environ", {XGB_TIME_DECAY_HALFLIFE_ENV: "180"}):
            self.assertEqual(_resolve_time_decay_half_life(), 180.0)
        with patch.dict("os.environ", {XGB_TIME_DECAY_HALFLIFE_ENV: "garbage"}):
            self.assertEqual(
                _resolve_time_decay_half_life(), float(DEFAULT_XGB_TIME_DECAY_HALFLIFE_DAYS)
            )

    def test_fit_goal_models_reports_half_life(self) -> None:
        fitted = _fit_goal_models(_feature_df(60))
        self.assertEqual(
            fitted["time_decay_half_life_days"], float(DEFAULT_XGB_TIME_DECAY_HALFLIFE_DAYS)
        )


if __name__ == "__main__":
    unittest.main()
