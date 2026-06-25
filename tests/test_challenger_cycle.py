from __future__ import annotations

import unittest
from unittest.mock import patch

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
    diebold_mariano,
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

        promote, reason, _detail = _promotion_decision(champion, challenger)

        self.assertTrue(promote)
        self.assertIn("Brier", reason)

    def test_promotion_rejects_material_log_loss_decay(self) -> None:
        champion = {"brier_score": 0.6200, "log_loss": 0.9800}
        challenger = {"brier_score": 0.6100, "log_loss": 1.0500}

        promote, reason, _detail = _promotion_decision(champion, challenger)

        self.assertFalse(promote)
        self.assertIn("Log Loss", reason)

    def test_promotion_blocked_when_not_beating_active_model(self) -> None:
        champion = {"brier_score": 0.6500, "log_loss": 1.0000}
        challenger = {"brier_score": 0.6000, "log_loss": 0.9500}  # beats Poisson champion
        # Active model is already better (0.55) -> must not regress production.
        promote, reason, _detail = _promotion_decision(champion, challenger, incumbent_brier=0.5500)
        self.assertFalse(promote)
        self.assertIn("activo", reason)

    def test_promotion_allowed_when_beating_active_model(self) -> None:
        champion = {"brier_score": 0.6500, "log_loss": 1.0000}
        challenger = {"brier_score": 0.5000, "log_loss": 0.9000}
        promote, reason, _detail = _promotion_decision(champion, challenger, incumbent_brier=0.5500)
        self.assertTrue(promote)

    def test_promotion_allows_small_brier_win_over_active_model(self) -> None:
        champion = {"brier_score": 0.6250, "log_loss": 1.0300}
        challenger = {"brier_score": 0.5433, "log_loss": 0.9170}
        promote, reason, _detail = _promotion_decision(champion, challenger, incumbent_brier=0.5446)
        self.assertTrue(promote)
        self.assertIn("incumbente", reason)

    def test_diebold_mariano_identical_series_is_non_significant(self) -> None:
        series = [0.2, 0.3, 0.25, 0.4, 0.1, 0.35]
        dm_stat, p_value = diebold_mariano(series, series)
        # Diferencia identica -> sin evidencia -> p ~ 1.0 (no significativo).
        self.assertAlmostEqual(p_value, 1.0)
        self.assertEqual(dm_stat, 0.0)

    def test_diebold_mariano_separated_series_is_significant(self) -> None:
        # Modelo A consistentemente con menor perdida que B, con diferencia variable
        # por partido (varianza no nula -> el estadistico DM es genuino).
        loss_a = [0.10, 0.15, 0.08, 0.20, 0.12, 0.09, 0.18, 0.11, 0.14, 0.07]
        loss_b = [0.45, 0.38, 0.50, 0.41, 0.55, 0.39, 0.48, 0.52, 0.40, 0.47]
        dm_stat, p_value = diebold_mariano(loss_a, loss_b)
        # A es mejor -> dm_stat negativo y altamente significativo.
        self.assertLess(dm_stat, 0.0)
        self.assertLess(p_value, 0.05)

    def test_diebold_mariano_handles_short_and_mismatched_series(self) -> None:
        self.assertEqual(diebold_mariano([0.2], [0.3]), (0.0, 1.0))
        self.assertEqual(diebold_mariano([], []), (0.0, 1.0))
        self.assertEqual(diebold_mariano([0.2, 0.3, 0.4], [0.2, 0.3]), (0.0, 1.0))

    def test_promotion_decision_carries_rps_and_dm_fields(self) -> None:
        champion = {"brier_score": 0.6200, "log_loss": 0.9800, "rps": 0.2200}
        challenger = {"brier_score": 0.6100, "log_loss": 0.9810, "rps": 0.2100}
        promote, reason, detail = _promotion_decision(
            champion,
            challenger,
            challenger_brier_losses=[0.10, 0.12, 0.11, 0.09, 0.13],
            incumbent_brier_losses=[0.40, 0.42, 0.41, 0.39, 0.43],
        )
        self.assertTrue(promote)
        self.assertIn("RPS", reason)
        self.assertEqual(detail["challenger_rps"], 0.2100)
        self.assertEqual(detail["champion_rps"], 0.2200)
        self.assertIn("dm_p_value", detail)
        self.assertIsInstance(detail["dm_p_value"], float)

    def test_promotion_blocked_when_rps_worsens_despite_better_brier(self) -> None:
        # Retador mejora Brier nominal pero empeora el RPS ordinal -> NO elegible.
        champion = {"brier_score": 0.6200, "log_loss": 0.9800, "rps": 0.2000}
        challenger = {"brier_score": 0.6100, "log_loss": 0.9790, "rps": 0.2500}
        promote, reason, detail = _promotion_decision(champion, challenger)
        self.assertFalse(promote)
        self.assertIn("RPS", reason)
        self.assertEqual(detail["challenger_rps"], 0.2500)
        self.assertEqual(detail["champion_rps"], 0.2000)

    def test_governance_dm_gate_blocks_auto_promote_when_insignificant(self) -> None:
        # En la ruta de auto-promocion, un p-valor DM no significativo cae a manual.
        with (
            patch("alttez.training.challenger_cycle.auto_promotion_enabled", return_value=True),
            patch(
                "alttez.training.challenger_cycle.production_supported_model_families",
                return_value={"xgboost"},
            ),
        ):
            status, mode, reason = _governance_decision(
                promote=True, challenger_family="xgboost", dm_p_value=0.30
            )
            self.assertEqual(status, "eligible")
            self.assertEqual(mode, "manual")
            self.assertIn("Diebold-Mariano", reason)

            # p significativo -> se permite la auto-promocion.
            status2, mode2, _reason2 = _governance_decision(
                promote=True, challenger_family="xgboost", dm_p_value=0.01
            )
            self.assertEqual(status2, "active")
            self.assertEqual(mode2, "auto")

    def test_governance_dm_gate_does_not_affect_manual_path(self) -> None:
        # Con auto-promocion OFF, el p-valor DM es irrelevante (no cambia comportamiento).
        with (
            patch("alttez.training.challenger_cycle.auto_promotion_enabled", return_value=False),
            patch(
                "alttez.training.challenger_cycle.production_supported_model_families",
                return_value={"xgboost"},
            ),
        ):
            status, mode, _reason = _governance_decision(
                promote=True, challenger_family="xgboost", dm_p_value=0.30
            )
            self.assertEqual(status, "eligible")
            self.assertEqual(mode, "manual")

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

    def test_lambda_blend_is_no_worse_than_either_endpoint(self) -> None:
        from alttez.training.challenger_cycle import _score_lambda_stream, evaluate_lambda_blend

        holdout = _feature_df(40)
        n = len(holdout)
        champion = [(1.2, 1.0)] * n
        challenger = [(1.8, 0.6)] * n

        blend = evaluate_lambda_blend(champion, challenger, holdout)
        champion_only = _score_lambda_stream(champion, holdout, model_version="p", model_family="poisson")
        challenger_only = _score_lambda_stream(challenger, holdout, model_version="x", model_family="xgboost")

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
