from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pandas as pd

from alttez.inference.router import (
    DEFAULT_POISSON_KWARGS,
    DEFAULT_POISSON_MODEL_VERSION,
)
from alttez.training.challenger_cycle import (
    FEATURE_COLUMNS,
    _baseline_comparison,
    _champion_lambda_stream,
    _evaluate_poisson_champion,
    _gated_promotion_decision,
    _promotion_decision,
    _score_lambda_stream,
    run_challenger_cycle,
)
from alttez.training.production_champion import (
    ProductionChampionConfig,
    ProductionConfigUnavailable,
    resolve_production_champion_config,
)

FITTED_RHO = "alttez.training.production_champion.load_fitted_dixon_coles_rho"


class RecordingEngine:
    instances: ClassVar[list[dict]] = []

    def __init__(self, **kwargs) -> None:
        RecordingEngine.instances.append(kwargs)

    def fit(self, df: pd.DataFrame, target_date: str) -> None:
        pass

    def predict_xg(self, home_team: str, away_team: str, is_neutral: bool = False) -> tuple[float, float]:
        return 1.3, 0.9


class ConstantGoalModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame):
        return [self.value] * len(frame)


def _matches(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp(start) + pd.Timedelta(days=index),
            "home_team": "Mexico",
            "away_team": "Brazil",
            "neutral": index % 2 == 0,
            "home_score": index % 4,
            "away_score": (index * 3 + 1) % 3,
            "elo_difference": float((index % 11) * 20 - 100),
        }
        for index in range(n)
    ]
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def _config(**overrides) -> ProductionChampionConfig:
    values = {
        "version_key": "poisson-champion-v2:2026-01-01",
        "model_version": "poisson-champion-v2",
        "model_family": "poisson",
        "poisson_kwargs": {"years_window": 7, "prior_matches": 3},
        "rho": -0.25,
    }
    values.update(overrides)
    return ProductionChampionConfig(**values)


class TestProductionChampionConfig(unittest.TestCase):
    def test_empty_registry_resolves_to_router_default_with_fitted_rho(self) -> None:
        with patch(FITTED_RHO, return_value=-0.08):
            config = resolve_production_champion_config(None)

        self.assertEqual(config.model_version, DEFAULT_POISSON_MODEL_VERSION)
        self.assertEqual(dict(config.poisson_kwargs), DEFAULT_POISSON_KWARGS)
        self.assertEqual(config.rho, -0.08)

    def test_active_poisson_record_is_carried_into_config(self) -> None:
        record = {"model_family": "poisson", "model_version": "poisson-champion-v3", "version_key": "k"}
        with patch(FITTED_RHO, return_value=-0.1):
            config = resolve_production_champion_config(record)

        self.assertEqual((config.model_version, config.version_key), ("poisson-champion-v3", "k"))

    def test_refuses_when_active_model_is_not_poisson(self) -> None:
        with patch(FITTED_RHO, return_value=-0.1), self.assertRaises(ProductionConfigUnavailable):
            resolve_production_champion_config({"model_family": "xgboost", "model_version": "x"})

    def test_refuses_when_fitted_rho_is_unavailable(self) -> None:
        with patch(FITTED_RHO, return_value=None), self.assertRaises(ProductionConfigUnavailable):
            resolve_production_champion_config(None)


class TestChampionStreamUsesProductionConfig(unittest.TestCase):
    def test_engine_is_built_from_config_kwargs(self) -> None:
        RecordingEngine.instances = []
        with patch("alttez.training.challenger_cycle.PoissonEngine", RecordingEngine):
            _champion_lambda_stream(_matches(5), _matches(3, "2024-03-01"), config=_config())

        self.assertEqual(RecordingEngine.instances, [{"years_window": 7, "prior_matches": 3}])

    def test_champion_is_scored_with_config_rho(self) -> None:
        train, holdout = _matches(5), _matches(30, "2024-03-01")
        with patch("alttez.training.challenger_cycle.PoissonEngine", RecordingEngine):
            metrics = _evaluate_poisson_champion(train, holdout, config=_config(rho=-0.25))

        lambdas = [(1.3, 0.9)] * len(holdout)
        at_config_rho = _score_lambda_stream(lambdas, holdout, rho=-0.25, model_version="m", model_family="poisson")
        at_default_rho = _score_lambda_stream(lambdas, holdout, rho=-0.13, model_version="m", model_family="poisson")
        self.assertEqual(metrics["brier_score"], at_config_rho["brier_score"])
        self.assertNotEqual(metrics["brier_score"], at_default_rho["brier_score"])
        self.assertEqual(metrics["model_version"], "poisson-champion-v2")


class TestBaselineComparison(unittest.TestCase):
    def test_reports_all_three_baselines_with_paired_diffs_on_same_matches(self) -> None:
        train, holdout = _matches(120), _matches(60, "2024-06-01")
        champion_losses = [0.5] * len(holdout)
        challenger_losses = [0.4] * len(holdout)

        report = _baseline_comparison(train, holdout, champion_losses, challenger_losses)

        self.assertEqual(set(report), {"uniform", "base_rate", "elo_only"})
        for entry in report.values():
            self.assertEqual(entry["champion_vs_baseline"]["n"], len(holdout))
            self.assertEqual(entry["challenger_vs_baseline"]["n"], len(holdout))
        self.assertAlmostEqual(report["uniform"]["brier_score"], 2.0 / 3.0)
        self.assertTrue(report["uniform"]["champion_vs_baseline"]["a_better"])


def _loss_metrics(losses: np.ndarray, *, log_loss: float) -> dict:
    return {"brier_losses": [float(value) for value in losses], "log_loss": log_loss, "rps": None}


class TestPromotionAgainstTheServedRule(unittest.TestCase):
    def _series(self, challenger_vs_elo: float) -> tuple[dict, dict, dict]:
        rng = np.random.default_rng(7)
        elo = rng.normal(0.4961, 0.2, 300)
        elo += 0.4961 - elo.mean()
        poisson = elo + 0.0599 + rng.normal(0.0, 0.01, 300)
        challenger = elo + challenger_vs_elo + rng.normal(0.0, 0.01, 300)
        return (
            _loss_metrics(poisson, log_loss=1.05),
            _loss_metrics(challenger, log_loss=1.00),
            _loss_metrics(elo, log_loss=0.95),
        )

    def test_a_challenger_that_beats_poisson_but_not_the_served_elo_is_not_promoted(self) -> None:
        champion, challenger, elo = self._series(challenger_vs_elo=0.0294)

        plain_promote, _, plain_detail = _promotion_decision(champion, challenger)
        promote, reason, detail = _gated_promotion_decision(champion, challenger, elo)

        self.assertTrue(plain_promote)
        self.assertLess(plain_detail["paired_brier"]["ci_high"], 0.0)
        self.assertFalse(promote)
        self.assertIn("Elo", reason)
        self.assertGreater(detail["elo_only_paired_brier"]["ci_low"], 0.0)
        self.assertGreaterEqual(detail["elo_only_paired_brier"]["n"], 200)

    def test_a_challenger_that_beats_both_references_is_promoted(self) -> None:
        champion, challenger, elo = self._series(challenger_vs_elo=-0.03)
        challenger["log_loss"] = elo["log_loss"] - 0.01

        promote, reason, detail = _gated_promotion_decision(champion, challenger, elo)

        self.assertTrue(promote, reason)
        self.assertLess(detail["paired_brier"]["ci_high"], 0.0)
        self.assertLess(detail["elo_only_paired_brier"]["ci_high"], 0.0)

    def test_a_log_loss_decay_against_elo_blocks_promotion(self) -> None:
        champion, challenger, elo = self._series(challenger_vs_elo=-0.03)
        challenger["log_loss"] = elo["log_loss"] + 0.05

        promote, reason, _ = _gated_promotion_decision(champion, challenger, elo)

        self.assertFalse(promote)
        self.assertIn("Elo", reason)
        self.assertIn("Log Loss", reason)


class TestRunChallengerCycle(unittest.TestCase):
    def _run(self, *, active_record, fitted_rho, n_rows=400, **kwargs):
        artifact = {
            "models": {"home_goals": ConstantGoalModel(1.8), "away_goals": ConstantGoalModel(0.7)},
            "feature_columns": FEATURE_COLUMNS,
            "model_version": "xgboost-challenger-v2",
            "model_family": "xgboost",
            "best_iteration": None,
        }
        RecordingEngine.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "model-evals"
            with (
                patch("alttez.training.challenger_cycle.load_active_model_record", return_value=active_record),
                patch(FITTED_RHO, return_value=fitted_rho),
                patch("alttez.training.challenger_cycle.load_enriched_historical_matches", return_value=_matches(n_rows)),
                patch("alttez.training.challenger_cycle._build_challenger_dataset", side_effect=lambda df, **_: df),
                patch("alttez.training.challenger_cycle._train_challenger", return_value=artifact),
                patch("alttez.training.challenger_cycle.PoissonEngine", RecordingEngine),
                patch("alttez.training.challenger_cycle.get_engine", side_effect=RuntimeError("no database in tests")),
            ):
                report = run_challenger_cycle(date(2026, 1, 1), report_dir=report_dir, **kwargs)
            written = json.loads((report_dir / "challenger-cycle-2026-01-01.json").read_text(encoding="utf-8"))
        return report, written

    def test_cycle_refuses_and_says_so_when_production_config_is_unavailable(self) -> None:
        with patch("alttez.training.challenger_cycle.load_enriched_historical_matches") as load_matches:
            report, written = self._run(active_record=None, fitted_rho=None)
            load_matches.assert_not_called()

        self.assertFalse(report["promotion_decision"])
        self.assertIn("rho", report["production_config_error"])
        self.assertIn("rechazado", report["promotion_reason"])
        self.assertEqual(report["activation_mode"], "blocked")
        self.assertNotIn("champion", report)
        self.assertEqual(written["production_config_error"], report["production_config_error"])

    def test_cycle_refuses_when_active_model_is_not_reproducible(self) -> None:
        report, _ = self._run(active_record={"model_family": "xgboost"}, fitted_rho=-0.1)

        self.assertFalse(report["promotion_decision"])
        self.assertIn("xgboost", report["production_config_error"])

    def test_cycle_uses_production_config_and_reports_paired_comparison_and_baselines(self) -> None:
        report, written = self._run(active_record=None, fitted_rho=-0.07)

        self.assertEqual(report["production_config"]["rho"], -0.07)
        self.assertEqual(report["production_config"]["poisson_kwargs"], DEFAULT_POISSON_KWARGS)
        self.assertEqual(RecordingEngine.instances[0], DEFAULT_POISSON_KWARGS)
        self.assertEqual(report["champion"]["model_version"], DEFAULT_POISSON_MODEL_VERSION)
        self.assertEqual(report["paired_comparison"]["n"], report["evaluation_matches"])
        self.assertEqual(set(report["baselines"]), {"uniform", "base_rate", "elo_only"})
        self.assertFalse(report["promotion_decision"])
        self.assertIn("Muestra insuficiente", report["promotion_reason"])
        self.assertEqual(written["baselines"].keys(), report["baselines"].keys())
        self.assertTrue(np.isfinite(report["paired_comparison"]["mean_diff"]))

    def test_cycle_report_carries_the_elo_comparison(self) -> None:
        report, written = self._run(active_record=None, fitted_rho=-0.07)

        self.assertEqual(report["elo_comparison"]["n"], report["evaluation_matches"])
        self.assertEqual(written["elo_comparison"], report["elo_comparison"])


if __name__ == "__main__":
    unittest.main()
