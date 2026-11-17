from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import poisson

from alttez.inference.calibration import (
    ProbabilityCalibration,
    fit_temperature,
    log_loss,
    matrix_1x2,
    rescale_matrix_to_triplet,
    temperature_scale,
)
from scripts.evaluate_calibration import (
    closed_predictions_query,
    load_closed_predictions,
)
from scripts.fit_probability_calibration import select_fit_triplets


def _closed_row(match_id, created, **overrides) -> dict:
    row = {
        "match_id": match_id,
        "model_version": "committee-consensus-v1",
        "prediction_created_at": created,
        "kickoff_at": "2026-06-01T18:00:00+00:00",
        "prob_home": 0.5,
        "prob_draw": 0.3,
        "prob_away": 0.2,
        "raw_prob_home": 0.6,
        "raw_prob_draw": 0.25,
        "raw_prob_away": 0.15,
        "actual_outcome": "H",
    }
    row.update(overrides)
    return row


class TestSelectFitTriplets(unittest.TestCase):
    def test_raw_probabilities_are_used_when_present(self) -> None:
        frame = pd.DataFrame([_closed_row("m1", "2026-05-30T08:00:00+00:00")])

        triplets, raw_rows, legacy_rows = select_fit_triplets(frame)

        np.testing.assert_allclose(triplets, [[0.6, 0.25, 0.15]])
        self.assertEqual((raw_rows, legacy_rows), (1, 0))

    def test_legacy_rows_fall_back_to_stored_probabilities(self) -> None:
        frame = pd.DataFrame(
            [
                _closed_row("m1", "2026-05-30T08:00:00+00:00"),
                _closed_row("m2", "2026-05-30T08:00:00+00:00", raw_prob_home=None, raw_prob_draw=None, raw_prob_away=None),
                _closed_row("m3", "2026-05-30T08:00:00+00:00", raw_prob_draw=None),
            ]
        )

        triplets, raw_rows, legacy_rows = select_fit_triplets(frame)

        np.testing.assert_allclose(triplets[0], [0.6, 0.25, 0.15])
        np.testing.assert_allclose(triplets[1], [0.5, 0.3, 0.2])
        np.testing.assert_allclose(triplets[2], [0.5, 0.3, 0.2])
        self.assertEqual((raw_rows, legacy_rows), (1, 2))

    def test_frame_without_raw_columns_is_all_legacy(self) -> None:
        frame = pd.DataFrame([_closed_row("m1", "2026-05-30T08:00:00+00:00")]).drop(
            columns=["raw_prob_home", "raw_prob_draw", "raw_prob_away"]
        )

        triplets, raw_rows, legacy_rows = select_fit_triplets(frame)

        np.testing.assert_allclose(triplets, [[0.5, 0.3, 0.2]])
        self.assertEqual((raw_rows, legacy_rows), (0, 1))


class TestLoadClosedPredictions(unittest.TestCase):
    def _duplicated_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _closed_row("m1", "2026-05-29T08:00:00+00:00", raw_prob_home=0.1),
                _closed_row("m1", "2026-05-30T08:00:00+00:00", raw_prob_home=0.6),
                _closed_row("m1", "2026-06-02T08:00:00+00:00", raw_prob_home=0.99),
                _closed_row("m2", "2026-05-30T08:00:00+00:00"),
            ]
        )

    def test_query_selects_raw_columns_or_null_placeholders(self) -> None:
        self.assertIn("mp.raw_predicted_home_win_prob", closed_predictions_query(include_raw=True))
        legacy = closed_predictions_query(include_raw=False)
        self.assertNotIn("raw_predicted", legacy)
        self.assertIn("null::numeric as raw_prob_home", legacy)

    def test_one_prediction_per_match_and_model_is_loaded(self) -> None:
        with patch("scripts.evaluate_calibration.read_dataframe", return_value=self._duplicated_frame()):
            frame = load_closed_predictions(database_url=None, lookback_days=365)

        self.assertEqual(sorted(frame["match_id"]), ["m1", "m2"])
        self.assertEqual(frame[frame["match_id"] == "m1"].iloc[0]["raw_prob_home"], 0.6)

    def test_missing_raw_columns_fall_back_to_legacy_query_with_warning(self) -> None:
        legacy = self._duplicated_frame().assign(raw_prob_home=None, raw_prob_draw=None, raw_prob_away=None)
        reader = patch(
            "scripts.evaluate_calibration.read_dataframe",
            side_effect=[RuntimeError("column raw_predicted_home_win_prob does not exist"), legacy],
        )
        with reader as mocked, self.assertLogs("scripts.evaluate_calibration", level=logging.WARNING) as logs:
            frame = load_closed_predictions(database_url=None, lookback_days=365)

        self.assertEqual(len(frame), 2)
        self.assertNotIn("raw_predicted", mocked.call_args_list[1].args[0])
        self.assertIn("014", logs.output[0])


def _poisson_matrix(lambda_home: float, lambda_away: float, max_goals: int = 8) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))
    return matrix / matrix.sum()


def _overconfident_dataset(n: int, seed: int, sharpening: float = 1.8):
    """Partidos Poisson cuya matriz servida es una version sobreconfiada de la verdad."""
    rng = np.random.default_rng(seed)
    matrices, outcomes = [], []
    for _ in range(n):
        true_matrix = _poisson_matrix(rng.uniform(0.6, 2.8), rng.uniform(0.4, 2.2))
        true_triplet = np.array(matrix_1x2(true_matrix))
        outcomes.append("HDA"[rng.choice(3, p=true_triplet)])
        sharpened = true_triplet**sharpening
        matrices.append(rescale_matrix_to_triplet(true_matrix, sharpened / sharpened.sum()))
    return matrices, outcomes


def _brier(triplets: np.ndarray, outcomes: list[str]) -> float:
    targets = np.array([[float(o == c) for c in "HDA"] for o in outcomes])
    return float(np.mean(np.sum((triplets - targets) ** 2, axis=1)))


def heldout_calibration_report(*, n_train: int = 1500, n_test: int = 1500, seed: int = 11) -> dict[str, float]:
    train_matrices, train_outcomes = _overconfident_dataset(n_train, seed)
    test_matrices, test_outcomes = _overconfident_dataset(n_test, seed + 1)

    temperature = fit_temperature(np.array([matrix_1x2(m) for m in train_matrices]), train_outcomes)
    calibration = ProbabilityCalibration(temperature=temperature)

    before = np.array([matrix_1x2(m) for m in test_matrices])
    after = np.array([matrix_1x2(calibration.calibrate_matrix(m)) for m in test_matrices])
    return {
        "temperature": temperature,
        "log_loss_before": log_loss(before, test_outcomes),
        "log_loss_after": log_loss(after, test_outcomes),
        "brier_before": _brier(before, test_outcomes),
        "brier_after": _brier(after, test_outcomes),
    }


class TestHeldOutCalibration(unittest.TestCase):
    def test_matrix_calibration_does_not_worsen_heldout_log_loss_and_brier(self) -> None:
        report = heldout_calibration_report()

        self.assertGreater(report["temperature"], 1.0)
        self.assertLess(report["log_loss_after"], report["log_loss_before"])
        self.assertLess(report["brier_after"], report["brier_before"])

    def test_calibrated_matrix_marginals_match_temperature_scaled_triplet(self) -> None:
        matrix = _poisson_matrix(1.9, 0.8)
        calibration = ProbabilityCalibration(temperature=1.8)

        np.testing.assert_allclose(
            matrix_1x2(calibration.calibrate_matrix(matrix)),
            temperature_scale(matrix_1x2(matrix), 1.8),
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
