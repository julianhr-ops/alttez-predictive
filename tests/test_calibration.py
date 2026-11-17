from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from alttez.inference.calibration import (
    NEUTRAL_CALIBRATION,
    IsotonicCalibration,
    ProbabilityCalibration,
    apply_class_floor,
    expected_calibration_error,
    fit_isotonic,
    fit_temperature,
    load_active_calibration,
    log_loss,
    matrix_1x2,
    rescale_matrix_to_triplet,
    temperature_scale,
)


def _score_matrix(max_goals: int = 7, lambda_h: float = 1.7, lambda_a: float = 0.9) -> np.ndarray:
    from scipy.stats import poisson

    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lambda_h), poisson.pmf(goals, lambda_a))
    return matrix / matrix.sum()


class TestMatrixCalibration(unittest.TestCase):
    def test_marginals_equal_calibrated_triplet_when_temperature_not_one(self) -> None:
        matrix = _score_matrix()
        for temperature in (0.6, 1.5, 3.0):
            cal = ProbabilityCalibration(temperature=temperature)
            calibrated = cal.calibrate_matrix(matrix)
            expected = cal.calibrate_triplet(*matrix_1x2(matrix))
            np.testing.assert_allclose(matrix_1x2(calibrated), expected, atol=1e-9)
            self.assertAlmostEqual(float(calibrated.sum()), 1.0, places=9)
            self.assertGreaterEqual(float(calibrated.min()), 0.0)

    def test_temperature_one_is_identity(self) -> None:
        matrix = _score_matrix()
        np.testing.assert_allclose(ProbabilityCalibration(temperature=1.0).calibrate_matrix(matrix), matrix, atol=1e-12)

    def test_class_floor_is_respected_by_the_matrix(self) -> None:
        matrix = _score_matrix(lambda_h=4.0, lambda_a=0.3)
        cal = ProbabilityCalibration(temperature=1.0, class_floor=0.05)
        self.assertGreaterEqual(min(matrix_1x2(cal.calibrate_matrix(matrix))), 0.05 - 1e-9)

    def test_within_class_shape_is_preserved(self) -> None:
        matrix = _score_matrix()
        calibrated = ProbabilityCalibration(temperature=2.0).calibrate_matrix(matrix)
        ratio = calibrated / matrix
        home_cells = np.indices(matrix.shape)[0] > np.indices(matrix.shape)[1]
        np.testing.assert_allclose(ratio[home_cells], ratio[home_cells][0], rtol=1e-9)

    def test_softening_lowers_the_top_class_of_the_matrix(self) -> None:
        matrix = _score_matrix(lambda_h=3.0, lambda_a=0.5)
        raw = matrix_1x2(matrix)
        calibrated = matrix_1x2(ProbabilityCalibration(temperature=3.0).calibrate_matrix(matrix))
        self.assertLess(max(calibrated), max(raw))

    def test_class_without_mass_does_not_break_rescale(self) -> None:
        out = rescale_matrix_to_triplet(np.array([[1.0]]), (0.5, 0.3, 0.2))
        np.testing.assert_allclose(out, [[1.0]])


class TestTemperatureScale(unittest.TestCase):
    def test_identity_at_temperature_one(self) -> None:
        probs = np.array([0.6, 0.25, 0.15])
        np.testing.assert_allclose(temperature_scale(probs, 1.0), probs, atol=1e-9)

    def test_high_temperature_softens_toward_uniform(self) -> None:
        probs = np.array([0.8, 0.15, 0.05])
        softened = temperature_scale(probs, 3.0)
        # Max class drops, min class rises; still sums to 1.
        self.assertLess(softened[0], probs[0])
        self.assertGreater(softened[2], probs[2])
        self.assertAlmostEqual(float(softened.sum()), 1.0, places=9)

    def test_low_temperature_sharpens(self) -> None:
        probs = np.array([0.5, 0.3, 0.2])
        sharp = temperature_scale(probs, 0.5)
        self.assertGreater(sharp[0], probs[0])

    def test_matrix_renormalizes(self) -> None:
        matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
        scaled = temperature_scale(matrix, 2.0)
        self.assertAlmostEqual(float(scaled.sum()), 1.0, places=9)

    def test_degenerate_input_returns_uniform(self) -> None:
        probs = np.zeros(3)
        out = temperature_scale(probs, 1.5)
        np.testing.assert_allclose(out, np.full(3, 1 / 3), atol=1e-9)


class TestClassFloor(unittest.TestCase):
    def test_floor_lifts_low_class_and_renormalizes(self) -> None:
        out = apply_class_floor(np.array([0.97, 0.02, 0.01]), 0.05)
        self.assertGreaterEqual(out.min(), 0.05 - 1e-9)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=9)

    def test_zero_floor_is_identity(self) -> None:
        probs = np.array([0.5, 0.3, 0.2])
        np.testing.assert_allclose(apply_class_floor(probs, 0.0), probs, atol=1e-9)


class TestProbabilityCalibration(unittest.TestCase):
    def test_neutral_is_neutral(self) -> None:
        self.assertTrue(NEUTRAL_CALIBRATION.is_neutral)
        self.assertFalse(ProbabilityCalibration(temperature=1.5).is_neutral)

    def test_calibrate_triplet_sums_to_one(self) -> None:
        cal = ProbabilityCalibration(temperature=1.5, class_floor=0.03)
        h, d, a = cal.calibrate_triplet(0.8, 0.15, 0.05)
        self.assertAlmostEqual(h + d + a, 1.0, places=9)
        self.assertGreaterEqual(min(h, d, a), 0.03 - 1e-9)


class TestFitTemperature(unittest.TestCase):
    def test_overconfident_data_fits_temperature_above_one(self) -> None:
        # Confident-but-often-wrong probabilities -> needs softening (T > 1).
        rng = np.random.default_rng(0)
        triplets, outcomes = [], []
        for _ in range(300):
            if rng.random() < 0.5:
                triplets.append([0.85, 0.10, 0.05])  # confident home
                outcomes.append("H" if rng.random() < 0.55 else "A")  # but only 55% right
            else:
                triplets.append([0.05, 0.10, 0.85])
                outcomes.append("A" if rng.random() < 0.55 else "H")
        temp = fit_temperature(np.array(triplets), outcomes)
        self.assertGreater(temp, 1.0)

    def test_empty_returns_neutral(self) -> None:
        self.assertEqual(fit_temperature(np.empty((0, 3)), []), 1.0)


class TestMetrics(unittest.TestCase):
    def test_log_loss_uniform_is_about_ln3(self) -> None:
        triplets = np.tile([1 / 3, 1 / 3, 1 / 3], (10, 1))
        outcomes = ["H"] * 10
        self.assertAlmostEqual(log_loss(triplets, outcomes), np.log(3), places=4)

    def test_ece_perfect_calibration_is_low(self) -> None:
        # Always 100% on the realized class -> zero calibration gap.
        triplets = np.array([[1.0, 0.0, 0.0]] * 5)
        self.assertAlmostEqual(expected_calibration_error(triplets, ["H"] * 5), 0.0, places=6)


def _miscalibrated_dataset(n: int = 800, seed: int = 7):
    """Genera tripletes sobreconfiados: prob alta en una clase pero acierta ~55%."""
    rng = np.random.default_rng(seed)
    triplets, outcomes = [], []
    for _ in range(n):
        which = rng.integers(0, 3)
        base = [0.05, 0.05, 0.05]
        base[which] = 0.90
        triplets.append(base)
        # Solo acierta 55% de las veces -> sobreconfianza marcada.
        if rng.random() < 0.55:
            outcomes.append("HDA"[which])
        else:
            other = (which + rng.integers(1, 3)) % 3
            outcomes.append("HDA"[other])
    return np.array(triplets, dtype=float), outcomes


class TestIsotonicCalibration(unittest.TestCase):
    def test_neutral_when_empty(self) -> None:
        cal = IsotonicCalibration()
        self.assertTrue(cal.is_neutral)
        h, d, a = cal.calibrate_triplet(0.7, 0.2, 0.1)
        # Neutra normaliza pero no deforma (ya suma 1).
        self.assertAlmostEqual(h + d + a, 1.0, places=9)
        self.assertAlmostEqual(h, 0.7, places=9)

    def test_calibrate_matrix_marginals_match_calibrated_triplet(self) -> None:
        cal = fit_isotonic(*_miscalibrated_dataset())
        matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
        calibrated = cal.calibrate_matrix(matrix)
        self.assertAlmostEqual(float(calibrated.sum()), 1.0, places=9)
        np.testing.assert_allclose(
            matrix_1x2(calibrated), cal.calibrate_triplet(*matrix_1x2(matrix)), atol=1e-9
        )

    def test_neutral_calibrate_matrix_is_identity(self) -> None:
        matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_allclose(IsotonicCalibration().calibrate_matrix(matrix), matrix, atol=1e-12)

    def test_fit_reduces_log_loss_vs_raw(self) -> None:
        triplets, outcomes = _miscalibrated_dataset()
        cal = fit_isotonic(triplets, outcomes)
        self.assertFalse(cal.is_neutral)
        before = log_loss(triplets, outcomes)
        calibrated = np.array([cal.calibrate_triplet(*row) for row in triplets])
        after = log_loss(calibrated, outcomes)
        self.assertLess(after, before)

    def test_triplet_sums_to_one(self) -> None:
        cal = fit_isotonic(*_miscalibrated_dataset())
        h, d, a = cal.calibrate_triplet(0.9, 0.05, 0.05)
        self.assertAlmostEqual(h + d + a, 1.0, places=9)

    def test_serialization_round_trips_via_arrays(self) -> None:
        cal = fit_isotonic(*_miscalibrated_dataset())
        payload = cal.to_payload()
        # El payload solo contiene arreglos (sin objetos sklearn) y es JSON-serializable.
        text = json.dumps(payload)
        self.assertIn("curves", payload)
        for klass in ("home", "draw", "away"):
            self.assertIsInstance(payload["curves"][klass]["x"], list)
            self.assertIsInstance(payload["curves"][klass]["y"], list)
        restored = IsotonicCalibration.from_payload(json.loads(text))
        # Predicciones identicas tras round-trip.
        for row in (np.array([0.9, 0.05, 0.05]), np.array([0.2, 0.3, 0.5])):
            np.testing.assert_allclose(
                cal.calibrate_triplet(*row), restored.calibrate_triplet(*row), atol=1e-12
            )

    def test_class_floor_applied(self) -> None:
        triplets, outcomes = _miscalibrated_dataset()
        cal = fit_isotonic(triplets, outcomes, class_floor=0.05)
        h, d, a = cal.calibrate_triplet(0.95, 0.03, 0.02)
        self.assertGreaterEqual(min(h, d, a), 0.05 - 1e-9)


class TestLoadActiveCalibration(unittest.TestCase):
    def test_missing_artifact_returns_neutral(self) -> None:
        cal = load_active_calibration(path="does/not/exist.json")
        self.assertTrue(cal.is_neutral)

    def test_loads_isotonic_artifact(self) -> None:
        cal = fit_isotonic(*_miscalibrated_dataset())
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps(cal.to_payload()), encoding="utf-8")
            loaded = load_active_calibration(path=path)
        self.assertIsInstance(loaded, IsotonicCalibration)
        self.assertFalse(loaded.is_neutral)
        np.testing.assert_allclose(
            loaded.calibrate_triplet(0.9, 0.05, 0.05),
            cal.calibrate_triplet(0.9, 0.05, 0.05),
            atol=1e-12,
        )

    def test_loads_legacy_no_method_artifact_as_temperature(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps({"temperature": 2.0, "class_floor": 0.03}), encoding="utf-8")
            loaded = load_active_calibration(path=path)
        self.assertIsInstance(loaded, ProbabilityCalibration)
        self.assertEqual(loaded.temperature, 2.0)
        self.assertEqual(loaded.class_floor, 0.03)

    def test_loads_explicit_temperature_method(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(
                json.dumps({"method": "temperature", "temperature": 1.5, "class_floor": 0.0}),
                encoding="utf-8",
            )
            loaded = load_active_calibration(path=path)
        self.assertIsInstance(loaded, ProbabilityCalibration)
        self.assertEqual(loaded.temperature, 1.5)


if __name__ == "__main__":
    unittest.main()
