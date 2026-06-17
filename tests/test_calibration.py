from __future__ import annotations

import unittest

import numpy as np

from alttez.inference.calibration import (
    NEUTRAL_CALIBRATION,
    ProbabilityCalibration,
    apply_class_floor,
    expected_calibration_error,
    fit_temperature,
    load_active_calibration,
    log_loss,
    temperature_scale,
)


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


class TestLoadActiveCalibration(unittest.TestCase):
    def test_missing_artifact_returns_neutral(self) -> None:
        cal = load_active_calibration(path="does/not/exist.json")
        self.assertTrue(cal.is_neutral)


if __name__ == "__main__":
    unittest.main()
