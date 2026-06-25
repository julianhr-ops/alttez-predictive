from __future__ import annotations

import unittest

import numpy as np

from alttez.dashboard.model_monitoring import bootstrap_ci, murphy_decomposition


class MurphyDecompositionTests(unittest.TestCase):
    def test_identity_holds(self) -> None:
        rng = np.random.default_rng(7)
        n = 2000
        probs = rng.dirichlet([2.0, 1.5, 2.0], size=n)
        # Sample outcomes from the predicted distribution -> well-calibrated.
        outcomes = []
        labels = ["H", "D", "A"]
        for row in probs:
            outcomes.append(labels[int(rng.choice(3, p=row))])

        decomposition = murphy_decomposition(probs, outcomes, bins=10)
        identity = (
            decomposition["reliability"]
            - decomposition["resolution"]
            + decomposition["uncertainty"]
        )
        # The calibration-refinement identity holds up to a small within-bin
        # discretization residual (confidences vary inside a bin). Tolerance
        # reflects that grouping error, which shrinks as bin count grows.
        self.assertAlmostEqual(decomposition["brier"], identity, places=3)

    def test_identity_exact_for_single_confidence_per_bin(self) -> None:
        # When every confidence lands at a single point per bin there is no
        # within-bin variance, so the identity holds to machine precision.
        probs = [[0.6, 0.2, 0.2]] * 7 + [[0.6, 0.2, 0.2]] * 3
        outcomes = ["H"] * 7 + ["A"] * 3
        decomposition = murphy_decomposition(probs, outcomes, bins=10)
        identity = (
            decomposition["reliability"]
            - decomposition["resolution"]
            + decomposition["uncertainty"]
        )
        self.assertAlmostEqual(decomposition["brier"], identity, places=9)

    def test_reliability_small_for_calibrated(self) -> None:
        rng = np.random.default_rng(11)
        n = 5000
        probs = rng.dirichlet([3.0, 2.0, 3.0], size=n)
        labels = ["H", "D", "A"]
        outcomes = [labels[int(rng.choice(3, p=row))] for row in probs]

        decomposition = murphy_decomposition(probs, outcomes, bins=10)
        # Well-calibrated synthetic data: reliability close to zero.
        self.assertLess(decomposition["reliability"], 0.01)

    def test_uncertainty_equals_base_rate_brier(self) -> None:
        # Build a deterministic set where we control the hit rate (base rate).
        # confidence column = max prob; predicted class = argmax.
        # 7 hits out of 10 -> base rate 0.7 -> uncertainty 0.7*0.3 = 0.21
        probs = []
        outcomes = []
        # 7 correct: predict H (argmax) and outcome H
        for _ in range(7):
            probs.append([0.6, 0.2, 0.2])
            outcomes.append("H")
        # 3 wrong: predict H but outcome A
        for _ in range(3):
            probs.append([0.6, 0.2, 0.2])
            outcomes.append("A")

        decomposition = murphy_decomposition(probs, outcomes, bins=10)
        base_rate = 0.7
        self.assertAlmostEqual(decomposition["uncertainty"], base_rate * (1 - base_rate), places=9)

        # Cross-check: uncertainty is the Brier of always predicting the base rate.
        correct = np.array([1.0] * 7 + [0.0] * 3)
        base_rate_brier = float(np.mean((correct.mean() - correct) ** 2))
        self.assertAlmostEqual(decomposition["uncertainty"], base_rate_brier, places=9)

    def test_empty_inputs_neutral(self) -> None:
        decomposition = murphy_decomposition([], [])
        self.assertEqual(decomposition["reliability"], 0.0)
        self.assertEqual(decomposition["resolution"], 0.0)
        self.assertEqual(decomposition["uncertainty"], 0.0)
        self.assertEqual(decomposition["brier"], 0.0)


class BootstrapCITests(unittest.TestCase):
    def test_deterministic_with_seed(self) -> None:
        values = [0.2, 0.3, 0.15, 0.4, 0.25, 0.1, 0.35]
        first = bootstrap_ci(values, n=500, seed=42)
        second = bootstrap_ci(values, n=500, seed=42)
        self.assertEqual(first, second)

    def test_ci_brackets_point(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(0.25, 0.05, size=400)
        point, low, high = bootstrap_ci(values, n=1000, seed=99)
        self.assertLessEqual(low, point)
        self.assertLessEqual(point, high)
        self.assertLess(low, high)

    def test_degenerate_zero_width(self) -> None:
        values = [0.21] * 50
        point, low, high = bootstrap_ci(values, n=300, seed=1)
        self.assertAlmostEqual(point, 0.21, places=9)
        self.assertAlmostEqual(low, 0.21, places=9)
        self.assertAlmostEqual(high, 0.21, places=9)
        self.assertAlmostEqual(high - low, 0.0, places=12)

    def test_empty_returns_zeros(self) -> None:
        self.assertEqual(bootstrap_ci([], n=10), (0.0, 0.0, 0.0))

    def test_different_seeds_differ(self) -> None:
        values = [0.2, 0.3, 0.15, 0.4, 0.25, 0.1, 0.35]
        a = bootstrap_ci(values, n=500, seed=1)
        b = bootstrap_ci(values, n=500, seed=2)
        # Point estimate identical (same data), CI bounds should differ.
        self.assertAlmostEqual(a[0], b[0], places=12)
        self.assertNotEqual((a[1], a[2]), (b[1], b[2]))


if __name__ == "__main__":
    unittest.main()
