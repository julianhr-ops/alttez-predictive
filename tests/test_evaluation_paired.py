from __future__ import annotations

import unittest

import numpy as np

from alttez.evaluation.paired import (
    brier_1x2_per_game,
    brier_binary_per_game,
    expected_calibration_error,
    log_loss_binary_per_game,
    paired_bootstrap_diff,
    reliability_table,
)


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_losses_and_clear_winner(self):
        losses = np.linspace(0, 1, 100)
        same = paired_bootstrap_diff(losses, losses)
        self.assertEqual(same.mean_diff, 0)
        self.assertLessEqual(same.ci_low, 0)
        self.assertGreaterEqual(same.ci_high, 0)
        self.assertFalse(same.a_better)
        better = paired_bootstrap_diff(losses, losses + 0.2)
        self.assertTrue(better.a_better)
        self.assertFalse(better.b_better)
        self.assertTrue(paired_bootstrap_diff(losses + 0.2, losses).b_better)

    def test_seed_reproducibility(self):
        a = np.arange(40) / 40
        one = paired_bootstrap_diff(a, np.zeros(40), seed=7)
        self.assertEqual(one, paired_bootstrap_diff(a, np.zeros(40), seed=7))
        self.assertNotEqual(one, paired_bootstrap_diff(a, np.zeros(40), seed=8))

    def test_cluster_resampling_preserves_whole_blocks(self):
        values = np.repeat(np.r_[-np.ones(20), np.ones(20)], 20)
        ids = np.repeat(np.arange(40), 20)
        row = paired_bootstrap_diff(values, np.zeros_like(values), seed=11)
        block = paired_bootstrap_diff(values, np.zeros_like(values), seed=11, block_ids=ids)
        self.assertGreater(block.ci_high - block.ci_low, 3 * (row.ci_high - row.ci_low))

    def test_unequal_blocks_weight_games_not_blocks(self):
        result = paired_bootstrap_diff([1, 1, -1], [0, 0, 0], block_ids=["a", "a", "b"])
        self.assertAlmostEqual(result.mean_diff, 1 / 3)
        self.assertEqual(result.n, 3)
        self.assertGreaterEqual(result.ci_low, -1)
        self.assertLessEqual(result.ci_high, 1)


class ProbabilityMetricTests(unittest.TestCase):
    def test_binary_scores(self):
        np.testing.assert_array_equal(brier_binary_per_game([1, 0], [1, 0]), [0, 0])
        np.testing.assert_array_equal(brier_binary_per_game([0.5, 0.5], [1, 0]), [0.25, 0.25])
        np.testing.assert_allclose(log_loss_binary_per_game([0.5, 0.5], [1, 0]), np.log(2))
        self.assertTrue(np.isfinite(log_loss_binary_per_game([0, 1], [1, 0])).all())

    def test_multiclass_brier(self):
        result = brier_1x2_per_game([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], ["H", "D"])
        np.testing.assert_allclose(result, [0.14, 0.86])

    def test_calibration_and_bin_edges(self):
        p = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 1000)
        y = np.concatenate([np.r_[np.ones(k), np.zeros(1000 - k)] for k in [100, 300, 500, 700, 900]])
        self.assertAlmostEqual(expected_calibration_error(p, y), 0, places=12)
        self.assertAlmostEqual(expected_calibration_error(np.full(1000, 0.9), np.tile([0, 1], 500)), 0.4)
        table = reliability_table([0, 0.5, 1], [0, 1, 1], n_bins=2)
        self.assertEqual(list(table.columns), ["bin", "n", "mean_p", "freq"])
        self.assertEqual(table["n"].tolist(), [1, 2])
        np.testing.assert_allclose(table["mean_p"], [0, 0.75])

    def test_invalid_inputs_raise_value_error(self):
        cases = [
            lambda: paired_bootstrap_diff([], []),
            lambda: paired_bootstrap_diff([1], [1, 2]),
            lambda: paired_bootstrap_diff([np.nan], [1]),
            lambda: paired_bootstrap_diff([1], [1], n_boot=0),
            lambda: paired_bootstrap_diff([1], [1], alpha=1),
            lambda: paired_bootstrap_diff([1], [1], block_ids=[None]),
            lambda: paired_bootstrap_diff([1], [1], block_ids=[1, 2]),
            lambda: brier_binary_per_game([], []),
            lambda: brier_binary_per_game([0.5], [0, 1]),
            lambda: brier_binary_per_game([1.1], [1]),
            lambda: brier_binary_per_game([np.nan], [1]),
            lambda: brier_binary_per_game([0.5], [2]),
            lambda: log_loss_binary_per_game([0.5], [1], eps=0),
            lambda: brier_1x2_per_game([[0.2, 0.3, 0.5]], ["X"]),
            lambda: brier_1x2_per_game([[0.2, -0.1, 0.9]], ["H"]),
            lambda: brier_1x2_per_game([[0.2, 0.8]], ["H"]),
            lambda: brier_1x2_per_game([], []),
            lambda: expected_calibration_error([0.5], [1], n_bins=0),
            lambda: reliability_table([0.5], [1], n_bins=1.5),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                case()


if __name__ == "__main__":
    unittest.main()
