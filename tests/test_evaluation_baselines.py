import unittest

import numpy as np

from alttez.evaluation.baselines import (
    base_rate_probabilities,
    elo_only_probabilities,
    fit_draw_prob_at_parity,
    outcome_frequencies,
    uniform_probabilities,
)
from alttez.evaluation.paired import brier_1x2_per_game
from alttez.features.elo import HOME_ADVANTAGE_ELO


class UniformBaselineTest(unittest.TestCase):
    def test_every_row_is_one_third(self) -> None:
        probs = uniform_probabilities(4)
        self.assertEqual(probs.shape, (4, 3))
        np.testing.assert_allclose(probs, 1.0 / 3.0)

    def test_uniform_brier_is_two_thirds_for_any_outcome(self) -> None:
        brier = brier_1x2_per_game(uniform_probabilities(3), ["H", "D", "A"])
        np.testing.assert_allclose(brier, 2.0 / 3.0)

    def test_rejects_non_positive_size(self) -> None:
        with self.assertRaises(ValueError):
            uniform_probabilities(0)


class BaseRateBaselineTest(unittest.TestCase):
    def test_frequencies_follow_training_window(self) -> None:
        train = ["H", "H", "H", "D", "A", "A", "H", "D"]
        np.testing.assert_allclose(outcome_frequencies(train), [0.5, 0.25, 0.25])

    def test_rows_repeat_frequencies_and_sum_to_one(self) -> None:
        probs = base_rate_probabilities(["H", "D", "A", "H"], 5)
        self.assertEqual(probs.shape, (5, 3))
        np.testing.assert_allclose(probs, [[0.5, 0.25, 0.25]] * 5)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_rejects_empty_or_unknown_outcomes(self) -> None:
        with self.assertRaises(ValueError):
            base_rate_probabilities([], 3)
        with self.assertRaises(ValueError):
            base_rate_probabilities(["H", "X"], 3)


class EloOnlyBaselineTest(unittest.TestCase):
    def test_rows_are_valid_distributions(self) -> None:
        diffs = np.array([-600.0, -150.0, 0.0, 150.0, 600.0])
        probs = elo_only_probabilities(diffs, draw_prob_at_parity=0.3)
        self.assertEqual(probs.shape, (5, 3))
        self.assertTrue((probs >= 0).all())
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_bigger_gap_means_more_home_and_less_draw(self) -> None:
        probs = elo_only_probabilities(np.array([0.0, 200.0, 500.0]), draw_prob_at_parity=0.3)
        self.assertTrue(np.all(np.diff(probs[:, 0]) > 0))
        self.assertTrue(np.all(np.diff(probs[:, 2]) < 0))

    def test_neutral_venue_removes_home_advantage(self) -> None:
        home = elo_only_probabilities(np.array([0.0]), draw_prob_at_parity=0.3, neutral=[False])
        neutral = elo_only_probabilities(np.array([0.0]), draw_prob_at_parity=0.3, neutral=[True])
        self.assertGreater(home[0, 0], home[0, 2])
        self.assertAlmostEqual(neutral[0, 0], neutral[0, 2])
        also_home = elo_only_probabilities(np.array([-HOME_ADVANTAGE_ELO]), draw_prob_at_parity=0.3)
        np.testing.assert_allclose(also_home, neutral)

    def test_expected_score_is_preserved(self) -> None:
        probs = elo_only_probabilities(np.array([120.0]), draw_prob_at_parity=0.3, neutral=[True])
        expected = 1.0 / (1.0 + 10.0 ** (-120.0 / 400.0))
        self.assertAlmostEqual(probs[0, 0] + probs[0, 1] / 2.0, expected)

    def test_rejects_invalid_draw_probability_and_shapes(self) -> None:
        with self.assertRaises(ValueError):
            elo_only_probabilities(np.array([0.0]), draw_prob_at_parity=0.9)
        with self.assertRaises(ValueError):
            elo_only_probabilities(np.array([0.0, 1.0]), draw_prob_at_parity=0.3, neutral=[True])

    def test_fitted_draw_probability_reproduces_training_draw_rate(self) -> None:
        rng = np.random.default_rng(1)
        diffs = rng.normal(0.0, 200.0, size=400)
        outcomes = np.where(rng.random(400) < 0.24, "D", "H")
        neutral = np.ones(400, dtype=bool)
        p0 = fit_draw_prob_at_parity(diffs, outcomes, neutral)
        fitted = elo_only_probabilities(diffs, draw_prob_at_parity=p0, neutral=neutral)
        self.assertAlmostEqual(fitted[:, 1].mean(), (outcomes == "D").mean())

    def test_elo_only_beats_uniform_when_elo_is_informative(self) -> None:
        rng = np.random.default_rng(2)
        diffs = rng.normal(0.0, 250.0, size=2000)
        p_home = 1.0 / (1.0 + 10.0 ** (-diffs / 400.0))
        outcomes = np.where(rng.random(2000) < p_home, "H", "A")
        p0 = fit_draw_prob_at_parity(diffs, outcomes, np.ones(2000, dtype=bool))
        elo = brier_1x2_per_game(elo_only_probabilities(diffs, draw_prob_at_parity=p0, neutral=np.ones(2000, dtype=bool)), outcomes)
        uniform = brier_1x2_per_game(uniform_probabilities(2000), outcomes)
        self.assertLess(elo.mean(), uniform.mean())


if __name__ == "__main__":
    unittest.main()
