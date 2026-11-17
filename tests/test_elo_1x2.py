from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alttez.features.elo import EloEngine, rate_matches
from alttez.features.elo_1x2 import KINDS, DrawModel, fit_blend_weight, fit_draw_model, log_linear_blend


def simulate(model: DrawModel, n: int = 6000, seed: int = 3) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    diff = rng.normal(0.0, 200.0, n)
    neutral = rng.random(n) < 0.3
    probs = model.probabilities(diff, neutral)
    cumulative = probs.cumsum(axis=1)
    draws = rng.random(n)[:, None]
    outcomes = np.array(["H", "D", "A"])[(draws > cumulative).sum(axis=1)]
    return diff, neutral, outcomes.tolist()


class DrawModelTests(unittest.TestCase):
    def test_probabilities_sum_to_one_and_draw_falls_with_the_gap(self) -> None:
        gaps = np.array([0.0, 100.0, 300.0, 600.0])
        for kind in KINDS:
            probs = DrawModel(kind, 80.0, 0.5, 1.0).probabilities(gaps, np.ones(4, dtype=bool))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
            self.assertTrue((probs >= 0).all(), kind)
            self.assertTrue((np.diff(probs[:, 1]) < 0).all(), kind)
            self.assertTrue((np.diff(probs[:, 0]) > 0).all(), kind)

    def test_the_home_advantage_applies_only_off_neutral_ground(self) -> None:
        model = DrawModel("ordinal", 100.0, 0.5, 1.0)
        home, neutral = model.probabilities([0.0, 0.0], [False, True])
        self.assertGreater(home[0], home[2])
        self.assertAlmostEqual(neutral[0], neutral[2])

    def test_a_mirrored_neutral_match_swaps_home_and_away(self) -> None:
        for kind in KINDS:
            model = DrawModel(kind, 100.0, 0.6, 1.0)
            forward, backward = model.probabilities([150.0, -150.0], [True, True])
            np.testing.assert_allclose(forward, backward[::-1], atol=1e-12)

    def test_fit_recovers_the_generating_parameters(self) -> None:
        truth = {"ordinal": DrawModel("ordinal", 80.0, 0.55, 1.2), "davidson": DrawModel("davidson", 80.0, 0.7, 1.2)}
        for kind, model in truth.items():
            diff, neutral, outcomes = simulate(model)
            fitted = fit_draw_model(kind, diff, neutral, outcomes, fit_scale=True)
            self.assertAlmostEqual(fitted.home_advantage, model.home_advantage, delta=25.0, msg=kind)
            self.assertAlmostEqual(fitted.draw_param, model.draw_param, delta=0.08, msg=kind)
            self.assertAlmostEqual(fitted.scale, model.scale, delta=0.25, msg=kind)

    def test_without_fit_scale_the_scale_stays_at_one(self) -> None:
        diff, neutral, outcomes = simulate(DrawModel("ordinal", 80.0, 0.55, 1.0), n=800)
        self.assertEqual(fit_draw_model("ordinal", diff, neutral, outcomes).scale, 1.0)

    def test_fit_beats_a_misplaced_model_on_its_own_training_rows(self) -> None:
        diff, neutral, outcomes = simulate(DrawModel("ordinal", 60.0, 0.9, 1.0), n=2000)
        fitted = fit_draw_model("ordinal", diff, neutral, outcomes)
        index = np.array(["H", "D", "A"]).tolist()
        rows = np.arange(len(outcomes))
        actual = np.array([index.index(o) for o in outcomes])

        def log_loss(model: DrawModel) -> float:
            return float(-np.log(model.probabilities(diff, neutral)[rows, actual]).mean())

        self.assertLess(log_loss(fitted), log_loss(DrawModel("ordinal", 100.0, 0.3, 1.0)))

    def test_bad_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_draw_model("probit", [0.0], [False], ["H"])
        with self.assertRaises(ValueError):
            fit_draw_model("ordinal", [0.0, 1.0], [False, False], ["H"])
        with self.assertRaises(ValueError):
            fit_draw_model("ordinal", [0.0], [False], ["X"])
        with self.assertRaises(ValueError):
            fit_draw_model("ordinal", [float("nan")], [False], ["H"])
        with self.assertRaises(ValueError):
            DrawModel("ordinal", 100.0, 0.5).probabilities([0.0, 1.0], [False])


class BlendTests(unittest.TestCase):
    FIRST = np.array([[0.6, 0.2, 0.2], [0.2, 0.3, 0.5]])
    SECOND = np.array([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]])

    def test_weights_one_and_zero_return_each_input(self) -> None:
        np.testing.assert_allclose(log_linear_blend(self.FIRST, self.SECOND, 1.0), self.FIRST, atol=1e-12)
        np.testing.assert_allclose(log_linear_blend(self.FIRST, self.SECOND, 0.0), self.SECOND, atol=1e-12)
        mixed = log_linear_blend(self.FIRST, self.SECOND, 0.4)
        np.testing.assert_allclose(mixed.sum(axis=1), 1.0, atol=1e-12)

    def test_blend_is_proportional_to_the_weighted_geometric_mean(self) -> None:
        mixed = log_linear_blend(self.FIRST, self.SECOND, 0.25)
        expected = self.FIRST**0.25 * self.SECOND**0.75
        np.testing.assert_allclose(mixed, expected / expected.sum(axis=1, keepdims=True), atol=1e-12)

    def test_fit_weight_goes_to_the_informative_side(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.dirichlet([2.0, 1.5, 2.0], size=3000)
        outcomes = np.array(["H", "D", "A"])[(rng.random(3000)[:, None] > truth.cumsum(axis=1)).sum(axis=1)].tolist()
        noise = rng.dirichlet([2.0, 2.0, 2.0], size=3000)
        self.assertGreater(fit_blend_weight(truth, noise, outcomes), 0.9)
        self.assertLess(fit_blend_weight(noise, truth, outcomes), 0.1)

    def test_bad_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            log_linear_blend(self.FIRST, self.SECOND, 1.5)
        with self.assertRaises(ValueError):
            log_linear_blend(self.FIRST, self.SECOND[:1], 0.5)
        with self.assertRaises(ValueError):
            fit_blend_weight(self.FIRST, self.SECOND, ["H"])


def matches(rows: list[tuple[str, str, str, int, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([r[0] for r in rows]),
            "home_team": [r[1] for r in rows],
            "away_team": [r[2] for r in rows],
            "home_score": [r[3] for r in rows],
            "away_score": [r[4] for r in rows],
            "neutral": [r[5] for r in rows],
            "tournament": "Friendly",
        }
    )


class RateMatchesTests(unittest.TestCase):
    def test_final_ratings_match_the_engine_when_dates_are_unique(self) -> None:
        frame = matches(
            [
                ("2020-01-01", "A", "B", 2, 0, False),
                ("2020-01-05", "B", "C", 1, 1, True),
                ("2020-02-01", "C", "A", 0, 3, False),
            ]
        )
        run = rate_matches(frame)
        engine_frame = frame.assign(home_result=[1.0, 0.5, 0.0])
        engine = EloEngine(mov_enabled=False)
        engine.compute_elo_features(engine_frame)
        self.assertEqual(set(run.ratings), {"A", "B", "C"})
        for team, rating in run.ratings.items():
            self.assertAlmostEqual(rating, engine.ratings[team], places=9)

    def test_only_teams_that_played_have_a_rating(self) -> None:
        run = rate_matches(matches([("2020-01-01", "A", "B", 1, 0, False)]))
        self.assertNotIn("Z", run.ratings)

    def test_matches_of_one_date_do_not_see_each_other(self) -> None:
        run = rate_matches(matches([("2020-05-01", "A", "B", 2, 0, False), ("2020-05-01", "A", "C", 0, 1, False)]))
        self.assertEqual(run.home_before.tolist(), [1500.0, 1500.0])

    def test_unsorted_dates_are_rejected(self) -> None:
        frame = matches([("2020-05-02", "A", "B", 1, 0, False), ("2020-05-01", "A", "C", 0, 1, False)])
        with self.assertRaises(ValueError):
            rate_matches(frame)


if __name__ == "__main__":
    unittest.main()
