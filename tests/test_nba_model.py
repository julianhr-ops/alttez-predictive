import unittest

import numpy as np

from alttez.nba import data
from alttez.nba.model import (
    BlendModel,
    EloParams,
    EloTracker,
    IsotonicCalibrator,
    MarketModel,
    ModelFitError,
    PlattCalibrator,
    fit_margin_slope,
    fit_models,
    predict_games,
    run_elo,
)
from tests.nba_synthetic import SMALL_GRID, SMALL_VANILLA, synthetic_games


class EloMechanicsTests(unittest.TestCase):
    def test_probabilities_strictly_inside_unit_interval(self):
        games = synthetic_games()
        for params in (*SMALL_GRID, *SMALL_VANILLA):
            p = run_elo(games, params).predictions["p_home"].to_numpy()
            self.assertTrue(((p > 0) & (p < 1)).all())

    def test_equal_teams_favour_home_by_the_home_advantage(self):
        tracker = EloTracker(EloParams(home_advantage=100.0))
        self.assertEqual(tracker.diff(1, 2, 0, neutral=False), 100.0)
        self.assertEqual(tracker.diff(1, 2, 0, neutral=True), 0.0)

    def test_back_to_back_penalty_hits_only_the_tired_side(self):
        tracker = EloTracker(EloParams(home_advantage=0.0, b2b_penalty=30.0))
        tracker.update(1, 3, day=10, diff=0.0, margin=5, season_type=2)
        self.assertEqual(tracker.diff(1, 2, 11, neutral=False), -30.0 + tracker.rating(1) - tracker.rating(2))
        self.assertEqual(tracker.diff(1, 2, 12, neutral=False), tracker.rating(1) - tracker.rating(2))

    def test_bigger_margin_moves_ratings_more_only_with_mov(self):
        def shift(margin, use_mov):
            tracker = EloTracker(EloParams(use_mov=use_mov))
            tracker.update(1, 2, 0, 0.0, margin, 2)
            return tracker.rating(1) - 1500.0

        self.assertGreater(shift(25, True), shift(3, True))
        self.assertEqual(shift(25, False), shift(3, False))

    def test_losing_streak_lowers_probability(self):
        tracker = EloTracker(EloParams())
        before = tracker.diff(1, 2, 0, False)
        tracker.update(1, 2, 0, before, -10, 2)
        self.assertLess(tracker.diff(1, 2, 5, False), before)

    def test_season_change_regresses_toward_the_mean(self):
        tracker = EloTracker(EloParams(season_regression=0.5))
        tracker.enter_season(2010)
        tracker.ratings = {1: 1700.0, 2: 1300.0}
        tracker.enter_season(2010)
        self.assertEqual(tracker.ratings[1], 1700.0)
        tracker.enter_season(2011)
        self.assertEqual((tracker.ratings[1], tracker.ratings[2]), (1600.0, 1400.0))

    def test_playoff_multiplier_scales_the_update(self):
        base, playoff = EloTracker(EloParams(playoff_k_multiplier=2.0, use_mov=False)), EloTracker(EloParams(playoff_k_multiplier=2.0, use_mov=False))
        base.update(1, 2, 0, 0.0, 5, 2)
        playoff.update(1, 2, 0, 0.0, 5, 3)
        self.assertAlmostEqual(playoff.rating(1) - 1500.0, 2 * (base.rating(1) - 1500.0))

    def test_unsorted_input_is_rejected(self):
        games = synthetic_games().iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            run_elo(games, EloParams())


class NoLeakageTests(unittest.TestCase):
    def setUp(self):
        self.games = synthetic_games()

    def test_future_results_do_not_change_earlier_predictions(self):
        cutoff = self.games["game_date"].iloc[len(self.games) // 2]
        future = (self.games["game_date"] >= cutoff).to_numpy()
        shuffled = self.games.copy()
        rng = np.random.default_rng(0)
        permutation = rng.permutation(future.sum())
        for column in ("home_score", "away_score"):
            values = shuffled.loc[future, column].to_numpy()
            shuffled.loc[future, column] = values[permutation]
        for params in (SMALL_GRID[1], SMALL_VANILLA[0]):
            original = run_elo(self.games, params).predictions["elo_diff"].to_numpy()
            altered = run_elo(shuffled, params).predictions["elo_diff"].to_numpy()
            on_cutoff_day = (self.games["game_date"] <= cutoff).to_numpy()
            np.testing.assert_array_equal(original[on_cutoff_day], altered[on_cutoff_day])
            self.assertFalse(np.array_equal(original[future], altered[future]))

    def test_a_game_never_sees_its_own_result(self):
        index = 200
        flipped = self.games.copy()
        flipped.loc[index, ["home_score", "away_score"]] = [
            self.games.loc[index, "away_score"], self.games.loc[index, "home_score"],
        ]
        original = run_elo(self.games, SMALL_GRID[0]).predictions["elo_diff"].to_numpy()
        altered = run_elo(flipped, SMALL_GRID[0]).predictions["elo_diff"].to_numpy()
        np.testing.assert_array_equal(original[: index + 1], altered[: index + 1])

    def test_fitted_models_ignore_rows_after_the_history(self):
        history = self.games[self.games["season"] < 2016].reset_index(drop=True)
        later = self.games[self.games["season"] >= 2016].copy()
        later[["home_score", "away_score"]] = later[["away_score", "home_score"]].to_numpy()
        first = fit_models(history, elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)
        second = fit_models(history, elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)
        self.assertEqual(first.train_seasons, tuple(range(2010, 2016)))
        self.assertEqual(first.elo_params, second.elo_params)
        self.assertEqual(first.platt, second.platt)
        self.assertEqual(first.blend, second.blend)

    def test_predict_games_rejects_targets_that_overlap_history(self):
        models = fit_models(self.games, elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)
        with self.assertRaises(ValueError):
            predict_games(models, self.games, self.games.iloc[:5])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.p = rng.uniform(0.05, 0.95, 2000)
        self.y = (rng.uniform(size=2000) < 0.5 * self.p + 0.25).astype(float)

    def test_platt_is_monotone_and_shrinks_overconfidence(self):
        platt = PlattCalibrator.fit(self.p, self.y)
        grid = np.linspace(0.01, 0.99, 99)
        out = platt(grid)
        self.assertTrue((np.diff(out) > 0).all())
        self.assertLess(platt.slope, 1.0)
        self.assertTrue(((out > 0) & (out < 1)).all())

    def test_isotonic_is_monotone_and_inside_unit_interval(self):
        iso = IsotonicCalibrator.fit(self.p, self.y)
        out = iso(np.linspace(0.0, 1.0, 201))
        self.assertTrue((np.diff(out) >= 0).all())
        self.assertTrue(((out > 0) & (out < 1)).all())

    def test_platt_refuses_an_inverted_relationship(self):
        with self.assertRaises(ModelFitError):
            PlattCalibrator.fit(self.p, 1.0 - self.y)

    def test_too_little_data_is_an_error_not_a_fallback(self):
        with self.assertRaises(ModelFitError):
            PlattCalibrator.fit(self.p[:10], self.y[:10])


class MarketAndBlendTests(unittest.TestCase):
    def test_zero_line_is_a_coin_flip_and_the_curve_is_symmetric(self):
        market = MarketModel(sd=12.0)
        np.testing.assert_allclose(market.p_home(np.array([0.0, 6.0, -6.0])), [0.5, 1 - market.p_home(-6.0)[()], market.p_home(-6.0)], atol=1e-9)

    def test_fit_recovers_the_residual_sd(self):
        rng = np.random.default_rng(5)
        line = rng.normal(0, 6, 20000)
        margin = line + rng.normal(0, 12.5, 20000)
        self.assertAlmostEqual(MarketModel.fit(line, margin).sd, 12.5, delta=0.4)

    def test_blend_follows_the_informative_input(self):
        rng = np.random.default_rng(6)
        market_p = rng.uniform(0.1, 0.9, 4000)
        elo_p = rng.uniform(0.1, 0.9, 4000)
        y = (rng.uniform(size=4000) < market_p).astype(float)
        blend = BlendModel.fit(elo_p, market_p, y)
        self.assertGreater(blend.market_coef, 0.8)
        self.assertLess(abs(blend.elo_coef), 0.25)
        self.assertTrue(((blend.p_home(elo_p, market_p) > 0) & (blend.p_home(elo_p, market_p) < 1)).all())

    def test_margin_slope_through_origin(self):
        diff = np.array([-100.0, 0.0, 50.0, 100.0])
        self.assertAlmostEqual(fit_margin_slope(diff, diff * 0.04), 0.04)


class PredictGamesTests(unittest.TestCase):
    def test_all_probabilities_are_valid_and_blend_only_where_a_line_exists(self):
        games = synthetic_games()
        history = games[games["season"] < 2017].reset_index(drop=True)
        target = games[games["season"] == 2017].reset_index(drop=True)
        target.loc[:9, "home_margin_line"] = np.nan
        models = fit_models(history, elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)
        out = predict_games(models, history, target)
        self.assertEqual(len(out), len(target))
        for column in ("p_const", "p_vanilla", "p_elo", "p_elo_platt", "p_elo_isotonic"):
            self.assertTrue(((out[column] > 0) & (out[column] < 1)).all(), column)
        self.assertTrue(out.loc[:9, ["p_market", "p_blend"]].isna().all().all())
        lined = out.loc[10:]
        self.assertTrue(((lined["p_blend"] > 0) & (lined["p_blend"] < 1)).all())
        self.assertTrue(((lined["p_market"] > 0) & (lined["p_market"] < 1)).all())

    def test_history_without_lines_has_no_market_or_blend(self):
        games = synthetic_games(lined_from=3000)
        history = games[games["season"] < 2017].reset_index(drop=True)
        models = fit_models(history, elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)
        self.assertIsNone(models.market)
        self.assertIsNone(models.blend)


class DataCompatibilityTests(unittest.TestCase):
    def test_synthetic_frame_matches_the_real_game_columns(self):
        self.assertTrue(set(data.GAME_COLUMNS) <= set(synthetic_games().columns))


if __name__ == "__main__":
    unittest.main()
