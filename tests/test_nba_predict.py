import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from alttez.nba import data, predict
from alttez.nba.model import fit_models, predict_games
from alttez.nba.store import ServedModel
from tests.nba_synthetic import SMALL_GRID, SMALL_VANILLA, synthetic_games


def scheduled(game_id, home, away, start, *, line=None, state="pre", season=2017, season_type=2, provider="DraftKings"):
    return data.NbaScheduledGame(
        game_id=game_id, start_time=start, season=season, season_type=season_type, home_id=home, away_id=away,
        home_abbreviation=f"T{home}", away_abbreviation=f"T{away}", neutral=False, state=state, completed=state == "post", home_score=None,
        away_score=None, home_margin_line=line, total_line=None if line is None else 215.5,
        odds_provider=None if line is None else provider,
    )


class PredictionTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        games = synthetic_games()
        cls.games = games
        cls.inputs = predict.NbaInputs(
            games=games,
            complete_seasons=tuple(sorted(games["season"].unique())),
            scheduled_dates=(),
        )
        cls.models = fit_models(games, elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)
        last = games["game_date"].max()
        cls.today = (last + pd.Timedelta(days=1)).date()
        cls.tip = datetime.combine(cls.today, datetime.min.time(), tzinfo=UTC) + timedelta(hours=20)
        cls.now = cls.tip - timedelta(hours=10)


class FreshnessGateTests(unittest.TestCase):
    TODAY = date(2026, 1, 20)

    def test_stale_while_the_schedule_lists_games_since_the_newest_completed(self):
        newest = date(2026, 1, 10)
        scheduled = [date(2026, 1, 11), date(2026, 1, 15), date(2026, 3, 1)]
        with self.assertRaises(predict.StaleDataError) as caught:
            predict.check_freshness(newest, scheduled, self.TODAY)
        self.assertEqual(caught.exception.code, "stale_data")
        self.assertIn("stale_data", str(caught.exception))

    def test_exactly_three_days_old_is_still_fresh(self):
        predict.check_freshness(date(2026, 1, 17), [date(2026, 1, 18)], self.TODAY)

    def test_four_days_old_with_games_in_between_is_stale(self):
        with self.assertRaises(predict.StaleDataError):
            predict.check_freshness(date(2026, 1, 16), [date(2026, 1, 18)], self.TODAY)

    def test_offseason_and_all_star_break_do_not_trip_the_gate(self):
        predict.check_freshness(date(2025, 6, 20), [date(2025, 10, 22), date(2026, 4, 12)], date(2025, 8, 15))
        predict.check_freshness(date(2026, 1, 10), [date(2026, 1, 20), date(2026, 1, 21)], self.TODAY)

    def test_schedule_dates_cover_pending_games_of_nba_franchises_only(self):
        raw = pd.DataFrame(
            {
                "home_id": [1, 2, 31, 3],
                "away_id": [2, 3, 1, 4],
                "season_type": [2, 3, 2, 1],
                "date": ["2026-01-11T01:00Z", "2026-01-12T20:00Z", "2026-01-13T20:00Z", "2026-01-14T20:00Z"],
            }
        )
        self.assertEqual(predict.schedule_dates(raw), (date(2026, 1, 10), date(2026, 1, 12)))


class StaleRefusalTests(PredictionTestBase):
    def test_predict_upcoming_refuses_before_fitting_or_fetching(self):
        stale_today = self.today + timedelta(days=10)
        inputs = replace(self.inputs, scheduled_dates=(self.today + timedelta(days=2),))

        def fetch(_day):
            raise AssertionError("scoreboard must not be fetched on stale data")

        with self.assertRaises(predict.StaleDataError):
            predict.predict_upcoming(inputs, today=stale_today, now=self.now, fetch=fetch)


class ServedModelTests(unittest.TestCase):
    def test_blend_serves_when_it_exists(self):
        self.assertEqual(predict.select_served(0.60, 0.70), (ServedModel.BLEND, 0.70))

    def test_calibrated_elo_serves_without_a_blend(self):
        self.assertEqual(predict.select_served(0.60, float("nan")), (ServedModel.ELO_PLATT, 0.60))


class BuildPredictionsTests(PredictionTestBase):
    def build(self, upcoming, models=None):
        return predict.build_predictions(models or self.models, self.inputs, upcoming, now=self.now)

    def test_line_present_serves_the_blend_and_records_the_market(self):
        run = self.build([scheduled(1, 1, 2, self.tip, line=4.5)])
        row = run.predictions[0].row
        self.assertIs(row.served_model, ServedModel.BLEND)
        expected = self.models.blend.p_home(np.array([self._raw_elo(1, 2)]), np.array([row.p_home_market]))[0]
        self.assertEqual(row.p_home_served, round(float(expected), 6))
        self.assertAlmostEqual(row.p_home_market, float(self.models.market.p_home(4.5)), places=6)
        self.assertEqual((row.market_spread, row.market_total, row.market_source), (4.5, 215.5, "espn:DraftKings"))
        self.assertEqual(row.model_version, predict.MODEL_VERSION)
        self.assertEqual(row.tipoff_utc, self.tip)
        self.assertEqual(row.game_date, self.today)
        self.assertEqual(row.created_at, self.now)

    def test_no_line_serves_calibrated_elo_and_stores_no_market(self):
        run = self.build([scheduled(1, 1, 2, self.tip)])
        row = run.predictions[0].row
        self.assertIs(row.served_model, ServedModel.ELO_PLATT)
        self.assertEqual(row.p_home_served, row.p_home_elo)
        self.assertEqual(row.p_home_elo, round(float(self.models.platt(self._raw_elo(1, 2))), 6))
        self.assertIsNone(row.p_home_market)
        self.assertEqual((row.market_spread, row.market_total, row.market_source), (None, None, None))

    def test_without_a_fitted_blend_a_line_still_serves_elo_platt(self):
        models = replace(self.models, blend=None)
        row = self.build([scheduled(1, 1, 2, self.tip, line=4.5)], models).predictions[0].row
        self.assertIs(row.served_model, ServedModel.ELO_PLATT)
        self.assertIsNotNone(row.p_home_market)

    def test_teams_without_ratings_and_odd_season_types_are_skipped_and_reported(self):
        run = self.build(
            [
                scheduled(1, 1, 99, self.tip),
                scheduled(2, 1, 2, self.tip, season_type=1),
                scheduled(3, 3, 4, self.tip),
            ]
        )
        self.assertEqual([p.row.game_id for p in run.predictions], [3])
        reasons = {s.game_id: s.reason for s in run.skipped}
        self.assertIn("no rating for team id(s) [99]", reasons[1])
        self.assertIn("season type 1", reasons[2])

    def test_started_games_are_not_predicted(self):
        run = self.build(
            [scheduled(1, 1, 2, self.tip, state="in"), scheduled(2, 3, 4, self.now - timedelta(minutes=5))]
        )
        self.assertEqual(run.predictions, ())
        self.assertEqual([s.reason for s in run.skipped], ["already started", "already started"])

    def test_back_to_back_across_upcoming_dates_is_penalised(self):
        models = replace(self.models, elo_params=replace(self.models.elo_params, b2b_penalty=60.0))
        tomorrow = self.tip + timedelta(days=1)
        both_days = self.build([scheduled(1, 1, 2, self.tip), scheduled(2, 3, 1, tomorrow)], models)
        only_tomorrow = self.build([scheduled(2, 3, 1, tomorrow)], models)
        with_b2b = {p.row.game_id: p.row for p in both_days.predictions}[2]
        without = only_tomorrow.predictions[0].row
        self.assertGreater(with_b2b.p_home_elo, without.p_home_elo)

    def test_each_date_uses_only_earlier_results(self):
        tomorrow = self.tip + timedelta(days=1)
        together = {p.row.game_id: p.row for p in self.build([scheduled(1, 1, 2, self.tip), scheduled(2, 3, 4, tomorrow)]).predictions}
        alone = self.build([scheduled(2, 3, 4, tomorrow)]).predictions[0].row
        self.assertEqual(together[2].p_home_elo, alone.p_home_elo)

    def test_predict_day_matches_the_validated_model_when_nothing_earlier_is_upcoming(self):
        target = predict._target_frame([scheduled(1, 1, 2, self.tip, line=-3.0), scheduled(2, 3, 4, self.tip)])
        prior = self.games[self.games["game_date"] < target["game_date"].iloc[0]].reset_index(drop=True)
        mine = predict.predict_day(self.models, prior, target, {})
        reference = predict_games(self.models, prior, target)
        for mine_col, ref_col in (
            ("p_elo_platt", "p_elo_platt"), ("elo_margin", "elo_margin"), ("p_market", "p_market"),
            ("p_blend", "p_blend"), ("market_margin", "market_margin"),
        ):
            np.testing.assert_allclose(mine[mine_col], reference[ref_col], equal_nan=True, err_msg=mine_col)

    def test_results_are_sorted_by_tipoff_and_carry_abbreviations(self):
        later = scheduled(1, 1, 2, self.tip + timedelta(hours=2))
        earlier = scheduled(2, 3, 4, self.tip)
        run = self.build([later, earlier])
        self.assertEqual([p.row.game_id for p in run.predictions], [2, 1])
        self.assertEqual(run.predictions[0].matchup, "T4 @ T3")
        self.assertEqual(run.newest_completed, self.inputs.newest_completed)

    def _raw_elo(self, home, away):
        target = predict._target_frame([scheduled(1, home, away, self.tip)])
        prior = self.games[self.games["game_date"] < target["game_date"].iloc[0]].reset_index(drop=True)
        return float(predict_games(self.models, prior, target)["p_elo"].iloc[0])


class FetchTests(unittest.TestCase):
    def test_today_and_tomorrow_are_fetched_and_deduplicated(self):
        seen = []
        tip = datetime(2026, 11, 18, 0, 30, tzinfo=UTC)

        def fetch(day):
            seen.append(day)
            return [scheduled(1, 1, 2, tip), scheduled(10 + len(seen), 3, 4, tip)]

        games = predict.fetch_upcoming(date(2026, 11, 17), fetch=fetch)
        self.assertEqual(seen, [date(2026, 11, 17), date(2026, 11, 18)])
        self.assertEqual([g.game_id for g in games], [1, 11, 12])


if __name__ == "__main__":
    unittest.main()
