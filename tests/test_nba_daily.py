import argparse
import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd
from sqlalchemy import text

from alttez.nba import data, predict, store
from alttez.nba.store import DateWindow, InsertOutcome
from scripts import nba_daily
from tests.nba_sqlite import make_engine
from tests.nba_synthetic import synthetic_games
from tests.test_nba_predict import scheduled
from tests.test_nba_store import NOW, TIPOFF, make_row

TODAY = date(2026, 11, 18)


def finished(game_id, home_score, away_score, *, home=28, away=18):
    return replace(
        scheduled(game_id, home, away, TIPOFF, state="post"), home_score=home_score, away_score=away_score
    )


class GradeFinishedTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine(self)
        store.insert_prediction_if_absent(self.engine, make_row(1, p_served=0.8), now=NOW)
        store.insert_prediction_if_absent(self.engine, make_row(2, p_served=0.6), now=NOW)
        self.calls = []

    def fetch(self, day):
        self.calls.append(day)
        return [finished(1, 112, 101), replace(scheduled(2, 28, 18, TIPOFF), state="in")]

    def grade(self, **kwargs):
        return nba_daily.grade_finished(
            self.engine, TODAY, now=TIPOFF + timedelta(hours=5), fetch=self.fetch, write=kwargs.get("write", True)
        )

    def test_finished_games_are_graded_and_the_rest_stay_scheduled(self):
        report = self.grade()
        self.assertEqual(report.graded, 1)
        window = DateWindow(date(2026, 11, 1), date(2026, 11, 30))
        self.assertEqual([r.game_id for r in store.load_ungraded(self.engine, window)], [2])
        (graded,) = store.load_graded_summary(self.engine, window).rows
        self.assertEqual((graded.game_id, graded.home_win, graded.home_score), (1, True, 112))
        self.assertAlmostEqual(graded.brier, 0.04)

    def test_scoreboard_covers_today_and_the_last_three_days(self):
        self.grade()
        self.assertEqual(sorted(self.calls), [TODAY - timedelta(days=n) for n in (3, 2, 1, 0)])

    def test_grading_twice_is_idempotent(self):
        self.grade()
        before = store.load_graded_summary(self.engine, DateWindow(date(2026, 1, 1), date(2026, 12, 31))).rows
        self.assertEqual(self.grade().graded, 0)
        after = store.load_graded_summary(self.engine, DateWindow(date(2026, 1, 1), date(2026, 12, 31))).rows
        self.assertEqual(before, after)

    def test_dry_run_counts_but_does_not_write(self):
        self.assertEqual(self.grade(write=False).graded, 1)
        self.assertEqual(len(store.load_ungraded(self.engine, DateWindow(date(2026, 11, 1), date(2026, 11, 30)))), 2)

    def test_predictions_that_never_got_a_result_are_counted_not_dropped(self):
        self.assertEqual(self.grade().unresolved, 1)
        old = make_row(3, game_date=date(2026, 11, 1), tipoff_utc=TIPOFF - timedelta(days=17))
        store.insert_prediction_if_absent(self.engine, old, now=NOW - timedelta(days=20))
        self.assertEqual(self.grade().unresolved, 2)

    def test_nothing_is_fetched_when_nothing_is_pending(self):
        engine = make_engine(self)
        report = nba_daily.grade_finished(engine, TODAY, now=NOW, fetch=self.fetch, write=True)
        self.assertEqual((report.graded, self.calls), (0, []))

    def test_postponed_and_invalid_results_do_not_block_other_games(self):
        postponed = {
            "id": "401705090", "date": "2025-01-10T03:30Z", "season": {"year": 2025, "type": 2},
            "competitions": [{
                "status": {"type": {"name": "STATUS_POSTPONED", "state": "post", "completed": False}},
                "competitors": [
                    {"homeAway": "home", "team": {"id": "13", "abbreviation": "LAL"}, "score": "0"},
                    {"homeAway": "away", "team": {"id": "30", "abbreviation": "CHA"}, "score": "0"},
                ],
            }],
        }
        game = data.parse_espn_scoreboard({"events": [postponed]})[0]
        store.insert_prediction_if_absent(self.engine, make_row(401705090), now=NOW)
        board = [game, finished(1, 0, 0), finished(2, 112, 101)]
        with self.assertLogs("scripts.nba_daily", level="WARNING") as logs:
            report = nba_daily.grade_finished(
                self.engine, TODAY, now=TIPOFF + timedelta(hours=5), fetch=lambda _: board, write=True
            )
        self.assertEqual(report.graded, 1)
        self.assertIn("1", " ".join(logs.output))
        pending = store.load_ungraded(self.engine, DateWindow(date(2026, 11, 1), date(2026, 11, 30)))
        self.assertEqual({row.game_id for row in pending}, {1, 401705090})


class StorePredictionsTests(unittest.TestCase):
    def test_second_run_adds_nothing_and_past_tipoff_is_refused(self):
        engine = make_engine(self)
        rows = (make_row(1), make_row(2, tipoff_utc=NOW - timedelta(hours=1)))
        run = predict.PredictionRun(
            tuple(predict.PlannedPrediction(r, "A", "B") for r in rows), (), date(2026, 11, 16), (1994, 2026), 100
        )
        first = nba_daily.store_predictions(engine, run, now=NOW)
        second = nba_daily.store_predictions(engine, run, now=NOW)
        self.assertEqual(first[InsertOutcome.INSERTED], 1)
        self.assertEqual(second[InsertOutcome.ALREADY_EXISTS], 1)
        self.assertEqual(second[InsertOutcome.PAST_TIPOFF], 1)
        with engine.connect() as connection:
            self.assertEqual(connection.execute(text("select count(*) from public.nba_predictions")).scalar(), 1)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        games = synthetic_games()
        cls.games = games
        cls.today = (games["game_date"].max() + pd.Timedelta(days=1)).date()
        cls.tip = datetime.combine(cls.today, datetime.min.time(), tzinfo=UTC) + timedelta(hours=20)
        cls.now = cls.tip - timedelta(hours=10)

    def args(self, **overrides):
        values = dict(date=self.today.isoformat(), database_url=None, dry_run=True, deliver=False, cache_dir=Path("."))
        values.update(overrides)
        return argparse.Namespace(**values)

    def inputs(self, scheduled_dates=()):
        return predict.NbaInputs(
            games=self.games, complete_seasons=tuple(sorted(self.games["season"].unique())), scheduled_dates=scheduled_dates
        )

    def fetch(self, day):
        return [scheduled(1, 1, 2, self.tip, line=3.5), scheduled(2, 3, 4, self.tip)] if day == self.today else []

    def run_script(self, args, engine=None, inputs=None):
        with mock.patch.object(nba_daily, "refresh_inputs", return_value=inputs or self.inputs()), mock.patch.object(
            nba_daily, "resolve_engine", return_value=engine
        ):
            return nba_daily.run(args, fetch=self.fetch, now=self.now)

    def test_dry_run_prints_predictions_and_never_touches_a_database(self):
        code, output = self.run_script(self.args())
        self.assertEqual(code, 0)
        self.assertIn("mode: dry-run", output)
        self.assertIn("Predictions (2)", output)
        self.assertIn("[blend]", output)
        self.assertIn("[elo+platt]", output)

    def test_write_mode_stores_once_and_a_rerun_stores_nothing_new(self):
        engine = make_engine(self)
        code, output = self.run_script(self.args(dry_run=False), engine=engine)
        self.assertEqual(code, 0)
        self.assertIn("stored 2 new, 0 upgraded to the blend after a line appeared, 0 already present, 0 past tipoff", output)
        _, again = self.run_script(self.args(dry_run=False), engine=engine)
        self.assertIn("stored 0 new, 0 upgraded to the blend after a line appeared, 2 already present, 0 past tipoff", again)
        with engine.connect() as connection:
            self.assertEqual(connection.execute(text("select count(*) from public.nba_predictions")).scalar(), 2)

    def test_a_line_that_appears_on_a_later_run_upgrades_the_stored_game(self):
        engine = make_engine(self)
        tip = self.tip

        def unlined(day):
            return [scheduled(1, 1, 2, tip)] if day == self.today else []

        def lined(day):
            return [scheduled(1, 1, 2, tip, line=3.5)] if day == self.today else []

        args = self.args(dry_run=False)
        with mock.patch.object(nba_daily, "refresh_inputs", return_value=self.inputs()), mock.patch.object(
            nba_daily, "resolve_engine", return_value=engine
        ):
            _, first = nba_daily.run(args, fetch=unlined, now=self.now)
            _, second = nba_daily.run(args, fetch=lined, now=self.now + timedelta(hours=2))
            _, third = nba_daily.run(args, fetch=lined, now=self.now + timedelta(hours=3))
            _, late = nba_daily.run(args, fetch=lined, now=tip - timedelta(minutes=1))
        self.assertIn("stored 1 new, 0 upgraded", first)
        self.assertIn("stored 0 new, 1 upgraded", second)
        self.assertIn("stored 0 new, 0 upgraded to the blend after a line appeared, 1 already present", third)
        self.assertIn("1 already present", late)
        with engine.connect() as connection:
            rows = connection.execute(text("select served_model from public.nba_predictions order by served_model")).scalars().all()
        self.assertEqual(rows, ["blend", "elo_platt"])

    def test_stale_data_exits_with_its_own_code_and_writes_nothing(self):
        engine = make_engine(self)
        stale_today = self.today + timedelta(days=9)
        inputs = self.inputs(scheduled_dates=(self.today + timedelta(days=2),))
        code, message = self.run_script(self.args(date=stale_today.isoformat(), dry_run=False), engine=engine, inputs=inputs)
        self.assertEqual(code, nba_daily.EXIT_STALE_DATA)
        self.assertIn("stale_data", message)
        with engine.connect() as connection:
            self.assertEqual(connection.execute(text("select count(*) from public.nba_predictions")).scalar(), 0)

    def test_data_errors_exit_two(self):
        with mock.patch.object(nba_daily, "refresh_inputs", side_effect=data.NbaDataError("down")), mock.patch.object(
            nba_daily, "resolve_engine", return_value=None
        ):
            code, _ = nba_daily.run(self.args(), fetch=self.fetch, now=self.now)
        self.assertEqual(code, nba_daily.EXIT_DATA_ERROR)


class EngineResolutionTests(unittest.TestCase):
    def test_dry_run_without_an_explicit_url_never_opens_a_database(self):
        args = argparse.Namespace(dry_run=True, database_url=None)
        with mock.patch.object(nba_daily, "get_engine") as get_engine:
            self.assertIsNone(nba_daily.resolve_engine(args))
        get_engine.assert_not_called()

    def test_no_configured_url_means_print_only(self):
        args = argparse.Namespace(dry_run=False, database_url=None)
        with mock.patch.object(nba_daily, "get_database_url", side_effect=RuntimeError("none")):
            self.assertIsNone(nba_daily.resolve_engine(args))


if __name__ == "__main__":
    unittest.main()
