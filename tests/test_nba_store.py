import re
import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text

from alttez.nba import store
from alttez.nba.store import (
    DateWindow,
    InsertOutcome,
    NbaPredictionRow,
    PredictionStatus,
    ServedModel,
)
from tests.nba_sqlite import make_engine

TIPOFF = datetime(2026, 11, 18, 0, 30, tzinfo=UTC)
NOW = TIPOFF - timedelta(hours=12)
MIGRATIONS = Path(__file__).resolve().parents[1] / "sql" / "supabase" / "migrations"
MIGRATION = MIGRATIONS / "015_nba_predictions.sql"
MIGRATION_016 = MIGRATIONS / "016_nba_predictions_key_served_model.sql"


def make_row(game_id=1, *, model_version="nba-elo-v1", p_served=0.7, market=0.65, **overrides) -> NbaPredictionRow:
    values = dict(
        game_id=game_id,
        game_date=date(2026, 11, 17),
        tipoff_utc=TIPOFF,
        home_team_id=28,
        away_team_id=18,
        model_version=model_version,
        served_model=ServedModel.BLEND if market is not None else ServedModel.ELO_PLATT,
        p_home_elo=0.68,
        p_home_market=market,
        p_home_served=p_served,
        expected_margin=4.5,
        market_spread=-3.5 if market is not None else None,
        market_total=221.5 if market is not None else None,
        market_source="espn:DraftKings" if market is not None else None,
        created_at=NOW,
    )
    values.update(overrides)
    return NbaPredictionRow(**values)


class RowMappingTests(unittest.TestCase):
    def test_round_trip_through_a_database_style_record(self):
        row = make_row()
        record = store.row_to_params(row)
        self.assertEqual(store.row_from_mapping(record), row)

    def test_decimal_numerics_and_iso_strings_are_normalised(self):
        record = store.row_to_params(make_row())
        record.update(
            p_home_served=Decimal("0.700000"),
            expected_margin=Decimal("4.500"),
            tipoff_utc="2026-11-18 00:30:00.000000",
            created_at=NOW.replace(tzinfo=None),
            game_date="2026-11-17",
        )
        row = store.row_from_mapping(record)
        self.assertEqual(row.p_home_served, 0.7)
        self.assertIsInstance(row.expected_margin, float)
        self.assertEqual(row.tipoff_utc, TIPOFF)
        self.assertEqual(row.created_at, NOW)
        self.assertEqual(row.game_date, date(2026, 11, 17))

    def test_graded_record_keeps_result_columns(self):
        record = store.row_to_params(make_row())
        record.update(
            status="graded", home_score=110, away_score=99, home_win=1, brier=Decimal("0.09"),
            graded_at="2026-11-18 03:00:00",
        )
        row = store.row_from_mapping(record)
        self.assertIs(row.status, PredictionStatus.GRADED)
        self.assertIs(row.home_win, True)
        self.assertEqual((row.home_score, row.away_score, row.brier), (110, 99, 0.09))
        self.assertEqual(row.graded_at, datetime(2026, 11, 18, 3, 0, tzinfo=UTC))

    def test_params_are_utc_and_use_stored_strings(self):
        eastern = make_row(tipoff_utc=TIPOFF.astimezone(timezone(timedelta(hours=-5))))
        params = store.row_to_params(eastern)
        self.assertEqual(params["tipoff_utc"].utcoffset(), timedelta(0))
        self.assertEqual(params["tipoff_utc"], TIPOFF)
        self.assertEqual(params["served_model"], "blend")
        self.assertEqual(params["status"], "scheduled")

    def test_invalid_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            make_row(tipoff_utc=TIPOFF.replace(tzinfo=None))
        with self.assertRaises(ValueError):
            make_row(p_served=1.2)
        with self.assertRaises(ValueError):
            make_row(market=None, served_model=ServedModel.BLEND)


class InsertSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine(self)

    def _stored(self, game_id=1):
        with self.engine.connect() as connection:
            return connection.execute(
                text("select p_home_served, served_model, created_at from public.nba_predictions where game_id = :g"),
                {"g": game_id},
            ).one()

    def test_first_insert_writes_and_second_does_not_change_the_row(self):
        first = make_row(p_served=0.70)
        second = make_row(p_served=0.31, market=None, created_at=NOW + timedelta(hours=3))
        self.assertIs(store.insert_prediction_if_absent(self.engine, first, now=NOW), InsertOutcome.INSERTED)
        before = self._stored()
        self.assertIs(
            store.insert_prediction_if_absent(self.engine, second, now=NOW + timedelta(hours=3)),
            InsertOutcome.ALREADY_EXISTS,
        )
        self.assertEqual(tuple(self._stored()), tuple(before))
        self.assertEqual(before.p_home_served, 0.70)
        self.assertEqual(before.served_model, "blend")

    def test_a_new_model_version_is_a_separate_row(self):
        store.insert_prediction_if_absent(self.engine, make_row(), now=NOW)
        outcome = store.insert_prediction_if_absent(self.engine, make_row(model_version="nba-elo-v2"), now=NOW)
        self.assertIs(outcome, InsertOutcome.INSERTED)

    def test_no_prediction_is_written_once_the_game_has_tipped_off(self):
        for now in (TIPOFF, TIPOFF + timedelta(minutes=1)):
            outcome = store.insert_prediction_if_absent(self.engine, make_row(), now=now)
            self.assertIs(outcome, InsertOutcome.PAST_TIPOFF)
        with self.engine.connect() as connection:
            self.assertEqual(connection.execute(text("select count(*) from public.nba_predictions")).scalar(), 0)

    def test_a_graded_row_cannot_be_inserted(self):
        graded = replace(make_row(), status=PredictionStatus.GRADED)
        with self.assertRaises(ValueError):
            store.insert_prediction_if_absent(self.engine, graded, now=NOW)

    def test_stored_row_reads_back_equal(self):
        row = make_row()
        store.insert_prediction_if_absent(self.engine, row, now=NOW)
        loaded = store.load_ungraded(self.engine, DateWindow(date(2026, 11, 17), date(2026, 11, 17)))
        self.assertEqual(loaded, [row])


class GradingTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine(self)

    def test_grade_outcome_math(self):
        win = store.grade_outcome(0.8, 110, 100)
        self.assertTrue(win.home_win)
        self.assertAlmostEqual(win.brier, 0.04)
        loss = store.grade_outcome(0.8, 90, 100)
        self.assertFalse(loss.home_win)
        self.assertAlmostEqual(loss.brier, 0.64)
        with self.assertRaises(ValueError):
            store.grade_outcome(0.5, 100, 100)

    def test_grading_fills_results_once_and_leaves_the_prediction_untouched(self):
        row = make_row(p_served=0.7)
        store.insert_prediction_if_absent(self.engine, row, now=NOW)
        graded_at = TIPOFF + timedelta(hours=4)
        self.assertTrue(
            store.grade_prediction(self.engine, 1, row.model_version, row.served_model, 100, 108, 0.7, now=graded_at)
        )
        self.assertFalse(
            store.grade_prediction(
                self.engine, 1, row.model_version, row.served_model, 120, 90, 0.7, now=graded_at + timedelta(hours=1)
            )
        )
        window = DateWindow(date(2026, 11, 1), date(2026, 11, 30))
        self.assertEqual(store.load_ungraded(self.engine, window), [])
        summary = store.load_graded_summary(self.engine, window)
        self.assertEqual(summary.n, 1)
        stored = summary.rows[0]
        self.assertEqual((stored.home_score, stored.away_score, stored.home_win), (100, 108, False))
        self.assertAlmostEqual(stored.brier, 0.49)
        self.assertEqual(stored.graded_at, graded_at)
        self.assertEqual(replace(stored, status=PredictionStatus.SCHEDULED, home_score=None, away_score=None,
                                 home_win=None, brier=None, graded_at=None), row)

    def test_grading_an_unknown_game_changes_nothing(self):
        self.assertFalse(store.grade_prediction(self.engine, 99, "nba-elo-v1", ServedModel.BLEND, 100, 90, 0.5, now=NOW))

    def test_window_filters_by_game_date(self):
        store.insert_prediction_if_absent(self.engine, make_row(1), now=NOW)
        store.insert_prediction_if_absent(
            self.engine, make_row(2, game_date=date(2026, 11, 19), tipoff_utc=TIPOFF + timedelta(days=2)), now=NOW
        )
        loaded = store.load_ungraded(self.engine, DateWindow(date(2026, 11, 18), date(2026, 11, 30)))
        self.assertEqual([r.game_id for r in loaded], [2])
        with self.assertRaises(ValueError):
            DateWindow(date(2026, 11, 2), date(2026, 11, 1))


class GradedSummaryTests(unittest.TestCase):
    def _graded(self, game_id, p_served, market, home_win):
        row = make_row(game_id, p_served=p_served, market=market, game_date=date(2026, 11, 1) + timedelta(days=game_id))
        grade = store.grade_outcome(p_served, 100 + (5 if home_win else -5), 100)
        return replace(
            row, status=PredictionStatus.GRADED, home_score=grade.home_score, away_score=grade.away_score,
            home_win=grade.home_win, brier=grade.brier, graded_at=TIPOFF,
        )

    def test_paired_numbers_use_only_games_with_a_market_line(self):
        rows = [
            self._graded(1, 0.8, 0.6, True),
            self._graded(2, 0.7, 0.9, False),
            self._graded(3, 0.6, None, True),
        ]
        summary = store.summarise_graded(rows)
        self.assertEqual((summary.n, summary.n_market), (3, 2))
        self.assertAlmostEqual(summary.served_brier(), (0.04 + 0.49 + 0.16) / 3)
        self.assertAlmostEqual(summary.served_brier(market_games_only=True), (0.04 + 0.49) / 2)
        self.assertAlmostEqual(summary.market_brier(), (0.16 + 0.81) / 2)
        self.assertEqual(summary.served_accuracy(), 2 / 3)
        self.assertEqual(summary.market_accuracy(), 0.5)
        self.assertEqual(len(summary.served_losses()), len(summary.market_losses()))

    def test_scheduled_rows_are_ignored(self):
        summary = store.summarise_graded([make_row(1)])
        self.assertEqual(summary.n, 0)


class LineUpgradeTests(unittest.TestCase):
    LATER = NOW + timedelta(hours=6)

    def setUp(self):
        self.engine = make_engine(self)
        self.elo = make_row(p_served=0.6, market=None)
        self.blend = make_row(p_served=0.72, market=0.7, created_at=self.LATER)

    def _rows(self):
        with self.engine.connect() as connection:
            return connection.execute(text("select * from public.nba_predictions order by served_model")).all()

    def _loaded(self):
        return store.load_ungraded(self.engine, DateWindow(date(2026, 11, 17), date(2026, 11, 17)))

    def test_a_line_that_appears_later_adds_a_blend_row_and_leaves_the_elo_row_untouched(self):
        self.assertIs(store.insert_prediction_if_absent(self.engine, self.elo, now=NOW), InsertOutcome.INSERTED)
        elo_before = self._rows()[0]
        outcome = store.insert_prediction_if_absent(self.engine, self.blend, now=self.LATER)
        self.assertIs(outcome, InsertOutcome.UPGRADED)
        blend_row, elo_row = self._rows()
        self.assertEqual(tuple(elo_row), tuple(elo_before))
        self.assertEqual((blend_row.served_model, blend_row.p_home_served), ("blend", 0.72))
        self.assertEqual(sorted(r.served_model for r in self._loaded()), [ServedModel.BLEND, ServedModel.ELO_PLATT])

    def test_a_game_that_never_gets_a_line_keeps_a_single_row(self):
        for now in (NOW, self.LATER):
            store.insert_prediction_if_absent(self.engine, replace(self.elo, created_at=now), now=now)
        (stored,) = self._loaded()
        self.assertEqual((stored.served_model, stored.created_at), (ServedModel.ELO_PLATT, NOW))

    def test_a_line_that_first_appears_after_tipoff_writes_nothing(self):
        store.insert_prediction_if_absent(self.engine, self.elo, now=NOW)
        before = self._rows()
        for now in (TIPOFF, TIPOFF + timedelta(hours=1)):
            self.assertIs(store.insert_prediction_if_absent(self.engine, self.blend, now=now), InsertOutcome.PAST_TIPOFF)
        self.assertEqual(self._rows(), before)

    def test_rerunning_the_upgrade_is_idempotent(self):
        store.insert_prediction_if_absent(self.engine, self.elo, now=NOW)
        store.insert_prediction_if_absent(self.engine, self.blend, now=self.LATER)
        before = self._rows()
        again = replace(self.blend, p_home_served=0.5, created_at=self.LATER + timedelta(hours=1))
        self.assertIs(
            store.insert_prediction_if_absent(self.engine, again, now=self.LATER + timedelta(hours=1)),
            InsertOutcome.ALREADY_EXISTS,
        )
        self.assertIs(store.insert_prediction_if_absent(self.engine, self.elo, now=self.LATER), InsertOutcome.ALREADY_EXISTS)
        self.assertEqual(self._rows(), before)

    def test_an_elo_row_never_follows_an_existing_blend_row(self):
        store.insert_prediction_if_absent(self.engine, self.blend, now=NOW)
        outcome = store.insert_prediction_if_absent(self.engine, self.elo, now=self.LATER)
        self.assertIs(outcome, InsertOutcome.ALREADY_EXISTS)
        self.assertEqual([r.served_model for r in self._rows()], ["blend"])

    def test_both_rows_are_graded_each_from_its_own_probability(self):
        store.insert_prediction_if_absent(self.engine, self.elo, now=NOW)
        store.insert_prediction_if_absent(self.engine, self.blend, now=self.LATER)
        at = TIPOFF + timedelta(hours=4)
        for row in (self.elo, self.blend):
            self.assertTrue(
                store.grade_prediction(self.engine, 1, row.model_version, row.served_model, 110, 100, row.p_home_served, now=at)
            )
        window = DateWindow(date(2026, 11, 1), date(2026, 11, 30))
        graded = {r.served_model: r for r in store.load_graded_summary(self.engine, window).all_rows}
        self.assertEqual(set(graded), {ServedModel.BLEND, ServedModel.ELO_PLATT})
        for model, p in ((ServedModel.ELO_PLATT, 0.6), (ServedModel.BLEND, 0.72)):
            self.assertEqual((graded[model].home_score, graded[model].away_score, graded[model].home_win), (110, 100, True))
            self.assertAlmostEqual(graded[model].brier, (p - 1) ** 2)

    def test_grading_one_row_leaves_its_sibling_scheduled(self):
        store.insert_prediction_if_absent(self.engine, self.elo, now=NOW)
        store.insert_prediction_if_absent(self.engine, self.blend, now=self.LATER)
        store.grade_prediction(self.engine, 1, self.elo.model_version, ServedModel.ELO_PLATT, 90, 100, 0.6, now=TIPOFF)
        self.assertEqual([r.served_model for r in self._loaded()], [ServedModel.BLEND])


class ServedRowTests(unittest.TestCase):
    @staticmethod
    def _graded(row, home_win=True):
        grade = store.grade_outcome(row.p_home_served, 105 if home_win else 95, 100)
        return replace(
            row, status=PredictionStatus.GRADED, home_score=grade.home_score, away_score=grade.away_score,
            home_win=grade.home_win, brier=grade.brier, graded_at=TIPOFF,
        )

    def test_the_blend_serves_a_game_that_has_both_rows_whatever_the_order(self):
        elo = make_row(1, p_served=0.6, market=None)
        blend = make_row(1, p_served=0.72, market=0.7, created_at=NOW + timedelta(hours=6))
        for rows in ([elo, blend], [blend, elo]):
            (served,) = store.served_rows(rows)
            self.assertIs(served.served_model, ServedModel.BLEND)

    def test_one_row_per_game_in_tipoff_order(self):
        late = make_row(1, tipoff_utc=TIPOFF + timedelta(hours=3))
        early_elo = make_row(2, market=None)
        early_blend = make_row(2)
        self.assertEqual([r.game_id for r in store.served_rows([late, early_elo, early_blend])], [2, 1])

    def test_ties_between_model_versions_go_to_the_newest_then_the_larger_version(self):
        old = make_row(1, model_version="nba-elo-v1", created_at=NOW)
        new = make_row(1, model_version="nba-elo-v2", created_at=NOW + timedelta(hours=1))
        same_time = make_row(1, model_version="nba-elo-v3", created_at=NOW)
        self.assertEqual(store.served_rows([old, new])[0].model_version, "nba-elo-v2")
        self.assertEqual(store.served_rows([same_time, old])[0].model_version, "nba-elo-v3")

    def test_the_scoreboard_counts_an_upgraded_game_once_with_its_blend_row(self):
        elo = self._graded(make_row(1, p_served=0.6, market=None))
        blend = self._graded(make_row(1, p_served=0.8, market=0.7, created_at=NOW + timedelta(hours=6)))
        other = self._graded(make_row(2, p_served=0.5, market=None))
        summary = store.summarise_graded([elo, blend, other])
        self.assertEqual((summary.n, summary.n_market, len(summary.all_rows)), (2, 1, 3))
        self.assertAlmostEqual(summary.served_brier(), (0.04 + 0.25) / 2)
        self.assertAlmostEqual(summary.served_brier(market_games_only=True), 0.04)
        self.assertEqual(summary.upgraded_pairs, ((elo, blend),))


class MigrationContractTests(unittest.TestCase):
    def test_migration_columns_match_the_row_dataclass(self):
        sql = MIGRATION.read_text(encoding="utf-8")
        body = sql.split("create table if not exists public.nba_predictions (", 1)[1]
        body = body.split("constraint pk_nba_predictions", 1)[0]
        columns = re.findall(r"^\s{4}([a-z_]+) [a-z]+", body, flags=re.MULTILINE)
        self.assertEqual(tuple(columns), store.COLUMNS)

    def test_migration_is_idempotent_and_keys_on_game_and_version(self):
        sql = MIGRATION.read_text(encoding="utf-8").lower()
        self.assertIn("create table if not exists public.nba_predictions", sql)
        self.assertIn("primary key (game_id, model_version)", sql)
        self.assertNotIn("drop table", sql)

    def test_migration_016_rekeys_on_served_model_without_touching_data(self):
        sql = MIGRATION_016.read_text(encoding="utf-8").lower()
        self.assertIn("drop constraint if exists pk_nba_predictions", sql)
        self.assertIn("primary key (game_id, model_version, served_model)", sql)
        for forbidden in ("drop table", "delete ", "update ", "truncate", "--"):
            self.assertNotIn(forbidden, sql)

    def test_sqlite_fixture_key_matches_migration_016(self):
        sql = MIGRATION_016.read_text(encoding="utf-8").lower()
        key = [c.strip() for c in re.search(r"primary key \(([^)]*)\)", sql).group(1).split(",")]
        with make_engine(self).connect() as connection:
            info = connection.execute(text("pragma public.table_info(nba_predictions)")).all()
        in_key = [r.name for r in sorted((r for r in info if r.pk), key=lambda r: r.pk)]
        self.assertEqual(in_key, key)


if __name__ == "__main__":
    unittest.main()
