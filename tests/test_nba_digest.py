import unittest
from dataclasses import replace
from datetime import date, timedelta
from unittest import mock

from alttez.nba import predict, store
from alttez.nba.store import PredictionStatus
from alttez.reports import nba_digest
from alttez.reports.nba_digest import DigestInput, build_nba_digest
from tests.test_nba_store import TIPOFF, make_row


def planned(game_id, row):
    return predict.PlannedPrediction(row, "TOR", "NY")


def graded(game_id, p_served, market, home_win):
    row = make_row(game_id, p_served=p_served, market=market, game_date=date(2026, 10, 1) + timedelta(days=game_id))
    grade = store.grade_outcome(p_served, 100 + (5 if home_win else -5), 100)
    return replace(
        row, status=PredictionStatus.GRADED, home_score=grade.home_score, away_score=grade.away_score,
        home_win=grade.home_win, brier=grade.brier, graded_at=TIPOFF,
    )


def prediction_run():
    blend = make_row(1, p_served=0.357, market=0.36, market_spread=4.5, expected_margin=-4.0)
    elo = make_row(2, p_served=0.684, market=None, expected_margin=5.6, tipoff_utc=TIPOFF + timedelta(days=1))
    return predict.PredictionRun(
        predictions=(planned(1, blend), planned(2, elo)),
        skipped=(predict.SkippedGame(3, "XXX @ YYY", "no rating for team id(s) [99]"),),
        newest_completed=date(2026, 11, 16),
        train_seasons=(1994, 2026),
        n_lined_train=7936,
    )


class DigestTextTests(unittest.TestCase):
    def test_digest_text_on_a_fixture(self):
        summary = store.summarise_graded([graded(1, 0.8, 0.6, True), graded(2, 0.7, 0.9, False), graded(3, 0.6, None, True)])
        text = build_nba_digest(
            DigestInput(
                today=date(2026, 11, 17), run=prediction_run(), graded_this_run=2, unresolved=1, summary=summary,
                notes=("data refresh failed, using cached copy: boom",), mode="dry-run",
            )
        )
        expected = """NBA digest 2026-11-17 (mode: dry-run)
Data: newest completed game 2026-11-16; fitted on seasons 1994-2026 (7936 lined games)
Note: data refresh failed, using cached copy: boom
Graded this run: 2; scheduled predictions still unresolved: 1

Predictions (2)
- NY @ TOR 11-18 00:30Z: home 35.7% [blend] margin -4.0 | market 36.0%, line TOR -4.5
- NY @ TOR 11-19 00:30Z: home 68.4% [elo+platt] margin +5.6
Skipped (1)
- XXX @ YYY (id 3): no rating for team id(s) [99]

Live scoreboard (graded games so far)
- served: n=3, Brier 0.2300, accuracy 66.7%
- same games with a market line (n=2): served Brier 0.2650, accuracy 50.0%; market Brier 0.4850, accuracy 50.0%
- served minus market Brier: no interval until n >= 30

Schedule and betting-line data: sportsdataverse/hoopR-nba-data (CC BY 4.0)."""
        self.assertEqual(text, expected)

    def test_interval_appears_from_thirty_market_games(self):
        rows = [graded(i, 0.65, 0.55, i % 3 != 0) for i in range(1, 31)]
        text = build_nba_digest(
            DigestInput(today=date(2026, 11, 17), run=None, graded_this_run=0, unresolved=0, summary=store.summarise_graded(rows))
        )
        self.assertIn("paired bootstrap by game date", text)
        self.assertIn("95% CI [", text)
        self.assertNotIn("no interval", text)

    def test_empty_scoreboard_and_empty_run(self):
        empty = predict.PredictionRun((), (), date(2026, 11, 16), (1994, 2026), 100)
        text = build_nba_digest(
            DigestInput(today=date(2026, 11, 17), run=empty, graded_this_run=0, unresolved=0, summary=store.summarise_graded([]))
        )
        self.assertIn("- none: no scheduled games in the window", text)
        self.assertIn("- no graded predictions yet", text)


class DeliveryTests(unittest.TestCase):
    def test_missing_telegram_config_reports_failure_without_sending(self):
        with mock.patch.object(nba_digest, "telegram_config", return_value=(None, None)), mock.patch.object(
            nba_digest, "send_message"
        ) as send:
            delivered, error = nba_digest.deliver_digest("hello")
        self.assertFalse(delivered)
        self.assertIn("not configured", error)
        send.assert_not_called()

    def test_text_is_html_escaped_and_chunked_through_the_shared_sender(self):
        with mock.patch.object(nba_digest, "telegram_config", return_value=("t", "c")), mock.patch.object(
            nba_digest, "send_message", return_value=(True, None)
        ) as send:
            delivered, _ = nba_digest.deliver_digest("A & B <b>\n\n" + "x" * 3000 + "\n\n" + "y" * 3000)
        self.assertTrue(delivered)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(send.call_args_list[0].args[:2], ("t", "c"))
        self.assertTrue(send.call_args_list[0].args[2].startswith("A &amp; B &lt;b&gt;"))

    def test_a_failed_chunk_stops_delivery(self):
        with mock.patch.object(nba_digest, "telegram_config", return_value=("t", "c")), mock.patch.object(
            nba_digest, "send_message", return_value=(False, "HTTP 400")
        ):
            self.assertEqual(nba_digest.deliver_digest("hello"), (False, "HTTP 400"))


if __name__ == "__main__":
    unittest.main()
