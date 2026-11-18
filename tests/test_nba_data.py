import copy
import io
import json
import os
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from alttez.nba import data
from alttez.nba.data import NbaDataError, SeasonType

FIXTURES = Path(__file__).parent / "fixtures"


def _sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads((FIXTURES / "nba_sample_games.json").read_text(encoding="utf-8"))
    return pd.DataFrame(payload["schedule"]), pd.DataFrame(payload["lines"])


def _junk_row(**overrides) -> dict:
    row = {
        "id": 999_000_001, "date": "2026-01-10T01:00Z", "season": 2026, "season_type": 2,
        "status_type_completed": True, "status_type_state": "post", "neutral_site": False,
        "home_id": 1, "away_id": 2, "home_score": 100, "away_score": 90,
    }
    row.update(overrides)
    return row


class NormaliseScheduleTests(unittest.TestCase):
    def test_types_and_filters(self):
        raw, _ = _sample()
        junk = pd.DataFrame(
            [
                _junk_row(id=900_000_001, home_id=31),  # all-star team
                _junk_row(id=900_000_002, home_score=0, away_score=0),  # lockout 0-0
                _junk_row(id=900_000_003, home_score=101, away_score=101),
                _junk_row(id=900_000_004, status_type_completed=False, status_type_state="post"),
                _junk_row(id=900_000_005, season_type=1),  # preseason
            ]
        )
        raw = pd.concat([raw, junk, raw.iloc[[0]]], ignore_index=True)
        schedule = data.normalise_schedule(raw)
        games = schedule.games
        self.assertEqual(len(games), 60)
        self.assertEqual(tuple(games.columns), data.GAME_COLUMNS)
        self.assertTrue(games["game_id"].is_unique)
        self.assertTrue(games["game_date"].is_monotonic_increasing)
        self.assertEqual(games["home_score"].dtype, np.int64)
        self.assertTrue(set(games["home_id"]) <= data.NBA_TEAM_IDS)

    def test_local_game_date_assigns_late_tipoffs_to_the_evening_before(self):
        raw = pd.DataFrame([_junk_row(id=1, date="2026-01-11T00:30Z"), _junk_row(id=2, date="2026-01-10T17:00Z")])
        games = data.normalise_schedule(raw).games.set_index("game_id")
        self.assertEqual(games.loc[1, "game_date"], pd.Timestamp("2026-01-10"))
        self.assertEqual(games.loc[2, "game_date"], pd.Timestamp("2026-01-10"))

    def test_season_completeness(self):
        raw = pd.DataFrame(
            [
                _junk_row(id=1, season=2025, season_type=3),
                _junk_row(id=2, season=2026, season_type=2),
                _junk_row(id=3, season=2026, season_type=2, status_type_completed=False, status_type_state="pre"),
            ]
        )
        schedule = data.normalise_schedule(raw)
        self.assertEqual(schedule.complete_seasons, (2025,))
        self.assertEqual(schedule.in_progress_season, 2026)

    def test_missing_columns_raise(self):
        with self.assertRaises(NbaDataError):
            data.normalise_schedule(pd.DataFrame({"id": [1]}))

    def test_season_types_kept(self):
        raw = pd.DataFrame([_junk_row(id=i + 1, season_type=int(t)) for i, t in enumerate(SeasonType)])
        self.assertEqual(set(data.normalise_schedule(raw).games["season_type"]), {2, 3, 5})


class ClosingLineSignTests(unittest.TestCase):
    """The hoopR ``home_team_spread`` is the expected home margin.

    Measured on the full file (7,936 closing lines joined to final scores,
    2020-21..2025-26): corr(home_team_spread, home_score - away_score) = +0.4725,
    corr(home_point, margin) = -0.4725, and the residual margin - spread has mean -0.06
    and sd 13.5 points. So a positive ``home_margin_line`` means the home team is favoured.
    """

    def setUp(self):
        raw_schedule, raw_lines = _sample()
        self.games = data.normalise_schedule(raw_schedule).games
        self.raw_lines = raw_lines
        self.lines = data.normalise_closing_lines(raw_lines)
        self.joined = data.attach_lines(self.games, self.lines)
        self.joined["margin"] = self.joined["home_score"] - self.joined["away_score"]

    def test_home_margin_line_is_negated_sportsbook_home_line(self):
        by_id = self.raw_lines.set_index("game_id")["home_point"]
        np.testing.assert_allclose(self.lines.set_index("game_id")["home_margin_line"], -by_id.loc[self.lines["game_id"]].to_numpy())

    def test_positive_line_predicts_home_margin(self):
        self.assertEqual(self.joined["home_margin_line"].isna().sum(), 0)
        self.assertGreater(np.corrcoef(self.joined["home_margin_line"], self.joined["margin"])[0, 1], 0.3)
        favourites = self.joined[self.joined["home_margin_line"] > 0]
        underdogs = self.joined[self.joined["home_margin_line"] < 0]
        self.assertGreater((favourites["margin"] > 0).mean(), (underdogs["margin"] > 0).mean())

    def test_duplicate_line_ids_raise(self):
        with self.assertRaises(NbaDataError):
            data.normalise_closing_lines(pd.concat([self.raw_lines, self.raw_lines.iloc[[0]]], ignore_index=True))

    def test_attach_lines_keeps_games_without_a_line(self):
        joined = data.attach_lines(self.games, self.lines.iloc[:10])
        self.assertEqual(len(joined), len(self.games))
        self.assertEqual(int(joined["home_margin_line"].notna().sum()), 10)


class EspnParserTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((FIXTURES / "nba_espn_scoreboard.json").read_text(encoding="utf-8"))
        self.games = {g.game_id: g for g in data.parse_espn_scoreboard(self.payload)}
        self.by_teams = {(g.away_abbreviation, g.home_abbreviation): g for g in self.games.values()}

    def test_away_favourite_gives_negative_home_margin(self):
        game = self.by_teams[("NY", "TOR")]
        self.assertEqual(game.home_margin_line, -4.5)
        self.assertEqual(game.total_line, 214.5)
        self.assertEqual(game.odds_provider, "DraftKings")
        self.assertEqual((game.home_id, game.away_id), (28, 18))
        self.assertEqual(game.state, "pre")
        self.assertIsNone(game.home_score)

    def test_home_favourite_gives_positive_home_margin(self):
        self.assertEqual(self.by_teams[("POR", "MIN")].home_margin_line, 1.5)

    def test_game_without_odds(self):
        game = self.by_teams[("MEM", "CLE")]
        self.assertIsNone(game.home_margin_line)
        self.assertIsNone(game.total_line)
        self.assertIsNone(game.odds_provider)
        self.assertFalse(game.neutral)
        self.assertEqual(game.season, 2027)
        self.assertEqual(game.season_type, 2)
        self.assertEqual(game.start_time.isoformat(), "2026-11-19T00:00:00+00:00")

    def test_finished_game_has_scores_and_play_in_type(self):
        final = self.by_teams[("IND", "PHI")]
        self.assertEqual((final.state, final.home_score, final.away_score), ("post", 131, 124))
        self.assertEqual(self.by_teams[("MIA", "CHA")].season_type, SeasonType.PLAY_IN)

    def test_contradicting_favourite_flags_raise(self):
        payload = copy.deepcopy(self.payload)
        odds = payload["events"][0]["competitions"][0]["odds"][0]
        odds["homeTeamOdds"]["favorite"] = True
        odds["awayTeamOdds"]["favorite"] = False
        with self.assertRaises(NbaDataError):
            data.parse_espn_scoreboard(payload)

    def test_pickem_is_zero_margin(self):
        payload = copy.deepcopy(self.payload)
        odds = payload["events"][0]["competitions"][0]["odds"][0]
        odds["spread"] = 0
        self.assertEqual(data.parse_espn_scoreboard(payload)[0].home_margin_line, 0.0)

    def test_malformed_payloads_raise(self):
        with self.assertRaises(NbaDataError):
            data.parse_espn_scoreboard({"leagues": []})
        with self.assertRaises(NbaDataError):
            data.parse_espn_scoreboard({"events": [{"id": "1"}]})
        self.assertEqual(data.parse_espn_scoreboard({"events": []}), [])


class FetchEspnTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((FIXTURES / "nba_espn_scoreboard.json").read_text(encoding="utf-8"))
        self.sleeps: list[float] = []

    def test_retries_with_backoff_then_succeeds(self):
        calls = []

        def flaky(url, timeout):
            calls.append(url)
            if len(calls) < 3:
                raise NbaDataError("HTTP 503")
            return self.payload

        games = data.fetch_espn_scoreboard(date(2026, 11, 17), get_json=flaky, sleep=self.sleeps.append)
        self.assertEqual(len(games), 5)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0].endswith("?dates=20261117"))
        self.assertIn(2.0, self.sleeps)
        self.assertIn(4.0, self.sleeps)

    def test_gives_up_with_explicit_error(self):
        def failing(url, timeout):
            raise NbaDataError("HTTP 503")

        with self.assertRaises(NbaDataError) as ctx:
            data.fetch_espn_scoreboard(date(2026, 11, 17), retries=2, get_json=failing, sleep=self.sleeps.append)
        self.assertIn("failed after 2 attempts", str(ctx.exception))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "nested" / "x.parquet"
        buffer = io.BytesIO()
        pd.DataFrame({"a": [1, 2]}).to_parquet(buffer)
        self.body = buffer.getvalue()

    def test_downloads_then_serves_fresh_cache(self):
        with mock.patch.object(data, "_read_url", return_value=self.body) as read:
            first = data.load_cached_parquet("http://x", self.path)
            second = data.load_cached_parquet("http://x", self.path)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(list(first["a"]), [1, 2])
        self.assertEqual(list(second["a"]), [1, 2])
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["x.parquet"])

    def test_refreshes_after_twelve_hours(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.body)
        old = time.time() - 13 * 3600
        os.utime(self.path, (old, old))
        with mock.patch.object(data, "_read_url", return_value=self.body) as read:
            data.load_cached_parquet("http://x", self.path)
        self.assertEqual(read.call_count, 1)

    def test_failed_refresh_raises_and_keeps_old_cache(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.body)
        old = time.time() - 13 * 3600
        os.utime(self.path, (old, old))
        with mock.patch.object(data, "_read_url", side_effect=NbaDataError("offline")), self.assertRaises(NbaDataError):
            data.load_cached_parquet("http://x", self.path)
        self.assertEqual(self.path.read_bytes(), self.body)

    def test_garbage_body_does_not_poison_cache(self):
        with mock.patch.object(data, "_read_url", return_value=b"<html>rate limited</html>"), self.assertRaises(NbaDataError):
            data.load_cached_parquet("http://x", self.path)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
