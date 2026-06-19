from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from alttez.ingestion.live_matches import (
    _dedupe_agenda_by_provider,
    filter_strict_world_cup_matches,
    infer_effective_is_neutral,
    map_espn_status,
    match_day_from_kickoff,
    parse_api_football_records,
    parse_espn_records,
    parse_football_data_records,
    parse_local_json_records,
)


class TestLiveMatchesTimezoneHandling(unittest.TestCase):
    @patch.dict(os.environ, {"ALTTEZ_MATCH_LOCAL_TIMEZONE": "America/Bogota"}, clear=False)
    def test_match_day_from_kickoff_uses_operational_timezone_for_utc_crossovers(self) -> None:
        match_day = match_day_from_kickoff("2026-06-12T01:30:00+00:00", date(2026, 6, 11))

        self.assertEqual(match_day, date(2026, 6, 11))

    @patch.dict(os.environ, {"ALTTEZ_MATCH_LOCAL_TIMEZONE": "America/Bogota"}, clear=False)
    def test_parse_api_football_records_keeps_late_local_match_on_requested_day(self) -> None:
        records = [
            {
                "fixture": {
                    "id": 501,
                    "date": "2026-06-12T01:30:00+00:00",
                    "status": {"short": "NS"},
                    "venue": {"name": "Metropolitano", "city": "Barranquilla"},
                },
                "league": {
                    "name": "International Friendly",
                    "round": "Friendly",
                    "country": "World",
                    "season": 2026,
                },
                "teams": {
                    "home": {"name": "Colombia"},
                    "away": {"name": "Peru"},
                },
                "goals": {"home": None, "away": None},
            }
        ]

        matches, _ = parse_api_football_records(records, fetched_at=None, target_date=date(2026, 6, 11))

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].match_date, date(2026, 6, 11))
        self.assertEqual(matches[0].kickoff_at, "2026-06-12T01:30:00+00:00")

    @patch.dict(os.environ, {"ALTTEZ_MATCH_LOCAL_TIMEZONE": "America/Bogota"}, clear=False)
    def test_parse_football_data_records_keeps_late_local_match_on_requested_day(self) -> None:
        records = [
            {
                "id": 601,
                "utcDate": "2026-06-12T02:15:00+00:00",
                "status": "SCHEDULED",
                "stage": "GROUP_STAGE",
                "group": "Group A",
                "homeTeam": {"name": "Ecuador"},
                "awayTeam": {"name": "Venezuela"},
                "competition": {"name": "Copa America", "area": {"name": "South America"}},
                "score": {"fullTime": {"home": None, "away": None}},
                "season": {"id": 2026},
                "venue": "Quito Arena",
            }
        ]

        matches, _ = parse_football_data_records(records, fetched_at=None, target_date=date(2026, 6, 11))

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].match_date, date(2026, 6, 11))

    @patch.dict(os.environ, {"ALTTEZ_MATCH_LOCAL_TIMEZONE": "America/Bogota"}, clear=False)
    def test_parse_local_json_records_keeps_late_local_match_on_requested_day(self) -> None:
        records = [
            {
                "id": "701",
                "kickoff_at": "2026-06-12T03:00:00+00:00",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "tournament": "Friendly Series",
                "stage": "Friendly",
                "country": "World",
                "status": "scheduled",
            }
        ]

        matches, _ = parse_local_json_records(records, fetched_at=None, target_date=date(2026, 6, 11))

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].match_date, date(2026, 6, 11))

    def test_live_team_aliases_canonicalize_united_states_to_usa(self) -> None:
        records = [
            {
                "id": "wc-usa-1",
                "kickoff_at": "2026-06-12T20:00:00+00:00",
                "home_team": "United States",
                "away_team": "Türkiye",
                "tournament": "FIFA World Cup 2026",
                "stage": "Group Stage",
                "status": "scheduled",
            }
        ]

        matches, _ = parse_local_json_records(records, fetched_at=None, target_date=date(2026, 6, 12))

        self.assertEqual(matches[0].home_team, "United States")
        self.assertEqual(matches[0].away_team, "Turkey")

    def test_world_cup_without_venue_country_is_neutral_for_non_hosts(self) -> None:
        self.assertTrue(
            infer_effective_is_neutral(
                tournament_name="FIFA World Cup",
                competition_class="world_cup",
                home_team="Australia",
                away_team="Turkey",
                venue_country_name=None,
                provider_is_neutral=False,
            )
        )

    def test_world_cup_without_venue_country_keeps_host_as_non_neutral(self) -> None:
        self.assertFalse(
            infer_effective_is_neutral(
                tournament_name="FIFA World Cup",
                competition_class="world_cup",
                home_team="United States",
                away_team="Paraguay",
                venue_country_name=None,
                provider_is_neutral=False,
            )
        )

    def test_strict_world_cup_filter_omits_friendlies_and_other_world_cups(self) -> None:
        records = [
            {
                "id": "1",
                "kickoff_at": "2026-06-12T20:00:00+00:00",
                "home_team": "USA",
                "away_team": "Mexico",
                "tournament": "FIFA World Cup 2026",
                "stage": "Group Stage",
                "status": "scheduled",
            },
            {
                "id": "2",
                "kickoff_at": "2026-06-12T22:00:00+00:00",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "tournament": "International Friendly",
                "stage": "Friendly",
                "status": "scheduled",
            },
            {
                "id": "3",
                "kickoff_at": "2026-06-12T23:00:00+00:00",
                "home_team": "France",
                "away_team": "Germany",
                "tournament": "FIFA World Cup",
                "stage": "Group Stage",
                "status": "scheduled",
            },
        ]
        matches, raw_records = parse_local_json_records(records, fetched_at=None, target_date=date(2026, 6, 12))

        filtered_matches, filtered_raw = filter_strict_world_cup_matches(matches, raw_records)

        self.assertEqual([match.external_match_id for match in filtered_matches], ["1"])
        self.assertEqual([record.external_match_id for record in filtered_raw], ["1"])


class TestEspnProvider(unittest.TestCase):
    def _event(self) -> dict:
        return {
            "id": 760435,
            "date": "2026-06-17T17:00Z",
            "season": {"year": 2026, "slug": "group-stage"},
            "_alttez_league_name": "FIFA World Cup",
            "_alttez_season_year": 2026,
            "competitions": [
                {
                    "date": "2026-06-17T17:00Z",
                    "neutralSite": True,
                    "status": {"type": {"state": "pre"}},
                    "venue": {"fullName": "NRG Stadium", "address": {"city": "Houston", "country": "USA"}},
                    "competitors": [
                        {"homeAway": "home", "score": "0", "team": {"displayName": "Portugal"}},
                        {"homeAway": "away", "score": "0", "team": {"displayName": "Congo DR"}},
                    ],
                }
            ],
        }

    def test_map_espn_status(self) -> None:
        self.assertEqual(map_espn_status("pre"), "scheduled")
        self.assertEqual(map_espn_status("in"), "in_progress")
        self.assertEqual(map_espn_status("post"), "finished")
        self.assertEqual(map_espn_status(None), "scheduled")

    def test_parse_espn_records_builds_world_cup_match(self) -> None:
        matches, raw = parse_espn_records([self._event()], fetched_at=None, target_date=date(2026, 6, 17))

        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match.provider, "espn")
        self.assertEqual(match.external_match_id, "760435")
        self.assertEqual(match.home_team, "Portugal")
        self.assertEqual(match.tournament_name, "FIFA World Cup")
        self.assertEqual(match.season_label, "2026")
        self.assertEqual(match.competition_class, "world_cup")
        self.assertTrue(match.is_neutral)
        self.assertEqual(match.venue_name, "NRG Stadium")
        self.assertEqual(match.status, "scheduled")
        self.assertEqual([record.external_match_id for record in raw], ["760435"])

    def test_parse_espn_records_passes_strict_world_cup_filter(self) -> None:
        matches, raw = parse_espn_records([self._event()], fetched_at=None, target_date=date(2026, 6, 17))
        filtered, _ = filter_strict_world_cup_matches(matches, raw)

        self.assertEqual([match.external_match_id for match in filtered], ["760435"])

    def test_parse_espn_records_skips_event_without_competitors(self) -> None:
        matches, _ = parse_espn_records([{"id": 1, "competitions": [{}]}], fetched_at=None, target_date=date(2026, 6, 17))

        self.assertEqual(matches, [])


class TestAgendaDedup(unittest.TestCase):
    def test_dedupe_prefers_higher_priority_provider(self) -> None:
        frame = pd.DataFrame(
            {
                "provider": ["football_data_org", "espn"],
                "home_team": ["Portugal", "Portugal"],
                "away_team": ["Congo DR", "Congo DR"],
                "match_date": [date(2026, 6, 17), date(2026, 6, 17)],
                "kickoff_at": ["2026-06-17T17:00Z", "2026-06-17T17:00Z"],
            }
        )

        deduped = _dedupe_agenda_by_provider(frame)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped.iloc[0]["provider"], "espn")

    def test_dedupe_keeps_distinct_fixtures(self) -> None:
        frame = pd.DataFrame(
            {
                "provider": ["espn", "espn"],
                "home_team": ["Portugal", "England"],
                "away_team": ["Congo DR", "Croatia"],
                "match_date": [date(2026, 6, 17), date(2026, 6, 17)],
                "kickoff_at": ["2026-06-17T17:00Z", "2026-06-17T20:00Z"],
            }
        )

        self.assertEqual(len(_dedupe_agenda_by_provider(frame)), 2)


if __name__ == "__main__":
    unittest.main()
