from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

from alttez.ingestion.live_matches import (
    filter_strict_world_cup_matches,
    infer_effective_is_neutral,
    match_day_from_kickoff,
    parse_api_football_records,
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


if __name__ == "__main__":
    unittest.main()
