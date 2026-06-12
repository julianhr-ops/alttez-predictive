from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

from alttez.ingestion.live_matches import (
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


if __name__ == "__main__":
    unittest.main()
