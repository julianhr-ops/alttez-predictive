from __future__ import annotations

import unittest
from datetime import date

from alttez.ingestion.espn_results import extract_espn_finished_results
from alttez.ingestion.results_backfill import (
    CanonicalRow,
    FinishedResult,
    extract_finished_results,
    match_result_to_row,
)


class TestExtractFinishedResults(unittest.TestCase):
    def test_parses_football_data_org_shape(self) -> None:
        records = [
            {
                "status": "FINISHED",
                "utcDate": "2026-06-12T18:00:00Z",
                "homeTeam": {"name": "South Korea"},
                "awayTeam": {"name": "Czechia"},
                "score": {"fullTime": {"home": 2, "away": 1}},
            }
        ]
        results = extract_finished_results(records)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].match_date, date(2026, 6, 12))
        self.assertEqual((results[0].home_score, results[0].away_score), (2, 1))

    def test_skips_unfinished_and_unscored(self) -> None:
        records = [
            {"status": "SCHEDULED", "utcDate": "2026-06-16T18:00:00Z", "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}},
            {"status": "FINISHED", "utcDate": "2026-06-16T18:00:00Z", "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}, "score": {"fullTime": {"home": None, "away": None}}},
        ]
        self.assertEqual(extract_finished_results(records), [])


class TestExtractEspnFinishedResults(unittest.TestCase):
    def _event(self, state: str, hs: str | None, as_: str | None) -> dict:
        return {
            "competitions": [
                {
                    "date": "2026-06-09T13:00Z",
                    "status": {"type": {"state": state}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Philippines"}, "score": hs},
                        {"homeAway": "away", "team": {"displayName": "Myanmar"}, "score": as_},
                    ],
                }
            ]
        }

    def test_parses_finished_event(self) -> None:
        results = extract_espn_finished_results([self._event("post", "5", "1")])
        self.assertEqual(len(results), 1)
        self.assertEqual((results[0].home_score, results[0].away_score), (5, 1))
        self.assertEqual(results[0].match_date, date(2026, 6, 9))

    def test_skips_in_progress_and_pre(self) -> None:
        self.assertEqual(extract_espn_finished_results([self._event("in", "1", "0")]), [])
        self.assertEqual(extract_espn_finished_results([self._event("pre", None, None)]), [])


class TestMatchResultToRow(unittest.TestCase):
    def _result(self, d: date) -> FinishedResult:
        return FinishedResult(home_norm="haiti", away_norm="scotland", match_date=d, home_score=1, away_score=0)

    def test_matches_within_date_tolerance(self) -> None:
        # Provider UTC date is one day ahead of the local canonical match_date.
        rows = [CanonicalRow(row_id="X", home_norm="haiti", away_norm="scotland", match_date=date(2026, 6, 13))]
        self.assertEqual(match_result_to_row(self._result(date(2026, 6, 14)), rows, day_tolerance=1), "X")

    def test_rejects_beyond_tolerance(self) -> None:
        rows = [CanonicalRow(row_id="X", home_norm="haiti", away_norm="scotland", match_date=date(2026, 6, 10))]
        self.assertIsNone(match_result_to_row(self._result(date(2026, 6, 14)), rows, day_tolerance=1))

    def test_requires_same_team_pair_and_order(self) -> None:
        rows = [CanonicalRow(row_id="X", home_norm="scotland", away_norm="haiti", match_date=date(2026, 6, 14))]
        self.assertIsNone(match_result_to_row(self._result(date(2026, 6, 14)), rows, day_tolerance=1))

    def test_picks_closest_date_when_multiple(self) -> None:
        rows = [
            CanonicalRow(row_id="far", home_norm="haiti", away_norm="scotland", match_date=date(2026, 6, 13)),
            CanonicalRow(row_id="near", home_norm="haiti", away_norm="scotland", match_date=date(2026, 6, 14)),
        ]
        self.assertEqual(match_result_to_row(self._result(date(2026, 6, 14)), rows, day_tolerance=2), "near")


if __name__ == "__main__":
    unittest.main()
