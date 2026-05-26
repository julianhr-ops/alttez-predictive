from __future__ import annotations

import unittest
from datetime import date

from alttez.ingestion.certification import build_daily_certification_report, build_raw_match_payload
from alttez.ingestion.live_matches import RawMatchRecord, ScheduledMatch


def build_statistics_payload(home_team: str, away_team: str) -> dict[str, object]:
    return {
        "home_statistics": {
            "xg": 1.4,
            "possession": 57,
            "shots_total": 14,
            "shots_on_target": 6,
            "shots_off_target": 5,
            "shots_blocked": 3,
            "shots_inside_box": 9,
            "shots_outside_box": 5,
            "corners": 7,
            "fouls": 11,
            "yellow_cards": 2,
            "red_cards": 0,
            "offsides": 1,
            "saves": 4,
            "passes_completed": 411,
            "pass_accuracy": 84.2,
        },
        "away_statistics": {
            "xg": 0.6,
            "possession": 43,
            "shots_total": 9,
            "shots_on_target": 3,
            "shots_off_target": 4,
            "shots_blocked": 2,
            "shots_inside_box": 4,
            "shots_outside_box": 5,
            "corners": 3,
            "fouls": 14,
            "yellow_cards": 3,
            "red_cards": 0,
            "offsides": 2,
            "saves": 2,
            "passes_completed": 287,
            "pass_accuracy": 77.5,
        },
        "teams": {
            "home": {"name": home_team},
            "away": {"name": away_team},
        },
    }


def build_match(status: str = "finished") -> ScheduledMatch:
    return ScheduledMatch(
        provider="api_football",
        external_match_id="9001",
        kickoff_at="2026-05-26T18:00:00+00:00",
        match_date=date(2026, 5, 26),
        season_label="2026",
        stage="Group Stage",
        group_label="Group A",
        tournament_name="Friendly Cup",
        competition_normalized_name="friendly cup",
        competition_class="friendly",
        tournament_region="Europe",
        home_team="Nigeria",
        away_team="India",
        venue_name="Test Stadium",
        city="Lagos",
        venue_country_name="Nigeria",
        referee_name="Ref Test",
        is_neutral=True,
        status=status,
        home_score=2 if status == "finished" else 0,
        away_score=1 if status == "finished" else 0,
        payload=build_statistics_payload("Nigeria", "India"),
    )


class TestIngestionCertification(unittest.TestCase):
    def test_build_raw_match_payload_extracts_statistics(self) -> None:
        match = build_match()
        record = RawMatchRecord(provider="api_football", external_match_id="9001", payload=match.payload)
        raw_payload = build_raw_match_payload(match, record)

        self.assertEqual(raw_payload.status, "finished")
        self.assertIsNotNone(raw_payload.home_metrics)
        self.assertIsNotNone(raw_payload.away_metrics)
        self.assertGreaterEqual(raw_payload.available_metric_count(), 30)
        self.assertGreater(raw_payload.coverage_pct(), 90.0)

    def test_build_daily_certification_report_certifies_finished_match(self) -> None:
        match = build_match()
        record = RawMatchRecord(provider="api_football", external_match_id="9001", payload=match.payload)
        raw_payload = build_raw_match_payload(match, record)

        report = build_daily_certification_report(
            [raw_payload],
            batch_id="2026-05-26_api_football",
            target_date=date(2026, 5, 26),
            provider_used="api_football",
        )

        self.assertTrue(report.certified)
        self.assertEqual(report.total_records, 1)
        self.assertEqual(report.certified_records, 1)
        self.assertEqual(report.rejected_records, 0)
        self.assertEqual(report.certification_status, "certified")


if __name__ == "__main__":
    unittest.main()

