from __future__ import annotations

import unittest
from datetime import date

from alttez.ingestion.contracts import (
    BatchCertification,
    CertifiedMatch,
    MatchTeamMetrics,
    RawMatchPayload,
)


def build_metric_payload(offset: int = 0) -> dict[str, object]:
    return {
        "xg": f"{1.1 + offset}",
        "possession": "58%",
        "shots_total": "14",
        "shots_on_target": "6",
        "shots_off_target": "5",
        "shots_blocked": "3",
        "shots_inside_box": "8",
        "shots_outside_box": "6",
        "corners": "7",
        "fouls": "11",
        "yellow_cards": "2",
        "red_cards": "0",
        "offsides": "1",
        "saves": "4",
        "passes_completed": "412",
        "pass_accuracy": "84.2%",
    }


def build_raw_payload(
    *,
    external_match_id: str,
    status: str = "finished",
    home_score: int | None = 2,
    away_score: int | None = 1,
    home_metrics: dict[str, object] | None = None,
    away_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "provider": "api_football",
        "external_match_id": external_match_id,
        "match_date": "2026-05-26",
        "kickoff_at": "2026-05-26T18:00:00+00:00",
        "tournament_name": "Amistoso Internacional",
        "home_team": "Nigeria",
        "away_team": "India",
        "status": status,
        "home_score": home_score,
        "away_score": away_score,
        "season_label": "2026",
        "stage": "Group Stage",
        "competition_class": "friendly",
        "is_neutral": True,
        "home_metrics": home_metrics or build_metric_payload(),
        "away_metrics": away_metrics or build_metric_payload(1),
        "payload": {"fixture": {"id": external_match_id}},
    }


class TestIngestionContracts(unittest.TestCase):
    def test_match_team_metrics_coerces_strings(self) -> None:
        metrics = MatchTeamMetrics.model_validate(build_metric_payload())

        self.assertEqual(metrics.shots_total, 14)
        self.assertEqual(metrics.possession, 58.0)
        self.assertEqual(metrics.pass_accuracy, 84.2)
        self.assertEqual(metrics.available_metric_count(), 16)

    def test_raw_payload_tracks_coverage(self) -> None:
        raw = RawMatchPayload.model_validate(build_raw_payload(external_match_id="1001"))

        self.assertTrue(raw.hard_required_complete())
        self.assertEqual(raw.available_metric_count(), 32)
        self.assertEqual(raw.total_metric_slots(), 32)
        self.assertEqual(raw.coverage_pct(), 100.0)
        self.assertTrue(raw.certification_ready())

    def test_certified_match_requires_finished_status_and_metrics(self) -> None:
        raw = RawMatchPayload.model_validate(build_raw_payload(external_match_id="1002"))
        certified = CertifiedMatch.from_raw(raw, batch_id="batch-20260526")

        self.assertEqual(certified.certified, True)
        self.assertEqual(certified.status, "finished")
        self.assertEqual(certified.coverage_pct, 100.0)
        self.assertEqual(certified.available_metric_count, 32)

    def test_batch_certification_quarantines_incomplete_records(self) -> None:
        certified_raw = build_raw_payload(external_match_id="1003")
        rejected_raw = build_raw_payload(
            external_match_id="1004",
            status="scheduled",
            home_score=None,
            away_score=None,
            home_metrics={"xg": "0.6", "shots_total": "4"},
            away_metrics={"xg": "0.4"},
        )

        batch = BatchCertification.certify_batch(
            [certified_raw, rejected_raw],
            batch_id="batch-20260526",
            target_date=date(2026, 5, 26),
            provider_used="api_football",
        )

        self.assertEqual(batch.total_records, 2)
        self.assertEqual(batch.certified_records, 1)
        self.assertEqual(batch.rejected_records, 1)
        self.assertFalse(batch.certified)
        self.assertTrue(batch.is_quarantined())
        self.assertIsNotNone(batch.quarantine_reason)
        self.assertEqual(len(batch.certified_matches), 1)
        self.assertEqual(len(batch.rejected_matches), 1)


if __name__ == "__main__":
    unittest.main()

