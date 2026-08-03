from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from alttez.reports.tournament_review.analysis import build_tournament_review
from alttez.reports.tournament_review.automation import (
    _is_ready,
    load_tournament_registry,
    process_tournament_reviews,
)
from alttez.reports.tournament_review.delivery import build_delivery_caption


def _review(*, matches: int = 100, evaluated: int = 95) -> dict:
    return {
        "tournament": "Test Cup",
        "generated_at": "2026-08-03T00:00:00+00:00",
        "coverage": {
            "unique_matches": matches,
            "evaluated_predictions": evaluated,
            "evaluable_rate": evaluated / matches,
        },
        "summary": {
            "accuracy_1x2": 0.61,
            "brier_score": 0.52,
            "rating": {"score": 72.5, "label": "Bueno"},
        },
    }


def _tournament() -> dict:
    return {
        "key": "test-cup",
        "name": "Test Cup",
        "query": "Test Cup",
        "date_from": "2026-06-01",
        "date_to": "2026-06-30",
        "deliver_after": "2026-07-02",
        "minimum_unique_matches": 90,
        "minimum_evaluable_rate": 0.8,
        "enabled": True,
    }


class TournamentReviewDeduplicationTest(unittest.TestCase):
    def test_selects_latest_prediction_before_kickoff_per_match(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "prediction_id": "old",
                    "match_id": "match-1",
                    "prediction_created_at": "2026-06-01T10:00:00Z",
                    "kickoff_at": "2026-06-02T18:00:00Z",
                    "tournament_name": "Test Cup",
                    "model_version": "v1",
                    "predicted_outcome": "A",
                    "prob_home": 0.2,
                    "prob_draw": 0.2,
                    "prob_away": 0.6,
                    "predicted_scoreline": "0 - 1",
                    "actual_home_score": 2,
                    "actual_away_score": 0,
                    "actual_outcome": "H",
                    "stage": "Group stage",
                },
                {
                    "prediction_id": "latest",
                    "match_id": "match-1",
                    "prediction_created_at": "2026-06-02T10:00:00Z",
                    "kickoff_at": "2026-06-02T18:00:00Z",
                    "tournament_name": "Test Cup",
                    "model_version": "v2",
                    "predicted_outcome": "H",
                    "prob_home": 0.7,
                    "prob_draw": 0.2,
                    "prob_away": 0.1,
                    "predicted_scoreline": "2 - 0",
                    "actual_home_score": 2,
                    "actual_away_score": 0,
                    "actual_outcome": "H",
                    "stage": "Group stage",
                },
            ]
        )
        review = build_tournament_review(rows, tournament="Test Cup")
        self.assertEqual(review["coverage"]["persisted_predictions"], 2)
        self.assertEqual(review["coverage"]["unique_matches"], 1)
        self.assertEqual(review["coverage"]["evaluated_predictions"], 1)
        self.assertEqual(review["summary"]["accuracy_1x2"], 1.0)
        self.assertEqual(review["matches"][0]["model_version"], "v2")


class TournamentReviewReadinessTest(unittest.TestCase):
    def test_requires_delivery_date_matches_and_coverage(self) -> None:
        ready, reason = _is_ready(_tournament(), _review(), as_of=date(2026, 7, 2))
        self.assertTrue(ready)
        self.assertIsNone(reason)

        ready, reason = _is_ready(_tournament(), _review(), as_of=date(2026, 7, 1))
        self.assertFalse(ready)
        self.assertEqual(reason, "before_delivery_date")

        ready, reason = _is_ready(
            _tournament(), _review(matches=80, evaluated=79), as_of=date(2026, 7, 2)
        )
        self.assertFalse(ready)
        self.assertEqual(reason, "insufficient_matches")

    def test_registry_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps({"tournaments": [{"key": "broken"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tournament_registry(path)


class TournamentReviewAutomationTest(unittest.TestCase):
    @patch("alttez.reports.tournament_review.automation.load_delivery_record")
    @patch("alttez.reports.tournament_review.automation.load_tournament_registry")
    def test_skips_already_delivered(self, registry_mock, record_mock) -> None:
        registry_mock.return_value = [_tournament()]
        record_mock.return_value = {"delivery_status": "delivered"}
        results = process_tournament_reviews(as_of=date(2026, 7, 3), deliver=True)
        self.assertEqual(results[0]["reason"], "already_delivered")

    @patch("alttez.reports.tournament_review.automation.record_delivery")
    @patch("alttez.reports.tournament_review.automation.deliver_tournament_review")
    @patch("alttez.reports.tournament_review.automation.write_tournament_review")
    @patch("alttez.reports.tournament_review.automation.build_tournament_review")
    @patch("alttez.reports.tournament_review.automation.load_tournament_predictions")
    @patch("alttez.reports.tournament_review.automation.load_delivery_record")
    @patch("alttez.reports.tournament_review.automation.load_tournament_registry")
    def test_delivers_and_records_once(
        self,
        registry_mock,
        record_mock,
        predictions_mock,
        build_mock,
        writer_mock,
        deliver_mock,
        record_delivery_mock,
    ) -> None:
        registry_mock.return_value = [_tournament()]
        record_mock.return_value = None
        predictions_mock.return_value = pd.DataFrame()
        build_mock.return_value = _review()
        writer_mock.return_value = {"html": Path("review.html"), "json": Path("review.json")}
        deliver_mock.return_value = (True, None)

        results = process_tournament_reviews(as_of=date(2026, 7, 3), deliver=True)

        self.assertEqual(results[0]["status"], "delivered")
        deliver_mock.assert_called_once()
        record_delivery_mock.assert_called_once()


class TournamentReviewDeliveryTextTest(unittest.TestCase):
    def test_caption_contains_decision_metrics(self) -> None:
        caption = build_delivery_caption(_review())
        self.assertIn("72.5/100", caption)
        self.assertIn("1X2: 61.0%", caption)
        self.assertIn("Cobertura: 95.0%", caption)


if __name__ == "__main__":
    unittest.main()
