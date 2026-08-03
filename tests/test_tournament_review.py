from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from alttez.reports.tournament_review.analysis import build_tournament_review
from alttez.reports.tournament_review.render_html import render_tournament_review_html
from alttez.reports.tournament_review.repository import load_tournament_predictions
from alttez.reports.tournament_review.writer import write_tournament_review


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_id": "p1",
                "prediction_created_at": "2026-06-10T10:00:00Z",
                "kickoff_at": "2026-06-11T18:00:00Z",
                "match_date": "2026-06-11",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "tournament_name": "FIFA World Cup 2026",
                "model_version": "committee-v1",
                "model_family": "committee",
                "predicted_outcome": "H",
                "prob_home": 0.6,
                "prob_draw": 0.25,
                "prob_away": 0.15,
                "predicted_scoreline": "2 - 1",
                "stage": "Group stage",
                "used_fallback": "false",
                "actual_home_score": 2,
                "actual_away_score": 1,
                "actual_outcome": "H",
            },
            {
                "prediction_id": "p2",
                "prediction_created_at": "2026-07-18T10:00:00Z",
                "kickoff_at": "2026-07-19T19:00:00Z",
                "match_date": "2026-07-19",
                "home_team": "Spain",
                "away_team": "Argentina",
                "tournament_name": "FIFA World Cup 2026",
                "model_version": "committee-v2",
                "model_family": "committee",
                "predicted_outcome": "D",
                "prob_home": 0.35,
                "prob_draw": 0.4,
                "prob_away": 0.25,
                "predicted_scoreline": "1:1",
                "stage": "Final",
                "used_fallback": True,
                "actual_home_score": 1,
                "actual_away_score": 1,
                "actual_outcome": "D",
            },
            {
                "prediction_id": "p3",
                "prediction_created_at": "2026-06-12T20:00:00Z",
                "kickoff_at": "2026-06-12T18:00:00Z",
                "match_date": "2026-06-12",
                "home_team": "Canada",
                "away_team": "Italy",
                "tournament_name": "FIFA World Cup 2026",
                "model_version": "committee-v1",
                "model_family": "committee",
                "predicted_outcome": "H",
                "prob_home": 0.5,
                "prob_draw": 0.3,
                "prob_away": 0.2,
                "predicted_scoreline": "1 - 0",
                "stage": "Group stage",
                "used_fallback": False,
                "actual_home_score": 1,
                "actual_away_score": 0,
                "actual_outcome": "H",
            },
            {
                "prediction_id": "p4",
                "prediction_created_at": "2026-06-14T10:00:00Z",
                "kickoff_at": None,
                "match_date": "2026-06-15",
                "home_team": "Brazil",
                "away_team": "Japan",
                "tournament_name": "FIFA World Cup 2026",
                "model_version": "committee-v1",
                "model_family": "committee",
                "predicted_outcome": "H",
                "prob_home": 60,
                "prob_draw": 25,
                "prob_away": 15,
                "predicted_scoreline": "2 - 0",
                "stage": "Group stage",
                "used_fallback": False,
                "actual_home_score": None,
                "actual_away_score": None,
                "actual_outcome": None,
            },
        ]
    )


class TournamentReviewAnalysisTest(unittest.TestCase):
    def test_builds_review_and_excludes_after_kickoff(self) -> None:
        review = build_tournament_review(
            _prediction_rows(),
            tournament="FIFA World Cup 2026",
            generated_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
        )

        self.assertEqual(review["coverage"]["persisted_predictions"], 4)
        self.assertEqual(review["coverage"]["evaluated_predictions"], 2)
        self.assertEqual(review["coverage"]["valid_pre_match"], 2)
        self.assertEqual(review["coverage"]["invalid_after_kickoff"], 1)
        self.assertEqual(review["coverage"]["unverifiable_timing"], 1)
        self.assertEqual(review["summary"]["accuracy_1x2"], 1.0)
        self.assertEqual(review["summary"]["exact_score_hit_rate"], 1.0)
        self.assertEqual(review["summary"]["pool_points"], 30)
        self.assertEqual(review["summary"]["pool_points_max"], 30)
        self.assertEqual(len(review["matches"]), 2)
        self.assertIsNotNone(review["summary"]["rating"])
        self.assertEqual(len(review["model_breakdown"]), 2)
        self.assertTrue(any("kickoff" in warning for warning in review["warnings"]))

    def test_empty_review_is_explicit(self) -> None:
        review = build_tournament_review(pd.DataFrame(), tournament="Empty Cup")
        self.assertEqual(review["coverage"]["persisted_predictions"], 0)
        self.assertIsNone(review["summary"]["rating"])
        self.assertEqual(review["sample_confidence"], "Baja")
        self.assertTrue(review["warnings"])

    def test_invalid_probabilities_are_not_evaluated(self) -> None:
        frame = _prediction_rows().iloc[[0]].copy()
        frame.loc[frame.index[0], "prob_home"] = -1
        review = build_tournament_review(frame, tournament="FIFA World Cup 2026")
        self.assertEqual(review["coverage"]["evaluated_predictions"], 0)


class TournamentReviewRenderingTest(unittest.TestCase):
    def test_html_contains_rating_coverage_and_matches(self) -> None:
        review = build_tournament_review(_prediction_rows(), tournament="FIFA World Cup 2026")
        rendered = render_tournament_review_html(review)
        self.assertTrue(rendered.startswith("<!DOCTYPE html>"))
        self.assertIn("Valoracion ALTTEZ", rendered)
        self.assertIn("Cobertura evaluable", rendered)
        self.assertIn("Mexico vs South Africa", rendered)
        self.assertIn("Desempeno por modelo", rendered)

    def test_writer_outputs_json_and_html(self) -> None:
        review = build_tournament_review(_prediction_rows(), tournament="FIFA World Cup 2026")
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            paths = write_tournament_review(review, output_dir=tmp)
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["html"].exists())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")


class TournamentReviewRepositoryTest(unittest.TestCase):
    @patch("alttez.reports.tournament_review.repository.read_dataframe")
    def test_query_is_parameterized(self, read_dataframe_mock) -> None:
        read_dataframe_mock.return_value = pd.DataFrame()
        load_tournament_predictions(
            "World Cup",
            date_from="2026-06-11",
            date_to="2026-07-19",
            database_url="postgresql://example",
        )
        sql = read_dataframe_mock.call_args.args[0]
        kwargs = read_dataframe_mock.call_args.kwargs
        self.assertIn(":tournament_pattern", sql)
        self.assertNotIn("%World Cup%", sql)
        self.assertEqual(kwargs["params"]["tournament_pattern"], "%World Cup%")
        self.assertEqual(kwargs["params"]["date_from"], "2026-06-11")
        self.assertEqual(kwargs["database_url"], "postgresql://example")


if __name__ == "__main__":
    unittest.main()
