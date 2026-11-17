from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from alttez.operations.expansion_policy import (
    ExpansionThresholds,
    aggregate_prediction_memory,
    classify_competition_readiness,
    load_expansion_candidates,
)


class TestExpansionPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = ExpansionThresholds(
            min_matches=60,
            min_teams=10,
            min_evaluated_predictions=20,
            min_analyzable_rate=0.55,
            max_brier_score=0.66,
        )

    def test_classify_competition_readiness_approves_strong_candidate(self) -> None:
        readiness, reason = classify_competition_readiness(
            {
                "competition_class": "continental",
                "matches_total": 120,
                "teams_total": 18,
                "evaluated_predictions": 45,
                "analyzable_rate": 0.72,
                "brier_score": 0.58,
            },
            self.thresholds,
        )
        self.assertEqual(readiness, "approved")
        self.assertIn("apta", reason.lower())

    def test_classify_competition_readiness_blocks_low_history(self) -> None:
        readiness, _ = classify_competition_readiness(
            {
                "competition_class": "friendly",
                "matches_total": 24,
                "teams_total": 8,
                "evaluated_predictions": 0,
                "analyzable_rate": 0.2,
                "brier_score": None,
            },
            self.thresholds,
        )
        self.assertEqual(readiness, "blocked")


def _prediction(match_id, created, *, brier=None, label="Analizable", competition="continental", kickoff="2026-06-01T18:00:00+00:00"):
    return {
        "match_id": match_id,
        "model_version": "committee-consensus-v1",
        "prediction_created_at": created,
        "kickoff_at": kickoff,
        "tournament_name": "Copa",
        "competition_class": competition,
        "support_label": label,
        "brier_1x2": brier,
        "log_loss_1x2": None if brier is None else brier + 0.4,
        "hit_1x2": None if brier is None else brier < 0.5,
    }


class TestExpansionPredictionCounts(unittest.TestCase):
    def test_daily_reruns_of_a_match_count_once(self) -> None:
        predictions = pd.DataFrame(
            [
                _prediction("m1", "2026-05-29T08:00:00+00:00", brier=0.9),
                _prediction("m1", "2026-05-30T08:00:00+00:00", brier=0.3),
                _prediction("m1", "2026-05-31T08:00:00+00:00", brier=0.2),
                _prediction("m1", "2026-06-02T08:00:00+00:00", brier=0.0),
                _prediction("m2", "2026-05-31T08:00:00+00:00", brier=0.6, label="Soporte parcial"),
                _prediction("m3", "2026-05-31T08:00:00+00:00", competition=None),
            ]
        )

        aggregated = aggregate_prediction_memory(predictions).set_index("competition_class")

        continental = aggregated.loc["continental"]
        self.assertEqual(continental["predicted_matches"], 2)
        self.assertAlmostEqual(continental["analyzable_rate"], 0.5)
        self.assertEqual(continental["evaluated_predictions"], 2)
        self.assertAlmostEqual(continental["brier_score"], 0.4)
        self.assertAlmostEqual(continental["accuracy_1x2"], 0.5)
        self.assertEqual(aggregated.loc["other_international", "predicted_matches"], 1)

    def test_load_expansion_candidates_merges_deduplicated_counts(self) -> None:
        matches = pd.DataFrame(
            [{"tournament_name": "Copa", "competition_class": "continental", "matches_total": 120, "teams_total": 18}]
        )
        predictions = pd.DataFrame(
            [_prediction("m1", "2026-05-30T08:00:00+00:00", brier=0.3), _prediction("m1", "2026-05-31T08:00:00+00:00", brier=0.2)]
        )
        thresholds = ExpansionThresholds(min_evaluated_predictions=1, min_analyzable_rate=0.5)
        with patch("alttez.operations.expansion_policy.read_dataframe", side_effect=[matches, predictions]):
            candidates = load_expansion_candidates(thresholds=thresholds)

        self.assertEqual(candidates.iloc[0]["predicted_matches"], 1)
        self.assertEqual(candidates.iloc[0]["evaluated_predictions"], 1)
        self.assertAlmostEqual(candidates.iloc[0]["brier_score"], 0.2)


if __name__ == "__main__":
    unittest.main()
