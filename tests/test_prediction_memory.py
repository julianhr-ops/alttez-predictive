from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sqlalchemy import Column, Float, MetaData, String, Table

from alttez.operations.prediction_memory import (
    RAW_PROBABILITY_COLUMNS,
    _filter_match_prediction_row,
    _restrict_to_table_columns,
    activate_model_version_by_key,
    brier_score_1x2,
    load_closed_evaluation_samples,
    load_prediction_learning_audit,
    load_prediction_metrics_by_competition,
    load_prediction_metrics_by_confidence,
    load_prediction_metrics_by_model,
    load_prediction_overview,
    log_loss_1x2,
    select_latest_pre_kickoff,
)


class TestPredictionMemory(unittest.TestCase):
    def test_brier_score_rewards_calibrated_prediction(self) -> None:
        confident_correct = brier_score_1x2(0.75, 0.15, 0.10, "H")
        flat_prediction = brier_score_1x2(0.34, 0.33, 0.33, "H")

        self.assertLess(confident_correct, flat_prediction)

    def test_log_loss_penalizes_wrong_overconfidence(self) -> None:
        calibrated = log_loss_1x2(0.60, 0.25, 0.15, "H")
        wrong_overconfident = log_loss_1x2(0.05, 0.10, 0.85, "H")

        self.assertLess(calibrated, wrong_overconfident)

    def test_learning_audit_summarizes_prediction_evaluations(self) -> None:
        df = pd.DataFrame(
            [
                _eval_row("m1", "2026-05-30T08:00:00+00:00", model_version="xgb-v1", brier_1x2=0.20, log_loss_1x2=0.5, hit_1x2=True, hit_exact_score=False),
                _eval_row("m2", "2026-05-30T08:00:00+00:00", model_version="xgb-v1", brier_1x2=0.40, log_loss_1x2=0.9, hit_1x2=False, hit_exact_score=True),
            ]
        )
        with patch("alttez.operations.prediction_memory.read_dataframe", return_value=df):
            audit = load_prediction_learning_audit()

        self.assertEqual(audit["evaluated_predictions"], 2)
        self.assertAlmostEqual(audit["brier_score"], 0.30)
        self.assertAlmostEqual(audit["accuracy_1x2"], 0.50)
        self.assertAlmostEqual(audit["exact_score_hit_rate"], 0.50)
        self.assertEqual(audit["models"][0]["model_version"], "xgb-v1")

    def test_activation_rejects_missing_non_poisson_artifact(self) -> None:
        record = {
            "model_family": "xgboost",
            "artifact_path": "artifacts/model-artifacts/missing-model.joblib",
        }
        with (
            patch("alttez.operations.prediction_memory.get_engine"),
            patch("alttez.operations.prediction_memory.reflect_prediction_tables", return_value={"model_versions": object()}),
            patch("alttez.operations.prediction_memory.load_model_version_record", return_value=record),
        ):
            result = activate_model_version_by_key("xgb-missing")

        self.assertFalse(result["activated"])
        self.assertIn("artifact", result["error"].lower())


def _eval_row(match_id, created, kickoff="2026-06-01T18:00:00+00:00", model_version="v1", **metrics) -> dict:
    row = {
        "match_id": match_id,
        "model_version": model_version,
        "prediction_created_at": created,
        "kickoff_at": kickoff,
        "competition_class": "friendly",
        "avg_confidence": 0.7,
        "brier_1x2": 0.2,
        "log_loss_1x2": 0.5,
        "rps_1x2": 0.1,
        "hit_1x2": True,
        "hit_exact_score": False,
    }
    row.update(metrics)
    return row


class TestSelectLatestPreKickoff(unittest.TestCase):
    def test_keeps_one_row_per_match_and_model_version(self) -> None:
        frame = pd.DataFrame(
            [
                _eval_row("m1", "2026-05-30T08:00:00+00:00", brier_1x2=0.9),
                _eval_row("m1", "2026-05-31T08:00:00+00:00", brier_1x2=0.3),
                _eval_row("m1", "2026-06-01T08:00:00+00:00", brier_1x2=0.1),
                _eval_row("m2", "2026-05-31T08:00:00+00:00"),
                _eval_row("m1", "2026-05-31T08:00:00+00:00", model_version="v2"),
            ]
        )

        result = select_latest_pre_kickoff(frame)

        self.assertEqual(len(result), 3)
        self.assertFalse(result.duplicated(subset=["match_id", "model_version"]).any())
        kept = result[(result["match_id"] == "m1") & (result["model_version"] == "v1")].iloc[0]
        self.assertEqual(kept["brier_1x2"], 0.1)

    def test_rows_created_at_or_after_kickoff_are_dropped(self) -> None:
        frame = pd.DataFrame(
            [
                _eval_row("m1", "2026-06-01T08:00:00+00:00", brier_1x2=0.3),
                _eval_row("m1", "2026-06-01T18:00:00+00:00", brier_1x2=0.0),
                _eval_row("m1", "2026-06-01T20:00:00+00:00", brier_1x2=0.0),
                _eval_row("m2", "2026-06-01T19:00:00+00:00"),
            ]
        )

        result = select_latest_pre_kickoff(frame)

        self.assertEqual(result["match_id"].tolist(), ["m1"])
        self.assertEqual(result.iloc[0]["brier_1x2"], 0.3)

    def test_unknown_kickoff_or_creation_time_is_kept(self) -> None:
        frame = pd.DataFrame(
            [
                _eval_row("m1", "2026-06-02T08:00:00+00:00", kickoff=None),
                _eval_row("m2", None),
            ]
        )

        self.assertEqual(sorted(select_latest_pre_kickoff(frame)["match_id"]), ["m1", "m2"])

    def test_mixed_timezone_representations_compare_in_utc(self) -> None:
        frame = pd.DataFrame(
            [
                _eval_row("m1", "2026-06-01T10:00:00-05:00", kickoff="2026-06-01T18:00:00+00:00", brier_1x2=0.0),
                _eval_row("m1", "2026-06-01T09:00:00+00:00", kickoff="2026-06-01T18:00:00+00:00", brier_1x2=0.3),
            ]
        )

        result = select_latest_pre_kickoff(frame)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["brier_1x2"], 0.0)

    def test_empty_frame_is_returned_and_missing_columns_are_rejected(self) -> None:
        empty = pd.DataFrame(columns=["match_id", "model_version", "prediction_created_at", "kickoff_at"])
        self.assertTrue(select_latest_pre_kickoff(empty).empty)
        with self.assertRaises(ValueError):
            select_latest_pre_kickoff(pd.DataFrame([{"model_version": "v1"}]))


class TestDeduplicatedLoaders(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _eval_row("m1", "2026-05-29T08:00:00+00:00", brier_1x2=0.9, hit_1x2=False),
                _eval_row("m1", "2026-05-30T08:00:00+00:00", brier_1x2=0.3, hit_1x2=True),
                _eval_row("m1", "2026-06-02T08:00:00+00:00", brier_1x2=0.0, hit_1x2=True),
                _eval_row("m2", "2026-05-30T08:00:00+00:00", brier_1x2=0.5, hit_1x2=False, competition_class=None),
            ]
        )

    def test_overview_and_audit_count_one_prediction_per_match(self) -> None:
        with patch("alttez.operations.prediction_memory.read_dataframe", return_value=self._frame()):
            overview = load_prediction_overview()
            audit = load_prediction_learning_audit()

        for summary in (overview, audit):
            self.assertEqual(summary["evaluated_predictions"], 2)
            self.assertAlmostEqual(summary["brier_score"], 0.4)
            self.assertAlmostEqual(summary["accuracy_1x2"], 0.5)
        self.assertEqual(audit["models"][0]["evaluated_predictions"], 2)

    def test_metrics_by_dimension_are_deduplicated(self) -> None:
        with patch("alttez.operations.prediction_memory.read_dataframe", return_value=self._frame()):
            by_model = load_prediction_metrics_by_model()
            by_competition = load_prediction_metrics_by_competition()
            by_confidence = load_prediction_metrics_by_confidence()

        self.assertEqual(by_model["evaluated_predictions"].tolist(), [2])
        self.assertAlmostEqual(by_model.iloc[0]["brier_score"], 0.4)
        self.assertEqual(
            by_competition.set_index("competition_class")["evaluated_predictions"].to_dict(),
            {"friendly": 1, "other_international": 1},
        )
        self.assertEqual(by_confidence["evaluated_predictions"].sum(), 2)
        self.assertEqual(by_confidence.iloc[0]["confidence_band"], "0.65-0.79")

    def test_closed_samples_are_deduplicated(self) -> None:
        with patch("alttez.operations.prediction_memory.read_dataframe", return_value=self._frame()):
            samples = load_closed_evaluation_samples()

        self.assertEqual(len(samples), 2)


def _match_payload(**overrides) -> dict:
    payload = {
        "match_id": "m1",
        "target_date": "2026-06-01",
        "kickoff_at": "2026-06-01T18:00:00+00:00",
        "generated_at": "2026-06-01T08:00:00+00:00",
        "home_team": "A",
        "away_team": "B",
        "prob_home": 0.50,
        "prob_draw": 0.30,
        "prob_away": 0.20,
    }
    payload.update(overrides)
    return payload


class TestRawProbabilityPersistence(unittest.TestCase):
    def test_row_carries_raw_probabilities_next_to_calibrated(self) -> None:
        row = _filter_match_prediction_row(
            _match_payload(raw_prob_home=0.70, raw_prob_draw=0.20, raw_prob_away=0.10), 1, "run", "v", "f"
        )

        self.assertEqual(row["predicted_home_win_prob"], 0.50)
        self.assertEqual(row["raw_predicted_home_win_prob"], 0.70)
        self.assertEqual(row["raw_predicted_draw_prob"], 0.20)
        self.assertEqual(row["raw_predicted_away_win_prob"], 0.10)

    def test_row_without_raw_probabilities_stores_null(self) -> None:
        row = _filter_match_prediction_row(_match_payload(), 1, "run", "v", "f")

        for column in RAW_PROBABILITY_COLUMNS:
            self.assertIsNone(row[column])

    def test_unknown_columns_are_dropped_with_a_warning_until_migration_014_runs(self) -> None:
        table = Table("match_predictions", MetaData(), Column("match_id", String), Column("predicted_home_win_prob", Float))
        rows = [{"match_id": "m1", "predicted_home_win_prob": 0.5, "raw_predicted_home_win_prob": 0.7}]

        with self.assertLogs("alttez.operations.prediction_memory", level="WARNING") as logs:
            restricted = _restrict_to_table_columns(rows, table)

        self.assertEqual(restricted, [{"match_id": "m1", "predicted_home_win_prob": 0.5}])
        self.assertIn("raw_predicted_home_win_prob", logs.output[0])

    def test_known_columns_pass_through_untouched(self) -> None:
        table = Table("t", MetaData(), Column("match_id", String), Column("raw_predicted_home_win_prob", Float))
        rows = [{"match_id": "m1", "raw_predicted_home_win_prob": 0.7}]

        self.assertIs(_restrict_to_table_columns(rows, table), rows)

    def test_migration_014_adds_nullable_columns_idempotently(self) -> None:
        sql = Path("sql/supabase/migrations/014_prediction_raw_probabilities.sql").read_text(encoding="utf-8").lower()

        for column in RAW_PROBABILITY_COLUMNS:
            self.assertRegex(sql, rf"add column if not exists {column} numeric\(10,6\)[,;]")
        self.assertNotIn("not null", sql)


if __name__ == "__main__":
    unittest.main()
