from __future__ import annotations

import unittest

import optuna
import pandas as pd

from alttez.training.trainer import Trainer
from alttez.training.tuner import Tuner


class DummyTemporalModel:
    def __init__(self, **kwargs) -> None:
        self.fold_assertions = []

    def fit(self, train_df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> pd.DataFrame:
        assert eval_df is not None
        train_max_date = pd.to_datetime(train_df["date"]).max()
        eval_min_date = pd.to_datetime(eval_df["date"]).min()
        if not train_max_date < eval_min_date:
            raise AssertionError("Temporal CV leaked future samples into train.")
        return pd.DataFrame(
            [
                {"Accuracy": 0.55, "data": "eval"},
                {"Accuracy": 0.65, "data": "train"},
            ]
        )


class TestTuner(unittest.TestCase):
    def test_objective_uses_temporal_cross_validation_without_future_leakage(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=24, freq="D"),
                "home_score": [1] * 24,
                "away_score": [0] * 24,
            }
        )
        tuner = Tuner(
            model_cls=DummyTemporalModel,
            fixed_params={},
            tunable_params={"dummy_param": [1]},
            df=df,
            metric="Accuracy",
        )

        score = tuner._objective(optuna.trial.FixedTrial({"dummy_param": 1}))

        self.assertEqual(score, 0.55)


class TestTemporalCrossValidationDateBoundary(unittest.TestCase):
    def test_no_match_day_straddles_train_eval_split(self) -> None:
        # 5 unique dates, 4 matches each (24 rows): an index-based cut would split a day.
        unique_dates = pd.date_range("2025-01-01", periods=6, freq="7D")
        rows = []
        for match_date in unique_dates:
            for _ in range(4):
                rows.append({"date": match_date, "home_score": 1, "away_score": 0})
        df = pd.DataFrame(rows)

        # DummyTemporalModel.fit raises if train_max_date >= eval_min_date.
        metrics = Trainer().temporal_cross_validation(
            model=DummyTemporalModel(),
            df=df,
            test_ratio=0.2,
            k_folds=4,
        )

        self.assertFalse(metrics.empty)
        # Every fold landed a clean date boundary (no AssertionError raised).
        self.assertTrue((metrics["data"] == "eval").any())


if __name__ == "__main__":
    unittest.main()
