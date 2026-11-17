from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from alttez.evaluation import soccer_backtest as sb
from alttez.experiments.tournament_oracle import build_probability_matrix
from alttez.features.elo import EloEngine
from alttez.inference.calibration import matrix_1x2, temperature_scale

TEAMS = ["A", "B", "C", "D", "E", "F"]
STRENGTH = {"A": 1.6, "B": 1.3, "C": 1.0, "D": 0.9, "E": 0.7, "F": 0.5}
LAST_YEAR = 2018


def synthetic_matches(seed: int = 7, first_year: int = 2010, per_year: int = 70) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(first_year, LAST_YEAR + 1):
        n = per_year if year < LAST_YEAR else 6
        days = np.sort(rng.choice(365, size=n, replace=False))
        for day in days:
            home, away = rng.choice(TEAMS, size=2, replace=False)
            rows.append(
                {
                    "date": pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(day)),
                    "home_team": home,
                    "away_team": away,
                    "home_score": int(rng.poisson(1.3 * STRENGTH[home] / STRENGTH[away] ** 0.5)),
                    "away_score": int(rng.poisson(1.1 * STRENGTH[away] / STRENGTH[home] ** 0.5)),
                    "tournament": "FIFA World Cup qualification" if day % 5 == 0 else "Friendly",
                    "neutral": bool(day % 4 == 0),
                }
            )
    frame = pd.DataFrame(rows)
    frame["year"] = frame["date"].dt.year
    frame["outcome"] = np.where(frame["home_score"] > frame["away_score"], "H", np.where(frame["home_score"] < frame["away_score"], "A", "D"))
    return frame


def run(frame: pd.DataFrame, test_years=(2016, 2017)):
    with_elo = sb.add_pre_match_elo(frame)
    return sb.walk_forward(with_elo, list(test_years), window_years=2, refit_days=60)


def with_changed_result(frame: pd.DataFrame, index: int) -> pd.DataFrame:
    changed = frame.copy()
    changed.loc[index, ["home_score", "away_score"]] = [changed.loc[index, "away_score"] + 3, changed.loc[index, "home_score"]]
    changed["outcome"] = np.where(changed["home_score"] > changed["away_score"], "H", np.where(changed["home_score"] < changed["away_score"], "A", "D"))
    return changed


class EloLeakageTests(unittest.TestCase):
    def test_matches_the_engine_when_dates_are_unique(self) -> None:
        frame = synthetic_matches(per_year=8, first_year=2015)
        frame["date"] = pd.date_range("2015-01-01", periods=len(frame), freq="D")
        ours = sb.add_pre_match_elo(frame)
        engine_frame = frame.copy()
        engine_frame["home_result"] = frame["outcome"].map({"H": 1.0, "D": 0.5, "A": 0.0})
        theirs = EloEngine(mov_enabled=False).compute_elo_features(engine_frame)
        np.testing.assert_allclose(ours["elo_home_before"], theirs["elo_home_before"])
        np.testing.assert_allclose(ours["elo_away_before"], theirs["elo_away_before"])

    def test_matches_of_the_same_date_see_start_of_day_ratings(self) -> None:
        day = pd.Timestamp("2020-05-01")
        frame = pd.DataFrame(
            {
                "date": [day, day],
                "home_team": ["A", "A"],
                "away_team": ["B", "C"],
                "home_score": [2, 0],
                "away_score": [0, 1],
                "tournament": ["Friendly"] * 2,
                "neutral": [False, False],
                "outcome": ["H", "A"],
            }
        )
        out = sb.add_pre_match_elo(frame)
        self.assertEqual(out["elo_home_before"].tolist(), [1500.0, 1500.0])

    def test_changing_a_future_result_leaves_earlier_ratings_untouched(self) -> None:
        frame = synthetic_matches()
        cut = len(frame) // 2
        base = sb.add_pre_match_elo(frame)
        changed = sb.add_pre_match_elo(with_changed_result(frame, cut))
        np.testing.assert_array_equal(base["elo_diff"].to_numpy()[: cut + 1], changed["elo_diff"].to_numpy()[: cut + 1])
        self.assertTrue((base["elo_diff"].to_numpy()[cut + 1 :] != changed["elo_diff"].to_numpy()[cut + 1 :]).any())

    def test_requires_date_order(self) -> None:
        frame = synthetic_matches(per_year=3, first_year=2015).iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            sb.add_pre_match_elo(frame)


class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_matches()
        cls.predictions, cls.folds = run(cls.frame)

    def test_probabilities_sum_to_one_for_every_model_on_scored_rows(self) -> None:
        scored = self.predictions[self.predictions["supported"]]
        self.assertGreater(len(scored), 100)
        for model in sb.MODELS:
            probs = sb.model_probabilities(scored, model)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9, err_msg=model)
            self.assertTrue(((probs >= 0) & (probs <= 1)).all(), model)

    def test_folds_are_fitted_on_earlier_years_only(self) -> None:
        for fold in self.folds:
            self.assertLess(fold.train_end, f"{fold.year}-01-01")
            self.assertGreaterEqual(fold.window_start, f"{fold.year - 2}-01-01")
            self.assertTrue(all(y < fold.year for y in fold.calibration_years))
            self.assertLess(fold.temperature, 100)
        self.assertEqual([f.year for f in self.folds], [2016, 2017])

    def test_global_fits_never_receive_test_year_rows(self) -> None:
        seen_rho, seen_temperature = [], []
        real_rho, real_temperature = sb.fit_training_rho, sb.fit_temperature

        def spy_rho(train, **kwargs):
            seen_rho.append(train["date"].max())
            return real_rho(train, **kwargs)

        def spy_temperature(probs, outcomes, **kwargs):
            seen_temperature.append(len(outcomes))
            return real_temperature(probs, outcomes, **kwargs)

        with mock.patch.object(sb, "fit_training_rho", spy_rho), mock.patch.object(sb, "fit_temperature", spy_temperature):
            predictions, _ = run(self.frame)
        year_starts = [pd.Timestamp(year=y, month=1, day=1) for y in range(2014, 2018)]
        self.assertEqual(len(seen_rho), 4)
        for latest, start in zip(seen_rho, year_starts, strict=True):
            self.assertLess(latest, start)
        with_elo = sb.add_pre_match_elo(self.frame)
        supported = {y: int(sb.raw_poisson_year(with_elo[with_elo["year"] <= y], y, refit_days=60).frame["supported"].sum()) for y in range(2014, 2017)}
        self.assertEqual(seen_temperature, [supported[2014] + supported[2015], supported[2015] + supported[2016]])
        self.assertEqual(len(predictions), len(self.predictions))

    def test_changing_a_future_result_leaves_every_earlier_prediction_untouched(self) -> None:
        index = int(self.frame.index[(self.frame["year"] == 2017) & (self.frame["date"] > "2017-06-01")][0])
        cut_date = self.frame.loc[index, "date"]
        changed_predictions, _ = run(with_changed_result(self.frame, index))
        columns = [f"{m}_{c}" for m in sb.MODELS for c in sb.CLASSES] + ["lambda_home", "lambda_away", "supported"]
        before = self.predictions[self.predictions["date"] <= cut_date]
        after = changed_predictions[changed_predictions["date"] <= cut_date]
        self.assertGreater(len(before), 100)
        np.testing.assert_allclose(before[columns].to_numpy(float), after[columns].to_numpy(float), equal_nan=True, rtol=0, atol=0)
        later_before = self.predictions[self.predictions["date"] > cut_date]
        later_after = changed_predictions[changed_predictions["date"] > cut_date]
        self.assertFalse(np.allclose(later_before[columns].to_numpy(float), later_after[columns].to_numpy(float), equal_nan=True, rtol=0, atol=0))

    def test_changing_a_test_year_result_does_not_move_earlier_years_or_their_fit(self) -> None:
        index = int(self.frame.index[(self.frame["year"] == 2017)][0])
        changed_predictions, changed_folds = run(with_changed_result(self.frame, index))
        columns = [f"{m}_{c}" for m in sb.MODELS for c in sb.CLASSES]
        year_2016 = self.predictions["year"] == 2016
        np.testing.assert_allclose(
            self.predictions.loc[year_2016, columns].to_numpy(float),
            changed_predictions.loc[changed_predictions["year"] == 2016, columns].to_numpy(float),
            equal_nan=True,
            rtol=0,
            atol=0,
        )
        self.assertEqual(self.folds[1].temperature, changed_folds[1].temperature)
        self.assertEqual(self.folds[1].rho, changed_folds[1].rho)
        self.assertEqual(self.folds[1].base_rate, changed_folds[1].base_rate)


class MetricTests(unittest.TestCase):
    PROBS = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6], [0.1, 0.3, 0.6]])
    OUTCOMES = ["H", "A", "D"]

    def test_scores_match_hand_computation(self) -> None:
        losses = sb.per_game_losses(self.PROBS, self.OUTCOMES)
        np.testing.assert_allclose(losses["brier"], [0.38, 0.24, 0.01 + 0.49 + 0.36])
        np.testing.assert_allclose(losses["rps"], [(0.25 + 0.04) / 2, (0.04 + 0.16) / 2, (0.01 + 0.36) / 2])
        np.testing.assert_allclose(losses["log_loss"], [-np.log(0.5), -np.log(0.6), -np.log(0.3)])
        metrics = sb.point_metrics(self.PROBS, self.OUTCOMES)
        self.assertAlmostEqual(metrics["brier"], (0.38 + 0.24 + 0.86) / 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["n"], 3)

    def test_log_loss_is_finite_for_a_zero_probability(self) -> None:
        loss = sb.log_loss_1x2_per_game(np.array([[1.0, 0.0, 0.0]]), ["A"])
        self.assertTrue(np.isfinite(loss).all())

    def test_class_calibration_matches_hand_computation(self) -> None:
        probs = np.array([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
        outcomes = ["H", "D", "H", "H"]
        home = sb.class_calibration(probs, outcomes)["H"]
        # bin 2: two games predicted 0.2, one home win -> |0.2 - 0.5|; bin 8: two at 0.8, both wins -> |0.8 - 1|
        self.assertAlmostEqual(home["ece"], (2 * 0.3 + 2 * 0.2) / 4)
        self.assertAlmostEqual(home["frequency"], 0.75)
        self.assertAlmostEqual(home["mean_predicted"], 0.5)


class CalibrationTransformTests(unittest.TestCase):
    def test_matrix_calibration_keeps_distributions_and_hits_the_scaled_triplet(self) -> None:
        matrices = np.stack([build_probability_matrix(1.9, 0.8, max_goals=7, rho=-0.04), build_probability_matrix(0.6, 1.4, max_goals=7, rho=-0.04)])
        supported = np.array([True, True])
        out = sb.calibrate_matrices(matrices, supported, 1.7)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
        for row, matrix in zip(out, matrices, strict=True):
            np.testing.assert_allclose(row, temperature_scale(np.array(matrix_1x2(matrix)), 1.7), atol=1e-12)

    def test_temperature_one_is_the_identity_and_unsupported_rows_stay_nan(self) -> None:
        matrix = build_probability_matrix(1.4, 1.1, max_goals=7, rho=-0.04)
        matrices = np.stack([matrix, np.full_like(matrix, np.nan)])
        out = sb.calibrate_matrices(matrices, np.array([True, False]), 1.0)
        np.testing.assert_allclose(out[0], matrix_1x2(matrix), atol=1e-12)
        self.assertTrue(np.isnan(out[1]).all())


class SupportGateTests(unittest.TestCase):
    def test_both_teams_need_history_and_recent_support(self) -> None:
        ok = {"matches": 9.0, "effective_matches": 3.0}
        self.assertTrue(sb.is_supported(ok, ok))
        self.assertFalse(sb.is_supported(ok, {"matches": 4.0, "effective_matches": 3.0}))
        self.assertFalse(sb.is_supported(ok, {"matches": 9.0, "effective_matches": 0.0}))


class DataTests(unittest.TestCase):
    CSV = (
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2020-03-02,B,A,0,2,Friendly,X,Y,FALSE\n"
        "2020-03-01,A,B,1,1,Friendly,X,Y,TRUE\n"
        "2020-03-09,A,B,,,Friendly,X,Y,FALSE\n"
    )

    def test_prepare_matches_orders_by_date_and_reports_unscored_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text(self.CSV, encoding="utf-8")
            matches = sb.prepare_matches(path)
        self.assertEqual((matches.n_rows, matches.n_without_score), (3, 1))
        self.assertEqual(matches.frame["outcome"].tolist(), ["D", "A"])
        self.assertTrue(matches.frame["date"].is_monotonic_increasing)
        self.assertEqual(matches.frame["neutral"].tolist(), [True, False])

    def test_download_is_cached_validated_and_never_falls_back(self) -> None:
        calls: list[str] = []

        def fetch(url: str, timeout: float) -> bytes:
            calls.append(url)
            return self.CSV.encode()

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "nested"
            path = sb.load_results_csv(cache, fetch=fetch)
            self.assertEqual(path.read_text(encoding="utf-8"), self.CSV)
            sb.load_results_csv(cache, fetch=fetch)
            self.assertEqual(len(calls), 1)
            sb.load_results_csv(cache, fetch=fetch, refresh=True)
            self.assertEqual(len(calls), 2)

            with self.assertRaises(sb.SoccerDataError):
                sb.load_results_csv(cache, fetch=lambda url, timeout: b"<html>not csv</html>", refresh=True)
            self.assertEqual(path.read_text(encoding="utf-8"), self.CSV)
            self.assertEqual(list(cache.glob("*.tmp")), [])

    def test_a_failed_download_without_cache_raises(self) -> None:
        def fetch(url: str, timeout: float) -> bytes:
            raise sb.SoccerDataError("offline")

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(sb.SoccerDataError):
            sb.load_results_csv(Path(tmp), fetch=fetch)
            self.fail("unreachable")


class EndToEndTests(unittest.TestCase):
    def test_report_is_complete_and_serialisable(self) -> None:
        frame = synthetic_matches()
        matches = sb.Matches(frame=frame, n_rows=len(frame), n_without_score=0)
        report = sb.run_backtest(matches, first_test_year=2016, window_years=2, refit_days=60, n_boot=50)
        self.assertEqual((report["config"]["first_test_year"], report["config"]["last_test_year"]), (2016, 2017))
        self.assertEqual(set(report["pooled"]["metrics"]), set(sb.MODELS))
        pooled = {(r["model"], r["baseline"], r["metric"]) for r in report["pooled"]["comparisons"]}
        self.assertEqual(len(pooled), len(sb.reported_comparisons(sb.CANDIDATE_NAMES)) * len(sb.METRICS))
        self.assertEqual(set(report["by_year"]), {"2016", "2017"})
        json.dumps(report)
        self.assertIn("Poisson + temperature", sb.render_markdown(report))


if __name__ == "__main__":
    unittest.main()
