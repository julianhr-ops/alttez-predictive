from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from alttez.evaluation import soccer_backtest as sb
from alttez.evaluation import soccer_candidates as sc
from alttez.experiments.tournament_oracle import build_probability_matrix
from alttez.inference.calibration import matrix_1x2
from tests.test_soccer_backtest import run, synthetic_matches


def training_frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    diff = rng.normal(0.0, 180.0, n)
    p_home = 1 / (1 + 10 ** (-(diff + 60.0) / 400))
    draw = rng.random(n) < 0.25
    home = rng.random(n) < p_home
    outcome = np.where(draw, "D", np.where(home, "H", "A"))
    frame = pd.DataFrame({"elo_diff": diff, "neutral": rng.random(n) < 0.3, "outcome": outcome})
    poisson = rng.dirichlet([3.0, 2.0, 2.5], size=n)
    for column, cls in enumerate(sc.CLASSES):
        frame[f"poisson_raw_{cls}"] = poisson[:, column]
    return frame


class FitCandidateTests(unittest.TestCase):
    def test_every_candidate_gives_normalised_probabilities(self) -> None:
        train = training_frame()
        poisson = train[[f"poisson_raw_{c}" for c in sc.CLASSES]].to_numpy()
        for spec in sc.CANDIDATES:
            fitted = sc.fit_candidate(spec, train, train)
            probs = fitted.predict(train["elo_diff"].to_numpy(), train["neutral"].to_numpy(), poisson)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12, err_msg=spec.name)
            self.assertEqual(fitted.blend_weight is not None, spec.blend, spec.name)

    def test_the_fit_depends_only_on_the_rows_it_receives(self) -> None:
        train = training_frame()
        spec = next(s for s in sc.CANDIDATES if s.blend and not s.fit_scale and s.kind == "ordinal")
        first = sc.fit_candidate(spec, train, train)
        later = pd.concat([train, training_frame(seed=9, n=200).assign(outcome="A")], ignore_index=True)
        self.assertEqual(first, sc.fit_candidate(spec, train, train))
        self.assertNotEqual(first.draw_model, sc.fit_candidate(spec, later, train).draw_model)

    def test_a_blend_needs_poisson_rows_to_fit_and_to_predict(self) -> None:
        train = training_frame()
        blend = next(s for s in sc.CANDIDATES if s.blend)
        with self.assertRaises(ValueError):
            sc.fit_candidate(blend, train, None)
        with self.assertRaises(ValueError):
            sc.fit_candidate(blend, train, train.iloc[:0])
        fitted = sc.fit_candidate(blend, train, train)
        with self.assertRaises(ValueError):
            fitted.predict(train["elo_diff"].to_numpy(), train["neutral"].to_numpy())

    def test_a_blend_is_nan_where_the_poisson_1x2_is_missing(self) -> None:
        train = training_frame()
        blend = next(s for s in sc.CANDIDATES if s.blend)
        fitted = sc.fit_candidate(blend, train, train)
        poisson = train[[f"poisson_raw_{c}" for c in sc.CLASSES]].to_numpy().copy()
        poisson[0] = np.nan
        probs = fitted.predict(train["elo_diff"].to_numpy(), train["neutral"].to_numpy(), poisson)
        self.assertTrue(np.isnan(probs[0]).all())
        self.assertTrue(np.isfinite(probs[1:]).all())

    def test_candidate_names_are_unique_and_registered(self) -> None:
        self.assertEqual(len(set(sc.CANDIDATE_NAMES)), len(sc.CANDIDATES))
        self.assertTrue(set(sc.CANDIDATE_NAMES) <= set(sb.MODELS))


class ServedScorelineTests(unittest.TestCase):
    def test_served_matrix_marginals_equal_the_target_and_scorelines_stay_scored(self) -> None:
        matrices = np.stack([build_probability_matrix(1.9, 0.8, max_goals=7, rho=-0.04), build_probability_matrix(0.6, 1.4, max_goals=7, rho=-0.04)])
        targets = np.array([[0.30, 0.30, 0.40], [0.55, 0.25, 0.20]])
        served = sc.served_scoreline(matrices, np.array([True, True]), targets, np.array([2, 0]), np.array([1, 1]))
        self.assertLess(np.nanmax(served.marginal_gap), 1e-9)
        self.assertTrue(np.isfinite(served.log_loss).all())

    def test_goals_beyond_the_matrix_and_unusable_rows_are_nan(self) -> None:
        matrix = build_probability_matrix(1.4, 1.1, max_goals=7, rho=-0.04)
        matrices = np.stack([matrix, matrix, matrix])
        target = np.array([[0.4, 0.3, 0.3]] * 3)
        served = sc.served_scoreline(matrices, np.array([True, True, False]), target, np.array([9, 1, 1]), np.array([0, 1, 1]))
        self.assertTrue(np.isnan(served.log_loss[0]) and np.isnan(served.log_loss[2]))
        self.assertTrue(np.isfinite(served.log_loss[1]))
        raw = sc.scoreline_log_loss(matrices, np.array([True, True, False]), np.array([9, 1, 1]), np.array([0, 1, 1]))
        self.assertTrue(np.isnan(raw[0]) and np.isnan(raw[2]))
        self.assertAlmostEqual(raw[1], float(-np.log(matrix[1, 1] / matrix.sum())))

    def test_rescaling_to_the_matrix_own_1x2_changes_no_scoreline(self) -> None:
        matrix = build_probability_matrix(1.4, 1.1, max_goals=7, rho=-0.04)
        own = np.array([matrix_1x2(matrix)])
        served = sc.served_scoreline(matrix[None], np.array([True]), own, np.array([2]), np.array([0]))
        raw = sc.scoreline_log_loss(matrix[None], np.array([True]), np.array([2]), np.array([0]))
        np.testing.assert_allclose(served.log_loss, raw, atol=1e-12)


def scored_candidates(brier_offsets: dict[str, float], jitter: dict[str, float] | None = None, n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Rows where each candidate is the true 1X2 pulled toward a bad guess by its offset, and toward noise by its jitter."""
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet([3.0, 2.0, 3.0], size=n)
    outcomes = np.array(["H", "D", "A"])[(rng.random(n)[:, None] > truth.cumsum(axis=1)).sum(axis=1)]
    frame = pd.DataFrame({"date": pd.date_range("2018-01-01", periods=n, freq="D"), "outcome": outcomes})
    bad = np.tile([0.05, 0.05, 0.9], (n, 1))
    for name in sc.CANDIDATE_NAMES:
        offset = brier_offsets.get(name, 0.0)
        noise = (jitter or {}).get(name, 0.0)
        probs = (1 - offset - noise) * truth + offset * bad + noise * rng.dirichlet([1.0, 1.0, 1.0], size=n)
        for column, cls in enumerate(sc.CLASSES):
            frame[f"{name}_{cls}"] = probs[:, column]
    return frame


class SelectionTests(unittest.TestCase):
    def test_a_clearly_better_complex_candidate_is_chosen(self) -> None:
        offsets = {name: 0.6 for name in sc.CANDIDATE_NAMES}
        offsets["blend_davidson_scaled"] = 0.0
        picked = sc.select_candidate(scored_candidates(offsets), n_boot=200)
        self.assertEqual(picked.best, "blend_davidson_scaled")
        self.assertEqual(picked.chosen, "blend_davidson_scaled")

    def test_the_simplest_candidate_wins_among_statistical_ties(self) -> None:
        offsets = {name: 0.6 for name in sc.CANDIDATE_NAMES}
        offsets["blend_davidson_scaled"] = 0.0
        offsets["elo_ordinal"] = 0.0
        picked = sc.select_candidate(scored_candidates(offsets, jitter={"elo_ordinal": 0.02}), n_boot=200)
        self.assertEqual(picked.chosen, "elo_ordinal")
        self.assertEqual(set(picked.versus_best), set(sc.CANDIDATE_NAMES) - {picked.best})

    def test_a_significantly_worse_simple_candidate_is_not_chosen(self) -> None:
        offsets = {name: 0.0 for name in sc.CANDIDATE_NAMES}
        offsets["elo_ordinal"] = 0.7
        picked = sc.select_candidate(scored_candidates(offsets), n_boot=200)
        self.assertNotEqual(picked.chosen, "elo_ordinal")


class BacktestIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_matches()
        cls.predictions, cls.folds = run(cls.frame)

    def test_served_matrices_reproduce_every_candidate_1x2_to_1e9(self) -> None:
        scored = self.predictions[self.predictions["supported"]]
        for spec in sc.CANDIDATES:
            gap = scored[f"marginal_gap_{spec.name}"]
            self.assertTrue(gap.notna().all(), spec.name)
            self.assertLess(gap.max(), 1e-9, spec.name)

    def test_unsupported_rows_carry_no_candidate_probability(self) -> None:
        unsupported = self.predictions[~self.predictions["supported"]]
        for spec in sc.CANDIDATES:
            if spec.blend:
                self.assertTrue(unsupported[[f"{spec.name}_{c}" for c in sc.CLASSES]].isna().all().all())

    def test_folds_record_fitted_candidate_parameters_from_earlier_years(self) -> None:
        for fold in self.folds:
            self.assertEqual(set(fold.candidates), set(sc.CANDIDATE_NAMES))
            for name, params in fold.candidates.items():
                self.assertEqual(params["blend_weight"] is not None, name.startswith("blend_"), name)

    def test_report_years_restrict_what_is_scored_and_selection_respects_the_cutoff(self) -> None:
        matches = sb.Matches(frame=self.frame, n_rows=len(self.frame), n_without_score=0)
        options = dict(first_test_year=2016, window_years=2, refit_days=60, n_boot=30)
        report = sb.run_backtest(matches, report_years=(2016, 2016), select=True, **options)
        self.assertEqual(set(report["by_year"]), {"2016"})
        self.assertEqual([f["year"] for f in report["folds"]], [2016])
        self.assertIn(report["selection"]["chosen"], sc.CANDIDATE_NAMES)
        with mock.patch.object(sb, "SELECTION_LAST_YEAR", 2016), self.assertRaises(ValueError):
            sb.run_backtest(matches, report_years=(2016, 2017), select=True, **options)
        with self.assertRaises(ValueError):
            sb.run_backtest(matches, candidates=["nope"], **options)

    def test_rendered_report_includes_draw_calibration_and_scoreline_tables(self) -> None:
        matches = sb.Matches(frame=self.frame, n_rows=len(self.frame), n_without_score=0)
        report = sb.run_backtest(matches, first_test_year=2016, window_years=2, refit_days=60, n_boot=30)
        text = sb.render_markdown(report)
        self.assertIn("Mean predicted 1X2 against observed", text)
        self.assertIn("Scoreline coherence", text)
        self.assertIn("Candidate parameters per fold", text)


class RawCacheTests(unittest.TestCase):
    def test_a_second_run_reuses_the_stored_raw_year_and_changed_data_misses(self) -> None:
        with_elo = sb.add_pre_match_elo(synthetic_matches())
        subset = with_elo[with_elo["year"] <= 2014]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            first = sb._raw_job((subset, 2014, cache), 60)
            with mock.patch.object(sb, "raw_poisson_year", side_effect=AssertionError("cache miss")):
                again = sb._raw_job((subset, 2014, cache), 60)
            np.testing.assert_array_equal(first.matrices, again.matrices)
            pd.testing.assert_frame_equal(first.frame, again.frame)
            changed = subset.copy()
            changed.loc[changed.index[-1], "home_score"] += 1
            self.assertNotEqual(sb._raw_cache_path(cache, subset, 2014, 60), sb._raw_cache_path(cache, changed, 2014, 60))
            self.assertNotEqual(sb._raw_cache_path(cache, subset, 2014, 60), sb._raw_cache_path(cache, subset, 2014, 1))
            self.assertEqual(len(list(cache.glob("*.pkl"))), 1)


if __name__ == "__main__":
    unittest.main()
