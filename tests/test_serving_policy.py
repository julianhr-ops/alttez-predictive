from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from alttez.evaluation import soccer_backtest as sb
from alttez.evaluation.soccer_candidates import CANDIDATES, PRIMARY_CANDIDATE, fit_candidate
from alttez.features.elo import rate_matches
from alttez.inference.calibration import matrix_1x2
from alttez.inference.prediction_committee import build_prediction_committee
from alttez.inference.router import (
    XGBOOST_FEATURE_COLUMNS,
    ActiveModelSelection,
    ActiveRecordLookup,
    BivariatePoissonMatchPredictor,
    build_active_predictor,
)
from alttez.inference.serving_policy import (
    POLICY_ELO_ORDINAL,
    POLICY_NONE,
    POLICY_POISSON,
    SERVING_POLICY_ENV,
    SERVING_WINDOW_YEARS,
    EloUnavailableError,
    ServingPolicy,
    build_serving_policy,
    serving_policy_enabled,
)
from tests.test_inference_router import DummyGoalModel, _historical_df
from alttez.reports.daily_brief import _build_telegram_summary_text, prob_over_2_5, render_daily_brief_html
from alttez.reports.daily_brief.payload import _build_match_brief
from tests.test_soccer_backtest import synthetic_matches

SIDES = ("home", "draw", "away")


def _score_matrix(shape: int = 8) -> np.ndarray:
    """A fixed, strictly positive score matrix that is not symmetric."""
    rng = np.random.default_rng(3)
    matrix = rng.random((shape, shape)) ** 3
    return matrix / matrix.sum()


def _live_predictor(**environ: str):
    with patch.dict(os.environ, environ):
        return build_active_predictor(_historical_df(), "2026-05-27")


class EnabledSwitchTests(unittest.TestCase):
    def test_default_is_on(self) -> None:
        self.assertTrue(serving_policy_enabled(env={}))

    def test_env_turns_it_off_and_on(self) -> None:
        self.assertFalse(serving_policy_enabled(env={SERVING_POLICY_ENV: "off"}))
        self.assertFalse(serving_policy_enabled(env={SERVING_POLICY_ENV: POLICY_POISSON}))
        self.assertTrue(serving_policy_enabled(env={SERVING_POLICY_ENV: POLICY_ELO_ORDINAL}))

    def test_explicit_argument_beats_the_environment(self) -> None:
        self.assertFalse(serving_policy_enabled(False, env={SERVING_POLICY_ENV: "on"}))

    def test_unknown_value_is_rejected_not_ignored(self) -> None:
        with self.assertRaises(ValueError):
            serving_policy_enabled(env={SERVING_POLICY_ENV: "elo-ish"})


class MatchesTheValidatedRuleTests(unittest.TestCase):
    def test_window_and_rule_are_the_ones_that_were_validated(self) -> None:
        self.assertEqual(SERVING_WINDOW_YEARS, sb.WINDOW_YEARS)
        spec = next(candidate for candidate in CANDIDATES if candidate.name == PRIMARY_CANDIDATE)
        self.assertEqual((spec.kind, spec.fit_scale, spec.blend), ("ordinal", False, False))

    def test_live_policy_equals_the_backtest_fit_at_the_same_date(self) -> None:
        frame = synthetic_matches()
        year = 2016
        policy = build_serving_policy(frame, f"{year}-01-01")

        with_elo = sb.add_pre_match_elo(frame)
        spec = next(candidate for candidate in CANDIDATES if candidate.name == PRIMARY_CANDIDATE)
        expected = fit_candidate(spec, sb._window(with_elo, year, sb.WINDOW_YEARS)).draw_model

        self.assertAlmostEqual(policy.draw_model.home_advantage, expected.home_advantage, places=9)
        self.assertAlmostEqual(policy.draw_model.draw_param, expected.draw_param, places=9)
        before = frame[frame["date"] < pd.Timestamp(year=year, month=1, day=1)].reset_index(drop=True)
        self.assertEqual(dict(policy.ratings), rate_matches(before).ratings)


class NoLeakageTests(unittest.TestCase):
    def test_matches_on_or_after_the_target_date_change_nothing(self) -> None:
        frame = synthetic_matches()
        target = pd.Timestamp("2016-06-30")
        baseline = build_serving_policy(frame, target)

        altered = frame.copy()
        altered.loc[altered["date"] >= target, ["home_score", "away_score"]] = [9, 0]
        policy = build_serving_policy(altered, target)

        self.assertEqual(dict(policy.ratings), dict(baseline.ratings))
        self.assertEqual(policy.draw_model, baseline.draw_model)

    def test_a_match_on_the_target_date_itself_is_excluded(self) -> None:
        frame = synthetic_matches()
        target = frame.loc[frame["date"].dt.year == 2016, "date"].iloc[5]
        without_day = frame[frame["date"] < target]
        self.assertEqual(dict(build_serving_policy(frame, target).ratings), dict(build_serving_policy(without_day, target).ratings))


class ServedMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = build_serving_policy(synthetic_matches(), "2017-01-01")
        self.team_a, self.team_b = sorted(self.policy.ratings)[:2]

    def test_target_is_a_probability_triplet(self) -> None:
        for neutral in (False, True):
            target = self.policy.target_1x2(self.team_a, self.team_b, neutral)
            self.assertAlmostEqual(sum(target), 1.0, places=12)
            self.assertTrue(all(0.0 < p < 1.0 for p in target))

    def test_matrix_marginals_equal_the_served_1x2(self) -> None:
        matrix = _score_matrix()
        for neutral in (False, True):
            target = self.policy.target_1x2(self.team_a, self.team_b, neutral)
            served = self.policy.serve(matrix, self.team_a, self.team_b, neutral)
            self.assertAlmostEqual(float(served.sum()), 1.0, places=12)
            for got, want in zip(matrix_1x2(served), target, strict=True):
                self.assertAlmostEqual(got, want, delta=1e-9)

    def test_scoreline_shape_is_kept_within_each_class(self) -> None:
        matrix = _score_matrix()
        served = self.policy.serve(matrix, self.team_a, self.team_b, False)
        home_wins = np.tril_indices_from(matrix, k=-1)
        np.testing.assert_allclose(served[home_wins] / served[home_wins].sum(), matrix[home_wins] / matrix[home_wins].sum(), atol=1e-12)

    def test_a_team_without_a_rating_is_loud(self) -> None:
        with self.assertRaises(EloUnavailableError):
            self.policy.serve(_score_matrix(), self.team_a, "Atlantis", False)


class RouterIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(None)))
        self.enterContext(patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0"}))
        os.environ.pop(SERVING_POLICY_ENV, None)

    def test_served_probabilities_are_the_elo_1x2_and_raw_poisson_is_kept(self) -> None:
        served = _live_predictor()
        off = _live_predictor(**{SERVING_POLICY_ENV: "off"})
        prediction = served.predict_match("Mexico", "Brazil", is_neutral=True)
        poisson = off.predict_match("Mexico", "Brazil", is_neutral=True)

        target = served.serving_policy.target_1x2("Mexico", "Brazil", True)
        self.assertEqual(prediction["serving_policy"], POLICY_ELO_ORDINAL)
        self.assertFalse(prediction["used_fallback"])
        for side, want in zip(SIDES, target, strict=True):
            self.assertAlmostEqual(float(prediction[f"prob_{side}"]), want, delta=1e-9)
            self.assertAlmostEqual(float(prediction[f"raw_prob_{side}"]), float(poisson[f"prob_{side}"]), places=9)
        for got, want in zip(matrix_1x2(prediction["prob_matrix"]), target, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-9)
        self.assertAlmostEqual(float(prediction["lambda_h"]), float(poisson["lambda_h"]), places=9)

    def test_policy_off_serves_plain_poisson(self) -> None:
        prediction = _live_predictor(**{SERVING_POLICY_ENV: "off"}).predict_match("Mexico", "Brazil", is_neutral=True)

        self.assertEqual(prediction["serving_policy"], POLICY_POISSON)
        self.assertFalse(prediction["used_fallback"])
        for side in SIDES:
            self.assertAlmostEqual(float(prediction[f"raw_prob_{side}"]), float(prediction[f"prob_{side}"]), places=9)

    def test_missing_elo_falls_back_to_poisson_loudly(self) -> None:
        predictor = _live_predictor()
        predictor.serving_policy = ServingPolicy(POLICY_ELO_ORDINAL, predictor.serving_policy.draw_model, {"Mexico": 1500.0})

        with self.assertLogs("alttez.inference.router", level="WARNING"):
            prediction = predictor.predict_match("Mexico", "Brazil", is_neutral=True)

        self.assertTrue(prediction["prediction_supported"])
        self.assertTrue(prediction["used_fallback"])
        self.assertIn("Brazil", prediction["fallback_reason"])
        self.assertEqual(prediction["serving_policy"], POLICY_POISSON)
        for side in SIDES:
            self.assertAlmostEqual(float(prediction[f"raw_prob_{side}"]), float(prediction[f"prob_{side}"]), places=9)

    def test_a_broken_elo_build_is_a_run_level_fallback(self) -> None:
        history = _historical_df().drop(columns=["tournament"])
        with self.assertLogs("alttez.inference.router", level="WARNING"):
            predictor = build_active_predictor(history, "2026-05-27")
            prediction = predictor.predict_match("Mexico", "Brazil", is_neutral=True)

        self.assertTrue(predictor.used_fallback)
        self.assertIn("serving policy", predictor.fallback_reason)
        self.assertTrue(prediction["used_fallback"])
        self.assertEqual(prediction["serving_policy"], POLICY_POISSON)

    def test_committee_member_baselines_stay_raw_poisson(self) -> None:
        selection = ActiveModelSelection(
            version_key=None,
            model_version="bivariate-test",
            model_family="bivariate_poisson",
            artifact_path=None,
            requested_production_status="committee",
        )
        member = BivariatePoissonMatchPredictor(_historical_df(), "2026-05-27", selection=selection)
        self.assertIsNone(member.poisson_baseline.serving_policy)


def _committee():
    with (
        patch("alttez.inference.prediction_committee.load_latest_challenger_record", return_value=None),
        patch("alttez.inference.prediction_committee._resolve_member_weights", return_value=None),
    ):
        return build_prediction_committee(_historical_df(), "2026-05-27")


class CommitteeServingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(None)))
        self.enterContext(patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0"}))
        os.environ.pop(SERVING_POLICY_ENV, None)

    def test_served_total_is_used_in_brief_and_telegram(self) -> None:
        committee = _committee()
        prediction = committee.predict_match("Mexico", "Brazil", is_neutral=True)
        matrix = prediction["prob_matrix"]
        expected = sum(float(matrix[h, a]) for h in range(matrix.shape[0]) for a in range(matrix.shape[1]) if h + a >= 3)
        totals = [sum(float(matrix[h, a]) for h in range(matrix.shape[0]) for a in range(matrix.shape[1]) if h + a == k)
                  for k in range(sum(matrix.shape) - 1)]
        self.assertAlmostEqual(prediction["prob_over_2_5"], expected)
        self.assertEqual(prediction["most_likely_total"], int(np.argmax(totals)))
        self.assertNotAlmostEqual(expected, prob_over_2_5(prediction["lambda_h"], prediction["lambda_a"]), places=3)

        match = _build_match_brief({"id": "m1", "home_team": "Mexico", "away_team": "Brazil", "is_neutral": True}, committee)
        payload = {"target_date": "2026-05-27", "generated_at": "2026-05-27", "batch_summary": {},
                   "alert_summary": {"items": [], "critical": 0, "warning": 0, "info": 0},
                   "summary": {"ranked_matches": [match], "total_matches": 1, "analyzable_matches": 1,
                               "high_priority_matches": 0, "certification_status": "certified", "coverage_pct": 100},
                   "matches": [match]}
        expected_text = f"{expected * 100:.1f}%"
        self.assertEqual(match["prob_over_2_5"], prediction["prob_over_2_5"])
        self.assertIn(expected_text, render_daily_brief_html(payload))
        self.assertIn(expected_text, _build_telegram_summary_text(payload))

    def test_committee_serves_the_elo_1x2_and_keeps_the_members_as_diagnostics(self) -> None:
        committee = _committee()
        target = committee.members[0].predictor.serving_policy.target_1x2("Mexico", "Brazil", True)

        prediction = committee.predict_match("Mexico", "Brazil", is_neutral=True)

        for side, want in zip(SIDES, target, strict=True):
            self.assertAlmostEqual(float(prediction[f"prob_{side}"]), want, delta=1e-9)
        for got, want in zip(matrix_1x2(prediction["prob_matrix"]), target, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-9)
        self.assertAlmostEqual(float(prediction["prob_matrix"].sum()), 1.0, places=12)
        self.assertEqual(prediction["serving_policy"], POLICY_ELO_ORDINAL)
        self.assertFalse(prediction["calibration_applied"])
        members = {item["role"]: item for item in prediction["committee_members"]}
        self.assertEqual(set(members), {"champion", "poisson_baseline", "bivariate"})
        self.assertGreater(members["poisson_baseline"]["prob_draw"], members["champion"]["prob_draw"])

    def test_committee_follows_the_champion_when_the_policy_is_off(self) -> None:
        with patch.dict(os.environ, {SERVING_POLICY_ENV: "off"}):
            committee = _committee()
            prediction = committee.predict_match("Mexico", "Brazil", is_neutral=True)

        self.assertEqual(prediction["serving_policy"], POLICY_POISSON)
        self.assertFalse(prediction["used_fallback"])

    def test_committee_reports_a_per_match_elo_fallback(self) -> None:
        committee = _committee()
        self.assertFalse(committee.used_fallback)

        with self.assertLogs("alttez.inference.router", level="WARNING"):
            prediction = committee.predict_match("Mexico", "Atlantis", is_neutral=True)

        self.assertTrue(prediction["prediction_supported"])
        self.assertTrue(prediction["used_fallback"])
        self.assertIn("champion", prediction["fallback_reason"])
        self.assertIn("Atlantis", prediction["fallback_reason"])
        self.assertEqual(prediction["serving_policy"], POLICY_POISSON)

    def test_a_match_without_fallback_does_not_inherit_the_previous_one(self) -> None:
        committee = _committee()
        with self.assertLogs("alttez.inference.router", level="WARNING"):
            committee.predict_match("Mexico", "Atlantis", is_neutral=True)

        prediction = committee.predict_match("Mexico", "Brazil", is_neutral=True)

        self.assertFalse(prediction["used_fallback"])
        self.assertIsNone(prediction["fallback_reason"])


class NonPoissonChampionTests(unittest.TestCase):
    def test_an_active_xgboost_model_serves_without_the_elo_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "xgboost-active.joblib"
            joblib.dump(
                {
                    "artifact_schema_version": 2,
                    "models": {"home_goals": DummyGoalModel(1.8), "away_goals": DummyGoalModel(0.9)},
                    "feature_columns": list(XGBOOST_FEATURE_COLUMNS),
                    "model_version": "xgboost-active-v1",
                    "model_family": "xgboost",
                },
                artifact_path,
            )
            record = {
                "version_key": "xgboost-active-v1:2026-05-27",
                "model_version": "xgboost-active-v1",
                "model_family": "xgboost",
                "artifact_path": str(artifact_path),
                "production_status": "active",
            }
            with (
                patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0", "ALTTEZ_ARTIFACTS_DIR": tmp}),
                patch("alttez.inference.router._load_active_model_record_safe", return_value=ActiveRecordLookup(record)),
                patch("alttez.inference.prediction_committee.load_latest_challenger_record", return_value=None),
                patch("alttez.inference.prediction_committee._resolve_member_weights", return_value=None),
            ):
                predictor = build_active_predictor(_historical_df(), "2026-05-27")
                single = predictor.predict_match("Mexico", "USA", is_neutral=True)
                committee = build_prediction_committee(_historical_df(), "2026-05-27")
                blended = committee.predict_match("Mexico", "USA", is_neutral=True)

        self.assertEqual(predictor.model_family, "xgboost")
        self.assertEqual(single["serving_policy"], POLICY_NONE)
        self.assertEqual(blended["serving_policy"], POLICY_NONE)


if __name__ == "__main__":
    unittest.main()
