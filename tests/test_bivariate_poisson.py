from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from alttez.experiments.tournament_oracle import build_probability_matrix
from alttez.features.bivariate_poisson import (
    BivariatePoissonEngine,
    BivariatePoissonParams,
    load_active_bivariate_params,
)


def _draw_mass(matrix: np.ndarray) -> float:
    return float(np.trace(matrix))


class BivariatePoissonEngineTest(unittest.TestCase):
    def test_neutral_reduces_to_independent_poisson(self):
        engine = BivariatePoissonEngine(lambda3=0.0, diagonal_weight=0.0)
        bp = engine.build_matrix(1.4, 1.1, max_goals=7)
        indep = build_probability_matrix(1.4, 1.1, max_goals=7, rho=0.0)
        np.testing.assert_allclose(bp, indep, atol=1e-9)

    def test_matrix_normalizes(self):
        engine = BivariatePoissonEngine(lambda3=0.3, diagonal_weight=0.2)
        matrix = engine.build_matrix(1.6, 1.2, max_goals=7)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=9)

    def test_diagonal_inflation_increases_draw_mass(self):
        neutral = BivariatePoissonEngine(0.0, 0.0).build_matrix(1.5, 1.5, max_goals=7)
        inflated = BivariatePoissonEngine(0.0, 0.3).build_matrix(1.5, 1.5, max_goals=7)
        self.assertGreater(_draw_mass(inflated), _draw_mass(neutral))

    def test_lambda3_bounded_to_marginal_min(self):
        # lambda3 mayor que min(lh, la) no debe romper (se acota internamente).
        engine = BivariatePoissonEngine(lambda3=5.0, diagonal_weight=0.0)
        matrix = engine.build_matrix(0.8, 1.2, max_goals=7)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=9)
        self.assertTrue(np.all(matrix >= 0.0))


class BivariateParamsLoaderTest(unittest.TestCase):
    def test_neutral_when_artifact_absent(self):
        with patch.dict(os.environ, {"ALTTEZ_BIVARIATE_PARAMS_PATH": "does/not/exist.json"}):
            params = load_active_bivariate_params()
        self.assertTrue(params.is_neutral)

    def test_reads_and_clips_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "active.json"
            artifact.write_text(
                json.dumps({"lambda3": 0.4, "diagonal_weight": 5.0}), encoding="utf-8"
            )
            params = load_active_bivariate_params(artifact)
        self.assertEqual(params.lambda3, 0.4)
        self.assertLessEqual(params.diagonal_weight, 0.9)  # clipped
        self.assertFalse(params.is_neutral)

    def test_negative_lambda3_floored(self):
        self.assertTrue(BivariatePoissonParams(lambda3=-1.0, diagonal_weight=0.0).is_neutral)


def _historical_df() -> pd.DataFrame:
    rows = [
        ("2025-01-10", "Mexico", "USA", 1, 0, True),
        ("2025-02-15", "USA", "Canada", 2, 1, False),
        ("2025-03-12", "Canada", "Mexico", 0, 1, False),
        ("2025-07-18", "Mexico", "Canada", 2, 0, False),
        ("2025-08-14", "USA", "Mexico", 1, 1, False),
        ("2025-09-09", "Canada", "USA", 0, 1, False),
        ("2026-01-15", "Mexico", "USA", 1, 0, True),
        ("2026-02-12", "USA", "Canada", 0, 2, True),
    ]
    frame = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


class BivariatePredictorTest(unittest.TestCase):
    def test_predictor_returns_valid_distribution(self):
        from alttez.inference.router import ActiveModelSelection, BivariatePoissonMatchPredictor

        with patch.dict(os.environ, {"ALTTEZ_MIN_SUPPORTED_MATCHES": "0"}, clear=False):
            predictor = BivariatePoissonMatchPredictor(
                _historical_df(),
                "2026-05-27",
                selection=ActiveModelSelection(
                    version_key=None,
                    model_version="bivariate-poisson-committee",
                    model_family="bivariate_poisson",
                    artifact_path=None,
                    requested_production_status="committee",
                ),
            )
            prediction = predictor.predict_match("Mexico", "USA", is_neutral=True)

        self.assertEqual(predictor.model_family, "bivariate_poisson")
        self.assertTrue(prediction["prediction_supported"])
        self.assertIsNotNone(prediction["prob_matrix"])
        total = (
            float(prediction["prob_home"])
            + float(prediction["prob_draw"])
            + float(prediction["prob_away"])
        )
        self.assertAlmostEqual(total, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
