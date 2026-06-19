from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from alttez.features.hierarchical import (
    HierarchicalBayesEngine,
    HierarchicalStrengths,
    TeamStrength,
    load_active_hierarchical_strengths,
)

_HAS_PYMC = importlib.util.find_spec("pymc") is not None


def _strengths() -> HierarchicalStrengths:
    return HierarchicalStrengths(
        mu=0.1,
        gamma=0.3,
        sigma_att=0.4,
        sigma_def=0.4,
        teams={
            "Brazil": TeamStrength(attack=0.5, defense=0.4),
            "Bolivia": TeamStrength(attack=-0.4, defense=-0.5),
        },
        samples=500,
    )


class HierarchicalEngineTest(unittest.TestCase):
    def test_strong_team_has_higher_expected_goals(self):
        engine = HierarchicalBayesEngine(_strengths())
        lh, la = engine.expected_goals("Brazil", "Bolivia", is_neutral=True)
        # El favorito (mejor ataque, rival peor defensa) marca mas.
        self.assertGreater(lh, la)

    def test_home_advantage_only_when_not_neutral(self):
        engine = HierarchicalBayesEngine(_strengths())
        neutral_h, _ = engine.expected_goals("Brazil", "Bolivia", is_neutral=True)
        home_h, _ = engine.expected_goals("Brazil", "Bolivia", is_neutral=False)
        # gamma>0 sube la lambda local en sede no neutral.
        self.assertGreater(home_h, neutral_h)

    def test_matrix_normalizes(self):
        engine = HierarchicalBayesEngine(_strengths())
        matrix = engine.build_matrix("Brazil", "Bolivia", is_neutral=True, max_goals=7)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=9)

    def test_cold_start_unknown_team_is_neutral_strength(self):
        engine = HierarchicalBayesEngine(_strengths())
        strength = engine.strengths.strength_for("Atlantis")
        self.assertEqual((strength.attack, strength.defense), (0.0, 0.0))

    def test_engine_not_ready_when_empty(self):
        self.assertFalse(HierarchicalBayesEngine(HierarchicalStrengths()).is_ready)
        self.assertTrue(HierarchicalBayesEngine(_strengths()).is_ready)


class HierarchicalLoaderTest(unittest.TestCase):
    def test_empty_when_absent(self):
        with patch.dict(os.environ, {"ALTTEZ_HIERARCHICAL_PATH": "does/not/exist.json"}):
            self.assertTrue(load_active_hierarchical_strengths().is_empty)

    def test_reads_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "strengths.json"
            artifact.write_text(
                json.dumps(
                    {
                        "mu": 0.2,
                        "gamma": 0.25,
                        "sigma_att": 0.3,
                        "sigma_def": 0.3,
                        "samples": 100,
                        "teams": {"Brazil": {"attack": 0.6, "defense": 0.5}},
                    }
                ),
                encoding="utf-8",
            )
            strengths = load_active_hierarchical_strengths(artifact)
        self.assertFalse(strengths.is_empty)
        self.assertAlmostEqual(strengths.mu, 0.2)
        self.assertAlmostEqual(strengths.strength_for("Brazil").attack, 0.6)


class HierarchicalPredictorTest(unittest.TestCase):
    def _selection(self):
        from alttez.inference.router import ActiveModelSelection

        return ActiveModelSelection(
            version_key=None,
            model_version="hierarchical-bayes-committee",
            model_family="hierarchical_bayes",
            artifact_path=None,
            requested_production_status="committee",
        )

    def test_unsupported_without_artifact(self):
        from alttez.inference.router import HierarchicalMatchPredictor

        with patch.dict(os.environ, {"ALTTEZ_HIERARCHICAL_PATH": "does/not/exist.json"}):
            predictor = HierarchicalMatchPredictor(selection=self._selection())
            prediction = predictor.predict_match("Brazil", "Bolivia", is_neutral=True)
        self.assertFalse(prediction["prediction_supported"])

    def test_supported_with_artifact(self):
        from alttez.inference.router import HierarchicalMatchPredictor

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "strengths.json"
            artifact.write_text(
                json.dumps(
                    {
                        "mu": 0.1,
                        "gamma": 0.3,
                        "sigma_att": 0.4,
                        "sigma_def": 0.4,
                        "samples": 500,
                        "teams": {
                            "Brazil": {"attack": 0.5, "defense": 0.4},
                            "Bolivia": {"attack": -0.4, "defense": -0.5},
                        },
                    }
                ),
                encoding="utf-8",
            )
            predictor = HierarchicalMatchPredictor(selection=self._selection(), strengths_path=str(artifact))
            prediction = predictor.predict_match("Brazil", "Bolivia", is_neutral=True)
        self.assertTrue(prediction["prediction_supported"])
        total = float(prediction["prob_home"]) + float(prediction["prob_draw"]) + float(prediction["prob_away"])
        self.assertAlmostEqual(total, 1.0, places=3)


@unittest.skipUnless(_HAS_PYMC, "PyMC no instalado: el ajuste es dependencia de fit-time.")
class HierarchicalFitTest(unittest.TestCase):
    def test_fit_recovers_team_ordering(self):
        from alttez.features.hierarchical import fit_hierarchical_model

        # Dataset sintetico: 'Strong' golea a 'Weak' consistentemente.
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(60):
            rows.append(("Strong", "Weak", int(rng.poisson(2.5)), int(rng.poisson(0.6)), True))
            rows.append(("Weak", "Strong", int(rng.poisson(0.6)), int(rng.poisson(2.5)), True))
        df = pd.DataFrame(rows, columns=["home_team", "away_team", "home_score", "away_score", "neutral"])

        result, _idata = fit_hierarchical_model(df, draws=300, tune=300, chains=2, seed=0)
        # El equipo fuerte debe tener mayor ataque y mejor (mayor) defensa.
        self.assertGreater(result["teams"]["Strong"]["attack"], result["teams"]["Weak"]["attack"])
        self.assertGreater(result["teams"]["Strong"]["defense"], result["teams"]["Weak"]["defense"])


if __name__ == "__main__":
    unittest.main()
