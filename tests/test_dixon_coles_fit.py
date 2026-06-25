from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from alttez.experiments.tournament_oracle import (
    DEFAULT_DIXON_COLES_RHO,
    build_probability_matrix,
    load_active_dixon_coles_rho,
)
from scripts.fit_dixon_coles_rho import dixon_coles_log_likelihood, fit_rho


class TestRhoLoader(unittest.TestCase):
    def test_missing_artifact_falls_back_to_default(self) -> None:
        missing = os.path.join(tempfile.gettempdir(), "alttez_no_such_dc_artifact.json")
        if os.path.exists(missing):
            os.remove(missing)
        self.assertEqual(load_active_dixon_coles_rho(missing), DEFAULT_DIXON_COLES_RHO)

    def test_reads_temp_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps({"rho": -0.18, "samples": 100}), encoding="utf-8")
            self.assertAlmostEqual(load_active_dixon_coles_rho(str(path)), -0.18, places=9)

    def test_env_override_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps({"rho": -0.07}), encoding="utf-8")
            prev = os.environ.get("ALTTEZ_DIXON_COLES_RHO_PATH")
            os.environ["ALTTEZ_DIXON_COLES_RHO_PATH"] = str(path)
            try:
                self.assertAlmostEqual(load_active_dixon_coles_rho(), -0.07, places=9)
            finally:
                if prev is None:
                    os.environ.pop("ALTTEZ_DIXON_COLES_RHO_PATH", None)
                else:
                    os.environ["ALTTEZ_DIXON_COLES_RHO_PATH"] = prev

    def test_out_of_bounds_rho_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps({"rho": -0.9}), encoding="utf-8")
            self.assertEqual(load_active_dixon_coles_rho(str(path)), DEFAULT_DIXON_COLES_RHO)

    def test_malformed_artifact_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text("{ not valid json", encoding="utf-8")
            self.assertEqual(load_active_dixon_coles_rho(str(path)), DEFAULT_DIXON_COLES_RHO)


class TestBuildMatrixRhoResolution(unittest.TestCase):
    def test_none_resolves_active_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps({"rho": -0.2}), encoding="utf-8")
            prev = os.environ.get("ALTTEZ_DIXON_COLES_RHO_PATH")
            os.environ["ALTTEZ_DIXON_COLES_RHO_PATH"] = str(path)
            try:
                resolved = build_probability_matrix(1.4, 1.2, max_goals=7, rho=None)
                explicit = build_probability_matrix(1.4, 1.2, max_goals=7, rho=-0.2)
            finally:
                if prev is None:
                    os.environ.pop("ALTTEZ_DIXON_COLES_RHO_PATH", None)
                else:
                    os.environ["ALTTEZ_DIXON_COLES_RHO_PATH"] = prev
            # rho=None debe producir exactamente la misma matriz que el rho activo.
            self.assertTrue(np.allclose(resolved, explicit))

    def test_default_is_none_resolution(self) -> None:
        # Sin pasar rho, debe equivaler a pasar rho=None (resolver activo).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps({"rho": -0.15}), encoding="utf-8")
            prev = os.environ.get("ALTTEZ_DIXON_COLES_RHO_PATH")
            os.environ["ALTTEZ_DIXON_COLES_RHO_PATH"] = str(path)
            try:
                default_call = build_probability_matrix(1.3, 1.1, max_goals=7)
                explicit = build_probability_matrix(1.3, 1.1, max_goals=7, rho=-0.15)
            finally:
                if prev is None:
                    os.environ.pop("ALTTEZ_DIXON_COLES_RHO_PATH", None)
                else:
                    os.environ["ALTTEZ_DIXON_COLES_RHO_PATH"] = prev
            self.assertTrue(np.allclose(default_call, explicit))

    def test_explicit_rho_overrides_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps({"rho": -0.25}), encoding="utf-8")
            prev = os.environ.get("ALTTEZ_DIXON_COLES_RHO_PATH")
            os.environ["ALTTEZ_DIXON_COLES_RHO_PATH"] = str(path)
            try:
                # rho=0.0 explicito NO debe aplicar correccion, ignorando el activo.
                no_corr = build_probability_matrix(1.4, 1.2, max_goals=7, rho=0.0)
                active = build_probability_matrix(1.4, 1.2, max_goals=7, rho=None)
            finally:
                if prev is None:
                    os.environ.pop("ALTTEZ_DIXON_COLES_RHO_PATH", None)
                else:
                    os.environ["ALTTEZ_DIXON_COLES_RHO_PATH"] = prev
            self.assertFalse(np.allclose(no_corr, active))

    def test_matrix_still_normalized_with_resolved_rho(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            path.write_text(json.dumps({"rho": -0.2}), encoding="utf-8")
            # Ruta explicita para no depender de variables de entorno.
            matrix = build_probability_matrix(
                1.5, 1.1, max_goals=7, rho=load_active_dixon_coles_rho(str(path))
            )
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=9)
        self.assertTrue((matrix >= 0).all())


class TestLogLikelihood(unittest.TestCase):
    def _draw_heavy_samples(self) -> np.ndarray:
        # Dataset sintetico con muchos empates 0-0 y 1-1 (donde la dependencia DC
        # negativa eleva la masa). lambdas moderadas alrededor de 1.0.
        lambda_h, lambda_a = 1.1, 1.0
        rows = []
        # 40 partidos 0-0, 40 partidos 1-1, y algunos resultados dispersos.
        for _ in range(40):
            rows.append((0, 0, lambda_h, lambda_a))
        for _ in range(40):
            rows.append((1, 1, lambda_h, lambda_a))
        for hg, ag in [(2, 1), (1, 2), (0, 1), (1, 0), (2, 0), (0, 2)]:
            for _ in range(5):
                rows.append((hg, ag, lambda_h, lambda_a))
        return np.asarray(rows, dtype=float)

    def test_fitted_rho_beats_zero_on_draw_heavy_data(self) -> None:
        samples = self._draw_heavy_samples()
        ll_zero = dixon_coles_log_likelihood(0.0, samples)
        fitted = fit_rho(samples)
        ll_fitted = dixon_coles_log_likelihood(fitted, samples)
        # En datos cargados de empates, un rho negativo ajustado mejora la
        # verosimilitud frente a rho=0 (independencia).
        self.assertLess(fitted, 0.0)
        self.assertGreater(ll_fitted, ll_zero)


if __name__ == "__main__":
    unittest.main()
