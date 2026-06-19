"""Bivariate Poisson con inflacion diagonal (Karlis-Ntzoufris 2003).

El parche Dixon-Coles corrige post-hoc la subestimacion de empates del producto
Poisson independiente. El Bivariate Poisson (BP) modela la correlacion de forma
nativa via un termino comun lambda3, PERO el BP plano solo captura correlacion
POSITIVA (Cov = lambda3 >= 0); no puede reproducir la dependencia negativa /
exceso de empates de marcador bajo que justamente motiva a Dixon-Coles.

Por eso aqui se usa el Bivariate Poisson de INFLACION DIAGONAL (DI-BP): una mezcla
entre el BP y una distribucion concentrada en la diagonal (x == y), que sube la
masa de empates sin forzar correlacion positiva en el resto de la matriz.

    P(x, y) = (1 - p) * BP(x, y; l1, l2, l3) + p * D(x) * 1[x == y]

con D(x) = Poisson(x; lambda_d) y lambda_d fijo a la media de goles (no se ajusta,
mantiene el numero de parametros libres en dos: lambda3 y p).

Parametrizacion de medias marginales: el BP(l1,l2,l3) tiene marginales
Poisson(l1+l3) y Poisson(l2+l3). Para preservar las medias lambda_h/lambda_a que
entrega el PoissonEngine se fija l1 = lambda_h - l3, l2 = lambda_a - l3, con l3
acotado a [0, min(lambda_h, lambda_a)] para que l1, l2 >= 0.

Funciones puras, O(max_goals^2) por celda con cache de factoriales. Neutra
(lambda3 = 0, p = 0) -> producto Poisson independiente, preservando el arranque
sin artefacto.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import poisson

LOGGER = logging.getLogger(__name__)

DEFAULT_BIVARIATE_PATH = Path("artifacts/bivariate/active.json")
BIVARIATE_PATH_ENV = "ALTTEZ_BIVARIATE_PARAMS_PATH"
# Cotas de seguridad para los parametros cargados desde artefacto.
MAX_DIAGONAL_WEIGHT = 0.9


@dataclass(slots=True, frozen=True)
class BivariatePoissonParams:
    """Parametros del DI-BP ajustados offline. Neutro = Poisson independiente."""

    lambda3: float = 0.0
    diagonal_weight: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return self.lambda3 <= 0.0 and self.diagonal_weight <= 0.0


class BivariatePoissonEngine:
    """Construye la matriz de marcadores DI-BP a partir de lambdas marginales."""

    def __init__(self, lambda3: float = 0.0, diagonal_weight: float = 0.0) -> None:
        self.lambda3 = max(float(lambda3), 0.0)
        self.diagonal_weight = float(np.clip(diagonal_weight, 0.0, MAX_DIAGONAL_WEIGHT))

    def _bivariate_matrix(self, lambda1: float, lambda2: float, lambda3: float, max_goals: int) -> np.ndarray:
        """PMF del Bivariate Poisson estandar (Karlis-Ntzoufris).

        P(x,y) = exp(-(l1+l2+l3)) * sum_{k=0}^{min(x,y)}
                 (l1^{x-k}/(x-k)!) (l2^{y-k}/(y-k)!) (l3^k/k!).
        """
        size = max_goals + 1
        factorial = [math.factorial(i) for i in range(size)]
        matrix = np.zeros((size, size), dtype=float)
        base = math.exp(-(lambda1 + lambda2 + lambda3))
        for x in range(size):
            for y in range(size):
                acc = 0.0
                for k in range(min(x, y) + 1):
                    acc += (
                        (lambda1 ** (x - k) / factorial[x - k])
                        * (lambda2 ** (y - k) / factorial[y - k])
                        * (lambda3 ** k / factorial[k])
                    )
                matrix[x, y] = base * acc
        return matrix

    def build_matrix(self, lambda_home: float, lambda_away: float, max_goals: int = 7) -> np.ndarray:
        """Matriz de marcadores DI-BP normalizada (goles local x visitante)."""
        lambda_home = max(float(lambda_home), 0.05)
        lambda_away = max(float(lambda_away), 0.05)
        # l3 acotado para no volver negativas las marginales l1, l2.
        l3 = float(min(self.lambda3, lambda_home, lambda_away))
        l1 = lambda_home - l3
        l2 = lambda_away - l3

        matrix = self._bivariate_matrix(l1, l2, l3, max_goals)

        if self.diagonal_weight > 0.0:
            # Componente diagonal: Poisson centrado en la media de goles, solo x==y.
            lambda_d = (lambda_home + lambda_away) / 2.0
            diag = np.zeros_like(matrix)
            idx = np.arange(max_goals + 1)
            diag[idx, idx] = poisson.pmf(idx, lambda_d)
            diag_sum = diag.sum()
            if diag_sum > 0:
                diag /= diag_sum
            matrix = (1.0 - self.diagonal_weight) * matrix + self.diagonal_weight * diag

        total = matrix.sum()
        if total <= 0:
            raise ValueError("La matriz Bivariate Poisson no pudo normalizarse.")
        return matrix / total


def load_active_bivariate_params(path: str | Path | None = None) -> BivariatePoissonParams:
    """Carga los parametros DI-BP del artefacto activo.

    Devuelve parametros NEUTROS (Poisson independiente) si el artefacto falta o es
    ilegible: la inferencia nunca debe romperse por su ausencia.
    """
    resolved = Path(path or os.getenv(BIVARIATE_PATH_ENV) or DEFAULT_BIVARIATE_PATH)
    if not resolved.exists():
        return BivariatePoissonParams()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return BivariatePoissonParams(
            lambda3=max(float(payload.get("lambda3", 0.0)), 0.0),
            diagonal_weight=float(np.clip(float(payload.get("diagonal_weight", 0.0)), 0.0, MAX_DIAGONAL_WEIGHT)),
        )
    except Exception as exc:  # noqa: BLE001 - artefacto corrupto no debe tumbar inferencia.
        LOGGER.warning("bivariate_artifact_unreadable path=%s detail=%s", resolved, exc)
        return BivariatePoissonParams()
