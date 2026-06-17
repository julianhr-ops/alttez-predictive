"""Calibracion post-hoc de probabilidades para el motor de inferencia.

El modelo activo tiene buena precision de punto (~50% en 1X2) pero esta mal
calibrado: log loss historico ~3.85 frente a ~1.10 de un uniforme, senal de
sobreconfianza (asigna prob ~0 al resultado que luego ocurre). Esta capa corrige
eso sin reentrenar, mediante:

  1. Temperature scaling: p -> p^(1/T) / Z. T>1 suaviza la sobreconfianza.
  2. Piso por clase: ninguna clase queda en ~0, lo que acota el log loss.

Es post-hoc, reversible y barato. El parametro T se ajusta offline sobre las
evaluaciones cerradas (ver scripts/fit_probability_calibration.py) y se persiste
en un artefacto JSON. Si el artefacto no existe, la calibracion es NEUTRA
(T=1.0, piso=0.0): el comportamiento es identico al no calibrado, preservando el
invariante de que la inferencia arranca sin dependencias externas.

Funciones puras y unitariamente testeables segun la politica anti-monolitica.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_CALIBRATION_PATH = Path("artifacts/calibration/active.json")
CALIBRATION_PATH_ENV = "ALTTEZ_CALIBRATION_PATH"
# Piso por clase aplicado a 1X2 en inferencia. 0.0 = neutro (sin piso).
DEFAULT_CLASS_FLOOR = 0.0
MIN_TEMPERATURE = 0.05


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Aplica temperature scaling a un vector/matriz de probabilidades y renormaliza.

    T=1.0 es identidad. T>1 aplana (menos confianza); 0<T<1 agudiza.
    """
    array = np.asarray(probabilities, dtype=float)
    temp = max(float(temperature), MIN_TEMPERATURE)
    if temp == 1.0:
        scaled = array.copy()
    else:
        # Trabaja en log-espacio para estabilidad: p^(1/T) = exp(log(p)/T).
        with np.errstate(divide="ignore"):
            log_p = np.where(array > 0, np.log(array), -np.inf)
        scaled = np.exp(log_p / temp)
    total = scaled.sum()
    if not np.isfinite(total) or total <= 0:
        # Degenerado: devuelve uniforme antes que propagar NaN.
        return np.full(array.shape, 1.0 / array.size, dtype=float)
    return scaled / total


def apply_class_floor(triplet: np.ndarray, floor: float) -> np.ndarray:
    """Garantiza prob >= floor en cada clase y suma 1. floor=0 es identidad.

    Reserva floor para las clases por debajo del piso y reparte la masa restante
    proporcionalmente entre las clases por encima. Asi el piso se respeta tras
    normalizar (a diferencia de floor+renormalizar, que lo vuelve a hundir).
    """
    array = np.asarray(triplet, dtype=float)
    total = array.sum()
    normalized = array / total if total > 0 else np.full(array.shape, 1.0 / array.size)
    if floor <= 0.0:
        return normalized
    below = normalized < floor
    if not below.any():
        return normalized
    reserved = floor * int(below.sum())
    above_sum = normalized[~below].sum()
    remaining = 1.0 - reserved
    if remaining <= 0.0 or above_sum <= 0.0:
        # Demasiadas clases bajo el piso: reparte uniforme.
        return np.full(array.shape, 1.0 / array.size)
    out = normalized.copy()
    out[below] = floor
    out[~below] = normalized[~below] / above_sum * remaining
    return out


@dataclass(slots=True, frozen=True)
class ProbabilityCalibration:
    """Calibracion activa: una temperatura y un piso por clase."""

    temperature: float = 1.0
    class_floor: float = DEFAULT_CLASS_FLOOR

    @property
    def is_neutral(self) -> bool:
        return self.temperature == 1.0 and self.class_floor <= 0.0

    def calibrate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Calibra una matriz de marcadores (goles local x visitante). Coherente
        para 1X2 y marcadores porque se aplica antes de derivar ambos."""
        return temperature_scale(matrix, self.temperature)

    def calibrate_triplet(self, home: float, draw: float, away: float) -> tuple[float, float, float]:
        """Calibra y pisa un vector 1X2. Usar cuando solo se tienen las 3 clases."""
        scaled = temperature_scale(np.array([home, draw, away], dtype=float), self.temperature)
        floored = apply_class_floor(scaled, self.class_floor)
        return float(floored[0]), float(floored[1]), float(floored[2])


NEUTRAL_CALIBRATION = ProbabilityCalibration()


def load_active_calibration(path: str | Path | None = None) -> ProbabilityCalibration:
    """Carga la calibracion activa desde el artefacto JSON.

    Devuelve la calibracion NEUTRA si el archivo no existe o es ilegible: la
    inferencia nunca debe romperse por falta del artefacto.
    """
    resolved = Path(path or os.getenv(CALIBRATION_PATH_ENV) or DEFAULT_CALIBRATION_PATH)
    if not resolved.exists():
        return NEUTRAL_CALIBRATION
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return ProbabilityCalibration(
            temperature=float(payload.get("temperature", 1.0)),
            class_floor=float(payload.get("class_floor", DEFAULT_CLASS_FLOOR)),
        )
    except Exception as exc:  # noqa: BLE001 - artefacto corrupto no debe tumbar inferencia.
        LOGGER.warning("calibration_artifact_unreadable path=%s detail=%s", resolved, exc)
        return NEUTRAL_CALIBRATION


# --- Metricas de calibracion (para el arnes de evaluacion) -------------------

_OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def log_loss(triplets: np.ndarray, outcomes: list[str], *, epsilon: float = 1e-3) -> float:
    """Log loss multiclase con piso epsilon. Menor es mejor; uniforme ~1.0986."""
    probs = np.clip(np.asarray(triplets, dtype=float), epsilon, 1.0)
    losses = []
    for row, outcome in zip(probs, outcomes):
        idx = _OUTCOME_INDEX.get(outcome)
        if idx is None:
            continue
        losses.append(-np.log(row[idx] / row.sum()))
    return float(np.mean(losses)) if losses else 0.0


def expected_calibration_error(triplets: np.ndarray, outcomes: list[str], *, bins: int = 10) -> float:
    """ECE sobre la clase predicha (confianza = max prob). 0 = perfecto.

    Mide |confianza media - acierto real| por bin de confianza, ponderado.
    """
    probs = np.asarray(triplets, dtype=float)
    if probs.size == 0:
        return 0.0
    confidences = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    actual = np.array([_OUTCOME_INDEX.get(o, -1) for o in outcomes])
    correct = (predicted == actual).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(confidences)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(confidences[mask].mean() - correct[mask].mean())
    return float(ece)


def fit_temperature(
    triplets: np.ndarray,
    outcomes: list[str],
    *,
    grid: np.ndarray | None = None,
    refine_iterations: int = 40,
) -> float:
    """Ajusta la temperatura que minimiza log loss sobre las clases 1X2.

    Barrido en rejilla + refinamiento por seccion dorada. Sin dependencias de
    scipy para mantener la inferencia liviana.
    """
    probs = np.asarray(triplets, dtype=float)
    if probs.size == 0:
        return 1.0
    search = grid if grid is not None else np.linspace(0.5, 6.0, 23)

    def loss_at(temp: float) -> float:
        scaled = np.array([temperature_scale(row, temp) for row in probs])
        return log_loss(scaled, outcomes)

    best_temp = float(min(search, key=loss_at))

    # Refinamiento por seccion dorada alrededor del mejor punto de la rejilla.
    step = (search[1] - search[0]) if len(search) > 1 else 0.5
    low, high = max(MIN_TEMPERATURE, best_temp - step), best_temp + step
    phi = (np.sqrt(5) - 1) / 2
    c, d = high - phi * (high - low), low + phi * (high - low)
    for _ in range(refine_iterations):
        if loss_at(c) < loss_at(d):
            high = d
        else:
            low = c
        c, d = high - phi * (high - low), low + phi * (high - low)
    return float(round((low + high) / 2, 4))
