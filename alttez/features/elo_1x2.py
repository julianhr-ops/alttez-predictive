"""1X2 probabilities from the Elo gap with a fitted draw model, and a log-linear blend with another 1X2.

Everything here is a pure function of arrays or a frozen dataclass, so the walk-forward
backtest and live serving fit and apply exactly the same model.

Draw model. The home side's Elo gap (plus a fitted home advantage unless the venue is
neutral) is turned into a latent strength ``x = scale * ln(10) / 400 * gap``. Two links
turn ``x`` into H, D, A:

* ``ordinal``: cumulative logit with symmetric cut points, ``P(H) = sigmoid(x - c)``,
  ``P(A) = sigmoid(-x - c)``, ``P(D)`` the remainder. ``c`` is the draw width.
* ``davidson``: ``P(H), P(A), P(D)`` proportional to ``exp(x/2), exp(-x/2), nu``.

In both, the draw probability falls as the gap grows, which the one-parameter
``p0 * 4E(1-E)`` baseline only approximates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from alttez.evaluation.baselines import OUTCOME_ORDER
from alttez.features.elo import HOME_ADVANTAGE_ELO

LOGIT_PER_ELO = math.log(10.0) / 400.0
KINDS = ("ordinal", "davidson")
LOG_EPS = 1e-12

ELO_PER_UNIT = 100.0
_HOME_ADVANTAGE_BOUNDS = (-100.0, 400.0)
_DRAW_PARAM_BOUNDS = (1e-3, 5.0)
_SCALE_BOUNDS = (0.2, 3.0)


def _validated_inputs(elo_diff, neutral) -> tuple[np.ndarray, np.ndarray]:
    diff = np.asarray(elo_diff, dtype=float)
    flags = np.asarray(neutral, dtype=bool)
    if diff.ndim != 1 or not diff.size or not np.isfinite(diff).all():
        raise ValueError("elo_diff must be a non-empty finite vector")
    if flags.shape != diff.shape:
        raise ValueError("neutral must match elo_diff in length")
    return diff, flags


def _outcome_index(outcomes: Sequence[str]) -> np.ndarray:
    actual = np.asarray(outcomes)
    if actual.ndim != 1 or not actual.size or not np.isin(actual, OUTCOME_ORDER).all():
        raise ValueError("outcomes must be a non-empty vector of H, D or A")
    return np.array([OUTCOME_ORDER.index(o) for o in actual])


def _link(kind: str, latent: np.ndarray, draw_param: float) -> np.ndarray:
    if kind == "ordinal":
        home = expit(latent - draw_param)
        away = 1.0 - expit(latent + draw_param)
        return np.column_stack((home, 1.0 - home - away, away))
    if kind == "davidson":
        home, away = np.exp(latent / 2.0), np.exp(-latent / 2.0)
        total = home + away + draw_param
        return np.column_stack((home / total, draw_param / total, away / total))
    raise ValueError(f"kind must be one of {KINDS}")


@dataclass(frozen=True, slots=True)
class DrawModel:
    """Fitted Elo-gap to 1X2 map. ``draw_param`` is the ordinal cut point or the Davidson ``nu``."""

    kind: str
    home_advantage: float
    draw_param: float
    scale: float = 1.0

    def probabilities(self, elo_diff, neutral) -> np.ndarray:
        """``(n, 3)`` probabilities in H, D, A order; rows sum to 1."""
        diff, flags = _validated_inputs(elo_diff, neutral)
        latent = self.scale * LOGIT_PER_ELO * (diff + np.where(flags, 0.0, self.home_advantage))
        return _link(self.kind, latent, self.draw_param)


def fit_draw_model(kind: str, elo_diff, neutral, outcomes: Sequence[str], *, fit_scale: bool = False) -> DrawModel:
    """Maximum-likelihood ``DrawModel`` on the training rows only.

    Fits the home advantage and the draw parameter; ``fit_scale`` also fits a multiplier of
    the Elo logit scale (kept at 1 otherwise, i.e. Elo's own 400-point scale).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    diff, flags = _validated_inputs(elo_diff, neutral)
    index = _outcome_index(outcomes)
    if len(index) != len(diff):
        raise ValueError("outcomes must match elo_diff in length")
    rows = np.arange(len(diff))
    start = [HOME_ADVANTAGE_ELO / ELO_PER_UNIT, 0.5 if kind == "ordinal" else 1.0, 1.0]
    bounds = [(_HOME_ADVANTAGE_BOUNDS[0] / ELO_PER_UNIT, _HOME_ADVANTAGE_BOUNDS[1] / ELO_PER_UNIT), _DRAW_PARAM_BOUNDS, _SCALE_BOUNDS]
    free = 3 if fit_scale else 2

    def unpack(theta: np.ndarray) -> DrawModel:
        return DrawModel(kind, float(theta[0]) * ELO_PER_UNIT, float(theta[1]), float(theta[2]) if fit_scale else 1.0)

    def loss(theta: np.ndarray) -> float:
        probs = unpack(theta).probabilities(diff, flags)
        return float(-np.log(np.clip(probs[rows, index], LOG_EPS, 1.0)).mean())

    result = minimize(loss, start[:free], bounds=bounds[:free], method="L-BFGS-B", options={"ftol": 1e-13, "gtol": 1e-9})
    return unpack(result.x)


def log_linear_blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """Probabilities proportional to ``first**weight * second**(1 - weight)``, row-normalised."""
    first_p, second_p = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first_p.shape != second_p.shape or first_p.ndim != 2 or first_p.shape[1] != 3:
        raise ValueError("both inputs must be (n, 3) probability arrays of the same shape")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    logits = weight * np.log(np.clip(first_p, LOG_EPS, 1.0)) + (1.0 - weight) * np.log(np.clip(second_p, LOG_EPS, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    mass = np.exp(logits)
    return mass / mass.sum(axis=1, keepdims=True)


def fit_blend_weight(first: np.ndarray, second: np.ndarray, outcomes: Sequence[str]) -> float:
    """Weight on ``first`` in [0, 1] that minimises log loss of the log-linear blend on the training rows."""
    index = _outcome_index(outcomes)
    if len(index) != len(first):
        raise ValueError("outcomes must match the probability rows in length")
    rows = np.arange(len(index))

    def loss(weight: float) -> float:
        probs = log_linear_blend(first, second, weight)
        return float(-np.log(np.clip(probs[rows, index], LOG_EPS, 1.0)).mean())

    return float(minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6}).x)
