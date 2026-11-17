"""Candidate 1X2 serving rules for the soccer backtest and the choice among them.

Every candidate is an Elo-gap draw model (``alttez.features.elo_1x2``), optionally blended
log-linearly with the raw Poisson 1X2. A rule is fitted on training rows only; the backtest
calls :func:`fit_candidate` once per test year on the preceding window.

Selection is declared up front and pure: :func:`select_candidate` takes scored rows of the
SELECTION years only and picks the lowest pooled Brier, then the simplest candidate whose
paired Brier difference against that best one has a CI containing 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alttez.evaluation.paired import PairedDiff, brier_1x2_per_game, paired_bootstrap_diff
from alttez.features.elo_1x2 import KINDS, DrawModel, fit_blend_weight, fit_draw_model, log_linear_blend
from alttez.inference.calibration import matrix_1x2, rescale_matrix_to_triplet

CLASSES = ("H", "D", "A")
# Years up to this one choose the candidate; later years are the untouched final test.
SELECTION_LAST_YEAR = 2022
# Declared from the 2018-2022 selection run (docs/validation_soccer.md), before 2023-2025 was scored.
PRIMARY_CANDIDATE = "elo_ordinal"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    kind: str
    fit_scale: bool
    blend: bool

    @property
    def name(self) -> str:
        return f"{'blend' if self.blend else 'elo'}_{self.kind}{'_scaled' if self.fit_scale else ''}"

    @property
    def label(self) -> str:
        link = "ordinal" if self.kind == "ordinal" else "Davidson"
        base = f"Elo {link} draw" + (", fitted scale" if self.fit_scale else "")
        return f"{base} + Poisson blend" if self.blend else base

    @property
    def complexity(self) -> int:
        """Fitted parameters beyond the Elo ratings: home advantage, draw width, scale, blend weight."""
        return 2 + int(self.fit_scale) + int(self.blend)


CANDIDATES = tuple(
    CandidateSpec(kind, fit_scale, blend) for blend in (False, True) for fit_scale in (False, True) for kind in KINDS
)
CANDIDATE_NAMES = tuple(spec.name for spec in CANDIDATES)


@dataclass(frozen=True, slots=True)
class FittedCandidate:
    spec: CandidateSpec
    draw_model: DrawModel
    blend_weight: float | None

    def predict(self, elo_diff, neutral, poisson_probs: np.ndarray | None = None) -> np.ndarray:
        """``(n, 3)`` H, D, A probabilities. A blend candidate needs ``poisson_probs`` and is NaN where they are."""
        elo_probs = self.draw_model.probabilities(elo_diff, neutral)
        if self.blend_weight is None:
            return elo_probs
        if poisson_probs is None:
            raise ValueError(f"{self.spec.name} blends the Poisson 1X2, so poisson_probs is required")
        out = np.full(elo_probs.shape, np.nan)
        usable = np.isfinite(poisson_probs).all(axis=1)
        out[usable] = log_linear_blend(elo_probs[usable], poisson_probs[usable], self.blend_weight)
        return out


def fit_candidate(spec: CandidateSpec, train: pd.DataFrame, poisson_train: pd.DataFrame | None = None) -> FittedCandidate:
    """Fit ``spec`` on ``train`` (``elo_diff``, ``neutral``, ``outcome``) and, for a blend, on ``poisson_train``.

    ``poisson_train`` holds the walk-forward raw Poisson 1X2 of supported training matches
    (``poisson_raw_H/D/A`` plus the same three columns as ``train``). The blend weight is
    fitted with the draw model already fitted on ``train``.
    """
    draw_model = fit_draw_model(
        spec.kind, train["elo_diff"].to_numpy(float), train["neutral"].to_numpy(bool), train["outcome"].tolist(), fit_scale=spec.fit_scale
    )
    if not spec.blend:
        return FittedCandidate(spec, draw_model, None)
    if poisson_train is None or poisson_train.empty:
        raise ValueError(f"{spec.name} needs training rows with a raw Poisson 1X2 to fit its blend weight")
    elo_probs = draw_model.probabilities(poisson_train["elo_diff"].to_numpy(float), poisson_train["neutral"].to_numpy(bool))
    poisson_probs = poisson_train[[f"poisson_raw_{c}" for c in CLASSES]].to_numpy(float)
    weight = fit_blend_weight(elo_probs, poisson_probs, poisson_train["outcome"].tolist())
    return FittedCandidate(spec, draw_model, weight)


@dataclass(frozen=True, slots=True)
class Selection:
    best: str
    chosen: str
    pooled_brier: dict[str, float]
    versus_best: dict[str, PairedDiff]


def select_candidate(frame: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> Selection:
    """Choose among the candidates using ``frame`` (selection-year, scored rows) alone.

    ``best`` has the lowest pooled Brier. ``chosen`` is the lowest-complexity candidate whose
    paired Brier difference to ``best`` is not significantly positive (``ci_low <= 0``); ties
    on complexity go to the lower Brier. Blocks are match dates.
    """
    outcomes = frame["outcome"].tolist()
    losses = {name: brier_1x2_per_game(frame[[f"{name}_{c}" for c in CLASSES]].to_numpy(float), outcomes) for name in CANDIDATE_NAMES}
    pooled = {name: float(values.mean()) for name, values in losses.items()}
    best = min(pooled, key=pooled.__getitem__)
    blocks = frame["date"].to_numpy()
    versus = {
        name: paired_bootstrap_diff(losses[name], losses[best], n_boot=n_boot, seed=seed, block_ids=blocks) for name in CANDIDATE_NAMES if name != best
    }
    plausible = [best] + [name for name, diff in versus.items() if diff.ci_low <= 0.0]
    complexity = {spec.name: spec.complexity for spec in CANDIDATES}
    chosen = min(plausible, key=lambda name: (complexity[name], pooled[name]))
    return Selection(best=best, chosen=chosen, pooled_brier=pooled, versus_best=versus)


@dataclass(frozen=True, slots=True)
class ServedScorelines:
    """Per match: -log of the served matrix cell of the played score, and how far its 1X2 is from the target."""

    log_loss: np.ndarray
    marginal_gap: np.ndarray


def _cell_log_loss(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    size = matrix.shape[0]
    if home_goals >= size or away_goals >= size:
        return np.nan
    return float(-np.log(max(matrix[home_goals, away_goals], 1e-12)))


def scoreline_log_loss(matrices: np.ndarray, usable: np.ndarray, home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """Log loss of the played scoreline under each (normalised) matrix; NaN if unusable or beyond the matrix."""
    out = np.full(len(matrices), np.nan)
    for i in np.flatnonzero(usable):
        out[i] = _cell_log_loss(matrices[i] / matrices[i].sum(), int(home_goals[i]), int(away_goals[i]))
    return out


def served_scoreline(
    matrices: np.ndarray, usable: np.ndarray, targets: np.ndarray, home_goals: np.ndarray, away_goals: np.ndarray
) -> ServedScorelines:
    """Rescale every usable matrix to its target 1X2 as serving does and score the result."""
    loss = np.full(len(matrices), np.nan)
    gap = np.full(len(matrices), np.nan)
    for i in np.flatnonzero(usable):
        served = rescale_matrix_to_triplet(matrices[i], tuple(targets[i]))
        loss[i] = _cell_log_loss(served, int(home_goals[i]), int(away_goals[i]))
        gap[i] = float(np.abs(np.array(matrix_1x2(served)) - targets[i]).max())
    return ServedScorelines(log_loss=loss, marginal_gap=gap)
