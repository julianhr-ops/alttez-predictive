"""Reference 1X2 probability models that any real model has to beat.

Every function returns an ``(n, 3)`` array of probabilities in H, D, A order, the
layout ``alttez.evaluation.paired.brier_1x2_per_game`` expects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alttez.features.elo import HOME_ADVANTAGE_ELO, EloEngine

OUTCOME_ORDER = ("H", "D", "A")
MAX_DRAW_PROB_AT_PARITY = 0.5


def _outcome_vector(outcomes: Sequence[str]) -> np.ndarray:
    actual = np.asarray(outcomes)
    if actual.ndim != 1 or not actual.size or not np.isin(actual, OUTCOME_ORDER).all():
        raise ValueError("outcomes must be a non-empty vector of H, D or A")
    return actual


def _row_count(n: int) -> int:
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    return int(n)


def uniform_probabilities(n: int) -> np.ndarray:
    return np.full((_row_count(n), 3), 1.0 / 3.0)


def outcome_frequencies(train_outcomes: Sequence[str]) -> np.ndarray:
    actual = _outcome_vector(train_outcomes)
    return np.array([(actual == label).mean() for label in OUTCOME_ORDER])


def base_rate_probabilities(train_outcomes: Sequence[str], n: int) -> np.ndarray:
    """The H/D/A frequencies of the training window, repeated for ``n`` matches."""

    return np.tile(outcome_frequencies(train_outcomes), (_row_count(n), 1))


def _effective_difference(elo_diff, neutral) -> np.ndarray:
    diff = np.asarray(elo_diff, dtype=float)
    if diff.ndim != 1 or not diff.size or not np.isfinite(diff).all():
        raise ValueError("elo_diff must be a non-empty finite vector")
    if neutral is None:
        return diff + HOME_ADVANTAGE_ELO
    neutral_flags = np.asarray(neutral, dtype=bool)
    if neutral_flags.shape != diff.shape:
        raise ValueError("neutral must match elo_diff in length")
    return diff + np.where(neutral_flags, 0.0, HOME_ADVANTAGE_ELO)


def _expected_score(effective_diff: np.ndarray) -> np.ndarray:
    return EloEngine().get_expected_score(effective_diff, 0.0)


def fit_draw_prob_at_parity(train_elo_diff, train_outcomes: Sequence[str], train_neutral=None) -> float:
    """Draw probability of two evenly matched sides, by moment matching.

    The Elo-only model draws with probability ``p0 * 4 * E * (1 - E)``. ``p0`` is the
    value that reproduces the training draw frequency on the training Elo gaps.
    """

    actual = _outcome_vector(train_outcomes)
    expected = _expected_score(_effective_difference(train_elo_diff, train_neutral))
    if len(expected) != len(actual):
        raise ValueError("train_elo_diff and train_outcomes must have the same length")
    shape = (4.0 * expected * (1.0 - expected)).mean()
    return float(min((actual == "D").mean() / shape, MAX_DRAW_PROB_AT_PARITY))


def elo_only_probabilities(elo_diff, *, draw_prob_at_parity: float, neutral=None) -> np.ndarray:
    """1X2 probabilities from the pre-match Elo gap (home minus away).

    The Elo expected score ``E`` is split into win and draw mass so that
    ``P(H) + P(D) / 2 == E``. ``neutral`` flags matches without home advantage; when
    omitted every match gets the home-advantage bonus.
    """

    if not 0.0 <= draw_prob_at_parity <= MAX_DRAW_PROB_AT_PARITY:
        raise ValueError(f"draw_prob_at_parity must be in [0, {MAX_DRAW_PROB_AT_PARITY}]")
    expected = _expected_score(_effective_difference(elo_diff, neutral))
    draw = draw_prob_at_parity * 4.0 * expected * (1.0 - expected)
    return np.column_stack((expected - draw / 2.0, draw, 1.0 - expected - draw / 2.0))


def baseline_probabilities(
    train_outcomes: Sequence[str],
    train_elo_diff,
    train_neutral,
    holdout_elo_diff,
    holdout_neutral,
) -> dict[str, np.ndarray]:
    """All three baselines for a holdout window, fitted on the training window only."""

    n = len(np.asarray(holdout_elo_diff))
    draw_prob = fit_draw_prob_at_parity(train_elo_diff, train_outcomes, train_neutral)
    return {
        "uniform": uniform_probabilities(n),
        "base_rate": base_rate_probabilities(train_outcomes, n),
        "elo_only": elo_only_probabilities(holdout_elo_diff, draw_prob_at_parity=draw_prob, neutral=holdout_neutral),
    }
