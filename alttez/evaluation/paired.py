"""Paired uncertainty estimates and per-game probability metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PairedDiff:
    mean_diff: float
    ci_low: float
    ci_high: float
    n: int
    a_better: bool
    b_better: bool


def _vector(values, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _binary(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = _vector(p, "probabilities")
    y = _vector(y, "outcomes")
    if len(p) != len(y) or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError("probabilities and binary outcomes must be valid and equally sized")
    return p, y


def paired_bootstrap_diff(loss_a, loss_b, *, n_boot=2000, seed=0, alpha=0.05, block_ids=None) -> PairedDiff:
    """Bootstrap the mean paired loss difference, resampling rows or whole blocks."""
    a, b = _vector(loss_a, "loss_a"), _vector(loss_b, "loss_b")
    if len(a) != len(b):
        raise ValueError("loss vectors must have equal length")
    if not isinstance(n_boot, Integral) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not np.isscalar(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    diff = a - b
    rng = np.random.default_rng(seed)
    if block_ids is None:
        samples = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    else:
        ids = np.asarray(block_ids)
        if ids.ndim != 1 or len(ids) != len(diff) or pd.isna(ids).any():
            raise ValueError("block_ids must be a non-missing vector matching the losses")
        try:
            codes, unique = pd.factorize(ids, sort=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("block_ids must contain valid block labels") from exc
        sums = np.bincount(codes, weights=diff, minlength=len(unique))
        counts = np.bincount(codes, minlength=len(unique))
        selected = rng.integers(0, len(unique), size=(n_boot, len(unique)))
        samples = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return PairedDiff(float(diff.mean()), float(low), float(high), len(diff), bool(high < 0), bool(low > 0))


def brier_binary_per_game(p_home, y_home) -> np.ndarray:
    p, y = _binary(p_home, y_home)
    return (p - y) ** 2


def log_loss_binary_per_game(p_home, y_home, *, eps=1e-12) -> np.ndarray:
    p, y = _binary(p_home, y_home)
    if not np.isscalar(eps) or not 0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5")
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def brier_1x2_per_game(probs_hda: np.ndarray, outcomes: Sequence[str]) -> np.ndarray:
    try:
        probs = np.asarray(probs_hda, dtype=float)
        actual = np.asarray(outcomes)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid 1x2 inputs") from exc
    if (probs.ndim != 2 or probs.shape[1] != 3 or not len(probs)
            or actual.ndim != 1 or len(actual) != len(probs)
            or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1))
            or not np.isin(actual, ["H", "D", "A"]).all()):
        raise ValueError("invalid 1x2 probabilities or outcomes")
    target = actual[:, None] == np.array(["H", "D", "A"])
    return np.sum((probs - target) ** 2, axis=1)


def reliability_table(p, y, *, n_bins=10) -> pd.DataFrame:
    """Return equal-width bins, including empty bins with NaN summaries."""
    p, y = _binary(p, y)
    if not isinstance(n_bins, Integral) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    n = np.bincount(bins, minlength=n_bins)
    mean_p = np.full(n_bins, np.nan)
    freq = np.full(n_bins, np.nan)
    np.divide(np.bincount(bins, weights=p, minlength=n_bins), n, out=mean_p, where=n > 0)
    np.divide(np.bincount(bins, weights=y, minlength=n_bins), n, out=freq, where=n > 0)
    return pd.DataFrame({"bin": np.arange(n_bins), "n": n, "mean_p": mean_p, "freq": freq})


def expected_calibration_error(p, y, *, n_bins=10) -> float:
    table = reliability_table(p, y, n_bins=n_bins)
    populated = table[table["n"] > 0]
    return float((populated["n"] * (populated["mean_p"] - populated["freq"]).abs()).sum() / populated["n"].sum())
