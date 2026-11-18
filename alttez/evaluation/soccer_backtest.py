"""Walk-forward backtest of the soccer 1X2 stack against baselines, with paired uncertainty.

Data: the CC0 ``martj42/international_results`` CSV, parsed with the same loader the
historical seed uses. For every test year ``Y``:

* Elo ratings (``alttez.features.elo`` rules, fixed ``tournament_k_factor``) are updated
  with matches strictly before each match date, so they cannot see the match itself or
  anything later.
* The production Poisson champion (``DEFAULT_POISSON_KWARGS``, refit from matches before
  each block start, Dixon-Coles ``rho`` fitted by the production procedure on years
  before ``Y`` only) produces the raw score matrix and 1X2.
* The temperature is fitted on the RAW 1X2 outputs of the preceding ``window`` years,
  each of which was itself predicted walk-forward, and applied through the class-ratio
  matrix transform of ``ProbabilityCalibration.calibrate_matrix``.
* Baselines (uniform, base rate, Elo-only) are fitted on the same preceding window.

Everything scored on a test year is scored on identical matches: those the production
router would serve (both teams with enough recent history).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import pickle
import tempfile
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alttez.evaluation.baselines import baseline_probabilities, fit_draw_prob_at_parity
from alttez.evaluation.metrics import ranked_probability_score
from alttez.evaluation.paired import (
    PairedDiff,
    brier_1x2_per_game,
    expected_calibration_error,
    paired_bootstrap_diff,
    reliability_table,
)
from alttez.evaluation.soccer_candidates import (
    CANDIDATE_NAMES,
    CANDIDATES,
    SELECTION_LAST_YEAR,
    fit_candidate,
    scoreline_log_loss,
    select_candidate,
    served_scoreline,
)
from alttez.experiments.tournament_oracle import build_probability_matrix
from alttez.features.elo import HOME_ADVANTAGE_ELO, EloEngine, rate_matches
from alttez.features.poisson import PoissonEngine
from alttez.inference.calibration import ProbabilityCalibration, fit_temperature, matrix_1x2
from alttez.inference.router import DEFAULT_MAX_GOALS, DEFAULT_POISSON_KWARGS
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.seed_historical import load_historical_dataframe

DATA_URL = InternationalLoader.DATA_URL
DATA_LICENSE = "CC0 1.0 (github.com/martj42/international_results)"
CACHE_FILENAME = "results.csv"
REQUIRED_COLUMNS = ("date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral")
USER_AGENT = "ALTTEZ soccer validation"

FIRST_TEST_YEAR = 2018
WINDOW_YEARS = 4
REFIT_DAYS = 7
MIN_SUPPORTED_MATCHES = DEFAULT_POISSON_KWARGS["min_history_matches"]

CLASSES = ("H", "D", "A")
BASE_MODELS = ("uniform", "base_rate", "elo_only", "poisson_raw", "poisson_calibrated")
MODELS = (*BASE_MODELS, *CANDIDATE_NAMES)
LABELS = {
    "uniform": "uniform",
    "base_rate": "base rate",
    "elo_only": "Elo only",
    "poisson_raw": "Poisson raw",
    "poisson_calibrated": "Poisson + temperature",
    **{spec.name: spec.label for spec in CANDIDATES},
}
COMPARISONS = (
    ("poisson_raw", "uniform"),
    ("poisson_raw", "base_rate"),
    ("poisson_raw", "elo_only"),
    ("poisson_calibrated", "uniform"),
    ("poisson_calibrated", "base_rate"),
    ("poisson_calibrated", "elo_only"),
    ("poisson_calibrated", "poisson_raw"),
    ("elo_only", "base_rate"),
)
PRIMARY_METRIC = "brier"
METRICS = ("brier", "rps", "log_loss")
LOG_LOSS_EPS = 1e-12


class SoccerDataError(RuntimeError):
    """The results CSV could not be downloaded or is not usable."""


def _read_url(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SoccerDataError(f"download failed for {url}: {exc}") from exc
    if body[:2] == b"\x1f\x8b":
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise SoccerDataError(f"bad gzip body from {url}: {exc}") from exc
    return body


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def validate_results_csv(body: bytes) -> int:
    """Number of rows of a usable results CSV; raises :class:`SoccerDataError` otherwise."""
    try:
        frame = pd.read_csv(io.BytesIO(body))
    except Exception as exc:
        raise SoccerDataError(f"results body is not a readable CSV: {exc}") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SoccerDataError(f"results CSV lacks columns {missing}")
    if frame.empty:
        raise SoccerDataError("results CSV has no rows")
    return len(frame)


def load_results_csv(
    cache_dir: Path,
    *,
    refresh: bool = False,
    url: str = DATA_URL,
    timeout: float = 60.0,
    fetch: Callable[[str, float], bytes] = _read_url,
) -> Path:
    """Path of the cached results CSV, downloading it first when absent or ``refresh``.

    The body is validated before it replaces the cache, and a failed download raises
    instead of falling back to anything else.
    """
    cache_path = Path(cache_dir) / CACHE_FILENAME
    if cache_path.exists() and not refresh:
        return cache_path
    body = fetch(url, timeout)
    validate_results_csv(body)
    _atomic_write(cache_path, body)
    return cache_path


@dataclass(frozen=True, slots=True)
class Matches:
    """Scored matches in date order, plus what the loader excluded."""

    frame: pd.DataFrame
    n_rows: int
    n_without_score: int


def prepare_matches(csv_path: Path) -> Matches:
    """Parse the CSV with the seed loader (rows without scores dropped) and order by date."""
    n_rows = len(pd.read_csv(csv_path, usecols=["date"]))
    frame = load_historical_dataframe(Path(csv_path))
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    frame["year"] = frame["date"].dt.year
    frame["outcome"] = np.where(frame["home_score"] > frame["away_score"], "H", np.where(frame["home_score"] < frame["away_score"], "A", "D"))
    return Matches(frame=frame, n_rows=n_rows, n_without_score=n_rows - len(frame))


def add_pre_match_elo(matches: pd.DataFrame, *, engine: EloEngine | None = None) -> pd.DataFrame:
    """Add ``elo_home_before``, ``elo_away_before`` and ``elo_diff`` using only earlier dates.

    ``matches`` must be sorted by date; the walk is :func:`alttez.features.elo.rate_matches`.
    """
    run = rate_matches(matches, engine=engine)
    out = matches.copy()
    out["elo_home_before"] = run.home_before
    out["elo_away_before"] = run.away_before
    out["elo_diff"] = run.home_before - run.away_before
    return out


def fit_training_rho(train: pd.DataFrame, *, min_matches: int = MIN_SUPPORTED_MATCHES) -> float:
    """Dixon-Coles rho by the production procedure (``scripts/fit_dixon_coles_rho.py``) on ``train``."""
    from scripts.fit_dixon_coles_rho import _build_samples, _build_team_rates, fit_rho

    rates, global_avg = _build_team_rates(train, min_matches=min_matches)
    samples = _build_samples(train, rates, global_avg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fit_rho(samples)


def is_supported(profile_home: Mapping[str, float], profile_away: Mapping[str, float], *, threshold: int = MIN_SUPPORTED_MATCHES) -> bool:
    """The router's gate: both teams need ``threshold`` recent matches and recency-weighted support."""
    if min(profile_home["matches"], profile_away["matches"]) < threshold:
        return False
    return min(profile_home["effective_matches"], profile_away["effective_matches"]) > 0


@dataclass(frozen=True, slots=True)
class RawYear:
    """Raw Poisson output of one predicted year: per-match lambdas, supported flag, and score matrices."""

    year: int
    rho: float
    train_end: pd.Timestamp
    frame: pd.DataFrame
    matrices: np.ndarray


def raw_poisson_year(
    matches: pd.DataFrame,
    year: int,
    *,
    refit_days: int = REFIT_DAYS,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> RawYear:
    """Raw production-configured Poisson predictions for every match of ``year``.

    ``rho`` is fitted on matches before January 1 of ``year``. The engine is refit at the
    first match date of every ``refit_days`` block, from matches strictly before it.
    """
    year_start = pd.Timestamp(year=year, month=1, day=1)
    train = matches[matches["date"] < year_start]
    target = matches[matches["year"] == year].reset_index(drop=True)
    if train.empty or target.empty:
        raise ValueError(f"year {year} needs earlier history and its own matches")
    rho = fit_training_rho(train)

    block = ((target["date"] - year_start).dt.days // refit_days).to_numpy()
    lambdas = np.full((len(target), 2), np.nan)
    supported = np.zeros(len(target), dtype=bool)
    matrices = np.full((len(target), max_goals + 1, max_goals + 1), np.nan)
    home = target["home_team"].to_numpy()
    away = target["away_team"].to_numpy()
    neutral = target["neutral"].to_numpy(bool)
    for block_id in np.unique(block):
        rows = np.flatnonzero(block == block_id)
        cutoff = target["date"].iloc[rows[0]]
        engine = PoissonEngine(**DEFAULT_POISSON_KWARGS)
        engine.fit(matches[matches["date"] < cutoff], target_date=cutoff.date().isoformat())
        for i in rows:
            if not is_supported(engine.get_team_profile(home[i]), engine.get_team_profile(away[i])):
                continue
            lambda_home, lambda_away = engine.predict_xg(home[i], away[i], is_neutral=bool(neutral[i]))
            lambdas[i] = (lambda_home, lambda_away)
            supported[i] = True
            matrices[i] = build_probability_matrix(lambda_home, lambda_away, max_goals=max_goals, rho=rho)

    frame = target[["date", "year", "home_team", "away_team", "tournament", "neutral", "outcome"]].copy()
    frame["supported"] = supported
    frame["lambda_home"], frame["lambda_away"] = lambdas[:, 0], lambdas[:, 1]
    raw_1x2 = np.full((len(target), 3), np.nan)
    for i in np.flatnonzero(supported):
        raw_1x2[i] = matrix_1x2(matrices[i])
    frame = _with_probabilities(frame, "poisson_raw", raw_1x2)
    return RawYear(year, rho, train["date"].max(), frame, matrices)


def _with_probabilities(frame: pd.DataFrame, model: str, probabilities: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    for column, cls in enumerate(CLASSES):
        out[f"{model}_{cls}"] = probabilities[:, column]
    return out


def model_probabilities(frame: pd.DataFrame, model: str) -> np.ndarray:
    return frame[[f"{model}_{cls}" for cls in CLASSES]].to_numpy(float)


@dataclass(frozen=True, slots=True)
class FoldInfo:
    year: int
    rho: float
    temperature: float
    train_end: str
    window_start: str
    calibration_years: list[int]
    n_calibration_matches: int
    base_rate: dict[str, float]
    draw_prob_at_parity: float
    n_matches: int
    n_supported: int
    candidates: dict[str, dict[str, float | None]]


def _window(matches: pd.DataFrame, year: int, window_years: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year - window_years, month=1, day=1)
    end = pd.Timestamp(year=year, month=1, day=1)
    return matches[(matches["date"] >= start) & (matches["date"] < end)]


def fit_calibration(raw_years: Mapping[int, RawYear], year: int, window_years: int) -> tuple[float, list[int], int]:
    """Temperature fitted on the raw 1X2 of the ``window_years`` years before ``year`` only."""
    years = sorted(y for y in raw_years if year - window_years <= y < year)
    if not years:
        raise ValueError(f"no raw predictions before {year} to fit the temperature on")
    train = pd.concat([raw_years[y].frame for y in years], ignore_index=True)
    train = train[train["supported"]]
    temperature = fit_temperature(model_probabilities(train, "poisson_raw"), train["outcome"].tolist())
    return temperature, years, len(train)


def calibrate_matrices(matrices: np.ndarray, supported: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated 1X2 per match through the class-ratio matrix transform; NaN where unsupported."""
    calibration = ProbabilityCalibration(temperature=temperature)
    out = np.full((len(matrices), 3), np.nan)
    for i in np.flatnonzero(supported):
        out[i] = matrix_1x2(calibration.calibrate_matrix(matrices[i]))
    return out


def walk_forward(
    matches: pd.DataFrame,
    test_years: Sequence[int],
    *,
    window_years: int = WINDOW_YEARS,
    refit_days: int = REFIT_DAYS,
    map_fn: Callable[..., Any] = map,
    raw_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[FoldInfo]]:
    """Predictions of every model for ``test_years`` and the per-fold fitted parameters.

    ``matches`` must carry the columns of :func:`add_pre_match_elo`. Raw Poisson output is
    produced for ``window_years`` years before the first test year as well, so the
    temperature of every test year is fitted on earlier raw output only. ``map_fn`` may be
    a pool's ``map`` to run the years in parallel.
    """
    test_years = sorted(test_years)
    predicted = list(range(test_years[0] - window_years, test_years[-1] + 1))
    raw_by_year = {}
    cut = [matches[matches["year"] <= y] for y in predicted]
    jobs = [(subset, y, raw_cache_dir) for subset, y in zip(cut, predicted, strict=True)]
    results = map_fn(_raw_job, jobs, [refit_days] * len(jobs))
    for raw in results:
        raw_by_year[raw.year] = raw

    frames, folds = [], []
    for year in test_years:
        raw = raw_by_year[year]
        temperature, cal_years, n_cal = fit_calibration(raw_by_year, year, window_years)
        train = _window(matches, year, window_years)
        target = matches[matches["year"] == year].reset_index(drop=True)
        base = baseline_probabilities(
            train["outcome"].tolist(),
            train["elo_diff"].to_numpy(float),
            train["neutral"].to_numpy(bool),
            target["elo_diff"].to_numpy(float),
            target["neutral"].to_numpy(bool),
        )
        frame = raw.frame.copy()
        frame["home_score"] = target["home_score"].to_numpy(int)
        frame["away_score"] = target["away_score"].to_numpy(int)
        for name, probs in base.items():
            frame = _with_probabilities(frame, name, probs)
        supported = frame["supported"].to_numpy(bool)
        frame = _with_probabilities(frame, "poisson_calibrated", calibrate_matrices(raw.matrices, supported, temperature))
        frame["elo_diff"] = target["elo_diff"].to_numpy(float)
        poisson_train = _poisson_training_rows(raw_by_year, matches, cal_years)
        candidate_fits, candidate_columns = _candidate_predictions(train, poisson_train, frame, raw.matrices)
        frame = pd.concat([frame, candidate_columns], axis=1)
        frames.append(frame)
        draw_at_parity = fit_draw_prob_at_parity(train["elo_diff"].to_numpy(float), train["outcome"].tolist(), train["neutral"].to_numpy(bool))
        folds.append(
            FoldInfo(
                year=year,
                rho=raw.rho,
                temperature=temperature,
                train_end=raw.train_end.date().isoformat(),
                window_start=train["date"].min().date().isoformat(),
                calibration_years=cal_years,
                n_calibration_matches=n_cal,
                base_rate=dict(zip(CLASSES, base["base_rate"][0].tolist(), strict=True)),
                draw_prob_at_parity=draw_at_parity,
                n_matches=len(frame),
                n_supported=int(frame["supported"].sum()),
                candidates=candidate_fits,
            )
        )
    return pd.concat(frames, ignore_index=True), folds


def _poisson_training_rows(raw_by_year: Mapping[int, RawYear], matches: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """Supported rows of the walk-forward raw Poisson output of ``years`` with their Elo gap, for the blend weight."""
    parts = []
    for y in years:
        part = raw_by_year[y].frame.copy()
        part["elo_diff"] = matches.loc[matches["year"] == y, "elo_diff"].to_numpy(float)
        parts.append(part[part["supported"]])
    return pd.concat(parts, ignore_index=True)


def _candidate_predictions(
    train: pd.DataFrame, poisson_train: pd.DataFrame, frame: pd.DataFrame, matrices: np.ndarray
) -> tuple[dict[str, dict[str, float | None]], pd.DataFrame]:
    """Fit every candidate on the training window and predict the test year, with scoreline checks."""
    supported = frame["supported"].to_numpy(bool)
    poisson_probs = model_probabilities(frame, "poisson_raw")
    columns: dict[str, np.ndarray] = {}
    fits: dict[str, dict[str, float | None]] = {}
    raw_scoreline = scoreline_log_loss(matrices, supported, frame["home_score"].to_numpy(int), frame["away_score"].to_numpy(int))
    columns["scoreline_ll_poisson_raw"] = raw_scoreline
    for spec in CANDIDATES:
        fitted = fit_candidate(spec, train, poisson_train)
        probs = fitted.predict(frame["elo_diff"].to_numpy(float), frame["neutral"].to_numpy(bool), poisson_probs)
        for column, cls in enumerate(CLASSES):
            columns[f"{spec.name}_{cls}"] = probs[:, column]
        served = served_scoreline(matrices, supported & np.isfinite(probs).all(axis=1), probs, frame["home_score"].to_numpy(int), frame["away_score"].to_numpy(int))
        columns[f"scoreline_ll_{spec.name}"], columns[f"marginal_gap_{spec.name}"] = served.log_loss, served.marginal_gap
        fits[spec.name] = {
            "home_advantage": fitted.draw_model.home_advantage,
            "draw_param": fitted.draw_model.draw_param,
            "scale": fitted.draw_model.scale,
            "blend_weight": fitted.blend_weight,
        }
    return fits, pd.DataFrame(columns, index=frame.index)


def _raw_cache_path(cache_dir: Path, subset: pd.DataFrame, year: int, refit_days: int) -> Path:
    """Cache file keyed by everything the raw output depends on, so a stale file is never reused."""
    digest = hashlib.sha256(
        repr(
            (
                year,
                refit_days,
                DEFAULT_MAX_GOALS,
                sorted(DEFAULT_POISSON_KWARGS.items()),
                len(subset),
                str(subset["date"].max()),
                int(subset["home_score"].sum()),
                int(subset["away_score"].sum()),
            )
        ).encode()
    ).hexdigest()[:16]
    return Path(cache_dir) / f"raw_poisson_{year}_refit{refit_days}_{digest}.pkl"


def _raw_job(job: tuple[pd.DataFrame, int, Path | None], refit_days: int) -> RawYear:
    subset, year, cache_dir = job
    if cache_dir is None:
        return raw_poisson_year(subset, year, refit_days=refit_days)
    path = _raw_cache_path(cache_dir, subset, year, refit_days)
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    raw = raw_poisson_year(subset, year, refit_days=refit_days)
    _atomic_write(path, pickle.dumps(raw))
    return raw


def log_loss_1x2_per_game(probs_hda: np.ndarray, outcomes: Sequence[str], *, eps: float = LOG_LOSS_EPS) -> np.ndarray:
    probs = np.clip(np.asarray(probs_hda, dtype=float), eps, 1.0)
    index = np.array([CLASSES.index(o) for o in outcomes])
    return -np.log(probs[np.arange(len(probs)), index])


def rps_1x2_per_game(probs_hda: np.ndarray, outcomes: Sequence[str]) -> np.ndarray:
    return np.array([ranked_probability_score(p[0], p[1], p[2], o) for p, o in zip(probs_hda, outcomes, strict=True)])


def per_game_losses(probs_hda: np.ndarray, outcomes: Sequence[str]) -> dict[str, np.ndarray]:
    return {
        "brier": brier_1x2_per_game(probs_hda, outcomes),
        "rps": rps_1x2_per_game(probs_hda, outcomes),
        "log_loss": log_loss_1x2_per_game(probs_hda, outcomes),
    }


def class_calibration(probs_hda: np.ndarray, outcomes: Sequence[str], *, n_bins: int = 10) -> dict[str, Any]:
    """One-vs-rest reliability table and ECE for each of H, D, A."""
    actual = np.asarray(outcomes)
    out: dict[str, Any] = {}
    for column, cls in enumerate(CLASSES):
        p, y = probs_hda[:, column], (actual == cls).astype(int)
        out[cls] = {
            "ece": expected_calibration_error(p, y, n_bins=n_bins),
            "mean_predicted": float(p.mean()),
            "frequency": float(y.mean()),
            "reliability": reliability_table(p, y, n_bins=n_bins).round(4).to_dict("records"),
        }
    return out


def point_metrics(probs_hda: np.ndarray, outcomes: Sequence[str]) -> dict[str, Any]:
    losses = per_game_losses(probs_hda, outcomes)
    predicted = np.asarray(CLASSES)[probs_hda.argmax(axis=1)]
    calibration = class_calibration(probs_hda, outcomes)
    return {
        "n": len(probs_hda),
        **{name: float(values.mean()) for name, values in losses.items()},
        "accuracy": float((predicted == np.asarray(outcomes)).mean()),
        "ece": {cls: calibration[cls]["ece"] for cls in CLASSES},
        "calibration": calibration,
    }


def verdict(diff: PairedDiff) -> str:
    if diff.a_better:
        return "model better"
    if diff.b_better:
        return "baseline better"
    return "no significant difference"


def compare(frame: pd.DataFrame, model: str, baseline: str, *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    outcomes = frame["outcome"].tolist()
    losses_a = per_game_losses(model_probabilities(frame, model), outcomes)
    losses_b = per_game_losses(model_probabilities(frame, baseline), outcomes)
    blocks = frame["date"].to_numpy()
    rows = []
    for metric in METRICS:
        diff = paired_bootstrap_diff(losses_a[metric], losses_b[metric], n_boot=n_boot, seed=seed, block_ids=blocks)
        rows.append({"model": model, "baseline": baseline, "metric": metric, **asdict(diff), "verdict": verdict(diff)})
    return rows


def _scoreline_report(frame: pd.DataFrame, models: Sequence[str], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Scoreline log loss of the raw Poisson matrix and of each reported candidate's rescaled matrix, on identical rows."""
    columns = {m: f"scoreline_ll_{m}" for m in ["poisson_raw", *[c for c in models if c in CANDIDATE_NAMES]]}
    rows = frame[list(columns.values())].notna().all(axis=1).to_numpy()
    subset = frame[rows]
    blocks = subset["date"].to_numpy()
    raw_loss = subset[columns["poisson_raw"]].to_numpy(float)
    out: dict[str, Any] = {}
    for model, column in columns.items():
        loss = subset[column].to_numpy(float)
        gap = subset[f"marginal_gap_{model}"].max() if model != "poisson_raw" else None
        vs_raw = None
        if model != "poisson_raw":
            vs_raw = asdict(paired_bootstrap_diff(loss, raw_loss, n_boot=n_boot, seed=seed, block_ids=blocks))
        out[model] = {"n": int(rows.sum()), "mean_log_loss": float(loss.mean()), "max_marginal_gap": None if gap is None else float(gap), "vs_raw": vs_raw}
    return out


def scope_report(frame: pd.DataFrame, *, models: Sequence[str], comparisons: Sequence[tuple[str, str]], n_boot: int, seed: int) -> dict[str, Any]:
    outcomes = frame["outcome"].tolist()
    return {
        "n": len(frame),
        "metrics": {m: point_metrics(model_probabilities(frame, m), outcomes) for m in models},
        "comparisons": [row for a, b in comparisons for row in compare(frame, a, b, n_boot=n_boot, seed=seed)],
        "scorelines": _scoreline_report(frame, models, n_boot=n_boot, seed=seed),
    }


def reported_models(candidates: Sequence[str]) -> tuple[str, ...]:
    return (*BASE_MODELS, *candidates)


def reported_comparisons(candidates: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return (*COMPARISONS, *[(c, b) for c in candidates for b in ("poisson_raw", "elo_only", "base_rate", "uniform")])


def run_backtest(
    matches: Matches,
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    window_years: int = WINDOW_YEARS,
    refit_days: int = REFIT_DAYS,
    n_boot: int = 2000,
    seed: int = 0,
    map_fn: Callable[..., Any] = map,
    raw_cache_dir: Path | None = None,
    report_years: tuple[int, int] | None = None,
    candidates: Sequence[str] = CANDIDATE_NAMES,
    select: bool = False,
) -> dict[str, Any]:
    """Full report: the data audit, per-fold parameters, pooled and per-year scores.

    Every year in the test range is predicted, but only ``report_years`` (inclusive) is
    scored and reported, so a development run can leave the held-out years unread.
    ``select`` runs the candidate selection on the reported rows and refuses to look at any
    year after :data:`SELECTION_LAST_YEAR`. ``candidates`` limits which candidate models are reported.
    """
    frame = add_pre_match_elo(matches.frame)
    last_complete = int(frame["year"].max()) - 1
    earliest = int(frame["year"].min()) + window_years + 1
    test_years = list(range(max(first_test_year, earliest), last_complete + 1))
    if not test_years:
        raise ValueError("no complete test year with enough earlier history")
    first_report, last_report = report_years or (test_years[0], test_years[-1])
    if select and last_report > SELECTION_LAST_YEAR:
        raise ValueError(f"candidate selection may only use years up to {SELECTION_LAST_YEAR}, not {last_report}")
    unknown = sorted(set(candidates) - set(CANDIDATE_NAMES))
    if unknown:
        raise ValueError(f"unknown candidates {unknown}")
    predictions, folds = walk_forward(
        frame, test_years, window_years=window_years, refit_days=refit_days, map_fn=map_fn, raw_cache_dir=raw_cache_dir
    )
    reported = predictions[predictions["year"].between(first_report, last_report)]
    scored = reported[reported["supported"]].reset_index(drop=True)
    in_test = frame[frame["year"].isin(range(first_report, last_report + 1))]
    models, comparisons = reported_models(candidates), reported_comparisons(candidates)
    selection = None
    if select:
        chosen = select_candidate(scored, n_boot=n_boot, seed=seed)
        selection = {
            "best": chosen.best,
            "chosen": chosen.chosen,
            "pooled_brier": chosen.pooled_brier,
            "versus_best": {name: asdict(diff) for name, diff in chosen.versus_best.items()},
        }
    scope = {"models": models, "comparisons": comparisons, "n_boot": n_boot, "seed": seed}
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {"url": DATA_URL, "license": DATA_LICENSE},
        "config": {
            "first_test_year": test_years[0],
            "last_test_year": test_years[-1],
            "report_years": [first_report, last_report],
            "window_years": window_years,
            "refit_days": refit_days,
            "poisson_kwargs": dict(DEFAULT_POISSON_KWARGS),
            "max_goals": DEFAULT_MAX_GOALS,
            "elo_base_k": EloEngine().k_factor,
            "elo_home_advantage": HOME_ADVANTAGE_ELO,
            "n_boot": n_boot,
            "seed": seed,
            "primary_metric": PRIMARY_METRIC,
            "models": list(models),
            "selection_last_year": SELECTION_LAST_YEAR,
        },
        "data": {
            "rows_in_csv": matches.n_rows,
            "rows_without_score": matches.n_without_score,
            "scored_matches": len(frame),
            "first_date": frame["date"].min().date().isoformat(),
            "last_date": frame["date"].max().date().isoformat(),
            "test_year_matches": len(in_test),
            "test_year_neutral": int(in_test["neutral"].sum()),
            "test_year_unsupported_excluded": int((~reported["supported"]).sum()),
            "scored_in_backtest": len(scored),
        },
        "labels": LABELS,
        "folds": [asdict(f) for f in folds if first_report <= f.year <= last_report],
        "pooled": scope_report(scored, **scope),
        "by_year": {str(y): scope_report(g.reset_index(drop=True), **scope) for y, g in scored.groupby("year")},
    }
    if selection is not None:
        report["selection"] = selection
    return report


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _metrics_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    lines = [_row(["model", "n", "Brier", "RPS", "log loss", "accuracy", "ECE H", "ECE D", "ECE A"]), _row(["---"] * 9)]
    for name, m in metrics.items():
        lines.append(
            _row(
                [LABELS[name], str(m["n"]), f"{m['brier']:.4f}", f"{m['rps']:.4f}", f"{m['log_loss']:.4f}", f"{m['accuracy']:.3f}"]
                + [f"{m['ece'][c]:.3f}" for c in CLASSES]
            )
        )
    return lines


def _comparison_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [_row(["model", "vs", "metric", "mean diff", "95% CI", "verdict"]), _row(["---"] * 6)]
    for r in rows:
        lines.append(
            _row([LABELS[r["model"]], LABELS[r["baseline"]], r["metric"], f"{r['mean_diff']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", r["verdict"]])
        )
    return lines


def _fmt_diff(r: Mapping[str, Any]) -> str:
    return f"{r['mean_diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"


def _scoreline_table(scorelines: Mapping[str, Any]) -> list[str]:
    lines = [_row(["served matrix", "n", "scoreline log loss", "vs raw Poisson matrix", "max 1X2 gap to target"]), _row(["---"] * 5)]
    for name, s in scorelines.items():
        diff = _fmt_diff(s["vs_raw"]) if s["vs_raw"] else "reference"
        gap = f"{s['max_marginal_gap']:.1e}" if s["max_marginal_gap"] is not None else "n/a"
        lines.append(_row([LABELS[name], str(s["n"]), f"{s['mean_log_loss']:.4f}", diff, gap]))
    return lines


def _draw_table(metrics: Mapping[str, Mapping[str, Any]], models: Sequence[str]) -> list[str]:
    observed = metrics[models[0]]["calibration"]
    lines = [_row(["model", "mean P(H)", "mean P(D)", "mean P(A)"]), _row(["---"] * 4)]
    lines.append(_row(["observed", *[f"{observed[c]['frequency']:.3f}" for c in CLASSES]]))
    for name in models:
        lines.append(_row([LABELS[name], *[f"{metrics[name]['calibration'][c]['mean_predicted']:.3f}" for c in CLASSES]]))
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    cfg, data, pooled = report["config"], report["data"], report["pooled"]
    models = cfg["models"]
    out = [
        f"# Soccer walk-forward validation, test years {cfg['first_test_year']} to {cfg['last_test_year']}",
        "",
        f"Source: {report['source']['url']} ({report['source']['license']}).",
        f"Predicted years {cfg['first_test_year']} to {cfg['last_test_year']}, reported years {cfg['report_years'][0]} to {cfg['report_years'][1]}, refit every {cfg['refit_days']} day(s).",
        f"{data['test_year_matches']} matches in the reported years, {data['scored_in_backtest']} scored "
        f"({data['test_year_unsupported_excluded']} excluded: a team had fewer than {MIN_SUPPORTED_MATCHES} matches in the last {cfg['poisson_kwargs']['years_window']} years).",
        "",
        f"## Pooled, n={pooled['n']}",
        "",
        *_metrics_table(pooled["metrics"]),
        "",
        "## Mean predicted 1X2 against observed, pooled",
        "",
        *_draw_table(pooled["metrics"], models),
        "",
        "## Paired bootstrap, pooled (negative mean diff = model better; blocks = match dates)",
        "",
        *_comparison_table(pooled["comparisons"]),
        "",
        "## Scoreline coherence, pooled (matrix rescaled to the served 1X2 by class ratios)",
        "",
        *_scoreline_table(pooled["scorelines"]),
        "",
        "## By year (Brier / RPS / log loss)",
        "",
        _row(["year", "n", *[LABELS[m] for m in models]]),
        _row(["---"] * (2 + len(models))),
    ]
    for year, scope in report["by_year"].items():
        cells = [f"{scope['metrics'][m]['brier']:.4f} / {scope['metrics'][m]['rps']:.4f} / {scope['metrics'][m]['log_loss']:.4f}" for m in models]
        out.append(_row([year, str(scope["n"]), *cells]))
    out += ["", f"## Per-year verdict on {PRIMARY_METRIC}", "", _row(["year", "model", "vs", "mean diff", "95% CI", "verdict"]), _row(["---"] * 6)]
    for year, scope in report["by_year"].items():
        for r in scope["comparisons"]:
            if r["metric"] == PRIMARY_METRIC:
                out.append(_row([year, LABELS[r["model"]], LABELS[r["baseline"]], f"{r['mean_diff']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", r["verdict"]]))
    out += ["", "## Fitted parameters per fold", "", _row(["year", "rho", "temperature", "calibration years", "calibration n", "draw at parity"]), _row(["---"] * 6)]
    for f in report["folds"]:
        out.append(
            _row([str(f["year"]), f"{f['rho']:.4f}", f"{f['temperature']:.3f}", f"{f['calibration_years'][0]}-{f['calibration_years'][-1]}", str(f["n_calibration_matches"]), f"{f['draw_prob_at_parity']:.3f}"])
        )
    candidate_names = [m for m in models if m in CANDIDATE_NAMES]
    if candidate_names:
        out += ["", "## Candidate parameters per fold", "", _row(["year", "model", "home advantage (Elo)", "draw param", "scale", "blend weight on Elo"]), _row(["---"] * 6)]
        for f in report["folds"]:
            for name in candidate_names:
                p = f["candidates"][name]
                weight = "n/a" if p["blend_weight"] is None else f"{p['blend_weight']:.3f}"
                out.append(_row([str(f["year"]), LABELS[name], f"{p['home_advantage']:.1f}", f"{p['draw_param']:.3f}", f"{p['scale']:.3f}", weight]))
    selection = report.get("selection")
    if selection:
        out += ["", "## Candidate selection (selection years only)", "", _row(["model", "pooled Brier", "Brier minus best (95% CI)"]), _row(["---"] * 3)]
        for name, brier in sorted(selection["pooled_brier"].items(), key=lambda kv: kv[1]):
            diff = selection["versus_best"].get(name)
            out.append(_row([LABELS[name], f"{brier:.4f}", "best" if diff is None else _fmt_diff(diff)]))
        out += ["", f"Best by Brier: {LABELS[selection['best']]}. Chosen (simplest not significantly worse): **{LABELS[selection['chosen']]}**."]
    return "\n".join(out) + "\n"
