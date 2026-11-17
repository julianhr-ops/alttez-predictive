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
import io
import os
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
from alttez.experiments.tournament_oracle import build_probability_matrix
from alttez.features.elo import HOME_ADVANTAGE_ELO, EloEngine, tournament_k_factor
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
MODELS = ("uniform", "base_rate", "elo_only", "poisson_raw", "poisson_calibrated")
LABELS = {
    "uniform": "uniform",
    "base_rate": "base rate",
    "elo_only": "Elo only",
    "poisson_raw": "Poisson raw",
    "poisson_calibrated": "Poisson + temperature",
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

    All matches on one date see the ratings as they stood at the start of that date, and
    their updates are applied together afterwards. ``matches`` must be sorted by date.
    """
    elo = engine or EloEngine(mov_enabled=False)
    ratings: dict[str, float] = {}
    home_before = np.empty(len(matches))
    away_before = np.empty(len(matches))
    home = matches["home_team"].to_numpy()
    away = matches["away_team"].to_numpy()
    neutral = matches["neutral"].to_numpy(bool)
    tournament = matches["tournament"].to_numpy()
    result = np.where(matches["outcome"].to_numpy() == "H", 1.0, np.where(matches["outcome"].to_numpy() == "D", 0.5, 0.0))
    if not matches["date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date")
    dates = matches["date"].to_numpy()

    start = 0
    while start < len(matches):
        stop = start
        while stop < len(matches) and dates[stop] == dates[start]:
            stop += 1
        pending: list[tuple[str, float]] = []
        for i in range(start, stop):
            h = ratings.get(home[i], elo.base_rating)
            a = ratings.get(away[i], elo.base_rating)
            home_before[i], away_before[i] = h, a
            h_eff = h + (0 if neutral[i] else HOME_ADVANTAGE_ELO)
            expected_home = elo.get_expected_score(h_eff, a)
            k = tournament_k_factor(tournament[i], elo.k_factor)
            pending.append((home[i], elo.update_rating(0.0, expected_home, result[i], k)))
            pending.append((away[i], elo.update_rating(0.0, 1 - expected_home, 1 - result[i], k)))
        for team, delta in pending:
            ratings[team] = ratings.get(team, elo.base_rating) + delta
        start = stop

    out = matches.copy()
    out["elo_home_before"] = home_before
    out["elo_away_before"] = away_before
    out["elo_diff"] = home_before - away_before
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
    jobs = [(subset, y) for subset, y in zip(cut, predicted, strict=True)]
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
        for name, probs in base.items():
            frame = _with_probabilities(frame, name, probs)
        frame = _with_probabilities(frame, "poisson_calibrated", calibrate_matrices(raw.matrices, frame["supported"].to_numpy(bool), temperature))
        frame["elo_diff"] = target["elo_diff"].to_numpy(float)
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
            )
        )
    return pd.concat(frames, ignore_index=True), folds


def _raw_job(job: tuple[pd.DataFrame, int], refit_days: int) -> RawYear:
    subset, year = job
    return raw_poisson_year(subset, year, refit_days=refit_days)


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


def scope_report(frame: pd.DataFrame, *, n_boot: int, seed: int) -> dict[str, Any]:
    outcomes = frame["outcome"].tolist()
    return {
        "n": len(frame),
        "metrics": {m: point_metrics(model_probabilities(frame, m), outcomes) for m in MODELS},
        "comparisons": [row for a, b in COMPARISONS for row in compare(frame, a, b, n_boot=n_boot, seed=seed)],
    }


def run_backtest(
    matches: Matches,
    *,
    first_test_year: int = FIRST_TEST_YEAR,
    window_years: int = WINDOW_YEARS,
    refit_days: int = REFIT_DAYS,
    n_boot: int = 2000,
    seed: int = 0,
    map_fn: Callable[..., Any] = map,
) -> dict[str, Any]:
    """Full report: the data audit, per-fold parameters, pooled and per-year scores."""
    frame = add_pre_match_elo(matches.frame)
    last_complete = int(frame["year"].max()) - 1
    earliest = int(frame["year"].min()) + window_years + 1
    test_years = list(range(max(first_test_year, earliest), last_complete + 1))
    if not test_years:
        raise ValueError("no complete test year with enough earlier history")
    predictions, folds = walk_forward(frame, test_years, window_years=window_years, refit_days=refit_days, map_fn=map_fn)
    scored = predictions[predictions["supported"]].reset_index(drop=True)
    in_test = frame[frame["year"].isin(test_years)]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {"url": DATA_URL, "license": DATA_LICENSE},
        "config": {
            "first_test_year": test_years[0],
            "last_test_year": test_years[-1],
            "window_years": window_years,
            "refit_days": refit_days,
            "poisson_kwargs": dict(DEFAULT_POISSON_KWARGS),
            "max_goals": DEFAULT_MAX_GOALS,
            "elo_base_k": EloEngine().k_factor,
            "elo_home_advantage": HOME_ADVANTAGE_ELO,
            "n_boot": n_boot,
            "seed": seed,
            "primary_metric": PRIMARY_METRIC,
        },
        "data": {
            "rows_in_csv": matches.n_rows,
            "rows_without_score": matches.n_without_score,
            "scored_matches": len(frame),
            "first_date": frame["date"].min().date().isoformat(),
            "last_date": frame["date"].max().date().isoformat(),
            "test_year_matches": len(in_test),
            "test_year_neutral": int(in_test["neutral"].sum()),
            "test_year_unsupported_excluded": int((~predictions["supported"]).sum()),
            "scored_in_backtest": len(scored),
        },
        "labels": LABELS,
        "folds": [asdict(f) for f in folds],
        "pooled": scope_report(scored, n_boot=n_boot, seed=seed),
        "by_year": {str(y): scope_report(g.reset_index(drop=True), n_boot=n_boot, seed=seed) for y, g in scored.groupby("year")},
    }


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


def render_markdown(report: Mapping[str, Any]) -> str:
    cfg, data, pooled = report["config"], report["data"], report["pooled"]
    out = [
        f"# Soccer walk-forward validation, test years {cfg['first_test_year']} to {cfg['last_test_year']}",
        "",
        f"Source: {report['source']['url']} ({report['source']['license']}).",
        f"{data['test_year_matches']} matches in the test years, {data['scored_in_backtest']} scored "
        f"({data['test_year_unsupported_excluded']} excluded: a team had fewer than {MIN_SUPPORTED_MATCHES} matches in the last {cfg['poisson_kwargs']['years_window']} years).",
        "",
        f"## Pooled, n={pooled['n']}",
        "",
        *_metrics_table(pooled["metrics"]),
        "",
        "## Paired bootstrap, pooled (negative mean diff = model better; blocks = match dates)",
        "",
        *_comparison_table(pooled["comparisons"]),
        "",
        "## By year (Brier / RPS / log loss)",
        "",
        _row(["year", "n", *[LABELS[m] for m in MODELS]]),
        _row(["---"] * (2 + len(MODELS))),
    ]
    for year, scope in report["by_year"].items():
        cells = [f"{scope['metrics'][m]['brier']:.4f} / {scope['metrics'][m]['rps']:.4f} / {scope['metrics'][m]['log_loss']:.4f}" for m in MODELS]
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
    return "\n".join(out) + "\n"
