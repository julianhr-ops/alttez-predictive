"""NBA win-probability models: MOV-adjusted Elo, market spread model, blend and calibrators.

Every model here is causal. The Elo run scores all games of a date from the ratings
as they stood before that date and only then applies the date's results, so a game's
prediction cannot depend on its own result or on anything later.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, ndtr
from sklearn.linear_model import LogisticRegression

from alttez.nba.data import SeasonType

INITIAL_RATING = 1500.0
MOV_EXPONENT = 0.8
PROBABILITY_FLOOR = 1e-3
TUNE_SEASONS = 5
UNREGULARISED_C = 1e10


class ModelFitError(ValueError):
    """A fit that cannot be trusted (too little data, degenerate or non-monotone result)."""


def clip_probability(p: np.ndarray | float) -> np.ndarray:
    return np.clip(p, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)


def logit(p: np.ndarray | float) -> np.ndarray:
    p = clip_probability(p)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True, slots=True)
class EloParams:
    k: float = 20.0
    home_advantage: float = 90.0
    season_regression: float = 0.3
    b2b_penalty: float = 0.0
    playoff_k_multiplier: float = 1.0
    use_mov: bool = True


class EloTracker:
    """Ratings by ESPN franchise id plus each team's last game date (for rest days)."""

    def __init__(self, params: EloParams) -> None:
        self.params = params
        self.ratings: dict[int, float] = {}
        self.last_game_day: dict[int, int] = {}
        self.season: int | None = None

    def rating(self, team: int) -> float:
        return self.ratings.get(team, INITIAL_RATING)

    def enter_season(self, season: int) -> None:
        if self.season is not None and season > self.season:
            keep = 1.0 - self.params.season_regression
            self.ratings = {t: INITIAL_RATING + keep * (r - INITIAL_RATING) for t, r in self.ratings.items()}
        if self.season is None or season > self.season:
            self.season = season

    def is_back_to_back(self, team: int, day: int) -> bool:
        last = self.last_game_day.get(team)
        return last is not None and day - last == 1

    def diff(self, home: int, away: int, day: int, neutral: bool) -> float:
        """Home rating advantage in Elo points, including home court and back-to-back effects."""
        p = self.params
        value = self.rating(home) - self.rating(away)
        if not neutral:
            value += p.home_advantage
        value -= p.b2b_penalty * (self.is_back_to_back(home, day) - self.is_back_to_back(away, day))
        return value

    def update(self, home: int, away: int, day: int, diff: float, margin: int, season_type: int) -> None:
        p = self.params
        p_home = expit(diff * math.log(10.0) / 400.0)
        k = p.k * (p.playoff_k_multiplier if season_type == SeasonType.PLAYOFFS else 1.0)
        if p.use_mov:
            winner_diff = diff if margin > 0 else -diff
            k *= (abs(margin) + 3.0) ** MOV_EXPONENT / (7.5 + 0.006 * winner_diff)
        shift = k * ((1.0 if margin > 0 else 0.0) - p_home)
        self.ratings[home] = self.rating(home) + shift
        self.ratings[away] = self.rating(away) - shift
        self.last_game_day[home] = day
        self.last_game_day[away] = day


@dataclass(frozen=True, slots=True)
class EloRun:
    predictions: pd.DataFrame
    tracker: EloTracker


def elo_probability(diff: np.ndarray | float) -> np.ndarray:
    return expit(np.asarray(diff, dtype=float) * math.log(10.0) / 400.0)


def run_elo(games: pd.DataFrame, params: EloParams) -> EloRun:
    """Walk through games in date order; ``games`` must be sorted by ``game_date``.

    Returns pre-game ``elo_diff`` and ``p_home`` for every row. Outcomes are read only
    after the whole date has been scored.
    """
    days = games["game_date"].to_numpy("datetime64[D]").astype(np.int64)
    if len(days) > 1 and (np.diff(days) < 0).any():
        raise ValueError("games must be sorted by game_date")
    home = games["home_id"].to_numpy(np.int64).tolist()
    away = games["away_id"].to_numpy(np.int64).tolist()
    neutral = games["neutral"].to_numpy(bool).tolist()
    season = games["season"].to_numpy(np.int64).tolist()
    season_type = games["season_type"].to_numpy(np.int64).tolist()
    margin = (games["home_score"] - games["away_score"]).to_numpy(np.int64).tolist()

    tracker = EloTracker(params)
    n = len(games)
    diffs = [0.0] * n
    start = 0
    while start < n:
        day = int(days[start])
        stop = start
        while stop < n and days[stop] == day:
            stop += 1
        tracker.enter_season(season[start])
        for g in range(start, stop):
            diffs[g] = tracker.diff(home[g], away[g], day, neutral[g])
        for g in range(start, stop):
            tracker.update(home[g], away[g], day, diffs[g], margin[g], season_type[g])
        start = stop
    diff_array = np.asarray(diffs, dtype=float)
    frame = pd.DataFrame(
        {"game_id": games["game_id"].to_numpy(), "elo_diff": diff_array, "p_home": elo_probability(diff_array)}
    )
    return EloRun(predictions=frame, tracker=tracker)


def elo_grid() -> tuple[EloParams, ...]:
    return tuple(
        EloParams(k=k, home_advantage=hca, season_regression=reg, b2b_penalty=b2b, playoff_k_multiplier=po)
        for k, hca, reg, b2b, po in itertools.product((12.0, 20.0, 28.0), (30.0, 60.0, 90.0), (0.2, 0.35, 0.5), (0.0, 30.0, 60.0), (0.5, 0.75, 1.0))
    )


def vanilla_elo_grid() -> tuple[EloParams, ...]:
    return tuple(
        EloParams(k=k, home_advantage=hca, season_regression=reg, use_mov=False)
        for k, hca, reg in itertools.product((12.0, 20.0, 28.0), (30.0, 60.0, 90.0), (0.2, 0.35, 0.5))
    )


def recent_seasons(games: pd.DataFrame, count: int) -> list[int]:
    return sorted(games["season"].unique())[-count:]


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = clip_probability(p)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def select_elo_params(history: pd.DataFrame, grid: Sequence[EloParams], *, tune_seasons: int = TUNE_SEASONS) -> EloParams:
    """Grid search on the last ``tune_seasons`` seasons of ``history`` only (log loss)."""
    window = history["season"].isin(recent_seasons(history, tune_seasons)).to_numpy()
    y = (history["home_score"] > history["away_score"]).to_numpy(float)[window]
    best, best_loss = None, math.inf
    for params in grid:
        p = run_elo(history, params).predictions["p_home"].to_numpy()[window]
        loss = _log_loss(p, y)
        if loss < best_loss:
            best, best_loss = params, loss
    if best is None:
        raise ModelFitError("empty hyperparameter grid")
    return best


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    slope: float
    intercept: float

    def __call__(self, p: np.ndarray | float) -> np.ndarray:
        return clip_probability(expit(self.slope * logit(p) + self.intercept))

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        _require_binary(y)
        model = LogisticRegression(C=UNREGULARISED_C, max_iter=1000).fit(logit(p).reshape(-1, 1), y.astype(int))
        slope = float(model.coef_[0, 0])
        if slope <= 0:
            raise ModelFitError(f"Platt slope {slope:.4f} is not positive; calibration would not be monotone")
        return cls(slope, float(model.intercept_[0]))


@dataclass(frozen=True, slots=True, eq=False)
class IsotonicCalibrator:
    knots: np.ndarray
    values: np.ndarray

    def __call__(self, p: np.ndarray | float) -> np.ndarray:
        return clip_probability(np.interp(np.asarray(p, dtype=float), self.knots, self.values))

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        _require_binary(y)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(np.asarray(p, dtype=float), y.astype(float))
        return cls(np.asarray(iso.X_thresholds_, dtype=float), np.asarray(iso.y_thresholds_, dtype=float))


def _require_binary(y: np.ndarray) -> None:
    if len(y) < 50 or len(np.unique(y)) < 2:
        raise ModelFitError(f"calibration needs at least 50 games with both outcomes, got {len(y)}")


@dataclass(frozen=True, slots=True)
class MarketModel:
    """Closing spread to home win probability: Phi(spread / sd)."""

    sd: float

    def p_home(self, home_margin_line: np.ndarray | float) -> np.ndarray:
        return clip_probability(ndtr(np.asarray(home_margin_line, dtype=float) / self.sd))

    @classmethod
    def fit(cls, home_margin_line: np.ndarray, margin: np.ndarray) -> MarketModel:
        if len(margin) < 100:
            raise ModelFitError(f"market model needs at least 100 lined games, got {len(margin)}")
        y = (margin > 0).astype(float)
        line = np.asarray(home_margin_line, dtype=float)

        def loss(sd: float) -> float:
            return _log_loss(ndtr(line / sd), y)

        result = minimize_scalar(loss, bounds=(4.0, 30.0), method="bounded")
        return cls(float(result.x))


@dataclass(frozen=True, slots=True)
class BlendModel:
    """Logistic regression on [elo_logit, market_logit]; only defined where a line exists."""

    intercept: float
    elo_coef: float
    market_coef: float

    def p_home(self, elo_p: np.ndarray, market_p: np.ndarray) -> np.ndarray:
        return clip_probability(expit(self.intercept + self.elo_coef * logit(elo_p) + self.market_coef * logit(market_p)))

    @classmethod
    def fit(cls, elo_p: np.ndarray, market_p: np.ndarray, y: np.ndarray) -> BlendModel:
        if len(y) < 100 or len(np.unique(y)) < 2:
            raise ModelFitError(f"blend needs at least 100 lined games with both outcomes, got {len(y)}")
        features = np.column_stack([logit(elo_p), logit(market_p)])
        model = LogisticRegression(C=UNREGULARISED_C, max_iter=1000).fit(features, y.astype(int))
        return cls(float(model.intercept_[0]), float(model.coef_[0, 0]), float(model.coef_[0, 1]))


def fit_margin_slope(elo_diff: np.ndarray, margin: np.ndarray) -> float:
    """Points of expected home margin per Elo point (least squares through the origin)."""
    denom = float(np.dot(elo_diff, elo_diff))
    if denom <= 0:
        raise ModelFitError("cannot fit margin slope on zero Elo differences")
    return float(np.dot(elo_diff, margin) / denom)


@dataclass(frozen=True, slots=True)
class FittedModels:
    """Everything learned from ``train_seasons`` and nothing else."""

    train_seasons: tuple[int, ...]
    tune_seasons: tuple[int, ...]
    elo_params: EloParams
    vanilla_params: EloParams
    margin_slope: float
    home_rate: float
    platt: PlattCalibrator
    isotonic: IsotonicCalibrator
    market: MarketModel | None
    blend: BlendModel | None
    n_lined_train: int


def _with_outcomes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    margin = (frame["home_score"] - frame["away_score"]).to_numpy(float)
    return margin, (margin > 0).astype(float)


def fit_models(
    history: pd.DataFrame,
    *,
    elo_candidates: Sequence[EloParams] | None = None,
    vanilla_candidates: Sequence[EloParams] | None = None,
    tune_seasons: int = TUNE_SEASONS,
) -> FittedModels:
    """Fit hyperparameters, calibrators, market sd and blend on ``history`` alone.

    ``history`` is the completed games of the training seasons, sorted by date, with the
    optional ``home_margin_line`` column. The caller must not include any test-season row.
    """
    elo_params = select_elo_params(history, elo_candidates or elo_grid(), tune_seasons=tune_seasons)
    vanilla_params = select_elo_params(history, vanilla_candidates or vanilla_elo_grid(), tune_seasons=tune_seasons)
    run = run_elo(history, elo_params).predictions
    seasons = recent_seasons(history, tune_seasons)
    window = history["season"].isin(seasons).to_numpy()
    margin, y = _with_outcomes(history)
    p_elo = run["p_home"].to_numpy()

    market = blend = None
    lined = history["home_margin_line"].notna().to_numpy() if "home_margin_line" in history else np.zeros(len(history), bool)
    if lined.sum() >= 100:
        market = MarketModel.fit(history["home_margin_line"].to_numpy()[lined], margin[lined])
        market_p = market.p_home(history["home_margin_line"].to_numpy()[lined])
        blend = BlendModel.fit(p_elo[lined], market_p, y[lined])

    return FittedModels(
        train_seasons=tuple(sorted(int(s) for s in history["season"].unique())),
        tune_seasons=tuple(int(s) for s in seasons),
        elo_params=elo_params,
        vanilla_params=vanilla_params,
        margin_slope=fit_margin_slope(run["elo_diff"].to_numpy()[window], margin[window]),
        home_rate=float(y[window].mean()),
        platt=PlattCalibrator.fit(p_elo[window], y[window]),
        isotonic=IsotonicCalibrator.fit(p_elo[window], y[window]),
        market=market,
        blend=blend,
        n_lined_train=int(lined.sum()),
    )


PREDICTION_COLUMNS = (
    "game_id", "p_const", "p_vanilla", "p_elo", "p_elo_platt", "p_elo_isotonic",
    "elo_margin", "p_market", "p_blend", "market_margin",
)


def predict_games(models: FittedModels, history: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Predict ``target`` games with models fitted on ``history`` (which must precede them).

    The Elo state is built by replaying ``history`` followed by ``target`` in date order;
    each target game still sees only results from earlier dates.
    """
    if len(history) and len(target) and target["game_date"].min() < history["game_date"].max():
        raise ValueError("target games must not start before the end of the history")
    combined = pd.concat([history, target], ignore_index=True)
    n_history = len(history)
    elo = run_elo(combined, models.elo_params).predictions.iloc[n_history:].reset_index(drop=True)
    vanilla = run_elo(combined, models.vanilla_params).predictions.iloc[n_history:].reset_index(drop=True)

    p_elo = elo["p_home"].to_numpy()
    out = pd.DataFrame(
        {
            "game_id": target["game_id"].to_numpy(),
            "p_const": np.full(len(target), models.home_rate),
            "p_vanilla": vanilla["p_home"].to_numpy(),
            "p_elo": p_elo,
            "p_elo_platt": models.platt(p_elo),
            "p_elo_isotonic": models.isotonic(p_elo),
            "elo_margin": elo["elo_diff"].to_numpy() * models.margin_slope,
            "p_market": np.nan,
            "p_blend": np.nan,
            "market_margin": np.nan,
        }
    )
    if models.market is not None and "home_margin_line" in target:
        line = target["home_margin_line"].to_numpy(float)
        lined = ~np.isnan(line)
        out.loc[lined, "market_margin"] = line[lined]
        out.loc[lined, "p_market"] = models.market.p_home(line[lined])
        if models.blend is not None:
            out.loc[lined, "p_blend"] = models.blend.p_home(p_elo[lined], out.loc[lined, "p_market"].to_numpy())
    return out[list(PREDICTION_COLUMNS)]

