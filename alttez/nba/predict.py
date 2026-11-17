"""Operational NBA predictions for the games ESPN lists as upcoming.

Ratings are replayed from every completed game in the hoopR schedule. Calibrator, market
model and blend are fitted on the completed seasons only (the same ``fit_models`` the
walk-forward validation uses). The served probability follows the measured policy: the
Elo+market blend when a line exists, the Platt-calibrated Elo otherwise, and the row
records which one served. Nothing is guessed: stale data stops the run, and a game whose
teams have no rating is skipped and reported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alttez.nba import data
from alttez.nba.model import FittedModels, elo_probability, fit_models, run_elo
from alttez.nba.store import NbaPredictionRow, ServedModel

MODEL_VERSION = "nba-elo-v1"
MAX_DATA_AGE_DAYS = 3
SCHEDULE_CACHE = "schedule.parquet"
CLOSING_LINES_CACHE = "closing_lines.parquet"
STALE_DATA = "stale_data"

ScoreboardFetch = Callable[[date], list[data.NbaScheduledGame]]


class StaleDataError(RuntimeError):
    """The hoopR schedule lags a season that is being played, so ratings would be old."""

    code = STALE_DATA

    def __init__(self, newest_completed: date, today: date, missing_days: int) -> None:
        self.newest_completed = newest_completed
        self.today = today
        super().__init__(
            f"{STALE_DATA}: newest completed game is {newest_completed} ({(today - newest_completed).days} days "
            f"before {today}) and the schedule lists games after it ({missing_days} dates), so ratings are out of date"
        )


@dataclass(frozen=True, slots=True)
class NbaInputs:
    """Completed games with closing lines attached, plus what the freshness gate needs."""

    games: pd.DataFrame
    complete_seasons: tuple[int, ...]
    scheduled_dates: tuple[date, ...]

    @property
    def newest_completed(self) -> date:
        return self.games["game_date"].max().date()


@dataclass(frozen=True, slots=True)
class SkippedGame:
    game_id: int
    matchup: str
    reason: str


@dataclass(frozen=True, slots=True)
class PlannedPrediction:
    row: NbaPredictionRow
    home_abbreviation: str
    away_abbreviation: str

    @property
    def matchup(self) -> str:
        return f"{self.away_abbreviation} @ {self.home_abbreviation}"


@dataclass(frozen=True, slots=True)
class PredictionRun:
    predictions: tuple[PlannedPrediction, ...]
    skipped: tuple[SkippedGame, ...]
    newest_completed: date
    train_seasons: tuple[int, int]
    n_lined_train: int


def schedule_dates(raw: pd.DataFrame) -> tuple[date, ...]:
    """US dates of every NBA-franchise game in the schedule, completed or not."""
    frame = raw[raw["home_id"].isin(data.NBA_TEAM_IDS) & raw["away_id"].isin(data.NBA_TEAM_IDS)]
    frame = frame[frame["season_type"].isin([int(t) for t in data.SeasonType])]
    start = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
    return tuple(sorted({d.date() for d in data.local_game_date(start)}))


def load_inputs(cache_dir: Path = data.DEFAULT_CACHE_DIR, **cache_kwargs) -> NbaInputs:
    raw = data.load_cached_parquet(data.SCHEDULE_URL, cache_dir / SCHEDULE_CACHE, **cache_kwargs)
    lines = data.normalise_closing_lines(
        data.load_cached_parquet(data.CLOSING_LINES_URL, cache_dir / CLOSING_LINES_CACHE, **cache_kwargs)
    )
    schedule = data.normalise_schedule(raw)
    return NbaInputs(
        games=data.attach_lines(schedule.games, lines),
        complete_seasons=schedule.complete_seasons,
        scheduled_dates=schedule_dates(raw),
    )


def check_freshness(
    newest_completed: date, scheduled: Iterable[date], today: date, *, max_age_days: int = MAX_DATA_AGE_DAYS
) -> None:
    """Raise :class:`StaleDataError` when games should have finished since the newest completed one.

    The data is stale only when it is both older than ``max_age_days`` and the schedule
    itself lists games dated after the newest completed game and before today. The
    offseason and the All-Star break list no such games, so they never trip the gate.
    """
    if (today - newest_completed).days <= max_age_days:
        return
    missing = [d for d in set(scheduled) if newest_completed < d < today]
    if missing:
        raise StaleDataError(newest_completed, today, len(missing))


def fit_for_prediction(inputs: NbaInputs) -> FittedModels:
    history = inputs.games[inputs.games["season"].isin(inputs.complete_seasons)].reset_index(drop=True)
    return fit_models(history)


def select_served(p_elo_platt: float, p_blend: float) -> tuple[ServedModel, float]:
    """Blend when it exists (a market line was available), otherwise the calibrated Elo."""
    if math.isfinite(p_blend):
        return ServedModel.BLEND, p_blend
    return ServedModel.ELO_PLATT, p_elo_platt


def _target_frame(games: list[data.NbaScheduledGame]) -> pd.DataFrame:
    start = pd.to_datetime([g.start_time for g in games], utc=True).tz_localize(None)
    frame = pd.DataFrame(
        {
            "game_id": [g.game_id for g in games],
            "start_time": start,
            "game_date": data.local_game_date(pd.Series(start)).to_numpy(),
            "season": [g.season for g in games],
            "season_type": [g.season_type for g in games],
            "home_id": [g.home_id for g in games],
            "away_id": [g.away_id for g in games],
            "home_score": 0,
            "away_score": 0,
            "neutral": [g.neutral for g in games],
            "home_margin_line": [np.nan if g.home_margin_line is None else g.home_margin_line for g in games],
        }
    )
    return frame.astype({"game_id": "int64", "season": "int64", "season_type": "int64", "home_id": "int64", "away_id": "int64"})


def _skip_reason(game: data.NbaScheduledGame, rated_teams: set[int]) -> str | None:
    if game.season_type not in {int(t) for t in data.SeasonType}:
        return f"season type {game.season_type} is not regular season, playoffs or play-in"
    missing = sorted({game.home_id, game.away_id} - rated_teams)
    if missing:
        return f"no rating for team id(s) {missing}"
    return None


def _day_number(day: pd.Timestamp) -> int:
    return int(pd.Timestamp(day).to_datetime64().astype("datetime64[D]").astype(np.int64))


def predict_day(models: FittedModels, prior: pd.DataFrame, group: pd.DataFrame, last_played: dict[int, int]) -> pd.DataFrame:
    """Predict the games of one date from ``prior`` (strictly earlier results).

    ``last_played`` maps a team to the day it plays earlier in the upcoming window, so a
    back-to-back across two upcoming dates is penalised even though neither result is
    known. Same output as ``model.predict_games`` for the columns it shares.
    """
    day = _day_number(group["game_date"].iloc[0])
    tracker = run_elo(prior, models.elo_params).tracker
    tracker.enter_season(int(group["season"].iloc[0]))
    for team, played in last_played.items():
        tracker.last_game_day[team] = max(tracker.last_game_day.get(team, played), played)
    diffs = np.array(
        [
            tracker.diff(int(h), int(a), day, bool(n))
            for h, a, n in zip(group["home_id"], group["away_id"], group["neutral"], strict=True)
        ]
    )
    p_elo = elo_probability(diffs)
    out = pd.DataFrame(
        {
            "game_id": group["game_id"].to_numpy(),
            "p_elo_platt": models.platt(p_elo),
            "elo_margin": diffs * models.margin_slope,
            "p_market": np.nan,
            "p_blend": np.nan,
            "market_margin": np.nan,
        }
    )
    line = group["home_margin_line"].to_numpy(float)
    lined = ~np.isnan(line)
    if models.market is not None and lined.any():
        out.loc[lined, "market_margin"] = line[lined]
        out.loc[lined, "p_market"] = models.market.p_home(line[lined])
        if models.blend is not None:
            out.loc[lined, "p_blend"] = models.blend.p_home(p_elo[lined], out.loc[lined, "p_market"].to_numpy())
    return out


def build_predictions(
    models: FittedModels,
    inputs: NbaInputs,
    upcoming: list[data.NbaScheduledGame],
    *,
    now: datetime,
) -> PredictionRun:
    """Predict each scheduled game from the ratings as they stood before its own date.

    Each date is predicted on its own with only earlier results, so a game never
    inherits a placeholder result from another upcoming game.
    """
    history = inputs.games
    rated = set(history["home_id"]) | set(history["away_id"])
    skipped: list[SkippedGame] = []
    ready: list[data.NbaScheduledGame] = []
    for game in upcoming:
        reason = _skip_reason(game, rated)
        label = f"{game.away_abbreviation} @ {game.home_abbreviation}"
        if reason:
            skipped.append(SkippedGame(game.game_id, label, reason))
        elif game.state != "pre" or game.start_time <= now:
            skipped.append(SkippedGame(game.game_id, label, "already started"))
        else:
            ready.append(game)

    abbreviations = {g.game_id: (g.home_abbreviation, g.away_abbreviation) for g in ready}
    sources = {g.game_id: g.odds_provider for g in ready}
    totals = {g.game_id: g.total_line for g in ready}
    planned: list[PlannedPrediction] = []
    if ready:
        target = _target_frame(ready)
        last_played: dict[int, int] = {}
        for day, group in target.groupby("game_date", sort=True):
            prior = history[history["game_date"] < day].reset_index(drop=True)
            group = group.reset_index(drop=True)
            out = predict_day(models, prior, group, last_played)
            merged = group.merge(out, on="game_id", validate="one_to_one")
            day_number = _day_number(day)
            last_played.update({int(t): day_number for t in (*group["home_id"], *group["away_id"])})
            for rec in merged.itertuples(index=False):
                served, p_served = select_served(float(rec.p_elo_platt), float(rec.p_blend))
                lined = math.isfinite(rec.p_market)
                gid = int(rec.game_id)
                source = sources[gid]
                row = NbaPredictionRow(
                    game_id=gid,
                    game_date=rec.game_date.date(),
                    tipoff_utc=rec.start_time.to_pydatetime().replace(tzinfo=UTC),
                    home_team_id=int(rec.home_id),
                    away_team_id=int(rec.away_id),
                    model_version=MODEL_VERSION,
                    served_model=served,
                    p_home_elo=round(float(rec.p_elo_platt), 6),
                    p_home_market=round(float(rec.p_market), 6) if lined else None,
                    p_home_served=round(p_served, 6),
                    expected_margin=round(float(rec.elo_margin), 3),
                    market_spread=float(rec.market_margin) if lined else None,
                    market_total=totals[gid] if lined else None,
                    market_source=(f"espn:{source}" if source else "espn") if lined else None,
                    created_at=now,
                )
                planned.append(PlannedPrediction(row, *abbreviations[gid]))
    planned.sort(key=lambda p: (p.row.tipoff_utc, p.row.game_id))
    return PredictionRun(
        predictions=tuple(planned),
        skipped=tuple(skipped),
        newest_completed=inputs.newest_completed,
        train_seasons=(models.train_seasons[0], models.train_seasons[-1]),
        n_lined_train=models.n_lined_train,
    )


def fetch_upcoming(today: date, *, fetch: ScoreboardFetch = data.fetch_espn_scoreboard) -> list[data.NbaScheduledGame]:
    """ESPN scoreboard for ``today`` and the next day (UTC), deduplicated by game id."""
    seen: dict[int, data.NbaScheduledGame] = {}
    for offset in (0, 1):
        for game in fetch(today + timedelta(days=offset)):
            seen.setdefault(game.game_id, game)
    return list(seen.values())


def predict_upcoming(
    inputs: NbaInputs,
    *,
    today: date,
    now: datetime,
    fetch: ScoreboardFetch = data.fetch_espn_scoreboard,
    models: FittedModels | None = None,
) -> PredictionRun:
    """Gate on freshness, then fit (unless ``models`` is given) and predict the upcoming games."""
    check_freshness(inputs.newest_completed, inputs.scheduled_dates, today)
    fitted = models if models is not None else fit_for_prediction(inputs)
    return build_predictions(fitted, inputs, fetch_upcoming(today, fetch=fetch), now=now)
