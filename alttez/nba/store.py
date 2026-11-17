"""Persistence of NBA predictions in ``public.nba_predictions`` (migration 015).

A prediction row is written once, while its game is still scheduled, and is never
rewritten: the insert is ``ON CONFLICT DO NOTHING`` and :func:`grade_prediction` only
fills the result columns. Row mapping and grading math are pure and tested apart from
the SQL. Every function takes the engine explicitly; callers own its lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from sqlalchemy import Boolean, Date, DateTime, bindparam, text
from sqlalchemy.engine import Engine

TABLE = "public.nba_predictions"


class ServedModel(StrEnum):
    BLEND = "blend"
    ELO_PLATT = "elo_platt"


class PredictionStatus(StrEnum):
    SCHEDULED = "scheduled"
    GRADED = "graded"


class InsertOutcome(StrEnum):
    INSERTED = "inserted"
    ALREADY_EXISTS = "already_exists"
    PAST_TIPOFF = "past_tipoff"


@dataclass(frozen=True, slots=True)
class NbaPredictionRow:
    """One row of ``nba_predictions``. Probabilities are for the home team to win.

    ``p_home_elo`` is the Platt-calibrated Elo-only probability. ``market_spread`` is the
    market-expected home margin (positive means the home team is favoured) and
    ``expected_margin`` is the Elo model's expected home margin.
    """

    game_id: int
    game_date: date
    tipoff_utc: datetime
    home_team_id: int
    away_team_id: int
    model_version: str
    served_model: ServedModel
    p_home_elo: float
    p_home_market: float | None
    p_home_served: float
    expected_margin: float
    market_spread: float | None
    market_total: float | None
    market_source: str | None
    created_at: datetime
    status: PredictionStatus = PredictionStatus.SCHEDULED
    home_score: int | None = None
    away_score: int | None = None
    home_win: bool | None = None
    brier: float | None = None
    graded_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("tipoff_utc", "created_at"):
            if getattr(self, name).tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if not 0.0 <= self.p_home_served <= 1.0:
            raise ValueError(f"p_home_served out of range: {self.p_home_served}")
        if self.served_model is ServedModel.BLEND and self.p_home_market is None:
            raise ValueError("a blend prediction needs a market probability")


COLUMNS = tuple(f.name for f in fields(NbaPredictionRow))
_TIMESTAMPS = ("tipoff_utc", "created_at", "graded_at")
_FLOATS = (
    "p_home_elo", "p_home_market", "p_home_served", "expected_margin",
    "market_spread", "market_total", "brier",
)
_OPTIONAL_INTS = ("home_score", "away_score")


def _to_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(value) if isinstance(value, str) else value


def _optional(value: Any, convert: Any) -> Any:
    return None if value is None else convert(value)


def row_from_mapping(record: Any) -> NbaPredictionRow:
    """Build a row from a DB record. Tolerates ``Decimal`` numerics and ISO-string timestamps."""
    values: dict[str, Any] = {
        "game_id": int(record["game_id"]),
        "game_date": _to_date(record["game_date"]),
        "home_team_id": int(record["home_team_id"]),
        "away_team_id": int(record["away_team_id"]),
        "model_version": str(record["model_version"]),
        "served_model": ServedModel(record["served_model"]),
        "market_source": record["market_source"],
        "status": PredictionStatus(record["status"]),
        "home_win": _optional(record["home_win"], bool),
    }
    values.update({name: _to_datetime(record[name]) if name != "graded_at" else _optional(record[name], _to_datetime) for name in _TIMESTAMPS})
    values.update({name: _optional(record[name], float) for name in _FLOATS})
    values.update({name: _optional(record[name], int) for name in _OPTIONAL_INTS})
    return NbaPredictionRow(**values)


def row_to_params(row: NbaPredictionRow) -> dict[str, Any]:
    """Bind parameters for the insert. Enums become their stored string."""
    params = {name: getattr(row, name) for name in COLUMNS}
    params["served_model"] = row.served_model.value
    params["status"] = row.status.value
    for name in _TIMESTAMPS:
        if params[name] is not None:
            params[name] = params[name].astimezone(UTC)
    return params


@dataclass(frozen=True, slots=True)
class DateWindow:
    """Inclusive range of ``game_date``."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"window start {self.start} is after end {self.end}")


@dataclass(frozen=True, slots=True)
class Grade:
    home_score: int
    away_score: int
    home_win: bool
    brier: float


def grade_outcome(p_home_served: float, home_score: int, away_score: int) -> Grade:
    if home_score == away_score:
        raise ValueError("basketball games cannot end tied; refusing to grade a draw")
    home_win = home_score > away_score
    return Grade(home_score, away_score, home_win, (p_home_served - float(home_win)) ** 2)


_INSERT_SQL = text(
    f"""
    insert into {TABLE} ({", ".join(COLUMNS)})
    values ({", ".join(f":{c}" for c in COLUMNS)})
    on conflict (game_id, model_version) do nothing
    """
).bindparams(
    bindparam("game_date", type_=Date),
    bindparam("tipoff_utc", type_=DateTime(timezone=True)),
    bindparam("created_at", type_=DateTime(timezone=True)),
    bindparam("graded_at", type_=DateTime(timezone=True)),
    bindparam("home_win", type_=Boolean),
)

_UNGRADED_SQL = text(
    f"""
    select {", ".join(COLUMNS)} from {TABLE}
    where status = 'scheduled' and game_date between :start and :end
    order by tipoff_utc, game_id
    """
).bindparams(bindparam("start", type_=Date), bindparam("end", type_=Date))

_GRADED_SQL = text(
    f"""
    select {", ".join(COLUMNS)} from {TABLE}
    where status = 'graded' and game_date between :start and :end
    order by tipoff_utc, game_id
    """
).bindparams(bindparam("start", type_=Date), bindparam("end", type_=Date))

_GRADE_SQL = text(
    f"""
    update {TABLE}
    set status = 'graded', home_score = :home_score, away_score = :away_score,
        home_win = :home_win, brier = :brier, graded_at = :graded_at
    where game_id = :game_id and model_version = :model_version and status = 'scheduled'
    """
).bindparams(bindparam("graded_at", type_=DateTime(timezone=True)), bindparam("home_win", type_=Boolean))


def insert_prediction_if_absent(engine: Engine, row: NbaPredictionRow, *, now: datetime) -> InsertOutcome:
    """Insert ``row`` unless one exists for (game_id, model_version) or the game has tipped off."""
    if row.status is not PredictionStatus.SCHEDULED:
        raise ValueError("only scheduled predictions can be inserted")
    if row.tipoff_utc <= now:
        return InsertOutcome.PAST_TIPOFF
    with engine.begin() as connection:
        result = connection.execute(_INSERT_SQL, row_to_params(row))
    return InsertOutcome.INSERTED if result.rowcount == 1 else InsertOutcome.ALREADY_EXISTS


def _load(engine: Engine, statement: Any, window: DateWindow) -> list[NbaPredictionRow]:
    with engine.connect() as connection:
        records = connection.execute(statement, {"start": window.start, "end": window.end}).mappings().all()
    return [row_from_mapping(record) for record in records]


def load_ungraded(engine: Engine, window: DateWindow) -> list[NbaPredictionRow]:
    return _load(engine, _UNGRADED_SQL, window)


def grade_prediction(
    engine: Engine, game_id: int, model_version: str, home_score: int, away_score: int, p_home_served: float, *, now: datetime
) -> bool:
    """Fill the result columns of a scheduled prediction. Returns False when it was already graded or absent."""
    grade = grade_outcome(p_home_served, home_score, away_score)
    params = {
        "game_id": game_id,
        "model_version": model_version,
        "home_score": grade.home_score,
        "away_score": grade.away_score,
        "home_win": grade.home_win,
        "brier": grade.brier,
        "graded_at": now,
    }
    with engine.begin() as connection:
        return connection.execute(_GRADE_SQL, params).rowcount == 1


@dataclass(frozen=True, slots=True)
class GradedSummary:
    """Served probabilities against the stored market probability on the games that have both.

    Losses are per game and aligned, so the caller can bootstrap the paired difference.
    """

    rows: tuple[NbaPredictionRow, ...]
    market_rows: tuple[NbaPredictionRow, ...]

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def n_market(self) -> int:
        return len(self.market_rows)

    @staticmethod
    def _outcome(rows: tuple[NbaPredictionRow, ...]) -> np.ndarray:
        return np.array([bool(r.home_win) for r in rows], dtype=float)

    def served_brier(self, *, market_games_only: bool = False) -> float:
        rows = self.market_rows if market_games_only else self.rows
        return float(np.mean([r.brier for r in rows])) if rows else float("nan")

    def served_accuracy(self, *, market_games_only: bool = False) -> float:
        rows = self.market_rows if market_games_only else self.rows
        return float(np.mean([(r.p_home_served > 0.5) == r.home_win for r in rows])) if rows else float("nan")

    def served_losses(self) -> np.ndarray:
        return np.array([r.brier for r in self.market_rows], dtype=float)

    def market_losses(self) -> np.ndarray:
        p = np.array([r.p_home_market for r in self.market_rows], dtype=float)
        return (p - self._outcome(self.market_rows)) ** 2

    def market_brier(self) -> float:
        return float(self.market_losses().mean()) if self.market_rows else float("nan")

    def market_accuracy(self) -> float:
        if not self.market_rows:
            return float("nan")
        p = np.array([r.p_home_market for r in self.market_rows], dtype=float)
        return float(((p > 0.5) == (self._outcome(self.market_rows) == 1)).mean())


def summarise_graded(rows: list[NbaPredictionRow]) -> GradedSummary:
    graded = tuple(r for r in rows if r.status is PredictionStatus.GRADED)
    return GradedSummary(rows=graded, market_rows=tuple(r for r in graded if r.p_home_market is not None))


def load_graded_summary(engine: Engine, window: DateWindow) -> GradedSummary:
    return summarise_graded(_load(engine, _GRADED_SQL, window))


__all__ = [
    "COLUMNS", "DateWindow", "Grade", "GradedSummary", "InsertOutcome", "NbaPredictionRow",
    "PredictionStatus", "ServedModel", "grade_outcome", "grade_prediction", "insert_prediction_if_absent",
    "load_graded_summary", "load_ungraded", "row_from_mapping", "row_to_params", "summarise_graded",
]
