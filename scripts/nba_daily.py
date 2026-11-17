"""Daily NBA stage: refresh data, grade finished predictions, predict upcoming games, print a digest.

    python scripts/nba_daily.py [--date YYYY-MM-DD] [--database-url URL] [--dry-run] [--deliver]

Without a database URL, or with ``--dry-run``, nothing is written: predictions and the
digest are only printed. ``--dry-run`` opens the database (for reads) only when
``--database-url`` is given explicitly. Exit codes: 0 ok, 1 delivery failed, 2 data or
database error, 3 ``stale_data`` (the hoopR schedule lags a season in progress).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from alttez.api.db import get_database_url, get_engine
from alttez.nba import data, predict, store
from alttez.nba.data import NbaDataError
from alttez.reports.nba_digest import DigestInput, build_nba_digest, deliver_digest

DEFAULT_CACHE_DIR = _REPO_ROOT / "artifacts" / "nba"
GRADING_LOOKBACK_DAYS = 3
EARLIEST_GAME_DATE = date(2000, 1, 1)
EXIT_DELIVERY_FAILED = 1
EXIT_DATA_ERROR = 2
EXIT_STALE_DATA = 3

ScoreboardFetch = Callable[[date], list[data.NbaScheduledGame]]


@dataclass(frozen=True, slots=True)
class GradeReport:
    graded: int
    unresolved: int


def refresh_inputs(cache_dir: Path, notes: list[str]) -> predict.NbaInputs:
    """Force a download; on failure fall back to the cached copy and say so."""
    try:
        return predict.load_inputs(cache_dir, max_age_hours=0)
    except NbaDataError as exc:
        notes.append(f"data refresh failed, using cached copy: {exc}")
        return predict.load_inputs(cache_dir, max_age_hours=float("inf"))


def grade_finished(
    engine: Engine, today: date, *, now: datetime, fetch: ScoreboardFetch, write: bool
) -> GradeReport:
    """Grade scheduled predictions whose game finished on the ESPN scoreboard in the last days.

    Idempotent: only rows still ``scheduled`` are loaded, and the update itself is
    conditioned on that status.
    """
    window = store.DateWindow(today - timedelta(days=GRADING_LOOKBACK_DAYS + 1), today + timedelta(days=1))
    pending = store.load_ungraded(engine, window)
    unresolved_old = len(store.load_ungraded(engine, store.DateWindow(EARLIEST_GAME_DATE, window.start - timedelta(days=1))))
    finished: dict[int, data.NbaScheduledGame] = {}
    if pending:
        for offset in range(GRADING_LOOKBACK_DAYS + 1):
            for game in fetch(today - timedelta(days=offset)):
                if game.state == "post" and game.home_score is not None and game.away_score is not None:
                    finished[game.game_id] = game
    graded = 0
    for row in pending:
        game = finished.get(row.game_id)
        if game is None:
            continue
        if not write:
            graded += 1
        elif store.grade_prediction(
            engine, row.game_id, row.model_version, game.home_score, game.away_score, row.p_home_served, now=now
        ):
            graded += 1
    still_open = sum(1 for r in pending if r.game_id not in finished and r.game_date < today)
    return GradeReport(graded=graded, unresolved=unresolved_old + still_open)


def store_predictions(engine: Engine, run: predict.PredictionRun, *, now: datetime) -> dict[store.InsertOutcome, int]:
    counts = {outcome: 0 for outcome in store.InsertOutcome}
    for planned in run.predictions:
        counts[store.insert_prediction_if_absent(engine, planned.row, now=now)] += 1
    return counts


def resolve_engine(args: argparse.Namespace) -> Engine | None:
    if args.dry_run and not args.database_url:
        return None
    try:
        get_database_url(args.database_url)
    except RuntimeError:
        return None
    return get_engine(args.database_url)


def run(
    args: argparse.Namespace, *, fetch: ScoreboardFetch = data.fetch_espn_scoreboard, now: datetime | None = None
) -> tuple[int, str]:
    now = now or datetime.now(UTC)
    today = date.fromisoformat(args.date) if args.date else now.date()
    notes: list[str] = []
    engine = resolve_engine(args)
    write = engine is not None and not args.dry_run
    mode = "write" if write else "dry-run" if args.dry_run else "print-only (no database URL)"
    if engine is None:
        notes.append("no database connection: grading and the live scoreboard are skipped")

    try:
        inputs = refresh_inputs(args.cache_dir, notes)
        report = grade_finished(engine, today, now=now, fetch=fetch, write=write) if engine is not None else GradeReport(0, 0)
        try:
            prediction_run = predict.predict_upcoming(inputs, today=today, now=now, fetch=fetch)
        except predict.StaleDataError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_STALE_DATA, str(exc)
        if write:
            counts = store_predictions(engine, prediction_run, now=now)
            notes.append(
                f"stored {counts[store.InsertOutcome.INSERTED]} new, "
                f"{counts[store.InsertOutcome.ALREADY_EXISTS]} already present, "
                f"{counts[store.InsertOutcome.PAST_TIPOFF]} past tipoff"
            )
        summary = store.load_graded_summary(engine, store.DateWindow(EARLIEST_GAME_DATE, today)) if engine is not None else None
    except (NbaDataError, SQLAlchemyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR, str(exc)

    text = build_nba_digest(
        DigestInput(
            today=today, run=prediction_run, graded_this_run=report.graded, unresolved=report.unresolved,
            summary=summary, notes=notes, mode=mode,
        )
    )
    return 0, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", help="UTC date to run for (default: today)")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--dry-run", action="store_true", help="never write to the database")
    parser.add_argument("--deliver", action="store_true", help="send the digest through the Telegram sender")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)

    code, text = run(args)
    if code != 0:
        return code
    print(text)
    if args.deliver:
        delivered, error = deliver_digest(text)
        if not delivered:
            print(f"ERROR: digest delivery failed: {error}", file=sys.stderr)
            return EXIT_DELIVERY_FAILED
        print("digest delivered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
