"""Backfill real finished results into the canonical ``matches`` table.

Recent tournament fixtures land in ``matches`` as ``scheduled`` rows with a placeholder
``0-0`` score; the actual results were never promoted (the raw->canonical transform gap).
Because ``close_finished_prediction_evaluations`` only scores predictions whose match is
``finished``, the prediction feedback loop stays starved of labels.

This module pulls finished results from a configured provider (football-data.org covers the
FIFA World Cup) and reconciles them into the canonical rows, matching by normalized team pair
within a small date tolerance (provider ``utcDate`` vs local ``match_date`` can differ by a day).
It then re-runs the close sweep so freshly-scored matches become evaluated predictions.

Design: the parsing and matching steps are pure functions so they are unit-testable without a
database or network; only ``backfill_finished_results`` touches IO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable

from sqlalchemy import text

from alttez.api.db import get_engine
from alttez.ingestion.live_matches import canonical_team_name
from alttez.ingestion.providers.base import ProviderFetchRequest
from alttez.ingestion.providers.football_data_org import FootballDataOrgProvider
from alttez.operations.prediction_memory import (
    close_finished_prediction_evaluations,
    reflect_prediction_tables,
)


LOGGER = logging.getLogger(__name__)

FINISHED_PROVIDER_STATUSES = {"FINISHED", "FT", "AET", "PEN"}


def _norm(value: str | None) -> str:
    return canonical_team_name(value or "").strip().lower()


@dataclass(frozen=True)
class FinishedResult:
    """A finished match result, normalized for matching."""

    home_norm: str
    away_norm: str
    match_date: date
    home_score: int
    away_score: int


@dataclass(frozen=True)
class CanonicalRow:
    """A candidate canonical ``matches`` row, normalized for matching."""

    row_id: Any
    home_norm: str
    away_norm: str
    match_date: date


def extract_finished_results(records: Iterable[dict[str, Any]]) -> list[FinishedResult]:
    """Parses provider records into normalized finished results.

    Ignores records that are not finished or lack a full-time score. Pure function.
    """

    results: list[FinishedResult] = []
    for record in records:
        fixture = record.get("fixture") if isinstance(record.get("fixture"), dict) else None
        status = (
            (fixture.get("status") or {}).get("short")
            if fixture
            else record.get("status")
        )
        if str(status) not in FINISHED_PROVIDER_STATUSES:
            continue

        score = record.get("score") if isinstance(record.get("score"), dict) else {}
        full_time = score.get("fullTime") if isinstance(score.get("fullTime"), dict) else {}
        goals = record.get("goals") if isinstance(record.get("goals"), dict) else {}
        home_score = full_time.get("home", goals.get("home"))
        away_score = full_time.get("away", goals.get("away"))
        if home_score is None or away_score is None:
            continue

        home_team = record.get("homeTeam") or {}
        away_team = record.get("awayTeam") or {}
        home_name = home_team.get("name") if isinstance(home_team, dict) else home_team
        away_name = away_team.get("name") if isinstance(away_team, dict) else away_team

        raw_date = str(record.get("utcDate") or record.get("date") or "")[:10]
        try:
            match_date = date.fromisoformat(raw_date)
        except ValueError:
            continue

        results.append(
            FinishedResult(
                home_norm=_norm(home_name),
                away_norm=_norm(away_name),
                match_date=match_date,
                home_score=int(home_score),
                away_score=int(away_score),
            )
        )
    return results


def match_result_to_row(
    result: FinishedResult,
    candidates: Iterable[CanonicalRow],
    *,
    day_tolerance: int = 1,
) -> Any | None:
    """Finds the canonical row matching a result by team pair within a date tolerance.

    Same team pair, closest match_date within ``day_tolerance`` days wins (UTC vs local skew).
    Returns the row id or ``None``. Pure function.
    """

    best_id: Any | None = None
    best_delta: int | None = None
    for row in candidates:
        if row.home_norm != result.home_norm or row.away_norm != result.away_norm:
            continue
        delta = abs((row.match_date - result.match_date).days)
        if delta > day_tolerance:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_id = row.row_id
    return best_id


def _load_candidate_rows(connection, date_from: date, date_to: date) -> list[CanonicalRow]:
    rows = connection.execute(
        text(
            """
            select m.id, th.name as home_name, ta.name as away_name, m.match_date
            from matches m
            join teams th on th.id = m.home_team_id
            join teams ta on ta.id = m.away_team_id
            where m.match_date between :lo and :hi
            """
        ),
        {"lo": (date_from - timedelta(days=1)).isoformat(), "hi": (date_to + timedelta(days=1)).isoformat()},
    ).fetchall()
    return [
        CanonicalRow(row_id=r[0], home_norm=_norm(r[1]), away_norm=_norm(r[2]), match_date=r[3])
        for r in rows
    ]


def _fetch_results(
    source: str,
    date_from: date,
    date_to: date,
    *,
    competition: str,
    leagues: Iterable[str] | None,
) -> list[FinishedResult]:
    """Dispatches to the configured results source. Returns normalized finished results."""

    if source == "football_data":
        provider = FootballDataOrgProvider()
        fetch = provider.fetch_matches(
            ProviderFetchRequest(date_from=date_from, date_to=date_to, competition=competition, extra={})
        )
        return extract_finished_results(fetch.records)
    if source == "espn":
        # Lazy import avoids a circular dependency (espn_results imports FinishedResult/_norm).
        from alttez.ingestion.espn_results import ESPN_DEFAULT_LEAGUES, fetch_espn_finished_results

        return fetch_espn_finished_results(
            date_from, date_to, leagues=leagues or ESPN_DEFAULT_LEAGUES
        )
    raise ValueError(f"Unknown results source: {source!r} (expected 'football_data' or 'espn').")


def backfill_finished_results(
    date_from: date,
    date_to: date,
    *,
    source: str = "espn",
    competition: str = "WC",
    leagues: Iterable[str] | None = None,
    database_url: str | None = None,
    day_tolerance: int = 1,
    apply: bool = False,
) -> dict[str, Any]:
    """Fetches finished results, reconciles them into canonical ``matches``, and closes evaluations.

    ``source`` selects the feed: ``espn`` (keyless, covers friendlies) or ``football_data``
    (free tier, World Cup only — needs ``competition``).
    With ``apply=False`` (default) nothing is written: returns the match plan for inspection.
    With ``apply=True`` updates scores/status and re-runs the close sweep.
    """

    report: dict[str, Any] = {
        "source": source,
        "competition": competition,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "applied": apply,
        "provider_finished": 0,
        "matched": 0,
        "updated": 0,
        "skipped_already_finished": 0,
        "unmatched_results": [],
        "evaluations_closed": 0,
    }

    results = _fetch_results(source, date_from, date_to, competition=competition, leagues=leagues)
    report["provider_finished"] = len(results)

    engine = get_engine(database_url)
    with engine.begin() as connection:
        candidates = _load_candidate_rows(connection, date_from, date_to)
        for result in results:
            row_id = match_result_to_row(result, candidates, day_tolerance=day_tolerance)
            if row_id is None:
                report["unmatched_results"].append(
                    f"{result.home_norm} vs {result.away_norm} {result.match_date.isoformat()}"
                )
                continue
            report["matched"] += 1
            current = connection.execute(
                text("select status, home_score, away_score from matches where id = :id"),
                {"id": row_id},
            ).first()
            already = (
                current is not None
                and current[0] == "finished"
                and current[1] == result.home_score
                and current[2] == result.away_score
            )
            if already:
                report["skipped_already_finished"] += 1
                continue
            if apply:
                connection.execute(
                    text(
                        """
                        update matches
                        set home_score = :hs, away_score = :as_, status = 'finished', updated_at = now()
                        where id = :id
                        """
                    ),
                    {"hs": result.home_score, "as_": result.away_score, "id": row_id},
                )
                report["updated"] += 1

    if apply and report["updated"]:
        engine = get_engine(database_url)
        tables = reflect_prediction_tables(engine)
        close_report = close_finished_prediction_evaluations(engine=engine, tables=tables, target_date=None)
        report["evaluations_closed"] = close_report.get("prediction_evaluations_closed", 0)

    LOGGER.info(
        "results_backfill competition=%s matched=%s updated=%s closed=%s applied=%s",
        competition,
        report["matched"],
        report["updated"],
        report["evaluations_closed"],
        apply,
    )
    return report
