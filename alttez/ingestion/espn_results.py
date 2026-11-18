"""ESPN scoreboard results source (keyless) for national-team matches.

ESPN exposes an undocumented but stable scoreboard JSON per league per date at
``site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates=YYYYMMDD``.
It requires no API key and — crucially — covers international friendlies
(``fifa.friendly``), which football-data.org's free tier does not. This fills the
result-ingestion gap that starves the prediction feedback loop of labels.

Parsing is a pure function; only ``fetch_*`` touches the network.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Iterable

import requests

from alttez.ingestion.espn_competitions import ESPN_DEFAULT_LEAGUES
from alttez.ingestion.results_backfill import FinishedResult, _norm


LOGGER = logging.getLogger(__name__)

ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
# National-team competitions that map to our prediction set. Friendlies are the key gap.
ESPN_FINISHED_STATE = "post"
_TIMEOUT = 20


def extract_espn_finished_results(events: Iterable[dict[str, Any]]) -> list[FinishedResult]:
    """Parses ESPN scoreboard ``events`` into normalized finished results. Pure function."""

    results: list[FinishedResult] = []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        state = (((competition.get("status") or {}).get("type")) or {}).get("state")
        if state != ESPN_FINISHED_STATE:
            continue
        competitors = competition.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            home_score = int(home.get("score"))
            away_score = int(away.get("score"))
        except (TypeError, ValueError):
            continue
        raw_date = str(competition.get("date") or event.get("date") or "")[:10]
        try:
            match_date = date.fromisoformat(raw_date)
        except ValueError:
            continue
        results.append(
            FinishedResult(
                home_norm=_norm((home.get("team") or {}).get("displayName")),
                away_norm=_norm((away.get("team") or {}).get("displayName")),
                match_date=match_date,
                home_score=home_score,
                away_score=away_score,
            )
        )
    return results


def fetch_espn_scoreboard(league: str, day: date, *, timeout: int = _TIMEOUT) -> list[dict[str, Any]]:
    """Fetches the raw events list for one league on one date."""

    response = requests.get(
        f"{ESPN_BASE_URL}/{league}/scoreboard",
        params={"dates": day.strftime("%Y%m%d")},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    events = payload.get("events", [])
    return [event for event in events if isinstance(event, dict)]


def fetch_espn_finished_results(
    date_from: date,
    date_to: date,
    *,
    leagues: Iterable[str] = ESPN_DEFAULT_LEAGUES,
    timeout: int = _TIMEOUT,
) -> list[FinishedResult]:
    """Sweeps each league across each date in the range and returns finished results.

    Deduplicated by (home, away, date) to avoid the same fixture appearing in two leagues.
    """

    seen: set[tuple[str, str, date]] = set()
    results: list[FinishedResult] = []
    leagues = list(leagues)
    day = date_from
    while day <= date_to:
        for league in leagues:
            try:
                events = fetch_espn_scoreboard(league, day, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - one league/day must not kill the sweep.
                LOGGER.warning("espn_fetch_failed league=%s day=%s detail=%s", league, day.isoformat(), exc)
                continue
            for result in extract_espn_finished_results(events):
                key = (result.home_norm, result.away_norm, result.match_date)
                if key in seen:
                    continue
                seen.add(key)
                results.append(result)
        day += timedelta(days=1)
    return results
