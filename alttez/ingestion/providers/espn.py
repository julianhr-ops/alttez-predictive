"""ESPN scoreboard provider for live national-team fixtures (keyless).

ESPN exposes a stable, key-free scoreboard JSON per league per date at
``site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates=YYYYMMDD``.
Unlike football-data.org's free tier it covers both scheduled and finished
fixtures across the World Cup (``fifa.world``), international friendlies
(``fifa.friendly``) and the national-team competitions listed in
``alttez.ingestion.espn_competitions``, which is why it replaces the retired
api-football key in the live agenda stack.

This provider only sweeps the network and returns raw ESPN events, each stamped
with the league/season context needed downstream. Parsing into ``ScheduledMatch``
lives in ``alttez.ingestion.live_matches.parse_espn_records`` (pure function),
keeping fetch and transform decoupled.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

from alttez.ingestion.espn_competitions import ESPN_DEFAULT_LEAGUES, tournament_name
from alttez.ingestion.providers.base import MatchProvider, ProviderFetchRequest, ProviderFetchResult


LOGGER = logging.getLogger(__name__)

ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
DEFAULT_TIMEOUT = 20


def resolve_espn_leagues(explicit: str | None = None) -> list[str]:
    configured = explicit or os.getenv("ALTTEZ_ESPN_LEAGUES", "")
    if configured.strip():
        return [item.strip() for item in configured.split(",") if item.strip()]
    return list(ESPN_DEFAULT_LEAGUES)


def stamp_espn_event(event: dict[str, Any], league_meta: dict[str, Any]) -> dict[str, Any]:
    """Annotates a raw ESPN event with league/season context. Pure function."""

    stamped = dict(event)
    stamped["_alttez_league_name"] = league_meta.get("name")
    stamped["_alttez_league_slug"] = league_meta.get("slug")
    stamped["_alttez_season_year"] = league_meta.get("season_year")
    return stamped


def extract_league_meta(payload: dict[str, Any], league_slug: str) -> dict[str, Any]:
    """Pulls league display name + season year from a scoreboard payload. Pure function.

    The name is the one the competition is stored under (see ``espn_competitions``).
    """

    leagues = payload.get("leagues") or []
    league = leagues[0] if leagues and isinstance(leagues[0], dict) else {}
    season = league.get("season") or {}
    return {
        "name": tournament_name(league_slug, league.get("name")) or "FIFA World Cup",
        "slug": league.get("slug") or league_slug,
        "season_year": season.get("year"),
    }


class EspnProvider(MatchProvider):
    provider_name = "espn"

    def __init__(
        self,
        leagues: list[str] | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.leagues = leagues or resolve_espn_leagues()
        self.base_url = (base_url or os.getenv("ALTTEZ_ESPN_BASE_URL") or ESPN_BASE_URL).rstrip("/")
        self.timeout = timeout or int(os.getenv("ALTTEZ_ESPN_TIMEOUT", str(DEFAULT_TIMEOUT)))

    def is_configured(self) -> bool:
        # Keyless source: enabled by default, opt-out via env for testing/isolation.
        return os.getenv("ALTTEZ_ESPN_ENABLED", "true").strip().lower() != "false"

    def _fetch_scoreboard(self, league: str, day: date) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/{league}/scoreboard",
            params={"dates": day.strftime("%Y%m%d")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected ESPN payload shape: root is not an object.")
        return payload

    def fetch_matches(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        if not self.is_configured():
            raise RuntimeError("ESPN provider is disabled (ALTTEZ_ESPN_ENABLED=false).")

        date_from = request.date_from or date.today()
        date_to = request.date_to or date_from
        leagues = [request.competition] if request.competition else self.leagues

        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        day = date_from
        while day <= date_to:
            for league in leagues:
                try:
                    payload = self._fetch_scoreboard(league, day)
                except Exception as exc:  # noqa: BLE001 - one league/day must not kill the sweep.
                    LOGGER.warning("espn_scoreboard_failed league=%s day=%s detail=%s", league, day.isoformat(), exc)
                    continue
                league_meta = extract_league_meta(payload, league)
                for event in payload.get("events", []):
                    if not isinstance(event, dict):
                        continue
                    event_id = str(event.get("id") or "")
                    if not event_id or event_id in seen:
                        continue
                    seen.add(event_id)
                    records.append(stamp_espn_event(event, league_meta))
            day += timedelta(days=1)

        return ProviderFetchResult(
            provider=self.provider_name,
            records=records,
            fetched_at=datetime.now(timezone.utc),
            metadata={"leagues": leagues},
        )
