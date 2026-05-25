from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from alttez.ingestion.providers.base import MatchProvider, ProviderFetchRequest, ProviderFetchResult


class FootballDataOrgProvider(MatchProvider):
    provider_name = "football_data_org"
    base_url = "https://api.football-data.org/v4"
    timeout = 30

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALTTEZ_FOOTBALL_DATA_KEY", "")
        self.base_url = (base_url or os.getenv("ALTTEZ_FOOTBALL_DATA_BASE_URL") or self.base_url).rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def fetch_matches(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        if not self.is_configured():
            raise RuntimeError("football-data.org is not configured. Set ALTTEZ_FOOTBALL_DATA_KEY.")

        if not request.competition:
            raise ValueError("football-data.org requires a competition code, e.g. WC or PL.")

        params: dict[str, Any] = {}
        if request.date_from:
            params["dateFrom"] = request.date_from.isoformat()
        if request.date_to:
            params["dateTo"] = request.date_to.isoformat()
        if request.season:
            params["season"] = request.season
        params.update(request.extra)

        response = requests.get(
            f"{self.base_url}/competitions/{request.competition}/matches",
            headers={
                "X-Auth-Token": self.api_key,
                "Accept": "application/json",
            },
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        records = payload.get("matches", [])
        if not isinstance(records, list):
            raise ValueError("Unexpected football-data.org payload shape: matches is not a list.")

        return ProviderFetchResult(
            provider=self.provider_name,
            records=[record for record in records if isinstance(record, dict)],
            fetched_at=datetime.now(timezone.utc),
            metadata={
                "result_set": payload.get("resultSet"),
                "competition": payload.get("competition"),
                "filters": payload.get("filters"),
            },
        )
