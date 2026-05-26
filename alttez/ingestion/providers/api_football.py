from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from alttez.ingestion.providers.base import MatchProvider, ProviderFetchRequest, ProviderFetchResult


PLACEHOLDER_KEYS = {
    "",
    "replace-me",
    "your_api_key_here",
    "changeme",
}


class ApiFootballProvider(MatchProvider):
    provider_name = "api_football"
    base_url = "https://v3.football.api-sports.io"
    timeout = 30

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        use_system_proxy: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALTTEZ_API_FOOTBALL_KEY", "")
        self.base_url = (base_url or os.getenv("ALTTEZ_API_FOOTBALL_BASE_URL") or self.base_url).rstrip("/")
        configured_proxy_mode = os.getenv("ALTTEZ_USE_SYSTEM_PROXY", "false").strip().lower() == "true"
        self.use_system_proxy = configured_proxy_mode if use_system_proxy is None else use_system_proxy

    def is_configured(self) -> bool:
        normalized = (self.api_key or "").strip().lower()
        return normalized not in PLACEHOLDER_KEYS

    def fetch_matches(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        if not self.is_configured():
            raise RuntimeError("API-Football is not configured. Set ALTTEZ_API_FOOTBALL_KEY.")

        params: dict[str, Any] = {}
        if request.date_from and request.date_to and request.date_from == request.date_to:
            params["date"] = request.date_from.isoformat()
        else:
            if request.date_from:
                params["from"] = request.date_from.isoformat()
            if request.date_to:
                params["to"] = request.date_to.isoformat()

        if request.competition:
            params["league"] = request.competition
        if request.season:
            params["season"] = request.season
        params.update(request.extra)

        session = requests.Session()
        session.trust_env = self.use_system_proxy
        response = session.get(
            f"{self.base_url}/fixtures",
            headers={
                "x-apisports-key": self.api_key,
                "Accept": "application/json",
            },
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("errors") or []
        if isinstance(errors, dict):
            error_message = "; ".join(f"{key}: {value}" for key, value in errors.items() if value)
            if error_message:
                raise ValueError(f"API-Football returned errors: {error_message}")
        elif isinstance(errors, list) and errors:
            raise ValueError(f"API-Football returned errors: {'; '.join(str(item) for item in errors)}")

        records = payload.get("response", [])
        if not isinstance(records, list):
            raise ValueError("Unexpected API-Football payload shape: response is not a list.")

        return ProviderFetchResult(
            provider=self.provider_name,
            records=[record for record in records if isinstance(record, dict)],
            fetched_at=datetime.now(timezone.utc),
            metadata={
                "results": payload.get("results"),
                "paging": payload.get("paging"),
                "parameters": payload.get("parameters"),
                "errors": errors,
                "rate_limit_remaining": response.headers.get("x-ratelimit-requests-remaining"),
                "burst_limit_remaining": response.headers.get("X-Ratelimit-Remaining"),
            },
        )
