from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(slots=True)
class ProviderFetchRequest:
    date_from: date | None = None
    date_to: date | None = None
    competition: str | None = None
    season: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderFetchResult:
    provider: str
    records: list[dict[str, Any]]
    fetched_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class MatchProvider:
    provider_name: str = ""

    def is_configured(self) -> bool:
        raise NotImplementedError

    def fetch_matches(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        raise NotImplementedError
