from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alttez.ingestion.providers.base import MatchProvider, ProviderFetchRequest, ProviderFetchResult


class LocalJsonProvider(MatchProvider):
    provider_name = "local_json"

    def __init__(self, input_path: str | Path | None = None) -> None:
        configured = input_path or os.getenv("ALTTEZ_LOCAL_JSON_PATH", "")
        self.input_path = Path(configured) if configured else None

    def is_configured(self) -> bool:
        return self.input_path is not None and self.input_path.exists()

    def fetch_matches(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        if not self.is_configured():
            raise RuntimeError("Local JSON provider is not configured. Set ALTTEZ_LOCAL_JSON_PATH.")

        assert self.input_path is not None
        payload: Any = json.loads(self.input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("response", "matches", "data", "fixtures"):
                value = payload.get(key)
                if isinstance(value, list):
                    payload = value
                    break

        if not isinstance(payload, list):
            raise ValueError("Local JSON payload must be a list or a known wrapper with a list inside.")

        return ProviderFetchResult(
            provider=self.provider_name,
            records=[item for item in payload if isinstance(item, dict)],
            fetched_at=datetime.now(timezone.utc),
            metadata={"path": str(self.input_path)},
        )
