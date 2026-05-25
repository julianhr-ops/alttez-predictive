from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alttez.ingestion.providers.base import MatchProvider, ProviderFetchRequest, ProviderFetchResult


class StatsBombOpenProvider(MatchProvider):
    provider_name = "statsbomb_open"

    def __init__(self, root_path: str | Path | None = None) -> None:
        configured_root = root_path or os.getenv("ALTTEZ_STATSBOMB_OPEN_ROOT", "")
        self.root_path = Path(configured_root) if configured_root else None

    def is_configured(self) -> bool:
        return self.root_path is not None and self.root_path.exists()

    def fetch_matches(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        if not self.is_configured():
            raise RuntimeError(
                "StatsBomb Open Data is not configured. Set ALTTEZ_STATSBOMB_OPEN_ROOT to a local clone/path."
            )

        candidates = self._resolve_match_files(request)
        records: list[dict[str, Any]] = []
        for path in candidates:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records.extend(item for item in payload if isinstance(item, dict))

        return ProviderFetchResult(
            provider=self.provider_name,
            records=records,
            fetched_at=datetime.now(timezone.utc),
            metadata={"files_read": [str(path) for path in candidates]},
        )

    def _resolve_match_files(self, request: ProviderFetchRequest) -> list[Path]:
        assert self.root_path is not None
        matches_root = self.root_path / "data" / "matches"
        if not matches_root.exists():
            raise RuntimeError(f"StatsBomb matches directory not found: {matches_root}")

        if request.competition and request.season:
            path = matches_root / str(request.competition) / f"{request.season}.json"
            if not path.exists():
                raise RuntimeError(f"StatsBomb season file not found: {path}")
            return [path]

        return sorted(matches_root.glob("*/*.json"))
