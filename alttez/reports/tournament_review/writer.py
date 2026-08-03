from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .render_html import render_tournament_review_html


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "tournament"


def write_tournament_review(
    review: dict[str, Any],
    *,
    output_dir: str | Path = "artifacts/tournament-reviews",
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(str(review['tournament']))}-review"
    json_path = directory / f"{stem}.json"
    html_path = directory / f"{stem}.html"
    json_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    html_path.write_text(render_tournament_review_html(review), encoding="utf-8")
    return {"json": json_path, "html": html_path}
