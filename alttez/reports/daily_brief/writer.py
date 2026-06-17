from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_REPORT_DIR
from .render_html import render_daily_brief_html


def write_daily_brief(payload: dict[str, Any], report_dir: str | Path = DEFAULT_REPORT_DIR) -> tuple[Path, Path]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    stem = f"daily-brief-{payload['target_date']}"
    html_path = report_path / f"{stem}.html"
    json_path = report_path / f"{stem}.json"
    html_path.write_text(render_daily_brief_html(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return html_path, json_path


