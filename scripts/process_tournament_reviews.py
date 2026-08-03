from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.settings import load_local_env  # noqa: E402
from alttez.reports.tournament_review.automation import process_tournament_reviews  # noqa: E402


def clear_loopback_proxy_trap() -> None:
    """Match the daily PowerShell guard for blocked local proxy placeholders."""

    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    for name in names:
        value = os.environ.get(name, "")
        if re.match(r"^https?://(?:127\.0\.0\.1|localhost):9/?$", value, flags=re.IGNORECASE):
            os.environ.pop(name, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Procesa cierres de torneo pendientes.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--registry", default="storage/tournament_reviews.json")
    parser.add_argument("--output-dir", default="artifacts/tournament-reviews")
    parser.add_argument("--database-url")
    parser.add_argument("--deliver", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_local_env()
    clear_loopback_proxy_trap()
    args = parse_args()
    results = process_tournament_reviews(
        as_of=args.date,
        registry_path=args.registry,
        output_dir=args.output_dir,
        database_url=args.database_url,
        deliver=args.deliver,
        force=args.force,
    )
    failed = False
    for result in results:
        print(
            f"tournament_key={result['key']} status={result['status']} "
            f"reason={result.get('reason') or ''} error={result.get('error') or ''}"
        )
        failed = failed or result["status"] == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
