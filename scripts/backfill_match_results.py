"""CLI to backfill finished match results into the canonical store and close evaluations.

Dry-run by default (no writes). Use --apply to persist.

    venv\\Scripts\\python.exe scripts\\backfill_match_results.py --date-from 2026-06-11 --date-to 2026-06-16
    venv\\Scripts\\python.exe scripts\\backfill_match_results.py --date-from 2026-06-11 --date-to 2026-06-16 --apply
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.settings import load_local_env
from alttez.ingestion.results_backfill import backfill_finished_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill finished results into canonical matches.")
    parser.add_argument("--date-from", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--date-to", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument("--source", default="espn", choices=["espn", "football_data"], help="Results feed (default espn, covers friendlies).")
    parser.add_argument("--competition", default="WC", help="football_data competition code (default WC).")
    parser.add_argument("--leagues", default=None, help="Comma-separated ESPN league slugs (default fifa.friendly,fifa.world).")
    parser.add_argument("--day-tolerance", type=int, default=1, help="UTC/local date skew tolerance in days.")
    parser.add_argument("--apply", action="store_true", help="Persist updates and close evaluations.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL.")
    return parser.parse_args()


def main() -> int:
    load_local_env(override=True)
    args = parse_args()
    leagues = [s.strip() for s in args.leagues.split(",") if s.strip()] if args.leagues else None
    report = backfill_finished_results(
        date.fromisoformat(args.date_from),
        date.fromisoformat(args.date_to),
        source=args.source,
        competition=args.competition,
        leagues=leagues,
        database_url=args.database_url,
        day_tolerance=args.day_tolerance,
        apply=args.apply,
    )
    print(f"source={report['source']}")
    print(f"applied={report['applied']}")
    print(f"provider_finished={report['provider_finished']}")
    print(f"matched={report['matched']}")
    print(f"updated={report['updated']}")
    print(f"skipped_already_finished={report['skipped_already_finished']}")
    print(f"evaluations_closed={report['evaluations_closed']}")
    if report["unmatched_results"]:
        print(f"unmatched={len(report['unmatched_results'])}")
        for item in report["unmatched_results"]:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
