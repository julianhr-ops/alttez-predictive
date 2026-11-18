"""Write the soccer data files of the public page. No database and no secrets.

    python scripts/soccer_site_data.py --out <dir> [--cache-dir <dir>]

Writes ``<out>/data/soccer_ratings.json`` and ``<out>/data/soccer_validation.json``. The results
come from the CC0 CSV that ``scripts/validate_soccer.py`` uses, cached in ``--cache-dir``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttez.evaluation import soccer_backtest as sb
from alttez.inference.serving_policy import SERVING_WINDOW_YEARS
from alttez.site import soccer_data

DEFAULT_CACHE_DIR = _REPO_ROOT / "artifacts" / "soccer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true", help="download the CSV again even if cached")
    args = parser.parse_args(argv)
    data_dir = args.out / "data"
    started = time.monotonic()

    try:
        csv_path = sb.load_results_csv(args.cache_dir, refresh=args.refresh)
    except sb.SoccerDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"results: {csv_path} ({csv_path.stat().st_size} bytes)")

    matches = sb.prepare_matches(csv_path)
    print(f"parsed: {len(matches.frame)} scored matches of {matches.n_rows} rows")

    as_of = datetime.now(timezone.utc).date()
    payload = soccer_data.build_ratings_payload(matches.frame, as_of, source=sb.DATA_LICENSE, window_years=SERVING_WINDOW_YEARS)
    soccer_data.write_json(data_dir / soccer_data.RATINGS_FILENAME, payload)
    print(f"ratings: {payload['n']} teams as of {payload['as_of']}, top {payload['teams'][0]['name']}")

    soccer_data.write_json(data_dir / soccer_data.VALIDATION_FILENAME, soccer_data.load_validation_snapshot())
    print(f"validation: snapshot copied to {data_dir / soccer_data.VALIDATION_FILENAME}")
    print(f"done in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
