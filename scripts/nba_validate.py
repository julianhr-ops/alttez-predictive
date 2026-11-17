"""Walk-forward validation of the NBA models on live free data.

    python scripts/nba_validate.py [--first-test-season 2021] [--n-boot 2000]

Downloads (or reuses a cache younger than 12 h of) the hoopR schedule and closing lines,
fits every model on seasons before each test season only, writes
``artifacts/nba/validation.json`` and prints a markdown report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttez.nba import data
from alttez.nba.data import NbaDataError
from alttez.nba.evaluate import FIRST_TEST_SEASON, render_markdown, run_validation

DEFAULT_CACHE_DIR = _REPO_ROOT / "artifacts" / "nba"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE_DIR / "validation.json")
    parser.add_argument("--first-test-season", type=int, default=FIRST_TEST_SEASON)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        schedule = data.load_schedule(args.cache_dir)
        lines = data.load_closing_lines(args.cache_dir)
    except NbaDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = run_validation(schedule, lines, first_test_season=args.first_test_season, n_boot=args.n_boot, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render_markdown(report))
    print(f"JSON report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
