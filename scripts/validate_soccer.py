"""Walk-forward validation of the soccer 1X2 stack on the free international results.

    python scripts/validate_soccer.py [--first-test-year 2018] [--n-boot 2000] [--workers 8]

Downloads (or reuses ``artifacts/soccer/results.csv``) the CC0 results, fits every model on
years before each test year only, writes ``artifacts/soccer/validation.json`` and prints a
markdown report.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttez.evaluation import soccer_backtest as sb

DEFAULT_CACHE_DIR = _REPO_ROOT / "artifacts" / "soccer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE_DIR / "validation.json")
    parser.add_argument("--refresh", action="store_true", help="download the CSV again even if cached")
    parser.add_argument("--first-test-year", type=int, default=sb.FIRST_TEST_YEAR)
    parser.add_argument("--window-years", type=int, default=sb.WINDOW_YEARS)
    parser.add_argument("--refit-days", type=int, default=sb.REFIT_DAYS)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="processes used to predict years in parallel")
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=None,
        help="reuse or store the raw Poisson output of every predicted year here (keyed by data and cadence)",
    )
    parser.add_argument(
        "--report-years",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="score and report only these years (inclusive); every test year is still predicted",
    )
    parser.add_argument("--select", action="store_true", help="choose the candidate on the reported years (must end by 2022)")
    parser.add_argument("--candidates", nargs="+", choices=sb.CANDIDATE_NAMES, help="candidate models to report (default all)")
    args = parser.parse_args(argv)

    try:
        csv_path = sb.load_results_csv(args.cache_dir, refresh=args.refresh)
    except sb.SoccerDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    matches = sb.prepare_matches(csv_path)

    options = dict(
        first_test_year=args.first_test_year,
        window_years=args.window_years,
        refit_days=args.refit_days,
        n_boot=args.n_boot,
        seed=args.seed,
        report_years=tuple(args.report_years) if args.report_years else None,
        candidates=args.candidates or sb.CANDIDATE_NAMES,
        select=args.select,
    )
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            report = sb.run_backtest(matches, map_fn=pool.map, raw_cache_dir=args.raw_cache_dir, **options)
    else:
        report = sb.run_backtest(matches, raw_cache_dir=args.raw_cache_dir, **options)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(sb.render_markdown(report))
    print(f"JSON report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
