from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.settings import load_local_env
from alttez.inference import build_active_predictor
from alttez.inference.world_cup_2026 import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORT_DIR,
    load_world_cup_2026_config,
    simulate_world_cup_2026,
    write_world_cup_2026_report,
)
from alttez.ingestion.international import InternationalLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official-format FIFA World Cup 2026 simulator.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Prediction cutoff date in YYYY-MM-DD format.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="World Cup 2026 configuration JSON path.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Artifact directory for simulation outputs.")
    parser.add_argument("--iterations", type=int, default=500, help="Monte Carlo iterations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> int:
    load_local_env(override=True)
    args = parse_args()
    historical_df = InternationalLoader().fetch_data()
    predictor = build_active_predictor(historical_df, args.date, database_url=args.database_url)
    config = load_world_cup_2026_config(args.config)
    report = simulate_world_cup_2026(
        predictor,
        config=config,
        iterations=args.iterations,
        random_seed=args.seed,
    )
    output_path = write_world_cup_2026_report(report, report_dir=args.report_dir)
    print(f"world_cup_report_path={output_path}")
    print(f"world_cup_model_version={report['model_version']}")
    print(f"world_cup_model_family={report['model_family']}")
    print(f"world_cup_used_fallback={report.get('used_fallback', False)}")
    top_team = report["team_probabilities"][0] if report.get("team_probabilities") else None
    if top_team:
        print(f"world_cup_top_favorite={top_team['team']}")
        print(f"world_cup_top_favorite_champion_probability={top_team['champion_probability']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
