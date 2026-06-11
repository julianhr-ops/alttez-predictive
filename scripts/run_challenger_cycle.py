from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.settings import load_local_env
from alttez.training.challenger_cycle import run_challenger_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ALTTEZ champion vs challenger batch cycle.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--report-dir", default="artifacts/model-evals", help="Output directory for challenger artifacts.")
    parser.add_argument("--evaluation-matches", type=int, default=250, help="Number of recent matches to evaluate.")
    return parser.parse_args()


def main() -> int:
    load_local_env(override=True)
    args = parse_args()
    result = run_challenger_cycle(
        date.fromisoformat(args.date),
        database_url=args.database_url,
        report_dir=args.report_dir,
        evaluation_matches=args.evaluation_matches,
    )
    print(f"challenger_artifact_path={result['artifact_path']}")
    if result.get("model_artifact_path"):
        print(f"challenger_model_artifact_path={result['model_artifact_path']}")
    print(f"champion_brier={result['champion']['brier_score']:.6f}")
    print(f"challenger_brier={result['challenger']['brier_score']:.6f}")
    ensemble = result.get("ensemble") or {}
    if ensemble:
        print(f"ensemble_brier={ensemble.get('brier_score', float('nan')):.6f}")
        print(f"ensemble_blend_weight={ensemble.get('blend_weight')}")
    print(f"promotion_decision={result['promotion_decision']}")
    print(f"promotion_reason={result['promotion_reason']}")
    print(f"production_status={result['production_status']}")
    print(f"activation_mode={result['activation_mode']}")
    print(f"governance_reason={result['governance_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
