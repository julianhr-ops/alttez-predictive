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


def _paired_line(label: str, paired: dict) -> str:
    return (
        f"{label}: mean_diff={paired['mean_diff']:+.4f} "
        f"ci=[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}] n={paired['n']}"
    )


def summary_lines(result: dict) -> list[str]:
    lines = [f"challenger_artifact_path={result['artifact_path']}"]
    if result.get("production_config_error"):
        lines.append(f"CYCLE_REFUSED production_config_error={result['production_config_error']}")
    else:
        config = result["production_config"]
        lines.append(
            f"production_champion={config['model_version']} rho={config['rho']} poisson_kwargs={config['poisson_kwargs']}"
        )
        if result.get("model_artifact_path"):
            lines.append(f"challenger_model_artifact_path={result['model_artifact_path']}")
        lines.append(f"champion_brier={result['champion']['brier_score']:.6f}")
        lines.append(f"challenger_brier={result['challenger']['brier_score']:.6f}")
        lines.append(_paired_line("challenger_minus_champion_brier", result["paired_comparison"]))
        lines.append(f"min_promotion_matches={result['min_promotion_matches']}")
        for name, baseline in result["baselines"].items():
            lines.append(f"baseline_{name}_brier={baseline['brier_score']:.6f}")
            lines.append(_paired_line(f"  champion_minus_{name}", baseline["champion_vs_baseline"]))
            lines.append(_paired_line(f"  challenger_minus_{name}", baseline["challenger_vs_baseline"]))
        ensemble = result.get("ensemble") or {}
        if ensemble:
            lines.append(f"ensemble_brier={ensemble.get('brier_score', float('nan')):.6f}")
            lines.append(f"ensemble_blend_weight={ensemble.get('blend_weight')}")
    lines.append(f"promotion_decision={result['promotion_decision']}")
    lines.append(f"promotion_reason={result['promotion_reason']}")
    lines.append(f"production_status={result['production_status']}")
    lines.append(f"activation_mode={result['activation_mode']}")
    lines.append(f"governance_reason={result['governance_reason']}")
    return lines


def main() -> int:
    load_local_env(override=True)
    args = parse_args()
    result = run_challenger_cycle(
        date.fromisoformat(args.date),
        database_url=args.database_url,
        report_dir=args.report_dir,
        evaluation_matches=args.evaluation_matches,
    )
    for line in summary_lines(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
