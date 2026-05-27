from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.settings import load_local_env
from alttez.operations.prediction_memory import rollback_active_model_to_poisson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback the active ALTTEZ production model to the latest Poisson version.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    return parser.parse_args()


def main() -> int:
    load_local_env(override=True)
    args = parse_args()
    result = rollback_active_model_to_poisson(database_url=args.database_url)
    print(f"rollback_to_poisson={result.get('rollback_to_poisson', False)}")
    print(f"activated={result.get('activated', False)}")
    if result.get("version_key"):
        print(f"active_version_key={result['version_key']}")
    if result.get("error"):
        print(f"rollback_error={result['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
