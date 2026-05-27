from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.settings import load_local_env
from alttez.operations.prediction_memory import activate_model_version_by_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activate a governed ALTTEZ model version for production inference.")
    parser.add_argument("--version-key", required=True, help="Exact version_key from public.model_versions.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--activation-mode", default="manual", help="Activation mode label to persist.")
    return parser.parse_args()


def main() -> int:
    load_local_env(override=True)
    args = parse_args()
    result = activate_model_version_by_key(
        args.version_key,
        database_url=args.database_url,
        activation_mode=args.activation_mode,
    )
    print(f"activated={result.get('activated', False)}")
    if result.get("version_key"):
        print(f"active_version_key={result['version_key']}")
    if result.get("model_family"):
        print(f"active_model_family={result['model_family']}")
    if result.get("error"):
        print(f"activation_error={result['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
