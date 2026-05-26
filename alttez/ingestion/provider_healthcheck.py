from __future__ import annotations

import argparse
import json

from alttez.api.settings import load_local_env
from alttez.ingestion.live_matches import get_live_provider_order
from alttez.ingestion.provider_stack import build_stack


PLACEHOLDER_KEYS = {
    "",
    "replace-me",
    "your_api_key_here",
    "changeme",
}


def _secret_state(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in PLACEHOLDER_KEYS:
        return "missing"
    return "set"


def run_healthcheck(explicit_order: str | None = None) -> list[dict[str, str]]:
    load_local_env(override=True)
    results: list[dict[str, str]] = []
    live_order = get_live_provider_order()
    stack_order = [item.strip() for item in (explicit_order or ",".join(live_order)).split(",") if item.strip()]

    for index, provider in enumerate(build_stack(explicit_order), start=1):
        item = {
            "provider": provider.provider_name,
            "configured": "yes" if provider.is_configured() else "no",
            "detail": "",
            "stack_position": str(index),
            "stack_scope": "live" if provider.provider_name in live_order else "raw_only",
            "env_state": "n/a",
        }

        if provider.provider_name == "api_football":
            item["detail"] = "expects ALTTEZ_API_FOOTBALL_KEY"
            item["env_state"] = _secret_state(getattr(provider, "api_key", ""))
        elif provider.provider_name == "football_data_org":
            item["detail"] = "expects ALTTEZ_FOOTBALL_DATA_KEY"
            item["env_state"] = _secret_state(getattr(provider, "api_key", ""))
        elif provider.provider_name == "statsbomb_open":
            item["detail"] = "expects ALTTEZ_STATSBOMB_OPEN_ROOT"
            item["env_state"] = "set" if getattr(provider, "root_path", None) else "missing"
        elif provider.provider_name == "local_json":
            item["detail"] = "expects ALTTEZ_LOCAL_JSON_PATH"
            item["env_state"] = "set" if getattr(provider, "input_path", None) else "missing"

        results.append(item)

    known = {item["provider"] for item in results}
    for index, provider_name in enumerate(stack_order, start=1):
        if provider_name not in known:
            results.append(
                {
                    "provider": provider_name,
                    "configured": "no",
                    "detail": "unknown provider in configured order",
                    "stack_position": str(index),
                    "stack_scope": "live" if provider_name in live_order else "raw_only",
                    "env_state": "n/a",
                }
            )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check which providers in the ALTTEZ stack are configured.")
    parser.add_argument("--provider-order", default=None, help="Comma-separated provider priority override.")
    parser.add_argument("--json", action="store_true", help="Emit the healthcheck as JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_healthcheck(args.provider_order)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for item in results:
            print(f"{item['provider']}: configured={item['configured']} | {item['detail']}")
