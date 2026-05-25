from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from alttez.api.settings import load_local_env
from alttez.ingestion.provider_stack import build_stack


def run_healthcheck(explicit_order: str | None = None) -> list[dict[str, str]]:
    load_local_env()
    results: list[dict[str, str]] = []

    for provider in build_stack(explicit_order):
        item = {
            "provider": provider.provider_name,
            "configured": "yes" if provider.is_configured() else "no",
            "detail": "",
        }

        if provider.provider_name == "api_football":
            item["detail"] = "expects ALTTEZ_API_FOOTBALL_KEY"
        elif provider.provider_name == "football_data_org":
            item["detail"] = "expects ALTTEZ_FOOTBALL_DATA_KEY"
        elif provider.provider_name == "statsbomb_open":
            item["detail"] = "expects ALTTEZ_STATSBOMB_OPEN_ROOT"
        elif provider.provider_name == "local_json":
            item["detail"] = "expects ALTTEZ_LOCAL_JSON_PATH"

        results.append(item)

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
