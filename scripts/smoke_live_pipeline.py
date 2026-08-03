from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.settings import load_local_env
from alttez.dashboard.analysis_helpers import build_recommendation_summary
from alttez.inference import BaseMatchPredictor, build_active_predictor
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import (
    fetch_live_fixtures_from_stack,
    ingest_live_matches_for_date,
    list_matches_for_date,
)
from alttez.ingestion.provider_healthcheck import run_healthcheck

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_NO_SCHEDULE = 3


def _pick_fixture(fixtures, predictor: BaseMatchPredictor):
    for fixture in fixtures:
        if predictor.get_team_profile(fixture.home_team).get("matches", 0) > 0 or predictor.get_team_profile(
            fixture.away_team
        ).get("matches", 0) > 0:
            return fixture
    return fixtures[0]


def _render_attempts(attempts: list[dict[str, object]]) -> None:
    for item in attempts:
        print(f"[attempt] {item['provider']} status={item['status']} detail={item['detail']}")


def run_smoke(
    target_date: date,
    provider_order: str | None,
    database_url: str | None,
    schema: str,
    write: bool,
    skip_analysis: bool = False,
) -> int:
    load_local_env(override=True)

    print("[health]")
    health = run_healthcheck(provider_order)
    for item in health:
        print(
            f"{item['provider']} configured={item['configured']} "
            f"stack_position={item['stack_position']} scope={item['stack_scope']} env={item['env_state']}"
        )

    print("\n[live-fetch]")
    fixtures, raw_records, provider_used, attempts = fetch_live_fixtures_from_stack(target_date, provider_order)
    _render_attempts(attempts)
    print(f"provider_used={provider_used or 'none'}")
    print(f"fixtures={len(fixtures)}")
    print(f"raw_records={len(raw_records)}")
    if not fixtures:
        provider_query_succeeded = any(item.get("status") == "ok" for item in attempts)
        if provider_query_succeeded:
            print("live_fetch=no_schedule")
            print("operational_status=idle")
            return EXIT_NO_SCHEDULE
        print("live_fetch=error")
        print("operational_status=failed")
        return EXIT_FAILURE

    if not skip_analysis:
        print("\n[historical]")
        historical_loader = InternationalLoader()
        historical_df = historical_loader.fetch_data()
        print(f"historical_rows={len(historical_df)}")

        predictor = build_active_predictor(historical_df, target_date, database_url=database_url)
        fixture = _pick_fixture(fixtures, predictor)
        prediction = predictor.predict_match(
            fixture.home_team,
            fixture.away_team,
            is_neutral=fixture.is_neutral,
            is_knockout=False,
            max_goals=7,
        )
        recommendations = prediction.get("recommendations", pd.DataFrame())
        result_summary = prediction.get("result_summary", pd.DataFrame())
        recommendation_summary = build_recommendation_summary(
            {
                "recomendaciones": recommendations,
                "result_summary": result_summary,
            }
        )

        print("\n[analysis]")
        print(f"fixture={fixture.home_team} vs {fixture.away_team}")
        print(f"provider={fixture.provider}")
        print(f"tournament={fixture.tournament_name}")
        print(f"kickoff={fixture.kickoff_at or 'n/a'}")
        print(f"model_version={prediction.get('model_version')}")
        print(f"model_family={prediction.get('model_family')}")
        print(f"used_fallback={prediction.get('used_fallback', False)}")
        if prediction.get("fallback_reason"):
            print(f"fallback_reason={prediction['fallback_reason']}")
        print(f"xg_home={prediction.get('lambda_h')}")
        print(f"xg_away={prediction.get('lambda_a')}")
        print(f"primary_prediction={recommendation_summary.get('primary_score', 'n/a')}")
        print(f"primary_ev={recommendation_summary.get('primary_ev', 'n/a')}")
        print(
            "best_market="
            + (
                str(recommendation_summary["best_market"]["Mercado"])
                if recommendation_summary.get("best_market")
                else "n/a"
            )
        )

    if write:
        print("\n[persist]")
        result = ingest_live_matches_for_date(target_date, database_url=database_url, schema=schema)
        print(f"status={result['status']}")
        print(f"provider_used={result.get('provider_used') or 'none'}")
        print(f"scheduled_matches={result.get('scheduled_matches', 0)}")
        print(f"raw_records={result.get('raw_records', 0)}")
        print(f"certification_status={result.get('certification_status') or 'n/a'}")
        print(f"certification_records={result.get('certification_certified_records', 0)}")
        print(f"certification_coverage_pct={result.get('certification_coverage_pct', 0.0)}")
        if result.get("certification_report_path"):
            print(f"certification_report_path={result['certification_report_path']}")
        print(f"certification_audit_persisted={result.get('certification_audit_persisted', False)}")
        if result.get("certification_audit_error"):
            print(f"certification_audit_error={result['certification_audit_error']}")
        print(f"alerts_total={result.get('alerts_total', 0)}")
        print(f"alerts_persisted={result.get('alerts_persisted', False)}")
        if result.get("alerts_storage_status"):
            print(f"alerts_storage_status={result['alerts_storage_status']}")
        if result.get("alerts_delivery_status"):
            print(f"alerts_delivery_status={result['alerts_delivery_status']}")
        if result.get("alerts_delivery_channel"):
            print(f"alerts_delivery_channel={result['alerts_delivery_channel']}")
        if result.get("alerts_error"):
            print(f"alerts_error={result['alerts_error']}")
        if result.get("status") != "success":
            print(f"error_category={result.get('error_category')}")
            print(f"error={result.get('error') or 'n/a'}")
            if result.get("status") == "empty":
                print("operational_status=idle")
                return EXIT_NO_SCHEDULE
            return EXIT_FAILURE

        rows = list_matches_for_date(target_date, database_url=database_url)
        print(f"readback_matches={len(rows)}")
        if rows.empty:
            print("readback=empty")
            return EXIT_FAILURE

        if "provider" in rows.columns:
            print(f"readback_provider={rows.iloc[0]['provider']}")

    print("\n[summary]")
    print("smoke=ok")
    return EXIT_SUCCESS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end ALTTEZ live pipeline smoke test.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--provider-order", default=None, help="Comma-separated provider priority override.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--schema", default="public", help="Target PostgreSQL schema.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the live agenda into Supabase and verify readback.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip historical model analysis; useful when the daily brief will run inference later.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    return run_smoke(
        target_date=target_date,
        provider_order=args.provider_order,
        database_url=args.database_url,
        schema=args.schema,
        write=args.write,
        skip_analysis=args.skip_analysis,
    )


if __name__ == "__main__":
    raise SystemExit(main())
