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
from alttez.experiments.tournament_oracle import build_probability_matrix, optimize_prediction
from alttez.features.poisson import PoissonEngine
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import (
    fetch_live_fixtures_from_stack,
    ingest_live_matches_for_date,
    list_matches_for_date,
)
from alttez.ingestion.provider_healthcheck import run_healthcheck


def _build_engine(df: pd.DataFrame, target_date: str) -> PoissonEngine:
    engine = PoissonEngine(
        years_window=4,
        prior_matches=8,
        min_history_matches=5,
        recency_half_life_days=365,
        recent_form_matches=6,
    )
    engine.fit(df, target_date=target_date)
    return engine


def _build_result_summary(prob_matrix, team_a: str, team_b: str) -> pd.DataFrame:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    max_goals = prob_matrix.shape[0] - 1

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = prob_matrix[home_goals, away_goals]
            if home_goals > away_goals:
                home_win += probability
            elif home_goals < away_goals:
                away_win += probability
            else:
                draw += probability

    return pd.DataFrame(
        {
            "Mercado": [f"Victoria {team_a}", "Empate", f"Victoria {team_b}"],
            "Probabilidad": [home_win, draw, away_win],
        }
    )


def _pick_fixture(fixtures, engine: PoissonEngine):
    for fixture in fixtures:
        if engine.get_team_profile(fixture.home_team).get("matches", 0) > 0 or engine.get_team_profile(fixture.away_team).get(
            "matches", 0
        ) > 0:
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
        print("live_fetch=empty")
        return 1

    print("\n[historical]")
    historical_loader = InternationalLoader()
    historical_df = historical_loader.fetch_data()
    print(f"historical_rows={len(historical_df)}")

    engine = _build_engine(historical_df, target_date.isoformat())
    fixture = _pick_fixture(fixtures, engine)
    lambda_home, lambda_away = engine.predict_xg(fixture.home_team, fixture.away_team, is_neutral=fixture.is_neutral)
    prob_matrix = build_probability_matrix(lambda_home, lambda_away, max_goals=7)
    recommendations = optimize_prediction(prob_matrix, is_knockout=False).head(5).copy()
    recommendations["EV"] = recommendations["EV"].map(lambda value: round(value, 3))
    result_summary = _build_result_summary(prob_matrix, fixture.home_team, fixture.away_team)
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
    print(f"xg_home={lambda_home}")
    print(f"xg_away={lambda_away}")
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
        if result.get("status") != "success":
            print(f"error_category={result.get('error_category')}")
            print(f"error={result.get('error') or 'n/a'}")
            return 1

        rows = list_matches_for_date(target_date, database_url=database_url)
        print(f"readback_matches={len(rows)}")
        if rows.empty:
            print("readback=empty")
            return 1

        if "provider" in rows.columns:
            print(f"readback_provider={rows.iloc[0]['provider']}")

    print("\n[summary]")
    print("smoke=ok")
    return 0


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
    )


if __name__ == "__main__":
    raise SystemExit(main())
