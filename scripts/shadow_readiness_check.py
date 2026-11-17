from __future__ import annotations

import argparse
import ast
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.settings import DEFAULT_ENV_PATH, load_local_env
from alttez.api.migrations import discover_migration_files, render_migration_plan
from alttez.features.poisson import PoissonEngine
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import fetch_live_fixtures_from_stack
from alttez.ingestion.provider_healthcheck import run_healthcheck


REQUIRED_ENV_KEYS = (
    "SUPABASE_DB_URL",
)
PLACEHOLDERS = {"", "replace-me", "your_api_key_here", "changeme"}


def _secret_state(key: str) -> str:
    value = os.getenv(key, "")
    normalized = value.strip().lower()
    if normalized in PLACEHOLDERS:
        return "missing-or-placeholder"
    return f"set(len={len(value.strip())})"


def _check_env() -> bool:
    print("[env]")
    print(f"cwd={Path.cwd()}")
    print(f"default_env_path={DEFAULT_ENV_PATH}")
    print(f"default_env_exists={DEFAULT_ENV_PATH.exists()}")
    ok = DEFAULT_ENV_PATH.exists()

    load_local_env(override=True)
    for key in REQUIRED_ENV_KEYS:
        state = _secret_state(key)
        print(f"{key}={state}")
        ok = ok and state != "missing-or-placeholder"
    return ok


def _check_delivery() -> bool:
    print("\n[delivery]")
    deliver_enabled = os.getenv("ALTTEZ_DAILY_BRIEF_DELIVER", "").strip().lower() == "true"
    delivery_channel = os.getenv("ALTTEZ_DAILY_BRIEF_DELIVERY_CHANNEL", "").strip().lower() or "auto"
    telegram_token = _secret_state("ALTTEZ_DAILY_BRIEF_TELEGRAM_BOT_TOKEN")
    telegram_chat_id = _secret_state("ALTTEZ_DAILY_BRIEF_TELEGRAM_CHAT_ID")
    webhook_url = _secret_state("ALTTEZ_DAILY_BRIEF_WEBHOOK_URL")

    telegram_ready = telegram_token != "missing-or-placeholder" and telegram_chat_id != "missing-or-placeholder"
    webhook_ready = webhook_url != "missing-or-placeholder"
    effective_channel = "telegram" if telegram_ready and delivery_channel in {"telegram", "auto"} else (
        "webhook" if webhook_ready and delivery_channel in {"webhook", "auto"} else "none"
    )

    print(f"daily_brief_deliver={'true' if deliver_enabled else 'false'}")
    print(f"daily_brief_channel={delivery_channel}")
    print(f"telegram_bot_token={telegram_token}")
    print(f"telegram_chat_id={telegram_chat_id}")
    print(f"daily_brief_webhook={webhook_url}")
    print(f"effective_delivery_channel={effective_channel}")

    if not deliver_enabled:
        return True
    if delivery_channel == "telegram":
        return telegram_ready
    if delivery_channel == "webhook":
        return webhook_ready
    return telegram_ready or webhook_ready


def _check_python_sources() -> bool:
    print("\n[syntax]")
    paths = (
        ROOT / "alttez" / "api" / "settings.py",
        ROOT / "alttez" / "features" / "poisson.py",
        ROOT / "alttez" / "ingestion" / "contracts.py",
        ROOT / "alttez" / "ingestion" / "certification.py",
        ROOT / "alttez" / "ingestion" / "certification_audit.py",
        ROOT / "alttez" / "operations" / "alerts.py",
        ROOT / "alttez" / "operations" / "expansion_policy.py",
        ROOT / "alttez" / "operations" / "prediction_memory.py",
        ROOT / "alttez" / "reports" / "daily_brief" / "__init__.py",
        ROOT / "alttez" / "reports" / "daily_brief" / "cli.py",
        ROOT / "alttez" / "reports" / "daily_brief" / "payload.py",
        ROOT / "alttez" / "reports" / "daily_brief" / "formatting.py",
        ROOT / "alttez" / "reports" / "daily_brief" / "render_html.py",
        ROOT / "alttez" / "reports" / "daily_brief" / "delivery.py",
        ROOT / "alttez" / "reports" / "daily_brief" / "writer.py",
        ROOT / "alttez" / "training" / "challenger_cycle.py",
        ROOT / "alttez" / "ingestion" / "providers" / "api_football.py",
        ROOT / "alttez" / "ingestion" / "providers" / "espn.py",
        ROOT / "alttez" / "ingestion" / "live_matches.py",
        ROOT / "alttez" / "dashboard" / "app.py",
        ROOT / "scripts" / "send_pipeline_alert.py",
    )
    ok = True
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            print(f"{path.relative_to(ROOT)}=ok")
        except FileNotFoundError:
            print(f"{path.relative_to(ROOT)}=missing")
            ok = False
        except SyntaxError as exc:
            print(f"{path.relative_to(ROOT)}=syntax-error:{exc}")
            ok = False
    return ok


def _check_providers() -> bool:
    print("\n[providers]")
    ok = True
    for item in run_healthcheck("espn,football_data_org,local_json"):
        print(f"{item['provider']} configured={item['configured']}")
        if item["provider"] == "espn":
            ok = ok and item["configured"] == "yes"
    return ok


def _check_migrations() -> bool:
    print("\n[migrations]")
    migrations_dir = ROOT / "sql" / "supabase" / "migrations"
    if not migrations_dir.exists():
        print(f"migrations_dir=missing path={migrations_dir}")
        return False

    migration_files = discover_migration_files(migrations_dir)
    for item in render_migration_plan(migration_files):
        print(item)

    if not migration_files:
        print("migration_count=0")
        return False

    print(f"migration_count={len(migration_files)}")
    return True


def _check_model() -> bool:
    print("\n[model]")
    csv_path = ROOT / "storage" / "network" / "international_results.csv"
    data_source = "local_csv"
    try:
        loader = InternationalLoader()
        df = loader.fetch_data()
        data_source = "international_loader"
    except Exception as exc:
        print(f"historical_loader=error detail={exc}")
        if not csv_path.exists():
            print(f"historical_csv=missing path={csv_path}")
            return False
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["home_score", "away_score"])

    engine = PoissonEngine(
        years_window=4,
        prior_matches=8,
        min_history_matches=5,
        recency_half_life_days=365,
    )
    engine.fit(df, target_date=date.today().isoformat())

    home_team = "Nigeria"
    away_team = "India"
    home_profile = engine.get_team_profile(home_team)
    away_profile = engine.get_team_profile(away_team)
    xg_home, xg_away = engine.predict_xg(home_team, away_team, is_neutral=False)

    print(f"historical_source={data_source}")
    print(f"historical_rows={len(df)}")
    print(f"{home_team}_matches={int(home_profile.get('matches', 0))}")
    print(f"{away_team}_matches={int(away_profile.get('matches', 0))}")
    print(f"sample_xg={home_team}:{xg_home} {away_team}:{xg_away}")
    return home_profile.get("matches", 0) > 0 and away_profile.get("matches", 0) > 0


def _check_live_api(target_date: date) -> bool:
    print("\n[live-api]")
    try:
        fixtures, _, provider_used, attempts = fetch_live_fixtures_from_stack(target_date)
    except Exception as exc:
        print(f"live_api=error detail={exc}")
        return False

    for item in attempts:
        print(f"{item['provider']} status={item['status']} detail={item['detail']}")
    print(f"provider_used={provider_used or 'none'}")
    print(f"fixtures={len(fixtures)}")
    return any(item["status"] == "ok" for item in attempts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ALTTEZ shadow-test readiness checks.")
    parser.add_argument("--live", action="store_true", help="Call the configured live API providers.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Live API date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    checks = [
        ("env", _check_env()),
        ("delivery", _check_delivery()),
        ("syntax", _check_python_sources()),
        ("providers", _check_providers()),
        ("migrations", _check_migrations()),
        ("model", _check_model()),
    ]
    if args.live:
        checks.append(("live-api", _check_live_api(target_date)))

    print("\n[summary]")
    for name, ok in checks:
        print(f"{name}={'ok' if ok else 'fail'}")

    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
