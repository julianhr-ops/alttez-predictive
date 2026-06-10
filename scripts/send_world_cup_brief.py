from __future__ import annotations

import argparse
import json
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
    predict_world_cup_2026_bracket,
    simulate_world_cup_2026,
)
from alttez.ingestion.international import InternationalLoader
from alttez.reports.telegram_delivery import (
    send_document,
    send_message,
    telegram_config,
)
from alttez.reports.world_cup_brief import (
    build_world_cup_brief_text,
    build_world_cup_summary_text,
    render_world_cup_brief_html,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and deliver the full World Cup 2026 tournament prediction brief.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Prediction cutoff date (YYYY-MM-DD).")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="World Cup 2026 configuration JSON path.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Output directory for the brief artifacts.")
    parser.add_argument("--iterations", type=int, default=300, help="Monte Carlo iterations for the odds context.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the Monte-Carlo odds.")
    parser.add_argument("--deliver", action="store_true", help="Send the brief to Telegram.")
    parser.add_argument("--dry-run", action="store_true", help="Build and print/save the brief without sending.")
    return parser.parse_args()


def main() -> int:
    # Windows consoles default to cp1252 and choke on emoji in the brief preview.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    load_local_env(override=True)
    args = parse_args()

    historical_df = InternationalLoader().fetch_data()
    predictor = build_active_predictor(historical_df, args.date, database_url=args.database_url)
    config = load_world_cup_2026_config(args.config)

    bracket = predict_world_cup_2026_bracket(predictor, config=config)
    odds_report = simulate_world_cup_2026(
        predictor,
        config=config,
        iterations=args.iterations,
        random_seed=args.seed,
    )
    odds = odds_report.get("team_probabilities")
    summary_text = build_world_cup_summary_text(bracket, odds=odds)
    full_text = build_world_cup_brief_text(bracket, odds=odds)
    html_report = render_world_cup_brief_html(bracket, odds=odds)

    output_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "world-cup-2026-brief.txt"
    html_path = output_dir / "world-cup-2026-brief.html"
    bracket_path = output_dir / "world-cup-2026-bracket.json"
    text_path.write_text(full_text, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    bracket_path.write_text(json.dumps(bracket, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"world_cup_brief_champion={bracket.get('champion')}")
    print(f"world_cup_brief_runner_up={bracket.get('runner_up')}")
    print(f"world_cup_brief_model_family={bracket.get('model_family')}")
    print(f"world_cup_brief_html_path={html_path}")
    print(f"world_cup_brief_bracket_path={bracket_path}")

    deliver = args.deliver and not args.dry_run
    if not deliver:
        print("=== DRY RUN (no Telegram delivery) ===")
        print(summary_text)
        print(f"\n[HTML report written to {html_path}]")
        print("delivery_status=skipped")
        return 0

    token, chat_id = telegram_config()
    if not token or not chat_id:
        print("delivery_status=failed")
        print("delivery_error=Telegram bot token o chat id no configurados.")
        return 1

    # Short caption message; the full tournament report is delivered as a branded HTML file.
    ok, err = send_message(token, chat_id, summary_text)
    if not ok:
        print("delivery_status=failed")
        print(f"delivery_error=mensaje resumen: {err}")
        return 1

    ok, err = send_document(token, chat_id, html_path, caption="ALTTEZ | Predicción Mundial 2026 (HTML)")
    if not ok:
        print("delivery_status=failed")
        print(f"delivery_error=adjunto HTML: {err}")
        return 1

    ok, err = send_document(token, chat_id, bracket_path, caption="Cuadro Mundial 2026 (datos JSON)")
    if not ok:
        print("delivery_status=failed")
        print(f"delivery_error=adjunto JSON: {err}")
        return 1

    print("delivery_status=delivered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
