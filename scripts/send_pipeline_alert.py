from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.operations.alerts import send_pipeline_incident_telegram


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a pipeline incident alert to Telegram.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--level", default="critical")
    args = parser.parse_args()

    sent, error_message = send_pipeline_incident_telegram(
        stage=args.stage,
        target_date=args.date,
        message=args.message,
        level=args.level,
    )
    print(f"pipeline_alert_sent={'true' if sent else 'false'}")
    if error_message:
        print(f"pipeline_alert_error={error_message}")
    return 0 if sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
