from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from alttez.api.settings import load_local_env
from alttez.operations.prediction_memory import persist_daily_predictions
from .constants import DEFAULT_REPORT_DIR
from .delivery import dispatch_daily_brief
from .payload import build_daily_brief_payload
from .reliability import evaluate_delivery_gate
from .writer import write_daily_brief


def generate_daily_brief(
    target_date: date,
    *,
    database_url: str | None = None,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    deliver: bool = False,
    persist_predictions: bool = True,
) -> dict[str, Any]:
    payload = build_daily_brief_payload(target_date, database_url=database_url)
    # Fallback is no longer expected to be silent: if the active model could not load and the
    # brief is served by the Poisson safety net, raise a non-critical incident so ops sees it.
    if payload.get("used_fallback"):
        try:
            from alttez.operations.alerts import send_pipeline_incident_telegram

            send_pipeline_incident_telegram(
                stage="daily brief model serving",
                target_date=target_date.isoformat(),
                message=(
                    "El brief se generó con el modelo de respaldo (Poisson) porque el modelo "
                    f"activo no pudo cargarse. Motivo: {payload.get('fallback_reason') or 'desconocido'}"
                ),
                level="warning",
            )
        except Exception:  # noqa: BLE001 - alerting must never block brief generation.
            pass
    delivery = {
        "delivery_attempted": False,
        "delivery_status": "skipped",
        "delivery_channel": "none",
        "delivery_error": None,
    }
    persistence = {
        "prediction_run_id": None,
        "prediction_run_row_id": None,
        "prediction_persisted": False,
        "prediction_rows": 0,
        "prediction_error": None,
    }
    if persist_predictions:
        try:
            persistence = persist_daily_predictions(payload, database_url=database_url)
            if persistence.get("prediction_run_id"):
                payload["prediction_run_id"] = persistence["prediction_run_id"]
        except Exception as exc:  # noqa: BLE001 - report generation should survive persistence issues.
            persistence["prediction_error"] = str(exc)

    # Delivery gate: never ship a generic report. The gate keeps every match in
    # the brief (each already carries a reliability marker); it only refuses to
    # *send* when the run as a whole is untrustworthy (global Poisson fallback,
    # or no match backed by the active model).
    gate = evaluate_delivery_gate(payload)
    payload["delivery_gate"] = gate

    payload.update(delivery)
    html_path, json_path = write_daily_brief(payload, report_dir=report_dir)

    if deliver:
        delivery["delivery_attempted"] = True
        if not gate["should_deliver"]:
            delivery["delivery_status"] = "blocked"
            delivery["delivery_channel"] = "none"
            delivery["delivery_error"] = gate["block_reason"]
            _raise_gate_incident(target_date, gate)
        else:
            html_content = html_path.read_text(encoding="utf-8")
            delivered, error_message, channel = dispatch_daily_brief(payload, html_content, html_path, json_path)
            delivery["delivery_status"] = "delivered" if delivered else "failed"
            delivery["delivery_channel"] = channel
            delivery["delivery_error"] = error_message
        payload.update(delivery)
        html_path, json_path = write_daily_brief(payload, report_dir=report_dir)

    return {
        "payload": payload,
        "html_path": str(html_path),
        "json_path": str(json_path),
        **persistence,
        **delivery,
    }


def _raise_gate_incident(target_date: date, gate: dict[str, Any]) -> None:
    """Alerts ops that delivery was blocked by the reliability gate. Best-effort."""

    try:
        from alttez.operations.alerts import send_pipeline_incident_telegram

        send_pipeline_incident_telegram(
            stage="daily brief delivery gate",
            target_date=target_date.isoformat(),
            message=(
                "Envío del brief BLOQUEADO por el gate de fiabilidad: "
                f"{gate.get('block_reason') or 'motivo desconocido'}. "
                f"Partidos respaldados por el modelo: {gate.get('model_backed_matches', 0)}/"
                f"{gate.get('total_matches', 0)}."
            ),
            level="critical",
        )
    except Exception:  # noqa: BLE001 - alerting must never block brief generation.
        pass


def exit_code_for_brief_result(result: dict[str, Any], *, deliver_requested: bool) -> int:
    if deliver_requested and result.get("delivery_status") != "delivered":
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the ALTTEZ Daily Decision Brief.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Output directory for the report artifacts.")
    parser.add_argument("--deliver", action="store_true", help="Deliver the brief to the configured webhook.")
    return parser.parse_args()


def main() -> int:
    load_local_env(override=True)
    args = parse_args()
    result = generate_daily_brief(
        date.fromisoformat(args.date),
        database_url=args.database_url,
        report_dir=args.report_dir,
        deliver=args.deliver,
    )
    print(f"daily_brief_html={result['html_path']}")
    print(f"daily_brief_json={result['json_path']}")
    print(f"prediction_run_id={result.get('prediction_run_id') or 'none'}")
    print(f"prediction_persisted={result.get('prediction_persisted', False)}")
    print(f"prediction_rows={result.get('prediction_rows', 0)}")
    print(f"daily_brief_delivery_status={result['delivery_status']}")
    print(f"daily_brief_delivery_channel={result['delivery_channel']}")
    brief_payload = result.get("payload") or {}
    gate = brief_payload.get("delivery_gate") or {}
    print(f"daily_brief_gate_should_deliver={gate.get('should_deliver')}")
    print(f"daily_brief_gate_model_backed={gate.get('model_backed_matches', 0)}/{gate.get('total_matches', 0)}")
    if gate.get("block_reason"):
        print(f"daily_brief_gate_block_reason={gate.get('block_reason')}")
    print(f"daily_brief_model_version={brief_payload.get('model_version') or 'unknown'}")
    print(f"daily_brief_used_fallback={bool(brief_payload.get('used_fallback'))}")
    if brief_payload.get("used_fallback"):
        print(f"daily_brief_fallback_reason={brief_payload.get('fallback_reason') or 'unknown'}")
    if result.get("prediction_error"):
        print(f"prediction_error={result['prediction_error']}")
    if result.get("delivery_error"):
        print(f"daily_brief_delivery_error={result['delivery_error']}")
    return exit_code_for_brief_result(result, deliver_requested=args.deliver)


if __name__ == "__main__":
    raise SystemExit(main())
