from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any
from urllib import error, request
import html

from sqlalchemy import Table, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from alttez.api.settings import load_local_env


logging.basicConfig(level=logging.INFO)


DEFAULT_ALERT_WEBHOOK_TIMEOUT_SECONDS = 8
DEFAULT_ALERTS_TELEGRAM_ENABLED = False


def _normalize_alert_key(value: Any) -> str:
    return str(value).strip().replace(" ", "_").replace(":", "_").replace("/", "_").lower()


def _build_alert(
    *,
    batch_id: str,
    target_date: date,
    provider_used: str | None,
    alert_scope: str,
    alert_level: str,
    alert_code: str,
    title: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "alert_key": _normalize_alert_key(f"{batch_id}:{alert_code}"),
        "batch_id": batch_id,
        "target_date": target_date,
        "provider_used": provider_used,
        "alert_scope": alert_scope,
        "alert_level": alert_level,
        "alert_code": alert_code,
        "title": title,
        "message": message,
        "storage_status": "pending",
        "delivery_channel": "dashboard",
        "delivery_status": "skipped",
        "payload": payload or {},
    }


def build_operational_alerts(
    sync_result: dict[str, Any],
    *,
    target_date: date,
    provider_used: str | None,
) -> list[dict[str, Any]]:
    batch_id = str(sync_result.get("certification_batch_id") or f"{target_date.isoformat()}_{provider_used or 'none'}")
    status = str(sync_result.get("status") or "error")
    certification_status = str(sync_result.get("certification_status") or "skipped")
    attempts = sync_result.get("attempted_providers") or []
    failed_attempts = [item for item in attempts if item.get("status") in {"error", "skipped"}]
    alerts: list[dict[str, Any]] = []

    if status == "error":
        alerts.append(
            _build_alert(
                batch_id=batch_id,
                target_date=target_date,
                provider_used=provider_used,
                alert_scope="sync",
                alert_level="critical",
                alert_code="sync_error",
                title="La sincronizacion diaria fallo",
                message=sync_result.get("error") or "El proceso diario no pudo completar la ingesta live.",
                payload={
                    "error_category": sync_result.get("error_category"),
                    "error": sync_result.get("error"),
                },
            )
        )
    elif status == "empty":
        alerts.append(
            _build_alert(
                batch_id=batch_id,
                target_date=target_date,
                provider_used=provider_used,
                alert_scope="sync",
                alert_level="info",
                alert_code="no_fixtures_found",
                title="No se encontraron partidos para certificar",
                message=(
                    f"No hubo partidos internacionales disponibles para la fecha {target_date.isoformat()}."
                ),
                payload={
                    "provider_used": provider_used,
                },
            )
        )
    else:
        if certification_status == "certified":
            alerts.append(
                _build_alert(
                    batch_id=batch_id,
                    target_date=target_date,
                    provider_used=provider_used,
                    alert_scope="certification",
                    alert_level="info",
                    alert_code="batch_certified",
                    title="Lote certificado correctamente",
                    message=(
                        f"{sync_result.get('certification_certified_records', 0)} registros certificados "
                        f"con cobertura {sync_result.get('certification_coverage_pct', 0.0):.2f}%."
                    ),
                    payload={
                        "certified_records": sync_result.get("certification_certified_records", 0),
                        "coverage_pct": sync_result.get("certification_coverage_pct", 0.0),
                    },
                )
            )
        elif certification_status == "quarantined":
            alerts.append(
                _build_alert(
                    batch_id=batch_id,
                    target_date=target_date,
                    provider_used=provider_used,
                    alert_scope="certification",
                    alert_level="warning",
                    alert_code="batch_quarantined",
                    title="Lote en cuarentena",
                    message=(
                        f"{sync_result.get('certification_certified_records', 0)} registros certificados, "
                        f"{sync_result.get('certification_rejected_records', 0)} rechazados "
                        f"y {sync_result.get('certification_coverage_pct', 0.0):.2f}% de cobertura."
                    ),
                    payload={
                        "certified_records": sync_result.get("certification_certified_records", 0),
                        "rejected_records": sync_result.get("certification_rejected_records", 0),
                        "coverage_pct": sync_result.get("certification_coverage_pct", 0.0),
                        "quarantine_reason": sync_result.get("certification_quarantine_reason"),
                    },
                )
            )
        elif certification_status == "skipped":
            alerts.append(
                _build_alert(
                    batch_id=batch_id,
                    target_date=target_date,
                    provider_used=provider_used,
                    alert_scope="certification",
                    alert_level="info",
                    alert_code="batch_skipped",
                    title="Sin partidos finalizados para certificar",
                    message="La agenda se sincronizo, pero no hubo partidos finalizados para certificar.",
                    payload={
                        "provider_used": provider_used,
                    },
                )
            )

        if failed_attempts:
            alerts.append(
                _build_alert(
                    batch_id=batch_id,
                    target_date=target_date,
                    provider_used=provider_used,
                    alert_scope="provider",
                    alert_level="warning",
                    alert_code="provider_degraded",
                    title="El stack live opero con degradacion parcial",
                    message=(
                        f"{len(failed_attempts)} proveedor(es) reportaron problemas, "
                        "pero el lote pudo continuar con el fallback disponible."
                    ),
                    payload={
                        "failed_attempts": failed_attempts,
                    },
                )
            )

    if sync_result.get("certification_audit_persisted") is False and sync_result.get("status") in {"success", "empty"}:
        alerts.append(
            _build_alert(
                batch_id=batch_id,
                target_date=target_date,
                provider_used=provider_used,
                alert_scope="audit",
                alert_level="warning",
                alert_code="audit_persistence_failed",
                title="La auditoria del lote no quedo persistida",
                message=sync_result.get("certification_audit_error") or "No se pudo guardar el lote en Supabase.",
                payload={
                    "audit_error": sync_result.get("certification_audit_error"),
                },
            )
        )

    return alerts


def _webhook_timeout_seconds() -> int:
    raw_value = os.getenv("ALTTEZ_ALERT_WEBHOOK_TIMEOUT_SECONDS", str(DEFAULT_ALERT_WEBHOOK_TIMEOUT_SECONDS))
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_ALERT_WEBHOOK_TIMEOUT_SECONDS


def dispatch_alert_webhook(alerts: list[dict[str, Any]], webhook_url: str | None = None) -> tuple[bool, str | None]:
    load_local_env(override=False)
    target_url = webhook_url or os.getenv("ALTTEZ_ALERT_WEBHOOK_URL", "").strip()
    if not target_url:
        return False, None

    payload = {
        "source": "alttez",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alerts": alerts,
    }
    data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    request_obj = request.Request(
        target_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(request_obj, timeout=_webhook_timeout_seconds()) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Webhook returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)


def _telegram_enabled() -> bool:
    raw = os.getenv("ALTTEZ_ALERTS_TELEGRAM_ENABLED", str(DEFAULT_ALERTS_TELEGRAM_ENABLED)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _telegram_token() -> str:
    return os.getenv("ALTTEZ_ALERTS_TELEGRAM_BOT_TOKEN", "").strip()


def _telegram_chat_id() -> str:
    return os.getenv("ALTTEZ_ALERTS_TELEGRAM_CHAT_ID", "").strip()


def _telegram_api_url(bot_token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{bot_token}/{method}"


def _send_telegram_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str | None]:
    data = json.dumps(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request_obj = request.Request(
        _telegram_api_url(bot_token, "sendMessage"),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(request_obj, timeout=_webhook_timeout_seconds()) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Telegram returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)


def _build_telegram_alert_text(alerts: list[dict[str, Any]]) -> str:
    critical_alerts = [alert for alert in alerts if str(alert.get("alert_level")) == "critical"]
    warning_alerts = [alert for alert in alerts if str(alert.get("alert_level")) == "warning"]
    headline = "Incidente critico" if critical_alerts else "Alerta operativa"
    primary = critical_alerts[0] if critical_alerts else warning_alerts[0] if warning_alerts else alerts[0]
    target_date = str(primary.get("target_date") or "")
    provider_used = str(primary.get("provider_used") or "N/D")
    lines = [
        f"<b>ALTTEZ | {headline}</b>",
        f"<b>Fecha</b> {html.escape(target_date)}",
        f"<b>Provider</b> {html.escape(provider_used)}",
        f"<b>Evento</b> {html.escape(str(primary.get('title') or 'Incidente'))}",
        html.escape(str(primary.get("message") or "")),
    ]
    if len(alerts) > 1:
        lines.append(f"<b>Alertas en lote</b> {len(alerts)}")
    return "\n".join(lines)


def dispatch_alert_telegram(alerts: list[dict[str, Any]]) -> tuple[bool, str | None]:
    load_local_env(override=False)
    if not _telegram_enabled():
        return False, "Telegram alerts disabled."
    token = _telegram_token()
    chat_id = _telegram_chat_id()
    if not token or not chat_id:
        return False, "Telegram alert destination is not configured."
    telegram_text = _build_telegram_alert_text(alerts)
    return _send_telegram_message(token, chat_id, telegram_text)


def send_pipeline_incident_telegram(*, stage: str, target_date: str, message: str, level: str = "critical") -> tuple[bool, str | None]:
    alert = {
        "title": f"Pipeline fallo en {stage}",
        "message": message,
        "target_date": target_date,
        "provider_used": "pipeline",
        "alert_level": level,
    }
    return dispatch_alert_telegram([alert])


def upsert_alert_events(engine: Engine, alert_table: Table | None, alerts: list[dict[str, Any]]) -> bool:
    if alert_table is None or not alerts:
        return False

    rows = []
    for alert in alerts:
        row = {key: value for key, value in alert.items() if key in alert_table.c}
        rows.append(row)

    if not rows:
        return False

    with engine.begin() as connection:
        stmt = insert(alert_table).values(rows)
        update_map = {
            key: getattr(stmt.excluded, key)
            for key in rows[0]
            if key != "alert_key" and key in alert_table.c
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["alert_key"],
            set_=update_map,
        )
        connection.execute(stmt)

    return True


def record_operational_alerts(
    engine: Engine,
    alert_table: Table | None,
    sync_result: dict[str, Any],
    *,
    target_date: date,
    provider_used: str | None,
    webhook_url: str | None = None,
) -> dict[str, Any]:
    alerts = build_operational_alerts(sync_result, target_date=target_date, provider_used=provider_used)
    if not alerts:
        return {
            "alerts": [],
            "alerts_total": 0,
            "alerts_persisted": False,
            "alerts_storage_status": "skipped",
            "alerts_delivery_status": "skipped",
            "alerts_delivery_channel": "dashboard",
            "alerts_error": None,
        }

    delivery_status = "skipped"
    delivery_channel = "dashboard"
    delivery_error = None
    webhook_configured = bool(webhook_url or os.getenv("ALTTEZ_ALERT_WEBHOOK_URL", "").strip())
    sent = False
    if webhook_configured:
        sent, delivery_error = dispatch_alert_webhook(alerts, webhook_url=webhook_url)
        delivery_channel = "dashboard+webhook"
        delivery_status = "delivered" if sent else "failed"

    should_send_telegram = any(str(alert.get("alert_level")) in {"critical", "warning"} for alert in alerts)
    telegram_sent = False
    telegram_error = None
    if should_send_telegram:
        telegram_sent, telegram_error = dispatch_alert_telegram(alerts)
        if _telegram_enabled():
            delivery_channel = f"{delivery_channel}+telegram" if delivery_channel != "dashboard" else "dashboard+telegram"
            if telegram_sent:
                delivery_status = "delivered"
            elif delivery_status == "skipped":
                delivery_status = "failed"

    persisted = upsert_alert_events(engine, alert_table, alerts)
    storage_status = "skipped" if alert_table is None else ("stored" if persisted else "failed")
    for alert in alerts:
        alert["delivery_channel"] = delivery_channel
        alert["delivery_status"] = delivery_status
        alert["storage_status"] = storage_status

    return {
        "alerts": alerts,
        "alerts_total": len(alerts),
        "alerts_persisted": persisted,
        "alerts_storage_status": storage_status,
        "alerts_delivery_status": delivery_status,
        "alerts_delivery_channel": delivery_channel,
        "alerts_error": (
            delivery_error
            if webhook_configured and not sent and delivery_channel != "dashboard"
            else telegram_error
            if not telegram_sent and should_send_telegram and _telegram_enabled()
            else None
        ),
    }


def load_recent_alert_events(
    *,
    limit: int = 10,
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from alttez.api.db import get_engine

        engine = get_engine(database_url)
    except Exception as exc:  # noqa: BLE001 - dashboard should degrade gracefully.
        logging.warning("Could not load recent alert events: %s", exc)
        return []

    sql = text(
        """
        select
            alert_key,
            batch_id,
            target_date,
            provider_used,
            alert_scope,
            alert_level,
            alert_code,
            title,
            message,
            storage_status,
            delivery_channel,
            delivery_status,
            payload,
            created_at
        from public.alert_events
        order by created_at desc
        limit :limit
        """
    )
    try:
        with engine.connect() as connection:
            rows = connection.execute(sql, {"limit": limit}).mappings().all()
    except Exception as exc:  # noqa: BLE001 - optional panel must not crash the dashboard.
        logging.warning("Could not query recent alert events: %s", exc)
        return []

    return [dict(row) for row in rows]
