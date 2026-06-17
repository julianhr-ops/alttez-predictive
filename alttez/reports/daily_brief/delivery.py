from __future__ import annotations

import html
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from alttez.dashboard.analysis_helpers import safe_text
from .constants import DEFAULT_DELIVERY_TIMEOUT_SECONDS
from .formatting import _fmt_float, _fmt_pct


def _delivery_timeout_seconds() -> int:
    timeout_raw = os.getenv("ALTTEZ_DAILY_BRIEF_DELIVERY_TIMEOUT_SECONDS", "").strip()
    if not timeout_raw:
        timeout_raw = os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_TIMEOUT_SECONDS", "10").strip() or "10"
    try:
        return max(1, int(timeout_raw))
    except ValueError:
        return DEFAULT_DELIVERY_TIMEOUT_SECONDS


def _telegram_delivery_config() -> tuple[str | None, str | None]:
    token = os.getenv("ALTTEZ_DAILY_BRIEF_TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("ALTTEZ_DAILY_BRIEF_TELEGRAM_CHAT_ID", "").strip()
    return (token or None, chat_id or None)


def _delivery_channel() -> str:
    configured = os.getenv("ALTTEZ_DAILY_BRIEF_DELIVERY_CHANNEL", "").strip().lower()
    if configured in {"telegram", "webhook", "auto"}:
        return configured
    return "auto"


def _executive_decision_label(match: dict[str, Any]) -> str:
    decision_tier = safe_text(match.get("decision_tier")).lower()
    support_label = safe_text(match.get("support_label")).lower()
    if decision_tier == "alta prioridad":
        return "Decision principal"
    if decision_tier == "seguimiento" or support_label == "soporte parcial":
        return "Vigilancia"
    if support_label == "analizable":
        return "Decision principal"
    return "Descarte"


def _build_telegram_summary_text(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    batch = payload["batch_summary"]
    ranked_matches = summary.get("ranked_matches", [])
    certification_status = safe_text(batch.get("certification_status"), "unknown").upper()
    provider_used = safe_text(batch.get("provider_used"), "N/D")
    model_family = safe_text(payload.get("model_family"), "poisson")
    model_version = safe_text(payload.get("model_version"), "ALTTEZ")
    fallback_suffix = " · fallback" if payload.get("used_fallback") else ""
    status_icon = "🟢" if certification_status == "CERTIFIED" else "🟠"
    critical_alerts = summary.get("critical_alerts", 0)
    risk_icon = "🚨" if critical_alerts else ("⚠️" if summary.get("warning_alerts", 0) else "✅")
    lines = [
        f"🏟️ <b>ALTTEZ | Agenda Ejecutiva</b> · {payload['target_date']}",
        "",
        f"{status_icon} <b>Lote</b> {html.escape(certification_status)} · Cobertura {html.escape(_fmt_float(summary.get('coverage_pct')))}% · {html.escape(provider_used)}",
        f"🤖 <b>Modelo</b> {html.escape(model_family)} / {html.escape(model_version)}{fallback_suffix}",
        f"📅 <b>Agenda</b> {summary.get('total_matches', 0)} partidos · {summary.get('analyzable_matches', 0)} analizables · {summary.get('high_priority_matches', 0)} alta prioridad",
        f"{risk_icon} <b>Riesgo</b> {critical_alerts} criticas · {summary.get('warning_alerts', 0)} advertencias",
    ]
    if batch.get("quarantine_reason"):
        lines.extend(["", f"🚫 <b>Bloqueo lote</b> {html.escape(safe_text(batch.get('quarantine_reason')))}"])

    if ranked_matches:
        lines.extend(["", "🎯 <b>Top decisiones</b>"])
        for index, match in enumerate(ranked_matches, start=1):
            executive_decision = _executive_decision_label(match)
            lines.append(
                (
                    f"{index}. <b>{html.escape(match['home_team'])} vs {html.escape(match['away_team'])}</b>\n"
                    f"    🧭 {html.escape(executive_decision)} · {html.escape(safe_text(match.get('support_label'), 'N/D'))}\n"
                    f"    💡 {html.escape(safe_text(match.get('best_market'), 'N/D'))} "
                    f"{_fmt_pct(match.get('best_market_probability'))}\n"
                    f"    🔢 Score {html.escape(safe_text(match.get('primary_score'), 'N/D'))} · "
                    f"EV {_fmt_float(match.get('primary_ev'), digits=3)}"
                )
            )
    else:
        lines.extend(["", "Sin partidos analizables para hoy."])

    lines.extend(["", "📎 <b>Adjuntos</b> brief HTML y payload JSON"])
    return "\n".join(lines)


def _telegram_api_url(bot_token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{bot_token}/{method}"


def _send_telegram_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str | None]:
    data = parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = request.Request(
        _telegram_api_url(bot_token, "sendMessage"),
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=_delivery_timeout_seconds()) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Telegram sendMessage returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)


def _build_multipart_payload(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----ALTTEZBoundary{datetime.now(timezone.utc).timestamp()}".replace(".", "")
    file_bytes = file_path.read_bytes()
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks: list[bytes] = []

    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), boundary


def _send_telegram_document(bot_token: str, chat_id: str, file_path: Path, caption: str | None = None) -> tuple[bool, str | None]:
    fields = {"chat_id": chat_id}
    if caption:
        fields["caption"] = caption
        fields["parse_mode"] = "HTML"
    payload, boundary = _build_multipart_payload(fields, "document", file_path)
    req = request.Request(
        _telegram_api_url(bot_token, "sendDocument"),
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=_delivery_timeout_seconds()) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Telegram sendDocument returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)


def _dispatch_daily_brief_webhook(payload: dict[str, Any], html_content: str, webhook_url: str | None = None) -> tuple[bool, str | None]:
    target_url = webhook_url or os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_URL", "").strip()
    if not target_url:
        return False, "No daily brief webhook configured."

    body = {
        "source": "alttez-daily-brief",
        "generated_at": payload["generated_at"],
        "target_date": payload["target_date"],
        "summary": payload["summary"],
        "batch_summary": payload["batch_summary"],
        "alert_summary": payload["alert_summary"],
        "matches": payload["matches"],
        "html_report": html_content,
    }
    data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
    timeout_raw = os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_TIMEOUT_SECONDS", "10").strip() or "10"
    try:
        timeout = max(1, int(timeout_raw))
    except ValueError:
        timeout = 10

    req = request.Request(
        target_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Webhook returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)


def _dispatch_daily_brief_telegram(payload: dict[str, Any], html_path: Path, json_path: Path) -> tuple[bool, str | None]:
    bot_token, chat_id = _telegram_delivery_config()
    if not bot_token or not chat_id:
        return False, "Telegram bot token or chat id is not configured."

    summary_text = _build_telegram_summary_text(payload)
    sent, error_message = _send_telegram_message(bot_token, chat_id, summary_text)
    if not sent:
        return False, error_message

    html_sent, html_error = _send_telegram_document(
        bot_token,
        chat_id,
        html_path,
        caption=f"<b>ALTTEZ brief</b> {html.escape(payload['target_date'])}",
    )
    if not html_sent:
        return False, html_error

    json_sent, json_error = _send_telegram_document(bot_token, chat_id, json_path)
    if not json_sent:
        return False, json_error

    return True, None


def dispatch_daily_brief(
    payload: dict[str, Any],
    html_content: str,
    html_path: Path,
    json_path: Path,
    webhook_url: str | None = None,
) -> tuple[bool, str | None, str]:
    delivery_channel = _delivery_channel()
    telegram_token, telegram_chat_id = _telegram_delivery_config()

    if delivery_channel in {"telegram", "auto"} and telegram_token and telegram_chat_id:
        delivered, error_message = _dispatch_daily_brief_telegram(payload, html_path, json_path)
        return delivered, error_message, "telegram"

    delivered, error_message = _dispatch_daily_brief_webhook(payload, html_content, webhook_url=webhook_url)
    if delivered or delivery_channel == "webhook" or webhook_url or os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_URL", "").strip():
        return delivered, error_message, "webhook"

    return False, "No Telegram or webhook delivery channel configured.", "none"


