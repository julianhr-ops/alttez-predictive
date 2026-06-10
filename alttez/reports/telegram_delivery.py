"""Shared Telegram delivery helpers.

Centralises Telegram bot config, message/document sending and message chunking so
the daily brief and the world-cup brief can deliver through the same code path.
Uses only the standard library (urllib) to avoid extra dependencies.
"""

from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, parse, request


DEFAULT_DELIVERY_TIMEOUT_SECONDS = 10
TELEGRAM_MESSAGE_LIMIT = 4096


def telegram_config() -> tuple[str | None, str | None]:
    """Returns ``(bot_token, chat_id)`` from the daily-brief Telegram env vars."""

    token = os.getenv("ALTTEZ_DAILY_BRIEF_TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("ALTTEZ_DAILY_BRIEF_TELEGRAM_CHAT_ID", "").strip()
    return (token or None, chat_id or None)


def delivery_timeout_seconds() -> int:
    timeout_raw = os.getenv("ALTTEZ_DAILY_BRIEF_DELIVERY_TIMEOUT_SECONDS", "").strip()
    if not timeout_raw:
        timeout_raw = os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_TIMEOUT_SECONDS", "10").strip() or "10"
    try:
        return max(1, int(timeout_raw))
    except ValueError:
        return DEFAULT_DELIVERY_TIMEOUT_SECONDS


def _api_url(bot_token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{bot_token}/{method}"


def chunk_messages(text: str, *, limit: int = 3900) -> list[str]:
    """Split a long message into Telegram-safe chunks (<= ``limit`` chars).

    Splits on blank-line section boundaries when possible, falling back to hard line
    splits so no chunk ever exceeds the limit. ``limit`` defaults below the 4096 hard
    cap to leave room for HTML entities.
    """

    if limit <= 0:
        raise ValueError("limit must be positive.")
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.rstrip("\n"))
        current = ""

    for block in text.split("\n\n"):
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= limit:
            current = candidate
            continue
        flush()
        if len(block) <= limit:
            current = block
            continue
        # Single block too large: split line by line.
        for line in block.split("\n"):
            line_candidate = line if not current else f"{current}\n{line}"
            if len(line_candidate) <= limit:
                current = line_candidate
            else:
                flush()
                # As a last resort hard-slice an over-long single line.
                while len(line) > limit:
                    chunks.append(line[:limit])
                    line = line[limit:]
                current = line
    flush()
    return chunks or [text[:limit]]


def send_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str | None]:
    data = parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = request.Request(
        _api_url(bot_token, "sendMessage"),
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=delivery_timeout_seconds()) as response:
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
            (f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n').encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), boundary


def send_document(bot_token: str, chat_id: str, file_path: Path, caption: str | None = None) -> tuple[bool, str | None]:
    fields = {"chat_id": chat_id}
    if caption:
        fields["caption"] = caption
        fields["parse_mode"] = "HTML"
    payload, boundary = _build_multipart_payload(fields, "document", Path(file_path))
    req = request.Request(
        _api_url(bot_token, "sendDocument"),
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=delivery_timeout_seconds()) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Telegram sendDocument returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)
