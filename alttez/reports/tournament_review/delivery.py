from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from alttez.api.db import get_engine, read_dataframe
from alttez.reports.telegram_delivery import send_document, telegram_config


def load_delivery_record(
    tournament_key: str,
    *,
    database_url: str | None = None,
) -> dict[str, Any] | None:
    frame = read_dataframe(
        """
        select tournament_key, delivery_status, delivered_at, evaluated_predictions,
               rating, error_message
        from public.tournament_review_deliveries
        where tournament_key = :tournament_key
        """,
        params={"tournament_key": tournament_key},
        database_url=database_url,
    )
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def record_delivery(
    tournament: dict[str, Any],
    review: dict[str, Any],
    *,
    status: str,
    error_message: str | None = None,
    database_url: str | None = None,
) -> None:
    rating = (review.get("summary", {}).get("rating") or {}).get("score")
    delivered_at = datetime.now(timezone.utc) if status == "delivered" else None
    engine = get_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                insert into public.tournament_review_deliveries (
                    tournament_key, tournament_name, date_from, date_to,
                    delivery_status, report_generated_at, delivered_at,
                    evaluated_predictions, rating, report_payload, error_message
                ) values (
                    :tournament_key, :tournament_name, :date_from, :date_to,
                    :delivery_status, :report_generated_at, :delivered_at,
                    :evaluated_predictions, :rating, cast(:report_payload as jsonb),
                    :error_message
                )
                on conflict (tournament_key) do update set
                    delivery_status = excluded.delivery_status,
                    report_generated_at = excluded.report_generated_at,
                    delivered_at = excluded.delivered_at,
                    evaluated_predictions = excluded.evaluated_predictions,
                    rating = excluded.rating,
                    report_payload = excluded.report_payload,
                    error_message = excluded.error_message
                """
            ),
            {
                "tournament_key": tournament["key"],
                "tournament_name": tournament["name"],
                "date_from": tournament["date_from"],
                "date_to": tournament["date_to"],
                "delivery_status": status,
                "report_generated_at": review["generated_at"],
                "delivered_at": delivered_at,
                "evaluated_predictions": review["coverage"]["evaluated_predictions"],
                "rating": rating,
                "report_payload": json.dumps(review, ensure_ascii=False, default=str),
                "error_message": error_message,
            },
        )


def build_delivery_caption(review: dict[str, Any]) -> str:
    summary = review["summary"]
    rating = summary.get("rating") or {}
    return (
        f"ALTTEZ | Cierre {review['tournament']}\n"
        f"Valoracion: {rating.get('score', 'N/D')}/100 ({rating.get('label', 'Sin muestra')})\n"
        f"1X2: {float(summary.get('accuracy_1x2') or 0) * 100:.1f}% | "
        f"Brier: {float(summary.get('brier_score') or 0):.3f} | "
        f"Cobertura: {float(review['coverage'].get('evaluable_rate') or 0) * 100:.1f}%"
    )


def deliver_tournament_review(
    review: dict[str, Any],
    html_path: str | Path,
) -> tuple[bool, str | None]:
    token, chat_id = telegram_config()
    if not token or not chat_id:
        return False, "Telegram no esta configurado para el brief diario."
    return send_document(
        token,
        chat_id,
        Path(html_path),
        caption=build_delivery_caption(review),
    )
