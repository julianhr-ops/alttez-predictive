from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine


def build_certification_audit_row(
    sync_result: dict[str, Any],
    *,
    target_date: date,
    provider_used: str | None,
) -> dict[str, Any]:
    certification_status = str(
        sync_result.get("certification_status")
        or sync_result.get("status")
        or sync_result.get("error_category")
        or "error"
    )
    batch_id = str(sync_result.get("certification_batch_id") or f"{target_date.isoformat()}_{provider_used or 'none'}")

    return {
        "batch_id": batch_id,
        "target_date": target_date,
        "provider_used": provider_used,
        "certification_status": certification_status,
        "certified": certification_status == "certified",
        "total_records": int(sync_result.get("certification_total_records", 0) or 0),
        "certified_records": int(sync_result.get("certification_certified_records", 0) or 0),
        "rejected_records": int(sync_result.get("certification_rejected_records", 0) or 0),
        "coverage_pct": float(sync_result.get("certification_coverage_pct", 0.0) or 0.0),
        "quarantine_reason": sync_result.get("certification_quarantine_reason"),
        "report_path": sync_result.get("certification_report_path"),
        "payload": dict(sync_result),
    }


def upsert_certification_audit(
    engine: Engine,
    audit_table: Table | None,
    sync_result: dict[str, Any],
    *,
    target_date: date,
    provider_used: str | None,
) -> bool:
    if audit_table is None:
        return False

    row = build_certification_audit_row(
        sync_result,
        target_date=target_date,
        provider_used=provider_used,
    )
    row = {key: value for key, value in row.items() if key in audit_table.c}

    with engine.begin() as connection:
        stmt = insert(audit_table).values(row)
        update_map = {
            key: getattr(stmt.excluded, key)
            for key in row
            if key != "batch_id" and key in audit_table.c
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["batch_id"],
            set_=update_map,
        )
        connection.execute(stmt)

    return True
