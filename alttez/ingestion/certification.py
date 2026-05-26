from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from alttez.ingestion.contracts import BatchCertification, MatchTeamMetrics, RawMatchPayload


STAT_FIELD_ALIASES: dict[str, str] = {
    "xg": "xg",
    "expectedgoals": "xg",
    "expectedgoalsfor": "xg",
    "expectedgoalsagainst": "xg",
    "possession": "possession",
    "ballpossession": "possession",
    "shotstotal": "shots_total",
    "shots": "shots_total",
    "totalshots": "shots_total",
    "shotsongoal": "shots_on_target",
    "shotsontarget": "shots_on_target",
    "shotsoffgoal": "shots_off_target",
    "shotsofftarget": "shots_off_target",
    "shotsblocked": "shots_blocked",
    "shotsinsidebox": "shots_inside_box",
    "shotssinbox": "shots_inside_box",
    "shotsoutsidebox": "shots_outside_box",
    "corners": "corners",
    "corner": "corners",
    "fouls": "fouls",
    "yellowcards": "yellow_cards",
    "yellowcard": "yellow_cards",
    "redcards": "red_cards",
    "redcard": "red_cards",
    "offsides": "offsides",
    "offside": "offsides",
    "saves": "saves",
    "goalkeepersaves": "saves",
    "keepersaves": "saves",
    "passescompleted": "passes_completed",
    "completedpasses": "passes_completed",
    "passaccuracy": "pass_accuracy",
    "passingaccuracy": "pass_accuracy",
}


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _coerce_metric_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _extract_metrics_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in mapping.items():
        normalized = _normalize_key(key)
        field_name = STAT_FIELD_ALIASES.get(normalized)
        if field_name is not None:
            metrics[field_name] = value
    return metrics


def _extract_metrics_from_statistics_list(statistics: Iterable[Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for item in statistics:
        if not isinstance(item, dict):
            continue

        metric_value = item.get("value")
        if metric_value is None:
            continue

        metric_type = item.get("type") or item.get("name") or item.get("metric")
        field_name = STAT_FIELD_ALIASES.get(_normalize_key(metric_type))
        if field_name is not None:
            metrics[field_name] = metric_value
    return metrics


def _select_statistics_blob(payload: dict[str, Any], team_name: str | None, side: str) -> Any:
    candidates = (
        payload.get(f"{side}_statistics"),
        payload.get(f"{side}_stats"),
        payload.get(f"{side}_metrics"),
        payload.get(f"{side}_summary"),
        payload.get("statistics"),
        payload.get("stats"),
        payload.get("metrics"),
        payload.get("match_statistics"),
    )

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, dict):
            if team_name:
                direct_team = candidate.get(team_name)
                if isinstance(direct_team, (dict, list)):
                    return direct_team
            side_value = candidate.get(side)
            if isinstance(side_value, (dict, list)):
                return side_value
            return candidate
        if isinstance(candidate, list):
            if not candidate:
                continue
            first_item = candidate[0]
            if isinstance(first_item, dict) and "statistics" in first_item:
                for item in candidate:
                    if not isinstance(item, dict):
                        continue
                    team_block = item.get("team")
                    if isinstance(team_block, dict):
                        candidate_team_name = team_block.get("name") or team_block.get("team_name")
                        candidate_team_side = team_block.get("side") or item.get("side")
                        if team_name and candidate_team_name and str(candidate_team_name).strip().casefold() == team_name.strip().casefold():
                            return item.get("statistics") or item
                        if candidate_team_side and str(candidate_team_side).strip().casefold() == side.casefold():
                            return item.get("statistics") or item
                return candidate[0].get("statistics") if isinstance(candidate[0], dict) else candidate
            return candidate

    return None


def _build_team_metrics(payload: dict[str, Any], team_name: str | None, side: str) -> MatchTeamMetrics | None:
    blob = _select_statistics_blob(payload, team_name, side)
    if blob is None:
        return None

    metrics_payload: dict[str, Any]
    if isinstance(blob, dict):
        metrics_payload = _extract_metrics_from_mapping(blob)
    elif isinstance(blob, list):
        if blob and all(isinstance(item, dict) and "value" in item for item in blob):
            metrics_payload = _extract_metrics_from_statistics_list(blob)
        else:
            metrics_payload = {}
            for item in blob:
                if isinstance(item, dict):
                    metrics_payload.update(_extract_metrics_from_mapping(item))
                    nested_statistics = item.get("statistics")
                    if isinstance(nested_statistics, list):
                        metrics_payload.update(_extract_metrics_from_statistics_list(nested_statistics))
    else:
        metrics_payload = {}

    if not metrics_payload:
        return None

    return MatchTeamMetrics.model_validate(metrics_payload)


def build_raw_match_payload(
    match: Any,
    raw_record: Any,
    *,
    source_payload: dict[str, Any] | None = None,
) -> RawMatchPayload:
    payload = source_payload or getattr(raw_record, "payload", {}) or {}
    home_metrics = _build_team_metrics(payload, getattr(match, "home_team", None), "home")
    away_metrics = _build_team_metrics(payload, getattr(match, "away_team", None), "away")

    return RawMatchPayload.model_validate(
        {
            "provider": getattr(match, "provider", None),
            "external_match_id": getattr(match, "external_match_id", None),
            "match_date": getattr(match, "match_date", None),
            "kickoff_at": getattr(match, "kickoff_at", None),
            "tournament_name": getattr(match, "tournament_name", None),
            "home_team": getattr(match, "home_team", None),
            "away_team": getattr(match, "away_team", None),
            "status": getattr(match, "status", "scheduled"),
            "home_score": getattr(match, "home_score", None),
            "away_score": getattr(match, "away_score", None),
            "season_label": getattr(match, "season_label", None),
            "stage": getattr(match, "stage", None),
            "competition_class": getattr(match, "competition_class", None),
            "is_neutral": getattr(match, "is_neutral", False),
            "home_metrics": home_metrics.model_dump() if home_metrics is not None else None,
            "away_metrics": away_metrics.model_dump() if away_metrics is not None else None,
            "payload": payload,
        }
    )


def build_daily_certification_report(
    raw_payloads: Iterable[RawMatchPayload],
    *,
    batch_id: str,
    target_date: date,
    provider_used: str | None = None,
) -> BatchCertification:
    return BatchCertification.certify_batch(
        list(raw_payloads),
        batch_id=batch_id,
        target_date=target_date,
        provider_used=provider_used,
    )


def write_certification_report(report: BatchCertification, report_dir: str | Path = "artifacts/certification") -> Path:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_file = path / f"{report.batch_id}.json"
    output_file.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return output_file

