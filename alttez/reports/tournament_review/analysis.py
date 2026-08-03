from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from alttez.evaluation.metrics import ranked_probability_score
from alttez.evaluation.regulation_points import max_regulation_points, regulation_points
from alttez.inference.calibration import expected_calibration_error
from alttez.operations.prediction_memory import brier_score_1x2, log_loss_1x2


RATING_WEIGHTS = {
    "probability_quality": 0.40,
    "accuracy_1x2": 0.30,
    "calibration": 0.15,
    "scorelines": 0.15,
}
KNOCKOUT_MARKERS = (
    "knockout",
    "round of",
    "last 16",
    "quarter",
    "semi",
    "final",
    "octavos",
    "cuartos",
    "semifinal",
    "eliminatoria",
)
SCORELINE_PATTERN = re.compile(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "t"}


def _parse_scoreline(value: Any) -> tuple[int, int] | None:
    match = SCORELINE_PATTERN.match(str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _is_knockout(stage: Any) -> bool:
    normalized = str(stage or "").strip().lower()
    return any(marker in normalized for marker in KNOCKOUT_MARKERS)


def _normalize_probabilities(row: pd.Series) -> tuple[float, float, float] | None:
    values = [_safe_float(row.get(column)) for column in ("prob_home", "prob_draw", "prob_away")]
    if any(value is None or value < 0 for value in values):
        return None
    total = float(sum(values))
    if total <= 0:
        return None
    return tuple(float(value / total) for value in values)


def _prediction_timing(row: pd.Series) -> str:
    created = pd.to_datetime(row.get("prediction_created_at"), utc=True, errors="coerce")
    kickoff = pd.to_datetime(row.get("kickoff_at"), utc=True, errors="coerce")
    if pd.isna(created):
        return "unverifiable"
    if pd.isna(kickoff):
        return "unverifiable"
    return "valid_pre_match" if created < kickoff else "invalid_after_kickoff"


def _rating_label(score: float) -> str:
    if score >= 85:
        return "Excelente"
    if score >= 70:
        return "Bueno"
    if score >= 55:
        return "Aceptable"
    if score >= 40:
        return "Bajo"
    return "Deficiente"


def _sample_confidence(evaluated: int, coverage: float) -> str:
    if evaluated >= 40 and coverage >= 0.9:
        return "Alta"
    if evaluated >= 15 and coverage >= 0.65:
        return "Media"
    return "Baja"


def _empty_summary() -> dict[str, Any]:
    return {
        "evaluated_predictions": 0,
        "accuracy_1x2": None,
        "exact_score_hit_rate": None,
        "brier_score": None,
        "log_loss": None,
        "rps": None,
        "ece": None,
        "mean_score_abs_error": None,
        "pool_points": 0,
        "pool_points_max": 0,
        "pool_points_rate": None,
        "fallback_rate": None,
        "rating": None,
    }


def _summarize(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return _empty_summary()

    triplets = frame[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    outcomes = frame["actual_outcome"].astype(str).tolist()
    accuracy = float(frame["hit_1x2"].mean())
    exact_rate = float(frame["hit_exact_score"].mean())
    brier = float(frame["brier_1x2"].mean())
    log_loss = float(frame["log_loss_1x2"].mean())
    rps = float(frame["rps_1x2"].mean())
    ece = expected_calibration_error(triplets, outcomes)
    score_error = float(frame["score_abs_error"].mean())
    pool_points = int(frame["pool_points"].sum())
    pool_max = int(frame["pool_points_max"].sum())

    probability_quality = 100.0 * (
        0.6 * (1.0 - min(max(brier, 0.0) / 2.0, 1.0))
        + 0.4 * (1.0 - min(max(rps, 0.0), 1.0))
    )
    calibration_quality = 100.0 * (1.0 - min(max(ece, 0.0) / 0.5, 1.0))
    scoreline_quality = 100.0 * (
        0.7 * exact_rate + 0.3 * (1.0 - min(max(score_error, 0.0) / 6.0, 1.0))
    )
    rating = (
        RATING_WEIGHTS["probability_quality"] * probability_quality
        + RATING_WEIGHTS["accuracy_1x2"] * accuracy * 100.0
        + RATING_WEIGHTS["calibration"] * calibration_quality
        + RATING_WEIGHTS["scorelines"] * scoreline_quality
    )

    return {
        "evaluated_predictions": int(len(frame)),
        "accuracy_1x2": round(accuracy, 4),
        "exact_score_hit_rate": round(exact_rate, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "rps": round(rps, 4),
        "ece": round(float(ece), 4),
        "mean_score_abs_error": round(score_error, 3),
        "pool_points": pool_points,
        "pool_points_max": pool_max,
        "pool_points_rate": round(pool_points / pool_max, 4) if pool_max else None,
        "fallback_rate": round(float(frame["used_fallback"].mean()), 4),
        "rating": {
            "score": round(rating, 1),
            "label": _rating_label(rating),
            "components": {
                "probability_quality": round(probability_quality, 1),
                "accuracy_1x2": round(accuracy * 100.0, 1),
                "calibration": round(calibration_quality, 1),
                "scorelines": round(scoreline_quality, 1),
            },
            "weights": RATING_WEIGHTS,
        },
    }


def _prepare_evaluated(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in frame.iterrows():
        if pd.isna(source.get("actual_outcome")):
            continue
        probabilities = _normalize_probabilities(source)
        predicted_score = _parse_scoreline(source.get("predicted_scoreline"))
        if probabilities is None or predicted_score is None:
            continue
        actual_home = _safe_float(source.get("actual_home_score"))
        actual_away = _safe_float(source.get("actual_away_score"))
        if actual_home is None or actual_away is None:
            continue
        actual_home_int = int(actual_home)
        actual_away_int = int(actual_away)
        actual_outcome = str(source.get("actual_outcome"))
        prob_home, prob_draw, prob_away = probabilities
        predicted_outcome = str(source.get("predicted_outcome"))
        knockout = _is_knockout(source.get("stage"))
        pred_home, pred_away = predicted_score
        rows.append(
            {
                **source.to_dict(),
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "actual_outcome": actual_outcome,
                "hit_1x2": predicted_outcome == actual_outcome,
                "hit_exact_score": pred_home == actual_home_int and pred_away == actual_away_int,
                "brier_1x2": brier_score_1x2(prob_home, prob_draw, prob_away, actual_outcome),
                "log_loss_1x2": log_loss_1x2(prob_home, prob_draw, prob_away, actual_outcome),
                "rps_1x2": ranked_probability_score(prob_home, prob_draw, prob_away, actual_outcome),
                "score_abs_error": abs(pred_home - actual_home_int) + abs(pred_away - actual_away_int),
                "pool_points": regulation_points(
                    pred_home,
                    pred_away,
                    actual_home_int,
                    actual_away_int,
                    is_knockout=knockout,
                ),
                "pool_points_max": max_regulation_points(is_knockout=knockout),
                "used_fallback": _safe_bool(source.get("used_fallback")),
                "stage": str(source.get("stage") or "Sin fase"),
            }
        )
    return pd.DataFrame(rows)


def _group_summaries(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if frame.empty or column not in frame:
        return []
    rows = []
    for value, group in frame.groupby(column, dropna=False):
        rows.append({"name": str(value or "Sin dato"), **_summarize(group)})
    return sorted(rows, key=lambda row: row["evaluated_predictions"], reverse=True)


def _match_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "match_date": str(row.get("match_date") or ""),
                "stage": str(row.get("stage") or "Sin fase"),
                "home_team": str(row.get("home_team") or ""),
                "away_team": str(row.get("away_team") or ""),
                "predicted_scoreline": str(row.get("predicted_scoreline") or ""),
                "actual_scoreline": f"{int(row['actual_home_score'])} - {int(row['actual_away_score'])}",
                "hit_1x2": bool(row["hit_1x2"]),
                "hit_exact_score": bool(row["hit_exact_score"]),
                "brier_1x2": round(float(row["brier_1x2"]), 4),
                "rps_1x2": round(float(row["rps_1x2"]), 4),
                "pool_points": int(row["pool_points"]),
                "pool_points_max": int(row["pool_points_max"]),
                "model_version": str(row.get("model_version") or ""),
                "used_fallback": bool(row["used_fallback"]),
            }
        )
    return rows


def build_tournament_review(
    predictions: pd.DataFrame,
    *,
    tournament: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an auditable tournament closure from persisted pre-match predictions."""

    source = predictions.copy()
    if source.empty:
        return {
            "schema_version": "1.0",
            "tournament": tournament,
            "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
            "coverage": {
                "persisted_predictions": 0,
                "evaluated_predictions": 0,
                "valid_pre_match": 0,
                "unverifiable_timing": 0,
                "invalid_after_kickoff": 0,
                "evaluable_rate": 0.0,
            },
            "summary": _empty_summary(),
            "sample_confidence": "Baja",
            "model_breakdown": [],
            "stage_breakdown": [],
            "matches": [],
            "warnings": ["No se encontraron predicciones persistidas para el torneo."],
        }

    source["timing_status"] = source.apply(_prediction_timing, axis=1)
    persisted = int(len(source))
    match_key = "match_id" if "match_id" in source.columns else "prediction_id"
    unique_matches = int(source[match_key].nunique(dropna=True))
    valid_count = int((source["timing_status"] == "valid_pre_match").sum())
    invalid_count = int((source["timing_status"] == "invalid_after_kickoff").sum())
    unverifiable_count = int((source["timing_status"] == "unverifiable").sum())

    eligible = source[source["timing_status"] != "invalid_after_kickoff"].copy()
    eligible["_created_sort"] = pd.to_datetime(
        eligible["prediction_created_at"], utc=True, errors="coerce"
    )
    eligible = (
        eligible.sort_values("_created_sort", ascending=True, na_position="first")
        .drop_duplicates(subset=[match_key], keep="last")
        .drop(columns=["_created_sort"])
    )
    selected_count = int(len(eligible))
    evaluated = _prepare_evaluated(eligible)
    evaluated_count = int(len(evaluated))
    evaluable_rate = evaluated_count / unique_matches if unique_matches else 0.0
    summary = _summarize(evaluated)
    warnings = []
    if invalid_count:
        warnings.append(
            f"{invalid_count} predicciones creadas durante o despues del kickoff fueron excluidas."
        )
    if unverifiable_count:
        warnings.append(
            f"{unverifiable_count} predicciones no tienen timing verificable y se reportan con esa limitacion."
        )
    if evaluated_count < unique_matches:
        warnings.append(
            f"El cierre evalua {evaluated_count} de {unique_matches} partidos unicos."
        )
    if evaluated_count < 15:
        warnings.append("La muestra es pequena; la valoracion no debe interpretarse como concluyente.")

    return {
        "schema_version": "1.0",
        "tournament": tournament,
        "matched_tournament_names": sorted(source["tournament_name"].dropna().astype(str).unique().tolist()),
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "methodology": {
            "rating_weights": RATING_WEIGHTS,
            "timing_policy": "Se excluyen predicciones creadas durante o despues del kickoff.",
            "rating_note": "La valoracion tecnica no sustituye cobertura ni confianza de muestra.",
        },
        "coverage": {
            "persisted_predictions": persisted,
            "unique_matches": unique_matches,
            "selected_pre_match_predictions": selected_count,
            "evaluated_predictions": evaluated_count,
            "valid_pre_match": valid_count,
            "unverifiable_timing": unverifiable_count,
            "invalid_after_kickoff": invalid_count,
            "evaluable_rate": round(evaluable_rate, 4),
        },
        "summary": summary,
        "sample_confidence": _sample_confidence(evaluated_count, evaluable_rate),
        "model_breakdown": _group_summaries(evaluated, "model_version"),
        "stage_breakdown": _group_summaries(evaluated, "stage"),
        "matches": _match_rows(evaluated),
        "warnings": warnings,
    }
