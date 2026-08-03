"""Per-match reliability scoring and the delivery gate for the daily brief.

Two responsibilities, both pure (no I/O):

1. ``compute_match_reliability`` turns the signals already present in a match
   brief (historical support, effective sample size, model confidence,
   conviction of the top market, fallback usage, committee discordance) into a
   single honest reliability marker: a 0-100 score, a Spanish label, and the
   list of reasons behind it. Every match always gets one — including
   low-support fixtures — so the brief never ships an unlabelled number.

2. ``evaluate_delivery_gate`` decides whether the run as a whole is trustworthy
   enough to deliver. It never drops a match (all fixtures matter); it only
   blocks the *whole* send when the report would be generic — i.e. the active
   model could not load and the entire run fell back to the Poisson safety net,
   or not a single match is backed by a real model. Ops gets told why.
"""

from __future__ import annotations

from typing import Any


RELIABILITY_HIGH = "Alta"
RELIABILITY_MEDIUM = "Media"
RELIABILITY_LOW = "Baja"
RELIABILITY_NONE = "Sin soporte confiable"

# Score thresholds for the label bands.
_HIGH_THRESHOLD = 70
_MEDIUM_THRESHOLD = 45

# A fair 1X2 coin is ~0.33; conviction is only credited above that floor.
_MARKET_FLOOR = 1.0 / 3.0
_EFFECTIVE_MATCHES_TARGET = 20.0


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _support_component(min_effective: float) -> float:
    # Up to 40 points, linear in the weaker side's effective sample size.
    return _clamp(min_effective / _EFFECTIVE_MATCHES_TARGET, 0.0, 1.0) * 40.0


def _confidence_component(avg_confidence: float) -> float:
    # Up to 30 points from the model's own confidence in the two teams.
    return _clamp(avg_confidence, 0.0, 1.0) * 30.0


def _conviction_component(best_market_probability: float) -> float:
    # Up to 30 points for how far the top market clears a fair coin.
    span = 1.0 - _MARKET_FLOOR
    normalized = (best_market_probability - _MARKET_FLOOR) / span if span else 0.0
    return _clamp(normalized, 0.0, 1.0) * 30.0


def _label_for_score(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return RELIABILITY_HIGH
    if score >= _MEDIUM_THRESHOLD:
        return RELIABILITY_MEDIUM
    return RELIABILITY_LOW


def compute_match_reliability(match_brief: dict[str, Any], *, run_used_fallback: bool = False) -> dict[str, Any]:
    """Builds the always-present reliability marker for a single match brief. Pure function."""

    supported = bool(match_brief.get("prediction_supported"))
    used_fallback = bool(match_brief.get("used_fallback")) or run_used_fallback
    discordance = bool(match_brief.get("discordance_warning"))

    effective_local = float(match_brief.get("effective_matches_local") or 0.0)
    effective_visitante = float(match_brief.get("effective_matches_visitante") or 0.0)
    min_effective = min(effective_local, effective_visitante)
    avg_confidence = (
        float(match_brief.get("confidence_local") or 0.0) + float(match_brief.get("confidence_visitante") or 0.0)
    ) / 2.0
    best_market_probability = float(match_brief.get("best_market_probability") or 0.0)

    reasons: list[str] = []

    if not supported:
        reasons.append(match_brief.get("support_message") or "Sin historial compatible suficiente.")
        return {
            "score": 0,
            "label": RELIABILITY_NONE,
            "is_model_backed": False,
            "used_fallback": used_fallback,
            "reasons": reasons,
        }

    score = (
        _support_component(min_effective)
        + _confidence_component(avg_confidence)
        + _conviction_component(best_market_probability)
    )

    if min_effective < 10:
        reasons.append(f"Muestra efectiva limitada ({min_effective:.0f} partidos en el lado más débil).")
    if used_fallback:
        score *= 0.5
        reasons.append("Generado con el modelo de respaldo (Poisson), no con el modelo activo.")
    if discordance:
        score -= 10
        reasons.append("Los miembros del comité discrepan en el pronóstico.")
    if best_market_probability < 0.4:
        reasons.append("Mercado principal poco convincente (probabilidad cercana al azar).")

    score = int(round(_clamp(score)))
    label = RELIABILITY_NONE if not supported else _label_for_score(score)
    if not reasons:
        reasons.append("Soporte histórico sólido y pronóstico con convicción.")

    return {
        "score": score,
        "label": label,
        "is_model_backed": supported and not used_fallback,
        "used_fallback": used_fallback,
        "reasons": reasons,
    }


def summarize_reliability(match_briefs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates per-match markers into run-level counts. Pure function."""

    counts = {RELIABILITY_HIGH: 0, RELIABILITY_MEDIUM: 0, RELIABILITY_LOW: 0, RELIABILITY_NONE: 0}
    model_backed = 0
    for brief in match_briefs:
        marker = brief.get("reliability") or {}
        counts[marker.get("label", RELIABILITY_NONE)] = counts.get(marker.get("label", RELIABILITY_NONE), 0) + 1
        if marker.get("is_model_backed"):
            model_backed += 1
    return {
        "counts": counts,
        "model_backed_matches": model_backed,
        "total_matches": len(match_briefs),
    }


def evaluate_delivery_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Decides whether the run is trustworthy enough to deliver. Pure function.

    Never drops a match. Blocks the *whole* send only when the report would be
    generic: the run globally fell back to Poisson, or no match is model-backed.
    """

    matches = payload.get("matches") or []
    run_used_fallback = bool(payload.get("used_fallback"))

    if not matches:
        return {
            "should_deliver": False,
            "status": "no_schedule",
            "block_reason": "No hay partidos en la agenda para la fecha.",
            "model_backed_matches": 0,
            "total_matches": 0,
        }

    summary = summarize_reliability(matches)
    model_backed = summary["model_backed_matches"]

    missing_markers = [m for m in matches if not (m.get("reliability") or {}).get("label")]
    if missing_markers:
        return {
            "should_deliver": False,
            "status": "blocked",
            "block_reason": f"{len(missing_markers)} partido(s) sin marcador de fiabilidad.",
            "model_backed_matches": model_backed,
            "total_matches": summary["total_matches"],
        }

    if run_used_fallback:
        return {
            "should_deliver": False,
            "status": "blocked",
            "block_reason": (
                "El modelo activo no pudo cargarse; todo el informe usaría el respaldo Poisson "
                f"(motivo: {payload.get('fallback_reason') or 'desconocido'})."
            ),
            "model_backed_matches": model_backed,
            "total_matches": summary["total_matches"],
        }

    if model_backed == 0:
        return {
            "should_deliver": False,
            "status": "blocked",
            "block_reason": "Ningún partido está respaldado por el modelo activo (informe genérico).",
            "model_backed_matches": 0,
            "total_matches": summary["total_matches"],
        }

    return {
        "should_deliver": True,
        "status": "ready",
        "block_reason": None,
        "model_backed_matches": model_backed,
        "total_matches": summary["total_matches"],
    }
