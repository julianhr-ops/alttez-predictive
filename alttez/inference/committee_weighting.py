"""Resolucion de pesos del comite de prediccion (Fase 2).

El comite combina campeon + retador + baseline Poisson promediando sus matrices
de marcadores. Pero un promedio NO garantiza mejorar a sus miembros. Sin una
guardia, el comite puede degradar al mejor modelo disponible.

Dos mecanismos, en orden:

  1. Guardia de consenso (dura): si el consenso es peor que el campeon medido sobre
     LOS MISMOS partidos (diferencia Brier por partido, bootstrap pareado con IC
     completamente sobre 0), el comite DEFIERE al campeon (peso 1 al campeon, 0 al
     resto). Comparar Briers agregados de versiones evaluadas sobre partidos
     distintos no es una comparacion valida y ya no se usa.

  2. Pesos por error inverso (suave): cuando no se defiere, cada miembro se pondera
     por 1/Brier reciente (con piso de campeon), de modo que un retador debil pierde
     peso solo en vez de arrastrar el consenso.

Si no hay metricas suficientes, cae a los pesos por defecto (comportamiento previo).
Funciones puras y testeables segun la politica anti-monolitica.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from alttez.evaluation.paired import paired_bootstrap_diff

DEFAULT_MIN_SAMPLES = 15
DEFAULT_CHAMPION_FLOOR = 0.5
CHAMPION_ROLE = "champion"


@dataclass(slots=True)
class WeightingDecision:
    weights: dict[str, float]
    deferred_to_champion: bool = False
    strategy: str = "default"
    reason: str | None = None
    member_briers: dict[str, float] = field(default_factory=dict)


def _brier_lookup(audit_models: list[dict[str, Any]], *, min_samples: int) -> dict[str, float]:
    """model_version -> Brier, solo cuando hay muestras suficientes y es finito."""
    lookup: dict[str, float] = {}
    for row in audit_models or []:
        version = str(row.get("model_version") or "")
        samples = int(row.get("evaluated_predictions") or 0)
        brier = row.get("brier_score")
        if not version or samples < min_samples or brier is None:
            continue
        brier_value = float(brier)
        if brier_value > 0:
            lookup[version] = brier_value
    return lookup


def resolve_committee_weights(
    member_versions: dict[str, str],
    audit_models: list[dict[str, Any]],
    default_weights: dict[str, float],
    *,
    committee_version: str,
    paired_losses: tuple[Sequence[float], Sequence[float]] | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    champion_floor: float = DEFAULT_CHAMPION_FLOOR,
) -> WeightingDecision:
    """Decide los pesos por rol del comite a partir de las metricas recientes.

    member_versions: rol -> model_version de cada miembro presente.
    audit_models: salida de load_prediction_learning_audit()["models"].
    default_weights: pesos fijos previos (fallback).
    paired_losses: ``(perdidas_consenso, perdidas_campeon)`` por partido sobre LOS
        MISMOS partidos. Sin ellas la guardia de consenso no se evalua.
    committee_version: version bajo la que se registra el consenso (solo informativa).
    """
    roles = list(member_versions.keys())
    brier_by_version = _brier_lookup(audit_models, min_samples=min_samples)
    member_briers = {
        role: brier_by_version[version]
        for role, version in member_versions.items()
        if version in brier_by_version
    }

    guard_note = " Guardia de consenso no evaluada: faltan perdidas pareadas del consenso y el campeon."
    if CHAMPION_ROLE in roles and paired_losses is not None:
        consensus_losses, champion_losses = paired_losses
        paired = paired_bootstrap_diff(consensus_losses, champion_losses)
        guard_note = ""
        if paired.n >= min_samples and paired.b_better:
            weights = {role: (1.0 if role == CHAMPION_ROLE else 0.0) for role in roles}
            return WeightingDecision(
                weights=weights,
                deferred_to_champion=True,
                strategy="champion_guard",
                reason=(
                    f"Consenso peor que el campeon en {paired.n} partidos comunes "
                    f"(diferencia Brier {paired.mean_diff:+.4f}, IC [{paired.ci_low:+.4f}, {paired.ci_high:+.4f}]); "
                    "se defiere al campeon."
                ),
                member_briers=member_briers,
            )

    if member_briers:
        scores = {
            role: (1.0 / member_briers[role]) if role in member_briers else max(default_weights.get(role, 0.0), 1e-6)
            for role in roles
        }
        total = sum(scores.values())
        weights = {role: score / total for role, score in scores.items()} if total > 0 else dict(default_weights)
        weights = _enforce_champion_floor(weights, champion_floor=champion_floor)
        return WeightingDecision(
            weights=weights,
            strategy="inverse_brier",
            reason="Pesos por error inverso (1/Brier) con piso de campeon." + guard_note,
            member_briers=member_briers,
        )

    return WeightingDecision(
        weights=dict(default_weights),
        strategy="default",
        reason="Metricas insuficientes; pesos por defecto." + guard_note,
        member_briers=member_briers,
    )


def _enforce_champion_floor(weights: dict[str, float], *, champion_floor: float) -> dict[str, float]:
    """Garantiza que el campeon conserve al menos champion_floor del peso total."""
    if CHAMPION_ROLE not in weights or champion_floor <= 0:
        return weights
    total = sum(weights.values())
    normalized = {role: (weight / total if total > 0 else 0.0) for role, weight in weights.items()}
    if normalized[CHAMPION_ROLE] >= champion_floor:
        return normalized
    others = {role: weight for role, weight in normalized.items() if role != CHAMPION_ROLE}
    others_sum = sum(others.values())
    remaining = 1.0 - champion_floor
    adjusted = {CHAMPION_ROLE: champion_floor}
    for role, weight in others.items():
        adjusted[role] = (weight / others_sum * remaining) if others_sum > 0 else 0.0
    return adjusted
