"""Resolucion de pesos del comite de prediccion (Fase 2).

El comite combina campeon + retador + baseline Poisson promediando sus matrices
de marcadores. Pero un promedio NO garantiza mejorar a sus miembros: en los datos
historicos el consenso (Brier 0.696) rinde peor que el propio campeon (0.608). Sin
una guardia, el comite degrada al mejor modelo disponible.

Dos mecanismos, en orden:

  1. Guardia de consenso (dura): si el consenso historico es peor que el campeon
     por encima de un margen, el comite DEFIERE al campeon (peso 1 al campeon,
     0 al resto). Espeja la regla de gobernanza: Brier decide. Garantiza que el
     comite nunca entregue algo peor que su mejor miembro conocido.

  2. Pesos por error inverso (suave): cuando no se defiere, cada miembro se pondera
     por 1/Brier reciente (con piso de campeon), de modo que un retador debil pierde
     peso solo en vez de arrastrar el consenso.

Si no hay metricas suficientes, cae a los pesos por defecto (comportamiento previo).
Funciones puras y testeables segun la politica anti-monolitica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Umbral bajo a proposito: en torneos el modelo activo acumula pocas
# evaluaciones cerradas (ej. ~20). Exigir mas lo dejaria siempre en pesos por
# defecto. 15 es el minimo para que la guardia/ponderacion pueda actuar con la
# senal disponible, asumiendo el ruido de muestra pequena.
DEFAULT_MIN_SAMPLES = 15
DEFAULT_CHAMPION_FLOOR = 0.5
DEFAULT_REGRESSION_MARGIN = 0.03
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
    min_samples: int = DEFAULT_MIN_SAMPLES,
    champion_floor: float = DEFAULT_CHAMPION_FLOOR,
    regression_margin: float = DEFAULT_REGRESSION_MARGIN,
) -> WeightingDecision:
    """Decide los pesos por rol del comite a partir de las metricas recientes.

    member_versions: rol -> model_version de cada miembro presente.
    audit_models: salida de load_prediction_learning_audit()["models"].
    default_weights: pesos fijos previos (fallback).
    """
    roles = list(member_versions.keys())
    brier_by_version = _brier_lookup(audit_models, min_samples=min_samples)
    member_briers = {
        role: brier_by_version[version]
        for role, version in member_versions.items()
        if version in brier_by_version
    }

    champion_version = member_versions.get(CHAMPION_ROLE)
    champion_brier = brier_by_version.get(champion_version) if champion_version else None
    consensus_brier = brier_by_version.get(committee_version)

    # 1) Guardia de consenso: si el consenso historico es peor que el campeon,
    # defiere al campeon. Solo si ambos tienen metricas comparables.
    if (
        CHAMPION_ROLE in roles
        and champion_brier is not None
        and consensus_brier is not None
        and consensus_brier > champion_brier + regression_margin
    ):
        weights = {role: (1.0 if role == CHAMPION_ROLE else 0.0) for role in roles}
        return WeightingDecision(
            weights=weights,
            deferred_to_champion=True,
            strategy="champion_guard",
            reason=(
                f"Consenso historico (Brier {consensus_brier:.4f}) peor que el campeon "
                f"({champion_brier:.4f}); se defiere al campeon."
            ),
            member_briers=member_briers,
        )

    # 2) Pesos por error inverso entre miembros con metricas; resto usa default.
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
            reason="Pesos por error inverso (1/Brier) con piso de campeon.",
            member_briers=member_briers,
        )

    # 3) Sin metricas suficientes: pesos por defecto.
    return WeightingDecision(
        weights=dict(default_weights),
        strategy="default",
        reason="Metricas insuficientes; pesos por defecto.",
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
