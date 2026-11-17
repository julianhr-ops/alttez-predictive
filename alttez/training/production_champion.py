"""Production configuration of the champion that the challenger cycle must reproduce.

The router serves the active Poisson champion with ``DEFAULT_POISSON_KWARGS`` and the
Dixon-Coles rho fitted by ``scripts/fit_dixon_coles_rho.py``. Governance compares the
challenger against that exact configuration; when it cannot be reproduced the cycle
has no valid champion and must not promote anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from alttez.experiments.tournament_oracle import load_fitted_dixon_coles_rho
from alttez.inference.router import (
    DEFAULT_POISSON_KWARGS,
    DEFAULT_POISSON_MODEL_FAMILY,
    DEFAULT_POISSON_MODEL_VERSION,
)


class ProductionConfigUnavailable(RuntimeError):
    """The production champion cannot be reproduced, so no promotion may be decided."""


@dataclass(frozen=True, slots=True)
class ProductionChampionConfig:
    version_key: str | None
    model_version: str
    model_family: str
    poisson_kwargs: Mapping[str, Any]
    rho: float


def resolve_production_champion_config(active_record: Mapping[str, Any] | None) -> ProductionChampionConfig:
    """Rebuild the champion the router serves from the active model record.

    ``active_record=None`` is an empty registry: the router then serves the default
    Poisson model, which is the champion to compare against.
    """

    family = str((active_record or {}).get("model_family") or DEFAULT_POISSON_MODEL_FAMILY).strip().lower()
    if family != DEFAULT_POISSON_MODEL_FAMILY:
        raise ProductionConfigUnavailable(
            f"El modelo activo es de familia '{family}'; el champion Poisson no reproduce la produccion."
        )

    rho = load_fitted_dixon_coles_rho()
    if rho is None:
        raise ProductionConfigUnavailable(
            "No se pudo cargar el rho Dixon-Coles ajustado (artifacts/dixon_coles/active.json); "
            "el champion evaluado no seria el de produccion."
        )

    record = active_record or {}
    return ProductionChampionConfig(
        version_key=record.get("version_key"),
        model_version=str(record.get("model_version") or DEFAULT_POISSON_MODEL_VERSION),
        model_family=family,
        poisson_kwargs=dict(DEFAULT_POISSON_KWARGS),
        rho=rho,
    )
