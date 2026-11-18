"""Fuerzas jerarquicas bayesianas (Baio-Blangiardo) para seleccion nacional.

Modelo generativo:

    log(lambda_home) = mu + att[home] - def[away] + gamma * (1 - is_neutral)
    log(lambda_away) = mu + att[away] - def[home]

    att_i ~ Normal(0, sigma_att)   con restriccion suma-cero (identificabilidad)
    def_i ~ Normal(0, sigma_def)   con restriccion suma-cero
    sigma_att, sigma_def ~ HalfNormal(1)
    gamma ~ Normal(0.3, 0.1)       ventaja de localia (anulada en sede neutral)
    goals ~ Poisson(lambda)

El POOLING PARCIAL encoge las fuerzas de equipos con pocos partidos hacia la
media global: clave en un torneo corto (cold-start severo para Cabo Verde,
Curazao, Haiti, etc.).

SEPARACION DE DEPENDENCIAS (decision de arquitectura):
- El AJUSTE (MCMC/NUTS) vive en ``fit_hierarchical_model`` con import PEREZOSO de
  PyMC. Solo se necesita PyMC al correr el script offline de ajuste.
- La INFERENCIA carga un artefacto COMPACTO (medias posteriores por equipo) y usa
  unicamente numpy/scipy. Asi PyMC NO entra al path de produccion ni al daily job.

El artefacto activo es artifacts/hierarchical/strengths.json. Si falta, el engine
es NEUTRO (toda fuerza 0, lambda = exp(mu) o un baseline) y la prediccion se marca
como no soportada para no contaminar al comite con ruido.

Busquedas de fuerza O(1) por diccionario mapeado por equipo (proteje el runtime).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from alttez.ingestion.live_matches import canonical_team_name

LOGGER = logging.getLogger(__name__)

DEFAULT_HIERARCHICAL_PATH = Path("artifacts/hierarchical/strengths.json")
HIERARCHICAL_PATH_ENV = "ALTTEZ_HIERARCHICAL_PATH"
DEFAULT_HOME_ADVANTAGE = 0.3


@dataclass(slots=True, frozen=True)
class TeamStrength:
    attack: float = 0.0
    defense: float = 0.0


@dataclass(slots=True)
class HierarchicalStrengths:
    """Medias posteriores compactas para inferencia (sin dependencia de PyMC)."""

    mu: float = 0.0
    gamma: float = DEFAULT_HOME_ADVANTAGE
    sigma_att: float = 0.0
    sigma_def: float = 0.0
    teams: dict[str, TeamStrength] = field(default_factory=dict)
    samples: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.teams

    def strength_for(self, team: str) -> TeamStrength:
        """Fuerza del equipo (O(1)); cold-start -> fuerza neutra (pooling a la media)."""
        return self.teams.get(canonical_team_name(team), TeamStrength())


class HierarchicalBayesEngine:
    """Construye lambdas y matriz de marcadores desde fuerzas jerarquicas."""

    def __init__(self, strengths: HierarchicalStrengths | None = None) -> None:
        self.strengths = strengths or HierarchicalStrengths()

    @property
    def is_ready(self) -> bool:
        return not self.strengths.is_empty

    def expected_goals(self, home_team: str, away_team: str, *, is_neutral: bool = False) -> tuple[float, float]:
        """lambdas esperadas (medias posteriores) para el enfrentamiento."""
        s = self.strengths
        home = s.strength_for(home_team)
        away = s.strength_for(away_team)
        home_adv = s.gamma * (0.0 if is_neutral else 1.0)
        log_lambda_home = s.mu + home.attack - away.defense + home_adv
        log_lambda_away = s.mu + away.attack - home.defense
        lambda_home = max(math.exp(log_lambda_home), 0.05)
        lambda_away = max(math.exp(log_lambda_away), 0.05)
        return float(lambda_home), float(lambda_away)

    def build_matrix(self, home_team: str, away_team: str, *, is_neutral: bool = False, max_goals: int = 7) -> np.ndarray:
        """Matriz de marcadores (producto Poisson de las lambdas posteriores)."""
        lambda_home, lambda_away = self.expected_goals(home_team, away_team, is_neutral=is_neutral)
        size = max_goals + 1
        home_pmf = poisson.pmf(np.arange(size), lambda_home)
        away_pmf = poisson.pmf(np.arange(size), lambda_away)
        matrix = np.outer(home_pmf, away_pmf)
        total = matrix.sum()
        if total <= 0:
            raise ValueError("La matriz jerarquica no pudo normalizarse.")
        return matrix / total


def load_active_hierarchical_strengths(path: str | Path | None = None) -> HierarchicalStrengths:
    """Carga las fuerzas posteriores del artefacto activo.

    Devuelve fuerzas VACIAS (engine no listo) si el artefacto falta o es ilegible:
    la inferencia nunca debe romperse por su ausencia.
    """
    resolved = Path(path or os.getenv(HIERARCHICAL_PATH_ENV) or DEFAULT_HIERARCHICAL_PATH)
    if not resolved.exists():
        return HierarchicalStrengths()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        teams = {
            canonical_team_name(name): TeamStrength(
                attack=float(values.get("attack", 0.0)),
                defense=float(values.get("defense", 0.0)),
            )
            for name, values in payload.get("teams", {}).items()
        }
        return HierarchicalStrengths(
            mu=float(payload.get("mu", 0.0)),
            gamma=float(payload.get("gamma", DEFAULT_HOME_ADVANTAGE)),
            sigma_att=float(payload.get("sigma_att", 0.0)),
            sigma_def=float(payload.get("sigma_def", 0.0)),
            teams=teams,
            samples=int(payload.get("samples", 0)),
        )
    except Exception as exc:
        LOGGER.warning("hierarchical_artifact_unreadable path=%s detail=%s", resolved, exc)
        return HierarchicalStrengths()


def fit_hierarchical_model(
    matches: "pd.DataFrame",  # noqa: F821
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    seed: int = 0,
) -> dict:
    """Ajusta el modelo jerarquico via NUTS y devuelve las medias posteriores.

    Import PEREZOSO de PyMC: solo se requiere al ejecutar el ajuste offline. El
    DataFrame debe traer columnas home_team, away_team, home_score, away_score,
    neutral. Aplica restriccion suma-cero sobre att/def (ZeroSumNormal) para
    identificabilidad frente a mu.

    Devuelve un dict serializable con mu, gamma, sigma_att, sigma_def, teams
    (att/def por equipo) y samples.
    """
    import numpy as _np
    import pymc as pm

    home = matches["home_team"].map(canonical_team_name).to_numpy()
    away = matches["away_team"].map(canonical_team_name).to_numpy()
    home_goals = matches["home_score"].to_numpy().astype(int)
    away_goals = matches["away_score"].to_numpy().astype(int)
    neutral = matches["neutral"].astype(bool).to_numpy()

    teams = sorted(set(home) | set(away))
    team_index = {team: idx for idx, team in enumerate(teams)}
    home_idx = _np.array([team_index[t] for t in home])
    away_idx = _np.array([team_index[t] for t in away])
    not_neutral = (~neutral).astype(float)
    n_teams = len(teams)

    with pm.Model():
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        gamma = pm.Normal("gamma", mu=0.3, sigma=0.1)
        sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
        sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)
        att = pm.ZeroSumNormal("att", sigma=sigma_att, shape=n_teams)
        deff = pm.ZeroSumNormal("def", sigma=sigma_def, shape=n_teams)

        log_lambda_home = mu + att[home_idx] - deff[away_idx] + gamma * not_neutral
        log_lambda_away = mu + att[away_idx] - deff[home_idx]

        pm.Poisson("home_goals", mu=pm.math.exp(log_lambda_home), observed=home_goals)
        pm.Poisson("away_goals", mu=pm.math.exp(log_lambda_away), observed=away_goals)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
        )

    posterior = idata.posterior
    att_mean = posterior["att"].mean(dim=["chain", "draw"]).to_numpy()
    def_mean = posterior["def"].mean(dim=["chain", "draw"]).to_numpy()
    result = {
        "mu": float(posterior["mu"].mean().item()),
        "gamma": float(posterior["gamma"].mean().item()),
        "sigma_att": float(posterior["sigma_att"].mean().item()),
        "sigma_def": float(posterior["sigma_def"].mean().item()),
        "samples": int(len(matches)),
        "teams": {
            team: {"attack": float(att_mean[idx]), "defense": float(def_mean[idx])}
            for team, idx in team_index.items()
        },
    }
    return result, idata
