"""Ajusta el parametro rho de Dixon-Coles por maxima verosimilitud (MLE).

En lugar del valor fijo historico (-0.13), este script estima el rho que mejor
explica los marcadores observados en el historico internacional, optimizando la
log-verosimilitud del modelo Dixon-Coles (producto Poisson independiente con la
correccion tau aplicada SOLO a las 4 celdas de marcador bajo: 0-0, 0-1, 1-0,
1-1, identica a ``_apply_dixon_coles`` en tournament_oracle).

Eleccion de las lambdas (per-match):
    No reutilizamos PoissonEngine.predict_xg porque ese motor incorpora ponderacion
    por recencia y factores de forma reciente que (a) introducen fuga temporal si se
    aplican sobre el mismo conjunto que se ajusta y (b) acoplan el ajuste de rho a la
    parametrizacion del motor. En su lugar usamos lambdas basadas en tasas observadas:
    para cada equipo se calcula su fuerza de ataque y defensa como el cociente de su
    promedio de goles (anotados / recibidos) contra el promedio global del dataset, y
    la lambda esperada de un partido es ataque_local * defensa_visitante * media_global
    (y simetrico para el visitante). Es pragmatico, O(1) por partido tras precomputar
    los diccionarios por equipo, y mantiene exactamente la misma estructura de pmf que
    usa la inferencia, que es lo unico que importa para estimar rho. El rho ajustado es
    robusto a la escala exacta de las lambdas porque la correccion tau depende del
    producto lambda_h*lambda_a y de las lambdas marginales en las celdas bajas.

Salida: artifacts/dixon_coles/active.json con
    {"rho", "fitted_on", "samples", "ll_before", "ll_after"}.

Uso:
    venv\\Scripts\\python.exe scripts\\fit_dixon_coles_rho.py [--as-of 2026-06-19]
        [--output artifacts/dixon_coles/active.json] [--min-matches 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttez.api.settings import load_local_env
from alttez.experiments.tournament_oracle import (
    DEFAULT_DIXON_COLES_RHO,
    DIXON_COLES_RHO_MAX,
    DIXON_COLES_RHO_MIN,
)
from alttez.ingestion.international import InternationalLoader

DEFAULT_OUTPUT_PATH = _REPO_ROOT / "artifacts" / "dixon_coles" / "active.json"
DEFAULT_MIN_MATCHES = 5
LAMBDA_FLOOR = 0.05


def _build_team_rates(df: pd.DataFrame, min_matches: int) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Precalcula fuerza de ataque/defensa por equipo (busqueda O(1) en el fit)."""

    scored: Dict[str, float] = {}
    conceded: Dict[str, float] = {}
    played: Dict[str, int] = {}

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        hs, as_ = float(row.home_score), float(row.away_score)
        scored[home] = scored.get(home, 0.0) + hs
        conceded[home] = conceded.get(home, 0.0) + as_
        played[home] = played.get(home, 0) + 1
        scored[away] = scored.get(away, 0.0) + as_
        conceded[away] = conceded.get(away, 0.0) + hs
        played[away] = played.get(away, 0) + 1

    global_avg = float((df["home_score"].sum() + df["away_score"].sum()) / (2 * len(df)))
    if global_avg <= 0:
        raise ValueError("El promedio global de goles es no positivo; dataset invalido.")

    rates: Dict[str, Dict[str, float]] = {}
    for team, n in played.items():
        if n < min_matches:
            continue
        attack = (scored[team] / n) / global_avg
        defense = (conceded[team] / n) / global_avg
        rates[team] = {"attack": attack, "defense": defense}

    return rates, global_avg


def _match_lambdas(
    home: str,
    away: str,
    rates: Dict[str, Dict[str, float]],
    global_avg: float,
) -> Tuple[float, float]:
    """Lambdas esperadas (neutras) de un partido a partir de las tasas precomputadas."""

    home_rate = rates.get(home, {"attack": 1.0, "defense": 1.0})
    away_rate = rates.get(away, {"attack": 1.0, "defense": 1.0})
    lambda_h = home_rate["attack"] * away_rate["defense"] * global_avg
    lambda_a = away_rate["attack"] * home_rate["defense"] * global_avg
    return max(lambda_h, LAMBDA_FLOOR), max(lambda_a, LAMBDA_FLOOR)


def _tau(home_goals: int, away_goals: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Factor de correccion Dixon-Coles para las 4 celdas de marcador bajo.

    Misma estructura que ``_apply_dixon_coles``: solo (0,0),(0,1),(1,0),(1,1) se
    corrigen; el resto vale 1.0.
    """

    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_h * lambda_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_h * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_a * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _build_samples(df: pd.DataFrame, rates: Dict[str, Dict[str, float]], global_avg: float) -> np.ndarray:
    """Devuelve un array Nx4 [home_goals, away_goals, lambda_h, lambda_a]."""

    rows = []
    for row in df.itertuples(index=False):
        hg = int(round(float(row.home_score)))
        ag = int(round(float(row.away_score)))
        if hg < 0 or ag < 0:
            continue
        lambda_h, lambda_a = _match_lambdas(row.home_team, row.away_team, rates, global_avg)
        rows.append((hg, ag, lambda_h, lambda_a))
    return np.asarray(rows, dtype=float)


def dixon_coles_log_likelihood(rho: float, samples: np.ndarray) -> float:
    """Log-verosimilitud total Dixon-Coles de los marcadores observados dado rho.

    Cada partido aporta log( P_pois(hg|lh) * P_pois(ag|la) * tau(hg,ag,lh,la,rho) ).
    Solo el factor tau depende de rho; el termino Poisson es constante respecto a rho
    pero se incluye para una verosimilitud bien definida.
    """

    total = 0.0
    for hg, ag, lambda_h, lambda_a in samples:
        hg_i = int(hg)
        ag_i = int(ag)
        base = poisson.pmf(hg_i, lambda_h) * poisson.pmf(ag_i, lambda_a)
        tau = _tau(hg_i, ag_i, lambda_h, lambda_a, rho)
        joint = base * tau
        if joint <= 0:
            return -np.inf
        total += float(np.log(joint))
    return total


def fit_rho(samples: np.ndarray) -> float:
    """Optimiza rho en [-0.3, 0.0] maximizando la log-verosimilitud."""

    result = minimize_scalar(
        lambda r: -dixon_coles_log_likelihood(r, samples),
        bounds=(DIXON_COLES_RHO_MIN, DIXON_COLES_RHO_MAX),
        method="bounded",
    )
    return float(result.x)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ajusta el rho de Dixon-Coles por MLE.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Ruta del artefacto JSON de salida (default artifacts/dixon_coles/active.json).",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Fecha (YYYY-MM-DD) a registrar en 'fitted_on'. Si se omite queda null.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=DEFAULT_MIN_MATCHES,
        help="Minimo de partidos para incluir las tasas de un equipo (default 5).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_local_env()
    args = _parse_args(argv)

    print("=== AJUSTE MLE DE RHO (DIXON-COLES) ===")
    print("Cargando historico internacional...")
    df = InternationalLoader().fetch_data()
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
    if df.empty:
        print("ERROR: no hay partidos con marcadores validos.")
        return 1
    print(f"Partidos cargados: {len(df)}")

    rates, global_avg = _build_team_rates(df, min_matches=args.min_matches)
    print(f"Equipos con tasas (>= {args.min_matches} partidos): {len(rates)}")
    print(f"Promedio global de goles por equipo/partido: {global_avg:.4f}")

    samples = _build_samples(df, rates, global_avg)
    n_samples = int(samples.shape[0])
    if n_samples == 0:
        print("ERROR: no se construyeron muestras para el ajuste.")
        return 1
    print(f"Muestras para la verosimilitud: {n_samples}")

    ll_before = dixon_coles_log_likelihood(DEFAULT_DIXON_COLES_RHO, samples)
    fitted_rho = fit_rho(samples)
    ll_after = dixon_coles_log_likelihood(fitted_rho, samples)

    print(f"rho fijo historico = {DEFAULT_DIXON_COLES_RHO:.6f} -> LL = {ll_before:.4f}")
    print(f"rho ajustado (MLE) = {fitted_rho:.6f} -> LL = {ll_after:.4f}")
    delta = ll_after - ll_before
    print(f"Mejora de LL respecto al rho fijo: {delta:+.4f}")

    payload = {
        "rho": fitted_rho,
        "fitted_on": args.as_of,
        "samples": n_samples,
        "ll_before": float(ll_before),
        "ll_after": float(ll_after),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Artefacto escrito en: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
