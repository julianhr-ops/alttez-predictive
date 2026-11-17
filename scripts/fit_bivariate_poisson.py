"""Ajusta offline los parametros del Bivariate Poisson con inflacion diagonal.

Estima (lambda3, diagonal_weight) que maximizan la log-verosimilitud de los
marcadores observados bajo el DI-BP, usando las lambdas marginales del
PoissonEngine como senal de entrada (mismas que consume la inferencia). Escribe
artifacts/bivariate/active.json. Si no existe el artefacto, el engine es neutro
(Poisson independiente), por lo que este ajuste es opcional pero recomendado.

Uso:
    venv\\Scripts\\python.exe scripts/fit_bivariate_poisson.py --output artifacts/bivariate/active.json

Logs en texto plano (sin unicode) por compatibilidad con la terminal de Windows.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from alttez.api.settings import load_local_env
from alttez.features.bivariate_poisson import BivariatePoissonEngine
from alttez.features.klement import load_enriched_historical_matches
from alttez.features.poisson import PoissonEngine

LOGGER = logging.getLogger("fit_bivariate_poisson")
DEFAULT_OUTPUT = Path("artifacts/bivariate/active.json")
DEFAULT_MAX_GOALS = 7


def _log_likelihood(
    engine: BivariatePoissonEngine,
    lambdas: list[tuple[float, float]],
    scores: list[tuple[int, int]],
    max_goals: int,
) -> float:
    """Suma de log P(marcador) bajo el DI-BP. Marcadores fuera de rango se ignoran."""
    total = 0.0
    for (lambda_h, lambda_a), (home, away) in zip(lambdas, scores, strict=False):
        if home > max_goals or away > max_goals:
            continue
        matrix = engine.build_matrix(lambda_h, lambda_a, max_goals=max_goals)
        prob = float(matrix[home, away])
        total += float(np.log(max(prob, 1e-12)))
    return total


def fit_parameters(
    lambdas: list[tuple[float, float]],
    scores: list[tuple[int, int]],
    *,
    max_goals: int = DEFAULT_MAX_GOALS,
    lambda3_grid: np.ndarray | None = None,
    diagonal_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Grid search sobre (lambda3, diagonal_weight) maximizando log-verosimilitud."""
    lambda3_grid = lambda3_grid if lambda3_grid is not None else np.linspace(0.0, 0.5, 11)
    diagonal_grid = diagonal_grid if diagonal_grid is not None else np.linspace(0.0, 0.4, 9)

    baseline_ll = _log_likelihood(BivariatePoissonEngine(0.0, 0.0), lambdas, scores, max_goals)
    best = {"lambda3": 0.0, "diagonal_weight": 0.0, "ll": baseline_ll}
    for lambda3 in lambda3_grid:
        for diag in diagonal_grid:
            engine = BivariatePoissonEngine(float(lambda3), float(diag))
            ll = _log_likelihood(engine, lambdas, scores, max_goals)
            if ll > best["ll"]:
                best = {"lambda3": float(lambda3), "diagonal_weight": float(diag), "ll": ll}
    best["ll_baseline"] = baseline_ll
    return best


def _build_inputs(target_date: str) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    df = load_enriched_historical_matches()
    engine = PoissonEngine()
    engine.fit(df, target_date=target_date)
    lambdas: list[tuple[float, float]] = []
    scores: list[tuple[int, int]] = []
    for row in df.itertuples(index=False):
        is_neutral = bool(getattr(row, "neutral", False))
        lambda_h, lambda_a = engine.predict_xg(row.home_team, row.away_team, is_neutral=is_neutral)
        lambdas.append((float(lambda_h), float(lambda_a)))
        scores.append((int(row.home_score), int(row.away_score)))
    return lambdas, scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ajuste offline del Bivariate Poisson DI-BP.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-date", default="2026-06-01", help="Corte para fit del PoissonEngine.")
    parser.add_argument("--max-goals", type=int, default=DEFAULT_MAX_GOALS)
    parser.add_argument("--samples", type=int, default=0, help="Limita el numero de partidos (0 = todos).")
    args = parser.parse_args()

    load_local_env()
    lambdas, scores = _build_inputs(args.target_date)
    if args.samples and args.samples > 0:
        lambdas, scores = lambdas[-args.samples:], scores[-args.samples:]

    LOGGER.info("Ajustando DI-BP sobre %d partidos...", len(scores))
    best = fit_parameters(lambdas, scores, max_goals=args.max_goals)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lambda3": best["lambda3"],
        "diagonal_weight": best["diagonal_weight"],
        "samples": len(scores),
        "ll_baseline": best["ll_baseline"],
        "ll_fitted": best["ll"],
        "target_date": args.target_date,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info(
        "lambda3=%.4f diagonal_weight=%.4f ll_baseline=%.2f ll_fitted=%.2f -> %s",
        best["lambda3"],
        best["diagonal_weight"],
        best["ll_baseline"],
        best["ll"],
        output_path,
    )


if __name__ == "__main__":
    main()
