import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

from alttez.evaluation.regulation_points import rank_regulation_predictions, regulation_points
from alttez.features.poisson import PoissonEngine


DEFAULT_DIXON_COLES_RHO = -0.13

DIXON_COLES_RHO_MIN = -0.3
DIXON_COLES_RHO_MAX = 0.0

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIXON_COLES_RHO_PATH = _REPO_ROOT / "artifacts" / "dixon_coles" / "active.json"


def load_fitted_dixon_coles_rho(path: Optional[str] = None) -> Optional[float]:
    """Lee el rho ajustado por MLE de ``artifacts/dixon_coles/active.json``.

    Override via la variable de entorno ``ALTTEZ_DIXON_COLES_RHO_PATH`` o el argumento
    ``path``. Devuelve ``None`` si el archivo falta, es ilegible, no contiene un rho
    numerico o cae fuera del rango ``[-0.3, 0.0]``.
    """

    resolved = path or os.environ.get("ALTTEZ_DIXON_COLES_RHO_PATH") or DEFAULT_DIXON_COLES_RHO_PATH
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rho = float(payload["rho"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    if not np.isfinite(rho) or rho < DIXON_COLES_RHO_MIN or rho > DIXON_COLES_RHO_MAX:
        return None

    return rho


def load_active_dixon_coles_rho(path: Optional[str] = None) -> float:
    """Resuelve el rho de Dixon-Coles activo (ajustado por MLE).

    Nunca lanza excepciones: si el artefacto no es utilizable devuelve
    ``DEFAULT_DIXON_COLES_RHO``. La inferencia debe arrancar sin el artefacto.
    """

    fitted = load_fitted_dixon_coles_rho(path)
    return DEFAULT_DIXON_COLES_RHO if fitted is None else fitted


def calcular_puntos(pred_h: int, pred_a: int, real_h: int, real_a: int, is_knockout: bool = False) -> int:
    """Aplica el reglamento del torneo."""

    return regulation_points(pred_h, pred_a, real_h, real_a, is_knockout=is_knockout)


def _apply_dixon_coles(prob_matrix: np.ndarray, lambda_home: float, lambda_away: float, rho: float) -> None:
    """Aplica in-place la correccion Dixon-Coles a las 4 celdas de marcador bajo."""

    prob_matrix[0, 0] *= 1.0 - lambda_home * lambda_away * rho
    prob_matrix[0, 1] *= 1.0 + lambda_home * rho
    prob_matrix[1, 0] *= 1.0 + lambda_away * rho
    prob_matrix[1, 1] *= 1.0 - rho
    np.clip(prob_matrix, 0.0, None, out=prob_matrix)


def build_probability_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 7,
    rho: Optional[float] = None,
) -> np.ndarray:
    """Construye y normaliza la matriz de probabilidad de marcadores.

    ``rho`` controla la correccion Dixon-Coles sobre las celdas de marcador bajo
    (0-0, 0-1, 1-0, 1-1), que corrige la subestimacion de empates propia del
    producto Poisson independiente:

    - ``rho=None`` (por defecto): resuelve el valor ajustado activo via
      ``load_active_dixon_coles_rho()`` (cae a ``DEFAULT_DIXON_COLES_RHO`` si no
      hay artefacto). Esto permite que el sistema use el rho ajustado por MLE.
    - ``rho`` float explicito (incluye ``0.0`` y ``-0.13``): se usa tal cual. Asi
      cualquier llamador existente que pase un rho explicito mantiene su
      comportamiento sin cambios.

    Con ``rho == 0.0`` no se aplica correccion alguna.
    """

    if rho is None:
        rho = load_active_dixon_coles_rho()

    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_matrix[home_goals, away_goals] = (
                poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away)
            )

    if rho != 0.0:
        _apply_dixon_coles(prob_matrix, lambda_home, lambda_away, rho)

    matrix_sum = prob_matrix.sum()
    if matrix_sum <= 0:
        raise ValueError("La matriz de probabilidad no pudo normalizarse.")

    return prob_matrix / matrix_sum


def optimize_prediction(prob_matrix: np.ndarray, is_knockout: bool = False) -> pd.DataFrame:
    """Evalua todos los pronosticos y devuelve su valor esperado."""

    return rank_regulation_predictions(prob_matrix, is_knockout=is_knockout)


def validate_points_optimal_recommendations(
    prob_matrix: np.ndarray,
    recommendations: pd.DataFrame,
    is_knockout: bool = False,
    tolerance: float = 1e-9,
) -> None:
    """Valida que las recomendaciones maximicen puntos esperados del reglamento."""

    if recommendations.empty:
        raise ValueError("No hay recomendaciones para validar.")

    if "Pronostico" not in recommendations.columns or "EV" not in recommendations.columns:
        raise ValueError("Las recomendaciones deben incluir columnas Pronostico y EV.")

    full_ranking = optimize_prediction(prob_matrix, is_knockout=is_knockout)
    expected_best = full_ranking.iloc[0]
    actual_best = recommendations.iloc[0]
    actual_ev = float(actual_best["EV"])
    expected_ev = float(expected_best["EV"])

    if str(actual_best["Pronostico"]) != str(expected_best["Pronostico"]) or abs(actual_ev - expected_ev) > tolerance:
        raise ValueError(
            "La recomendacion principal no maximiza puntos esperados: "
            f"actual={actual_best['Pronostico']} EV={actual_ev:.6f}; "
            f"esperada={expected_best['Pronostico']} EV={expected_ev:.6f}."
        )


def main() -> None:
    from alttez.ingestion.international import InternationalLoader

    print("=== ORACULO ALTTEZ: PREDICTOR DE TORNEOS ===\n")

    target_date = "2026-05-24"
    years_window = 4
    equipo_a = "Colombia"
    equipo_b = "Brazil"
    is_neutral = True
    is_knockout = False

    loader = InternationalLoader()
    df = loader.fetch_data()

    print(f"Entrenando Motor de Poisson (analizando ultimos {years_window} anos)...")
    engine = PoissonEngine(years_window=years_window, prior_matches=8)
    engine.fit(df, target_date=target_date)

    lambda_a, lambda_b = engine.predict_xg(equipo_a, equipo_b, is_neutral=is_neutral)
    print(f"\n{equipo_a} (ataque) vs {equipo_b} (defensa) -> xG esperado: {lambda_a}")
    print(f"{equipo_b} (ataque) vs {equipo_a} (defensa) -> xG esperado: {lambda_b}\n")

    prob_matrix = build_probability_matrix(lambda_a, lambda_b, max_goals=7)
    resultados = optimize_prediction(prob_matrix, is_knockout=is_knockout)

    print("PRONOSTICO OPTIMO MATEMATICAMENTE:")
    print(resultados.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
