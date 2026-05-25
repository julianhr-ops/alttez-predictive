import numpy as np
import pandas as pd
from scipy.stats import poisson

from alttez.features.poisson import PoissonEngine
from alttez.ingestion.international import InternationalLoader


def calcular_puntos(pred_h: int, pred_a: int, real_h: int, real_a: int, is_knockout: bool = False) -> int:
    """Aplica el reglamento del torneo."""

    mult = 2 if is_knockout else 1
    pts = 0

    if np.sign(pred_h - pred_a) == np.sign(real_h - real_a):
        pts += 5 * mult
    if pred_h == real_h:
        pts += 2 * mult
    if pred_a == real_a:
        pts += 2 * mult
    if (pred_h - pred_a) == (real_h - real_a):
        pts += 1 * mult

    return pts


def build_probability_matrix(lambda_home: float, lambda_away: float, max_goals: int = 7) -> np.ndarray:
    """Construye y normaliza la matriz de probabilidad de marcadores."""

    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_matrix[home_goals, away_goals] = (
                poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away)
            )

    matrix_sum = prob_matrix.sum()
    if matrix_sum <= 0:
        raise ValueError("La matriz de probabilidad no pudo normalizarse.")

    return prob_matrix / matrix_sum


def optimize_prediction(prob_matrix: np.ndarray, is_knockout: bool = False) -> pd.DataFrame:
    """Evalua todos los pronosticos y devuelve su valor esperado."""

    max_goals = prob_matrix.shape[0] - 1
    resultados = []

    for pred_h in range(max_goals + 1):
        for pred_a in range(max_goals + 1):
            expected_points = 0.0
            for real_h in range(max_goals + 1):
                for real_a in range(max_goals + 1):
                    probability = prob_matrix[real_h, real_a]
                    points = calcular_puntos(pred_h, pred_a, real_h, real_a, is_knockout=is_knockout)
                    expected_points += probability * points

            resultados.append(
                {
                    "Pronostico": f"{pred_h} - {pred_a}",
                    "EV": expected_points,
                }
            )

    return pd.DataFrame(resultados).sort_values(by="EV", ascending=False).reset_index(drop=True)


def main() -> None:
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
