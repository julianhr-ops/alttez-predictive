"""Metricas de evaluacion para predicciones 1X2.

El Brier multiclase trata las tres clases (H, D, A) como nominales y penaliza
igual un error H->A que un error H->D. El 1X2 es en realidad *ordinal*
(victoria local -> empate -> victoria visitante): equivocarse por una posicion
debe doler menos que equivocarse por dos. El Ranked Probability Score (RPS)
respeta esa ordinalidad comparando las distribuciones acumuladas.

Funciones puras y unitariamente testeables segun la politica anti-monolitica.
"""

from __future__ import annotations

# Orden ordinal canonico del 1X2: local -> empate -> visitante.
_ORDINAL_OUTCOMES = ("H", "D", "A")


def _outcome_cumulative(actual_outcome: str) -> tuple[float, float]:
    """Distribucion acumulada del resultado real en el orden H, D, A.

    Devuelve las dos primeras acumuladas (la tercera siempre es 1.0).
    """
    home = 1.0 if actual_outcome == "H" else 0.0
    draw = 1.0 if actual_outcome == "D" else 0.0
    return home, home + draw


def ranked_probability_score(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    actual_outcome: str,
) -> float:
    """Ranked Probability Score para un pronostico 1X2 ordinal (menor es mejor).

    RPS = (1/(K-1)) * sum_{i=1}^{K-1} (F_i - O_i)^2, con K=3 clases ordinales.
    F_i es la probabilidad acumulada del pronostico hasta la clase i; O_i la del
    resultado real. Rango [0, 1]: 0 es pronostico perfecto.
    """
    if actual_outcome not in _ORDINAL_OUTCOMES:
        raise ValueError(f"actual_outcome debe ser uno de {_ORDINAL_OUTCOMES}, recibido {actual_outcome!r}")

    home = float(home_prob)
    draw = float(draw_prob)
    away = float(away_prob)
    total = home + draw + away
    if total <= 0:
        raise ValueError("Las probabilidades 1X2 deben sumar un total positivo.")

    # Normaliza por robustez: la entrada puede no sumar exactamente 1.0.
    forecast_c1 = home / total
    forecast_c2 = (home + draw) / total
    outcome_c1, outcome_c2 = _outcome_cumulative(actual_outcome)

    return ((forecast_c1 - outcome_c1) ** 2 + (forecast_c2 - outcome_c2) ** 2) / 2.0
