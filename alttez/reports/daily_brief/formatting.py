from __future__ import annotations

import math
from typing import Any

import pandas as pd

from alttez.dashboard.analysis_helpers import safe_text


def _decision_tier(support_label: str, avg_confidence: float, best_market_probability: float) -> str:
    if support_label == "Analizable" and avg_confidence >= 0.6 and best_market_probability >= 0.45:
        return "Alta prioridad"
    if support_label in {"Analizable", "Soporte parcial"} and avg_confidence >= 0.35:
        return "Seguimiento"
    return "Baja confianza"


def _format_probability(value: float) -> str:
    return f"{value:.1%}"


TEAM_DISPLAY_NAMES_ES = {
    "Algeria": "Argelia",
    "Argentina": "Argentina",
    "Australia": "Australia",
    "Austria": "Austria",
    "Belgium": "B\u00e9lgica",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Brazil": "Brasil",
    "Canada": "Canad\u00e1",
    "Cape Verde": "Cabo Verde",
    "Colombia": "Colombia",
    "Congo DR": "Republica Democratica del Congo",
    "DR Congo": "Republica Democratica del Congo",
    "Curaçao": "Curazao",
    "Czechia": "Chequia",
    "Czech Republic": "Chequia",
    "Ecuador": "Ecuador",
    "Egypt": "Egipto",
    "France": "Francia",
    "Germany": "Alemania",
    "Haiti": "Hait\u00ed",
    "Iran": "Ir\u00e1n",
    "Ivory Coast": "Costa de Marfil",
    "Japan": "Jap\u00f3n",
    "Jordan": "Jordania",
    "Mexico": "M\u00e9xico",
    "Morocco": "Marruecos",
    "Netherlands": "Pa\u00edses Bajos",
    "New Zealand": "Nueva Zelanda",
    "Paraguay": "Paraguay",
    "Portugal": "Portugal",
    "Qatar": "Catar",
    "Saudi Arabia": "Arabia Saudita",
    "Scotland": "Escocia",
    "Senegal": "Senegal",
    "South Africa": "Sud\u00e1frica",
    "South Korea": "Corea del Sur",
    "Spain": "Espa\u00f1a",
    "Sweden": "Suecia",
    "Switzerland": "Suiza",
    "Tunisia": "T\u00fanez",
    "Turkey": "Turqu\u00eda",
    "United States": "Estados Unidos",
    "Uruguay": "Uruguay",
    "Uzbekistan": "Uzbekist\u00e1n",
}


def team_display_name(team: str | None) -> str:
    team_name = safe_text(team)
    return TEAM_DISPLAY_NAMES_ES.get(team_name, team_name)


def localize_prediction_text(value: str | None) -> str:
    text = safe_text(value)
    if not text:
        return text
    for canonical_name in sorted(TEAM_DISPLAY_NAMES_ES, key=len, reverse=True):
        text = text.replace(canonical_name, TEAM_DISPLAY_NAMES_ES[canonical_name])
    return text


def _fmt_float(value: Any, digits: int = 2, default: str = "N/D") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any, default: str = "N/D") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return _format_probability(float(value))


def prob_over_2_5(lambda_h: Any, lambda_a: Any) -> float | None:
    """Probabilidad de mas de 2.5 goles totales desde las lambdas del modelo.

    La suma de dos Poisson independientes es Poisson(lambda_h + lambda_a), asi que
    P(total > 2.5) = 1 - P(0) - P(1) - P(2). Funcion pura para que el resumen de
    Telegram comunique el nivel de goles esperado (no solo el marcador modal, que
    cae por debajo de la media). Devuelve None si las lambdas no son validas.
    """
    if lambda_h is None or lambda_a is None:
        return None
    try:
        total = float(lambda_h) + float(lambda_a)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(total) or total <= 0:
        return None
    p_le_2 = math.exp(-total) * (1.0 + total + (total * total) / 2.0)
    return max(0.0, min(1.0, 1.0 - p_le_2))


def _fmt_expected_goals(lambda_h: Any, lambda_a: Any) -> str:
    """Texto de goles esperados: total y desglose local-visitante."""
    if lambda_h is None or lambda_a is None:
        return "N/D"
    try:
        lh = float(lambda_h)
        la = float(lambda_a)
    except (TypeError, ValueError):
        return "N/D"
    return f"{lh + la:.1f} ({lh:.1f}-{la:.1f})"


def _fmt_points_profile(points_probability: dict[Any, Any] | None) -> str:
    if not isinstance(points_probability, dict) or not points_probability:
        return "sin perfil"
    parts = []
    for points, probability in points_probability.items():
        try:
            point_label = int(points)
        except (TypeError, ValueError):
            point_label = points
        parts.append(f"{point_label} pts {_fmt_pct(probability)}")
    return ", ".join(parts[:4])


