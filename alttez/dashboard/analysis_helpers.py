from __future__ import annotations

from typing import Any, Callable

import pandas as pd


def safe_text(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def safe_key(value: Any, default: str = "section") -> str:
    text = safe_text(value, default).lower()
    normalized = "".join(char if char.isalnum() else "-" for char in text)
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or default


def competition_class_label(value: str | None) -> str:
    mapping = {
        "friendly": "Amistoso",
        "world_cup": "Copa Mundial",
        "qualifiers": "Eliminatorias",
        "continental": "Continental",
        "other_international": "Internacional",
    }
    normalized = safe_text(value).lower()
    return mapping.get(normalized, safe_text(value, "Internacional"))


def build_match_meta(row_dict: dict) -> str:
    tournament = safe_text(row_dict.get("tournament"), "Sin torneo")
    competition_class = competition_class_label(row_dict.get("competition_class"))
    stage = safe_text(row_dict.get("stage"), "Fase por confirmar")
    group_label = safe_text(row_dict.get("group_label"))
    detail = f"{competition_class} | {tournament}"
    if group_label:
        detail = f"{detail} | {group_label}"
    elif stage:
        detail = f"{detail} | {stage}"
    return detail


def is_youth_or_u17_match(row_dict: dict) -> bool:
    haystack = " ".join(
        safe_text(row_dict.get(key))
        for key in ("home_team", "away_team", "tournament", "stage", "group_label")
    ).lower()
    youth_patterns = (
        "u17",
        "u-17",
        "under 17",
        "under-17",
        "sub 17",
        "sub-17",
    )
    return any(pattern in haystack for pattern in youth_patterns)


def support_level(local_supported: bool, away_supported: bool) -> str:
    if local_supported and away_supported:
        return "Analizable"
    if local_supported or away_supported:
        return "Soporte parcial"
    return "Sin historial suficiente"


def support_status(support_level_value: str) -> str:
    if support_level_value == "Analizable":
        return "success"
    if support_level_value == "Soporte parcial":
        return "warning"
    return "error"


def filter_noise_matches(
    agenda_df: pd.DataFrame,
    support_lookup: Callable[[dict], tuple[bool, bool]],
) -> pd.DataFrame:
    if agenda_df.empty:
        return agenda_df

    keep_rows = []
    for row in agenda_df.itertuples(index=False):
        row_dict = row._asdict()
        local_supported, away_supported = support_lookup(row_dict)
        is_unsupported_youth = is_youth_or_u17_match(row_dict) and not (local_supported or away_supported)
        keep_rows.append(not is_unsupported_youth)

    return agenda_df.loc[keep_rows].copy()


def build_recommendation_summary(result: dict) -> dict:
    recommendations = result.get("recomendaciones")
    result_summary = result.get("result_summary")
    if not isinstance(recommendations, pd.DataFrame) or recommendations.empty:
        return {}

    primary = recommendations.iloc[0].to_dict()
    alternatives = recommendations.iloc[1:4].copy()
    best_market = None
    if isinstance(result_summary, pd.DataFrame) and not result_summary.empty:
        market_frame = result_summary.copy()
        market_values = pd.to_numeric(
            market_frame["Probabilidad"].astype(str).str.rstrip("%"),
            errors="coerce",
        )
        market_frame["Probabilidad_num"] = market_values
        market_frame = market_frame.dropna(subset=["Probabilidad_num"])
        if not market_frame.empty:
            best_market = market_frame.sort_values("Probabilidad_num", ascending=False).iloc[0].to_dict()

    return {
        "primary_score": primary.get("Pronostico", "N/D"),
        "primary_ev": primary.get("EV", "N/D"),
        "alternatives": alternatives,
        "best_market": best_market,
    }
