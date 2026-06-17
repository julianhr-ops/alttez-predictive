import argparse
import math
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alttez.api.settings import load_local_env
from alttez.inference import build_active_predictor, build_prediction_committee
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import WORLD_CUP_2026_HOST_TEAMS, canonical_team_name


NEUTRAL_STRENGTH = 1.0
METRICS_PATH = Path("poisson_team_metrics.csv")
HISTORICAL_SEEDS_PATH = Path("world_cup_historical_seeds.csv")
FIXTURES_PATH = Path("mundial_2026_fixtures.csv")
OUTPUT_PATH = Path("alttez_predictions_output.csv")


def normalizar_equipo(value):
    """Normaliza llaves de equipos para cruces estables entre capas."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def valor_fuerza(value, default=NEUTRAL_STRENGTH):
    """Convierte una fuerza a float finito; si no es valida, devuelve neutral."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    return numeric_value if np.isfinite(numeric_value) else default


def construir_lookup_fuerzas(df, *, source):
    """Construye un diccionario equipo -> fuerzas con acceso promedio O(1)."""
    if df.empty:
        return {}

    working_df = df.copy()
    alias_groups = {
        "equipo": ("team", "Team"),
        "fuerza_ataque": ("attack_strength", "fuerza_ataque_historica"),
        "fuerza_defensa": ("defense_strength", "fuerza_defensa_historica"),
    }
    for canonical_column, aliases in alias_groups.items():
        if canonical_column not in working_df.columns:
            alias_column = next((alias for alias in aliases if alias in working_df.columns), None)
            if alias_column is not None:
                working_df[canonical_column] = working_df[alias_column]
        else:
            working_df[canonical_column] = working_df[canonical_column].astype("object")
            for alias_column in aliases:
                if alias_column in working_df.columns:
                    missing_mask = working_df[canonical_column].isna() | (
                        working_df[canonical_column].astype(str).str.strip() == ""
                    )
                    working_df.loc[missing_mask, canonical_column] = working_df.loc[missing_mask, alias_column]

    required_columns = {"equipo", "fuerza_ataque", "fuerza_defensa"}
    missing_columns = required_columns - set(working_df.columns)
    if missing_columns:
        print(f"ADVERTENCIA: Capa {source} ignorada: faltan columnas {sorted(missing_columns)}")
        return {}

    working_df["equipo_key"] = working_df["equipo"].map(normalizar_equipo)
    working_df = working_df[working_df["equipo_key"] != ""]

    lookup = {}
    for _, row in working_df.iterrows():
        lookup[row["equipo_key"]] = {
            "fuerza_ataque": valor_fuerza(row["fuerza_ataque"]),
            "fuerza_defensa": valor_fuerza(row["fuerza_defensa"]),
            "source": source,
        }
    return lookup


def cargar_lookup_metricas():
    """Carga semillas historicas y metricas calientes; las metricas actuales tienen prioridad."""
    historical_lookup = {}
    if HISTORICAL_SEEDS_PATH.exists():
        historical_lookup = construir_lookup_fuerzas(pd.read_csv(HISTORICAL_SEEDS_PATH), source="historical_seed")
    else:
        print(
            "INFO: No se encontro 'world_cup_historical_seeds.csv'; "
            "se usara fallback neutro para equipos sin metricas actuales."
        )

    metrics_lookup = construir_lookup_fuerzas(pd.read_csv(METRICS_PATH), source="current_metrics")
    return {**historical_lookup, **metrics_lookup}


def obtener_fuerzas(metrics_lookup, team):
    team_key = normalizar_equipo(team)
    if not team_key:
        return NEUTRAL_STRENGTH, NEUTRAL_STRENGTH, "neutral_missing_team"

    strengths = metrics_lookup.get(team_key)
    if strengths is None:
        return NEUTRAL_STRENGTH, NEUTRAL_STRENGTH, "neutral_unseen_team"

    return (
        valor_fuerza(strengths.get("fuerza_ataque")),
        valor_fuerza(strengths.get("fuerza_defensa")),
        strengths.get("source", "unknown"),
    )


def calcular_probabilidad_poisson(lmbda, k):
    """Calcula la probabilidad de que ocurran exactamente k goles dado un lambda."""
    if not np.isfinite(lmbda) or lmbda < 0:
        return 0.0
    if lmbda == 0:
        return 1.0 if k == 0 else 0.0
    return (lmbda**k) * math.exp(-lmbda) / math.factorial(k)


def cargar_partidos_pendientes(fixtures_path: Path = FIXTURES_PATH) -> pd.DataFrame:
    df_fixtures = pd.read_csv(fixtures_path)
    df_fixtures["home_score"] = pd.to_numeric(df_fixtures.get("home_score"), errors="coerce")
    df_fixtures["away_score"] = pd.to_numeric(df_fixtures.get("away_score"), errors="coerce")
    df_fixtures["score"] = (
        df_fixtures.get("score", pd.Series(index=df_fixtures.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df_fixtures["home_team"] = df_fixtures["home_team"].map(normalizar_equipo)
    df_fixtures["away_team"] = df_fixtures["away_team"].map(normalizar_equipo)
    proximos_partidos = df_fixtures[df_fixtures["score"].eq("")].copy()
    return proximos_partidos[proximos_partidos["home_score"].isna()].copy()


def es_sede_neutral(row: pd.Series, home_team: str, away_team: str) -> bool:
    venue = str(row.get("venue") or "")
    if "Neutral Site" in venue:
        return home_team not in WORLD_CUP_2026_HOST_TEAMS and away_team not in WORLD_CUP_2026_HOST_TEAMS
    return True


def contexto_mundial(row: pd.Series) -> dict[str, Any]:
    return {
        "tournament": "FIFA World Cup 2026",
        "edition": "2026",
        "stage": str(row.get("round") or ""),
        "round": str(row.get("round") or ""),
        "date": str(row.get("date") or ""),
    }


def resultado_no_soportado(row: pd.Series, local: str, visita: str, prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "fecha": row.get("date"),
        "partido": row.get("game", "N/A"),
        "local": local,
        "visita": visita,
        "motor": prediction.get("model_family"),
        "model_version": prediction.get("model_version"),
        "prediction_supported": False,
        "support_message": prediction.get("support_message"),
        "used_fallback": bool(prediction.get("used_fallback", False)),
        "fallback_reason": prediction.get("fallback_reason"),
        "exp_goles_local": None,
        "exp_goles_visita": None,
        "prob_local_%": None,
        "prob_empate_%": None,
        "prob_visita_%": None,
        "marcador_mas_probable": "N/D",
        "marcador_ev": "N/D",
        "ev_marcador": None,
    }


def fila_prediccion_modelo(row: pd.Series, predictor, known_names: set[str]) -> dict[str, Any]:
    local = canonical_team_name(row["home_team"], known_names)
    visita = canonical_team_name(row["away_team"], known_names)
    round_name = str(row.get("round") or "")
    is_knockout = "group" not in round_name.lower()
    prediction = predictor.predict_match(
        local,
        visita,
        is_neutral=es_sede_neutral(row, local, visita),
        is_knockout=is_knockout,
        tournament_context=contexto_mundial(row),
    )

    if not prediction.get("prediction_supported", True):
        return resultado_no_soportado(row, local, visita, prediction)

    return {
        "fecha": row.get("date"),
        "partido": row.get("game", "N/A"),
        "local": local,
        "visita": visita,
        "motor": prediction.get("model_family"),
        "model_version": prediction.get("model_version"),
        "prediction_supported": True,
        "support_message": prediction.get("support_message"),
        "used_fallback": bool(prediction.get("used_fallback", False)),
        "fallback_reason": prediction.get("fallback_reason"),
        "exp_goles_local": round(float(prediction.get("lambda_h") or 0.0), 2),
        "exp_goles_visita": round(float(prediction.get("lambda_a") or 0.0), 2),
        "prob_local_%": round(float(prediction.get("prob_home") or 0.0) * 100, 2),
        "prob_empate_%": round(float(prediction.get("prob_draw") or 0.0) * 100, 2),
        "prob_visita_%": round(float(prediction.get("prob_away") or 0.0) * 100, 2),
        "marcador_mas_probable": str(prediction.get("primary_score") or prediction.get("most_likely_score") or "N/D"),
        "marcador_ev": str(prediction.get("ev_optimal_score") or "N/D"),
        "ev_marcador": prediction.get("ev_optimal_value"),
        "best_market": prediction.get("best_market"),
        "best_market_probability_%": round(float(prediction.get("best_market_probability") or 0.0) * 100, 2),
        "committee_consensus_used": bool(prediction.get("committee_consensus_used", False)),
        "discordance_warning": bool(prediction.get("discordance_warning", False)),
    }


def generar_predicciones_modelo(
    engine: str,
    target_date: str,
    database_url: str | None,
    min_supported_matches: int,
) -> None:
    print(f"Inicializando motor ALTTEZ: {engine}")
    load_local_env(override=True)
    os.environ["ALTTEZ_MIN_SUPPORTED_MATCHES"] = str(max(0, min_supported_matches))
    proximos_partidos = cargar_partidos_pendientes()
    total_a_predecir = len(proximos_partidos)
    print(f"Partidos futuros detectados en el fixture para prediccion: {total_a_predecir}")
    if total_a_predecir == 0:
        print("ADVERTENCIA: No hay partidos pendientes por jugar en el fixture cargado.")
        return

    historical_df = InternationalLoader().fetch_data()
    known_names = set(historical_df["home_team"].dropna()).union(set(historical_df["away_team"].dropna()))
    if engine == "active":
        predictor = build_active_predictor(historical_df, target_date, database_url=database_url)
    else:
        predictor = build_prediction_committee(historical_df, target_date, database_url=database_url)

    predicciones_list = [
        fila_prediccion_modelo(row, predictor, known_names)
        for _, row in proximos_partidos.iterrows()
    ]
    df_output = pd.DataFrame(predicciones_list)
    df_output.to_csv(OUTPUT_PATH, index=False)
    print(f"Predicciones calculadas con {predictor.model_family}/{predictor.model_version}. Archivo '{OUTPUT_PATH}' exportado.")
    print("\nProximos 5 partidos con sus probabilidades calculadas:")
    print(
        df_output[
            [
                "fecha",
                "local",
                "visita",
                "motor",
                "prob_local_%",
                "prob_empate_%",
                "prob_visita_%",
                "marcador_mas_probable",
            ]
        ].head()
    )


def generar_predicciones_legacy_poisson() -> None:
    print("Inicializando el Motor Predictivo ALTTEZ legacy Poisson...")
    try:
        metrics_dict = cargar_lookup_metricas()
        proximos_partidos = cargar_partidos_pendientes()
    except FileNotFoundError as e:
        print(f"ERROR: Faltan archivos base para predecir. {e}")
        return

    total_a_predecir = len(proximos_partidos)
    print(f"Partidos futuros detectados en el fixture para prediccion: {total_a_predecir}")
    if total_a_predecir == 0:
        print("ADVERTENCIA: No hay partidos pendientes por jugar en el fixture cargado.")
        return

    avg_goles_global = 1.188
    predicciones_list = []
    max_goles = 6
    fallback_counter = {"historical_seed": 0, "neutral_unseen_team": 0, "neutral_missing_team": 0}

    for _, row in proximos_partidos.iterrows():
        local = row["home_team"]
        visita = row["away_team"]
        f_ataque_local, f_defensa_local, source_local = obtener_fuerzas(metrics_dict, local)
        f_ataque_visita, f_defensa_visita, source_visita = obtener_fuerzas(metrics_dict, visita)
        for source in (source_local, source_visita):
            if source in fallback_counter:
                fallback_counter[source] += 1

        lambda_local = f_ataque_local * f_defensa_visita * avg_goles_global
        lambda_visita = f_ataque_visita * f_defensa_local * avg_goles_global

        prob_matrix = np.zeros((max_goles, max_goles))
        for i in range(max_goles):
            for j in range(max_goles):
                prob_matrix[i, j] = (
                    calcular_probabilidad_poisson(lambda_local, i)
                    * calcular_probabilidad_poisson(lambda_visita, j)
                )

        prob_empate = np.trace(prob_matrix)
        prob_local = np.sum(np.tril(prob_matrix, -1))
        prob_visita = np.sum(np.triu(prob_matrix, 1))
        total_p = prob_local + prob_empate + prob_visita
        if total_p > 0:
            prob_local /= total_p
            prob_empate /= total_p
            prob_visita /= total_p

        idx_marcador = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        predicciones_list.append(
            {
                "fecha": row.get("date"),
                "partido": row.get("game", "N/A"),
                "local": local,
                "visita": visita,
                "motor": "legacy_poisson_csv",
                "model_version": "legacy-poisson-csv",
                "prediction_supported": True,
                "fuente_local": source_local,
                "fuente_visita": source_visita,
                "exp_goles_local": round(lambda_local, 2),
                "exp_goles_visita": round(lambda_visita, 2),
                "prob_local_%": round(prob_local * 100, 2),
                "prob_empate_%": round(prob_empate * 100, 2),
                "prob_visita_%": round(prob_visita * 100, 2),
                "marcador_mas_probable": f"{idx_marcador[0]}-{idx_marcador[1]}",
            }
        )

    pd.DataFrame(predicciones_list).to_csv(OUTPUT_PATH, index=False)
    print(f"Predicciones legacy calculadas. Fallbacks aplicados: {fallback_counter}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera predicciones del Mundial 2026 con el motor ALTTEZ activo.")
    parser.add_argument(
        "--engine",
        choices=("committee", "active", "legacy-poisson"),
        default="committee",
        help="committee usa champion + challenger + Poisson; active usa solo el modelo activo; legacy-poisson usa CSV local.",
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="Fecha de corte de prediccion YYYY-MM-DD.")
    parser.add_argument("--database-url", default=None, help="URL de base de datos opcional para registry/modelos.")
    parser.add_argument(
        "--min-supported-matches",
        type=int,
        default=0,
        help="Minimo de partidos historicos para bloquear prediccion. 0 permite cold start con priors.",
    )
    return parser.parse_args()


def generar_predicciones() -> None:
    args = parse_args()
    if args.engine == "legacy-poisson":
        generar_predicciones_legacy_poisson()
        return
    generar_predicciones_modelo(args.engine, args.date, args.database_url, args.min_supported_matches)


if __name__ == "__main__":
    generar_predicciones()
