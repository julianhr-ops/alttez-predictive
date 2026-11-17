from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alttez.api.db import read_dataframe
from alttez.operations.prediction_memory import select_latest_pre_kickoff


DEFAULT_EXPANSION_LOOKBACK_YEARS = 4
DEFAULT_EXPANSION_MIN_MATCHES = 60
DEFAULT_EXPANSION_MIN_TEAMS = 10
DEFAULT_EXPANSION_MIN_EVALUATED = 20
DEFAULT_EXPANSION_MIN_ANALYZABLE_RATE = 0.55
DEFAULT_EXPANSION_MAX_BRIER = 0.66


@dataclass(frozen=True)
class ExpansionThresholds:
    lookback_years: int = DEFAULT_EXPANSION_LOOKBACK_YEARS
    min_matches: int = DEFAULT_EXPANSION_MIN_MATCHES
    min_teams: int = DEFAULT_EXPANSION_MIN_TEAMS
    min_evaluated_predictions: int = DEFAULT_EXPANSION_MIN_EVALUATED
    min_analyzable_rate: float = DEFAULT_EXPANSION_MIN_ANALYZABLE_RATE
    max_brier_score: float = DEFAULT_EXPANSION_MAX_BRIER


def load_expansion_thresholds() -> ExpansionThresholds:
    return ExpansionThresholds(
        lookback_years=max(1, int(os.getenv("ALTTEZ_EXPANSION_LOOKBACK_YEARS", str(DEFAULT_EXPANSION_LOOKBACK_YEARS)))),
        min_matches=max(1, int(os.getenv("ALTTEZ_EXPANSION_MIN_MATCHES", str(DEFAULT_EXPANSION_MIN_MATCHES)))),
        min_teams=max(2, int(os.getenv("ALTTEZ_EXPANSION_MIN_TEAMS", str(DEFAULT_EXPANSION_MIN_TEAMS)))),
        min_evaluated_predictions=max(1, int(os.getenv("ALTTEZ_EXPANSION_MIN_EVALUATED", str(DEFAULT_EXPANSION_MIN_EVALUATED)))),
        min_analyzable_rate=min(1.0, max(0.0, float(os.getenv("ALTTEZ_EXPANSION_MIN_ANALYZABLE_RATE", str(DEFAULT_EXPANSION_MIN_ANALYZABLE_RATE))))),
        max_brier_score=max(0.1, float(os.getenv("ALTTEZ_EXPANSION_MAX_BRIER", str(DEFAULT_EXPANSION_MAX_BRIER)))),
    )


def classify_competition_readiness(row: dict[str, Any], thresholds: ExpansionThresholds) -> tuple[str, str]:
    matches_total = int(row.get("matches_total") or 0)
    teams_total = int(row.get("teams_total") or 0)
    evaluated_predictions = int(row.get("evaluated_predictions") or 0)
    analyzable_rate = float(row.get("analyzable_rate") or 0.0)
    brier_score = row.get("brier_score")
    competition_class = str(row.get("competition_class") or "other_international").strip().lower()

    if matches_total < thresholds.min_matches:
        return "blocked", "Volumen historico insuficiente para abrir cobertura estable."
    if teams_total < thresholds.min_teams:
        return "blocked", "Poca diversidad competitiva para sostener calibracion."
    if analyzable_rate < thresholds.min_analyzable_rate:
        return "watchlist", "La tasa de partidos analizables aun es baja para escalar."
    if evaluated_predictions < thresholds.min_evaluated_predictions:
        return "watchlist", "Todavia no hay suficiente muestra cerrada para auditar el comportamiento."
    if brier_score is not None and float(brier_score) > thresholds.max_brier_score:
        return "watchlist", "La calibracion historica aun no es suficientemente solida."
    if competition_class in {"friendly", "other_international"} and evaluated_predictions < (thresholds.min_evaluated_predictions * 2):
        return "watchlist", "Competencia volatil; requiere mas evidencia cerrada."
    return "approved", "Competencia apta para ampliar cobertura diaria."


def aggregate_prediction_memory(predictions: pd.DataFrame) -> pd.DataFrame:
    """Conteos y metricas por competencia con UNA prediccion por (partido, model_version).

    Cada ejecucion diaria inserta una fila nueva por partido; contarlas todas infla
    ``predicted_matches`` y ``evaluated_predictions`` y adelanta el umbral de expansion.
    """
    keys = ["tournament_name", "competition_class"]
    frame = predictions.assign(competition_class=predictions["competition_class"].fillna("other_international"))
    predicted = select_latest_pre_kickoff(frame)
    analyzable = (
        predicted.assign(analyzable=(predicted["support_label"] == "Analizable").astype(float))
        .groupby(keys)
        .agg(analyzable_rate=("analyzable", "mean"), predicted_matches=("analyzable", "size"))
        .reset_index()
    )
    evaluated_rows = select_latest_pre_kickoff(frame[frame["brier_1x2"].notna()])
    evaluated = (
        evaluated_rows.assign(accuracy=evaluated_rows["hit_1x2"].astype(float))
        .groupby(keys)
        .agg(
            evaluated_predictions=("brier_1x2", "size"),
            brier_score=("brier_1x2", "mean"),
            log_loss=("log_loss_1x2", "mean"),
            accuracy_1x2=("accuracy", "mean"),
        )
        .reset_index()
    )
    return analyzable.merge(evaluated, on=keys, how="outer")


def load_expansion_candidates(
    *,
    database_url: str | None = None,
    thresholds: ExpansionThresholds | None = None,
) -> pd.DataFrame:
    effective_thresholds = thresholds or load_expansion_thresholds()
    try:
        df = read_dataframe(
            """
            with base_matches as (
                select
                    m.id,
                    m.match_date,
                    coalesce(t.name, m.competition_normalized_name, 'Sin torneo') as tournament_name,
                    coalesce(m.competition_class, 'other_international') as competition_class,
                    m.home_team_id,
                    m.away_team_id
                from public.matches as m
                left join public.tournaments as t
                    on t.id = m.tournament_id
                where m.match_date >= (current_date - (:lookback_years * interval '1 year'))
            ),
            team_universe as (
                select tournament_name, competition_class, home_team_id as team_id from base_matches
                union
                select tournament_name, competition_class, away_team_id as team_id from base_matches
            )
            select
                bm.tournament_name,
                bm.competition_class,
                count(*) as matches_total,
                count(distinct tu.team_id) as teams_total
            from base_matches as bm
            left join team_universe as tu
                on tu.tournament_name = bm.tournament_name
               and tu.competition_class = bm.competition_class
            group by
                bm.tournament_name,
                bm.competition_class
            order by matches_total desc, tournament_name
            """,
            params={"lookback_years": effective_thresholds.lookback_years},
            database_url=database_url,
        )
        predictions = read_dataframe(
            """
            select
                mp.match_id,
                mp.model_version,
                mp.prediction_created_at,
                mp.kickoff_at,
                mp.tournament_name,
                mp.competition_class,
                mp.support_label,
                pe.brier_1x2,
                pe.log_loss_1x2,
                pe.hit_1x2
            from public.match_predictions as mp
            left join public.prediction_evaluations as pe
                on pe.prediction_id = mp.id
            """,
            database_url=database_url,
        )
    except Exception:
        return pd.DataFrame()

    df = df.merge(aggregate_prediction_memory(predictions), on=["tournament_name", "competition_class"], how="left")
    df["predicted_matches"] = df["predicted_matches"].fillna(0).astype(int)
    df["analyzable_rate"] = df["analyzable_rate"].fillna(0.0)
    df["evaluated_predictions"] = df["evaluated_predictions"].fillna(0).astype(int)

    if df.empty:
        return df

    readiness = df.apply(
        lambda row: classify_competition_readiness(row.to_dict(), effective_thresholds),
        axis=1,
        result_type="expand",
    )
    readiness.columns = ["readiness", "readiness_reason"]
    df = pd.concat([df, readiness], axis=1)
    readiness_order = {"approved": 0, "watchlist": 1, "blocked": 2}
    df["readiness_rank"] = df["readiness"].map(readiness_order).fillna(9)
    df = df.sort_values(["readiness_rank", "matches_total", "tournament_name"], ascending=[True, False, True]).drop(columns=["readiness_rank"])
    return df.reset_index(drop=True)

