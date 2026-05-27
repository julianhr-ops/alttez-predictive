from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alttez.api.db import read_dataframe


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
            ),
            analyzable as (
                select
                    mp.tournament_name,
                    coalesce(mp.competition_class, 'other_international') as competition_class,
                    avg(case when mp.support_label = 'Analizable' then 1.0 else 0.0 end) as analyzable_rate,
                    count(*) as predicted_matches
                from public.match_predictions as mp
                group by mp.tournament_name, coalesce(mp.competition_class, 'other_international')
            ),
            evaluated as (
                select
                    mp.tournament_name,
                    coalesce(mp.competition_class, 'other_international') as competition_class,
                    count(*) as evaluated_predictions,
                    avg(pe.brier_1x2) as brier_score,
                    avg(pe.log_loss_1x2) as log_loss,
                    avg(case when pe.hit_1x2 then 1.0 else 0.0 end) as accuracy_1x2
                from public.prediction_evaluations as pe
                join public.match_predictions as mp
                    on mp.id = pe.prediction_id
                group by mp.tournament_name, coalesce(mp.competition_class, 'other_international')
            )
            select
                bm.tournament_name,
                bm.competition_class,
                count(*) as matches_total,
                count(distinct tu.team_id) as teams_total,
                coalesce(a.predicted_matches, 0) as predicted_matches,
                coalesce(a.analyzable_rate, 0.0) as analyzable_rate,
                coalesce(e.evaluated_predictions, 0) as evaluated_predictions,
                e.brier_score,
                e.log_loss,
                e.accuracy_1x2
            from base_matches as bm
            left join team_universe as tu
                on tu.tournament_name = bm.tournament_name
               and tu.competition_class = bm.competition_class
            left join analyzable as a
                on a.tournament_name = bm.tournament_name
               and a.competition_class = bm.competition_class
            left join evaluated as e
                on e.tournament_name = bm.tournament_name
               and e.competition_class = bm.competition_class
            group by
                bm.tournament_name,
                bm.competition_class,
                a.predicted_matches,
                a.analyzable_rate,
                e.evaluated_predictions,
                e.brier_score,
                e.log_loss,
                e.accuracy_1x2
            order by matches_total desc, tournament_name
            """,
            params={"lookback_years": effective_thresholds.lookback_years},
            database_url=database_url,
        )
    except Exception:
        return pd.DataFrame()

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

