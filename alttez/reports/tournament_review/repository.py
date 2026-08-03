from __future__ import annotations

from datetime import date

import pandas as pd

from alttez.api.db import read_dataframe


def load_tournament_predictions(
    tournament: str,
    *,
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Load every persisted prediction for a tournament, evaluated or pending."""

    normalized = str(tournament).strip()
    if not normalized:
        raise ValueError("Tournament name must not be empty.")

    filters = ["mp.tournament_name ilike :tournament_pattern"]
    params: dict[str, object] = {"tournament_pattern": f"%{normalized}%"}
    if date_from is not None:
        filters.append("mp.match_date >= :date_from")
        params["date_from"] = str(date_from)
    if date_to is not None:
        filters.append("mp.match_date <= :date_to")
        params["date_to"] = str(date_to)

    return read_dataframe(
        f"""
        select
            mp.id as prediction_id,
            mp.match_id,
            mp.prediction_created_at,
            mp.match_date,
            mp.kickoff_at,
            mp.home_team,
            mp.away_team,
            mp.tournament_name,
            mp.competition_class,
            mp.model_version,
            mp.model_family,
            mp.predicted_outcome,
            mp.predicted_home_win_prob as prob_home,
            mp.predicted_draw_prob as prob_draw,
            mp.predicted_away_win_prob as prob_away,
            mp.predicted_scoreline,
            mp.predicted_lambda_home,
            mp.predicted_lambda_away,
            mp.support_label,
            mp.avg_confidence,
            mp.decision_tier,
            mp.payload ->> 'stage' as stage,
            mp.payload ->> 'used_fallback' as used_fallback,
            pe.evaluation_status,
            pe.actual_home_score,
            pe.actual_away_score,
            pe.actual_outcome,
            pe.hit_1x2,
            pe.hit_exact_score,
            pe.brier_1x2,
            pe.log_loss_1x2,
            pe.rps_1x2,
            pe.score_abs_error,
            pe.evaluated_at
        from public.match_predictions as mp
        left join public.prediction_evaluations as pe
            on pe.prediction_id = mp.id
        where {' and '.join(filters)}
        order by mp.match_date, mp.kickoff_at, mp.home_team, mp.away_team
        """,
        params=params,
        database_url=database_url,
    )
