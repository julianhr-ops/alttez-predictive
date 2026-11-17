from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from alttez.api.db import read_dataframe
from alttez.api.settings import load_local_env
from alttez.dashboard.analysis_helpers import competition_class_label, filter_noise_matches, safe_text
from alttez.inference import BaseMatchPredictor, build_prediction_committee
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import list_matches_for_date
from .constants import DEFAULT_MAX_GOALS
from .formatting import _decision_tier, localize_prediction_text, team_display_name
from .reliability import compute_match_reliability, summarize_reliability


def _build_result_summary(prob_matrix, team_a: str, team_b: str) -> pd.DataFrame:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    max_goals = prob_matrix.shape[0] - 1

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = prob_matrix[home_goals, away_goals]
            if home_goals > away_goals:
                home_win += probability
            elif home_goals < away_goals:
                away_win += probability
            else:
                draw += probability

    return pd.DataFrame(
        {
            "Mercado": [f"Victoria {team_a}", "Empate", f"Victoria {team_b}"],
            "Probabilidad": [home_win, draw, away_win],
        }
    )


def _top_score_probabilities(prob_matrix, limit: int = 8) -> list[dict[str, Any]]:
    if prob_matrix is None:
        return []
    rows = []
    max_goals = prob_matrix.shape[0] - 1
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            rows.append(
                {
                    "score": f"{home_goals} - {away_goals}",
                    "probability": float(prob_matrix[home_goals, away_goals]),
                }
            )
    return sorted(rows, key=lambda item: item["probability"], reverse=True)[:limit]


def _is_prediction_supported(profile_local: dict[str, float], profile_visitante: dict[str, float]) -> tuple[bool, str | None]:
    local_matches = float(profile_local.get("matches", 0.0))
    away_matches = float(profile_visitante.get("matches", 0.0))
    local_effective = float(profile_local.get("effective_matches", 0.0))
    away_effective = float(profile_visitante.get("effective_matches", 0.0))

    if local_matches <= 0 and away_matches <= 0:
        return False, "Sin historial compatible para ambos equipos."
    if local_effective <= 0 and away_effective <= 0:
        return False, "Sin soporte estadistico efectivo para ambos equipos."
    return True, None


def _is_knockout_stage(stage: str | None) -> bool:
    lowered = safe_text(stage).lower()
    keywords = (
        "quarter-final",
        "quarterfinal",
        "semi-final",
        "semifinal",
        "final",
        "round of 16",
        "octavos",
        "cuartos",
        "semifinal",
        "elimin",
        "playoff",
    )
    return any(keyword in lowered for keyword in keywords)


def _support_lookup_from_predictor(predictor: BaseMatchPredictor, row_dict: dict[str, Any]) -> tuple[bool, bool]:
    home_profile = predictor.get_team_profile(row_dict["home_team"])
    away_profile = predictor.get_team_profile(row_dict["away_team"])
    return home_profile.get("matches", 0.0) > 0, away_profile.get("matches", 0.0) > 0


def _load_batch_summary(target_date: date, database_url: str | None = None) -> dict[str, Any]:
    summary = {
        "certification_status": "unknown",
        "certified": False,
        "total_records": 0,
        "certified_records": 0,
        "rejected_records": 0,
        "coverage_pct": 0.0,
        "provider_used": None,
        "quarantine_reason": None,
    }
    try:
        df = read_dataframe(
            """
            select
                batch_id,
                target_date,
                provider_used,
                certification_status,
                certified,
                total_records,
                certified_records,
                rejected_records,
                coverage_pct,
                quarantine_reason,
                created_at
            from public.certification_batches
            where target_date = :target_date
            order by created_at desc
            limit 1
            """,
            params={"target_date": target_date.isoformat()},
            database_url=database_url,
        )
    except Exception:
        return summary

    if df.empty:
        return summary

    row = df.iloc[0].to_dict()
    summary.update(
        {
            "batch_id": row.get("batch_id"),
            "provider_used": row.get("provider_used"),
            "certification_status": row.get("certification_status") or "unknown",
            "certified": bool(row.get("certified", False)),
            "total_records": int(row.get("total_records") or 0),
            "certified_records": int(row.get("certified_records") or 0),
            "rejected_records": int(row.get("rejected_records") or 0),
            "coverage_pct": float(row.get("coverage_pct") or 0.0),
            "quarantine_reason": row.get("quarantine_reason"),
        }
    )
    return summary


def _load_alert_summary(target_date: date, database_url: str | None = None) -> dict[str, Any]:
    base = {
        "total": 0,
        "critical": 0,
        "warning": 0,
        "info": 0,
        "items": [],
    }
    try:
        df = read_dataframe(
            """
            select
                created_at,
                alert_level,
                alert_code,
                title,
                message,
                provider_used,
                delivery_status,
                delivery_channel
            from public.alert_events
            where target_date = :target_date
            order by created_at desc
            limit 25
            """,
            params={"target_date": target_date.isoformat()},
            database_url=database_url,
        )
    except Exception:
        return base

    if df.empty:
        return base

    base["total"] = int(len(df))
    base["critical"] = int((df["alert_level"] == "critical").sum())
    base["warning"] = int((df["alert_level"] == "warning").sum())
    base["info"] = int((df["alert_level"] == "info").sum())
    base["items"] = df.fillna("").to_dict(orient="records")
    return base


def _build_match_brief(row_dict: dict[str, Any], predictor: BaseMatchPredictor, max_goals: int = DEFAULT_MAX_GOALS) -> dict[str, Any]:
    home_team = row_dict["home_team"]
    away_team = row_dict["away_team"]
    home_team_display = team_display_name(home_team)
    away_team_display = team_display_name(away_team)
    prediction = predictor.predict_match(
        home_team,
        away_team,
        is_neutral=bool(row_dict.get("is_neutral", False)),
        is_knockout=_is_knockout_stage(safe_text(row_dict.get("stage"), "Fase por confirmar")),
        max_goals=max_goals,
        tournament_context={
            "tournament": safe_text(row_dict.get("tournament")),
            "edition": safe_text(row_dict.get("season_label")),
            "city": safe_text(row_dict.get("city")),
            "venue_name": safe_text(row_dict.get("venue_name")),
            "weather": row_dict.get("weather") or None,
        },
    )
    profile_local = prediction["profile_local"]
    profile_visitante = prediction["profile_visitante"]
    prediction_supported = bool(prediction["prediction_supported"])
    support_message = prediction["support_message"]
    support_label = prediction["support_label"]

    brief = {
        "match_id": row_dict.get("id"),
        "kickoff_at": safe_text(row_dict.get("kickoff_at"), "Por confirmar"),
        "tournament": safe_text(row_dict.get("tournament"), "Sin torneo"),
        "competition_class": safe_text(row_dict.get("competition_class"), "other_international"),
        "competition_label": competition_class_label(row_dict.get("competition_class")),
        "season_label": safe_text(row_dict.get("season_label")),
        "stage": safe_text(row_dict.get("stage"), "Fase por confirmar"),
        "group_label": safe_text(row_dict.get("group_label")),
        "status": safe_text(row_dict.get("status"), "scheduled"),
        "home_team": home_team_display,
        "away_team": away_team_display,
        "home_team_canonical": home_team,
        "away_team_canonical": away_team,
        "city": safe_text(row_dict.get("city"), "Ciudad pendiente"),
        "venue_name": safe_text(row_dict.get("venue_name")),
        "is_neutral": bool(row_dict.get("is_neutral", False)),
        "support_label": support_label,
        "prediction_supported": prediction_supported,
        "support_message": support_message,
        "matches_local": int(profile_local.get("matches", 0.0)),
        "matches_visitante": int(profile_visitante.get("matches", 0.0)),
        "effective_matches_local": float(profile_local.get("effective_matches", 0.0)),
        "effective_matches_visitante": float(profile_visitante.get("effective_matches", 0.0)),
        "confidence_local": float(profile_local.get("confidence", 0.0)),
        "confidence_visitante": float(profile_visitante.get("confidence", 0.0)),
        "recent_points_local": float(profile_local.get("recent_points_per_match", 0.0)),
        "recent_points_visitante": float(profile_visitante.get("recent_points_per_match", 0.0)),
        "recent_goal_diff_local": float(profile_local.get("recent_goal_diff_per_match", 0.0)),
        "recent_goal_diff_visitante": float(profile_visitante.get("recent_goal_diff_per_match", 0.0)),
        "model_version": prediction["model_version"],
        "model_family": prediction["model_family"],
        "used_fallback": bool(prediction.get("used_fallback", False)),
        "fallback_reason": prediction.get("fallback_reason"),
        "committee_consensus_used": bool(prediction.get("committee_consensus_used", False)),
        "committee_members": prediction.get("committee_members", []),
        "committee_weights": prediction.get("committee_weights", {}),
        "discordance_warning": bool(prediction.get("discordance_warning", False)),
        "member_primary_scores": prediction.get("member_primary_scores", []),
    }

    if not prediction_supported:
        brief.update(
            {
                "lambda_h": None,
                "lambda_a": None,
                "xg_total": None,
                "primary_score": "N/D",
                "primary_score_probability": None,
                "primary_outcome": None,
                "most_likely_score": "N/D",
                "most_likely_score_probability": None,
                "primary_ev": None,
                "ev_optimal_score": "N/D",
                "ev_optimal_probability": None,
                "ev_optimal_value": None,
                "primary_score_scenarios": [],
                "scoreline_candidates": [],
                "score_probability_chart": [],
                "best_market": "Sin soporte",
                "best_market_probability": 0.0,
                "prob_home": None,
                "prob_draw": None,
                "prob_away": None,
                "decision_tier": "Baja confianza",
                "alternatives": [],
            }
        )
        return brief

    avg_confidence = (brief["confidence_local"] + brief["confidence_visitante"]) / 2
    best_market_probability = float(prediction.get("best_market_probability", 0.0) or 0.0)

    brief.update(
        {
            "lambda_h": prediction.get("lambda_h"),
            "lambda_a": prediction.get("lambda_a"),
            "xg_total": round(float(prediction["lambda_h"]) + float(prediction["lambda_a"]), 2),
            "primary_score": prediction.get("primary_score", "N/D"),
            "primary_score_probability": prediction.get("primary_score_probability"),
            "primary_outcome": prediction.get("primary_outcome"),
            "most_likely_score": prediction.get("most_likely_score"),
            "most_likely_score_probability": prediction.get("most_likely_score_probability"),
            "primary_ev": prediction.get("primary_ev"),
            "ev_optimal_score": prediction.get("ev_optimal_score"),
            "ev_optimal_probability": prediction.get("ev_optimal_probability"),
            "ev_optimal_value": prediction.get("ev_optimal_value"),
            "primary_score_scenarios": prediction.get("primary_score_scenarios", []),
            "scoreline_candidates": prediction.get("scoreline_candidates", []),
            "score_probability_chart": _top_score_probabilities(prediction.get("prob_matrix")),
            "best_market": localize_prediction_text(prediction.get("best_market") or "Por definir"),
            "best_market_probability": best_market_probability,
            "prob_home": prediction.get("prob_home"),
            "prob_draw": prediction.get("prob_draw"),
            "prob_away": prediction.get("prob_away"),
            "raw_prob_home": prediction.get("raw_prob_home"),
            "raw_prob_draw": prediction.get("raw_prob_draw"),
            "raw_prob_away": prediction.get("raw_prob_away"),
            "serving_policy": prediction.get("serving_policy"),
            "decision_tier": _decision_tier(support_label, avg_confidence, best_market_probability),
            "alternatives": prediction.get("recommendations", pd.DataFrame()).iloc[1:4].to_dict(orient="records")
            if isinstance(prediction.get("recommendations"), pd.DataFrame)
            else [],
        }
    )
    return brief


def _summarize_brief(match_briefs: list[dict[str, Any]], batch_summary: dict[str, Any], alert_summary: dict[str, Any]) -> dict[str, Any]:
    analyzable = [item for item in match_briefs if item["support_label"] == "Analizable"]
    partial = [item for item in match_briefs if item["support_label"] == "Soporte parcial"]
    unsupported = [item for item in match_briefs if item["support_label"] == "Sin historial suficiente"]
    high_priority = [item for item in match_briefs if item["decision_tier"] == "Alta prioridad"]
    ranked = sorted(
        match_briefs,
        key=lambda item: (
            0 if item["decision_tier"] == "Alta prioridad" else 1 if item["decision_tier"] == "Seguimiento" else 2,
            -((item["confidence_local"] + item["confidence_visitante"]) / 2),
            -(item.get("best_market_probability") or 0.0),
        ),
    )

    return {
        "total_matches": len(match_briefs),
        "analyzable_matches": len(analyzable),
        "partial_matches": len(partial),
        "unsupported_matches": len(unsupported),
        "high_priority_matches": len(high_priority),
        "critical_alerts": alert_summary["critical"],
        "warning_alerts": alert_summary["warning"],
        "certification_status": batch_summary.get("certification_status", "unknown"),
        "coverage_pct": float(batch_summary.get("coverage_pct") or 0.0),
        "ranked_matches": ranked,
    }


def build_daily_brief_payload(
    target_date: date,
    *,
    database_url: str | None = None,
    include_statuses: tuple[str, ...] = ("scheduled", "in_progress", "finished", "cancelled"),
) -> dict[str, Any]:
    load_local_env(override=True)
    agenda_df = list_matches_for_date(target_date, include_statuses=include_statuses, database_url=database_url)
    historical_loader = InternationalLoader()
    historical_df = historical_loader.fetch_data()
    predictor = build_prediction_committee(historical_df, target_date, database_url=database_url)
    filtered_agenda = filter_noise_matches(
        agenda_df,
        lambda row_dict: _support_lookup_from_predictor(predictor, row_dict),
    )

    match_briefs = [_build_match_brief(row._asdict(), predictor) for row in filtered_agenda.itertuples(index=False)]
    for brief in match_briefs:
        brief["reliability"] = compute_match_reliability(brief, run_used_fallback=predictor.used_fallback)
    batch_summary = _load_batch_summary(target_date, database_url=database_url)
    alert_summary = _load_alert_summary(target_date, database_url=database_url)
    summary = _summarize_brief(match_briefs, batch_summary, alert_summary)
    reliability_summary = summarize_reliability(match_briefs)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date.isoformat(),
        "model_version": predictor.model_version,
        "model_family": predictor.model_family,
        "used_fallback": predictor.used_fallback,
        "fallback_reason": predictor.fallback_reason,
        "batch_summary": batch_summary,
        "alert_summary": alert_summary,
        "summary": summary,
        "reliability_summary": reliability_summary,
        "matches": match_briefs,
    }


