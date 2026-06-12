from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from alttez.api.db import read_dataframe
from alttez.experiments.tournament_oracle import DEFAULT_DIXON_COLES_RHO, optimize_prediction
from alttez.inference.router import (
    ActiveModelSelection,
    BaseMatchPredictor,
    PoissonMatchPredictor,
    XGBoostMatchPredictor,
    _numeric_result_summary,
    _selection_from_record,
    build_active_predictor,
    most_likely_score_from_matrix,
)
from alttez.ingestion.live_matches import canonical_team_name


LOGGER = logging.getLogger(__name__)
DEFAULT_COMMITTEE_WEIGHTS = {
    "champion": 0.5,
    "challenger": 0.3,
    "poisson_baseline": 0.2,
}
DEFAULT_DISCORDANCE_GOAL_DISTANCE = 3


@dataclass(slots=True)
class CommitteeMember:
    role: str
    predictor: BaseMatchPredictor
    weight: float


def load_latest_challenger_record(*, database_url: str | None = None) -> dict[str, Any] | None:
    try:
        df = read_dataframe(
            """
            select
                version_key,
                model_version,
                model_family,
                training_cutoff,
                trained_at,
                status,
                is_promotion_eligible,
                production_status,
                activation_mode,
                activated_at,
                promotion_reason,
                artifact_path,
                metrics_payload
            from public.model_versions
            where lower(model_family) = 'xgboost'
              and status = 'challenger'
              and artifact_path is not null
            order by trained_at desc
            limit 1
            """,
            database_url=database_url,
        )
    except Exception:
        return None

    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _score_to_tuple(score: str | None) -> tuple[int, int] | None:
    try:
        home, away = str(score or "").split("-", 1)
        return int(home.strip()), int(away.strip())
    except Exception:
        return None


def _score_distance(left: str | None, right: str | None) -> int:
    left_tuple = _score_to_tuple(left)
    right_tuple = _score_to_tuple(right)
    if left_tuple is None or right_tuple is None:
        return 0
    return abs(left_tuple[0] - right_tuple[0]) + abs(left_tuple[1] - right_tuple[1])


def _discordance_warning(member_scores: list[str], threshold: int = DEFAULT_DISCORDANCE_GOAL_DISTANCE) -> bool:
    if len(member_scores) < 2:
        return False
    for left_index, left_score in enumerate(member_scores):
        for right_score in member_scores[left_index + 1:]:
            if _score_distance(left_score, right_score) >= threshold:
                return True
    return False


def _normalize_members(members: list[CommitteeMember]) -> list[CommitteeMember]:
    total_weight = sum(member.weight for member in members if member.weight > 0)
    if total_weight <= 0:
        return members
    return [
        CommitteeMember(member.role, member.predictor, member.weight / total_weight)
        for member in members
        if member.weight > 0
    ]


class PredictionCommittee(BaseMatchPredictor):
    def __init__(self, members: list[CommitteeMember], *, model_version: str = "committee-consensus-v1") -> None:
        if not members:
            raise ValueError("PredictionCommittee requires at least one member.")
        self.members = _normalize_members(members)
        super().__init__(
            selection=ActiveModelSelection(
                version_key=None,
                model_version=model_version,
                model_family="committee",
                artifact_path=None,
                requested_production_status="active",
            )
        )

    def get_team_profile(self, team: str) -> dict[str, float]:
        return self.members[0].predictor.get_team_profile(canonical_team_name(team))

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        is_neutral: bool = False,
        is_knockout: bool = False,
        max_goals: int = 7,
        tournament_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        home_team = canonical_team_name(home_team)
        away_team = canonical_team_name(away_team)
        member_predictions: list[dict[str, Any]] = []
        matrices: list[tuple[float, np.ndarray]] = []

        for member in self.members:
            try:
                prediction = member.predictor.predict_match(
                    home_team,
                    away_team,
                    is_neutral=is_neutral,
                    is_knockout=is_knockout,
                    max_goals=max_goals,
                    tournament_context=tournament_context,
                )
            except Exception as exc:  # noqa: BLE001 - one vote should not kill the committee.
                LOGGER.warning("committee_member_failed role=%s detail=%s", member.role, exc)
                continue

            prob_matrix = prediction.get("prob_matrix")
            if not prediction.get("prediction_supported") or prob_matrix is None:
                member_predictions.append(
                    {
                        "role": member.role,
                        "weight": member.weight,
                        "model_version": prediction.get("model_version"),
                        "model_family": prediction.get("model_family"),
                        "prediction_supported": False,
                        "support_message": prediction.get("support_message"),
                        "support_label": prediction.get("support_label"),
                    }
                )
                continue

            primary_score = prediction.get("primary_score")
            member_predictions.append(
                {
                    "role": member.role,
                    "weight": member.weight,
                    "model_version": prediction.get("model_version"),
                    "model_family": prediction.get("model_family"),
                    "prediction_supported": True,
                    "support_label": prediction.get("support_label"),
                    "primary_score": primary_score,
                    "primary_score_probability": prediction.get("primary_score_probability"),
                    "ev_optimal_score": prediction.get("ev_optimal_score"),
                    "ev_optimal_value": prediction.get("ev_optimal_value"),
                    "lambda_h": prediction.get("lambda_h"),
                    "lambda_a": prediction.get("lambda_a"),
                }
            )
            matrices.append((member.weight, np.asarray(prob_matrix, dtype=float)))

        if not matrices:
            fallback = self.members[0].predictor.predict_match(
                home_team,
                away_team,
                is_neutral=is_neutral,
                is_knockout=is_knockout,
                max_goals=max_goals,
                tournament_context=tournament_context,
            )
            fallback["committee_members"] = member_predictions
            fallback["committee_consensus_used"] = False
            return fallback

        total_weight = sum(weight for weight, _ in matrices)
        consensus_matrix = sum((weight / total_weight) * matrix for weight, matrix in matrices)
        consensus_matrix = consensus_matrix / consensus_matrix.sum()
        recommendations = optimize_prediction(consensus_matrix, is_knockout=is_knockout).head(5).copy()
        recommendations["EV"] = recommendations["EV"].map(lambda value: round(float(value), 3))
        primary_score, primary_score_probability = most_likely_score_from_matrix(consensus_matrix)
        ev_row = recommendations.iloc[0].to_dict() if not recommendations.empty else {"Pronostico": "N/D", "EV": None}
        result_summary = _numeric_result_summary(consensus_matrix, home_team, away_team)
        result_probabilities = {
            row["Mercado"]: float(row["Probabilidad"])
            for row in result_summary.to_dict(orient="records")
        }
        best_market = max(result_summary.to_dict(orient="records"), key=lambda row: float(row["Probabilidad"]))
        supported_members = [item for item in member_predictions if item.get("prediction_supported")]
        support_label = (
            supported_members[0].get("support_label")
            if supported_members
            else member_predictions[0].get("support_label") if member_predictions else "Sin historial suficiente"
        )
        member_scores = [str(item.get("primary_score")) for item in supported_members if item.get("primary_score")]
        member_lambdas_h = [float(item["lambda_h"]) for item in supported_members if item.get("lambda_h") is not None]
        member_lambdas_a = [float(item["lambda_a"]) for item in supported_members if item.get("lambda_a") is not None]
        profile_local = self.get_team_profile(home_team)
        profile_visitante = self.get_team_profile(away_team)

        return {
            "model_version": self.model_version,
            "model_family": self.model_family,
            "requested_model_version": self.requested_model_version,
            "requested_model_family": self.requested_model_family,
            "used_fallback": False,
            "fallback_reason": None,
            "profile_local": profile_local,
            "profile_visitante": profile_visitante,
            "prediction_supported": True,
            "support_message": None,
            "support_label": support_label,
            "lambda_h": float(np.mean(member_lambdas_h)) if member_lambdas_h else None,
            "lambda_a": float(np.mean(member_lambdas_a)) if member_lambdas_a else None,
            "prob_matrix": consensus_matrix,
            "prob_home": result_probabilities.get(f"Victoria {home_team}"),
            "prob_draw": result_probabilities.get("Empate"),
            "prob_away": result_probabilities.get(f"Victoria {away_team}"),
            "result_summary": result_summary,
            "recommendations": recommendations,
            "primary_score": primary_score,
            "primary_score_probability": primary_score_probability,
            "primary_ev": ev_row.get("EV"),
            "ev_optimal_score": str(ev_row.get("Pronostico") or "N/D"),
            "ev_optimal_value": ev_row.get("EV"),
            "score_selection_policy": "committee_consensus_matrix",
            "best_market": str(best_market.get("Mercado") or "Por definir"),
            "best_market_probability": float(best_market.get("Probabilidad") or 0.0),
            "committee_members": member_predictions,
            "committee_consensus_used": True,
            "committee_weights": {member.role: member.weight for member in self.members},
            "discordance_warning": _discordance_warning(member_scores),
            "member_primary_scores": member_scores,
        }


def build_prediction_committee(
    historical_df: pd.DataFrame,
    target_date: date | str,
    *,
    database_url: str | None = None,
) -> PredictionCommittee:
    champion = build_active_predictor(historical_df, target_date, database_url=database_url)
    if isinstance(target_date, date):
        target_date_str = target_date.isoformat()
    else:
        target_date_str = str(target_date)

    members = [
        CommitteeMember("champion", champion, DEFAULT_COMMITTEE_WEIGHTS["champion"]),
        CommitteeMember(
            "poisson_baseline",
            PoissonMatchPredictor(
                historical_df,
                target_date_str,
                selection=ActiveModelSelection(
                    version_key=None,
                    model_version="poisson-baseline-committee",
                    model_family="poisson",
                    artifact_path=None,
                    requested_production_status="committee",
                ),
            ),
            DEFAULT_COMMITTEE_WEIGHTS["poisson_baseline"],
        ),
    ]

    challenger_record = load_latest_challenger_record(database_url=database_url)
    if challenger_record:
        try:
            members.insert(
                1,
                CommitteeMember(
                    "challenger",
                    XGBoostMatchPredictor(
                        historical_df,
                        target_date_str,
                        selection=_selection_from_record(challenger_record),
                        database_url=database_url,
                    ),
                    DEFAULT_COMMITTEE_WEIGHTS["challenger"],
                ),
            )
        except Exception as exc:  # noqa: BLE001 - committee can run without challenger.
            LOGGER.warning("challenger_committee_load_failed detail=%s", exc)

    return PredictionCommittee(members)
