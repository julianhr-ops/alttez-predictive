from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from alttez.api.db import read_dataframe
from alttez.experiments.tournament_oracle import optimize_prediction
from alttez.inference.calibration import load_active_calibration, matrix_1x2
from alttez.inference.committee_weighting import CHAMPION_ROLE, WeightingDecision, resolve_committee_weights
from alttez.features.hierarchical import load_active_hierarchical_strengths
from alttez.inference.router import (
    _raw_probability_fields,
    ActiveModelSelection,
    BaseMatchPredictor,
    BivariatePoissonMatchPredictor,
    HierarchicalMatchPredictor,
    PoissonMatchPredictor,
    XGBoostMatchPredictor,
    _numeric_result_summary,
    _selection_from_record,
    build_active_predictor,
    exact_scoreline_decision,
)
from alttez.ingestion.live_matches import canonical_team_name


LOGGER = logging.getLogger(__name__)
DEFAULT_COMMITTEE_WEIGHTS = {
    "champion": 0.45,
    "challenger": 0.3,
    "poisson_baseline": 0.2,
    "bivariate": 0.15,
    "hierarchical": 0.25,
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
            limit 20
            """,
            database_url=database_url,
        )
    except Exception:
        return None

    if df.empty:
        return None
    for record in df.to_dict(orient="records"):
        artifact_path = str(record.get("artifact_path") or "").strip()
        artifact = Path(artifact_path)
        if not artifact_path or not artifact.exists():
            LOGGER.debug(
                "challenger_record_skipped_missing_artifact version_key=%s artifact_path=%s",
                record.get("version_key"),
                artifact_path or "N/D",
            )
            continue
        if artifact.suffix.lower() != ".joblib":
            LOGGER.debug(
                "challenger_record_skipped_non_model_artifact version_key=%s artifact_path=%s",
                record.get("version_key"),
                artifact_path,
            )
            continue
        try:
            payload = joblib.load(artifact)
        except Exception as exc:  # noqa: BLE001 - invalid challenger should not break committee construction.
            LOGGER.debug(
                "challenger_record_skipped_unloadable_artifact version_key=%s artifact_path=%s detail=%s",
                record.get("version_key"),
                artifact_path,
                exc,
            )
            continue
        if isinstance(payload, dict) and (payload.get("model") is not None or payload.get("models")):
            return record
        LOGGER.debug(
            "challenger_record_skipped_invalid_model_artifact version_key=%s artifact_path=%s",
            record.get("version_key"),
            artifact_path,
        )
    return None


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
    def __init__(
        self,
        members: list[CommitteeMember],
        *,
        model_version: str = "committee-consensus-v1",
        weighting: WeightingDecision | None = None,
    ) -> None:
        if not members:
            raise ValueError("PredictionCommittee requires at least one member.")
        self.members = _normalize_members(members)
        # Decision de pesos (guardia de consenso / error inverso). None = pesos previos.
        self.weighting = weighting
        fallback_members = [
            member for member in self.members if getattr(member.predictor, "used_fallback", False)
        ]
        fallback_reason = (
            "; ".join(
                f"{member.role}: {getattr(member.predictor, 'fallback_reason', None) or 'motivo desconocido'}"
                for member in fallback_members
            )
            if fallback_members
            else None
        )
        super().__init__(
            selection=ActiveModelSelection(
                version_key=None,
                model_version=model_version,
                model_family="committee",
                artifact_path=None,
                requested_production_status="active",
            ),
            used_fallback=bool(fallback_members),
            fallback_reason=fallback_reason,
        )
        # Calibracion post-hoc del consenso (temperature scaling ajustado offline
        # sobre las predicciones cerradas del comite). Neutra si no hay artefacto.
        # Se asigna DESPUES de super().__init__, que fija la calibracion del predictor unico.
        self.calibration = load_active_calibration()

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
            member_home, member_draw, member_away = (float(value) for value in matrix_1x2(prob_matrix))
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
                    "prob_home": member_home,
                    "prob_draw": member_draw,
                    "prob_away": member_away,
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
        # Calibracion del consenso antes de derivar 1X2 y marcadores: coherente
        # para ambos. Neutra (identidad) si no hay artefacto activo.
        # NO se aplica cuando el comite defiere al campeon: la temperatura se
        # ajusto sobre el consenso (blend) y sobre-suavizaria la salida de un
        # campeon ya bien calibrado (p.ej. Poisson).
        deferred = bool(self.weighting and self.weighting.deferred_to_champion)
        calibration_active = (not deferred) and (not self.calibration.is_neutral)
        raw_1x2 = matrix_1x2(consensus_matrix)
        if calibration_active:
            consensus_matrix = self.calibration.calibrate_matrix(consensus_matrix)
        recommendations = optimize_prediction(consensus_matrix, is_knockout=is_knockout).head(5).copy()
        recommendations["EV"] = recommendations["EV"].map(lambda value: round(float(value), 3))
        score_decision = exact_scoreline_decision(consensus_matrix, recommendations, is_knockout=is_knockout)
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
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
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
            **_raw_probability_fields(raw_1x2),
            "result_summary": result_summary,
            "recommendations": recommendations,
            "primary_score": score_decision["primary_score"],
            "primary_score_probability": score_decision["primary_score_probability"],
            "primary_outcome": score_decision["primary_outcome"],
            "most_likely_score": score_decision["most_likely_score"],
            "most_likely_score_probability": score_decision["most_likely_score_probability"],
            "primary_ev": score_decision["primary_ev"],
            "ev_optimal_score": score_decision["ev_optimal_score"],
            "ev_optimal_probability": score_decision["ev_optimal_probability"],
            "ev_optimal_value": score_decision["ev_optimal_value"],
            "primary_score_scenarios": score_decision["primary_score_scenarios"],
            "scoreline_candidates": score_decision["scoreline_candidates"],
            "score_selection_policy": "committee_most_likely_outcome_exact_score",
            "best_market": str(best_market.get("Mercado") or "Por definir"),
            "best_market_probability": float(best_market.get("Probabilidad") or 0.0),
            "committee_members": member_predictions,
            "committee_consensus_used": True,
            "committee_weights": {member.role: member.weight for member in self.members},
            "committee_weighting_strategy": self.weighting.strategy if self.weighting else "default",
            "committee_deferred_to_champion": bool(self.weighting.deferred_to_champion) if self.weighting else False,
            "calibration_temperature": getattr(self.calibration, "temperature", None),
            "calibration_applied": calibration_active,
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
                serving=False,
            ),
            DEFAULT_COMMITTEE_WEIGHTS["poisson_baseline"],
        ),
        CommitteeMember(
            "bivariate",
            BivariatePoissonMatchPredictor(
                historical_df,
                target_date_str,
                selection=ActiveModelSelection(
                    version_key=None,
                    model_version="bivariate-poisson-committee",
                    model_family="bivariate_poisson",
                    artifact_path=None,
                    requested_production_status="committee",
                ),
            ),
            DEFAULT_COMMITTEE_WEIGHTS["bivariate"],
        ),
    ]

    # Miembro jerarquico bayesiano: solo se anade si hay artefacto de fuerzas
    # activo (ajuste offline via scripts/fit_hierarchical_bayes.py). Sin artefacto
    # no aporta senal y se omite para no diluir el consenso.
    if not load_active_hierarchical_strengths().is_empty:
        try:
            members.append(
                CommitteeMember(
                    "hierarchical",
                    HierarchicalMatchPredictor(
                        selection=ActiveModelSelection(
                            version_key=None,
                            model_version="hierarchical-bayes-committee",
                            model_family="hierarchical_bayes",
                            artifact_path=None,
                            requested_production_status="committee",
                        ),
                    ),
                    DEFAULT_COMMITTEE_WEIGHTS["hierarchical"],
                )
            )
        except Exception as exc:  # noqa: BLE001 - committee can run without hierarchical.
            LOGGER.warning("hierarchical_committee_load_failed detail=%s", exc)

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

    weighting = _resolve_member_weights(members, database_url=database_url)
    if weighting is not None:
        members = [
            CommitteeMember(member.role, member.predictor, weighting.weights.get(member.role, member.weight))
            for member in members
        ]
    return PredictionCommittee(members, weighting=weighting)


def _load_champion_paired_losses(
    member_versions: dict[str, str],
    committee_version: str,
    database_url: str | None,
) -> tuple[Sequence[float], Sequence[float]] | None:
    """Perdidas Brier por partido (consenso, campeon) sobre partidos comunes, o None sin muestra.

    Sin ellas la guardia de consenso no se evalua; un fallo de lectura se avisa en
    WARNING en vez de silenciar la guardia.
    """
    champion_version = member_versions.get(CHAMPION_ROLE)
    if champion_version is None:
        return None
    try:
        from alttez.operations.prediction_memory import load_committee_paired_losses

        return load_committee_paired_losses(
            consensus_version=committee_version,
            member_role=CHAMPION_ROLE,
            member_version=champion_version,
            database_url=database_url,
        )
    except Exception as exc:  # noqa: BLE001 - the guard degrades to "not evaluated", never blocks the committee.
        LOGGER.warning("committee_guard_losses_unavailable detail=%s", exc)
        return None


def _resolve_member_weights(
    members: list[CommitteeMember],
    *,
    database_url: str | None,
    committee_version: str = "committee-consensus-v1",
) -> WeightingDecision | None:
    """Reasigna pesos por metricas recientes (guardia de consenso / error inverso).

    Devuelve None ante cualquier fallo de lectura para preservar el arranque del
    comite con sus pesos por defecto (la inferencia nunca depende del registro).
    """
    try:
        from alttez.operations.prediction_memory import load_prediction_learning_audit

        # Ventana amplia: las evaluaciones cerradas pueden ser mas viejas que el
        # default de 180 dias; necesitamos suficiente muestra para ponderar.
        audit = load_prediction_learning_audit(database_url=database_url, lookback_days=365)
        member_versions = {member.role: str(member.predictor.model_version) for member in members}
        return resolve_committee_weights(
            member_versions,
            audit.get("models", []),
            DEFAULT_COMMITTEE_WEIGHTS,
            committee_version=committee_version,
            paired_losses=_load_champion_paired_losses(member_versions, committee_version, database_url),
        )
    except Exception as exc:  # noqa: BLE001 - weighting is an optimization, never a hard dependency.
        LOGGER.warning("committee_weighting_failed detail=%s", exc)
        return None
