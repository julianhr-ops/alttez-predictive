from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from alttez.dashboard.analysis_helpers import support_level
from alttez.evaluation.regulation_points import (
    expected_regulation_points,
    parse_scoreline,
    scoreline_candidate_profiles,
    scoreline_scenarios_for_prediction,
)
from alttez.experiments.tournament_oracle import (
    build_probability_matrix,
    optimize_prediction,
    validate_points_optimal_recommendations,
)
from alttez.features.klement import (
    MODEL_FEATURE_COLUMNS,
    build_klement_feature_state,
    build_match_feature_payload,
)
from alttez.ingestion.live_matches import canonical_team_name
from alttez.features.poisson import PoissonEngine
from alttez.features.bivariate_poisson import BivariatePoissonEngine, load_active_bivariate_params
from alttez.features.hierarchical import HierarchicalBayesEngine, load_active_hierarchical_strengths
from alttez.inference.calibration import load_single_predictor_calibration, matrix_1x2
from alttez.inference.serving_policy import (
    POLICY_ELO_ORDINAL,
    POLICY_POISSON,
    EloUnavailableError,
    ServingPolicy,
    build_serving_policy,
    serving_policy_enabled,
)


def _join_reasons(*reasons: str | None) -> str | None:
    joined = "; ".join(reason for reason in reasons if reason)
    return joined or None


@dataclass(frozen=True, slots=True)
class _ServedMatrix:
    matrix: np.ndarray
    raw_1x2: tuple[float, float, float]
    policy: str
    used_fallback: bool
    fallback_reason: str | None




def _raw_probability_fields(raw_1x2: tuple[float, float, float]) -> dict[str, float]:
    return {"raw_prob_home": raw_1x2[0], "raw_prob_draw": raw_1x2[1], "raw_prob_away": raw_1x2[2]}


LOGGER = logging.getLogger(__name__)

ARTIFACTS_DIR_ENV = "ALTTEZ_ARTIFACTS_DIR"
DEFAULT_ARTIFACTS_DIR = "artifacts"


class ArtifactPathError(ValueError):
    """El artifact_path del registro apunta fuera del directorio de artefactos permitido."""


def resolve_artifact_path(raw_path: str | Path, *, artifacts_dir: str | Path | None = None) -> Path:
    """Resuelve `raw_path` y exige que quede dentro del directorio de artefactos.

    joblib.load ejecuta codigo arbitrario al deserializar, asi que una ruta venida de
    la base de datos no puede salirse de `artifacts/` (ni por `..` ni por rutas
    absolutas ni por enlaces simbolicos): se compara la ruta ya resuelta.
    """
    base = Path(artifacts_dir or os.getenv(ARTIFACTS_DIR_ENV) or DEFAULT_ARTIFACTS_DIR).resolve()
    resolved = Path(raw_path).resolve()
    if not resolved.is_relative_to(base):
        raise ArtifactPathError(f"artifact_path fuera del directorio de artefactos permitido ({base}): {raw_path}")
    return resolved


DEFAULT_POISSON_MODEL_VERSION = "poisson-champion-v2"
DEFAULT_POISSON_MODEL_FAMILY = "poisson"


DEFAULT_POISSON_KWARGS = {
    "years_window": 4,
    "prior_matches": 8,
    "min_history_matches": 5,
    "recency_half_life_days": 365,
    "recent_form_matches": 6,
}
# Single source of truth lives in alttez.features.klement; aliased for compatibility.
XGBOOST_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS
DEFAULT_MAX_GOALS = 7


def _recent_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_recent_form_state(df: pd.DataFrame, window: int = 6) -> dict[str, dict[str, float]]:
    ordered = df.sort_values("date").reset_index(drop=True).copy()
    team_points: dict[str, list[float]] = {}
    team_goal_diff: dict[str, list[float]] = {}
    latest_state: dict[str, dict[str, float]] = {}

    for row in ordered.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        team_points.setdefault(home_team, [])
        team_points.setdefault(away_team, [])
        team_goal_diff.setdefault(home_team, [])
        team_goal_diff.setdefault(away_team, [])

        latest_state[home_team] = {
            "recent_points_per_match": _recent_average(team_points[home_team][-window:]),
            "recent_goal_diff_per_match": _recent_average(team_goal_diff[home_team][-window:]),
        }
        latest_state[away_team] = {
            "recent_points_per_match": _recent_average(team_points[away_team][-window:]),
            "recent_goal_diff_per_match": _recent_average(team_goal_diff[away_team][-window:]),
        }

        home_points = 3.0 if row.home_score > row.away_score else 1.0 if row.home_score == row.away_score else 0.0
        away_points = 3.0 if row.away_score > row.home_score else 1.0 if row.home_score == row.away_score else 0.0
        goal_diff = float(row.home_score - row.away_score)
        team_points[home_team].append(home_points)
        team_points[away_team].append(away_points)
        team_goal_diff[home_team].append(goal_diff)
        team_goal_diff[away_team].append(-goal_diff)

    for team in set(ordered["home_team"]).union(set(ordered["away_team"])):
        latest_state.setdefault(
            team,
            {
                "recent_points_per_match": 0.0,
                "recent_goal_diff_per_match": 0.0,
            },
        )
    return latest_state


def _numeric_result_summary(prob_matrix: np.ndarray, team_a: str, team_b: str) -> pd.DataFrame:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    max_goals = prob_matrix.shape[0] - 1

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = float(prob_matrix[home_goals, away_goals])
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


def most_likely_score_from_matrix(prob_matrix: np.ndarray) -> tuple[str, float]:
    flat_index = int(np.argmax(prob_matrix))
    home_goals, away_goals = np.unravel_index(flat_index, prob_matrix.shape)
    return f"{int(home_goals)} - {int(away_goals)}", float(prob_matrix[home_goals, away_goals])


def outcome_aligned_score_from_matrix(prob_matrix: np.ndarray) -> tuple[str, float, str]:
    home_win = float(np.tril(prob_matrix, -1).sum())
    draw = float(prob_matrix.diagonal().sum())
    away_win = float(np.triu(prob_matrix, 1).sum())
    outcome = max((("H", home_win), ("D", draw), ("A", away_win)), key=lambda item: item[1])[0]

    best_score = "N/D"
    best_probability = -1.0
    max_goals = prob_matrix.shape[0] - 1
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            if outcome == "H" and home_goals <= away_goals:
                continue
            if outcome == "D" and home_goals != away_goals:
                continue
            if outcome == "A" and home_goals >= away_goals:
                continue
            probability = float(prob_matrix[home_goals, away_goals])
            if probability > best_probability:
                best_probability = probability
                best_score = f"{home_goals} - {away_goals}"

    return best_score, best_probability, outcome


def points_optimal_score_from_recommendations(
    prob_matrix: np.ndarray,
    recommendations: pd.DataFrame,
    *,
    is_knockout: bool = False,
) -> tuple[str, float | None, float | None]:
    validate_points_optimal_recommendations(
        prob_matrix,
        recommendations,
        is_knockout=is_knockout,
        tolerance=0.001,
    )
    if recommendations.empty:
        return "N/D", None, None

    row = recommendations.iloc[0].to_dict()
    score = str(row.get("Pronostico") or "N/D")
    try:
        home_goals, away_goals = score.split("-", 1)
        exact_probability = float(prob_matrix[int(home_goals.strip()), int(away_goals.strip())])
    except Exception:
        exact_probability = None
    return score, exact_probability, float(row["EV"])


def exact_scoreline_decision(
    prob_matrix: np.ndarray,
    recommendations: pd.DataFrame,
    *,
    is_knockout: bool = False,
) -> dict[str, Any]:
    most_likely_score, most_likely_probability = most_likely_score_from_matrix(prob_matrix)
    primary_score, primary_score_probability, primary_outcome = outcome_aligned_score_from_matrix(prob_matrix)
    primary_home, primary_away = parse_scoreline(primary_score)
    primary_ev = expected_regulation_points(
        prob_matrix,
        primary_home,
        primary_away,
        is_knockout=is_knockout,
    )
    ev_score, ev_probability, ev_value = points_optimal_score_from_recommendations(
        prob_matrix,
        recommendations,
        is_knockout=is_knockout,
    )
    return {
        "primary_score": primary_score,
        "primary_score_probability": primary_score_probability,
        "primary_outcome": primary_outcome,
        "primary_ev": primary_ev,
        "most_likely_score": most_likely_score,
        "most_likely_score_probability": most_likely_probability,
        "ev_optimal_score": ev_score,
        "ev_optimal_probability": ev_probability,
        "ev_optimal_value": ev_value,
        "primary_score_scenarios": scoreline_scenarios_for_prediction(
            prob_matrix,
            primary_home,
            primary_away,
            is_knockout=is_knockout,
        ),
        "scoreline_candidates": scoreline_candidate_profiles(
            prob_matrix,
            is_knockout=is_knockout,
        ),
    }


def _support_threshold() -> int:
    raw = os.getenv("ALTTEZ_MIN_SUPPORTED_MATCHES", "5").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 5


def _is_prediction_supported(profile_local: dict[str, float], profile_visitante: dict[str, float]) -> tuple[bool, str | None]:
    local_matches = float(profile_local.get("matches", 0.0))
    away_matches = float(profile_visitante.get("matches", 0.0))
    local_effective = float(profile_local.get("effective_matches", 0.0))
    away_effective = float(profile_visitante.get("effective_matches", 0.0))
    threshold = _support_threshold()
    if threshold <= 0:
        return True, None  # gating disabled

    # Stricter gating: BOTH teams must clear the minimum history and have effective
    # (recency-weighted) support. This drops matchups that would only produce noise
    # instead of letting a single well-covered team carry an unsupported prediction.
    if min(local_matches, away_matches) < threshold:
        return False, f"Historial insuficiente: se requieren >= {threshold} partidos por equipo."
    if min(local_effective, away_effective) <= 0:
        return False, "Sin soporte estadistico efectivo reciente para ambos equipos."
    return True, None


@dataclass(slots=True)
class ActiveModelSelection:
    version_key: str | None
    model_version: str
    model_family: str
    artifact_path: str | None
    requested_production_status: str


class BaseMatchPredictor(ABC):
    def __init__(
        self,
        *,
        selection: ActiveModelSelection,
        used_fallback: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        self.requested_version_key = selection.version_key
        self.requested_model_version = selection.model_version
        self.requested_model_family = selection.model_family
        self.requested_production_status = selection.requested_production_status
        self.model_version = selection.model_version
        self.model_family = selection.model_family
        self.used_fallback = used_fallback
        self.fallback_reason = fallback_reason
        # Calibracion del predictor unico (separada del consenso del comite).
        # NEUTRA salvo que ALTTEZ_CALIBRATE_SINGLE_PREDICTOR este encendido.
        self.calibration = load_single_predictor_calibration()

    def _calibrate_matrix(self, prob_matrix: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Calibra la matriz (si la calibracion esta activa) y devuelve tambien su 1X2 crudo, previo a calibrar."""
        raw_1x2 = matrix_1x2(prob_matrix)
        if not self.calibration.is_neutral:
            prob_matrix = self.calibration.calibrate_matrix(prob_matrix)
        return prob_matrix, raw_1x2

    @abstractmethod
    def get_team_profile(self, team: str) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        is_neutral: bool = False,
        is_knockout: bool = False,
        max_goals: int = DEFAULT_MAX_GOALS,
        tournament_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class PoissonMatchPredictor(BaseMatchPredictor):
    def __init__(
        self,
        historical_df: pd.DataFrame,
        target_date: str,
        *,
        selection: ActiveModelSelection,
        poisson_kwargs: dict[str, Any] | None = None,
        used_fallback: bool = False,
        fallback_reason: str | None = None,
        serving: bool | None = None,
    ) -> None:
        super().__init__(selection=selection, used_fallback=used_fallback, fallback_reason=fallback_reason)
        self.model_version = DEFAULT_POISSON_MODEL_VERSION if used_fallback else selection.model_version
        self.model_family = DEFAULT_POISSON_MODEL_FAMILY
        kwargs = DEFAULT_POISSON_KWARGS.copy()
        if poisson_kwargs:
            kwargs.update(poisson_kwargs)
        self.engine = PoissonEngine(**kwargs)
        self.engine.fit(historical_df, target_date=target_date)
        # ``serving`` None follows ALTTEZ_SERVING_POLICY (default ON); committee members that must
        # stay raw Poisson pass False.
        self.serving_policy: ServingPolicy | None = None
        if serving_policy_enabled(serving):
            try:
                self.serving_policy = build_serving_policy(historical_df, target_date)
            except Exception as exc:  # noqa: BLE001 - a broken Elo build must degrade to Poisson, loudly.
                reason = f"serving policy {POLICY_ELO_ORDINAL} unavailable: {type(exc).__name__}: {exc}"
                LOGGER.warning(reason)
                self.used_fallback = True
                self.fallback_reason = _join_reasons(self.fallback_reason, reason)

    def _serve_matrix(self, prob_matrix: np.ndarray, home_team: str, away_team: str, is_neutral: bool) -> _ServedMatrix:
        """Apply the serving policy; the raw Poisson 1X2 is kept either way for the calibration pipeline."""
        raw_1x2 = matrix_1x2(prob_matrix)
        if self.serving_policy is not None:
            try:
                served = self.serving_policy.serve(prob_matrix, home_team, away_team, is_neutral)
            except EloUnavailableError as exc:
                reason = f"serving policy {self.serving_policy.name} unavailable for {home_team} vs {away_team}: {exc}"
                LOGGER.warning(reason)
                matrix, _ = self._calibrate_matrix(prob_matrix)
                return _ServedMatrix(matrix, raw_1x2, POLICY_POISSON, True, _join_reasons(self.fallback_reason, reason))
            # The single-predictor calibration is fitted on Poisson output, so it is not applied to the Elo 1X2.
            return _ServedMatrix(served, raw_1x2, self.serving_policy.name, self.used_fallback, self.fallback_reason)
        matrix, _ = self._calibrate_matrix(prob_matrix)
        return _ServedMatrix(matrix, raw_1x2, POLICY_POISSON, self.used_fallback, self.fallback_reason)

    def get_team_profile(self, team: str) -> dict[str, float]:
        return self.engine.get_team_profile(canonical_team_name(team))

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        is_neutral: bool = False,
        is_knockout: bool = False,
        max_goals: int = DEFAULT_MAX_GOALS,
        tournament_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        home_team = canonical_team_name(home_team)
        away_team = canonical_team_name(away_team)
        profile_local = self.get_team_profile(home_team)
        profile_visitante = self.get_team_profile(away_team)
        prediction_supported, support_message = _is_prediction_supported(profile_local, profile_visitante)
        support_label = support_level(profile_local.get("matches", 0) > 0, profile_visitante.get("matches", 0) > 0)

        if not prediction_supported:
            return {
                "model_version": self.model_version,
                "model_family": self.model_family,
                "requested_model_version": self.requested_model_version,
                "requested_model_family": self.requested_model_family,
                "used_fallback": self.used_fallback,
                "fallback_reason": self.fallback_reason,
                "serving_policy": POLICY_POISSON if self.serving_policy is None else self.serving_policy.name,
                "profile_local": profile_local,
                "profile_visitante": profile_visitante,
                "prediction_supported": False,
                "support_message": support_message,
                "support_label": support_label,
                "lambda_h": None,
                "lambda_a": None,
                "prob_matrix": None,
                "prob_home": None,
                "prob_draw": None,
                "prob_away": None,
                "result_summary": pd.DataFrame(),
                "recommendations": pd.DataFrame(),
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
                "score_selection_policy": "most_likely_outcome_exact_score",
                "best_market": "Sin soporte",
                "best_market_probability": 0.0,
            }

        lambda_h, lambda_a = self.engine.predict_xg(home_team, away_team, is_neutral=is_neutral)
        # rho=None resuelve el rho Dixon-Coles ajustado por MLE (artifact activo),
        # con fallback a DEFAULT_DIXON_COLES_RHO cuando no hay artefacto.
        prob_matrix = build_probability_matrix(lambda_h, lambda_a, max_goals=max_goals, rho=None)
        served = self._serve_matrix(prob_matrix, home_team, away_team, is_neutral)
        prob_matrix, raw_1x2 = served.matrix, served.raw_1x2
        recommendations = optimize_prediction(prob_matrix, is_knockout=is_knockout).head(5).copy()
        recommendations["EV"] = recommendations["EV"].map(lambda value: round(float(value), 3))
        score_decision = exact_scoreline_decision(prob_matrix, recommendations, is_knockout=is_knockout)
        result_summary = _numeric_result_summary(prob_matrix, home_team, away_team)
        result_probabilities = {
            row["Mercado"]: float(row["Probabilidad"])
            for row in result_summary.to_dict(orient="records")
        }
        best_market = max(
            result_summary.to_dict(orient="records"),
            key=lambda row: float(row["Probabilidad"]),
        )

        return {
            "model_version": self.model_version,
            "model_family": self.model_family,
            "requested_model_version": self.requested_model_version,
            "requested_model_family": self.requested_model_family,
            "used_fallback": served.used_fallback,
            "fallback_reason": served.fallback_reason,
            "serving_policy": served.policy,
            "profile_local": profile_local,
            "profile_visitante": profile_visitante,
            "prediction_supported": True,
            "support_message": None,
            "support_label": support_label,
            "lambda_h": float(lambda_h),
            "lambda_a": float(lambda_a),
            "prob_matrix": prob_matrix,
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
            "score_selection_policy": "most_likely_outcome_exact_score",
            "best_market": str(best_market.get("Mercado") or "Por definir"),
            "best_market_probability": float(best_market.get("Probabilidad") or 0.0),
        }


class XGBoostMatchPredictor(BaseMatchPredictor):
    def __init__(
        self,
        historical_df: pd.DataFrame,
        target_date: str,
        *,
        selection: ActiveModelSelection,
        poisson_kwargs: dict[str, Any] | None = None,
        database_url: str | None = None,
    ) -> None:
        super().__init__(selection=selection)
        if not selection.artifact_path:
            raise FileNotFoundError("El modelo activo XGBoost no declara artifact_path.")
        artifact_path = resolve_artifact_path(selection.artifact_path)
        if not artifact_path.exists() and selection.version_key:
            # The binary may not be on this host (ephemeral runner / fresh checkout).
            # Hydrate it from the durable store keyed by version_key before giving up.
            try:
                from alttez.operations.artifact_store import download_artifact

                fetched = download_artifact(selection.version_key, artifact_path, database_url=database_url)
                if fetched is not None:
                    artifact_path = resolve_artifact_path(fetched)
            except Exception:  # noqa: BLE001 - fall through to the explicit not-found error below.
                pass
        if not artifact_path.exists():
            raise FileNotFoundError(f"No existe el artefacto XGBoost activo: {artifact_path}")

        artifact = joblib.load(artifact_path)
        self.artifact_schema_version = int(artifact.get("artifact_schema_version") or 1)
        self.model = artifact.get("model")
        self.models = dict(artifact.get("models") or {})
        self.feature_columns = list(artifact.get("feature_columns") or XGBOOST_FEATURE_COLUMNS)
        self.label_order = list(artifact.get("label_order") or ["H", "D", "A"])
        self.model_version = str(artifact.get("model_version") or selection.model_version)
        self.model_family = str(artifact.get("model_family") or selection.model_family)
        cutoff = pd.to_datetime(target_date)
        self.target_date = cutoff
        self.historical_df = historical_df.loc[historical_df["date"] < cutoff].copy()
        self.poisson_baseline = PoissonMatchPredictor(
            self.historical_df,
            target_date,
            selection=ActiveModelSelection(
                version_key=selection.version_key,
                model_version=self.model_version,
                model_family=self.model_family,
                artifact_path=selection.artifact_path,
                requested_production_status=selection.requested_production_status,
            ),
            poisson_kwargs=poisson_kwargs,
            serving=False,
        )
        self.recent_state = _build_recent_form_state(self.historical_df)
        _, self.klement_state = build_klement_feature_state(self.historical_df, database_url=database_url)

    def _feature_frame(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool,
        is_knockout: bool,
        tournament_context: dict[str, Any] | None,
    ) -> pd.DataFrame:
        home_team = canonical_team_name(home_team)
        away_team = canonical_team_name(away_team)
        feature_row = build_match_feature_payload(
            home_team=home_team,
            away_team=away_team,
            is_neutral=is_neutral,
            is_knockout=is_knockout,
            recent_state=self.recent_state,
            klement_state=self.klement_state,
            reference_date=self.target_date,
            tournament_context=tournament_context,
        )
        return pd.DataFrame([{column: feature_row.get(column, 0.0) for column in self.feature_columns}])

    def get_team_profile(self, team: str) -> dict[str, float]:
        return self.poisson_baseline.get_team_profile(canonical_team_name(team))

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        is_neutral: bool = False,
        is_knockout: bool = False,
        max_goals: int = DEFAULT_MAX_GOALS,
        tournament_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        home_team = canonical_team_name(home_team)
        away_team = canonical_team_name(away_team)
        baseline = self.poisson_baseline.predict_match(
            home_team,
            away_team,
            is_neutral=is_neutral,
            is_knockout=is_knockout,
            max_goals=max_goals,
            tournament_context=tournament_context,
        )
        baseline["model_version"] = self.model_version
        baseline["model_family"] = self.model_family
        baseline["requested_model_version"] = self.requested_model_version
        baseline["requested_model_family"] = self.requested_model_family
        baseline["used_fallback"] = self.used_fallback
        baseline["fallback_reason"] = self.fallback_reason
        if not baseline["prediction_supported"]:
            return baseline

        feature_df = self._feature_frame(home_team, away_team, is_neutral, is_knockout, tournament_context)
        if self.artifact_schema_version >= 2 and self.models:
            lambda_h = max(float(self.models["home_goals"].predict(feature_df)[0]), 0.05)
            lambda_a = max(float(self.models["away_goals"].predict(feature_df)[0]), 0.05)
            prob_matrix = build_probability_matrix(lambda_h, lambda_a, max_goals=max_goals, rho=None)
            prob_matrix, raw_1x2 = self._calibrate_matrix(prob_matrix)
            recommendations = optimize_prediction(prob_matrix, is_knockout=is_knockout).head(5).copy()
            recommendations["EV"] = recommendations["EV"].map(lambda value: round(float(value), 3))
            score_decision = exact_scoreline_decision(prob_matrix, recommendations, is_knockout=is_knockout)
            result_summary = _numeric_result_summary(prob_matrix, home_team, away_team)
            result_probabilities = {
                row["Mercado"]: float(row["Probabilidad"])
                for row in result_summary.to_dict(orient="records")
            }
            market_rows = result_summary.to_dict(orient="records")
            baseline["lambda_h"] = float(lambda_h)
            baseline["lambda_a"] = float(lambda_a)
            baseline["prob_matrix"] = prob_matrix
            baseline["prob_home"] = result_probabilities.get(f"Victoria {home_team}")
            baseline["prob_draw"] = result_probabilities.get("Empate")
            baseline["prob_away"] = result_probabilities.get(f"Victoria {away_team}")
            baseline.update(_raw_probability_fields(raw_1x2))
            baseline["result_summary"] = result_summary
            baseline["recommendations"] = recommendations
            baseline["primary_score"] = score_decision["primary_score"]
            baseline["primary_score_probability"] = score_decision["primary_score_probability"]
            baseline["primary_outcome"] = score_decision["primary_outcome"]
            baseline["most_likely_score"] = score_decision["most_likely_score"]
            baseline["most_likely_score_probability"] = score_decision["most_likely_score_probability"]
            baseline["primary_ev"] = score_decision["primary_ev"]
            baseline["ev_optimal_score"] = score_decision["ev_optimal_score"]
            baseline["ev_optimal_probability"] = score_decision["ev_optimal_probability"]
            baseline["ev_optimal_value"] = score_decision["ev_optimal_value"]
            baseline["primary_score_scenarios"] = score_decision["primary_score_scenarios"]
            baseline["scoreline_candidates"] = score_decision["scoreline_candidates"]
            baseline["score_selection_policy"] = "most_likely_outcome_exact_score"
        else:
            probabilities = self.model.predict_proba(feature_df)[0]
            probability_map = {label: float(probabilities[idx]) for idx, label in enumerate(self.label_order)}
            probability_map.setdefault("H", 0.0)
            probability_map.setdefault("D", 0.0)
            probability_map.setdefault("A", 0.0)
            raw_1x2 = (probability_map["H"], probability_map["D"], probability_map["A"])
            if not self.calibration.is_neutral:
                probability_map["H"], probability_map["D"], probability_map["A"] = self.calibration.calibrate_triplet(
                    *raw_1x2
                )
            market_rows = [
                {"Mercado": f"Victoria {home_team}", "Probabilidad": probability_map["H"]},
                {"Mercado": "Empate", "Probabilidad": probability_map["D"]},
                {"Mercado": f"Victoria {away_team}", "Probabilidad": probability_map["A"]},
            ]
            baseline["prob_home"] = probability_map["H"]
            baseline["prob_draw"] = probability_map["D"]
            baseline["prob_away"] = probability_map["A"]
            baseline.update(_raw_probability_fields(raw_1x2))
            baseline["result_summary"] = pd.DataFrame(market_rows)

        best_market = max(market_rows, key=lambda row: float(row["Probabilidad"]))
        baseline["best_market"] = str(best_market["Mercado"])
        baseline["best_market_probability"] = float(best_market["Probabilidad"])
        baseline["artifact_schema_version"] = self.artifact_schema_version
        baseline["xgboost_feature_payload"] = feature_df.iloc[0].to_dict()
        return baseline


class BivariatePoissonMatchPredictor(BaseMatchPredictor):
    """Miembro del comite basado en Bivariate Poisson con inflacion diagonal.

    Reusa las lambdas marginales del PoissonMatchPredictor (misma senal de entrada)
    y reemplaza la matriz de marcadores por la del DI-BP, que modela la correlacion
    nativamente. Si no hay artefacto de parametros, el engine es neutro (Poisson
    independiente) y el miembro sigue aportando una matriz valida.
    """

    def __init__(
        self,
        historical_df: pd.DataFrame,
        target_date: str,
        *,
        selection: ActiveModelSelection,
        poisson_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(selection=selection)
        self.model_family = "bivariate_poisson"
        self.poisson_baseline = PoissonMatchPredictor(
            historical_df, target_date, selection=selection, poisson_kwargs=poisson_kwargs, serving=False
        )
        params = load_active_bivariate_params()
        self.engine = BivariatePoissonEngine(
            lambda3=params.lambda3, diagonal_weight=params.diagonal_weight
        )

    def get_team_profile(self, team: str) -> dict[str, float]:
        return self.poisson_baseline.get_team_profile(team)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        is_neutral: bool = False,
        is_knockout: bool = False,
        max_goals: int = DEFAULT_MAX_GOALS,
        tournament_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        home_team = canonical_team_name(home_team)
        away_team = canonical_team_name(away_team)
        baseline = self.poisson_baseline.predict_match(
            home_team,
            away_team,
            is_neutral=is_neutral,
            is_knockout=is_knockout,
            max_goals=max_goals,
            tournament_context=tournament_context,
        )
        baseline["model_version"] = self.model_version
        baseline["model_family"] = self.model_family
        if not baseline["prediction_supported"]:
            return baseline

        lambda_h = float(baseline["lambda_h"])
        lambda_a = float(baseline["lambda_a"])
        prob_matrix = self.engine.build_matrix(lambda_h, lambda_a, max_goals=max_goals)
        prob_matrix, raw_1x2 = self._calibrate_matrix(prob_matrix)
        recommendations = optimize_prediction(prob_matrix, is_knockout=is_knockout).head(5).copy()
        recommendations["EV"] = recommendations["EV"].map(lambda value: round(float(value), 3))
        score_decision = exact_scoreline_decision(prob_matrix, recommendations, is_knockout=is_knockout)
        result_summary = _numeric_result_summary(prob_matrix, home_team, away_team)
        result_probabilities = {
            row["Mercado"]: float(row["Probabilidad"])
            for row in result_summary.to_dict(orient="records")
        }
        best_market = max(result_summary.to_dict(orient="records"), key=lambda row: float(row["Probabilidad"]))

        baseline["prob_matrix"] = prob_matrix
        baseline["prob_home"] = result_probabilities.get(f"Victoria {home_team}")
        baseline["prob_draw"] = result_probabilities.get("Empate")
        baseline["prob_away"] = result_probabilities.get(f"Victoria {away_team}")
        baseline.update(_raw_probability_fields(raw_1x2))
        baseline["result_summary"] = result_summary
        baseline["recommendations"] = recommendations
        baseline["primary_score"] = score_decision["primary_score"]
        baseline["primary_score_probability"] = score_decision["primary_score_probability"]
        baseline["primary_outcome"] = score_decision["primary_outcome"]
        baseline["most_likely_score"] = score_decision["most_likely_score"]
        baseline["most_likely_score_probability"] = score_decision["most_likely_score_probability"]
        baseline["primary_ev"] = score_decision["primary_ev"]
        baseline["ev_optimal_score"] = score_decision["ev_optimal_score"]
        baseline["ev_optimal_probability"] = score_decision["ev_optimal_probability"]
        baseline["ev_optimal_value"] = score_decision["ev_optimal_value"]
        baseline["primary_score_scenarios"] = score_decision["primary_score_scenarios"]
        baseline["scoreline_candidates"] = score_decision["scoreline_candidates"]
        baseline["score_selection_policy"] = "most_likely_outcome_exact_score"
        baseline["best_market"] = str(best_market.get("Mercado") or "Por definir")
        baseline["best_market_probability"] = float(best_market.get("Probabilidad") or 0.0)
        return baseline


class HierarchicalMatchPredictor(BaseMatchPredictor):
    """Predictor de fuerzas jerarquicas bayesianas (Baio-Blangiardo).

    Carga el artefacto compacto de medias posteriores (sin dependencia de PyMC en
    inferencia) y construye la matriz de marcadores. Si el artefacto no existe el
    engine no esta listo y la prediccion se marca como NO soportada, para que el
    comite lo omita en vez de promediar ruido neutro.
    """

    def __init__(
        self,
        *,
        selection: ActiveModelSelection,
        strengths_path: str | None = None,
    ) -> None:
        super().__init__(selection=selection)
        self.model_family = "hierarchical_bayes"
        self.engine = HierarchicalBayesEngine(load_active_hierarchical_strengths(strengths_path))

    def get_team_profile(self, team: str) -> dict[str, float]:
        strength = self.engine.strengths.strength_for(team)
        return {"attack": strength.attack, "defense": strength.defense, "matches": float(self.engine.strengths.samples)}

    def _unsupported(self, message: str) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "model_family": self.model_family,
            "requested_model_version": self.requested_model_version,
            "requested_model_family": self.requested_model_family,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "prediction_supported": False,
            "support_message": message,
            "support_label": "Sin historial suficiente",
            "lambda_h": None,
            "lambda_a": None,
            "prob_matrix": None,
            "prob_home": None,
            "prob_draw": None,
            "prob_away": None,
            "result_summary": pd.DataFrame(),
            "recommendations": pd.DataFrame(),
            "primary_score": "N/D",
            "best_market": "Sin soporte",
            "best_market_probability": 0.0,
        }

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        is_neutral: bool = False,
        is_knockout: bool = False,
        max_goals: int = DEFAULT_MAX_GOALS,
        tournament_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        home_team = canonical_team_name(home_team)
        away_team = canonical_team_name(away_team)
        if not self.engine.is_ready:
            return self._unsupported("Modelo jerarquico sin artefacto activo.")

        lambda_h, lambda_a = self.engine.expected_goals(home_team, away_team, is_neutral=is_neutral)
        prob_matrix = self.engine.build_matrix(home_team, away_team, is_neutral=is_neutral, max_goals=max_goals)
        prob_matrix, raw_1x2 = self._calibrate_matrix(prob_matrix)
        recommendations = optimize_prediction(prob_matrix, is_knockout=is_knockout).head(5).copy()
        recommendations["EV"] = recommendations["EV"].map(lambda value: round(float(value), 3))
        score_decision = exact_scoreline_decision(prob_matrix, recommendations, is_knockout=is_knockout)
        result_summary = _numeric_result_summary(prob_matrix, home_team, away_team)
        result_probabilities = {
            row["Mercado"]: float(row["Probabilidad"])
            for row in result_summary.to_dict(orient="records")
        }
        best_market = max(result_summary.to_dict(orient="records"), key=lambda row: float(row["Probabilidad"]))

        return {
            "model_version": self.model_version,
            "model_family": self.model_family,
            "requested_model_version": self.requested_model_version,
            "requested_model_family": self.requested_model_family,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "prediction_supported": True,
            "support_message": None,
            "support_label": "Modelo jerarquico",
            "lambda_h": float(lambda_h),
            "lambda_a": float(lambda_a),
            "prob_matrix": prob_matrix,
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
            "score_selection_policy": "most_likely_outcome_exact_score",
            "best_market": str(best_market.get("Mercado") or "Por definir"),
            "best_market_probability": float(best_market.get("Probabilidad") or 0.0),
        }


def _selection_from_record(record: dict[str, Any] | None) -> ActiveModelSelection:
    if not record:
        return ActiveModelSelection(
            version_key=None,
            model_version=DEFAULT_POISSON_MODEL_VERSION,
            model_family=DEFAULT_POISSON_MODEL_FAMILY,
            artifact_path=None,
            requested_production_status="active",
        )
    return ActiveModelSelection(
        version_key=record.get("version_key"),
        model_version=str(record.get("model_version") or DEFAULT_POISSON_MODEL_VERSION),
        model_family=str(record.get("model_family") or DEFAULT_POISSON_MODEL_FAMILY),
        artifact_path=record.get("artifact_path"),
        requested_production_status=str(record.get("production_status") or "active"),
    )


@dataclass(frozen=True, slots=True)
class ActiveRecordLookup:
    """Resultado de consultar el registro de modelos: el registro, o el error que impidio leerlo."""

    record: dict[str, Any] | None
    error: str | None = None


def _load_active_model_record_safe(*, database_url: str | None = None) -> ActiveRecordLookup:
    # La inferencia sigue arrancando sin registro, pero el fallo es visible: se loguea y viaja en `error`.
    try:
        from alttez.operations.prediction_memory import fetch_active_model_record

        return ActiveRecordLookup(record=fetch_active_model_record(database_url=database_url))
    except Exception as exc:  # noqa: BLE001 - registry stack may be unavailable; reported, not hidden.
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Model registry unavailable, serving Poisson fallback: %s", error)
        return ActiveRecordLookup(record=None, error=error)


def build_active_predictor(
    historical_df: pd.DataFrame,
    target_date: date | str,
    *,
    database_url: str | None = None,
    poisson_kwargs: dict[str, Any] | None = None,
) -> BaseMatchPredictor:
    if isinstance(target_date, date):
        target_date_str = target_date.isoformat()
    else:
        target_date_str = str(target_date)

    lookup = _load_active_model_record_safe(database_url=database_url)
    selection = _selection_from_record(lookup.record)
    family = selection.model_family.strip().lower()

    if family == "xgboost":
        try:
            return XGBoostMatchPredictor(
                historical_df,
                target_date_str,
                selection=selection,
                poisson_kwargs=poisson_kwargs,
                database_url=database_url,
            )
        except Exception as exc:  # noqa: BLE001 - production fallback should keep the pipeline alive.
            reason = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "Active %s model %s failed to load, serving Poisson fallback: %s",
                selection.model_family,
                selection.model_version,
                reason,
            )
            return PoissonMatchPredictor(
                historical_df,
                target_date_str,
                selection=selection,
                poisson_kwargs=poisson_kwargs,
                used_fallback=True,
                fallback_reason=reason,
            )

    registry_failed = lookup.error is not None
    return PoissonMatchPredictor(
        historical_df,
        target_date_str,
        selection=selection,
        poisson_kwargs=poisson_kwargs,
        used_fallback=registry_failed,
        fallback_reason=f"model registry unavailable: {lookup.error}" if registry_failed else None,
    )
