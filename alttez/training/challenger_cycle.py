from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBRegressor

from alttez.api.db import get_engine
from alttez.evaluation.metrics import ranked_probability_score
from alttez.evaluation.regulation_points import (
    max_regulation_points,
    points_optimal_prediction,
    regulation_points,
    summarize_regulation_points,
)
from alttez.experiments.tournament_oracle import DEFAULT_DIXON_COLES_RHO, build_probability_matrix
from alttez.features.klement import (
    MODEL_FEATURE_COLUMNS,
    build_klement_feature_state,
    load_enriched_historical_matches,
)
from alttez.features.poisson import PoissonEngine
from alttez.features.selection import snap_split_index
from alttez.operations.prediction_memory import (
    auto_promotion_enabled,
    brier_score_1x2,
    get_active_model_family,
    get_active_model_version,
    load_active_model_record,
    production_supported_model_families,
    reflect_prediction_tables,
    set_active_model_version_record,
    upsert_model_version_record,
)


logging.basicConfig(level=logging.INFO)


DEFAULT_REPORT_DIR = Path("artifacts/model-evals")
DEFAULT_CHALLENGER_VERSION = "xgboost-challenger-v2"
DEFAULT_EVALUATION_MATCHES = 250
DEFAULT_PROMOTION_LOG_LOSS_TOLERANCE = 0.01
DEFAULT_INCUMBENT_REGRESSION_EPSILON = 1e-9
# Tolerancia para el RPS: el retador debe igualar o mejorar el RPS del champion
# (mismo espiritu que el Brier, sin exigir margen arbitrario). Empeorar el RPS por
# encima de este epsilon lo descalifica aunque el Brier nominal mejore.
DEFAULT_PROMOTION_RPS_TOLERANCE = 1e-9
# Umbral de significancia para la prueba de Diebold-Mariano. Solo aplica en la ruta
# de auto-promocion: la diferencia de perdida debe ser estadisticamente significativa.
DEFAULT_DM_SIGNIFICANCE_LEVEL = 0.05
DEFAULT_CHAMPION_REFIT_INTERVAL = 25
DEFAULT_MAX_GOALS = 7
# Single source of truth lives in alttez.features.klement; aliased for compatibility.
FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS


def _outcome_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _recent_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_recent_form_features(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    team_points: dict[str, list[float]] = defaultdict(list)
    team_goal_diff: dict[str, list[float]] = defaultdict(list)

    home_points = []
    away_points = []
    home_goal_diff = []
    away_goal_diff = []

    for row in df.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        home_points.append(_recent_average(team_points[home_team][-window:]))
        away_points.append(_recent_average(team_points[away_team][-window:]))
        home_goal_diff.append(_recent_average(team_goal_diff[home_team][-window:]))
        away_goal_diff.append(_recent_average(team_goal_diff[away_team][-window:]))

        home_points_gained = 3.0 if row.home_score > row.away_score else 1.0 if row.home_score == row.away_score else 0.0
        away_points_gained = 3.0 if row.away_score > row.home_score else 1.0 if row.home_score == row.away_score else 0.0
        goal_diff = float(row.home_score - row.away_score)

        team_points[home_team].append(home_points_gained)
        team_points[away_team].append(away_points_gained)
        team_goal_diff[home_team].append(goal_diff)
        team_goal_diff[away_team].append(-goal_diff)

    df["home_recent_points"] = home_points
    df["away_recent_points"] = away_points
    df["home_recent_goal_diff"] = home_goal_diff
    df["away_recent_goal_diff"] = away_goal_diff
    return df


def _build_challenger_dataset(df: pd.DataFrame, *, database_url: str | None = None) -> pd.DataFrame:
    feature_df, _ = build_klement_feature_state(df.copy(), database_url=database_url)
    feature_df["is_neutral"] = feature_df["neutral"].astype(int)
    feature_df = _build_recent_form_features(feature_df)
    feature_df["outcome_label"] = feature_df.apply(
        lambda row: _outcome_label(int(row["home_score"]), int(row["away_score"])),
        axis=1,
    )
    return feature_df


def _score_prediction_row(
    *,
    prob_matrix: np.ndarray,
    actual_home_score: int,
    actual_away_score: int,
    is_knockout: bool = False,
) -> dict[str, Any]:
    prob_home = float(np.tril(prob_matrix, -1).sum())
    prob_draw = float(prob_matrix.diagonal().sum())
    prob_away = float(np.triu(prob_matrix, 1).sum())
    predicted_outcome = max(
        (("H", prob_home), ("D", prob_draw), ("A", prob_away)),
        key=lambda item: item[1],
    )[0]
    actual_outcome = _outcome_label(actual_home_score, actual_away_score)
    max_score = divmod(int(prob_matrix.argmax()), prob_matrix.shape[1])
    regulation_pick = points_optimal_prediction(prob_matrix, is_knockout=is_knockout)
    regulation_home = int(regulation_pick["home_goals"])
    regulation_away = int(regulation_pick["away_goals"])
    earned_points = regulation_points(
        regulation_home,
        regulation_away,
        actual_home_score,
        actual_away_score,
        is_knockout=is_knockout,
    )
    return {
        "predicted_outcome": predicted_outcome,
        "actual_outcome": actual_outcome,
        "brier": brier_score_1x2(prob_home, prob_draw, prob_away, actual_outcome),
        # RPS respeta la ordinalidad del 1X2 (H->D->A); complementa el Brier nominal
        # y alimenta tanto la metrica agregada como la guardia de promocion.
        "rps": ranked_probability_score(prob_home, prob_draw, prob_away, actual_outcome),
        "prob_home": prob_home,
        "prob_draw": prob_draw,
        "prob_away": prob_away,
        "exact_hit": int(max_score[0] == actual_home_score and max_score[1] == actual_away_score),
        "regulation_scoreline": regulation_pick["scoreline"],
        "regulation_expected_points": regulation_pick["ev"],
        "regulation_exact_probability": regulation_pick["exact_probability"],
        "regulation_points": earned_points,
        "regulation_max_points": max_regulation_points(is_knockout=is_knockout),
    }


def expected_calibration_error(eval_df: pd.DataFrame, *, n_bins: int = 10) -> float:
    """Expected Calibration Error sobre la confianza del 1X2 (max prob por fila).

    Mide la brecha entre confianza promedio y acierto real por bin; 0 = perfecta
    calibracion. Complementa Brier/LogLoss exponiendo el sesgo sistematico de
    confianza del modelo.
    """

    if eval_df.empty:
        return 0.0

    confidence = eval_df[["prob_home", "prob_draw", "prob_away"]].to_numpy().max(axis=1)
    correct = (eval_df["predicted_outcome"] == eval_df["actual_outcome"]).to_numpy().astype(float)
    total = len(eval_df)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        bin_size = int(in_bin.sum())
        if bin_size == 0:
            continue
        bin_confidence = float(confidence[in_bin].mean())
        bin_accuracy = float(correct[in_bin].mean())
        ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _summarize_probability_rows(
    rows: list[dict[str, Any]],
    *,
    model_version: str,
    model_family: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_df = pd.DataFrame(rows)
    probability_matrix = eval_df[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    y_true = eval_df["actual_outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    summary = {
        "model_version": model_version,
        "model_family": model_family,
        "brier_score": float(eval_df["brier"].mean()),
        "rps": float(eval_df["rps"].mean()),
        "log_loss": float(log_loss(y_true, probability_matrix, labels=[0, 1, 2])),
        "accuracy_1x2": float((eval_df["predicted_outcome"] == eval_df["actual_outcome"]).mean()),
        "exact_score_hit_rate": float(eval_df["exact_hit"].mean()),
        "expected_calibration_error": expected_calibration_error(eval_df),
        # Serie de perdida Brier por partido (orden del holdout) para la prueba de
        # Diebold-Mariano retador-vs-incumbente. Es metadata interna, no una metrica.
        "brier_losses": [float(value) for value in eval_df["brier"].tolist()],
        **summarize_regulation_points(rows),
    }
    if extra:
        summary.update(extra)
    return summary


def _append_history_row(history_df: pd.DataFrame, row: Any) -> pd.DataFrame:
    next_row = pd.DataFrame([row._asdict()])
    return pd.concat([history_df, next_row], ignore_index=True)


def _champion_lambda_stream(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    champion_refit_interval: int = DEFAULT_CHAMPION_REFIT_INTERVAL,
) -> tuple[list[tuple[float, float]], int]:
    """Per-row Poisson champion (lambda_home, lambda_away) with block refits.

    Returns the lambda stream (ordered by date) and the number of refits, so champion
    evaluation and the ensemble blend share one causal source of champion lambdas.
    """

    history_df = train_df.sort_values("date").reset_index(drop=True).copy()
    ordered_holdout = holdout_df.sort_values("date").reset_index(drop=True)
    lambdas: list[tuple[float, float]] = []
    refit_count = 0
    engine: PoissonEngine | None = None

    for index, row in enumerate(ordered_holdout.itertuples(index=False)):
        if engine is None or index % champion_refit_interval == 0:
            cutoff = pd.to_datetime(row.date).date().isoformat()
            engine = PoissonEngine(
                years_window=4,
                prior_matches=8,
                min_history_matches=5,
                recency_half_life_days=365,
                recent_form_matches=6,
            )
            engine.fit(history_df, target_date=cutoff)
            refit_count += 1

        lambda_home, lambda_away = engine.predict_xg(row.home_team, row.away_team, is_neutral=bool(row.neutral))
        lambdas.append((float(lambda_home), float(lambda_away)))
        history_df = _append_history_row(history_df, row)

    return lambdas, refit_count


def _xgb_lambda_stream(artifact_bundle: dict[str, Any], holdout_df: pd.DataFrame) -> list[tuple[float, float]]:
    """Per-row XGBoost challenger (lambda_home, lambda_away) from the dual goal models."""

    ordered_holdout = holdout_df.sort_values("date").reset_index(drop=True)
    feature_columns = list(artifact_bundle.get("feature_columns") or FEATURE_COLUMNS)
    x_holdout = ordered_holdout[feature_columns].fillna(0.0)
    models = artifact_bundle["models"]
    lambdas_home = np.maximum(np.asarray(models["home_goals"].predict(x_holdout), dtype=float), 0.05)
    lambdas_away = np.maximum(np.asarray(models["away_goals"].predict(x_holdout), dtype=float), 0.05)
    return [(float(h), float(a)) for h, a in zip(lambdas_home, lambdas_away)]


def _score_lambda_stream(
    lambdas: list[tuple[float, float]],
    holdout_df: pd.DataFrame,
    *,
    model_version: str,
    model_family: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_holdout = holdout_df.sort_values("date").reset_index(drop=True)
    rows = []
    for (lambda_home, lambda_away), row in zip(lambdas, ordered_holdout.itertuples(index=False)):
        prob_matrix = build_probability_matrix(
            lambda_home, lambda_away, max_goals=DEFAULT_MAX_GOALS, rho=DEFAULT_DIXON_COLES_RHO
        )
        rows.append(
            _score_prediction_row(
                prob_matrix=prob_matrix,
                actual_home_score=int(row.home_score),
                actual_away_score=int(row.away_score),
                is_knockout=bool(getattr(row, "is_knockout", False)),
            )
        )
    return _summarize_probability_rows(rows, model_version=model_version, model_family=model_family, extra=extra)


def _evaluate_poisson_champion(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    champion_refit_interval: int = DEFAULT_CHAMPION_REFIT_INTERVAL,
) -> dict[str, Any]:
    lambdas, refit_count = _champion_lambda_stream(
        train_df, holdout_df, champion_refit_interval=champion_refit_interval
    )
    return _score_lambda_stream(
        lambdas,
        holdout_df,
        model_version=get_active_model_version(),
        model_family=get_active_model_family(),
        extra={"refit_count": refit_count, "champion_refit_interval": champion_refit_interval},
    )


def evaluate_lambda_blend(
    champion_lambdas: list[tuple[float, float]],
    challenger_lambdas: list[tuple[float, float]],
    holdout_df: pd.DataFrame,
    *,
    weight_grid: list[float] | None = None,
) -> dict[str, Any]:
    """Grid-search the blend weight w for lambda = w*challenger + (1-w)*champion.

    Picks the weight minimising Brier on the holdout and returns its metrics. This
    measures whether *combining* the goal models beats either alone (the core of the
    ALTTEZ ensemble) before any production wiring.
    """

    if weight_grid is None:
        weight_grid = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0

    best: dict[str, Any] | None = None
    for weight in weight_grid:
        blended = [
            (weight * ch_h + (1.0 - weight) * cp_h, weight * ch_a + (1.0 - weight) * cp_a)
            for (cp_h, cp_a), (ch_h, ch_a) in zip(champion_lambdas, challenger_lambdas)
        ]
        summary = _score_lambda_stream(
            blended,
            holdout_df,
            model_version="alttez-ensemble-v1",
            model_family="ensemble",
            extra={"blend_weight": weight},
        )
        if best is None or summary["brier_score"] < best["brier_score"]:
            best = summary
    return best or {}


DEFAULT_MIN_EARLY_STOPPING_SAMPLES = 50

# Recencia para el ponderado del entrenamiento XGBoost. Default 365d para paridad
# con PoissonEngine.recency_half_life_days (features/poisson.py): el campeon y el
# retador decaen el pasado al mismo ritmo en vez de tratar todo el historico igual.
DEFAULT_XGB_TIME_DECAY_HALFLIFE_DAYS = 365
XGB_TIME_DECAY_HALFLIFE_ENV = "ALTTEZ_XGB_TIME_DECAY_HALFLIFE_DAYS"


def _resolve_time_decay_half_life() -> float:
    """Half-life en dias para el ponderado por recencia. Env override con fallback.

    Valores no numericos o <= 0 caen al default (paridad con Poisson).
    """
    raw = os.getenv(XGB_TIME_DECAY_HALFLIFE_ENV)
    if raw is None:
        return float(DEFAULT_XGB_TIME_DECAY_HALFLIFE_DAYS)
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_XGB_TIME_DECAY_HALFLIFE_DAYS)
    return value if value > 0 else float(DEFAULT_XGB_TIME_DECAY_HALFLIFE_DAYS)


def _compute_time_decay_weights(
    dates: pd.Series, reference_date: Any, half_life_days: float
) -> np.ndarray:
    """Pesos exponenciales por recencia: w = 0.5^(dias_atras / half_life).

    Espeja la formula de PoissonEngine._compute_match_weights. El partido mas
    reciente del nucleo de entrenamiento pesa ~1.0; los antiguos decaen. La
    referencia es el corte de entrenamiento (max fecha del nucleo), nunca una
    fecha futura -> sin fuga temporal.
    """
    if half_life_days <= 0:
        return np.ones(len(dates), dtype=float)
    parsed = pd.to_datetime(dates)
    reference = pd.to_datetime(reference_date)
    days_ago = (reference - parsed).dt.days.clip(lower=0).to_numpy(dtype=float)
    return np.power(0.5, days_ago / float(half_life_days))


def _split_training_and_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Carve a strictly disjoint, date-aligned validation tail for early stopping.

    Returns ``(core_train, validation_df)`` when there is enough data to split on a
    clean date boundary, or ``(ordered, None)`` when the dataset is too small to
    carve a disjoint validation set (in which case the caller must train without
    early stopping rather than validating on data the model has already seen).
    """

    ordered = train_df.sort_values("date").reset_index(drop=True)
    if len(ordered) < DEFAULT_MIN_EARLY_STOPPING_SAMPLES:
        return ordered, None

    validation_size = max(10, len(ordered) // 10)
    validation_size = min(validation_size, len(ordered) - 10)
    # Snap to a date boundary so the validation set never shares a match-day with
    # the training core (avoids intra-day leakage into eval_set).
    split_index = snap_split_index(ordered, len(ordered) - validation_size, "date")
    if split_index <= 0 or split_index >= len(ordered):
        return ordered, None

    core_train = ordered.iloc[:split_index].copy()
    validation_df = ordered.iloc[split_index:].copy()
    if core_train.empty or validation_df.empty:
        return ordered, None
    return core_train, validation_df


def _build_goal_regressor(use_early_stopping: bool = True) -> XGBRegressor:
    params: dict[str, Any] = dict(
        objective="count:poisson",
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="poisson-nloglik",
        random_state=0,
        n_jobs=-1,
    )
    if use_early_stopping:
        # XGBoost raises if early_stopping_rounds is set without an eval_set, so the
        # flag is only added when a disjoint validation set is available.
        params["early_stopping_rounds"] = 40
    return XGBRegressor(**params)


def _fit_goal_models(train_df: pd.DataFrame) -> dict[str, Any]:
    train_core, validation_df = _split_training_and_validation(train_df)
    use_early_stopping = validation_df is not None
    x_train = train_core[FEATURE_COLUMNS].fillna(0.0)

    # Ponderado por recencia: el corte (max fecha del nucleo) pesa ~1.0; el
    # pasado decae con half-life. sample_weight se alinea por posicion con x_train
    # (mismo orden de filas que train_core). El eval_set queda sin ponderar.
    half_life_days = _resolve_time_decay_half_life()
    reference_date = train_core["date"].max()
    sample_weight = _compute_time_decay_weights(train_core["date"], reference_date, half_life_days)

    home_model = _build_goal_regressor(use_early_stopping=use_early_stopping)
    away_model = _build_goal_regressor(use_early_stopping=use_early_stopping)

    if use_early_stopping:
        x_validation = validation_df[FEATURE_COLUMNS].fillna(0.0)
        home_model.fit(
            x_train,
            train_core["home_score"],
            sample_weight=sample_weight,
            eval_set=[(x_validation, validation_df["home_score"])],
            verbose=False,
        )
        away_model.fit(
            x_train,
            train_core["away_score"],
            sample_weight=sample_weight,
            eval_set=[(x_validation, validation_df["away_score"])],
            verbose=False,
        )
    else:
        home_model.fit(x_train, train_core["home_score"], sample_weight=sample_weight, verbose=False)
        away_model.fit(x_train, train_core["away_score"], sample_weight=sample_weight, verbose=False)

    # Average gain importance across both goal models, ranked, for auditability.
    home_imp = np.asarray(getattr(home_model, "feature_importances_", np.zeros(len(FEATURE_COLUMNS))), dtype=float)
    away_imp = np.asarray(getattr(away_model, "feature_importances_", np.zeros(len(FEATURE_COLUMNS))), dtype=float)
    mean_imp = (home_imp + away_imp) / 2.0
    feature_importance = dict(
        sorted(
            ((col, float(val)) for col, val in zip(FEATURE_COLUMNS, mean_imp)),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    return {
        "home_goals": home_model,
        "away_goals": away_model,
        "best_iteration": {
            "home_goals": int(getattr(home_model, "best_iteration", home_model.n_estimators - 1)),
            "away_goals": int(getattr(away_model, "best_iteration", away_model.n_estimators - 1)),
        },
        "validation_samples": int(len(validation_df)) if validation_df is not None else 0,
        "train_samples": int(len(train_core)),
        "time_decay_half_life_days": float(half_life_days),
        "feature_importance": feature_importance,
    }


def _train_challenger(train_df: pd.DataFrame) -> dict[str, Any]:
    fitted_models = _fit_goal_models(train_df)
    artifact_bundle = {
        "artifact_schema_version": 2,
        "models": {
            "home_goals": fitted_models["home_goals"],
            "away_goals": fitted_models["away_goals"],
        },
        "feature_columns": FEATURE_COLUMNS,
        "model_version": DEFAULT_CHALLENGER_VERSION,
        "model_family": "xgboost",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_iteration": fitted_models["best_iteration"],
        "train_samples": fitted_models["train_samples"],
        "validation_samples": fitted_models["validation_samples"],
        "feature_importance": fitted_models.get("feature_importance", {}),
    }
    return artifact_bundle


def _evaluate_challenger(
    artifact_bundle: dict[str, Any],
    holdout_df: pd.DataFrame,
) -> dict[str, Any]:
    # Holdout features in challenger_df are already causal (each row's elo_*_before,
    # recent form and knockout weight only depend on earlier matches), so we can score
    # the whole window in a single vectorized pass instead of rebuilding the feature
    # state per match. The lambda values are identical to a per-row rebuild.
    ordered_holdout = holdout_df.sort_values("date").reset_index(drop=True)
    feature_columns = list(artifact_bundle.get("feature_columns") or FEATURE_COLUMNS)
    x_holdout = ordered_holdout[feature_columns].fillna(0.0)

    models = artifact_bundle["models"]
    lambdas_home = np.maximum(np.asarray(models["home_goals"].predict(x_holdout), dtype=float), 0.05)
    lambdas_away = np.maximum(np.asarray(models["away_goals"].predict(x_holdout), dtype=float), 0.05)

    rows = []
    for index, row in enumerate(ordered_holdout.itertuples(index=False)):
        prob_matrix = build_probability_matrix(
            float(lambdas_home[index]),
            float(lambdas_away[index]),
            max_goals=DEFAULT_MAX_GOALS,
            rho=DEFAULT_DIXON_COLES_RHO,
        )
        rows.append(
            _score_prediction_row(
                prob_matrix=prob_matrix,
                actual_home_score=int(row.home_score),
                actual_away_score=int(row.away_score),
                is_knockout=bool(getattr(row, "is_knockout", False)),
            )
        )

    return _summarize_probability_rows(
        rows,
        model_version=artifact_bundle["model_version"],
        model_family=artifact_bundle["model_family"],
        extra={"best_iteration": artifact_bundle.get("best_iteration")},
    )


def _incumbent_brier(active_record: dict[str, Any] | None) -> float | None:
    """Brier of the currently-active production model, if recoverable."""

    if not active_record:
        return None
    payload = active_record.get("metrics_payload")
    if isinstance(payload, dict) and payload.get("brier_score") is not None:
        try:
            return float(payload["brier_score"])
        except (TypeError, ValueError):
            return None
    return None


def diebold_mariano(loss_a: list[float], loss_b: list[float]) -> tuple[float, float]:
    """Prueba de Diebold-Mariano sobre dos series de perdida por partido (menor es mejor).

    Compara la precision predictiva de dos modelos a partir de su perdida por partido
    (p.ej. Brier por partido del retador vs el incumbente). La diferencia ``d = loss_a -
    loss_b`` se promedia y se estandariza por su error estandar. Un ``dm_stat`` negativo
    indica que ``loss_a`` (modelo A) tiene menor perdida -> A es mejor.

    Aproximacion: se usa la varianza muestral simple (sin correccion HAC de Newey-West
    para autocorrelacion ni el ajuste de muestra chica de Harvey-Leybourne-Newbold) y un
    p-valor bilateral por aproximacion normal sobre ``dm_stat``. Es adecuado para perdidas
    por partido razonablemente independientes; documentado como aproximacion.

    Casos degenerados (longitudes distintas o < 2, o varianza ~0 de la diferencia) NO
    levantan excepcion: devuelven ``(0.0, 1.0)`` (no significativo).
    """

    series_a = np.asarray(list(loss_a), dtype=float)
    series_b = np.asarray(list(loss_b), dtype=float)
    n = series_a.shape[0]
    if n < 2 or series_b.shape[0] != n:
        return 0.0, 1.0

    diff = series_a - series_b
    mean_diff = float(diff.mean())
    # Varianza muestral (ddof=1) de la diferencia; error estandar de la media.
    variance = float(diff.var(ddof=1))
    if not np.isfinite(variance) or variance <= 1e-15:
        # Series identicas o diferencia constante: sin evidencia de superioridad.
        return 0.0, 1.0

    standard_error = (variance / n) ** 0.5
    if standard_error <= 0.0:
        return 0.0, 1.0

    dm_stat = mean_diff / standard_error
    # p-valor bilateral por aproximacion normal: 2 * (1 - Phi(|z|)).
    # math.erf evita una dependencia de scipy; Phi(z) = 0.5*(1+erf(z/sqrt(2))).
    import math

    cdf = 0.5 * (1.0 + math.erf(abs(dm_stat) / math.sqrt(2.0)))
    p_value = 2.0 * (1.0 - cdf)
    p_value = float(min(max(p_value, 0.0), 1.0))
    return float(dm_stat), p_value


def _promotion_decision(
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    *,
    incumbent_brier: float | None = None,
    challenger_brier_losses: list[float] | None = None,
    incumbent_brier_losses: list[float] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Decide elegibilidad del retador y devuelve el detalle de gobernanza.

    Reglas compuestas (orden de evaluacion):
      1. Brier (gate primario): el retador debe mejorar el Brier del champion.
      2. Log Loss (guardrail): no deteriorar el Log Loss materialmente.
      3. RPS (gate adicional ordinal): el retador debe igualar o mejorar el RPS del
         champion. Es un requisito ADICIONAL al Brier, no un reemplazo.
      4. Proteccion del incumbente: no regresar el Brier del modelo activo.

    Devuelve ``(promote, reason, detail)``. ``detail`` expone ``challenger_rps``,
    ``champion_rps`` y ``dm_p_value`` (Diebold-Mariano sobre las perdidas Brier por
    partido retador vs incumbente) para que el reporte/dashboard las muestren. El
    ``dm_p_value`` se calcula aqui pero NO bloquea esta decision: la guardia DM solo
    aplica en la ruta de auto-promocion (ver ``_governance_decision``).
    """

    challenger_rps = challenger_metrics.get("rps")
    champion_rps = champion_metrics.get("rps")
    dm_stat, dm_p_value = diebold_mariano(
        list(challenger_brier_losses or []), list(incumbent_brier_losses or [])
    )
    detail: dict[str, Any] = {
        "challenger_rps": float(challenger_rps) if challenger_rps is not None else None,
        "champion_rps": float(champion_rps) if champion_rps is not None else None,
        "dm_stat": float(dm_stat),
        "dm_p_value": float(dm_p_value),
    }

    if float(challenger_metrics["brier_score"]) >= float(champion_metrics["brier_score"]):
        return False, "Challenger no mejora Brier Score frente al champion actual.", detail
    if float(challenger_metrics["log_loss"]) > float(champion_metrics["log_loss"]) + DEFAULT_PROMOTION_LOG_LOSS_TOLERANCE:
        return False, "Challenger mejora Brier pero deteriora Log Loss materialmente.", detail
    # RPS gate (ordinal): el retador no puede empeorar el RPS aunque gane en Brier.
    # El 1X2 es ordinal; un retador que mejora el Brier nominal pero degrada el RPS
    # esta acertando peor en la *direccion* del resultado y no es elegible.
    if (
        challenger_rps is not None
        and champion_rps is not None
        and float(challenger_rps) > float(champion_rps) + DEFAULT_PROMOTION_RPS_TOLERANCE
    ):
        return False, "Challenger mejora Brier pero deteriora RPS ordinal frente al champion.", detail
    # Incumbent protection: never auto-replace the active model with a worse one.
    # Small wins are still wins; the tolerance above is only for material Log Loss
    # decay, not for demanding an arbitrary Brier margin over the active model.
    if incumbent_brier is not None and float(challenger_metrics["brier_score"]) > incumbent_brier + DEFAULT_INCUMBENT_REGRESSION_EPSILON:
        return False, "Challenger empeora frente al modelo activo (sin regresion permitida).", detail
    return True, "Challenger elegible por Brier y RPS; supera champion e incumbente.", detail


def _governance_decision(
    *,
    promote: bool,
    challenger_family: str,
    dm_p_value: float | None = None,
) -> tuple[str, str, str]:
    if not promote:
        return "shadow", "blocked", "Challenger descartado por politica de metricas."
    if challenger_family.lower() not in production_supported_model_families():
        return "eligible", "blocked", "Elegible por Brier, pero su familia aun no esta habilitada para inferencia productiva."
    if not auto_promotion_enabled():
        return "eligible", "manual", "Elegible por Brier; pendiente de aprobacion explicita para activacion productiva."
    # Auto-promote gate adicional (Diebold-Mariano): solo en esta ruta exigimos que la
    # mejora del retador sobre el incumbente sea estadisticamente significativa (p<0.05).
    # No afecta las rutas manual/shadow: la activacion manual sigue siendo discrecional.
    if dm_p_value is not None and float(dm_p_value) >= DEFAULT_DM_SIGNIFICANCE_LEVEL:
        return (
            "eligible",
            "manual",
            "Elegible por Brier/RPS, pero la mejora no es estadisticamente significativa "
            f"(Diebold-Mariano p={float(dm_p_value):.3f} >= {DEFAULT_DM_SIGNIFICANCE_LEVEL}); requiere aprobacion manual.",
        )
    return "active", "auto", "Promovido automaticamente por mejorar Brier sin deterioro material de Log Loss."


def champion_pointer_status(*, challenger_production_status: str, active_model_exists: bool) -> str:
    """Production-pointer policy for the champion baseline row.

    With auto-promotion OFF the challenger cycle must never move which model serves. The
    champion baseline only claims the ``active`` slot to bootstrap an empty registry; if an
    active model already exists (manual activation or a prior bootstrap) the freshly-trained
    champion is recorded as ``shadow`` so it cannot silently clobber the active pointer on the
    nightly run. When the challenger is auto-promoted, the old champion steps down to ``archived``.
    """

    if challenger_production_status == "active":
        return "archived"
    if not active_model_exists:
        return "active"
    return "shadow"


def _write_report(report: dict[str, Any], report_dir: str | Path = DEFAULT_REPORT_DIR) -> Path:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    output_path = report_path / f"challenger-cycle-{report['target_date']}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output_path


def _write_model_artifact(
    artifact_bundle: dict[str, Any],
    *,
    target_date: date,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> Path:
    artifact_dir = Path(report_dir).parent / "model-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{artifact_bundle['model_version']}-{target_date.isoformat()}.joblib"
    joblib.dump(artifact_bundle, artifact_path)
    return artifact_path


def run_challenger_cycle(
    target_date: date,
    *,
    database_url: str | None = None,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    evaluation_matches: int = DEFAULT_EVALUATION_MATCHES,
    champion_refit_interval: int = DEFAULT_CHAMPION_REFIT_INTERVAL,
) -> dict[str, Any]:
    historical_df = load_enriched_historical_matches(database_url=database_url).sort_values("date").reset_index(drop=True)
    challenger_df = _build_challenger_dataset(historical_df, database_url=database_url)
    challenger_df = challenger_df.sort_values("date").reset_index(drop=True)
    eval_size = min(evaluation_matches, max(60, len(challenger_df) // 20))
    # Snap the split to a date boundary so a single match-day never straddles train/holdout.
    split_index = snap_split_index(challenger_df, len(challenger_df) - eval_size, "date")
    split_index = min(max(split_index, 1), len(challenger_df) - 1)
    train_df = challenger_df.iloc[:split_index].copy()
    holdout_df = challenger_df.iloc[split_index:].copy()

    champion_lambdas, champion_refit_count = _champion_lambda_stream(
        train_df,
        holdout_df,
        champion_refit_interval=champion_refit_interval,
    )
    champion_metrics = _score_lambda_stream(
        champion_lambdas,
        holdout_df,
        model_version=get_active_model_version(),
        model_family=get_active_model_family(),
        extra={"refit_count": champion_refit_count, "champion_refit_interval": champion_refit_interval},
    )
    challenger_artifact = _train_challenger(train_df)
    challenger_metrics = _evaluate_challenger(challenger_artifact, holdout_df)
    challenger_metrics["feature_importance"] = challenger_artifact.get("feature_importance")

    # Ensemble probe: does blending champion + challenger lambdas beat either alone?
    challenger_lambdas = _xgb_lambda_stream(challenger_artifact, holdout_df)
    ensemble_metrics = evaluate_lambda_blend(champion_lambdas, challenger_lambdas, holdout_df)

    active_record = load_active_model_record(database_url=database_url)
    incumbent_brier = _incumbent_brier(active_record)
    # Series de perdida Brier por partido (mismo holdout, mismo orden) para la prueba
    # de Diebold-Mariano. El champion Poisson evaluado en el holdout es el baseline
    # incumbente por-partido comparable (las perdidas por-partido del modelo activo
    # historico no se reconstruyen aqui).
    challenger_brier_losses = list(challenger_metrics.get("brier_losses") or [])
    incumbent_brier_losses = list(champion_metrics.get("brier_losses") or [])
    promote, promotion_reason, promotion_detail = _promotion_decision(
        champion_metrics,
        challenger_metrics,
        incumbent_brier=incumbent_brier,
        challenger_brier_losses=challenger_brier_losses,
        incumbent_brier_losses=incumbent_brier_losses,
    )
    dm_p_value = promotion_detail.get("dm_p_value")
    production_status, activation_mode, governance_reason = _governance_decision(
        promote=promote,
        challenger_family=challenger_metrics["model_family"],
        dm_p_value=dm_p_value,
    )

    report = {
        "target_date": target_date.isoformat(),
        "training_cutoff": pd.to_datetime(train_df.iloc[-1]["date"]).date().isoformat(),
        "evaluation_window_start": pd.to_datetime(holdout_df.iloc[0]["date"]).date().isoformat(),
        "evaluation_window_end": pd.to_datetime(holdout_df.iloc[-1]["date"]).date().isoformat(),
        "evaluation_matches": int(len(holdout_df)),
        "champion": champion_metrics,
        "challenger": challenger_metrics,
        "ensemble": ensemble_metrics,
        "promotion_decision": promote,
        "promotion_reason": promotion_reason,
        "challenger_rps": promotion_detail.get("challenger_rps"),
        "champion_rps": promotion_detail.get("champion_rps"),
        "rps_comparison": (
            f"RPS retador={promotion_detail.get('challenger_rps')} vs "
            f"champion={promotion_detail.get('champion_rps')} "
            "(menor es mejor; el retador debe igualar o mejorar)."
        ),
        "dm_stat": promotion_detail.get("dm_stat"),
        "dm_p_value": dm_p_value,
        "production_status": production_status,
        "activation_mode": activation_mode,
        "governance_reason": governance_reason,
        "registry_persisted": False,
        "registry_error": None,
    }
    report_path = _write_report(report, report_dir=report_dir)
    model_artifact_path = _write_model_artifact(challenger_artifact, target_date=target_date, report_dir=report_dir)

    try:
        engine = get_engine(database_url)
        tables = reflect_prediction_tables(engine)
        training_cutoff = pd.to_datetime(train_df.iloc[-1]["date"]).date()
        evaluation_window_start = pd.to_datetime(holdout_df.iloc[0]["date"]).date()
        evaluation_window_end = pd.to_datetime(holdout_df.iloc[-1]["date"]).date()
        model_versions_table = tables.get("model_versions")

        if model_versions_table is None:
            report["registry_error"] = "Prediction memory tables are not available."
        else:
            champion_production_status = champion_pointer_status(
                challenger_production_status=production_status,
                active_model_exists=active_record is not None,
            )
            champion_row = {
                "version_key": f"{champion_metrics['model_version']}:{target_date.isoformat()}",
                "model_version": champion_metrics["model_version"],
                "model_family": champion_metrics["model_family"],
                "training_cutoff": training_cutoff,
                "trained_at": datetime.now(timezone.utc),
                "evaluation_window_start": evaluation_window_start,
                "evaluation_window_end": evaluation_window_end,
                "brier_score": champion_metrics["brier_score"],
                "log_loss": champion_metrics["log_loss"],
                "accuracy_1x2": champion_metrics["accuracy_1x2"],
                "exact_score_hit_rate": champion_metrics["exact_score_hit_rate"],
                "status": "champion",
                "promotion_reason": "Champion baseline productivo actual.",
                "is_promotion_eligible": True,
                "production_status": champion_production_status,
                "activation_mode": "manual",
                "artifact_path": str(report_path),
                "metrics_payload": report["champion"],
            }
            challenger_row = {
                "version_key": f"{challenger_metrics['model_version']}:{target_date.isoformat()}",
                "model_version": challenger_metrics["model_version"],
                "model_family": challenger_metrics["model_family"],
                "training_cutoff": training_cutoff,
                "trained_at": datetime.now(timezone.utc),
                "evaluation_window_start": evaluation_window_start,
                "evaluation_window_end": evaluation_window_end,
                "brier_score": challenger_metrics["brier_score"],
                "log_loss": challenger_metrics["log_loss"],
                "accuracy_1x2": challenger_metrics["accuracy_1x2"],
                "exact_score_hit_rate": challenger_metrics["exact_score_hit_rate"],
                "status": "challenger" if promote else "rejected",
                "promotion_reason": governance_reason,
                "is_promotion_eligible": promote,
                "production_status": production_status,
                "activation_mode": activation_mode,
                "activated_at": datetime.now(timezone.utc) if production_status == "active" else None,
                "artifact_path": str(model_artifact_path),
                "metrics_payload": report["challenger"],
            }
            upsert_model_version_record(engine, model_versions_table, champion_row)
            upsert_model_version_record(engine, model_versions_table, challenger_row)
            # Persist the binary into the durable store so any environment can hydrate it.
            # Without this the artifact only ever exists on the ephemeral training host and
            # inference silently falls back to Poisson at brief time.
            try:
                from alttez.operations.artifact_store import upload_artifact

                upload_artifact(challenger_row["version_key"], model_artifact_path, database_url=database_url)
                report["artifact_uploaded"] = True
            except Exception as exc:  # noqa: BLE001 - artifact upload must not break the registry write.
                report["artifact_upload_error"] = str(exc)
                logging.warning("artifact_upload_failed version_key=%s detail=%s", challenger_row["version_key"], exc)
            if production_status == "active":
                set_active_model_version_record(
                    engine,
                    model_versions_table,
                    active_version_key=challenger_row["version_key"],
                    activation_mode=activation_mode,
                )
            report["registry_persisted"] = True
    except Exception as exc:  # noqa: BLE001 - challenger evaluation should still produce the artifact.
        report["registry_error"] = str(exc)

    report["artifact_path"] = str(report_path)
    report["model_artifact_path"] = str(model_artifact_path)
    _write_report(report, report_dir=report_dir)
    return report
