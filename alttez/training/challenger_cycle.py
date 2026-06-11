from __future__ import annotations

import json
import logging
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
    return {
        "predicted_outcome": predicted_outcome,
        "actual_outcome": actual_outcome,
        "brier": brier_score_1x2(prob_home, prob_draw, prob_away, actual_outcome),
        "prob_home": prob_home,
        "prob_draw": prob_draw,
        "prob_away": prob_away,
        "exact_hit": int(max_score[0] == actual_home_score and max_score[1] == actual_away_score),
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
        "log_loss": float(log_loss(y_true, probability_matrix, labels=[0, 1, 2])),
        "accuracy_1x2": float((eval_df["predicted_outcome"] == eval_df["actual_outcome"]).mean()),
        "exact_score_hit_rate": float(eval_df["exact_hit"].mean()),
        "expected_calibration_error": expected_calibration_error(eval_df),
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

    home_model = _build_goal_regressor(use_early_stopping=use_early_stopping)
    away_model = _build_goal_regressor(use_early_stopping=use_early_stopping)

    if use_early_stopping:
        x_validation = validation_df[FEATURE_COLUMNS].fillna(0.0)
        home_model.fit(
            x_train,
            train_core["home_score"],
            eval_set=[(x_validation, validation_df["home_score"])],
            verbose=False,
        )
        away_model.fit(
            x_train,
            train_core["away_score"],
            eval_set=[(x_validation, validation_df["away_score"])],
            verbose=False,
        )
    else:
        home_model.fit(x_train, train_core["home_score"], verbose=False)
        away_model.fit(x_train, train_core["away_score"], verbose=False)

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


def _promotion_decision(
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    *,
    incumbent_brier: float | None = None,
) -> tuple[bool, str]:
    if float(challenger_metrics["brier_score"]) >= float(champion_metrics["brier_score"]):
        return False, "Challenger no mejora Brier Score frente al champion actual."
    if float(challenger_metrics["log_loss"]) > float(champion_metrics["log_loss"]) + DEFAULT_PROMOTION_LOG_LOSS_TOLERANCE:
        return False, "Challenger mejora Brier pero deteriora Log Loss materialmente."
    # Incumbent protection: never auto-replace the active model with a worse one.
    if incumbent_brier is not None and float(challenger_metrics["brier_score"]) >= incumbent_brier - DEFAULT_PROMOTION_LOG_LOSS_TOLERANCE:
        return False, "Challenger no supera al modelo activo (sin regresion permitida)."
    return True, "Challenger elegible por Brier; supera champion e incumbente."


def _governance_decision(
    *,
    promote: bool,
    challenger_family: str,
) -> tuple[str, str, str]:
    if not promote:
        return "shadow", "blocked", "Challenger descartado por politica de metricas."
    if challenger_family.lower() not in production_supported_model_families():
        return "eligible", "blocked", "Elegible por Brier, pero su familia aun no esta habilitada para inferencia productiva."
    if not auto_promotion_enabled():
        return "eligible", "manual", "Elegible por Brier; pendiente de aprobacion explicita para activacion productiva."
    return "active", "auto", "Promovido automaticamente por mejorar Brier sin deterioro material de Log Loss."


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

    champion_metrics = _evaluate_poisson_champion(
        train_df,
        holdout_df,
        champion_refit_interval=champion_refit_interval,
    )
    challenger_artifact = _train_challenger(train_df)
    challenger_metrics = _evaluate_challenger(challenger_artifact, holdout_df)
    challenger_metrics["feature_importance"] = challenger_artifact.get("feature_importance")

    # Ensemble probe: does blending champion + challenger lambdas beat either alone?
    champion_lambdas, _ = _champion_lambda_stream(
        train_df, holdout_df, champion_refit_interval=champion_refit_interval
    )
    challenger_lambdas = _xgb_lambda_stream(challenger_artifact, holdout_df)
    ensemble_metrics = evaluate_lambda_blend(champion_lambdas, challenger_lambdas, holdout_df)

    active_record = load_active_model_record(database_url=database_url)
    incumbent_brier = _incumbent_brier(active_record)
    promote, promotion_reason = _promotion_decision(
        champion_metrics, challenger_metrics, incumbent_brier=incumbent_brier
    )
    production_status, activation_mode, governance_reason = _governance_decision(
        promote=promote,
        challenger_family=challenger_metrics["model_family"],
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
                "production_status": "active" if production_status != "active" else "archived",
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
