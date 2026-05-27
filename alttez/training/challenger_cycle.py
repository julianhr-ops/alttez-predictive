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
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from alttez.api.db import get_engine
from alttez.features.elo import EloEngine
from alttez.features.poisson import PoissonEngine
from alttez.experiments.tournament_oracle import build_probability_matrix
from alttez.ingestion.international import InternationalLoader
from alttez.operations.prediction_memory import (
    auto_promotion_enabled,
    brier_score_1x2,
    get_active_model_family,
    get_active_model_version,
    production_supported_model_families,
    reflect_prediction_tables,
    set_active_model_version_record,
    upsert_model_version_record,
)


logging.basicConfig(level=logging.INFO)


DEFAULT_REPORT_DIR = Path("artifacts/model-evals")
DEFAULT_CHALLENGER_VERSION = "xgboost-challenger-v1"
DEFAULT_EVALUATION_MATCHES = 250
DEFAULT_PROMOTION_LOG_LOSS_TOLERANCE = 0.01
FEATURE_COLUMNS = [
    "elo_difference",
    "elo_home_before",
    "elo_away_before",
    "is_neutral",
    "home_recent_points",
    "away_recent_points",
    "home_recent_goal_diff",
    "away_recent_goal_diff",
]


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


def _build_challenger_dataset(df: pd.DataFrame) -> pd.DataFrame:
    elo_engine = EloEngine()
    elo_df = elo_engine.compute_elo_features(df.copy())
    elo_df["is_neutral"] = elo_df["neutral"].astype(int)
    elo_df = _build_recent_form_features(elo_df)
    elo_df["outcome_label"] = elo_df.apply(
        lambda row: _outcome_label(int(row["home_score"]), int(row["away_score"])),
        axis=1,
    )
    return elo_df


def _evaluate_poisson_champion(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict[str, Any]:
    engine = PoissonEngine(
        years_window=4,
        prior_matches=8,
        min_history_matches=5,
        recency_half_life_days=365,
        recent_form_matches=6,
    )
    cutoff = pd.to_datetime(holdout_df.iloc[0]["date"]).date().isoformat()
    engine.fit(train_df, target_date=cutoff)

    rows = []
    for row in holdout_df.itertuples(index=False):
        lambda_home, lambda_away = engine.predict_xg(row.home_team, row.away_team, is_neutral=bool(row.neutral))
        prob_matrix = build_probability_matrix(lambda_home, lambda_away, max_goals=7)
        prob_home = float(np.tril(prob_matrix, -1).sum())
        prob_draw = float(prob_matrix.diagonal().sum())
        prob_away = float(np.triu(prob_matrix, 1).sum())
        predicted_outcome = max(
            (("H", prob_home), ("D", prob_draw), ("A", prob_away)),
            key=lambda item: item[1],
        )[0]
        actual_outcome = _outcome_label(int(row.home_score), int(row.away_score))
        max_score = divmod(int(prob_matrix.argmax()), prob_matrix.shape[1])
        rows.append(
            {
                "predicted_outcome": predicted_outcome,
                "actual_outcome": actual_outcome,
                "brier": brier_score_1x2(prob_home, prob_draw, prob_away, actual_outcome),
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "exact_hit": int(max_score[0] == int(row.home_score) and max_score[1] == int(row.away_score)),
            }
        )

    eval_df = pd.DataFrame(rows)
    probability_matrix = eval_df[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    y_true = eval_df["actual_outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return {
        "model_version": get_active_model_version(),
        "model_family": get_active_model_family(),
        "brier_score": float(eval_df["brier"].mean()),
        "log_loss": float(log_loss(y_true, probability_matrix, labels=[0, 1, 2])),
        "accuracy_1x2": float((eval_df["predicted_outcome"] == eval_df["actual_outcome"]).mean()),
        "exact_score_hit_rate": float(eval_df["exact_hit"].mean()),
    }


def _build_feature_matrices(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    label_map = {"H": 0, "D": 1, "A": 2}
    x_train = train_df[FEATURE_COLUMNS].fillna(0.0)
    x_holdout = holdout_df[FEATURE_COLUMNS].fillna(0.0)
    y_train = train_df["outcome_label"].map(label_map)
    y_holdout = holdout_df["outcome_label"].map(label_map)
    return x_train, x_holdout, y_train, y_holdout


def _train_challenger(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    x_train, x_holdout, y_train, y_holdout = _build_feature_matrices(train_df, holdout_df)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=140,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.95,
        colsample_bytree=0.95,
        eval_metric="mlogloss",
        random_state=0,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_holdout)
    predictions = probabilities.argmax(axis=1)

    brier_values = []
    for index, actual_label in enumerate(y_holdout.to_numpy()):
        actual_outcome = "H" if actual_label == 0 else "D" if actual_label == 1 else "A"
        brier_values.append(
            brier_score_1x2(
                probabilities[index, 0],
                probabilities[index, 1],
                probabilities[index, 2],
                actual_outcome,
            )
        )

    metrics = {
        "model_version": DEFAULT_CHALLENGER_VERSION,
        "model_family": "xgboost",
        "brier_score": float(sum(brier_values) / len(brier_values)),
        "log_loss": float(log_loss(y_holdout, probabilities, labels=[0, 1, 2])),
        "accuracy_1x2": float(accuracy_score(y_holdout, predictions)),
        "exact_score_hit_rate": None,
    }
    artifact_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "label_order": ["H", "D", "A"],
        "model_version": DEFAULT_CHALLENGER_VERSION,
        "model_family": "xgboost",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return metrics, artifact_bundle


def _promotion_decision(champion_metrics: dict[str, Any], challenger_metrics: dict[str, Any]) -> tuple[bool, str]:
    if float(challenger_metrics["brier_score"]) >= float(champion_metrics["brier_score"]):
        return False, "Challenger no mejora Brier Score frente al champion actual."
    if float(challenger_metrics["log_loss"]) > float(champion_metrics["log_loss"]) + DEFAULT_PROMOTION_LOG_LOSS_TOLERANCE:
        return False, "Challenger mejora Brier pero deteriora Log Loss materialmente."
    return True, "Challenger elegible por Brier; promotion controlado pendiente de activacion productiva."


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
) -> dict[str, Any]:
    loader = InternationalLoader()
    historical_df = loader.fetch_data().sort_values("date").reset_index(drop=True)
    challenger_df = _build_challenger_dataset(historical_df)
    eval_size = min(evaluation_matches, max(60, len(challenger_df) // 20))
    train_df = challenger_df.iloc[:-eval_size].copy()
    holdout_df = challenger_df.iloc[-eval_size:].copy()

    champion_metrics = _evaluate_poisson_champion(train_df, holdout_df)
    challenger_metrics, challenger_artifact = _train_challenger(train_df, holdout_df)
    promote, promotion_reason = _promotion_decision(champion_metrics, challenger_metrics)
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
    return report
