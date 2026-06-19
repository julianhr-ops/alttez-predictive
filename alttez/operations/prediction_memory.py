from __future__ import annotations

import math
import os
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy import MetaData, Table, inspect, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from alttez.api.db import get_engine, read_dataframe
from alttez.api.settings import load_local_env


LOGGER = logging.getLogger(__name__)
DEFAULT_SCHEMA = "public"
DEFAULT_POISSON_MODEL_VERSION = "poisson-champion-v2"
DEFAULT_POISSON_MODEL_FAMILY = "poisson"
DEFAULT_SUPPORTED_PRODUCTION_MODEL_FAMILIES = "poisson,xgboost"
# Piso de probabilidad para el log loss almacenado. Un piso demasiado pequeno
# (p.ej. 1e-9 -> -log = 20.7) infla artificialmente el log loss cuando una clase
# recibe prob ~0, haciendo que la metrica de monitoreo reporte ~3.85 cuando el
# log loss real (piso sano) es ~0.98. 1e-3 acota el castigo a -log(1e-3)=6.9.
DEFAULT_EVALUATION_EPSILON = 1e-3


def get_active_model_version() -> str:
    return os.getenv("ALTTEZ_ACTIVE_MODEL_VERSION", DEFAULT_POISSON_MODEL_VERSION).strip() or DEFAULT_POISSON_MODEL_VERSION


def get_active_model_family() -> str:
    return os.getenv("ALTTEZ_ACTIVE_MODEL_FAMILY", DEFAULT_POISSON_MODEL_FAMILY).strip() or DEFAULT_POISSON_MODEL_FAMILY


def production_supported_model_families() -> set[str]:
    raw = os.getenv("ALTTEZ_SUPPORTED_PRODUCTION_MODEL_FAMILIES", DEFAULT_SUPPORTED_PRODUCTION_MODEL_FAMILIES)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def auto_promotion_enabled() -> bool:
    raw = os.getenv("ALTTEZ_AUTO_PROMOTE_MODELS", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def reflect_prediction_tables(engine: Engine, schema: str = DEFAULT_SCHEMA) -> dict[str, Table]:
    requested_tables = [
        "matches",
        "prediction_runs",
        "match_predictions",
        "prediction_evaluations",
        "model_versions",
    ]
    available_tables = set(inspect(engine).get_table_names(schema=schema))
    tables_to_reflect = [table_name for table_name in requested_tables if table_name in available_tables]
    if not tables_to_reflect:
        return {}

    metadata = MetaData(schema=schema)
    metadata.reflect(
        bind=engine,
        schema=schema,
        only=tables_to_reflect,
    )
    return {
        table_name: metadata.tables[f"{schema}.{table_name}"]
        for table_name in requested_tables
        if f"{schema}.{table_name}" in metadata.tables
    }


def _prediction_run_identifier(target_date: str, model_version: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{target_date}:{model_version}:{stamp}:{uuid4().hex[:8]}"


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _predicted_outcome(match_payload: dict[str, Any]) -> str:
    probabilities = {
        "H": float(match_payload.get("prob_home") or 0.0),
        "D": float(match_payload.get("prob_draw") or 0.0),
        "A": float(match_payload.get("prob_away") or 0.0),
    }
    return max(probabilities.items(), key=lambda item: item[1])[0]


def _decision_label(match_payload: dict[str, Any]) -> str:
    decision_tier = str(match_payload.get("decision_tier") or "").strip().lower()
    support_label = str(match_payload.get("support_label") or "").strip().lower()
    if decision_tier == "alta prioridad" or support_label == "analizable":
        return "Decision principal"
    if decision_tier == "seguimiento" or support_label == "soporte parcial":
        return "Vigilancia"
    return "Descarte"


def _avg_confidence(match_payload: dict[str, Any]) -> float:
    return (
        float(match_payload.get("confidence_local") or 0.0) + float(match_payload.get("confidence_visitante") or 0.0)
    ) / 2.0


def _filter_match_prediction_row(match_payload: dict[str, Any], run_row_id: Any, run_id: str, model_version: str, model_family: str) -> dict[str, Any]:
    return {
        "prediction_run_id": run_row_id,
        "match_id": match_payload["match_id"],
        "model_version": model_version,
        "model_family": model_family,
        "prediction_created_at": pd.to_datetime(match_payload.get("generated_at") or datetime.now(timezone.utc)),
        "match_date": pd.to_datetime(match_payload["target_date"]).date() if match_payload.get("target_date") else pd.to_datetime(match_payload["kickoff_at"]).date(),
        "kickoff_at": pd.to_datetime(match_payload["kickoff_at"]) if match_payload.get("kickoff_at") else None,
        "home_team": match_payload["home_team"],
        "away_team": match_payload["away_team"],
        "tournament_name": match_payload.get("tournament"),
        "competition_class": match_payload.get("competition_class"),
        "predicted_outcome": _predicted_outcome(match_payload),
        "predicted_home_win_prob": float(match_payload.get("prob_home") or 0.0),
        "predicted_draw_prob": float(match_payload.get("prob_draw") or 0.0),
        "predicted_away_win_prob": float(match_payload.get("prob_away") or 0.0),
        "predicted_scoreline": match_payload.get("primary_score"),
        "predicted_lambda_home": float(match_payload.get("lambda_h")) if match_payload.get("lambda_h") is not None else None,
        "predicted_lambda_away": float(match_payload.get("lambda_a")) if match_payload.get("lambda_a") is not None else None,
        "support_label": match_payload.get("support_label"),
        "avg_confidence": _avg_confidence(match_payload),
        "decision_tier": match_payload.get("decision_tier"),
        "decision_label": _decision_label(match_payload),
        "best_market": match_payload.get("best_market"),
        "best_market_probability": float(match_payload.get("best_market_probability") or 0.0),
        "payload": {
            "run_id": run_id,
            "generated_at": match_payload.get("generated_at"),
            "competition_label": match_payload.get("competition_label"),
            "city": match_payload.get("city"),
            "status": match_payload.get("status"),
            "support_message": match_payload.get("support_message"),
            "primary_score_probability": match_payload.get("primary_score_probability"),
            "ev_optimal_score": match_payload.get("ev_optimal_score"),
            "ev_optimal_value": match_payload.get("ev_optimal_value"),
            "score_selection_policy": match_payload.get("score_selection_policy"),
            "committee_consensus_used": match_payload.get("committee_consensus_used"),
            "committee_members": match_payload.get("committee_members"),
            "committee_weights": match_payload.get("committee_weights"),
            "discordance_warning": match_payload.get("discordance_warning"),
            "member_primary_scores": match_payload.get("member_primary_scores"),
        },
    }


def persist_daily_predictions(
    payload: dict[str, Any],
    *,
    database_url: str | None = None,
    schema: str = DEFAULT_SCHEMA,
    model_version: str | None = None,
    model_family: str | None = None,
) -> dict[str, Any]:
    load_local_env(override=True)
    engine = get_engine(database_url)
    tables = reflect_prediction_tables(engine, schema)
    runs_table = tables.get("prediction_runs")
    predictions_table = tables.get("match_predictions")
    if runs_table is None or predictions_table is None:
        return {
            "prediction_run_id": None,
            "prediction_run_row_id": None,
            "prediction_persisted": False,
            "prediction_rows": 0,
            "prediction_error": "Prediction memory tables are not available.",
        }

    effective_model_version = model_version or str(payload.get("model_version") or get_active_model_version())
    effective_model_family = model_family or str(payload.get("model_family") or get_active_model_family())
    run_id = _prediction_run_identifier(payload["target_date"], effective_model_version)
    generated_at = pd.to_datetime(payload.get("generated_at") or datetime.now(timezone.utc))
    run_row = {
        "run_id": run_id,
        "model_version": effective_model_version,
        "model_family": effective_model_family,
        "target_date": pd.to_datetime(payload["target_date"]).date(),
        "generated_at": generated_at,
        "training_cutoff": pd.to_datetime(payload["target_date"]).date(),
        "status": "completed",
        "summary_payload": _json_safe(
            {
                "summary": payload.get("summary", {}),
                "batch_summary": payload.get("batch_summary", {}),
                "alert_summary": payload.get("alert_summary", {}),
            }
        ),
    }

    with engine.begin() as connection:
        stmt = insert(runs_table).values(run_row).returning(runs_table.c.id)
        stmt = stmt.on_conflict_do_update(
            index_elements=["run_id"],
            set_={
                "model_version": stmt.excluded.model_version,
                "model_family": stmt.excluded.model_family,
                "generated_at": stmt.excluded.generated_at,
                "training_cutoff": stmt.excluded.training_cutoff,
                "status": stmt.excluded.status,
                "summary_payload": stmt.excluded.summary_payload,
            },
        ).returning(runs_table.c.id)
        run_row_id = connection.execute(stmt).scalar_one()

        rows = []
        for match_payload in payload.get("matches", []):
            enriched_match_payload = dict(match_payload)
            enriched_match_payload["generated_at"] = payload.get("generated_at")
            enriched_match_payload["target_date"] = payload.get("target_date")
            rows.append(
                _json_safe(
                    _filter_match_prediction_row(
                        enriched_match_payload,
                        run_row_id,
                        run_id,
                        effective_model_version,
                        effective_model_family,
                    )
                )
            )

        if rows:
            stmt = insert(predictions_table).values(rows)
            update_map = {
                key: getattr(stmt.excluded, key)
                for key in rows[0]
                if key not in {"prediction_run_id", "match_id"}
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["prediction_run_id", "match_id"],
                set_=update_map,
            )
            connection.execute(stmt)

    return {
        "prediction_run_id": run_id,
        "prediction_run_row_id": str(run_row_id),
        "prediction_persisted": True,
        "prediction_rows": len(payload.get("matches", [])),
        "prediction_error": None,
    }


def _actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _outcome_vector(outcome: str) -> tuple[int, int, int]:
    return (
        1 if outcome == "H" else 0,
        1 if outcome == "D" else 0,
        1 if outcome == "A" else 0,
    )


def brier_score_1x2(home_prob: float, draw_prob: float, away_prob: float, actual_outcome: str) -> float:
    actual_home, actual_draw, actual_away = _outcome_vector(actual_outcome)
    return (
        (float(home_prob) - actual_home) ** 2
        + (float(draw_prob) - actual_draw) ** 2
        + (float(away_prob) - actual_away) ** 2
    )


def log_loss_1x2(home_prob: float, draw_prob: float, away_prob: float, actual_outcome: str) -> float:
    probability_map = {
        "H": float(home_prob),
        "D": float(draw_prob),
        "A": float(away_prob),
    }
    actual_probability = max(min(probability_map.get(actual_outcome, 0.0), 1.0 - DEFAULT_EVALUATION_EPSILON), DEFAULT_EVALUATION_EPSILON)
    return -math.log(actual_probability)


def close_finished_prediction_evaluations(
    *,
    engine: Engine,
    tables: dict[str, Table],
    target_date: date | None = None,
) -> dict[str, Any]:
    predictions_table = tables.get("match_predictions")
    evaluations_table = tables.get("prediction_evaluations")
    matches_table = tables.get("matches")
    if predictions_table is None or evaluations_table is None or matches_table is None:
        return {
            "prediction_evaluations_closed": 0,
            "prediction_evaluation_error": "Prediction memory tables are not available.",
        }

    filters = ["m.status = 'finished'"]
    params: dict[str, Any] = {}
    if target_date is not None:
        filters.append("m.match_date = :target_date")
        params["target_date"] = target_date.isoformat()

    sql = text(
        f"""
        select
            mp.id as prediction_id,
            mp.match_id,
            mp.predicted_outcome,
            mp.predicted_home_win_prob,
            mp.predicted_draw_prob,
            mp.predicted_away_win_prob,
            mp.predicted_scoreline,
            m.home_score,
            m.away_score
        from public.match_predictions as mp
        join public.matches as m
            on m.id = mp.match_id
        left join public.prediction_evaluations as pe
            on pe.prediction_id = mp.id
        where pe.id is null
          and {' and '.join(filters)}
        """
    )

    with engine.begin() as connection:
        rows = connection.execute(sql, params).mappings().all()
        evaluation_rows = []
        for row in rows:
            actual_home_score = int(row["home_score"])
            actual_away_score = int(row["away_score"])
            actual_outcome = _actual_outcome(actual_home_score, actual_away_score)
            predicted_scoreline = str(row.get("predicted_scoreline") or "0 - 0")
            try:
                predicted_home_score, predicted_away_score = [int(item.strip()) for item in predicted_scoreline.split("-", 1)]
            except Exception:
                predicted_home_score, predicted_away_score = 0, 0

            evaluation_rows.append(
                {
                    "prediction_id": row["prediction_id"],
                    "match_id": row["match_id"],
                    "evaluation_status": "evaluated",
                    "actual_home_score": actual_home_score,
                    "actual_away_score": actual_away_score,
                    "actual_outcome": actual_outcome,
                    "hit_1x2": row["predicted_outcome"] == actual_outcome,
                    "hit_exact_score": predicted_home_score == actual_home_score and predicted_away_score == actual_away_score,
                    "brier_1x2": brier_score_1x2(
                        row["predicted_home_win_prob"],
                        row["predicted_draw_prob"],
                        row["predicted_away_win_prob"],
                        actual_outcome,
                    ),
                    "log_loss_1x2": log_loss_1x2(
                        row["predicted_home_win_prob"],
                        row["predicted_draw_prob"],
                        row["predicted_away_win_prob"],
                        actual_outcome,
                    ),
                    "score_residual_home": predicted_home_score - actual_home_score,
                    "score_residual_away": predicted_away_score - actual_away_score,
                    "score_abs_error": abs(predicted_home_score - actual_home_score) + abs(predicted_away_score - actual_away_score),
                    "evaluated_at": datetime.now(timezone.utc),
                    "payload": {
                        "predicted_scoreline": predicted_scoreline,
                    },
                }
            )

        if evaluation_rows:
            stmt = insert(evaluations_table).values(evaluation_rows)
            update_map = {
                key: getattr(stmt.excluded, key)
                for key in evaluation_rows[0]
                if key != "prediction_id"
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["prediction_id"],
                set_=update_map,
            )
            connection.execute(stmt)
        audit_filter = "where mp.match_date = :target_date" if target_date is not None else ""
        audit_params = {"target_date": target_date.isoformat()} if target_date is not None else {}
        audit_row = connection.execute(
            text(
                f"""
                select
                    count(*) as evaluated_predictions,
                    avg(pe.brier_1x2) as brier_score,
                    avg(pe.log_loss_1x2) as log_loss,
                    avg(case when pe.hit_1x2 then 1.0 else 0.0 end) as accuracy_1x2,
                    avg(case when pe.hit_exact_score then 1.0 else 0.0 end) as exact_score_hit_rate
                from public.prediction_evaluations as pe
                join public.match_predictions as mp
                    on mp.id = pe.prediction_id
                {audit_filter}
                """
            ),
            audit_params,
        ).mappings().first()

    if audit_row and int(audit_row["evaluated_predictions"] or 0) > 0:
        LOGGER.info(
            "prediction_learning_audit evaluated=%s brier=%.4f log_loss=%.4f accuracy_1x2=%.3f exact_score_hit=%.3f",
            int(audit_row["evaluated_predictions"] or 0),
            float(audit_row["brier_score"] or 0.0),
            float(audit_row["log_loss"] or 0.0),
            float(audit_row["accuracy_1x2"] or 0.0),
            float(audit_row["exact_score_hit_rate"] or 0.0),
        )

    return {
        "prediction_evaluations_closed": len(evaluation_rows if 'evaluation_rows' in locals() else []),
        "prediction_evaluation_error": None,
    }


def load_prediction_overview(*, database_url: str | None = None, lookback_days: int = 180) -> dict[str, Any]:
    params = {"lookback_start": (datetime.now(timezone.utc).date() - pd.Timedelta(days=lookback_days)).isoformat()}
    try:
        df = read_dataframe(
            """
            select
                pe.brier_1x2,
                pe.log_loss_1x2,
                pe.hit_1x2,
                pe.hit_exact_score
            from public.prediction_evaluations as pe
            join public.match_predictions as mp
                on mp.id = pe.prediction_id
            where mp.match_date >= :lookback_start
            """,
            params=params,
            database_url=database_url,
        )
    except Exception:
        return {
            "evaluated_predictions": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "accuracy_1x2": 0.0,
            "exact_score_hit_rate": 0.0,
        }

    if df.empty:
        return {
            "evaluated_predictions": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "accuracy_1x2": 0.0,
            "exact_score_hit_rate": 0.0,
        }

    return {
        "evaluated_predictions": int(len(df)),
        "brier_score": float(df["brier_1x2"].mean()),
        "log_loss": float(df["log_loss_1x2"].mean()),
        "accuracy_1x2": float(df["hit_1x2"].mean()),
        "exact_score_hit_rate": float(df["hit_exact_score"].mean()),
    }


def load_prediction_learning_audit(*, database_url: str | None = None, lookback_days: int = 180) -> dict[str, Any]:
    params = {"lookback_start": (datetime.now(timezone.utc).date() - pd.Timedelta(days=lookback_days)).isoformat()}
    try:
        df = read_dataframe(
            """
            select
                mp.model_version,
                pe.brier_1x2,
                pe.log_loss_1x2,
                pe.hit_1x2,
                pe.hit_exact_score
            from public.prediction_evaluations as pe
            join public.match_predictions as mp
                on mp.id = pe.prediction_id
            where mp.match_date >= :lookback_start
            """,
            params=params,
            database_url=database_url,
        )
    except Exception as exc:
        return {
            "evaluated_predictions": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "accuracy_1x2": 0.0,
            "exact_score_hit_rate": 0.0,
            "models": [],
            "error": str(exc),
        }

    if df.empty:
        return {
            "evaluated_predictions": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "accuracy_1x2": 0.0,
            "exact_score_hit_rate": 0.0,
            "models": [],
            "error": None,
        }

    models = []
    for model_version, group in df.groupby("model_version", dropna=False):
        models.append(
            {
                "model_version": str(model_version),
                "evaluated_predictions": int(len(group)),
                "brier_score": float(group["brier_1x2"].mean()),
                "log_loss": float(group["log_loss_1x2"].mean()),
                "accuracy_1x2": float(group["hit_1x2"].mean()),
                "exact_score_hit_rate": float(group["hit_exact_score"].mean()),
            }
        )

    return {
        "evaluated_predictions": int(len(df)),
        "brier_score": float(df["brier_1x2"].mean()),
        "log_loss": float(df["log_loss_1x2"].mean()),
        "accuracy_1x2": float(df["hit_1x2"].mean()),
        "exact_score_hit_rate": float(df["hit_exact_score"].mean()),
        "models": sorted(models, key=lambda row: row["evaluated_predictions"], reverse=True),
        "error": None,
    }


def load_calibration_overview(*, database_url: str | None = None, lookback_days: int = 365) -> dict[str, Any]:
    """Resumen de calibración: ECE por modelo + global y el artefacto activo.

    Alimenta el panel del dashboard. Lectura tolerante: ante cualquier fallo
    devuelve estructura vacía en vez de propagar (el dashboard nunca debe romper).
    """
    from alttez.inference.calibration import expected_calibration_error, load_active_calibration

    artifact = load_active_calibration()
    artifact_meta: dict[str, Any] = {
        "temperature": artifact.temperature,
        "class_floor": artifact.class_floor,
        "is_neutral": artifact.is_neutral,
    }
    artifact_path = Path(os.getenv("ALTTEZ_CALIBRATION_PATH") or "artifacts/calibration/active.json")
    if artifact_path.exists():
        try:
            artifact_meta.update(json.loads(artifact_path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - artefacto corrupto no debe romper el panel.
            pass

    empty = {"models": pd.DataFrame(), "global_ece": None, "artifact": artifact_meta, "samples": 0}
    params = {"lookback_start": (datetime.now(timezone.utc).date() - pd.Timedelta(days=lookback_days)).isoformat()}
    try:
        df = read_dataframe(
            """
            select
                mp.model_version,
                mp.predicted_home_win_prob as prob_home,
                mp.predicted_draw_prob as prob_draw,
                mp.predicted_away_win_prob as prob_away,
                pe.actual_outcome
            from public.prediction_evaluations as pe
            join public.match_predictions as mp on mp.id = pe.prediction_id
            where mp.match_date >= :lookback_start and pe.actual_outcome is not null
            """,
            params=params,
            database_url=database_url,
        )
    except Exception:  # noqa: BLE001
        return empty

    if df.empty:
        return empty

    def _ece(frame: pd.DataFrame) -> float:
        triplets = frame[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
        return round(expected_calibration_error(triplets, frame["actual_outcome"].tolist()), 4)

    rows = [
        {"model_version": str(model_version), "evaluated_predictions": int(len(group)), "ece": _ece(group)}
        for model_version, group in df.groupby("model_version")
    ]
    return {
        "models": pd.DataFrame(rows).sort_values("evaluated_predictions", ascending=False),
        "global_ece": _ece(df),
        "artifact": artifact_meta,
        "samples": int(len(df)),
    }


def load_prediction_metrics_by_model(*, database_url: str | None = None) -> pd.DataFrame:
    try:
        return read_dataframe(
            """
            select
                mp.model_version,
                count(*) as evaluated_predictions,
                avg(pe.brier_1x2) as brier_score,
                avg(pe.log_loss_1x2) as log_loss,
                avg(case when pe.hit_1x2 then 1.0 else 0.0 end) as accuracy_1x2,
                avg(case when pe.hit_exact_score then 1.0 else 0.0 end) as exact_score_hit_rate
            from public.prediction_evaluations as pe
            join public.match_predictions as mp
                on mp.id = pe.prediction_id
            group by mp.model_version
            order by mp.model_version desc
            """,
            database_url=database_url,
        )
    except Exception:
        return pd.DataFrame()


def load_prediction_metrics_by_competition(*, database_url: str | None = None) -> pd.DataFrame:
    try:
        return read_dataframe(
            """
            select
                coalesce(mp.competition_class, 'other_international') as competition_class,
                count(*) as evaluated_predictions,
                avg(pe.brier_1x2) as brier_score,
                avg(pe.log_loss_1x2) as log_loss,
                avg(case when pe.hit_1x2 then 1.0 else 0.0 end) as accuracy_1x2
            from public.prediction_evaluations as pe
            join public.match_predictions as mp
                on mp.id = pe.prediction_id
            group by coalesce(mp.competition_class, 'other_international')
            order by evaluated_predictions desc, competition_class
            """,
            database_url=database_url,
        )
    except Exception:
        return pd.DataFrame()


def load_prediction_metrics_by_confidence(*, database_url: str | None = None) -> pd.DataFrame:
    try:
        return read_dataframe(
            """
            select
                case
                    when mp.avg_confidence >= 0.80 then '0.80+'
                    when mp.avg_confidence >= 0.65 then '0.65-0.79'
                    when mp.avg_confidence >= 0.50 then '0.50-0.64'
                    else '<0.50'
                end as confidence_band,
                count(*) as evaluated_predictions,
                avg(pe.brier_1x2) as brier_score,
                avg(pe.log_loss_1x2) as log_loss,
                avg(case when pe.hit_1x2 then 1.0 else 0.0 end) as accuracy_1x2
            from public.prediction_evaluations as pe
            join public.match_predictions as mp
                on mp.id = pe.prediction_id
            group by confidence_band
            order by confidence_band
            """,
            database_url=database_url,
        )
    except Exception:
        return pd.DataFrame()


def load_recent_prediction_evaluations(*, database_url: str | None = None, limit: int = 50) -> pd.DataFrame:
    try:
        return read_dataframe(
            """
            select
                mp.prediction_created_at,
                mp.model_version,
                mp.home_team,
                mp.away_team,
                mp.tournament_name,
                mp.competition_class,
                mp.predicted_scoreline,
                mp.predicted_outcome,
                mp.predicted_home_win_prob,
                mp.predicted_draw_prob,
                mp.predicted_away_win_prob,
                mp.decision_label,
                pe.actual_home_score,
                pe.actual_away_score,
                pe.actual_outcome,
                pe.hit_1x2,
                pe.hit_exact_score,
                pe.brier_1x2,
                pe.log_loss_1x2,
                pe.score_abs_error,
                pe.evaluated_at
            from public.prediction_evaluations as pe
            join public.match_predictions as mp
                on mp.id = pe.prediction_id
            order by pe.evaluated_at desc
            limit :limit
            """,
            params={"limit": limit},
            database_url=database_url,
        )
    except Exception:
        return pd.DataFrame()


def upsert_model_version_record(
    engine: Engine,
    model_versions_table: Table | None,
    row: dict[str, Any],
) -> bool:
    if model_versions_table is None:
        return False
    filtered_row = {key: value for key, value in row.items() if key in model_versions_table.c}
    with engine.begin() as connection:
        stmt = insert(model_versions_table).values(filtered_row)
        update_map = {
            key: getattr(stmt.excluded, key)
            for key in filtered_row
            if key != "version_key"
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["version_key"],
            set_=update_map,
        )
        connection.execute(stmt)
    return True


def set_active_model_version_record(
    engine: Engine,
    model_versions_table: Table | None,
    *,
    active_version_key: str,
    activation_mode: str,
) -> bool:
    if model_versions_table is None:
        return False

    with engine.begin() as connection:
        connection.execute(
            model_versions_table.update()
            .where(model_versions_table.c.production_status == "active")
            .values(production_status="archived")
        )
        connection.execute(
            model_versions_table.update()
            .where(model_versions_table.c.version_key == active_version_key)
            .values(
                status="champion",
                production_status="active",
                activation_mode=activation_mode,
                activated_at=datetime.now(timezone.utc),
                is_promotion_eligible=True,
            )
        )
    return True


def load_model_versions(*, database_url: str | None = None, limit: int = 20) -> pd.DataFrame:
    try:
        return read_dataframe(
            """
            select
                version_key,
                model_version,
                model_family,
                training_cutoff,
                trained_at,
                evaluation_window_start,
                evaluation_window_end,
                brier_score,
                log_loss,
                accuracy_1x2,
                exact_score_hit_rate,
                status,
                is_promotion_eligible,
                production_status,
                activation_mode,
                activated_at,
                promotion_reason,
                artifact_path
            from public.model_versions
            order by trained_at desc
            limit :limit
            """,
            params={"limit": limit},
            database_url=database_url,
        )
    except Exception:
        return pd.DataFrame()


def load_active_model_record(*, database_url: str | None = None) -> dict[str, Any] | None:
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
            where production_status = 'active'
            order by coalesce(activated_at, trained_at) desc
            limit 1
            """,
            database_url=database_url,
        )
    except Exception:
        return None

    if df.empty:
        return None
    return df.iloc[0].to_dict()


def load_model_version_record(version_key: str, *, database_url: str | None = None) -> dict[str, Any] | None:
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
            where version_key = :version_key
            limit 1
            """,
            params={"version_key": version_key},
            database_url=database_url,
        )
    except Exception:
        return None

    if df.empty:
        return None
    return df.iloc[0].to_dict()


def activate_model_version_by_key(
    version_key: str,
    *,
    database_url: str | None = None,
    activation_mode: str = "manual",
) -> dict[str, Any]:
    load_local_env(override=True)
    engine = get_engine(database_url)
    tables = reflect_prediction_tables(engine)
    model_versions_table = tables.get("model_versions")
    if model_versions_table is None:
        return {"activated": False, "error": "model_versions table is not available."}

    target_record = load_model_version_record(version_key, database_url=database_url)
    if not target_record:
        return {"activated": False, "error": f"Model version `{version_key}` not found."}

    target_family = str(target_record.get("model_family") or "").strip().lower()
    if target_family not in production_supported_model_families():
        return {
            "activated": False,
            "error": f"Model family `{target_family}` is not enabled for production inference.",
        }
    artifact_path = str(target_record.get("artifact_path") or "").strip()
    if target_family != DEFAULT_POISSON_MODEL_FAMILY and (not artifact_path or not Path(artifact_path).exists()):
        return {
            "activated": False,
            "error": f"Model artifact is not available locally: {artifact_path or 'missing artifact_path'}.",
        }

    set_active_model_version_record(
        engine,
        model_versions_table,
        active_version_key=version_key,
        activation_mode=activation_mode,
    )
    return {"activated": True, "version_key": version_key, "model_family": target_family}


def rollback_active_model_to_poisson(*, database_url: str | None = None) -> dict[str, Any]:
    load_local_env(override=True)
    try:
        df = read_dataframe(
            """
            select version_key
            from public.model_versions
            where lower(model_family) = 'poisson'
            order by coalesce(activated_at, trained_at) desc
            limit 1
            """,
            database_url=database_url,
        )
    except Exception as exc:
        return {"activated": False, "error": str(exc)}

    if df.empty:
        return {"activated": False, "error": "No Poisson model version is available for rollback."}

    version_key = str(df.iloc[0]["version_key"])
    result = activate_model_version_by_key(version_key, database_url=database_url, activation_mode="manual")
    result["rollback_to_poisson"] = True
    return result
