"""Arnes de evaluacion de calibracion (Fase 0).

Lee las predicciones 1X2 cerradas (match_predictions x prediction_evaluations) y
reporta, por modelo y global: muestras, Brier, log loss, accuracy 1X2 y ECE.
Opcionalmente aplica la calibracion activa para mostrar el "antes/despues".

Es la base de medicion contra la que se compara cualquier cambio de calibracion.
No escribe nada; solo mide.

    python -m scripts.evaluate_calibration --lookback-days 365
    python -m scripts.evaluate_calibration --lookback-days 365 --with-calibration
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from alttez.api.db import read_dataframe
from alttez.api.settings import load_local_env
from alttez.inference.calibration import (
    expected_calibration_error,
    load_active_calibration,
    log_loss,
)
from alttez.operations.prediction_memory import select_latest_pre_kickoff

LOGGER = logging.getLogger(__name__)


RAW_COLUMNS_SQL = """
            mp.raw_predicted_home_win_prob as raw_prob_home,
            mp.raw_predicted_draw_prob as raw_prob_draw,
            mp.raw_predicted_away_win_prob as raw_prob_away,"""
LEGACY_RAW_COLUMNS_SQL = """
            null::numeric as raw_prob_home,
            null::numeric as raw_prob_draw,
            null::numeric as raw_prob_away,"""


def closed_predictions_query(*, include_raw: bool) -> str:
    raw_columns = RAW_COLUMNS_SQL if include_raw else LEGACY_RAW_COLUMNS_SQL
    return f"""
        select
            mp.match_id,
            mp.model_version,
            mp.prediction_created_at,
            mp.kickoff_at,
            mp.predicted_home_win_prob as prob_home,
            mp.predicted_draw_prob as prob_draw,
            mp.predicted_away_win_prob as prob_away,{raw_columns}
            pe.actual_outcome
        from public.prediction_evaluations as pe
        join public.match_predictions as mp on mp.id = pe.prediction_id
        where mp.match_date >= :lookback_start
          and pe.actual_outcome is not null
        """


def load_closed_predictions(*, database_url: str | None, lookback_days: int) -> pd.DataFrame:
    """Predicciones cerradas, una por (partido, model_version), con probabilidades servidas y crudas.

    Si la migracion 014 aun no esta aplicada las columnas crudas no existen: se
    avisa y se carga la consulta legacy con las columnas crudas en NULL.
    """
    params = {"lookback_start": (datetime.now(timezone.utc).date() - pd.Timedelta(days=lookback_days)).isoformat()}
    try:
        frame = read_dataframe(closed_predictions_query(include_raw=True), params=params, database_url=database_url)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "No se pudieron leer las probabilidades crudas (%s); aplica la migracion 014. "
            "Se usan solo las probabilidades almacenadas (ya calibradas).",
            exc,
        )
        frame = read_dataframe(closed_predictions_query(include_raw=False), params=params, database_url=database_url)
    return select_latest_pre_kickoff(frame)


def _brier(triplets: np.ndarray, outcomes: list[str]) -> float:
    index = {"H": 0, "D": 1, "A": 2}
    errors = []
    for row, outcome in zip(triplets, outcomes):
        target = np.zeros(3)
        idx = index.get(outcome)
        if idx is None:
            continue
        target[idx] = 1.0
        norm = row / row.sum() if row.sum() > 0 else row
        errors.append(float(np.sum((norm - target) ** 2)))
    return float(np.mean(errors)) if errors else 0.0


def _accuracy(triplets: np.ndarray, outcomes: list[str]) -> float:
    index = {"H": 0, "D": 1, "A": 2}
    hits = [
        1.0 if int(np.argmax(row)) == index.get(outcome, -1) else 0.0
        for row, outcome in zip(triplets, outcomes)
    ]
    return float(np.mean(hits)) if hits else 0.0


def compute_metrics(frame: pd.DataFrame, calibration=None) -> dict[str, float]:
    """Metricas sobre un frame de predicciones; calibra antes si se entrega calibracion."""
    triplets = frame[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    if calibration is not None and not calibration.is_neutral:
        triplets = np.array([calibration.calibrate_triplet(*row) for row in triplets])
    outcomes = frame["actual_outcome"].tolist()
    return {
        "samples": int(len(frame)),
        "brier": round(_brier(triplets, outcomes), 4),
        "log_loss": round(log_loss(triplets, outcomes), 4),
        "accuracy_1x2": round(_accuracy(triplets, outcomes), 4),
        "ece": round(expected_calibration_error(triplets, outcomes), 4),
    }


def _print_block(title: str, metrics: dict[str, float]) -> None:
    print(
        f"{title:28} n={metrics['samples']:>4}  "
        f"brier={metrics['brier']:.4f}  logloss={metrics['log_loss']:.4f}  "
        f"acc={metrics['accuracy_1x2']:.4f}  ece={metrics['ece']:.4f}"
    )


def main() -> int:
    load_local_env(override=True)
    parser = argparse.ArgumentParser(description="Evalua la calibracion del modelo.")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--with-calibration", action="store_true", help="Aplica la calibracion activa (antes/despues).")
    args = parser.parse_args()

    frame = load_closed_predictions(database_url=args.database_url, lookback_days=args.lookback_days)
    if frame.empty:
        print("Sin predicciones cerradas en la ventana.")
        return 0

    print(f"=== Evaluacion CRUDA (ultimos {args.lookback_days} dias) ===")
    _print_block("GLOBAL", compute_metrics(frame))
    for model_version, group in frame.groupby("model_version"):
        _print_block(str(model_version), compute_metrics(group))

    if args.with_calibration:
        calibration = load_active_calibration()
        print(f"\n=== Evaluacion CALIBRADA (T={calibration.temperature}, piso={calibration.class_floor}) ===")
        if calibration.is_neutral:
            print("(calibracion neutra: no hay artefacto activo; corre fit_probability_calibration.py)")
        _print_block("GLOBAL", compute_metrics(frame, calibration))
        for model_version, group in frame.groupby("model_version"):
            _print_block(str(model_version), compute_metrics(group, calibration))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
