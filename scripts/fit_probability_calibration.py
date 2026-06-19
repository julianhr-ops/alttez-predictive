"""Ajusta la calibracion de probabilidad (Fase 1) y la persiste.

Toma las predicciones 1X2 cerradas, ajusta la temperatura que minimiza el log
loss y escribe artifacts/calibration/active.json. La inferencia carga ese
artefacto via load_active_calibration; si no existe, usa calibracion neutra.

    python -m scripts.fit_probability_calibration --lookback-days 365 --apply
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from alttez.api.settings import load_local_env
from alttez.inference.calibration import (
    DEFAULT_CALIBRATION_PATH,
    ProbabilityCalibration,
    expected_calibration_error,
    fit_temperature,
    log_loss,
)
from scripts.evaluate_calibration import load_closed_predictions


def main() -> int:
    load_local_env(override=True)
    parser = argparse.ArgumentParser(description="Ajusta y persiste la calibracion de probabilidad.")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--model-version",
        default="committee-consensus-v1",
        help="Ajusta sobre este modelo (la ruta de produccion). Vacio = global.",
    )
    parser.add_argument("--class-floor", type=float, default=0.0, help="Piso por clase en inferencia (T>1 ya aleja del 0).")
    parser.add_argument("--output", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--apply", action="store_true", help="Escribe el artefacto. Sin esto solo reporta.")
    args = parser.parse_args()

    frame = load_closed_predictions(database_url=args.database_url, lookback_days=args.lookback_days)
    if args.model_version:
        frame = frame[frame["model_version"] == args.model_version]
    if frame.empty:
        print("Sin predicciones cerradas; no se puede ajustar.")
        return 1

    triplets = frame[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    outcomes = frame["actual_outcome"].tolist()

    temperature = fit_temperature(triplets, outcomes)
    calibration = ProbabilityCalibration(temperature=temperature, class_floor=args.class_floor)

    before_ll = log_loss(triplets, outcomes)
    before_ece = expected_calibration_error(triplets, outcomes)
    calibrated = np.array([calibration.calibrate_triplet(*row) for row in triplets])
    after_ll = log_loss(calibrated, outcomes)
    after_ece = expected_calibration_error(calibrated, outcomes)

    print(f"muestras={len(frame)}  T={temperature}  piso={args.class_floor}")
    print(f"log_loss: {before_ll:.4f} -> {after_ll:.4f}")
    print(f"ece:      {before_ece:.4f} -> {after_ece:.4f}")

    payload = {
        "temperature": temperature,
        "class_floor": args.class_floor,
        "fitted_on_model": args.model_version or "global",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.lookback_days,
        "samples": int(len(frame)),
        "log_loss_before": round(before_ll, 4),
        "log_loss_after": round(after_ll, 4),
        "ece_before": round(before_ece, 4),
        "ece_after": round(after_ece, 4),
    }

    if args.apply:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Artefacto escrito: {output}")
    else:
        print("(dry-run; usa --apply para persistir)")
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
