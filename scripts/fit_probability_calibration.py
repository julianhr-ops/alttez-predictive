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
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from alttez.api.settings import load_local_env
from alttez.inference.calibration import (
    DEFAULT_CALIBRATION_PATH,
    ProbabilityCalibration,
    expected_calibration_error,
    fit_isotonic,
    fit_temperature,
    log_loss,
)
from scripts.evaluate_calibration import load_closed_predictions

RAW_PROBABILITY_COLUMNS = ["raw_prob_home", "raw_prob_draw", "raw_prob_away"]
SERVED_PROBABILITY_COLUMNS = ["prob_home", "prob_draw", "prob_away"]


def select_fit_triplets(frame: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """Tripletes de ajuste: la probabilidad cruda cuando existe, la almacenada solo en filas legacy.

    Devuelve (tripletes, filas_con_cruda, filas_legacy).
    """
    served = frame[SERVED_PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not set(RAW_PROBABILITY_COLUMNS).issubset(frame.columns):
        return served, 0, len(frame)
    raw = frame[RAW_PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    has_raw = ~np.isnan(raw).any(axis=1)
    return np.where(has_raw[:, None], raw, served), int(has_raw.sum()), int((~has_raw).sum())


def _metrics(triplets: np.ndarray, outcomes: list[str]) -> tuple[float, float, float]:
    """Devuelve (brier, log_loss, ece) para un set de tripletes vs resultados."""
    index = {"H": 0, "D": 1, "A": 2}
    probs = np.asarray(triplets, dtype=float)
    target = np.zeros_like(probs)
    valid = 0
    for i, outcome in enumerate(outcomes):
        idx = index.get(outcome)
        if idx is None:
            continue
        target[i, idx] = 1.0
        valid += 1
    brier = float(np.mean(np.sum((probs - target) ** 2, axis=1))) if valid else 0.0
    return brier, log_loss(probs, outcomes), expected_calibration_error(probs, outcomes)


def _split_holdout(triplets: np.ndarray, outcomes: list[str], frac: float, seed: int):
    """Particion train/validation reproducible. Devuelve (tr_X, tr_y, va_X, va_y)."""
    n = len(triplets)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = max(1, int(round(n * (1.0 - frac))))
    tr_idx, va_idx = order[:cut], order[cut:]
    if va_idx.size == 0:  # dataset minusculo: valida sobre train.
        va_idx = tr_idx
    tr_y = [outcomes[i] for i in tr_idx]
    va_y = [outcomes[i] for i in va_idx]
    return triplets[tr_idx], tr_y, triplets[va_idx], va_y


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
    parser.add_argument(
        "--method",
        choices=["temperature", "isotonic"],
        default="temperature",
        help="Metodo de calibracion. temperature preserva el comportamiento previo; "
        "isotonic ajusta one-vs-rest y compara contra temperature en validacion.",
    )
    parser.add_argument("--validation-frac", type=float, default=0.3, help="Fraccion de holdout de validacion.")
    parser.add_argument("--seed", type=int, default=0, help="Semilla de la particion train/validacion.")
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

    triplets, raw_rows, legacy_rows = select_fit_triplets(frame)
    outcomes = frame["actual_outcome"].tolist()
    print(f"filas de ajuste: {raw_rows} con probabilidad cruda, {legacy_rows} legacy sin cruda")
    if legacy_rows:
        print(
            "AVISO: las filas legacy usan la probabilidad almacenada, que ya paso por la calibracion "
            "vigente de su dia; ajustar sobre ellas compone calibraciones. Aplica la migracion 014 y "
            "deja acumular filas crudas."
        )

    if args.method == "temperature":
        payload = _fit_temperature_payload(triplets, outcomes, args)
    else:
        payload = _fit_isotonic_payload(triplets, outcomes, args)
    payload["raw_probability_samples"] = raw_rows
    payload["legacy_calibrated_samples"] = legacy_rows

    if args.apply:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Artefacto escrito: {output}")
    else:
        print("(dry-run; usa --apply para persistir)")
        print(json.dumps(payload, indent=2))
    return 0


def _fit_temperature_payload(triplets: np.ndarray, outcomes: list[str], args) -> dict:
    temperature = fit_temperature(triplets, outcomes)
    calibration = ProbabilityCalibration(temperature=temperature, class_floor=args.class_floor)

    before_ll = log_loss(triplets, outcomes)
    before_ece = expected_calibration_error(triplets, outcomes)
    calibrated = np.array([calibration.calibrate_triplet(*row) for row in triplets])
    after_ll = log_loss(calibrated, outcomes)
    after_ece = expected_calibration_error(calibrated, outcomes)

    print(f"metodo=temperature  muestras={len(outcomes)}  T={temperature}  piso={args.class_floor}")
    print(f"log_loss: {before_ll:.4f} -> {after_ll:.4f}")
    print(f"ece:      {before_ece:.4f} -> {after_ece:.4f}")

    return {
        "method": "temperature",
        "temperature": temperature,
        "class_floor": args.class_floor,
        "fitted_on_model": args.model_version or "global",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.lookback_days,
        "samples": int(len(outcomes)),
        "log_loss_before": round(before_ll, 4),
        "log_loss_after": round(after_ll, 4),
        "ece_before": round(before_ece, 4),
        "ece_after": round(after_ece, 4),
    }


def _fit_isotonic_payload(triplets: np.ndarray, outcomes: list[str], args) -> dict:
    """Ajusta isotonic vs temperature en una particion de validacion y elige el mejor.

    El ganador se decide por log_loss en validacion (no en train, para no premiar
    el sobreajuste). El artefacto resultante usa el esquema versionado del metodo
    elegido.
    """
    tr_X, tr_y, va_X, va_y = _split_holdout(triplets, outcomes, args.validation_frac, args.seed)

    iso = fit_isotonic(tr_X, tr_y, class_floor=args.class_floor)
    temp = fit_temperature(tr_X, tr_y)
    temp_cal = ProbabilityCalibration(temperature=temp, class_floor=args.class_floor)

    raw_va = np.array([(h, d, a) for h, d, a in va_X])
    iso_va = np.array([iso.calibrate_triplet(*row) for row in va_X])
    temp_va = np.array([temp_cal.calibrate_triplet(*row) for row in va_X])

    raw_b, raw_ll, raw_ece = _metrics(raw_va, va_y)
    iso_b, iso_ll, iso_ece = _metrics(iso_va, va_y)
    temp_b, temp_ll, temp_ece = _metrics(temp_va, va_y)

    print(f"metodo=isotonic  train={len(tr_y)}  validacion={len(va_y)}  piso={args.class_floor}")
    print("                 brier    logloss   ece")
    print(f"crudo          : {raw_b:.4f}  {raw_ll:.4f}  {raw_ece:.4f}")
    print(f"temperature(T={temp}): {temp_b:.4f}  {temp_ll:.4f}  {temp_ece:.4f}")
    print(f"isotonic       : {iso_b:.4f}  {iso_ll:.4f}  {iso_ece:.4f}")

    chosen = "isotonic" if iso_ll <= temp_ll else "temperature"
    print(f"elegido (menor logloss en validacion): {chosen}")

    meta = {
        "fitted_on_model": args.model_version or "global",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.lookback_days,
        "samples": int(len(outcomes)),
        "validation_samples": int(len(va_y)),
        "chosen_method": chosen,
        "validation_logloss_temperature": round(temp_ll, 4),
        "validation_logloss_isotonic": round(iso_ll, 4),
        "validation_brier_temperature": round(temp_b, 4),
        "validation_brier_isotonic": round(iso_b, 4),
        "validation_ece_temperature": round(temp_ece, 4),
        "validation_ece_isotonic": round(iso_ece, 4),
    }

    if chosen == "isotonic":
        # Reajusta sobre TODO el set para el artefacto final (mas senal).
        full_iso = fit_isotonic(triplets, outcomes, class_floor=args.class_floor)
        payload = full_iso.to_payload()
    else:
        full_temp = fit_temperature(triplets, outcomes)
        payload = {
            "method": "temperature",
            "temperature": full_temp,
            "class_floor": args.class_floor,
        }
    payload.update(meta)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
