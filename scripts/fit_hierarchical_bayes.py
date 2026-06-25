"""Ajusta el modelo jerarquico bayesiano (Baio-Blangiardo) y persiste el posterior.

Ejecuta NUTS (4 cadenas, target_accept=0.95) sobre el historico y escribe:
  - artifacts/hierarchical/strengths.json : medias posteriores compactas que CONSUME
    la inferencia (sin dependencia de PyMC en produccion).
  - artifacts/hierarchical/posterior.nc   : InferenceData completo (arviz/NetCDF)
    para diagnostico (r_hat, ESS), opcional.

PyMC SOLO es dependencia de este script (fit-time), no de la inferencia.

Uso:
    venv\\Scripts\\python.exe scripts/fit_hierarchical_bayes.py --draws 1000 --tune 1000

Promocion: queda fuera del path automatico. Integrar como challenger y promover
manualmente via scripts/activate_model_version.py tras validar r_hat/ESS y metricas.
Logs en texto plano (sin unicode) por compatibilidad con Windows.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from alttez.api.settings import load_local_env
from alttez.features.hierarchical import fit_hierarchical_model
from alttez.features.klement import load_enriched_historical_matches

LOGGER = logging.getLogger("fit_hierarchical_bayes")
DEFAULT_STRENGTHS = Path("artifacts/hierarchical/strengths.json")
DEFAULT_POSTERIOR = Path("artifacts/hierarchical/posterior.nc")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ajuste del modelo jerarquico bayesiano.")
    parser.add_argument("--strengths-output", default=str(DEFAULT_STRENGTHS))
    parser.add_argument("--posterior-output", default=str(DEFAULT_POSTERIOR))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=0, help="Limita el numero de partidos (0 = todos).")
    parser.add_argument("--no-posterior", action="store_true", help="No guardar el NetCDF de diagnostico.")
    args = parser.parse_args()

    load_local_env()
    df = load_enriched_historical_matches()
    required = {"home_team", "away_team", "home_score", "away_score", "neutral"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas para el ajuste: {sorted(missing)}")
    if args.samples and args.samples > 0:
        df = df.tail(args.samples).reset_index(drop=True)

    LOGGER.info("Ajustando jerarquico sobre %d partidos (NUTS %d cadenas)...", len(df), args.chains)
    result, idata = fit_hierarchical_model(
        df,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )

    strengths_path = Path(args.strengths_output)
    strengths_path.parent.mkdir(parents=True, exist_ok=True)
    strengths_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info(
        "mu=%.4f gamma=%.4f sigma_att=%.4f sigma_def=%.4f equipos=%d -> %s",
        result["mu"],
        result["gamma"],
        result["sigma_att"],
        result["sigma_def"],
        len(result["teams"]),
        strengths_path,
    )

    if not args.no_posterior:
        try:
            posterior_path = Path(args.posterior_output)
            posterior_path.parent.mkdir(parents=True, exist_ok=True)
            idata.to_netcdf(str(posterior_path))
            LOGGER.info("posterior NetCDF -> %s", posterior_path)
        except Exception as exc:  # noqa: BLE001 - el diagnostico es opcional.
            LOGGER.warning("no se pudo guardar el posterior NetCDF: %s", exc)


if __name__ == "__main__":
    main()
