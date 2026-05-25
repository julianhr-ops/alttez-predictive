import logging
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from alttez.api.db import read_dataframe
from alttez.api.settings import load_local_env


logging.basicConfig(level=logging.INFO)


class InternationalLoader:
    """Descarga y formatea el dataset de partidos internacionales historicos."""

    DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    CACHE_PATH = Path("storage/network/international_results.csv")
    COMPATIBILITY_VIEW = "public.international_results_compat"
    REQUEST_TIMEOUT = 20
    DATA_SOURCE_ENV = "ALTTEZ_DATA_SOURCE"
    VIEW_ENV = "ALTTEZ_SUPABASE_VIEW"
    ALLOW_LOCAL_FALLBACK_ENV = "ALTTEZ_ALLOW_LOCAL_FALLBACK"

    def fetch_data(self) -> pd.DataFrame:
        load_local_env()

        data_source = os.getenv(self.DATA_SOURCE_ENV, "auto").strip().lower()
        allow_local_fallback = os.getenv(self.ALLOW_LOCAL_FALLBACK_ENV, "true").strip().lower() != "false"

        if data_source not in {"auto", "supabase", "csv"}:
            raise RuntimeError(
                f'Unsupported data source "{data_source}". Use one of: auto, supabase, csv.'
            )

        if data_source in {"auto", "supabase"}:
            try:
                df = self._fetch_from_supabase()
                logging.info("Dataset cargado desde Supabase.")
                return self._finalize_dataframe(df)
            except Exception as exc:  # noqa: BLE001 - we want to preserve uptime with a local fallback.
                if data_source == "supabase" and not allow_local_fallback:
                    raise RuntimeError(
                        "No se pudo cargar el dataset desde Supabase y el fallback local esta deshabilitado."
                    ) from exc
                logging.warning("Fallo la carga desde Supabase (%s). Intentando fallback local.", exc)

        try:
            df = self._load_local_or_remote_csv()
        except requests.RequestException as exc:
            if self.CACHE_PATH.exists():
                logging.warning(
                    "No fue posible descargar el dataset remoto (%s). Usando cache local: %s",
                    exc,
                    self.CACHE_PATH,
                )
                df = pd.read_csv(self.CACHE_PATH)
            else:
                raise RuntimeError(
                    "No se pudo cargar el dataset desde Supabase ni descargar el CSV remoto, y no existe cache local en "
                    f"{self.CACHE_PATH}. Verifica la conexion o ejecuta el seed historico."
                ) from exc

        return self._finalize_dataframe(df)

    def _fetch_from_supabase(self) -> pd.DataFrame:
        view_name = os.getenv(self.VIEW_ENV, self.COMPATIBILITY_VIEW)
        sql = f"""
            select
                date,
                home_team,
                away_team,
                home_score,
                away_score,
                tournament,
                city,
                country,
                neutral
            from {view_name}
        """
        return read_dataframe(sql)

    def _load_local_or_remote_csv(self) -> pd.DataFrame:
        if self.CACHE_PATH.exists():
            logging.info("Usando cache local en %s.", self.CACHE_PATH)
            return pd.read_csv(self.CACHE_PATH)

        df = self._download_remote_data()
        self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.CACHE_PATH, index=False)
        logging.info("Cache local actualizada en %s.", self.CACHE_PATH)
        return df

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Formateo de fechas y columnas clave.
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Limpiar partidos nulos.
        df = df.dropna(subset=["home_score", "away_score"])

        # Crear columna de resultado real (W = 1 para local, 0.5 empate, 0 derrota).
        df["home_result"] = 0.5
        df.loc[df["home_score"] > df["away_score"], "home_result"] = 1.0
        df.loc[df["home_score"] < df["away_score"], "home_result"] = 0.0

        logging.info("Dataset listo: %s partidos cargados.", len(df))
        return df

    def _download_remote_data(self) -> pd.DataFrame:
        logging.info("Descargando datos desde %s...", self.DATA_URL)
        response = requests.get(
            self.DATA_URL,
            timeout=self.REQUEST_TIMEOUT,
            headers={"User-Agent": "ALTTEZ Predictive Engine"},
        )
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
