"""NBA predictions and their graded, paired market comparison."""

from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, date, datetime

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from alttez.api.db import get_engine
from alttez.nba.store import COLUMNS, DateWindow, NbaPredictionRow, load_graded_summary, load_ungraded, served_rows


def rows_frame(rows: Iterable[NbaPredictionRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows], columns=COLUMNS)


def upcoming_games(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Return scheduled predictions that have not reached tipoff."""
    result = frame.copy()
    if result.empty:
        return result
    tipoff = pd.to_datetime(result["tipoff_utc"], utc=True)
    result = result.loc[tipoff >= pd.Timestamp(now)].copy()
    result["source"] = result["served_model"].map({"blend": "Market blend", "elo_platt": "Elo + Platt"})
    result["home_line"] = -pd.to_numeric(result["market_spread"], errors="coerce")
    return result.sort_values("tipoff_utc")


def graded_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the winning side and the served prediction's hit to graded rows."""
    result = frame.copy()
    if result.empty:
        return result
    result["outcome"] = result["home_win"].map({True: "Home", False: "Away"})
    result["hit"] = (result["p_home_served"] > 0.5) == result["home_win"].astype(bool)
    return result.sort_values("tipoff_utc", ascending=False)


def scoreboard(frame: pd.DataFrame) -> dict:
    """Calculate overall accuracy and paired Brier losses on market games."""
    if frame.empty:
        return {"n": 0, "hit_rate": None, "brier": None, "market_n": 0, "paired_brier": None, "market_brier": None}
    market = frame.loc[frame["p_home_market"].notna() & frame["market_spread"].notna()]
    return {
        "n": len(frame),
        "hit_rate": float(frame["hit"].mean()),
        "brier": float(frame["brier"].mean()),
        "market_n": len(market),
        "paired_brier": float(market["brier"].mean()) if len(market) else None,
        "market_brier": float(((market["p_home_market"] - market["home_win"].astype(float)) ** 2).mean()) if len(market) else None,
    }


def render_nba_view() -> None:
    st.title("NBA")
    try:
        engine = get_engine()
    except RuntimeError:
        st.info("No hay una base de datos configurada para NBA. Consulta la alternativa sin base de datos abajo.")
        st.markdown("[Ver el comando dry-run](#nba-sin-base-de-datos)")
        st.subheader("NBA sin base de datos")
        st.code("python scripts/nba_daily.py --dry-run --html out.html", language="bash")
        return
    today = datetime.now(UTC).date()
    try:
        upcoming = served_rows(load_ungraded(engine, DateWindow(date(2000, 1, 1), date.max)))
        summary = load_graded_summary(engine, DateWindow(date(2000, 1, 1), today))
    except SQLAlchemyError:
        st.error("No se pudieron leer las predicciones NBA. Revisa la conexión y las migraciones 015 y 016.")
        return
    finally:
        engine.dispose()

    scheduled = upcoming_games(rows_frame(upcoming), datetime.now(UTC))
    graded = graded_history(rows_frame(summary.all_rows))
    stats = scoreboard(graded_history(rows_frame(summary.rows)))
    st.subheader("Hoy y próximos partidos")
    if scheduled.empty:
        st.info("No hay predicciones programadas disponibles.")
    else:
        st.caption("Equipos identificados por ID de ESPN. Línea del local con signo de apuesta; horas UTC.")
        st.dataframe(scheduled[["tipoff_utc", "away_team_id", "home_team_id", "p_home_served", "source", "home_line", "market_source", "model_version"]], hide_index=True)
    st.subheader("Marcador acumulado")
    if not stats["n"]:
        st.info("Aún no hay predicciones calificadas.")
    else:
        st.write(f"n={stats['n']} · aciertos {stats['hit_rate']:.1%} · Brier {stats['brier']:.4f}")
        if stats["market_n"]:
            st.write(f"Mismos partidos con línea, n={stats['market_n']} · Brier modelo {stats['paired_brier']:.4f} · mercado {stats['market_brier']:.4f}")
        else:
            st.caption("No hay partidos calificados con línea y probabilidad de mercado.")
        st.subheader("Historial calificado")
        st.dataframe(graded[["game_date", "away_team_id", "home_team_id", "away_score", "home_score", "p_home_served", "served_model", "outcome", "hit", "brier", "model_version"]], hide_index=True)
    st.caption("Cómo leerlo: la probabilidad es de victoria local; un Brier menor es mejor. El marcador cuenta cada partido una vez, con la fila del blend si existe y si no con Elo. El historial muestra todas las filas. Elo no supera la línea de cierre.")
