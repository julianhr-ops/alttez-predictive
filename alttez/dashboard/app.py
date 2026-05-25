import pandas as pd
import streamlit as st

from alttez.experiments.tournament_oracle import build_probability_matrix, optimize_prediction
from alttez.features.poisson import PoissonEngine
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import ingest_live_matches_for_date, list_matches_for_date


st.set_page_config(
    page_title="ALTTEZ | Predictive Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #11161f 0%, #0d1219 100%);
        color: #e6edf5;
    }
    [data-testid="stSidebar"] {
        background: #131a23;
        border-right: 1px solid rgba(160, 176, 198, 0.12);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .alttez-panel {
        background: rgba(22, 29, 40, 0.92);
        border: 1px solid rgba(160, 176, 198, 0.12);
        border-radius: 8px;
        padding: 1rem 1.1rem;
    }
    .alttez-kicker {
        color: #7d8ea3;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .alttez-headline {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0.2rem 0 0.4rem 0;
        color: #edf3fb;
    }
    .alttez-subtle {
        color: #9eb0c5;
        font-size: 0.98rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(22, 29, 40, 0.92);
        border: 1px solid rgba(160, 176, 198, 0.12);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        background: #2c7be5;
        color: #f5f8fc;
        border: 0;
        padding: 0.7rem 0.95rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #2468bf;
        color: #f5f8fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86400)
def cargar_datos() -> pd.DataFrame:
    loader = InternationalLoader()
    return loader.fetch_data()


@st.cache_data(ttl=300)
def cargar_agenda(fecha: str) -> pd.DataFrame:
    return list_matches_for_date(pd.to_datetime(fecha).date(), include_statuses=("scheduled", "in_progress", "finished"))


def _get_team_profile(engine: PoissonEngine, team: str) -> dict:
    return engine._resolve_stats(team)


def _build_scoreline_table(prob_matrix, top_n: int = 8) -> pd.DataFrame:
    rows = []
    max_goals = prob_matrix.shape[0] - 1
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            rows.append(
                {
                    "Marcador probable": f"{home_goals} - {away_goals}",
                    "Probabilidad": prob_matrix[home_goals, away_goals],
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(by="Probabilidad", ascending=False)
        .head(top_n)
        .assign(Probabilidad=lambda frame: frame["Probabilidad"].map(lambda value: f"{value:.1%}"))
        .reset_index(drop=True)
    )


def _build_result_summary(prob_matrix, team_a: str, team_b: str) -> pd.DataFrame:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    max_goals = prob_matrix.shape[0] - 1

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = prob_matrix[home_goals, away_goals]
            if home_goals > away_goals:
                home_win += probability
            elif home_goals < away_goals:
                away_win += probability
            else:
                draw += probability

    return pd.DataFrame(
        {
            "Mercado": [f"Victoria {team_a}", "Empate", f"Victoria {team_b}"],
            "Probabilidad": [home_win, draw, away_win],
        }
    ).assign(Probabilidad=lambda frame: frame["Probabilidad"].map(lambda value: f"{value:.1%}"))


def _run_analysis(
    df: pd.DataFrame,
    fecha_simulacion: str,
    ventana: int,
    prior_matches: int,
    equipo_local: str,
    equipo_visitante: str,
    is_neutral: bool,
    is_knockout: bool,
    max_goles: int,
    match_context: dict | None = None,
) -> dict:
    engine = PoissonEngine(years_window=ventana, prior_matches=prior_matches, min_history_matches=5)
    engine.fit(df, target_date=fecha_simulacion)

    lambda_h, lambda_a = engine.predict_xg(equipo_local, equipo_visitante, is_neutral=is_neutral)
    prob_matrix = build_probability_matrix(lambda_h, lambda_a, max_goals=max_goles)

    recomendaciones = optimize_prediction(prob_matrix, is_knockout=is_knockout).head(5).copy()
    recomendaciones["EV"] = recomendaciones["EV"].map(lambda value: round(value, 3))

    profile_local = _get_team_profile(engine, equipo_local)
    profile_visitante = _get_team_profile(engine, equipo_visitante)

    strength_rows = [
        {
            "Equipo": equipo_local,
            "Ataque": round(profile_local["attack"], 3),
            "Defensa": round(profile_local["defense"], 3),
            "Local ataque": round(profile_local["home_attack"], 3),
            "Local defensa": round(profile_local["home_defense"], 3),
        },
        {
            "Equipo": equipo_visitante,
            "Ataque": round(profile_visitante["attack"], 3),
            "Defensa": round(profile_visitante["defense"], 3),
            "Visitante ataque": round(profile_visitante["away_attack"], 3),
            "Visitante defensa": round(profile_visitante["away_defense"], 3),
        },
    ]

    return {
        "equipo_local": equipo_local,
        "equipo_visitante": equipo_visitante,
        "fase_torneo": "Eliminacion Directa" if is_knockout else "Fase de Grupos",
        "is_neutral": is_neutral,
        "fecha_simulacion": fecha_simulacion,
        "ventana": ventana,
        "prior_matches": prior_matches,
        "max_goles": max_goles,
        "lambda_h": lambda_h,
        "lambda_a": lambda_a,
        "matches_local": int(profile_local["matches"]),
        "matches_visitante": int(profile_visitante["matches"]),
        "recomendaciones": recomendaciones,
        "scorelines": _build_scoreline_table(prob_matrix, top_n=8),
        "result_summary": _build_result_summary(prob_matrix, equipo_local, equipo_visitante),
        "strength_df": pd.DataFrame(strength_rows),
        "match_context": match_context or {},
    }


def _format_kickoff(value) -> str:
    if pd.isna(value) or value in {None, ""}:
        return "--:--"
    kickoff = pd.to_datetime(value)
    return kickoff.strftime("%H:%M")


def _build_fixture_button_label(row: pd.Series) -> str:
    kickoff_label = _format_kickoff(row.get("kickoff_at"))
    tournament = row.get("tournament") or "Sin torneo"
    return f"{kickoff_label} | {row['home_team']} vs {row['away_team']} | {tournament}"


def _select_match_from_agenda(
    agenda_df: pd.DataFrame,
    df_historico: pd.DataFrame,
    fecha_simulacion,
    ventana: int,
    prior_matches: int,
    fase_torneo: str,
    max_goles: int,
) -> None:
    for row in agenda_df.itertuples(index=False):
        row_dict = row._asdict()
        if st.button(_build_fixture_button_label(pd.Series(row_dict)), key=f"fixture-{row_dict['id']}"):
            st.session_state["selected_match_id"] = row_dict["id"]
            st.session_state["selected_match"] = row_dict
            st.session_state["analysis_result"] = _run_analysis(
                df=df_historico,
                fecha_simulacion=str(fecha_simulacion),
                ventana=ventana,
                prior_matches=prior_matches,
                equipo_local=row_dict["home_team"],
                equipo_visitante=row_dict["away_team"],
                is_neutral=bool(row_dict.get("is_neutral", False)),
                is_knockout=fase_torneo == "Eliminacion Directa",
                max_goles=max_goles,
                match_context=row_dict,
            )
            st.session_state.pop("analysis_error", None)


def main() -> None:
    st.markdown('<div class="alttez-kicker">Predictive Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="alttez-headline">ALTTEZ Agenda y Oraculo del Dia</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="alttez-subtle">Calendario real de partidos, sincronizacion en vivo y analisis automatico con Poisson y valor esperado.</div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Sincronizando base historica internacional..."):
            df_historico = cargar_datos()
    except RuntimeError as exc:
        st.error(str(exc))
        st.caption("Si ya tienes una copia del CSV, puedes guardarla en `storage/network/international_results.csv`.")
        return

    default_date = pd.Timestamp.today().date()

    with st.sidebar:
        st.header("Parametros")
        fecha_simulacion = st.date_input("Fecha de agenda", value=default_date)
        fase_torneo = st.radio("Formato de puntuacion", ["Fase de Grupos", "Eliminacion Directa"])
        ventana = st.slider("Ventana de analisis (anos)", min_value=1, max_value=10, value=4)
        prior_matches = st.slider("Suavizado por muestra", min_value=0, max_value=20, value=8)
        max_goles = st.slider("Maximo de goles simulados", min_value=5, max_value=10, value=7)

        if st.button("Sincronizar agenda del dia", key="sync-live-fixtures"):
            try:
                with st.spinner("Descargando agenda real desde API-Football..."):
                    result = ingest_live_matches_for_date(pd.to_datetime(fecha_simulacion).date())
                cargar_agenda.clear()
                st.session_state["sync_result"] = result
                st.session_state.pop("analysis_error", None)
            except Exception as exc:  # noqa: BLE001 - surface ingestion errors to the operator.
                st.session_state["analysis_error"] = f"No se pudo sincronizar la agenda: {exc}"

        sync_result = st.session_state.get("sync_result")
        if sync_result:
            st.caption(
                f"Ultima sincronizacion: {sync_result['scheduled_matches']} partidos programados, {sync_result['raw_records']} payloads raw."
            )

    try:
        agenda_df = cargar_agenda(str(fecha_simulacion))
    except Exception as exc:  # noqa: BLE001 - preserve UI even if database query fails.
        st.error(f"No se pudo leer la agenda diaria desde Supabase: {exc}")
        return

    summary_col, context_col = st.columns([1.2, 1], gap="large")

    with summary_col:
        st.markdown('<div class="alttez-panel">', unsafe_allow_html=True)
        st.caption("Agenda del dia")
        st.write(f"Fecha objetivo: `{fecha_simulacion}`")
        st.write(f"Partidos encontrados: `{len(agenda_df)}`")
        st.write(f"Formato: `{fase_torneo}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with context_col:
        st.markdown('<div class="alttez-panel">', unsafe_allow_html=True)
        st.caption("Blindaje del motor")
        st.write(f"Ventana reciente: `{ventana}` anos")
        st.write(f"Suavizado: `{prior_matches}`")
        st.write("Fallback equipos nuevos: `fuerza neutral si historial < 5 partidos`")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("analysis_error"):
        st.warning(st.session_state["analysis_error"])

    st.subheader("Partidos reales del dia")
    if agenda_df.empty:
        st.info("No hay partidos cargados para esta fecha. Sincroniza la agenda o cambia la fecha.")
    else:
        agenda_preview = agenda_df.copy()
        if "kickoff_at" in agenda_preview.columns:
            agenda_preview["kickoff_at"] = agenda_preview["kickoff_at"].map(_format_kickoff)
        st.dataframe(
            agenda_preview[["kickoff_at", "home_team", "away_team", "tournament", "status", "city"]],
            use_container_width=True,
            hide_index=True,
        )
        _select_match_from_agenda(
            agenda_df=agenda_df,
            df_historico=df_historico,
            fecha_simulacion=fecha_simulacion,
            ventana=ventana,
            prior_matches=prior_matches,
            fase_torneo=fase_torneo,
            max_goles=max_goles,
        )

    result = st.session_state.get("analysis_result")
    if result is None:
        st.info("Haz clic en un partido de la agenda para lanzar el analisis automatico.")
        return

    selected_match = result.get("match_context", {})
    selected_col, venue_col = st.columns([1.5, 1], gap="large")

    with selected_col:
        st.markdown('<div class="alttez-panel">', unsafe_allow_html=True)
        st.caption("Partido seleccionado")
        st.write(f"`{result['equipo_local']}` vs `{result['equipo_visitante']}`")
        st.write(f"Torneo: `{selected_match.get('tournament', 'Sin torneo')}`")
        st.write(f"Estado: `{selected_match.get('status', 'scheduled')}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with venue_col:
        st.markdown('<div class="alttez-panel">', unsafe_allow_html=True)
        st.caption("Contexto del encuentro")
        st.write(f"Hora: `{_format_kickoff(selected_match.get('kickoff_at'))}`")
        st.write(f"Ciudad: `{selected_match.get('city') or 'N/D'}`")
        st.write(f"Escenario: `{'Neutral' if result['is_neutral'] else 'Local/Visitante'}`")
        st.markdown("</div>", unsafe_allow_html=True)

    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric(f"xG esperado {result['equipo_local']}", f"{result['lambda_h']:.2f}")
    metric_cols[1].metric(f"xG esperado {result['equipo_visitante']}", f"{result['lambda_a']:.2f}")
    metric_cols[2].metric(f"Partidos {result['equipo_local']}", result["matches_local"])
    metric_cols[3].metric(f"Partidos {result['equipo_visitante']}", result["matches_visitante"])

    top_col, side_col = st.columns([1.45, 1], gap="large")

    with top_col:
        st.subheader("Pronosticos optimos")
        st.dataframe(result["recomendaciones"], use_container_width=True, hide_index=True)

        st.subheader("Marcadores reales mas probables")
        st.dataframe(result["scorelines"], use_container_width=True, hide_index=True)

    with side_col:
        st.subheader("Mapa del partido")
        st.dataframe(result["result_summary"], use_container_width=True, hide_index=True)

        st.subheader("Fortalezas del motor")
        st.dataframe(result["strength_df"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
