import pandas as pd
import streamlit as st

from alttez.experiments.tournament_oracle import build_probability_matrix, optimize_prediction
from alttez.features.poisson import PoissonEngine
from alttez.ingestion.international import InternationalLoader


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


def _get_team_profile(engine: PoissonEngine, team: str) -> dict:
    return engine.team_stats.get(team, engine._get_default_stats())


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
) -> dict:
    engine = PoissonEngine(years_window=ventana, prior_matches=prior_matches)
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
    }


def main() -> None:
    st.markdown('<div class="alttez-kicker">Predictive Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="alttez-headline">ALTTEZ Oraculo de Torneos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="alttez-subtle">Poisson reciente, suavizado por muestra y optimizacion de valor esperado sobre el reglamento del torneo.</div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Sincronizando base historica internacional..."):
            df = cargar_datos()
    except RuntimeError as exc:
        st.error(str(exc))
        st.caption("Si ya tienes una copia del CSV, puedes guardarla en `storage/network/international_results.csv`.")
        return

    equipos_disponibles = sorted(set(df["home_team"]).union(set(df["away_team"])))
    default_date = pd.Timestamp.today().date()

    with st.sidebar:
        st.header("Parametros")
        with st.form("analysis_form"):
            fase_torneo = st.radio("Formato de puntuacion", ["Fase de Grupos", "Eliminacion Directa"])
            is_neutral = st.toggle("Cancha neutral", value=True)
            ventana = st.slider("Ventana de analisis (anos)", min_value=1, max_value=10, value=4)
            prior_matches = st.slider("Suavizado por muestra", min_value=0, max_value=20, value=8)
            fecha_simulacion = st.date_input("Fecha de simulacion", value=default_date)
            max_goles = st.slider("Maximo de goles simulados", min_value=5, max_value=10, value=7)

            st.caption(
                "La app recalcula fortalezas con la historia anterior a la fecha elegida y optimiza el pronostico que maximiza puntos esperados."
            )

            equipo_local = st.selectbox(
                "Equipo A",
                equipos_disponibles,
                index=equipos_disponibles.index("Colombia") if "Colombia" in equipos_disponibles else 0,
            )
            equipo_visitante = st.selectbox(
                "Equipo B",
                equipos_disponibles,
                index=equipos_disponibles.index("Brazil") if "Brazil" in equipos_disponibles else 1,
            )
            ejecutar = st.form_submit_button("Ejecutar analisis de valor esperado")

    if ejecutar:
        if equipo_local == equipo_visitante:
            st.session_state["analysis_error"] = "Elige dos selecciones distintas para correr el oraculo."
            st.session_state.pop("analysis_result", None)
        else:
            try:
                st.session_state["analysis_result"] = _run_analysis(
                    df=df,
                    fecha_simulacion=str(fecha_simulacion),
                    ventana=ventana,
                    prior_matches=prior_matches,
                    equipo_local=equipo_local,
                    equipo_visitante=equipo_visitante,
                    is_neutral=is_neutral,
                    is_knockout=fase_torneo == "Eliminacion Directa",
                    max_goles=max_goles,
                )
                st.session_state.pop("analysis_error", None)
            except ValueError as exc:
                st.session_state["analysis_error"] = str(exc)
                st.session_state.pop("analysis_result", None)

    selector_col, summary_col = st.columns([1.5, 1], gap="large")

    with selector_col:
        st.markdown('<div class="alttez-panel">', unsafe_allow_html=True)
        st.caption("Cruce configurado desde el formulario lateral")
        st.write(f"Equipo A: `{equipo_local}`")
        st.write(f"Equipo B: `{equipo_visitante}`")
        st.write(f"Suavizado: `{prior_matches}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with summary_col:
        st.markdown('<div class="alttez-panel">', unsafe_allow_html=True)
        st.caption("Contexto de simulacion")
        st.write(f"Fecha objetivo: `{fecha_simulacion}`")
        st.write(f"Ventana reciente: `{ventana}` anos")
        st.write(f"Formato: `{fase_torneo}`")
        st.write(f"Escenario: `{'Neutral' if is_neutral else 'Local/Visitante'}`")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("analysis_error"):
        st.warning(st.session_state["analysis_error"])
        return

    result = st.session_state.get("analysis_result")
    if result is None:
        st.info("Selecciona un cruce y ejecuta el analisis para ver xG, probabilidades y pronosticos optimos.")
        return

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
