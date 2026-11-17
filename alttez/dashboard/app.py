from __future__ import annotations

import html
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from alttez.api.settings import load_local_env
from alttez.dashboard.analysis_helpers import (
    build_match_meta as _build_match_meta,
    build_recommendation_summary as _build_recommendation_summary,
    competition_class_label as _competition_class_label,
    filter_noise_matches as _filter_noise_matches,
    safe_key as _safe_key,
    safe_text as _safe_text,
    support_level as _support_level,
    support_status as _support_status,
)
from alttez.dashboard.model_monitoring import render_model_monitoring_dashboard
from alttez.dashboard.theme import build_brand_css
from alttez.inference import build_active_predictor
from alttez.inference.world_cup_2026 import (
    load_latest_world_cup_2026_report,
    load_world_cup_2026_config,
    simulate_world_cup_2026,
    write_world_cup_2026_report,
)
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import get_live_provider_order, ingest_live_matches_for_date, list_matches_for_date
from alttez.ingestion.provider_healthcheck import run_healthcheck
from alttez.operations.alerts import load_recent_alert_events


load_local_env(override=True)


st.set_page_config(
    page_title="ALTTEZ | Predictive Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --alttez-bg-0: #08111a;
        --alttez-bg-1: rgba(9, 20, 34, 0.88);
        --alttez-bg-2: rgba(12, 24, 40, 0.78);
        --alttez-bg-3: rgba(16, 30, 49, 0.92);
        --alttez-line-soft: rgba(109, 137, 182, 0.16);
        --alttez-line-strong: rgba(114, 148, 201, 0.28);
        --alttez-text-0: #f3f7ff;
        --alttez-text-1: #d6e3f7;
        --alttez-text-2: #93abc8;
        --alttez-accent: #52c7ff;
        --alttez-accent-2: #86f0cc;
        --alttez-shadow: 0 24px 70px rgba(2, 8, 18, 0.34);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(52, 122, 255, 0.22), transparent 24%),
            radial-gradient(circle at 85% 8%, rgba(0, 219, 176, 0.12), transparent 18%),
            radial-gradient(circle at 50% 100%, rgba(24, 70, 120, 0.18), transparent 30%),
            linear-gradient(180deg, #07111a 0%, #091521 42%, #08111a 100%);
        color: var(--alttez-text-0);
        font-family: "Segoe UI", Inter, system-ui, sans-serif;
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(9, 18, 31, 0.98), rgba(10, 20, 34, 0.94)),
            radial-gradient(circle at top, rgba(56, 114, 255, 0.08), transparent 30%);
        border-right: 1px solid var(--alttez-line-soft);
    }
    .block-container {
        padding-top: 1.55rem;
        padding-bottom: 2.2rem;
        max-width: 1360px;
    }
    .alttez-shell {
        display: grid;
        gap: 1.4rem;
    }
    .alttez-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--alttez-line-soft);
        border-radius: 22px;
        padding: 1.5rem 1.55rem;
        background:
            linear-gradient(135deg, rgba(10, 24, 42, 0.96), rgba(8, 18, 30, 0.92)),
            linear-gradient(180deg, rgba(43, 117, 255, 0.08), transparent 62%);
        box-shadow: var(--alttez-shadow);
    }
    .alttez-hero::after {
        content: "";
        position: absolute;
        inset: auto -8% -30% 58%;
        height: 280px;
        background: radial-gradient(circle, rgba(82, 199, 255, 0.14), transparent 62%);
        pointer-events: none;
    }
    .alttez-hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.9fr) minmax(320px, 0.95fr);
        gap: 1.15rem;
        align-items: stretch;
    }
    .alttez-kicker {
        color: var(--alttez-text-2);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin-bottom: 0.7rem;
    }
    .alttez-headline {
        font-size: 2.6rem;
        line-height: 0.98;
        font-weight: 730;
        letter-spacing: -0.03em;
        color: var(--alttez-text-0);
        margin: 0;
    }
    .alttez-subtle {
        color: var(--alttez-text-2);
        font-size: 1rem;
        margin-top: 0.7rem;
        max-width: 76ch;
        line-height: 1.58;
    }
    .alttez-hero-aside {
        position: relative;
        border: 1px solid var(--alttez-line-soft);
        border-radius: 18px;
        padding: 1.05rem 1.05rem 1rem;
        background:
            linear-gradient(180deg, rgba(12, 26, 43, 0.92), rgba(8, 17, 30, 0.88)),
            radial-gradient(circle at top, rgba(108, 193, 255, 0.08), transparent 55%);
        backdrop-filter: blur(12px);
    }
    .alttez-hero-aside::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent 38%);
        pointer-events: none;
    }
    .alttez-hero-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.78rem 0;
        border-bottom: 1px solid rgba(116, 145, 189, 0.12);
    }
    .alttez-hero-row:last-of-type {
        border-bottom: 0;
        padding-bottom: 0.2rem;
    }
    .alttez-hero-label {
        color: var(--alttez-text-2);
        font-size: 0.77rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .alttez-hero-value {
        color: var(--alttez-text-0);
        font-size: 1rem;
        font-weight: 650;
        text-align: right;
    }
    .alttez-micro-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.82rem;
        margin-top: 1rem;
    }
    .alttez-micro-card {
        border: 1px solid var(--alttez-line-soft);
        border-radius: 16px;
        padding: 0.9rem 0.92rem;
        background: linear-gradient(180deg, rgba(10, 20, 34, 0.86), rgba(8, 16, 28, 0.78));
        min-height: 104px;
    }
    .alttez-micro-label {
        color: var(--alttez-text-2);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.52rem;
    }
    .alttez-micro-value {
        color: var(--alttez-text-0);
        font-size: 1.28rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .alttez-micro-sub {
        color: var(--alttez-accent-2);
        font-size: 0.83rem;
    }
    .alttez-panel {
        border: 1px solid var(--alttez-line-soft);
        border-radius: 18px;
        padding: 1.05rem 1.08rem;
        background: linear-gradient(180deg, var(--alttez-bg-2), rgba(8, 17, 29, 0.78));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }
    .alttez-section-title {
        font-size: 1.08rem;
        font-weight: 680;
        color: var(--alttez-text-0);
        margin-bottom: 0.28rem;
        letter-spacing: -0.01em;
    }
    .alttez-section-copy {
        color: var(--alttez-text-2);
        font-size: 0.93rem;
        line-height: 1.55;
    }
    .alttez-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.36rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        line-height: 1;
        border: 1px solid rgba(96, 124, 170, 0.16);
        background: rgba(14, 27, 46, 0.82);
        color: var(--alttez-text-1);
    }
    .alttez-badge.ok {
        color: #7ff0c7;
        border-color: rgba(56, 218, 167, 0.24);
        background: rgba(8, 54, 43, 0.44);
    }
    .alttez-badge.warn {
        color: #ffd37d;
        border-color: rgba(247, 197, 74, 0.22);
        background: rgba(64, 51, 15, 0.42);
    }
    .alttez-badge.error {
        color: #ff9797;
        border-color: rgba(255, 116, 116, 0.2);
        background: rgba(68, 19, 19, 0.44);
    }
    .alttez-match-card {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(90, 116, 158, 0.14);
        border-radius: 18px;
        background:
            linear-gradient(180deg, rgba(10, 19, 33, 0.84), rgba(8, 17, 29, 0.74)),
            radial-gradient(circle at right top, rgba(82, 199, 255, 0.06), transparent 30%);
        padding: 1rem 1.04rem;
        margin-bottom: 0.82rem;
        box-shadow: 0 14px 32px rgba(2, 10, 20, 0.16);
    }
    .alttez-match-meta {
        color: var(--alttez-text-2);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        margin-bottom: 0.48rem;
    }
    .alttez-match-title {
        color: var(--alttez-text-0);
        font-size: 1.13rem;
        font-weight: 680;
        margin-bottom: 0.34rem;
    }
    .alttez-match-subtitle {
        color: var(--alttez-text-2);
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--alttez-line-soft);
        border-radius: 16px;
        padding: 0.84rem 0.92rem;
        background: linear-gradient(180deg, rgba(10, 20, 34, 0.82), rgba(8, 16, 29, 0.74));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--alttez-text-2);
    }
    div[data-testid="stMetricValue"] {
        color: var(--alttez-text-0);
        letter-spacing: -0.02em;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #2a7de7, #63d6ff);
        color: #07111a;
        border: 1px solid rgba(116, 194, 255, 0.12);
        padding: 0.76rem 1rem;
        font-weight: 700;
        box-shadow: 0 12px 28px rgba(34, 110, 255, 0.2);
        transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
    }
    .stButton > button:hover {
        color: #07111a;
        background: linear-gradient(135deg, #4a8ef0, #7ae0ff);
        transform: translateY(-1px);
        filter: saturate(1.04);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.16rem;
        background: rgba(9, 18, 31, 0.55);
        border: 1px solid var(--alttez-line-soft);
        border-radius: 999px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: transparent;
        color: #a6b8cf;
        border: 0;
        padding: 0.5rem 0.92rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(41, 118, 255, 0.24), rgba(39, 205, 255, 0.18));
        color: #ecf5ff;
        box-shadow: inset 0 0 0 1px rgba(111, 177, 255, 0.18);
    }
    .stSelectbox label, .stRadio label, .stDateInput label, .stSlider label {
        color: var(--alttez-text-1);
    }
    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stDateInput > div,
    [data-testid="stSidebar"] .stNumberInput > div,
    [data-testid="stSidebar"] .stTextInput > div {
        border-radius: 12px;
    }
    .alttez-feature-callout {
        border: 1px solid rgba(95, 151, 220, 0.18);
        border-radius: 16px;
        padding: 0.9rem 0.95rem;
        background: linear-gradient(180deg, rgba(12, 25, 42, 0.8), rgba(9, 17, 29, 0.72));
    }
    .alttez-feature-callout strong {
        color: var(--alttez-text-0);
    }
    .alttez-feature-hint {
        color: var(--alttez-text-2);
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .alttez-inline-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.82rem;
        margin-top: 0.95rem;
    }
    .alttez-inline-stat {
        border: 1px solid var(--alttez-line-soft);
        border-radius: 14px;
        padding: 0.82rem 0.88rem;
        background: linear-gradient(180deg, rgba(10, 21, 35, 0.78), rgba(8, 16, 29, 0.72));
    }
    .alttez-inline-stat-label {
        color: var(--alttez-text-2);
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.34rem;
    }
    .alttez-inline-stat-value {
        color: var(--alttez-text-0);
        font-size: 1.05rem;
        font-weight: 680;
    }
    .alttez-empty-state {
        border: 1px dashed rgba(112, 147, 198, 0.22);
        border-radius: 18px;
        padding: 1.15rem 1.15rem 1.05rem;
        background: linear-gradient(180deg, rgba(11, 22, 36, 0.7), rgba(8, 16, 28, 0.62));
    }
    .alttez-empty-title {
        color: var(--alttez-text-0);
        font-size: 1.02rem;
        font-weight: 680;
        margin-bottom: 0.3rem;
    }
    .alttez-empty-copy {
        color: var(--alttez-text-2);
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .alttez-subheader {
        color: var(--alttez-text-0);
        font-size: 1.28rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.78rem;
    }
    .alttez-data-note {
        color: var(--alttez-text-2);
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    .alttez-section-stack {
        display: grid;
        gap: 1rem;
    }
    .streamlit-expanderHeader {
        color: var(--alttez-text-1) !important;
        font-weight: 620 !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--alttez-line-soft);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(10, 20, 34, 0.68), rgba(8, 16, 28, 0.6));
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--alttez-line-soft);
        border-radius: 16px;
        overflow: hidden;
        background: rgba(8, 16, 28, 0.46);
    }
    [data-testid="stMarkdownContainer"] code {
        color: #9ee3ff;
    }
    @media (max-width: 900px) {
        .alttez-hero-grid,
        .alttez-micro-grid,
        .alttez-inline-stats {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 1100px) {
        .alttez-micro-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .alttez-inline-stats {
            grid-template-columns: 1fr;
        }
        .alttez-headline {
            font-size: 2.18rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(build_brand_css(), unsafe_allow_html=True)


@st.cache_data(ttl=86400)
def cargar_datos() -> pd.DataFrame:
    loader = InternationalLoader()
    return loader.fetch_data()


@st.cache_data(ttl=300)
def cargar_agenda(fecha: str) -> pd.DataFrame:
    return list_matches_for_date(
        pd.to_datetime(fecha).date(),
        include_statuses=("scheduled", "in_progress", "finished", "cancelled"),
    )


@st.cache_data(ttl=120)
def cargar_provider_healthcheck() -> list[dict[str, str]]:
    return run_healthcheck(",".join(get_live_provider_order()))


@st.cache_data(ttl=120)
def cargar_alertas_recientes(limit: int = 10) -> list[dict]:
    return load_recent_alert_events(limit=limit)


@st.cache_data(ttl=300)
def cargar_world_cup_2026_config() -> dict:
    return load_world_cup_2026_config()


@st.cache_data(ttl=300)
def cargar_world_cup_2026_reporte() -> dict | None:
    return load_latest_world_cup_2026_report()


def _get_team_profile(engine, team: str) -> dict:
    default_profile = {
        "home_attack": 1.0,
        "away_attack": 1.0,
        "home_defense": 1.0,
        "away_defense": 1.0,
        "attack": 1.0,
        "defense": 1.0,
        "matches": 0.0,
        "effective_matches": 0.0,
        "confidence": 0.0,
        "recent_points_per_match": 0.0,
        "recent_goal_diff_per_match": 0.0,
    }

    def _normalize(profile: dict | None) -> dict:
        normalized = default_profile.copy()
        if isinstance(profile, dict):
            normalized.update(profile)
        return normalized

    if hasattr(engine, "get_team_profile"):
        return _normalize(engine.get_team_profile(team))

    if hasattr(engine, "_resolve_stats"):
        return _normalize(engine._resolve_stats(team))

    team_stats = getattr(engine, "team_stats", {})
    if isinstance(team_stats, dict):
        return _normalize(team_stats.get(team))

    return default_profile


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


def _build_result_summary(prob_matrix, team_a: str, team_b: str, format_percent: bool = True) -> pd.DataFrame:
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

    summary_df = pd.DataFrame(
        {
            "Mercado": [f"Victoria {team_a}", "Empate", f"Victoria {team_b}"],
            "Probabilidad": [home_win, draw, away_win],
        }
    )
    if format_percent:
        summary_df["Probabilidad"] = summary_df["Probabilidad"].map(lambda value: f"{value:.1%}")
    return summary_df


def _build_score_heatmap_df(prob_matrix) -> pd.DataFrame:
    rows = []
    max_goals = prob_matrix.shape[0] - 1
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            rows.append(
                {
                    "Goles local": home_goals,
                    "Goles visitante": away_goals,
                    "Probabilidad": float(prob_matrix[home_goals, away_goals]),
                }
            )
    return pd.DataFrame(rows)


def _build_strength_chart_df(
    equipo_local: str,
    equipo_visitante: str,
    profile_local: dict,
    profile_visitante: dict,
    is_neutral: bool,
) -> pd.DataFrame:
    local_context_attack = profile_local.get("attack", 1.0) if is_neutral else profile_local.get("home_attack", 1.0)
    local_context_defense = profile_local.get("defense", 1.0) if is_neutral else profile_local.get("home_defense", 1.0)
    away_context_attack = profile_visitante.get("attack", 1.0) if is_neutral else profile_visitante.get("away_attack", 1.0)
    away_context_defense = profile_visitante.get("defense", 1.0) if is_neutral else profile_visitante.get("away_defense", 1.0)

    return pd.DataFrame(
        [
            {"Equipo": equipo_local, "Metrica": "Ataque base", "Valor": round(profile_local.get("attack", 1.0), 3)},
            {"Equipo": equipo_local, "Metrica": "Defensa base", "Valor": round(profile_local.get("defense", 1.0), 3)},
            {"Equipo": equipo_local, "Metrica": "Ataque contexto", "Valor": round(local_context_attack, 3)},
            {"Equipo": equipo_local, "Metrica": "Defensa contexto", "Valor": round(local_context_defense, 3)},
            {"Equipo": equipo_visitante, "Metrica": "Ataque base", "Valor": round(profile_visitante.get("attack", 1.0), 3)},
            {"Equipo": equipo_visitante, "Metrica": "Defensa base", "Valor": round(profile_visitante.get("defense", 1.0), 3)},
            {"Equipo": equipo_visitante, "Metrica": "Ataque contexto", "Valor": round(away_context_attack, 3)},
            {"Equipo": equipo_visitante, "Metrica": "Defensa contexto", "Valor": round(away_context_defense, 3)},
        ]
    )


def _build_form_chart_df(equipo_local: str, equipo_visitante: str, profile_local: dict, profile_visitante: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Equipo": equipo_local,
                "Metrica": "Puntos recientes p/partido",
                "Valor": round(profile_local.get("recent_points_per_match", 0.0), 3),
            },
            {
                "Equipo": equipo_local,
                "Metrica": "Dif gol reciente",
                "Valor": round(profile_local.get("recent_goal_diff_per_match", 0.0), 3),
            },
            {
                "Equipo": equipo_local,
                "Metrica": "Confianza modelo",
                "Valor": round(profile_local.get("confidence", 0.0), 3),
            },
            {
                "Equipo": equipo_visitante,
                "Metrica": "Puntos recientes p/partido",
                "Valor": round(profile_visitante.get("recent_points_per_match", 0.0), 3),
            },
            {
                "Equipo": equipo_visitante,
                "Metrica": "Dif gol reciente",
                "Valor": round(profile_visitante.get("recent_goal_diff_per_match", 0.0), 3),
            },
            {
                "Equipo": equipo_visitante,
                "Metrica": "Confianza modelo",
                "Valor": round(profile_visitante.get("confidence", 0.0), 3),
            },
        ]
    )


def _render_probability_chart(summary_df: pd.DataFrame) -> None:
    chart = (
        alt.Chart(summary_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Mercado:N", sort=None, title=None),
            y=alt.Y("Probabilidad:Q", axis=alt.Axis(format="%"), title="Probabilidad"),
            color=alt.Color(
                "Mercado:N",
                scale=alt.Scale(range=["#35d3ff", "#7d93ad", "#6fe5c7"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Mercado:N"),
                alt.Tooltip("Probabilidad:Q", format=".2%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_score_heatmap(heatmap_df: pd.DataFrame) -> None:
    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("Goles visitante:O", title="Goles visitante"),
            y=alt.Y("Goles local:O", title="Goles local"),
            color=alt.Color(
                "Probabilidad:Q",
                scale=alt.Scale(scheme="teals"),
                title="Probabilidad",
            ),
            tooltip=[
                alt.Tooltip("Goles local:O"),
                alt.Tooltip("Goles visitante:O"),
                alt.Tooltip("Probabilidad:Q", format=".2%"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(heatmap, use_container_width=True)


def _render_strength_chart(strength_chart_df: pd.DataFrame) -> None:
    chart = (
        alt.Chart(strength_chart_df)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("Valor:Q", title="Indice relativo"),
            y=alt.Y("Metrica:N", sort=None, title=None),
            color=alt.Color(
                "Equipo:N",
                scale=alt.Scale(range=["#35d3ff", "#6fe5c7"]),
                legend=alt.Legend(title=None),
            ),
            xOffset="Equipo:N",
            tooltip=["Equipo:N", "Metrica:N", alt.Tooltip("Valor:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_form_chart(form_chart_df: pd.DataFrame) -> None:
    chart = (
        alt.Chart(form_chart_df)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("Valor:Q", title="Valor"),
            y=alt.Y("Metrica:N", sort=None, title=None),
            color=alt.Color(
                "Equipo:N",
                scale=alt.Scale(range=["#35d3ff", "#6fe5c7"]),
                legend=alt.Legend(title=None),
            ),
            xOffset="Equipo:N",
            tooltip=["Equipo:N", "Metrica:N", alt.Tooltip("Valor:Q", format=".3f")],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def _format_kickoff(value) -> str:
    if pd.isna(value) or value in {None, ""}:
        return "--:--"
    kickoff = pd.to_datetime(value)
    return kickoff.strftime("%H:%M")


def _status_badge_class(status: str | None) -> str:
    normalized = (status or "").strip().lower()
    if normalized == "success":
        return "ok"
    if normalized in {"empty", "warning"}:
        return "warn"
    if normalized == "error":
        return "error"
    return ""


def _match_status_badge(status: str | None) -> str:
    mapping = {
        "scheduled": "Programado",
        "in_progress": "En juego",
        "finished": "Finalizado",
        "cancelled": "Cancelado",
    }
    return mapping.get((status or "").strip().lower(), status or "N/D")


def _sync_error_label(value: str | None) -> str:
    mapping = {
        "config_missing": "Configuracion incompleta",
        "provider_http_error": "Error HTTP del proveedor",
        "provider_payload_error": "Payload invalido del proveedor",
        "provider_runtime_error": "Fallo operativo del proveedor",
        "db_write_error": "Fallo de escritura en base de datos",
        "no_fixtures_found": "Sin partidos encontrados",
    }
    normalized = _safe_text(value).lower()
    return mapping.get(normalized, _safe_text(value, "Sin clasificar"))


def _render_status_pill(label: str, status: str | None) -> str:
    return (
        f'<span class="alttez-badge {_status_badge_class(status)}">'
        f"{html.escape(str(label))}: {html.escape(str(_match_status_badge(status) if label == 'Estado' else status or 'N/D'))}"
        "</span>"
    )


def _render_sync_status(sync_result: dict | None, fecha: str) -> None:
    if not sync_result:
        return

    provider_used = sync_result.get("provider_used") or "sin proveedor"
    status = sync_result.get("status")
    error_category = sync_result.get("error_category")
    certification_status = sync_result.get("certification_status")
    certification_report_path = sync_result.get("certification_report_path")
    certification_audit_persisted = sync_result.get("certification_audit_persisted")
    certification_audit_error = sync_result.get("certification_audit_error")

    if status == "success":
        st.success(
            f"Agenda del {fecha} sincronizada desde `{provider_used}` con "
            f"{sync_result.get('scheduled_matches', 0)} partidos listos para analisis."
        )
    elif status == "empty":
        st.info(
            f"ALTTEZ sincronizo la fecha `{fecha}` correctamente, pero no encontro partidos internacionales. "
            f"Ultimo proveedor revisado: `{provider_used}`."
        )
    elif status == "error":
        st.error(
            "La sincronizacion del stack live fallo. "
            f"Categoria: {_sync_error_label(error_category)}. "
            f"Detalle: {sync_result.get('error') or 'sin detalle operativo'}"
        )

    if certification_status == "certified":
        st.success(
            "Lote certificado correctamente. "
            f"{sync_result.get('certification_certified_records', 0)} registros certificados "
            f"con cobertura {sync_result.get('certification_coverage_pct', 0.0):.2f}%."
        )
    elif certification_status == "quarantined":
        st.warning(
            "Lote puesto en cuarentena por calidad insuficiente. "
            f"Certificados: {sync_result.get('certification_certified_records', 0)} | "
            f"Rechazados: {sync_result.get('certification_rejected_records', 0)} | "
            f"Cobertura: {sync_result.get('certification_coverage_pct', 0.0):.2f}%."
        )
    elif certification_status == "skipped":
        st.caption("No hubo partidos finalizados para certificar en este lote.")

    if certification_report_path:
        st.caption(f"Reporte de certificacion: `{certification_report_path}`")
    if certification_audit_persisted:
        st.caption("Auditoria del lote persistida en Supabase.")
    elif certification_audit_error:
        st.caption(f"No se pudo persistir la auditoria: {certification_audit_error}")

    alerts = sync_result.get("alerts") or []
    if alerts:
        alerts_df = pd.DataFrame(alerts).rename(
            columns={
                "alert_level": "Nivel",
                "alert_code": "Codigo",
                "title": "Titulo",
                "message": "Mensaje",
                "storage_status": "Persistencia",
                "delivery_channel": "Canal",
                "delivery_status": "Entrega",
            }
        )
        st.markdown("**Alertas del lote actual**")
        st.dataframe(
            alerts_df[["Nivel", "Codigo", "Titulo", "Mensaje", "Persistencia", "Canal", "Entrega"]],
            width="stretch",
            hide_index=True,
        )

    attempts = sync_result.get("attempted_providers") or []
    if attempts:
        with st.expander("Ver auditoria del stack live", expanded=False):
            attempts_df = pd.DataFrame(attempts).rename(
                columns={
                    "provider": "Proveedor",
                    "status": "Estado",
                    "detail": "Detalle",
                    "error_category": "Categoria",
                }
            )
            if "Categoria" in attempts_df.columns:
                attempts_df["Categoria"] = attempts_df["Categoria"].map(_sync_error_label)
            st.dataframe(attempts_df, width="stretch", hide_index=True)


def _render_provider_health_panel() -> None:
    healthcheck = cargar_provider_healthcheck()
    if not healthcheck:
        return

    with st.expander("Diagnostico del stack live", expanded=False):
        st.caption(
            "Si editas `.env`, reinicia Streamlit antes de sincronizar y valida el stack con "
            "`venv\\Scripts\\python.exe -m alttez.ingestion.provider_healthcheck`."
        )
        st.code(",".join(get_live_provider_order()), language="text")
        health_df = pd.DataFrame(healthcheck).rename(
            columns={
                "provider": "Proveedor",
                "configured": "Configurado",
                "env_state": "Secreto",
                "stack_position": "Orden",
                "stack_scope": "Uso",
                "detail": "Detalle",
            }
        )
        st.dataframe(health_df, width="stretch", hide_index=True)


def _render_recent_alerts_panel(limit: int = 10) -> None:
    alerts = cargar_alertas_recientes(limit)
    if not alerts:
        return

    with st.expander("Alertas recientes del sistema", expanded=False):
        alerts_df = pd.DataFrame(alerts).rename(
            columns={
                "target_date": "Fecha",
                "batch_id": "Lote",
                "provider_used": "Proveedor",
                "alert_level": "Nivel",
                "alert_code": "Codigo",
                "title": "Titulo",
                "message": "Mensaje",
                "storage_status": "Persistencia",
                "delivery_channel": "Canal",
                "delivery_status": "Entrega",
                "created_at": "Creado",
            }
        )
        display_columns = [
            "Creado",
            "Fecha",
            "Nivel",
            "Codigo",
            "Titulo",
            "Proveedor",
            "Persistencia",
            "Canal",
            "Entrega",
        ]
        st.dataframe(
            alerts_df[[column for column in display_columns if column in alerts_df.columns]],
            width="stretch",
            hide_index=True,
        )


def _render_recommendation_panel(result: dict) -> None:
    summary = _build_recommendation_summary(result)
    if not summary:
        return

    best_market = summary.get("best_market") or {}
    market_label = best_market.get("Mercado", "Mercado principal por definir")
    market_probability = best_market.get("Probabilidad", "N/D")
    alternatives = summary.get("alternatives")
    alternatives_text = "Sin escenarios alternos"
    if isinstance(alternatives, pd.DataFrame) and not alternatives.empty:
        alternatives_text = " | ".join(
            f"{row.Pronostico} (EV {row.EV})"
            for row in alternatives.itertuples(index=False)
        )

    st.markdown(
        f"""
        <div class="alttez-panel">
            <div class="alttez-section-title">Recomendacion ALTTEZ</div>
            <div class="alttez-section-copy">
                Pronostico principal: <strong>{html.escape(str(summary['primary_score']))}</strong> con EV <strong>{html.escape(str(summary['primary_ev']))}</strong><br>
                Mercado mas probable: <strong>{html.escape(str(market_label))}</strong> ({html.escape(str(market_probability))})<br>
                Escenarios cercanos: {html.escape(str(alternatives_text))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    decision_cols = st.columns(4, gap="medium")
    decision_cols[0].metric("Marcador ALTTEZ", summary["primary_score"])
    decision_cols[1].metric("EV esperado", summary["primary_ev"])
    decision_cols[2].metric("xG total", f"{result['lambda_h'] + result['lambda_a']:.2f}")
    decision_cols[3].metric(
        "Confianza media",
        f"{((result['confidence_local'] + result['confidence_visitante']) / 2):.0%}",
    )


def _current_sync_result(target_date) -> dict | None:
    sync_result = st.session_state.get("sync_result")
    sync_result_date = st.session_state.get("sync_result_date")
    if not sync_result or sync_result_date != str(target_date):
        return None
    return sync_result


def _clear_analysis_feedback() -> None:
    st.session_state.pop("analysis_error", None)
    st.session_state.pop("analysis_error_date", None)


def _reset_stale_analysis_state(target_date) -> None:
    date_key = str(target_date)
    result = st.session_state.get("analysis_result")
    if result and str(result.get("fecha_simulacion")) != date_key:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("selected_match_id", None)
        st.session_state.pop("selected_match", None)

    error_date = st.session_state.get("analysis_error_date")
    if error_date and error_date != date_key:
        _clear_analysis_feedback()


def _is_prediction_supported(profile_local: dict, profile_visitante: dict) -> tuple[bool, str | None]:
    local_matches = float(profile_local.get("matches", 0.0))
    away_matches = float(profile_visitante.get("matches", 0.0))
    local_effective = float(profile_local.get("effective_matches", 0.0))
    away_effective = float(profile_visitante.get("effective_matches", 0.0))

    if local_matches <= 0 and away_matches <= 0:
        return (
            False,
            "No hay historial compatible para ninguno de los dos equipos en la base internacional actual. "
            "ALTTEZ no debe proyectar este partido con perfil neutro.",
        )

    if local_effective <= 0 and away_effective <= 0:
        return (
            False,
            "Los dos equipos quedaron sin soporte estadistico efectivo en la ventana elegida. "
            "Ajusta la ventana o usa una competencia con historial compatible.",
        )

    return True, None


def _resolve_tournament_context(
    tournament_mode: str,
    match_context: dict | None = None,
) -> tuple[dict | None, str]:
    normalized_mode = _safe_text(tournament_mode, "Automatico")
    if normalized_mode == "Mundial 2026":
        return {"tournament": "FIFA World Cup 2026", "edition": 2026}, "Mundial 2026"
    if normalized_mode == "Sin contexto especial":
        return None, "Sin contexto especial"

    context = match_context or {}
    tournament_name = _safe_text(context.get("tournament"))
    haystack = f"{tournament_name} {_safe_text(context.get('competition_class'))}".lower()
    if "world cup 2026" in haystack or "fifa world cup 2026" in haystack:
        return {"tournament": "FIFA World Cup 2026", "edition": 2026}, "Automatico: Mundial 2026"
    return None, "Automatico"


def _has_historical_support(
    df: pd.DataFrame,
    team: str,
    fecha_simulacion,
    ventana: int,
    min_matches: int = 1,
) -> bool:
    target_date = pd.to_datetime(fecha_simulacion)
    start_date = target_date - pd.DateOffset(years=ventana)
    team_matches = df[
        (df["date"] >= start_date)
        & (df["date"] < target_date)
        & ((df["home_team"] == team) | (df["away_team"] == team))
    ]
    return len(team_matches) >= min_matches


def _run_analysis(
    df: pd.DataFrame,
    fecha_simulacion: str,
    ventana: int,
    prior_matches: int,
    recency_half_life_days: int,
    equipo_local: str,
    equipo_visitante: str,
    is_neutral: bool,
    is_knockout: bool,
    max_goles: int,
    match_context: dict | None = None,
    tournament_mode: str = "Automatico",
) -> dict:
    tournament_context, tournament_context_label = _resolve_tournament_context(tournament_mode, match_context)
    predictor = build_active_predictor(
        df,
        fecha_simulacion,
        poisson_kwargs={
            "years_window": ventana,
            "prior_matches": prior_matches,
            "min_history_matches": 5,
            "recency_half_life_days": recency_half_life_days,
            "recent_form_matches": 6,
        },
    )
    prediction = predictor.predict_match(
        equipo_local,
        equipo_visitante,
        is_neutral=is_neutral,
        is_knockout=is_knockout,
        max_goals=max_goles,
        tournament_context=tournament_context,
    )

    profile_local = prediction["profile_local"]
    profile_visitante = prediction["profile_visitante"]
    prediction_supported, support_message = _is_prediction_supported(profile_local, profile_visitante)
    if not prediction_supported:
        raise ValueError(
            f"{support_message} Equipos evaluados: `{equipo_local}` y `{equipo_visitante}`."
        )

    prob_matrix = prediction["prob_matrix"]
    recomendaciones = prediction.get("recommendations", pd.DataFrame())
    result_summary_chart = prediction.get("result_summary", pd.DataFrame()).copy()
    score_heatmap_df = _build_score_heatmap_df(prob_matrix)
    strength_chart_df = _build_strength_chart_df(
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        profile_local=profile_local,
        profile_visitante=profile_visitante,
        is_neutral=is_neutral,
    )
    form_chart_df = _build_form_chart_df(equipo_local, equipo_visitante, profile_local, profile_visitante)

    strength_rows = [
        {
            "Equipo": equipo_local,
            "Ataque": round(profile_local.get("attack", 1.0), 3),
            "Defensa": round(profile_local.get("defense", 1.0), 3),
            "Local ataque": round(profile_local.get("home_attack", 1.0), 3),
            "Local defensa": round(profile_local.get("home_defense", 1.0), 3),
            "Forma reciente": round(profile_local.get("recent_points_per_match", 0.0), 3),
            "Confianza": round(profile_local.get("confidence", 0.0), 3),
        },
        {
            "Equipo": equipo_visitante,
            "Ataque": round(profile_visitante.get("attack", 1.0), 3),
            "Defensa": round(profile_visitante.get("defense", 1.0), 3),
            "Visitante ataque": round(profile_visitante.get("away_attack", 1.0), 3),
            "Visitante defensa": round(profile_visitante.get("away_defense", 1.0), 3),
            "Forma reciente": round(profile_visitante.get("recent_points_per_match", 0.0), 3),
            "Confianza": round(profile_visitante.get("confidence", 0.0), 3),
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
        "recency_half_life_days": recency_half_life_days,
        "max_goles": max_goles,
        "tournament_context": tournament_context or {},
        "tournament_context_label": tournament_context_label,
        "lambda_h": prediction["lambda_h"],
        "lambda_a": prediction["lambda_a"],
        "model_version": prediction["model_version"],
        "model_family": prediction["model_family"],
        "used_fallback": bool(prediction.get("used_fallback", False)),
        "fallback_reason": prediction.get("fallback_reason"),
        "matches_local": int(profile_local.get("matches", 0.0)),
        "matches_visitante": int(profile_visitante.get("matches", 0.0)),
        "effective_matches_local": float(profile_local.get("effective_matches", 0.0)),
        "effective_matches_visitante": float(profile_visitante.get("effective_matches", 0.0)),
        "confidence_local": float(profile_local.get("confidence", 0.0)),
        "confidence_visitante": float(profile_visitante.get("confidence", 0.0)),
        "recomendaciones": recomendaciones,
        "scorelines": _build_scoreline_table(prob_matrix, top_n=8),
        "result_summary": _build_result_summary(prob_matrix, equipo_local, equipo_visitante),
        "result_summary_chart": result_summary_chart,
        "score_heatmap_df": score_heatmap_df,
        "strength_df": pd.DataFrame(strength_rows),
        "strength_chart_df": strength_chart_df,
        "form_chart_df": form_chart_df,
        "match_context": match_context or {},
        "xgboost_feature_payload": prediction.get("xgboost_feature_payload", {}),
    }


def _sync_selected_date(target_date, reason: str = "manual") -> dict:
    try:
        with st.spinner("Sincronizando agenda internacional desde el stack live..."):
            result = ingest_live_matches_for_date(pd.to_datetime(target_date).date())
    except Exception as exc:  # noqa: BLE001 - never let live sync crash the dashboard shell.
        result = {
            "status": "error",
            "provider_used": None,
            "attempted_providers": [],
            "scheduled_matches": 0,
            "raw_records": 0,
            "teams_catalog": 0,
            "tournaments_catalog": 0,
            "competition_groups_catalog": 0,
            "error_category": "provider_runtime_error",
            "error": str(exc),
        }
    cargar_agenda.clear()
    st.session_state["sync_result"] = result
    st.session_state["sync_result_date"] = str(target_date)
    st.session_state["sync_reason"] = reason
    return result


def _maybe_auto_sync_agenda(target_date, agenda_df: pd.DataFrame) -> pd.DataFrame:
    date_key = str(target_date)
    attempted_dates = st.session_state.setdefault("auto_sync_attempted_dates", set())
    if not agenda_df.empty or date_key in attempted_dates:
        return agenda_df

    if pd.to_datetime(target_date).date() < (pd.Timestamp.today().date() - timedelta(days=2)):
        return agenda_df

    attempted_dates.add(date_key)
    result = _sync_selected_date(target_date, reason="auto_empty_date")
    if result.get("status") in {"success", "empty"}:
        return cargar_agenda(date_key)
    return agenda_df


def _agenda_subset(agenda_df: pd.DataFrame, competition_class: str | None = None) -> pd.DataFrame:
    if competition_class is None:
        return agenda_df.copy()
    if "competition_class" not in agenda_df.columns:
        return agenda_df.iloc[0:0].copy()
    mask = agenda_df["competition_class"].fillna("").str.lower() == competition_class
    return agenda_df.loc[mask].copy()


def _world_cup_groups(agenda_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if agenda_df.empty:
        return []
    if "group_label" not in agenda_df.columns:
        return [("Programacion", agenda_df.copy())]

    grouped: list[tuple[str, pd.DataFrame]] = []
    labels = agenda_df["group_label"].fillna("Sin grupo").replace("", "Sin grupo")
    for group_name in labels.unique().tolist():
        grouped.append((group_name, agenda_df.loc[labels == group_name].copy()))
    return grouped


def _select_match_from_agenda(
    agenda_df: pd.DataFrame,
    df_historico: pd.DataFrame,
    fecha_simulacion,
    ventana: int,
    prior_matches: int,
    recency_half_life_days: int,
    fase_torneo: str,
    max_goles: int,
    tournament_mode: str,
    section_key: str,
) -> None:
    safe_section_key = _safe_key(section_key)
    for row in agenda_df.itertuples(index=False):
        row_dict = row._asdict()
        kickoff_label = _format_kickoff(row_dict.get("kickoff_at"))
        tournament_meta = _build_match_meta(row_dict)
        city = row_dict.get("city") or "Ciudad pendiente"
        local_supported = _has_historical_support(df_historico, row_dict["home_team"], fecha_simulacion, ventana)
        away_supported = _has_historical_support(df_historico, row_dict["away_team"], fecha_simulacion, ventana)
        model_supported = local_supported or away_supported
        support_label = _support_level(local_supported, away_supported)
        support_status = _support_status(support_label)
        support_badge = (
            f'<span class="alttez-badge {_status_badge_class(support_status)}">'
            f"{html.escape(str(support_label))}"
            "</span>"
        )

        st.markdown(
            f"""
            <div class="alttez-match-card">
                <div class="alttez-match-meta">{html.escape(str(kickoff_label))} | {html.escape(str(tournament_meta))}</div>
                <div class="alttez-match-title">{html.escape(str(row_dict['home_team']))} vs {html.escape(str(row_dict['away_team']))}</div>
                <div class="alttez-match-subtitle">{html.escape(str(city))} | {html.escape(str(_match_status_badge(row_dict.get('status'))))}</div>
                <div style="margin-top:0.65rem;">
                    {support_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not model_supported:
            continue

        if st.button("Lanzar analisis ALTTEZ", key=f"fixture-{safe_section_key}-{row_dict['id']}"):
            try:
                st.session_state["selected_match_id"] = row_dict["id"]
                st.session_state["selected_match"] = row_dict
                st.session_state["analysis_result"] = _run_analysis(
                    df=df_historico,
                    fecha_simulacion=str(fecha_simulacion),
                    ventana=ventana,
                    prior_matches=prior_matches,
                    recency_half_life_days=recency_half_life_days,
                    equipo_local=row_dict["home_team"],
                    equipo_visitante=row_dict["away_team"],
                    is_neutral=bool(row_dict.get("is_neutral", False)),
                    is_knockout=fase_torneo == "Eliminacion Directa",
                    max_goles=max_goles,
                    match_context=row_dict,
                    tournament_mode=tournament_mode,
                )
                _clear_analysis_feedback()
            except Exception as exc:  # noqa: BLE001 - keep dashboard responsive when a fixture lacks support.
                st.session_state.pop("analysis_result", None)
                st.session_state["analysis_error"] = str(exc)
                st.session_state["analysis_error_date"] = str(fecha_simulacion)


def _render_agenda_section(
    title: str,
    description: str,
    agenda_df: pd.DataFrame,
    df_historico: pd.DataFrame,
    fecha_simulacion,
    ventana: int,
    prior_matches: int,
    recency_half_life_days: int,
    fase_torneo: str,
    max_goles: int,
    tournament_mode: str,
    empty_message: str,
    section_key: str,
) -> None:
    match_count = len(agenda_df)
    st.markdown(
        f"""
        <div class="alttez-panel">
            <div class="alttez-section-title">{html.escape(str(title))}</div>
            <div class="alttez-section-copy">{html.escape(str(description))}</div>
            <div class="alttez-inline-stats">
                <div class="alttez-inline-stat">
                    <div class="alttez-inline-stat-label">Partidos visibles</div>
                    <div class="alttez-inline-stat-value">{html.escape(str(match_count))}</div>
                </div>
                <div class="alttez-inline-stat">
                    <div class="alttez-inline-stat-label">Formato activo</div>
                    <div class="alttez-inline-stat-value">{html.escape(str(fase_torneo))}</div>
                </div>
                <div class="alttez-inline-stat">
                    <div class="alttez-inline-stat-label">Contexto torneo</div>
                    <div class="alttez-inline-stat-value">{html.escape(str(tournament_mode))}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if agenda_df.empty:
        st.markdown(
            f"""
            <div class="alttez-empty-state">
                <div class="alttez-empty-title">Sin partidos analizables en este bloque</div>
                <div class="alttez-empty-copy">{html.escape(str(empty_message))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    _select_match_from_agenda(
        agenda_df=agenda_df,
        df_historico=df_historico,
        fecha_simulacion=fecha_simulacion,
        ventana=ventana,
        prior_matches=prior_matches,
        recency_half_life_days=recency_half_life_days,
        fase_torneo=fase_torneo,
        max_goles=max_goles,
        tournament_mode=tournament_mode,
        section_key=section_key,
    )


def _render_hero(fecha_simulacion, fase_torneo: str, agenda_df: pd.DataFrame, tournament_mode: str) -> None:
    sync_result = _current_sync_result(fecha_simulacion) or {}
    sync_status = sync_result.get("status") or "idle"
    provider_used = sync_result.get("provider_used") or "stack live"
    certified_records = int(sync_result.get("certification_certified_records") or 0)
    attempted_providers = sync_result.get("attempted_providers") or []
    status_label = {
        "idle": "En espera",
        "success": "Agenda sincronizada",
        "empty": "Sin partidos",
        "error": "Fallo operativo",
    }.get(sync_status, sync_status)

    st.markdown(
        f"""
        <div class="alttez-hero">
            <div class="alttez-hero-grid">
                <div>
                    <div class="alttez-kicker">Predictive Engine</div>
                    <div class="alttez-headline">ALTTEZ Agenda en Vivo y Oraculo del Dia</div>
                    <div class="alttez-subtle">
                        Monitoreo de amistosos internacionales, agenda real desde proveedores live y analisis automatico
                        con Poisson, valor esperado y blindaje para selecciones nuevas.
                    </div>
                    <div style="margin-top:0.9rem; display:flex; flex-wrap:wrap; gap:0.45rem;">
                        {_render_status_pill('Stack', status_label)}
                        {_render_status_pill('Proveedor', provider_used)}
                        {_render_status_pill('Formato', fase_torneo)}
                        {_render_status_pill('Contexto', tournament_mode)}
                    </div>
                    <div class="alttez-micro-grid">
                        <div class="alttez-micro-card">
                            <div class="alttez-micro-label">Partidos</div>
                            <div class="alttez-micro-value">{html.escape(str(len(agenda_df)))}</div>
                            <div class="alttez-micro-sub">Agenda lista para analisis</div>
                        </div>
                        <div class="alttez-micro-card">
                            <div class="alttez-micro-label">Proveedor activo</div>
                            <div class="alttez-micro-value">{html.escape(str(provider_used))}</div>
                            <div class="alttez-micro-sub">Ultima lectura del stack live</div>
                        </div>
                        <div class="alttez-micro-card">
                            <div class="alttez-micro-label">Lote certificado</div>
                            <div class="alttez-micro-value">{html.escape(str(certified_records))}</div>
                            <div class="alttez-micro-sub">Registros auditados</div>
                        </div>
                        <div class="alttez-micro-card">
                            <div class="alttez-micro-label">Proveedores sondeados</div>
                            <div class="alttez-micro-value">{html.escape(str(len(attempted_providers)))}</div>
                            <div class="alttez-micro-sub">Cobertura operativa</div>
                        </div>
                    </div>
                </div>
                <div class="alttez-hero-aside">
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Fecha objetivo</span>
                        <span class="alttez-hero-value">{html.escape(str(fecha_simulacion))}</span>
                    </div>
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Partidos detectados</span>
                        <span class="alttez-hero-value">{html.escape(str(len(agenda_df)))}</span>
                    </div>
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Stack live</span>
                        <span class="alttez-hero-value">{html.escape(str(provider_used))}</span>
                    </div>
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Modo de puntuacion</span>
                        <span class="alttez-hero-value">{html.escape(str(fase_torneo))}</span>
                    </div>
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Contexto torneo</span>
                        <span class="alttez-hero-value">{html.escape(str(tournament_mode))}</span>
                    </div>
                    <div class="alttez-section-copy">
                        {html.escape(str(status_label))} con agenda lista para analisis y auditoria operativa en tiempo real.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_control_summary(ventana: int, prior_matches: int, recency_half_life_days: int, max_goles: int) -> None:
    st.markdown(
        f"""
        <div class="alttez-panel alttez-feature-callout">
            <div class="alttez-section-title">Blindaje del motor</div>
            <div class="alttez-feature-hint">
                Ventana reciente: <strong>{html.escape(str(ventana))} anos</strong><br>
                Suavizado por muestra: <strong>{html.escape(str(prior_matches))}</strong><br>
                Vida media de recencia: <strong>{html.escape(str(recency_half_life_days))} dias</strong><br>
                Maximo de goles simulados: <strong>{html.escape(str(max_goles))}</strong><br>
                Equipos con menos de 5 partidos usan fuerza neutral por defecto.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_world_cup_2026_analysis(
    df: pd.DataFrame,
    fecha_simulacion: str,
    ventana: int,
    prior_matches: int,
    recency_half_life_days: int,
    iterations: int,
    random_seed: int,
) -> dict:
    predictor = build_active_predictor(
        df,
        fecha_simulacion,
        poisson_kwargs={
            "years_window": ventana,
            "prior_matches": prior_matches,
            "min_history_matches": 5,
            "recency_half_life_days": recency_half_life_days,
            "recent_form_matches": 6,
        },
    )
    config = cargar_world_cup_2026_config()
    report = simulate_world_cup_2026(
        predictor,
        config=config,
        iterations=iterations,
        random_seed=random_seed,
    )
    output_path = write_world_cup_2026_report(report)
    report["report_path"] = str(output_path)
    return report


def _render_world_cup_2026_dashboard(report: dict | None, fecha_simulacion: str, iterations: int, random_seed: int) -> None:
    st.markdown(
        f"""
        <div class="alttez-panel">
            <div class="alttez-section-title">Simulacion integral del Mundial 2026</div>
            <div class="alttez-section-copy">
                Esta vista corre el torneo completo con Monte Carlo usando el predictor activo, contexto Klement del
                Mundial 2026 y el draw oficial cargado en configuracion.
            </div>
            <div class="alttez-inline-stats">
                <div class="alttez-inline-stat">
                    <div class="alttez-inline-stat-label">Fecha de corte</div>
                    <div class="alttez-inline-stat-value">{html.escape(str(fecha_simulacion))}</div>
                </div>
                <div class="alttez-inline-stat">
                    <div class="alttez-inline-stat-label">Iteraciones</div>
                    <div class="alttez-inline-stat-value">{html.escape(str(iterations))}</div>
                </div>
                <div class="alttez-inline-stat">
                    <div class="alttez-inline-stat-label">Semilla</div>
                    <div class="alttez-inline-stat-value">{html.escape(str(random_seed))}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not report:
        st.markdown(
            """
            <div class="alttez-empty-state">
                <div class="alttez-empty-title">Todavia no hay una corrida materializada</div>
                <div class="alttez-empty-copy">
                    Ejecuta la simulacion desde el panel lateral para generar el cuadro completo, las tablas de grupos
                    y las probabilidades de campeon.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    probabilities = pd.DataFrame(report.get("team_probabilities") or [])
    standings_snapshot = report.get("standings_snapshot") or {}
    if probabilities.empty:
        st.warning("La simulacion no devolvio probabilidades agregadas.")
        return

    st.markdown(
        f"""
        <div class="alttez-panel">
            <div class="alttez-section-title">Corrida mas reciente</div>
            <div class="alttez-section-copy">
                Generada en: {html.escape(str(report.get('generated_at', 'N/D')))}<br>
                Modelo: <strong>{html.escape(str(report.get('model_version', 'N/D')))}</strong> ({html.escape(str(report.get('model_family', 'N/D')))})<br>
                Fallback operativo: {'Si' if report.get('used_fallback') else 'No'}<br>
                Cache de predicciones: {html.escape(str(report.get('prediction_cache_size', 0)))} cruces<br>
                Artefacto: {html.escape(str(report.get('report_path') or 'world-cup-2026-latest.json'))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4, gap="medium")
    top_row = probabilities.iloc[0]
    metric_cols[0].metric("Favorito al titulo", str(top_row.get("team") or "N/D"))
    metric_cols[1].metric("Prob. campeon", f"{float(top_row.get('champion_probability') or 0.0):.1%}")
    metric_cols[2].metric("Prob. final", f"{float(top_row.get('final_probability') or 0.0):.1%}")
    metric_cols[3].metric("Prob. semifinal", f"{float(top_row.get('semi_final_probability') or 0.0):.1%}")

    chart_df = probabilities.head(12).copy()
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("team:N", sort="-y", title=None),
            y=alt.Y("champion_probability:Q", axis=alt.Axis(format="%"), title="Probabilidad campeon"),
            color=alt.Color(
                "champion_probability:Q",
                scale=alt.Scale(range=["#1d7cc8", "#7ae0ff"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("team:N", title="Equipo"),
                alt.Tooltip("champion_probability:Q", format=".2%"),
                alt.Tooltip("final_probability:Q", format=".2%"),
                alt.Tooltip("semi_final_probability:Q", format=".2%"),
            ],
        )
        .properties(height=310)
    )
    st.altair_chart(chart, use_container_width=True)

    tab_probs, tab_groups = st.tabs(["Probabilidades", "Tablas de grupos"])

    with tab_probs:
        display_df = probabilities.copy()
        for column in (
            "group_stage_probability",
            "round_of_32_probability",
            "round_of_16_probability",
            "quarter_final_probability",
            "semi_final_probability",
            "final_probability",
            "champion_probability",
            "group_winner_probability",
            "group_runner_up_probability",
        ):
            if column in display_df.columns:
                display_df[column] = display_df[column].map(lambda value: f"{float(value):.1%}")
        st.dataframe(display_df, width="stretch", hide_index=True)

    with tab_groups:
        if not standings_snapshot:
            st.info("Esta corrida no dejo snapshot de grupos.")
        else:
            for group_name, rows in standings_snapshot.items():
                st.markdown(
                    f"""
                    <div class="alttez-panel">
                        <div class="alttez-section-title">Grupo {html.escape(str(group_name))}</div>
                        <div class="alttez-section-copy">
                            Snapshot de la ultima iteracion usada como referencia visual para la simulacion completa.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                standings_df = pd.DataFrame(rows)
                st.dataframe(standings_df, width="stretch", hide_index=True)


def main() -> None:
    with st.sidebar:
        dashboard_view = st.radio(
            "Vista", ["Predicciones diarias", "Simulacion Mundial 2026", "Rendimiento del modelo", "NBA"]
        )
    if dashboard_view == "NBA":
        from alttez.dashboard.nba_view import render_nba_view

        render_nba_view()
        return

    try:
        with st.spinner("Sincronizando base historica internacional..."):
            df_historico = cargar_datos()
    except RuntimeError as exc:
        st.error(str(exc))
        st.caption("Si ya tienes una copia del CSV, puedes guardarla en `storage/network/international_results.csv`.")
        return

    default_date = pd.Timestamp.today().date()

    with st.sidebar:
        st.header("Centro de control")
        fecha_simulacion = st.date_input("Fecha de agenda", value=default_date)
        fase_torneo = st.radio("Formato de puntuacion", ["Fase de Grupos", "Eliminacion Directa"])
        tournament_mode = st.selectbox(
            "Contexto de torneo",
            ["Automatico", "Mundial 2026", "Sin contexto especial"],
            index=0,
            help="Controla si ALTTEZ activa las señales Klement de contexto geográfico del Mundial 2026.",
        )
        ventana = st.slider("Ventana de analisis (anos)", min_value=1, max_value=10, value=4)
        prior_matches = st.slider("Suavizado por muestra", min_value=0, max_value=20, value=8)
        recency_half_life_days = st.slider("Vida media de recencia (dias)", min_value=90, max_value=720, value=365, step=15)
        max_goles = st.slider("Maximo de goles simulados", min_value=5, max_value=10, value=7)
        world_cup_iterations = st.slider("Iteraciones Mundial 2026", min_value=100, max_value=5000, value=500, step=100)
        world_cup_seed = st.number_input("Semilla Mundial 2026", min_value=1, max_value=999999, value=7, step=1)

        if dashboard_view == "Predicciones diarias" and st.button("Sincronizar fecha seleccionada", key="sync-live-fixtures"):
            result = _sync_selected_date(fecha_simulacion, reason="manual")
            _clear_analysis_feedback()
            if result.get("status") != "success":
                st.session_state.pop("analysis_result", None)

        if dashboard_view == "Simulacion Mundial 2026" and st.button("Correr simulacion Mundial 2026", key="run-world-cup-2026"):
            try:
                with st.spinner("Corriendo simulacion integral del Mundial 2026..."):
                    report = _run_world_cup_2026_analysis(
                        df=df_historico,
                        fecha_simulacion=str(fecha_simulacion),
                        ventana=ventana,
                        prior_matches=prior_matches,
                        recency_half_life_days=recency_half_life_days,
                        iterations=int(world_cup_iterations),
                        random_seed=int(world_cup_seed),
                    )
                st.session_state["world_cup_2026_report"] = report
                st.session_state.pop("world_cup_2026_error", None)
                cargar_world_cup_2026_reporte.clear()
                st.success("La simulacion del Mundial 2026 quedo actualizada y persistida en artifacts.")
            except Exception as exc:  # noqa: BLE001
                st.session_state["world_cup_2026_error"] = str(exc)

        sync_result = _current_sync_result(fecha_simulacion)
        sync_result_date = st.session_state.get("sync_result_date")
        if sync_result:
            st.caption(
                f"Ultima sincronizacion: `{sync_result.get('status')}` | "
                f"proveedor `{sync_result.get('provider_used') or 'n/a'}` | "
                f"fecha `{sync_result_date or 'n/a'}`"
            )

    agenda_df = pd.DataFrame()
    if dashboard_view != "Simulacion Mundial 2026":
        try:
            agenda_df = cargar_agenda(str(fecha_simulacion))
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo leer la agenda diaria desde Supabase: {exc}")
            return

        _reset_stale_analysis_state(fecha_simulacion)
        agenda_df = _maybe_auto_sync_agenda(fecha_simulacion, agenda_df)
        agenda_df = _filter_noise_matches(
            agenda_df,
            lambda row_dict: (
                _has_historical_support(df_historico, row_dict["home_team"], fecha_simulacion, ventana),
                _has_historical_support(df_historico, row_dict["away_team"], fecha_simulacion, ventana),
            ),
        )

    st.markdown('<div class="alttez-shell">', unsafe_allow_html=True)
    _render_hero(fecha_simulacion, fase_torneo, agenda_df, tournament_mode)

    if dashboard_view == "Rendimiento del modelo":
        render_model_monitoring_dashboard()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if dashboard_view == "Simulacion Mundial 2026":
        world_cup_report = st.session_state.get("world_cup_2026_report") or cargar_world_cup_2026_reporte()
        world_cup_error = st.session_state.get("world_cup_2026_error")
        if world_cup_error:
            st.warning(world_cup_error)
        _render_world_cup_2026_dashboard(
            report=world_cup_report,
            fecha_simulacion=str(fecha_simulacion),
            iterations=int(world_cup_iterations),
            random_seed=int(world_cup_seed),
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    top_left, top_right = st.columns([1.45, 1], gap="large")

    with top_left:
        st.markdown(
            """
            <div class="alttez-panel">
                <div class="alttez-section-title">Agenda real del dia</div>
                <div class="alttez-section-copy">
                    La agenda ahora separa amistosos, Copa Mundial y cruces internacionales generales.
                    Si una fecha viene vacia, ALTTEZ intenta una sincronizacion live una vez por sesion antes de declararla sin agenda.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        _render_control_summary(ventana, prior_matches, recency_half_life_days, max_goles)

    selected_sync_result = _current_sync_result(fecha_simulacion)
    _render_sync_status(selected_sync_result, str(fecha_simulacion))
    _render_provider_health_panel()
    _render_recent_alerts_panel()

    if st.session_state.get("analysis_error"):
        st.warning(st.session_state["analysis_error"])

    left_col, right_col = st.columns([1.1, 1.25], gap="large")

    with left_col:
        st.markdown('<div class="alttez-subheader">Partidos internacionales</div>', unsafe_allow_html=True)
        if agenda_df.empty:
            sync_result = selected_sync_result
            if sync_result and sync_result.get("status") == "empty":
                st.markdown(
                    """
                    <div class="alttez-empty-state">
                        <div class="alttez-empty-title">Fecha auditada sin agenda disponible</div>
                        <div class="alttez-empty-copy">
                            La fecha fue auditada y hoy no hay amistosos o cruces internacionales cargados en el stack.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif sync_result and sync_result.get("status") == "error":
                st.markdown(
                    """
                    <div class="alttez-empty-state">
                        <div class="alttez-empty-title">La agenda sigue vacia por un fallo operativo</div>
                        <div class="alttez-empty-copy">
                            La ultima sincronizacion live no completo correctamente y no hay partidos listos para analisis.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="alttez-empty-state">
                        <div class="alttez-empty-title">Aun no hay agenda cargada</div>
                        <div class="alttez-empty-copy">
                            No hay partidos cargados para esta fecha. Puedes forzar una sincronizacion desde el panel lateral.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            agenda_general_df = _agenda_subset(agenda_df)
            amistosos_df = _agenda_subset(agenda_df, "friendly")
            mundial_df = _agenda_subset(agenda_df, "world_cup")

            tab_general, tab_amistosos, tab_mundial = st.tabs(["Agenda General", "Amistosos", "Copa Mundial"])

            with tab_general:
                _render_agenda_section(
                    title="Agenda General",
                    description="Vista completa de la fecha con cualquier cruce internacional detectado.",
                    agenda_df=agenda_general_df,
                    df_historico=df_historico,
                    fecha_simulacion=fecha_simulacion,
                    ventana=ventana,
                    prior_matches=prior_matches,
                    recency_half_life_days=recency_half_life_days,
                    fase_torneo=fase_torneo,
                    max_goles=max_goles,
                    tournament_mode=tournament_mode,
                    empty_message="No hay partidos analizables para esta fecha.",
                    section_key="general",
                )

            with tab_amistosos:
                _render_agenda_section(
                    title="Amistosos",
                    description="Partidos amistosos internacionales con foco operativo y acceso al analisis del motor.",
                    agenda_df=amistosos_df,
                    df_historico=df_historico,
                    fecha_simulacion=fecha_simulacion,
                    ventana=ventana,
                    prior_matches=prior_matches,
                    recency_half_life_days=recency_half_life_days,
                    fase_torneo=fase_torneo,
                    max_goles=max_goles,
                    tournament_mode=tournament_mode,
                    empty_message="No hay amistosos analizables para esta fecha.",
                    section_key="friendlies",
                )

            with tab_mundial:
                world_cup_groups = _world_cup_groups(mundial_df)
                if len(world_cup_groups) <= 1:
                    _render_agenda_section(
                        title="Copa Mundial",
                        description="Programacion del Mundial agrupada por grupo o fase cuando el proveedor trae ese contexto.",
                        agenda_df=mundial_df,
                        df_historico=df_historico,
                        fecha_simulacion=fecha_simulacion,
                        ventana=ventana,
                        prior_matches=prior_matches,
                        recency_half_life_days=recency_half_life_days,
                        fase_torneo=fase_torneo,
                        max_goles=max_goles,
                        tournament_mode=tournament_mode,
                        empty_message="No hay partidos del Mundial analizables para esta fecha.",
                        section_key="world-cup",
                    )
                else:
                    _render_agenda_section(
                        title="Copa Mundial",
                        description="Programacion del Mundial agrupada por grupo o fase cuando el proveedor trae ese contexto.",
                        agenda_df=world_cup_groups[0][1],
                        df_historico=df_historico,
                        fecha_simulacion=fecha_simulacion,
                        ventana=ventana,
                        prior_matches=prior_matches,
                        recency_half_life_days=recency_half_life_days,
                        fase_torneo=fase_torneo,
                        max_goles=max_goles,
                        tournament_mode=tournament_mode,
                        empty_message="No hay partidos del Mundial analizables para esta fecha.",
                        section_key=f"world-cup-{world_cup_groups[0][0]}",
                    )
                    for group_name, group_df in world_cup_groups[1:]:
                        with st.expander(f"Ver {group_name}", expanded=False):
                            _select_match_from_agenda(
                                agenda_df=group_df,
                                df_historico=df_historico,
                                fecha_simulacion=fecha_simulacion,
                                ventana=ventana,
                                prior_matches=prior_matches,
                                recency_half_life_days=recency_half_life_days,
                                fase_torneo=fase_torneo,
                                max_goles=max_goles,
                                tournament_mode=tournament_mode,
                                section_key=f"world-cup-{group_name}",
                            )

    with right_col:
        result = st.session_state.get("analysis_result")
        if result is None:
            st.markdown(
                """
                <div class="alttez-panel">
                    <div class="alttez-section-title">Motor listo</div>
                    <div class="alttez-section-copy">
                        Selecciona un partido de la agenda para disparar el analisis automatico de ALTTEZ.
                        El panel derecho se convertira en la mesa de decision del dia.
                    </div>
                    <div class="alttez-data-note">
                        Cuando lances un cruce, aqui apareceran probabilidades, fortalezas estructurales,
                        señales Klement y soporte estadistico del modelo activo.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            selected_match = result.get("match_context", {})
            st.markdown(
                f"""
                <div class="alttez-panel">
                    <div class="alttez-section-title">{html.escape(str(result['equipo_local']))} vs {html.escape(str(result['equipo_visitante']))}</div>
                    <div class="alttez-section-copy">
                        {html.escape(str(selected_match.get('tournament', 'Sin torneo')))}<br>
                        {html.escape(str(_competition_class_label(selected_match.get('competition_class'))))}
                        {html.escape(' | ' + str(selected_match.get('group_label'))) if selected_match.get('group_label') else ''}<br>
                        {html.escape(str(_format_kickoff(selected_match.get('kickoff_at'))))} | {html.escape(str(selected_match.get('city') or 'Ciudad pendiente'))}
                    </div>
                    <div style="margin-top:0.85rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
                        {_render_status_pill('Estado', selected_match.get('status'))}
                        {_render_status_pill('Formato', result['fase_torneo'])}
                        {_render_status_pill('Entorno', 'Neutral' if result['is_neutral'] else 'Local/Visitante')}
                        {_render_status_pill('Contexto', result.get('tournament_context_label'))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_cols = st.columns(4, gap="medium")
            metric_cols[0].metric(f"xG {result['equipo_local']}", f"{result['lambda_h']:.2f}")
            metric_cols[1].metric(f"xG {result['equipo_visitante']}", f"{result['lambda_a']:.2f}")
            metric_cols[2].metric(f"Historial {result['equipo_local']}", result["matches_local"])
            metric_cols[3].metric(f"Historial {result['equipo_visitante']}", result["matches_visitante"])

            support_cols = st.columns(4, gap="medium")
            support_cols[0].metric(
                f"Peso efectivo {result['equipo_local']}",
                f"{result['effective_matches_local']:.1f}",
            )
            support_cols[1].metric(
                f"Peso efectivo {result['equipo_visitante']}",
                f"{result['effective_matches_visitante']:.1f}",
            )
            support_cols[2].metric(
                f"Confianza {result['equipo_local']}",
                f"{result['confidence_local']:.0%}",
            )
            support_cols[3].metric(
                f"Confianza {result['equipo_visitante']}",
                f"{result['confidence_visitante']:.0%}",
            )

            _render_recommendation_panel(result)

            tab_pred, tab_map, tab_strength, tab_features = st.tabs(
                ["Pronostico", "Mapa de probabilidades", "Fortalezas", "Features Klement"]
            )

            with tab_pred:
                st.caption("Detalle de escenarios calculados por el optimizador")
                _render_probability_chart(result["result_summary_chart"])
                st.dataframe(result["recomendaciones"], width="stretch", hide_index=True)
                st.dataframe(result["scorelines"], width="stretch", hide_index=True)

            with tab_map:
                st.caption("Calor de marcadores y probabilidad 1X2")
                _render_score_heatmap(result["score_heatmap_df"])
                st.dataframe(result["result_summary"], width="stretch", hide_index=True)

            with tab_strength:
                st.caption("Comparativa de fuerza estructural, forma reciente y estabilidad muestral")
                _render_strength_chart(result["strength_chart_df"])
                _render_form_chart(result["form_chart_df"])
                st.dataframe(result["strength_df"], width="stretch", hide_index=True)

            with tab_features:
                feature_payload = result.get("xgboost_feature_payload") or {}
                tournament_context = result.get("tournament_context") or {}
                if not feature_payload:
                    st.markdown(
                        """
                        <div class="alttez-empty-state">
                            <div class="alttez-empty-title">Sin payload enriquecido en este analisis</div>
                            <div class="alttez-empty-copy">
                                El modelo activo no expuso features XGBoost para este cruce. Si hubo fallback,
                                el predictor operativo sigue respondiendo con seguridad.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    feature_df = (
                        pd.DataFrame(
                            [{"Feature": key, "Valor": value} for key, value in feature_payload.items()]
                        )
                        .sort_values("Feature")
                        .reset_index(drop=True)
                    )
                    st.caption("Payload de features usado por el challenger para este partido.")
                    st.dataframe(feature_df, width="stretch", hide_index=True)

                if tournament_context:
                    st.caption("Contexto de torneo aplicado al predictor.")
                    st.json(tournament_context)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
