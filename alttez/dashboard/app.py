from __future__ import annotations

from datetime import timedelta
import inspect

import altair as alt
import pandas as pd
import streamlit as st

from alttez.api.settings import load_local_env
from alttez.dashboard.analysis_helpers import (
    build_match_meta as _build_match_meta,
    build_recommendation_summary as _build_recommendation_summary,
    competition_class_label as _competition_class_label,
    filter_noise_matches as _filter_noise_matches,
    is_youth_or_u17_match as _is_youth_or_u17_match,
    safe_key as _safe_key,
    safe_text as _safe_text,
    support_level as _support_level,
    support_status as _support_status,
)
from alttez.dashboard.theme import build_brand_css
from alttez.experiments.tournament_oracle import build_probability_matrix, optimize_prediction
from alttez.features.poisson import PoissonEngine
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
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(28, 107, 255, 0.18), transparent 26%),
            radial-gradient(circle at top right, rgba(0, 228, 176, 0.10), transparent 22%),
            linear-gradient(180deg, #07111c 0%, #0a1422 46%, #09111b 100%);
        color: #edf3ff;
    }
    [data-testid="stSidebar"] {
        background: rgba(10, 18, 31, 0.96);
        border-right: 1px solid rgba(95, 119, 160, 0.18);
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    .alttez-shell {
        display: grid;
        gap: 1.25rem;
    }
    .alttez-hero {
        border: 1px solid rgba(107, 137, 190, 0.18);
        border-radius: 16px;
        padding: 1.4rem 1.45rem;
        background:
            linear-gradient(145deg, rgba(10, 24, 42, 0.96), rgba(9, 18, 32, 0.92)),
            linear-gradient(180deg, rgba(32, 98, 255, 0.08), transparent 60%);
        box-shadow: 0 18px 45px rgba(3, 8, 18, 0.28);
    }
    .alttez-kicker {
        color: #7d93ad;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        margin-bottom: 0.55rem;
    }
    .alttez-headline {
        font-size: 2.35rem;
        line-height: 1.05;
        font-weight: 700;
        color: #f4f7ff;
        margin: 0;
    }
    .alttez-subtle {
        color: #9cb1ca;
        font-size: 1rem;
        margin-top: 0.5rem;
        max-width: 920px;
    }
    .alttez-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .alttez-tile {
        border: 1px solid rgba(92, 118, 158, 0.18);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        background: rgba(10, 18, 31, 0.68);
        min-height: 92px;
    }
    .alttez-tile-label {
        color: #8ea3bc;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.45rem;
    }
    .alttez-tile-value {
        color: #f5f8ff;
        font-size: 1.38rem;
        font-weight: 700;
    }
    .alttez-tile-sub {
        color: #6fe5c7;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .alttez-panel {
        border: 1px solid rgba(92, 118, 158, 0.18);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        background: rgba(10, 18, 31, 0.72);
    }
    .alttez-section-title {
        font-size: 1.18rem;
        font-weight: 650;
        color: #eff4ff;
        margin-bottom: 0.35rem;
    }
    .alttez-section-copy {
        color: #97aac3;
        font-size: 0.93rem;
    }
    .alttez-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        font-size: 0.78rem;
        line-height: 1;
        border: 1px solid rgba(96, 124, 170, 0.18);
        background: rgba(15, 27, 45, 0.86);
        color: #d8e4f5;
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
        border: 1px solid rgba(90, 116, 158, 0.14);
        border-radius: 14px;
        background: rgba(9, 18, 31, 0.72);
        padding: 0.95rem 1rem;
        margin-bottom: 0.7rem;
    }
    .alttez-match-meta {
        color: #85a0bb;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .alttez-match-title {
        color: #f2f6ff;
        font-size: 1.1rem;
        font-weight: 650;
        margin-bottom: 0.32rem;
    }
    .alttez-match-subtitle {
        color: #93aac6;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(92, 118, 158, 0.18);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        background: rgba(10, 18, 31, 0.72);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(135deg, #2c7be5, #35d3ff);
        color: #06111d;
        border: 0;
        padding: 0.72rem 0.98rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(34, 110, 255, 0.22);
    }
    .stButton > button:hover {
        color: #06111d;
        background: linear-gradient(135deg, #5a92ea, #66ddff);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: rgba(12, 22, 38, 0.76);
        color: #a6b8cf;
        border: 1px solid rgba(92, 118, 158, 0.14);
        padding: 0.45rem 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(41, 118, 255, 0.18);
        color: #ecf5ff;
    }
    @media (max-width: 900px) {
        .alttez-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
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


def _get_team_profile(engine: PoissonEngine, team: str) -> dict:
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


def _build_poisson_engine(ventana: int, prior_matches: int, recency_half_life_days: int) -> PoissonEngine:
    signature = inspect.signature(PoissonEngine.__init__)
    kwargs = {
        "years_window": ventana,
        "prior_matches": prior_matches,
    }
    if "min_history_matches" in signature.parameters:
        kwargs["min_history_matches"] = 5
    if "recency_half_life_days" in signature.parameters:
        kwargs["recency_half_life_days"] = recency_half_life_days
    if "recent_form_matches" in signature.parameters:
        kwargs["recent_form_matches"] = 6
    return PoissonEngine(**kwargs)


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
        f"{label}: {_match_status_badge(status) if label == 'Estado' else status or 'N/D'}"
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
                Pronostico principal: <strong>{summary['primary_score']}</strong> con EV <strong>{summary['primary_ev']}</strong><br>
                Mercado mas probable: <strong>{market_label}</strong> ({market_probability})<br>
                Escenarios cercanos: {alternatives_text}
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
) -> dict:
    engine = _build_poisson_engine(ventana, prior_matches, recency_half_life_days)
    engine.fit(df, target_date=fecha_simulacion)

    lambda_h, lambda_a = engine.predict_xg(equipo_local, equipo_visitante, is_neutral=is_neutral)
    prob_matrix = build_probability_matrix(lambda_h, lambda_a, max_goals=max_goles)

    recomendaciones = optimize_prediction(prob_matrix, is_knockout=is_knockout).head(5).copy()
    recomendaciones["EV"] = recomendaciones["EV"].map(lambda value: round(value, 3))

    profile_local = _get_team_profile(engine, equipo_local)
    profile_visitante = _get_team_profile(engine, equipo_visitante)
    prediction_supported, support_message = _is_prediction_supported(profile_local, profile_visitante)
    if not prediction_supported:
        raise ValueError(
            f"{support_message} Equipos evaluados: `{equipo_local}` y `{equipo_visitante}`."
        )

    result_summary_chart = _build_result_summary(prob_matrix, equipo_local, equipo_visitante, format_percent=False)
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
        "lambda_h": lambda_h,
        "lambda_a": lambda_a,
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
            f"{support_label}"
            "</span>"
        )

        st.markdown(
            f"""
            <div class="alttez-match-card">
                <div class="alttez-match-meta">{kickoff_label} | {tournament_meta}</div>
                <div class="alttez-match-title">{row_dict['home_team']} vs {row_dict['away_team']}</div>
                <div class="alttez-match-subtitle">{city} | {_match_status_badge(row_dict.get('status'))}</div>
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
    empty_message: str,
    section_key: str,
) -> None:
    st.markdown(
        f"""
        <div class="alttez-panel">
            <div class="alttez-section-title">{title}</div>
            <div class="alttez-section-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if agenda_df.empty:
        st.info(empty_message)
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
        section_key=section_key,
    )


def _render_hero(fecha_simulacion, fase_torneo: str, agenda_df: pd.DataFrame) -> None:
    sync_result = _current_sync_result(fecha_simulacion) or {}
    sync_status = sync_result.get("status") or "idle"
    provider_used = sync_result.get("provider_used") or "stack live"
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
                    </div>
                </div>
                <div class="alttez-hero-aside">
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Fecha objetivo</span>
                        <span class="alttez-hero-value">{fecha_simulacion}</span>
                    </div>
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Partidos detectados</span>
                        <span class="alttez-hero-value">{len(agenda_df)}</span>
                    </div>
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Stack live</span>
                        <span class="alttez-hero-value">{provider_used}</span>
                    </div>
                    <div class="alttez-hero-row">
                        <span class="alttez-hero-label">Modo de puntuacion</span>
                        <span class="alttez-hero-value">{fase_torneo}</span>
                    </div>
                    <div class="alttez-section-copy">
                        {status_label} con agenda lista para analisis y auditoria operativa en tiempo real.
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
        <div class="alttez-panel">
            <div class="alttez-section-title">Blindaje del motor</div>
            <div class="alttez-section-copy">
                Ventana reciente: <strong>{ventana} anos</strong><br>
                Suavizado por muestra: <strong>{prior_matches}</strong><br>
                Vida media de recencia: <strong>{recency_half_life_days} dias</strong><br>
                Maximo de goles simulados: <strong>{max_goles}</strong><br>
                Equipos con menos de 5 partidos usan fuerza neutral por defecto.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
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
        ventana = st.slider("Ventana de analisis (anos)", min_value=1, max_value=10, value=4)
        prior_matches = st.slider("Suavizado por muestra", min_value=0, max_value=20, value=8)
        recency_half_life_days = st.slider("Vida media de recencia (dias)", min_value=90, max_value=720, value=365, step=15)
        max_goles = st.slider("Maximo de goles simulados", min_value=5, max_value=10, value=7)

        if st.button("Sincronizar fecha seleccionada", key="sync-live-fixtures"):
            result = _sync_selected_date(fecha_simulacion, reason="manual")
            _clear_analysis_feedback()
            if result.get("status") != "success":
                st.session_state.pop("analysis_result", None)

        sync_result = _current_sync_result(fecha_simulacion)
        sync_result_date = st.session_state.get("sync_result_date")
        if sync_result:
            st.caption(
                f"Ultima sincronizacion: `{sync_result.get('status')}` | "
                f"proveedor `{sync_result.get('provider_used') or 'n/a'}` | "
                f"fecha `{sync_result_date or 'n/a'}`"
            )

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
    _render_hero(fecha_simulacion, fase_torneo, agenda_df)

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
        st.subheader("Partidos internacionales")
        if agenda_df.empty:
            sync_result = selected_sync_result
            if sync_result and sync_result.get("status") == "empty":
                st.info("La fecha fue auditada y hoy no hay amistosos o cruces internacionales cargados en el stack.")
            elif sync_result and sync_result.get("status") == "error":
                st.warning("La agenda sigue vacia porque la ultima sincronizacion live fallo.")
            else:
                st.info("No hay partidos cargados aun para esta fecha.")
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
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            selected_match = result.get("match_context", {})
            st.markdown(
                f"""
                <div class="alttez-panel">
                    <div class="alttez-section-title">{result['equipo_local']} vs {result['equipo_visitante']}</div>
                    <div class="alttez-section-copy">
                        {selected_match.get('tournament', 'Sin torneo')}<br>
                        {_competition_class_label(selected_match.get('competition_class'))}
                        {' | ' + str(selected_match.get('group_label')) if selected_match.get('group_label') else ''}<br>
                        {_format_kickoff(selected_match.get('kickoff_at'))} | {selected_match.get('city') or 'Ciudad pendiente'}
                    </div>
                    <div style="margin-top:0.85rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
                        {_render_status_pill('Estado', selected_match.get('status'))}
                        {_render_status_pill('Formato', result['fase_torneo'])}
                        {_render_status_pill('Entorno', 'Neutral' if result['is_neutral'] else 'Local/Visitante')}
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

            tab_pred, tab_map, tab_strength = st.tabs(["Pronostico", "Mapa de probabilidades", "Fortalezas"])

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

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
