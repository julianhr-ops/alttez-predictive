from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from alttez.dashboard.analysis_helpers import competition_class_label
from alttez.operations.prediction_memory import (
    load_model_versions,
    load_prediction_metrics_by_competition,
    load_prediction_metrics_by_confidence,
    load_prediction_metrics_by_model,
    load_prediction_overview,
    load_recent_prediction_evaluations,
)


@st.cache_data(ttl=300)
def load_monitoring_snapshot() -> dict[str, object]:
    return {
        "overview": load_prediction_overview(),
        "by_model": load_prediction_metrics_by_model(),
        "by_competition": load_prediction_metrics_by_competition(),
        "by_confidence": load_prediction_metrics_by_confidence(),
        "recent_evaluations": load_recent_prediction_evaluations(),
        "model_versions": load_model_versions(),
    }


def _format_pct(series: pd.Series) -> pd.Series:
    return series.map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "N/D")


def _format_float(series: pd.Series, digits: int = 4) -> pd.Series:
    return series.map(lambda value: f"{float(value):.{digits}f}" if pd.notna(value) else "N/D")


def _render_model_versions_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Todavia no hay corridas champion/challenger registradas.")
        return

    display_df = df.copy()
    for column in ("brier_score", "log_loss"):
        if column in display_df.columns:
            display_df[column] = _format_float(display_df[column], digits=4)
    for column in ("accuracy_1x2", "exact_score_hit_rate"):
        if column in display_df.columns:
            display_df[column] = _format_pct(display_df[column])
    st.dataframe(display_df, width="stretch", hide_index=True)


def _render_competition_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No hay evaluaciones cerradas por competencia todavia.")
        return
    display_df = df.copy()
    if "competition_class" in display_df.columns:
        display_df["competition_class"] = display_df["competition_class"].map(competition_class_label)
    display_df["accuracy_1x2"] = _format_pct(display_df["accuracy_1x2"])
    display_df["brier_score"] = _format_float(display_df["brier_score"])
    display_df["log_loss"] = _format_float(display_df["log_loss"])
    st.dataframe(display_df, width="stretch", hide_index=True)


def _render_recent_predictions(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No hay predicciones evaluadas todavia.")
        return
    display_df = df.copy()
    for column in ("predicted_home_win_prob", "predicted_draw_prob", "predicted_away_win_prob"):
        if column in display_df.columns:
            display_df[column] = _format_pct(display_df[column])
    for column in ("brier_1x2", "log_loss_1x2"):
        if column in display_df.columns:
            display_df[column] = _format_float(display_df[column])
    st.dataframe(display_df, width="stretch", hide_index=True)


def _render_accuracy_by_model_chart(df: pd.DataFrame) -> None:
    if df.empty:
        return
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("model_version:N", sort="-y", title=None),
            y=alt.Y("accuracy_1x2:Q", axis=alt.Axis(format="%"), title="Accuracy 1X2"),
            color=alt.Color("model_version:N", legend=None, scale=alt.Scale(scheme="teals")),
            tooltip=[
                alt.Tooltip("model_version:N", title="Modelo"),
                alt.Tooltip("accuracy_1x2:Q", format=".2%"),
                alt.Tooltip("brier_score:Q", format=".4f"),
                alt.Tooltip("log_loss:Q", format=".4f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_brier_by_confidence_chart(df: pd.DataFrame) -> None:
    if df.empty:
        return
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("confidence_band:N", title="Banda de confianza"),
            y=alt.Y("brier_score:Q", title="Brier Score"),
            color=alt.Color("confidence_band:N", legend=None, scale=alt.Scale(scheme="blues")),
            tooltip=[
                "confidence_band:N",
                alt.Tooltip("brier_score:Q", format=".4f"),
                alt.Tooltip("accuracy_1x2:Q", format=".2%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_model_monitoring_dashboard() -> None:
    snapshot = load_monitoring_snapshot()
    overview = snapshot["overview"]
    by_model = snapshot["by_model"]
    by_competition = snapshot["by_competition"]
    by_confidence = snapshot["by_confidence"]
    recent_evaluations = snapshot["recent_evaluations"]
    model_versions = snapshot["model_versions"]

    st.markdown(
        """
        <div class="alttez-panel">
            <div class="alttez-section-title">Panel operativo del modelo</div>
            <div class="alttez-section-copy">
                Baseline champion, challenger en sombra y memoria point-in-time de cada prediccion emitida.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(5, gap="medium")
    metrics[0].metric("Predicciones evaluadas", int(overview["evaluated_predictions"]))
    metrics[1].metric("Brier Score", f"{overview['brier_score']:.4f}")
    metrics[2].metric("Log Loss", f"{overview['log_loss']:.4f}")
    metrics[3].metric("Accuracy 1X2", f"{overview['accuracy_1x2']:.1%}")
    metrics[4].metric("Exact hit", f"{overview['exact_score_hit_rate']:.1%}")

    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        st.markdown(
            """
            <div class="alttez-panel">
                <div class="alttez-section-title">Desempeno por modelo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_accuracy_by_model_chart(by_model)
        if not by_model.empty:
            display_df = by_model.copy()
            display_df["accuracy_1x2"] = _format_pct(display_df["accuracy_1x2"])
            display_df["exact_score_hit_rate"] = _format_pct(display_df["exact_score_hit_rate"])
            display_df["brier_score"] = _format_float(display_df["brier_score"])
            display_df["log_loss"] = _format_float(display_df["log_loss"])
            st.dataframe(display_df, width="stretch", hide_index=True)

    with top_right:
        st.markdown(
            """
            <div class="alttez-panel">
                <div class="alttez-section-title">Calibracion por confianza</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_brier_by_confidence_chart(by_confidence)
        if not by_confidence.empty:
            display_df = by_confidence.copy()
            display_df["accuracy_1x2"] = _format_pct(display_df["accuracy_1x2"])
            display_df["brier_score"] = _format_float(display_df["brier_score"])
            display_df["log_loss"] = _format_float(display_df["log_loss"])
            st.dataframe(display_df, width="stretch", hide_index=True)

    lower_left, lower_right = st.columns([1.1, 1.3], gap="large")
    with lower_left:
        st.markdown(
            """
            <div class="alttez-panel">
                <div class="alttez-section-title">Competencias</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_competition_table(by_competition)

        st.markdown(
            """
            <div class="alttez-panel" style="margin-top:1rem;">
                <div class="alttez-section-title">Registry champion vs challenger</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_model_versions_table(model_versions)

    with lower_right:
        st.markdown(
            """
            <div class="alttez-panel">
                <div class="alttez-section-title">Predicciones cerradas</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_recent_predictions(recent_evaluations)
