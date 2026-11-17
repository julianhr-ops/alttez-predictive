from __future__ import annotations

from typing import Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from alttez.dashboard.analysis_helpers import competition_class_label
from alttez.inference.world_cup_2026 import load_latest_world_cup_2026_report
from alttez.operations.expansion_policy import load_expansion_candidates

_OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def murphy_decomposition(
    probs: "np.ndarray | list",
    outcomes: "list[str]",
    bins: int = 10,
) -> dict[str, float]:
    """Descomposicion calibracion-refinamiento (Murphy) del Brier 1X2.

    Formulacion multiclase elegida: se opera sobre la **confianza de la clase
    predicha** (max prob) tratada como un problema binario "acerto / no acerto",
    consistente con el ECE existente (`inference/calibration.py`, que tambien usa
    max-prob). Para cada partido el evento binario es z = 1[clase_predicha == real]
    y la prediccion es la confianza c = max(prob_home, prob_draw, prob_away). Sobre
    ese par (c, z) se aplica la descomposicion clasica de Murphy del Brier binario:

        Brier_bin = Reliability - Resolution + Uncertainty

    donde, agrupando en `bins` por confianza:
        Reliability = (1/N) * sum_k n_k * (c_bar_k - o_bar_k)^2   (menor mejor)
        Resolution  = (1/N) * sum_k n_k * (o_bar_k - o_bar)^2     (mayor mejor)
        Uncertainty = o_bar * (1 - o_bar)   (Brier de la tasa base; irreducible)

    con c_bar_k la confianza media del bin, o_bar_k la tasa de acierto del bin,
    o_bar la tasa de acierto global y n_k el tamano del bin.

    Devuelve {reliability, resolution, uncertainty, brier} donde `brier` es el
    Brier binario de la confianza (no el Brier 1X2 de 3 clases). La identidad
    brier ~= reliability - resolution + uncertainty se cumple por construccion.
    """
    probs_arr = np.asarray(probs, dtype=float)
    neutral = {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0, "brier": 0.0}
    if probs_arr.size == 0 or len(outcomes) == 0:
        return neutral
    if probs_arr.ndim != 2 or probs_arr.shape[1] != 3:
        return neutral

    confidences = probs_arr.max(axis=1)
    predicted = probs_arr.argmax(axis=1)
    actual = np.array([_OUTCOME_INDEX.get(o, -1) for o in outcomes])
    correct = (predicted == actual).astype(float)

    total = len(confidences)
    base_rate = float(correct.mean())
    uncertainty = base_rate * (1.0 - base_rate)
    brier = float(np.mean((confidences - correct) ** 2))

    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability = 0.0
    resolution = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        if low == 0.0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        conf_mean = float(confidences[mask].mean())
        out_mean = float(correct[mask].mean())
        reliability += (count / total) * (conf_mean - out_mean) ** 2
        resolution += (count / total) * (out_mean - base_rate) ** 2

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": brier,
    }


def bootstrap_ci(
    values: "np.ndarray | list",
    n: int = 1000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int | None = 12345,
) -> tuple[float, float, float]:
    """Intervalo de confianza bootstrap (percentil) de un estadistico.

    Determinista: usa `numpy.random.default_rng(seed)` para el remuestreo, por lo
    que dos llamadas con el mismo `seed` devuelven exactamente el mismo intervalo.
    Devuelve (punto, bajo, alto) con el punto = estadistico sobre la muestra
    original y [bajo, alto] los percentiles alpha/2 y 1-alpha/2 de la distribucion
    bootstrap. Caso degenerado (vacio -> ceros; valores todos iguales -> ancho 0).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)

    point = float(statistic(arr))
    rng = np.random.default_rng(seed)
    size = arr.size
    samples = np.empty(n, dtype=float)
    for i in range(n):
        resample = arr[rng.integers(0, size, size=size)]
        samples[i] = float(statistic(resample))

    low = float(np.percentile(samples, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return (point, low, high)


def _format_ci(point: float, low: float, high: float, digits: int = 4) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _prediction_memory_unavailable_snapshot() -> dict[str, object]:
    return {
        "overview": {
            "evaluated_predictions": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "accuracy_1x2": 0.0,
            "exact_score_hit_rate": 0.0,
        },
        "by_model": pd.DataFrame(),
        "by_competition": pd.DataFrame(),
        "by_confidence": pd.DataFrame(),
        "recent_evaluations": pd.DataFrame(),
        "model_versions": pd.DataFrame(),
        "active_model": None,
        "calibration": {"models": pd.DataFrame(), "global_ece": None, "artifact": {}, "samples": 0},
        "closed_samples": pd.DataFrame(),
        "prediction_memory_error": (
            "No se pudo cargar el registry/memory del modelo. "
            "La interfaz sigue operativa con fallback local."
        ),
    }


@st.cache_data(ttl=300)
def load_monitoring_snapshot() -> dict[str, object]:
    try:
        from alttez.operations.prediction_memory import (
            load_active_model_record,
            load_calibration_overview,
            load_closed_evaluation_samples,
            load_model_versions,
            load_prediction_metrics_by_competition,
            load_prediction_metrics_by_confidence,
            load_prediction_metrics_by_model,
            load_prediction_overview,
            load_recent_prediction_evaluations,
        )
    except Exception:
        snapshot = _prediction_memory_unavailable_snapshot()
    else:
        try:
            snapshot = {
                "overview": load_prediction_overview(),
                "by_model": load_prediction_metrics_by_model(),
                "by_competition": load_prediction_metrics_by_competition(),
                "by_confidence": load_prediction_metrics_by_confidence(),
                "recent_evaluations": load_recent_prediction_evaluations(),
                "model_versions": load_model_versions(),
                "active_model": load_active_model_record(),
                "calibration": load_calibration_overview(),
                "closed_samples": load_closed_evaluation_samples(),
                "prediction_memory_error": None,
            }
        except Exception:
            snapshot = _prediction_memory_unavailable_snapshot()

    snapshot["expansion_candidates"] = load_expansion_candidates()
    snapshot["world_cup_report"] = load_latest_world_cup_2026_report()
    return snapshot


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


def _render_expansion_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Todavia no hay datos suficientes para evaluar expansion de competiciones.")
        return

    display_df = df.copy()
    if "competition_class" in display_df.columns:
        display_df["competition_class"] = display_df["competition_class"].map(competition_class_label)
    for column in ("analyzable_rate", "accuracy_1x2"):
        if column in display_df.columns:
            display_df[column] = _format_pct(display_df[column])
    for column in ("brier_score", "log_loss"):
        if column in display_df.columns:
            display_df[column] = _format_float(display_df[column])
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


def _render_active_model_card(record: dict[str, object] | None) -> None:
    if not record:
        st.info("No hay modelo productivo activo registrado en Supabase.")
        return

    st.markdown(
        f"""
        <div class="alttez-panel">
            <div class="alttez-section-title">Modelo activo en inferencia</div>
            <div class="alttez-section-copy">
                <strong>{record.get('model_version', 'N/D')}</strong> ({record.get('model_family', 'N/D')})<br>
                Estado productivo: {record.get('production_status', 'shadow')} | Activacion: {record.get('activation_mode', 'manual')}<br>
                Activado en: {record.get('activated_at') or 'Pendiente'}<br>
                Motivo: {record.get('promotion_reason') or 'Sin comentario de gobernanza'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_world_cup_panel(report: dict[str, object] | None) -> None:
    if not report:
        st.info(
            "Todavia no hay una simulacion oficial del Mundial 2026 materializada. "
            "Ejecuta `scripts/run_world_cup_2026_simulation.py` cuando el draw oficial este cargado."
        )
        return

    probabilities = pd.DataFrame(report.get("team_probabilities") or [])
    if probabilities.empty:
        st.info("La simulacion del Mundial 2026 no devolvio probabilidades agregadas.")
        return

    display_df = probabilities.head(12).copy()
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
            display_df[column] = _format_pct(display_df[column])

    st.markdown(
        f"""
        <div class="alttez-panel">
            <div class="alttez-section-title">Mundial 2026</div>
            <div class="alttez-section-copy">
                Simulacion Monte Carlo con {report.get('iterations', 0)} iteraciones usando
                <strong>{report.get('model_version', 'N/D')}</strong> ({report.get('model_family', 'N/D')}).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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


def _render_calibration_panel(calibration: dict[str, object] | None) -> None:
    calibration = calibration or {}
    artifact = calibration.get("artifact") or {}
    temperature = artifact.get("temperature", 1.0)
    is_neutral = artifact.get("is_neutral", True)
    fitted_on = artifact.get("fitted_on_model", "—")
    ece_before = artifact.get("ece_before")
    ece_after = artifact.get("ece_after")
    estado = "Neutra (sin artefacto)" if is_neutral else f"Activa · T={float(temperature):.3f}"
    detalle = ""
    if ece_before is not None and ece_after is not None:
        detalle = f"<br>ECE del comité al ajustar: {float(ece_before):.4f} → {float(ece_after):.4f} (modelo: {fitted_on})"

    st.markdown(
        f"""
        <div class="alttez-panel">
            <div class="alttez-section-title">Calibración de probabilidad</div>
            <div class="alttez-section-copy">
                Estado: <strong>{estado}</strong>. La calibración (temperature scaling) se aplica
                al consenso del comité; se omite cuando el comité defiere al campeón.{detalle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    models = calibration.get("models")
    if isinstance(models, pd.DataFrame) and not models.empty:
        display_df = models.copy()
        display_df["ece"] = _format_float(display_df["ece"])
        st.dataframe(display_df, width="stretch", hide_index=True)
    else:
        st.info("Aún no hay predicciones cerradas para medir calibración (ECE).")


def _closed_samples_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if df is None or df.empty:
        return np.empty((0, 3)), []
    needed = {"prob_home", "prob_draw", "prob_away", "actual_outcome"}
    if not needed.issubset(set(df.columns)):
        return np.empty((0, 3)), []
    frame = df.dropna(subset=["actual_outcome"]).copy()
    if frame.empty:
        return np.empty((0, 3)), []
    triplets = frame[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    outcomes = [str(o) for o in frame["actual_outcome"].tolist()]
    return triplets, outcomes


def _render_murphy_diagnostic_card(closed_samples: pd.DataFrame) -> None:
    triplets, outcomes = _closed_samples_to_arrays(closed_samples)
    if len(outcomes) == 0:
        st.info("Aún no hay predicciones cerradas para descomponer el Brier (Murphy).")
        return

    decomposition = murphy_decomposition(triplets, outcomes)
    st.markdown(
        """
        <div class="alttez-panel">
            <div class="alttez-section-title">Diagnóstico Brier (descomposición de Murphy)</div>
            <div class="alttez-section-copy">
                Sobre la confianza de la clase predicha (máx. prob), consistente con el ECE.
                Brier ≈ Fiabilidad − Resolución + Incertidumbre.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3, gap="medium")
    cols[0].metric(
        "Fiabilidad",
        f"{decomposition['reliability']:.4f}",
        help="Reliability: distancia entre confianza y acierto real. Menor es mejor (0 = calibración perfecta).",
    )
    cols[1].metric(
        "Resolución",
        f"{decomposition['resolution']:.4f}",
        help="Resolution: cuánto separan las predicciones la tasa base. Mayor es mejor.",
    )
    cols[2].metric(
        "Incertidumbre",
        f"{decomposition['uncertainty']:.4f}",
        help="Uncertainty: Brier de la tasa base (irreducible).",
    )
    st.caption(
        f"Brier de confianza: {decomposition['brier']:.4f} ≈ "
        f"{decomposition['reliability']:.4f} − {decomposition['resolution']:.4f} + "
        f"{decomposition['uncertainty']:.4f}"
    )


def _render_bootstrap_ci_card(closed_samples: pd.DataFrame) -> None:
    if closed_samples is None or closed_samples.empty:
        st.info("Aún no hay predicciones cerradas para estimar intervalos de confianza (bootstrap).")
        return

    st.markdown(
        """
        <div class="alttez-panel">
            <div class="alttez-section-title">Intervalos de confianza (bootstrap)</div>
            <div class="alttez-section-copy">
                Remuestreo bootstrap (1000 réplicas, IC 95%) sobre las evaluaciones cerradas por partido.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_specs = [
        ("brier_1x2", "Brier 1X2"),
        ("rps_1x2", "RPS 1X2"),
        ("log_loss_1x2", "Log Loss 1X2"),
    ]
    cols = st.columns(len(metric_specs), gap="medium")
    for index, (column, label) in enumerate(metric_specs):
        if column not in closed_samples.columns:
            cols[index].metric(label, "N/D")
            continue
        values = pd.to_numeric(closed_samples[column], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            cols[index].metric(label, "N/D")
            continue
        point, low, high = bootstrap_ci(values)
        cols[index].metric(label, f"{point:.4f}", help=f"IC 95%: [{low:.4f}, {high:.4f}]")
        cols[index].caption(_format_ci(point, low, high))


def render_model_monitoring_dashboard() -> None:
    snapshot = load_monitoring_snapshot()
    overview = snapshot["overview"]
    by_model = snapshot["by_model"]
    by_competition = snapshot["by_competition"]
    by_confidence = snapshot["by_confidence"]
    recent_evaluations = snapshot["recent_evaluations"]
    model_versions = snapshot["model_versions"]
    expansion_candidates = snapshot["expansion_candidates"]
    active_model = snapshot["active_model"]
    world_cup_report = snapshot["world_cup_report"]
    closed_samples = snapshot.get("closed_samples", pd.DataFrame())
    prediction_memory_error = snapshot.get("prediction_memory_error")

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

    if prediction_memory_error:
        st.warning(str(prediction_memory_error))

    metrics = st.columns(6, gap="medium")
    metrics[0].metric("Predicciones evaluadas", int(overview["evaluated_predictions"]))
    metrics[1].metric("Brier Score", f"{overview['brier_score']:.4f}")
    metrics[2].metric("RPS 1X2", f"{overview.get('rps', 0.0):.4f}", help="Ranked Probability Score: penaliza errores ordinales (H<D<A). Menor es mejor.")
    metrics[3].metric("Log Loss", f"{overview['log_loss']:.4f}")
    metrics[4].metric("Accuracy 1X2", f"{overview['accuracy_1x2']:.1%}")
    metrics[5].metric("Exact hit", f"{overview['exact_score_hit_rate']:.1%}")

    _render_active_model_card(active_model)

    diag_left, diag_right = st.columns(2, gap="large")
    with diag_left:
        _render_murphy_diagnostic_card(closed_samples)
    with diag_right:
        _render_bootstrap_ci_card(closed_samples)

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
        _render_calibration_panel(snapshot.get("calibration"))

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

        st.markdown(
            """
            <div class="alttez-panel" style="margin-top:1rem;">
                <div class="alttez-section-title">Politica de expansion</div>
                <div class="alttez-section-copy">
                    Solo conviene abrir mas competiciones cuando hay volumen historico, partidos analizables y calibracion cerrada suficiente.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_expansion_table(expansion_candidates)

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

        st.markdown(
            """
            <div class="alttez-panel" style="margin-top:1rem;">
                <div class="alttez-section-title">Proyeccion Mundial 2026</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_world_cup_panel(world_cup_report)
