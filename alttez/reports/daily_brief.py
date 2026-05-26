from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.db import read_dataframe
from alttez.api.settings import load_local_env
from alttez.dashboard.analysis_helpers import (
    build_recommendation_summary,
    competition_class_label,
    filter_noise_matches,
    safe_text,
    support_level,
)
from alttez.experiments.tournament_oracle import build_probability_matrix, optimize_prediction
from alttez.features.poisson import PoissonEngine
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import list_matches_for_date


DEFAULT_REPORT_DIR = Path("artifacts/reports")
DEFAULT_MAX_GOALS = 7
DEFAULT_DELIVERY_TIMEOUT_SECONDS = 10


def _build_engine(df: pd.DataFrame, target_date: str) -> PoissonEngine:
    engine = PoissonEngine(
        years_window=4,
        prior_matches=8,
        min_history_matches=5,
        recency_half_life_days=365,
        recent_form_matches=6,
    )
    engine.fit(df, target_date=target_date)
    return engine


def _get_team_profile(engine: PoissonEngine, team: str) -> dict[str, float]:
    return engine.get_team_profile(team)


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
    )


def _is_prediction_supported(profile_local: dict[str, float], profile_visitante: dict[str, float]) -> tuple[bool, str | None]:
    local_matches = float(profile_local.get("matches", 0.0))
    away_matches = float(profile_visitante.get("matches", 0.0))
    local_effective = float(profile_local.get("effective_matches", 0.0))
    away_effective = float(profile_visitante.get("effective_matches", 0.0))

    if local_matches <= 0 and away_matches <= 0:
        return False, "Sin historial compatible para ambos equipos."
    if local_effective <= 0 and away_effective <= 0:
        return False, "Sin soporte estadistico efectivo para ambos equipos."
    return True, None


def _is_knockout_stage(stage: str | None) -> bool:
    lowered = safe_text(stage).lower()
    keywords = (
        "quarter-final",
        "quarterfinal",
        "semi-final",
        "semifinal",
        "final",
        "round of 16",
        "octavos",
        "cuartos",
        "semifinal",
        "elimin",
        "playoff",
    )
    return any(keyword in lowered for keyword in keywords)


def _support_lookup_from_engine(engine: PoissonEngine, row_dict: dict[str, Any]) -> tuple[bool, bool]:
    home_profile = _get_team_profile(engine, row_dict["home_team"])
    away_profile = _get_team_profile(engine, row_dict["away_team"])
    return home_profile.get("matches", 0.0) > 0, away_profile.get("matches", 0.0) > 0


def _load_batch_summary(target_date: date, database_url: str | None = None) -> dict[str, Any]:
    summary = {
        "certification_status": "unknown",
        "certified": False,
        "total_records": 0,
        "certified_records": 0,
        "rejected_records": 0,
        "coverage_pct": 0.0,
        "provider_used": None,
        "quarantine_reason": None,
    }
    try:
        df = read_dataframe(
            """
            select
                batch_id,
                target_date,
                provider_used,
                certification_status,
                certified,
                total_records,
                certified_records,
                rejected_records,
                coverage_pct,
                quarantine_reason,
                created_at
            from public.certification_batches
            where target_date = :target_date
            order by created_at desc
            limit 1
            """,
            params={"target_date": target_date.isoformat()},
            database_url=database_url,
        )
    except Exception:
        return summary

    if df.empty:
        return summary

    row = df.iloc[0].to_dict()
    summary.update(
        {
            "batch_id": row.get("batch_id"),
            "provider_used": row.get("provider_used"),
            "certification_status": row.get("certification_status") or "unknown",
            "certified": bool(row.get("certified", False)),
            "total_records": int(row.get("total_records") or 0),
            "certified_records": int(row.get("certified_records") or 0),
            "rejected_records": int(row.get("rejected_records") or 0),
            "coverage_pct": float(row.get("coverage_pct") or 0.0),
            "quarantine_reason": row.get("quarantine_reason"),
        }
    )
    return summary


def _load_alert_summary(target_date: date, database_url: str | None = None) -> dict[str, Any]:
    base = {
        "total": 0,
        "critical": 0,
        "warning": 0,
        "info": 0,
        "items": [],
    }
    try:
        df = read_dataframe(
            """
            select
                created_at,
                alert_level,
                alert_code,
                title,
                message,
                provider_used,
                delivery_status,
                delivery_channel
            from public.alert_events
            where target_date = :target_date
            order by created_at desc
            limit 25
            """,
            params={"target_date": target_date.isoformat()},
            database_url=database_url,
        )
    except Exception:
        return base

    if df.empty:
        return base

    base["total"] = int(len(df))
    base["critical"] = int((df["alert_level"] == "critical").sum())
    base["warning"] = int((df["alert_level"] == "warning").sum())
    base["info"] = int((df["alert_level"] == "info").sum())
    base["items"] = df.fillna("").to_dict(orient="records")
    return base


def _decision_tier(support_label: str, avg_confidence: float, best_market_probability: float) -> str:
    if support_label == "Analizable" and avg_confidence >= 0.6 and best_market_probability >= 0.45:
        return "Alta prioridad"
    if support_label in {"Analizable", "Soporte parcial"} and avg_confidence >= 0.35:
        return "Seguimiento"
    return "Baja confianza"


def _format_probability(value: float) -> str:
    return f"{value:.1%}"


def _build_match_brief(row_dict: dict[str, Any], engine: PoissonEngine, max_goals: int = DEFAULT_MAX_GOALS) -> dict[str, Any]:
    home_team = row_dict["home_team"]
    away_team = row_dict["away_team"]
    profile_local = _get_team_profile(engine, home_team)
    profile_visitante = _get_team_profile(engine, away_team)
    prediction_supported, support_message = _is_prediction_supported(profile_local, profile_visitante)
    support_label = support_level(profile_local.get("matches", 0) > 0, profile_visitante.get("matches", 0) > 0)

    brief = {
        "match_id": row_dict.get("id"),
        "kickoff_at": safe_text(row_dict.get("kickoff_at"), "Por confirmar"),
        "tournament": safe_text(row_dict.get("tournament"), "Sin torneo"),
        "competition_class": safe_text(row_dict.get("competition_class"), "other_international"),
        "competition_label": competition_class_label(row_dict.get("competition_class")),
        "season_label": safe_text(row_dict.get("season_label")),
        "stage": safe_text(row_dict.get("stage"), "Fase por confirmar"),
        "group_label": safe_text(row_dict.get("group_label")),
        "status": safe_text(row_dict.get("status"), "scheduled"),
        "home_team": home_team,
        "away_team": away_team,
        "city": safe_text(row_dict.get("city"), "Ciudad pendiente"),
        "venue_name": safe_text(row_dict.get("venue_name")),
        "is_neutral": bool(row_dict.get("is_neutral", False)),
        "support_label": support_label,
        "prediction_supported": prediction_supported,
        "support_message": support_message,
        "matches_local": int(profile_local.get("matches", 0.0)),
        "matches_visitante": int(profile_visitante.get("matches", 0.0)),
        "effective_matches_local": float(profile_local.get("effective_matches", 0.0)),
        "effective_matches_visitante": float(profile_visitante.get("effective_matches", 0.0)),
        "confidence_local": float(profile_local.get("confidence", 0.0)),
        "confidence_visitante": float(profile_visitante.get("confidence", 0.0)),
        "recent_points_local": float(profile_local.get("recent_points_per_match", 0.0)),
        "recent_points_visitante": float(profile_visitante.get("recent_points_per_match", 0.0)),
        "recent_goal_diff_local": float(profile_local.get("recent_goal_diff_per_match", 0.0)),
        "recent_goal_diff_visitante": float(profile_visitante.get("recent_goal_diff_per_match", 0.0)),
    }

    if not prediction_supported:
        brief.update(
            {
                "lambda_h": None,
                "lambda_a": None,
                "xg_total": None,
                "primary_score": "N/D",
                "primary_ev": None,
                "best_market": "Sin soporte",
                "best_market_probability": 0.0,
                "prob_home": None,
                "prob_draw": None,
                "prob_away": None,
                "decision_tier": "Baja confianza",
                "alternatives": [],
            }
        )
        return brief

    lambda_h, lambda_a = engine.predict_xg(home_team, away_team, is_neutral=brief["is_neutral"])
    prob_matrix = build_probability_matrix(lambda_h, lambda_a, max_goals=max_goals)
    recommendations = optimize_prediction(prob_matrix, is_knockout=_is_knockout_stage(brief["stage"])).head(4).copy()
    recommendations["EV"] = recommendations["EV"].map(lambda value: round(float(value), 3))
    result_summary = _build_result_summary(prob_matrix, home_team, away_team)
    recommendation_summary = build_recommendation_summary(
        {
            "recomendaciones": recommendations,
            "result_summary": result_summary,
        }
    )

    best_market = recommendation_summary.get("best_market") or {}
    probabilities = {
        row["Mercado"]: float(row["Probabilidad"])
        for row in result_summary.to_dict(orient="records")
    }
    avg_confidence = (brief["confidence_local"] + brief["confidence_visitante"]) / 2
    best_market_probability = float(best_market.get("Probabilidad", 0.0) or 0.0)

    brief.update(
        {
            "lambda_h": lambda_h,
            "lambda_a": lambda_a,
            "xg_total": round(lambda_h + lambda_a, 2),
            "primary_score": recommendation_summary.get("primary_score", "N/D"),
            "primary_ev": recommendation_summary.get("primary_ev"),
            "best_market": safe_text(best_market.get("Mercado"), "Por definir"),
            "best_market_probability": best_market_probability,
            "prob_home": probabilities.get(f"Victoria {home_team}"),
            "prob_draw": probabilities.get("Empate"),
            "prob_away": probabilities.get(f"Victoria {away_team}"),
            "decision_tier": _decision_tier(support_label, avg_confidence, best_market_probability),
            "alternatives": recommendations.iloc[1:4].to_dict(orient="records"),
        }
    )
    return brief


def _summarize_brief(match_briefs: list[dict[str, Any]], batch_summary: dict[str, Any], alert_summary: dict[str, Any]) -> dict[str, Any]:
    analyzable = [item for item in match_briefs if item["support_label"] == "Analizable"]
    partial = [item for item in match_briefs if item["support_label"] == "Soporte parcial"]
    unsupported = [item for item in match_briefs if item["support_label"] == "Sin historial suficiente"]
    high_priority = [item for item in match_briefs if item["decision_tier"] == "Alta prioridad"]
    ranked = sorted(
        match_briefs,
        key=lambda item: (
            0 if item["decision_tier"] == "Alta prioridad" else 1 if item["decision_tier"] == "Seguimiento" else 2,
            -((item["confidence_local"] + item["confidence_visitante"]) / 2),
            -(item.get("best_market_probability") or 0.0),
        ),
    )

    return {
        "total_matches": len(match_briefs),
        "analyzable_matches": len(analyzable),
        "partial_matches": len(partial),
        "unsupported_matches": len(unsupported),
        "high_priority_matches": len(high_priority),
        "critical_alerts": alert_summary["critical"],
        "warning_alerts": alert_summary["warning"],
        "certification_status": batch_summary.get("certification_status", "unknown"),
        "coverage_pct": float(batch_summary.get("coverage_pct") or 0.0),
        "ranked_matches": ranked,
    }


def build_daily_brief_payload(
    target_date: date,
    *,
    database_url: str | None = None,
    include_statuses: tuple[str, ...] = ("scheduled", "in_progress", "finished", "cancelled"),
) -> dict[str, Any]:
    load_local_env(override=True)
    agenda_df = list_matches_for_date(target_date, include_statuses=include_statuses, database_url=database_url)
    historical_loader = InternationalLoader()
    historical_df = historical_loader.fetch_data()
    engine = _build_engine(historical_df, target_date.isoformat())
    filtered_agenda = filter_noise_matches(
        agenda_df,
        lambda row_dict: _support_lookup_from_engine(engine, row_dict),
    )

    match_briefs = [_build_match_brief(row._asdict(), engine) for row in filtered_agenda.itertuples(index=False)]
    batch_summary = _load_batch_summary(target_date, database_url=database_url)
    alert_summary = _load_alert_summary(target_date, database_url=database_url)
    summary = _summarize_brief(match_briefs, batch_summary, alert_summary)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date.isoformat(),
        "batch_summary": batch_summary,
        "alert_summary": alert_summary,
        "summary": summary,
        "matches": match_briefs,
    }


def _fmt_float(value: Any, digits: int = 2, default: str = "N/D") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any, default: str = "N/D") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return _format_probability(float(value))


def _delivery_timeout_seconds() -> int:
    timeout_raw = os.getenv("ALTTEZ_DAILY_BRIEF_DELIVERY_TIMEOUT_SECONDS", "").strip()
    if not timeout_raw:
        timeout_raw = os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_TIMEOUT_SECONDS", "10").strip() or "10"
    try:
        return max(1, int(timeout_raw))
    except ValueError:
        return DEFAULT_DELIVERY_TIMEOUT_SECONDS


def _telegram_delivery_config() -> tuple[str | None, str | None]:
    token = os.getenv("ALTTEZ_DAILY_BRIEF_TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("ALTTEZ_DAILY_BRIEF_TELEGRAM_CHAT_ID", "").strip()
    return (token or None, chat_id or None)


def _delivery_channel() -> str:
    configured = os.getenv("ALTTEZ_DAILY_BRIEF_DELIVERY_CHANNEL", "").strip().lower()
    if configured in {"telegram", "webhook", "auto"}:
        return configured
    return "auto"


def _build_telegram_summary_text(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    batch = payload["batch_summary"]
    top_matches = summary.get("ranked_matches", [])[:5]
    lines = [
        f"<b>ALTTEZ Daily Decision Brief</b> - {payload['target_date']}",
        "",
        (
            f"<b>Lote:</b> {html.escape(safe_text(batch.get('certification_status'), 'unknown'))} | "
            f"<b>Cobertura:</b> {html.escape(_fmt_float(summary.get('coverage_pct')))}% | "
            f"<b>Proveedor:</b> {html.escape(safe_text(batch.get('provider_used'), 'N/D'))}"
        ),
        (
            f"<b>Agenda:</b> {summary.get('total_matches', 0)} partidos | "
            f"{summary.get('analyzable_matches', 0)} analizables | "
            f"{summary.get('high_priority_matches', 0)} alta prioridad"
        ),
        (
            f"<b>Alertas:</b> {summary.get('critical_alerts', 0)} criticas | "
            f"{summary.get('warning_alerts', 0)} advertencias"
        ),
    ]
    if batch.get("quarantine_reason"):
        lines.extend(["", f"<b>Quarantine:</b> {html.escape(safe_text(batch.get('quarantine_reason')))}"])

    if top_matches:
        lines.extend(["", "<b>Top partidos del dia</b>"])
        for index, match in enumerate(top_matches, start=1):
            lines.append(
                (
                    f"{index}. <b>{html.escape(match['home_team'])} vs {html.escape(match['away_team'])}</b> | "
                    f"{html.escape(match['competition_label'])} | "
                    f"{html.escape(safe_text(match.get('kickoff_at'), 'Por confirmar'))}\n"
                    f"   Mercado: {html.escape(safe_text(match.get('best_market'), 'N/D'))} "
                    f"({_fmt_pct(match.get('best_market_probability'))}) | "
                    f"Pronostico: {html.escape(safe_text(match.get('primary_score'), 'N/D'))} | "
                    f"EV: {_fmt_float(match.get('primary_ev'), digits=3)} | "
                    f"Soporte: {html.escape(match['support_label'])}"
                )
            )
    else:
        lines.extend(["", "No hay partidos analizables para esta fecha."])

    lines.extend(["", "Se adjuntan el brief HTML y el payload JSON para revision."])
    return "\n".join(lines)


def _telegram_api_url(bot_token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{bot_token}/{method}"


def _send_telegram_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str | None]:
    data = parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = request.Request(
        _telegram_api_url(bot_token, "sendMessage"),
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=_delivery_timeout_seconds()) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Telegram sendMessage returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)


def _build_multipart_payload(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----ALTTEZBoundary{datetime.now(timezone.utc).timestamp()}".replace(".", "")
    file_bytes = file_path.read_bytes()
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks: list[bytes] = []

    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), boundary


def _send_telegram_document(bot_token: str, chat_id: str, file_path: Path, caption: str | None = None) -> tuple[bool, str | None]:
    fields = {"chat_id": chat_id}
    if caption:
        fields["caption"] = caption
        fields["parse_mode"] = "HTML"
    payload, boundary = _build_multipart_payload(fields, "document", file_path)
    req = request.Request(
        _telegram_api_url(bot_token, "sendDocument"),
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=_delivery_timeout_seconds()) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Telegram sendDocument returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)


def render_daily_brief_html(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    batch = payload["batch_summary"]
    alerts = payload["alert_summary"]
    matches = payload["matches"]

    priority_rows = []
    for match in summary["ranked_matches"]:
        priority_rows.append(
            """
            <tr>
              <td>{kickoff}</td>
              <td>{competition}</td>
              <td><strong>{home}</strong> vs <strong>{away}</strong><div class="muted">{tournament}</div></td>
              <td>{support}</td>
              <td>{decision}</td>
              <td>{xg}</td>
              <td>{market}</td>
              <td>{market_prob}</td>
              <td>{prediction}</td>
              <td>{ev}</td>
            </tr>
            """.format(
                kickoff=html.escape(safe_text(match["kickoff_at"], "Por confirmar")),
                competition=html.escape(match["competition_label"]),
                home=html.escape(match["home_team"]),
                away=html.escape(match["away_team"]),
                tournament=html.escape(match["tournament"]),
                support=html.escape(match["support_label"]),
                decision=html.escape(match["decision_tier"]),
                xg=html.escape(_fmt_float(match.get("xg_total"))),
                market=html.escape(match.get("best_market") or "N/D"),
                market_prob=html.escape(_fmt_pct(match.get("best_market_probability"))),
                prediction=html.escape(match.get("primary_score") or "N/D"),
                ev=html.escape(_fmt_float(match.get("primary_ev"), digits=3)),
            )
        )

    detail_cards = []
    for match in matches:
        alt_text = " | ".join(
            f"{item['Pronostico']} (EV {_fmt_float(item['EV'], digits=3)})"
            for item in match.get("alternatives", [])
        ) or "Sin escenarios alternativos"
        risk_note = match.get("support_message") or "Cobertura aceptable para analisis diario."
        detail_cards.append(
            """
            <article class="match-card">
              <div class="match-topline">{competition} | {tournament} | {stage}{group}</div>
              <h3>{home} vs {away}</h3>
              <div class="meta">{kickoff} | {city} | {status}</div>
              <div class="pill-row">
                <span class="pill">{support}</span>
                <span class="pill accent">{decision}</span>
              </div>
              <div class="grid">
                <div class="metric"><span>xG local</span><strong>{xg_home}</strong></div>
                <div class="metric"><span>xG visitante</span><strong>{xg_away}</strong></div>
                <div class="metric"><span>1X2 principal</span><strong>{market}</strong></div>
                <div class="metric"><span>Probabilidad</span><strong>{market_prob}</strong></div>
                <div class="metric"><span>Pronostico ALTTEZ</span><strong>{prediction}</strong></div>
                <div class="metric"><span>EV esperado</span><strong>{ev}</strong></div>
                <div class="metric"><span>Confianza media</span><strong>{avg_confidence}</strong></div>
                <div class="metric"><span>Historial util</span><strong>{history}</strong></div>
              </div>
              <p><strong>Forma reciente:</strong> {home} {home_form} pts/j, GD {home_gd} | {away} {away_form} pts/j, GD {away_gd}</p>
              <p><strong>Escenarios cercanos:</strong> {alternatives}</p>
              <p><strong>Riesgo operativo:</strong> {risk}</p>
            </article>
            """.format(
                competition=html.escape(match["competition_label"]),
                tournament=html.escape(match["tournament"]),
                stage=html.escape(match["stage"]),
                group=(f" | {html.escape(match['group_label'])}" if match.get("group_label") else ""),
                home=html.escape(match["home_team"]),
                away=html.escape(match["away_team"]),
                kickoff=html.escape(safe_text(match["kickoff_at"], "Por confirmar")),
                city=html.escape(match["city"]),
                status=html.escape(match["status"]),
                support=html.escape(match["support_label"]),
                decision=html.escape(match["decision_tier"]),
                xg_home=html.escape(_fmt_float(match.get("lambda_h"))),
                xg_away=html.escape(_fmt_float(match.get("lambda_a"))),
                market=html.escape(match.get("best_market") or "N/D"),
                market_prob=html.escape(_fmt_pct(match.get("best_market_probability"))),
                prediction=html.escape(match.get("primary_score") or "N/D"),
                ev=html.escape(_fmt_float(match.get("primary_ev"), digits=3)),
                avg_confidence=html.escape(
                    _fmt_pct(((match["confidence_local"] + match["confidence_visitante"]) / 2))
                ),
                history=html.escape(
                    f"{match['matches_local']} / {match['matches_visitante']} partidos; "
                    f"{_fmt_float(match['effective_matches_local'])} / {_fmt_float(match['effective_matches_visitante'])} efectivos"
                ),
                home_form=html.escape(_fmt_float(match["recent_points_local"], digits=2)),
                home_gd=html.escape(_fmt_float(match["recent_goal_diff_local"], digits=2)),
                away_form=html.escape(_fmt_float(match["recent_points_visitante"], digits=2)),
                away_gd=html.escape(_fmt_float(match["recent_goal_diff_visitante"], digits=2)),
                alternatives=html.escape(alt_text),
                risk=html.escape(risk_note),
            )
        )

    alert_rows = []
    for item in alerts["items"][:8]:
        alert_rows.append(
            """
            <tr>
              <td>{level}</td>
              <td>{title}</td>
              <td>{message}</td>
              <td>{channel}</td>
              <td>{delivery}</td>
            </tr>
            """.format(
                level=html.escape(safe_text(item.get("alert_level")).upper()),
                title=html.escape(safe_text(item.get("title"))),
                message=html.escape(safe_text(item.get("message"))),
                channel=html.escape(safe_text(item.get("delivery_channel"), "dashboard")),
                delivery=html.escape(safe_text(item.get("delivery_status"), "skipped")),
            )
        )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ALTTEZ Daily Decision Brief | {html.escape(payload['target_date'])}</title>
  <style>
    :root {{
      --graphite: #111315;
      --ivory: #f5f7f8;
      --copper: #c27442;
      --blue: #3f6db5;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --muted: #667280;
      --line: rgba(17, 19, 21, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Inter", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(194, 116, 66, 0.14), transparent 28%),
        linear-gradient(180deg, #f8f8f6 0%, #eff2f4 100%);
      color: var(--graphite);
    }}
    .page {{ max-width: 1320px; margin: 0 auto; padding: 32px 24px 56px; }}
    .hero {{
      background: linear-gradient(145deg, #0f1215, #171a1d);
      color: white;
      border-radius: 22px;
      padding: 28px;
      box-shadow: 0 20px 60px rgba(17, 19, 21, 0.18);
    }}
    .eyebrow {{ color: rgba(255,255,255,0.68); text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; }}
    h1 {{ margin: 10px 0 8px; font-size: 42px; line-height: 1.02; font-family: "Segoe UI Semibold", "Inter", sans-serif; }}
    .subcopy {{ color: rgba(255,255,255,0.78); max-width: 920px; font-size: 15px; }}
    .stats {{
      display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 22px;
    }}
    .stat {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 14px;
    }}
    .stat span {{ display: block; color: rgba(255,255,255,0.62); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat strong {{ display: block; margin-top: 6px; font-size: 24px; }}
    .layout {{ display: grid; gap: 18px; margin-top: 22px; }}
    .panel {{
      background: rgba(255,255,255,0.82);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      backdrop-filter: blur(12px);
    }}
    h2 {{ margin: 0 0 14px; font-size: 24px; }}
    h3 {{ margin: 0 0 8px; font-size: 22px; }}
    .muted {{ color: var(--muted); }}
    .summary-grid {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--line); vertical-align: top; font-size: 14px; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 14px; }}
    .pill {{
      display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px;
      background: rgba(63, 109, 181, 0.12); color: #12355d; font-size: 12px; font-weight: 600;
    }}
    .pill.accent {{ background: rgba(194, 116, 66, 0.14); color: #7c431d; }}
    .match-card {{
      background: rgba(255,255,255,0.9);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      margin-top: 16px;
    }}
    .match-topline {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .meta {{ color: var(--muted); font-size: 14px; margin-bottom: 10px; }}
    .grid {{
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 14px 0;
    }}
    .metric {{
      background: rgba(17, 19, 21, 0.03);
      border: 1px solid rgba(17, 19, 21, 0.08);
      border-radius: 14px;
      padding: 12px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    .footer {{ margin-top: 24px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 1080px) {{
      .stats, .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .summary-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 720px) {{
      .stats, .grid {{ grid-template-columns: 1fr; }}
      .page {{ padding: 18px; }}
      h1 {{ font-size: 30px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">ALTTEZ Daily Decision Brief</div>
      <h1>Agenda analitica del {html.escape(payload['target_date'])}</h1>
      <p class="subcopy">
        Informe automatico generado por ALTTEZ despues de la ingesta, certificacion y evaluacion matematica del dia.
        Su objetivo es entregar una mesa de decisiones lista para analistas sin depender de la interfaz para descubrir el dato.
      </p>
      <div class="stats">
        <div class="stat"><span>Partidos</span><strong>{summary['total_matches']}</strong></div>
        <div class="stat"><span>Analizables</span><strong>{summary['analyzable_matches']}</strong></div>
        <div class="stat"><span>Alta prioridad</span><strong>{summary['high_priority_matches']}</strong></div>
        <div class="stat"><span>Estado lote</span><strong>{html.escape(safe_text(summary['certification_status'], 'N/D'))}</strong></div>
        <div class="stat"><span>Cobertura lote</span><strong>{html.escape(_fmt_float(summary['coverage_pct']))}%</strong></div>
        <div class="stat"><span>Alertas</span><strong>{alerts['critical']}/{alerts['warning']}/{alerts['info']}</strong></div>
      </div>
    </section>

    <section class="layout">
      <div class="summary-grid">
        <div class="panel">
          <h2>Agenda priorizada</h2>
          <table>
            <thead>
              <tr>
                <th>Hora</th>
                <th>Tipo</th>
                <th>Partido</th>
                <th>Soporte</th>
                <th>Decision</th>
                <th>xG total</th>
                <th>Mercado</th>
                <th>Prob.</th>
                <th>Pronostico</th>
                <th>EV</th>
              </tr>
            </thead>
            <tbody>
              {''.join(priority_rows) or '<tr><td colspan="10">No hay partidos analizables para esta fecha.</td></tr>'}
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Estado operativo</h2>
          <p><strong>Proveedor activo:</strong> {html.escape(safe_text(batch.get('provider_used'), 'N/D'))}</p>
          <p><strong>Certificacion:</strong> {html.escape(safe_text(batch.get('certification_status'), 'unknown'))}</p>
          <p><strong>Registros del lote:</strong> {batch.get('total_records', 0)} total | {batch.get('certified_records', 0)} certificados | {batch.get('rejected_records', 0)} rechazados</p>
          <p><strong>Cobertura:</strong> {html.escape(_fmt_float(batch.get('coverage_pct')))}%</p>
          <p><strong>Quarantine reason:</strong> {html.escape(safe_text(batch.get('quarantine_reason'), 'Sin motivo activo'))}</p>
          <h3 style="margin-top:18px;">Alertas recientes</h3>
          <table>
            <thead>
              <tr>
                <th>Nivel</th>
                <th>Titulo</th>
                <th>Mensaje</th>
                <th>Canal</th>
                <th>Entrega</th>
              </tr>
            </thead>
            <tbody>
              {''.join(alert_rows) or '<tr><td colspan="5">No hay alertas registradas para esta fecha.</td></tr>'}
            </tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <h2>Detalle por partido</h2>
        {''.join(detail_cards) or '<p>No hay detalle disponible.</p>'}
      </div>
    </section>

    <div class="footer">
      Generado en {html.escape(payload['generated_at'])} | Fuente de agenda: Supabase canonical matches | Motor: Poisson ALTTEZ
    </div>
  </div>
</body>
</html>
"""


def write_daily_brief(payload: dict[str, Any], report_dir: str | Path = DEFAULT_REPORT_DIR) -> tuple[Path, Path]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    stem = f"daily-brief-{payload['target_date']}"
    html_path = report_path / f"{stem}.html"
    json_path = report_path / f"{stem}.json"
    html_path.write_text(render_daily_brief_html(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return html_path, json_path


def _dispatch_daily_brief_webhook(payload: dict[str, Any], html_content: str, webhook_url: str | None = None) -> tuple[bool, str | None]:
    target_url = webhook_url or os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_URL", "").strip()
    if not target_url:
        return False, "No daily brief webhook configured."

    body = {
        "source": "alttez-daily-brief",
        "generated_at": payload["generated_at"],
        "target_date": payload["target_date"],
        "summary": payload["summary"],
        "batch_summary": payload["batch_summary"],
        "alert_summary": payload["alert_summary"],
        "matches": payload["matches"],
        "html_report": html_content,
    }
    data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
    timeout_raw = os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_TIMEOUT_SECONDS", "10").strip() or "10"
    try:
        timeout = max(1, int(timeout_raw))
    except ValueError:
        timeout = 10

    req = request.Request(
        target_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            if 200 <= int(getattr(response, "status", 200)) < 300:
                return True, None
            return False, f"Webhook returned HTTP {getattr(response, 'status', 'unknown')}."
    except error.URLError as exc:
        return False, str(exc)


def _dispatch_daily_brief_telegram(payload: dict[str, Any], html_path: Path, json_path: Path) -> tuple[bool, str | None]:
    bot_token, chat_id = _telegram_delivery_config()
    if not bot_token or not chat_id:
        return False, "Telegram bot token or chat id is not configured."

    summary_text = _build_telegram_summary_text(payload)
    sent, error_message = _send_telegram_message(bot_token, chat_id, summary_text)
    if not sent:
        return False, error_message

    html_sent, html_error = _send_telegram_document(
        bot_token,
        chat_id,
        html_path,
        caption=f"<b>ALTTEZ brief</b> {html.escape(payload['target_date'])}",
    )
    if not html_sent:
        return False, html_error

    json_sent, json_error = _send_telegram_document(bot_token, chat_id, json_path)
    if not json_sent:
        return False, json_error

    return True, None


def dispatch_daily_brief(
    payload: dict[str, Any],
    html_content: str,
    html_path: Path,
    json_path: Path,
    webhook_url: str | None = None,
) -> tuple[bool, str | None, str]:
    delivery_channel = _delivery_channel()
    telegram_token, telegram_chat_id = _telegram_delivery_config()

    if delivery_channel in {"telegram", "auto"} and telegram_token and telegram_chat_id:
        delivered, error_message = _dispatch_daily_brief_telegram(payload, html_path, json_path)
        return delivered, error_message, "telegram"

    delivered, error_message = _dispatch_daily_brief_webhook(payload, html_content, webhook_url=webhook_url)
    if delivered or delivery_channel == "webhook" or webhook_url or os.getenv("ALTTEZ_DAILY_BRIEF_WEBHOOK_URL", "").strip():
        return delivered, error_message, "webhook"

    return False, "No Telegram or webhook delivery channel configured.", "none"


def generate_daily_brief(
    target_date: date,
    *,
    database_url: str | None = None,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    deliver: bool = False,
) -> dict[str, Any]:
    payload = build_daily_brief_payload(target_date, database_url=database_url)
    html_path, json_path = write_daily_brief(payload, report_dir=report_dir)
    delivery = {
        "delivery_attempted": False,
        "delivery_status": "skipped",
        "delivery_channel": "none",
        "delivery_error": None,
    }
    if deliver:
        delivery["delivery_attempted"] = True
        html_content = html_path.read_text(encoding="utf-8")
        delivered, error_message, channel = dispatch_daily_brief(payload, html_content, html_path, json_path)
        delivery["delivery_status"] = "delivered" if delivered else "failed"
        delivery["delivery_channel"] = channel
        delivery["delivery_error"] = error_message

    return {
        "payload": payload,
        "html_path": str(html_path),
        "json_path": str(json_path),
        **delivery,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the ALTTEZ Daily Decision Brief.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--database-url", default=None, help="Explicit database URL. Falls back to environment.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Output directory for the report artifacts.")
    parser.add_argument("--deliver", action="store_true", help="Deliver the brief to the configured webhook.")
    return parser.parse_args()


def main() -> int:
    load_local_env(override=True)
    args = parse_args()
    result = generate_daily_brief(
        date.fromisoformat(args.date),
        database_url=args.database_url,
        report_dir=args.report_dir,
        deliver=args.deliver,
    )
    print(f"daily_brief_html={result['html_path']}")
    print(f"daily_brief_json={result['json_path']}")
    print(f"daily_brief_delivery_status={result['delivery_status']}")
    print(f"daily_brief_delivery_channel={result['delivery_channel']}")
    if result.get("delivery_error"):
        print(f"daily_brief_delivery_error={result['delivery_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
