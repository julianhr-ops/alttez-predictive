from __future__ import annotations

import html
import json
from typing import Any

from alttez.dashboard.analysis_helpers import safe_text
from alttez.reports.brand import ISOTYPE_SVG
from .formatting import (
    _fmt_expected_goals,
    _fmt_float,
    _fmt_pct,
    _fmt_points_profile,
    match_goal_totals,
)


def _decision_class(decision_tier: str) -> str:
    tier = (decision_tier or "").lower()
    if "alta" in tier:
        return "is-high"
    if "seguimiento" in tier:
        return "is-watch"
    return "is-low"


def _reliability_class(label: str) -> str:
    normalized = (label or "").lower()
    if "alta" in normalized:
        return "rel-high"
    if "media" in normalized:
        return "rel-medium"
    if "baja" in normalized:
        return "rel-low"
    return "rel-none"


def _prob_value(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value <= 1.0:
        value *= 100.0
    return round(value, 1)


def render_daily_brief_html(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    batch = payload["batch_summary"]
    alerts = payload["alert_summary"]
    matches = payload["matches"]

    priority_rows = []
    for match in summary["ranked_matches"]:
        over_prob, _ = match_goal_totals(match)
        priority_rows.append(
            """
            <tr>
              <td class="nowrap">{kickoff}</td>
              <td class="nowrap">{competition}</td>
              <td class="match-cell"><strong>{home}</strong> <span class="vs">vs</span> <strong>{away}</strong><div class="muted">{tournament}</div></td>
              <td>{support}</td>
              <td><span class="tag {reliability_class}">{reliability_label} {reliability_score}</span></td>
              <td><span class="tag {decision_class}">{decision}</span></td>
              <td class="num">{xg}</td>
              <td>{market}</td>
              <td class="num">{market_prob}</td>
              <td class="nowrap">{prediction}</td>
              <td class="num">{ev}</td>
            </tr>
            """.format(
                kickoff=html.escape(safe_text(match["kickoff_at"], "Por confirmar")),
                competition=html.escape(match["competition_label"]),
                home=html.escape(match["home_team"]),
                away=html.escape(match["away_team"]),
                tournament=html.escape(match["tournament"]),
                support=html.escape(match["support_label"]),
                reliability_label=html.escape((match.get("reliability") or {}).get("label", "N/D")),
                reliability_score=html.escape(
                    f"{(match.get('reliability') or {}).get('score', 0)}/100"
                    if (match.get("reliability") or {}).get("label") not in (None, "Sin soporte confiable")
                    else ""
                ),
                reliability_class=_reliability_class((match.get("reliability") or {}).get("label", "")),
                decision=html.escape(match["decision_tier"]),
                decision_class=_decision_class(match["decision_tier"]),
                xg=html.escape(
                    f"{_fmt_float(match.get('xg_total'))} (O2.5 {_fmt_pct(over_prob)})"
                    if over_prob is not None
                    else _fmt_float(match.get("xg_total"))
                ),
                market=html.escape(match.get("best_market") or "N/D"),
                market_prob=html.escape(_fmt_pct(match.get("best_market_probability"))),
                prediction=html.escape(match.get("primary_score") or "N/D"),
                ev=html.escape(_fmt_float(match.get("primary_ev"), digits=3)),
            )
        )

    score_chart_payload: dict[str, dict[str, Any]] = {}
    result_chart_payload: dict[str, dict[str, Any]] = {}
    detail_cards = []
    for match_index, match in enumerate(matches):
        over_prob, total_goals = match_goal_totals(match)
        chart_id = f"score-chart-{match_index}"
        result_id = f"result-chart-{match_index}"
        score_chart_payload[chart_id] = {
            "rows": match.get("score_probability_chart", []),
            "primary": safe_text(match.get("primary_score"), ""),
        }
        result_chart_payload[result_id] = {
            "home": _prob_value(match.get("prob_home")),
            "draw": _prob_value(match.get("prob_draw")),
            "away": _prob_value(match.get("prob_away")),
            "home_team": match["home_team"],
            "away_team": match["away_team"],
        }
        alt_text = " | ".join(
            f"{item['Pronostico']} (EV {_fmt_float(item['EV'], digits=3)})"
            for item in match.get("alternatives", [])
        ) or "Sin escenarios alternativos"
        candidate_text = " | ".join(
            f"{item['score']} [10 pts {_fmt_pct(item.get('full_points_probability'))}, "
            f"EV {_fmt_float(item.get('expected_points'), digits=3)}; "
            f"{_fmt_points_profile(item.get('points_probability'))}]"
            for item in match.get("scoreline_candidates", [])[:6]
        ) or "Sin pronosticos alternativos"
        risk_note = match.get("support_message") or "Cobertura aceptable para analisis diario."
        committee_note = (
            "Discordancia alta entre modelos: " + ", ".join(match.get("member_primary_scores", []))
            if match.get("discordance_warning")
            else "Consenso de comité activo."
        )
        detail_cards.append(
            """
            <article class="match-card">
              <div class="match-topline">{competition} &middot; {tournament} &middot; {stage}{group}</div>
              <div class="match-head">
                <h3>{home} <span class="vs">vs</span> {away}</h3>
                <span class="tag {decision_class}">{decision}</span>
              </div>
              <div class="meta">{kickoff} &middot; {city} &middot; {status}</div>
              <div class="pill-row">
                <span class="pill">{support}</span>
                <span class="tag {reliability_class}">Fiabilidad: {reliability_label} {reliability_score}</span>
              </div>

              <div class="goals-strip">
                <span>&#9917; Goles esperados <strong>{xg_total_disp}</strong></span>
                <span>Over 2.5: <strong>{over_25}</strong></span>
                <span>Total m&aacute;s probable: <strong>{total_goals} goles</strong></span>
                <span class="muted">Marcador sugerido {prediction} (pick, no la media)</span>
              </div>

              <div class="chart-block">
                <div class="chart-title">Resultado 1X2</div>
                <div class="result-chart-wrap"><canvas id="{result_id}" height="92" aria-label="Probabilidad 1X2"></canvas></div>
                <div class="legend">
                  <span><i class="dot dh"></i>{home} {p_home}%</span>
                  <span><i class="dot dd"></i>Empate {p_draw}%</span>
                  <span><i class="dot da"></i>{away} {p_away}%</span>
                </div>
              </div>

              <div class="grid">
                <div class="metric"><span>xG local</span><strong>{xg_home}</strong></div>
                <div class="metric"><span>xG visitante</span><strong>{xg_away}</strong></div>
                <div class="metric"><span>1X2 principal</span><strong>{market}</strong></div>
                <div class="metric"><span>Probabilidad</span><strong>{market_prob}</strong></div>
                <div class="metric"><span>Marcador sugerido (10 pts)</span><strong>{prediction}</strong></div>
                <div class="metric"><span>Prob. marcador</span><strong>{score_prob}</strong></div>
                <div class="metric"><span>Total mas probable</span><strong>{total_goals} goles</strong></div>
                <div class="metric"><span>Score estrategia EV</span><strong>{ev_score}</strong></div>
                <div class="metric"><span>EV pick exacto</span><strong>{ev}</strong></div>
                <div class="metric"><span>Confianza media</span><strong>{avg_confidence}</strong></div>
                <div class="metric"><span>Historial util</span><strong>{history}</strong></div>
              </div>

              <div class="chart-block">
                <div class="chart-title">Distribucion de marcadores</div>
                <div class="score-chart-wrap"><canvas id="{chart_id}" height="170" aria-label="Distribucion de probabilidad de marcadores"></canvas></div>
              </div>

              <div class="notes">
                <p><strong>Forma reciente:</strong> {home} {home_form} pts/j, GD {home_gd} &middot; {away} {away_form} pts/j, GD {away_gd}</p>
                <p><strong>Pronosticos alternativos:</strong> {candidates}</p>
                <p><strong>Estrategias EV alternativas:</strong> {alternatives}</p>
                <p><strong>Comite predictivo:</strong> {committee}</p>
                <p><strong>Fiabilidad ({reliability_label} {reliability_score}):</strong> {reliability_reasons}</p>
                <p><strong>Riesgo operativo:</strong> {risk}</p>
              </div>
            </article>
            """.format(
                chart_id=chart_id,
                result_id=result_id,
                competition=html.escape(match["competition_label"]),
                tournament=html.escape(match["tournament"]),
                stage=html.escape(match["stage"]),
                group=(f" &middot; {html.escape(match['group_label'])}" if match.get("group_label") else ""),
                home=html.escape(match["home_team"]),
                away=html.escape(match["away_team"]),
                decision=html.escape(match["decision_tier"]),
                decision_class=_decision_class(match["decision_tier"]),
                kickoff=html.escape(safe_text(match["kickoff_at"], "Por confirmar")),
                city=html.escape(match["city"]),
                status=html.escape(match["status"]),
                support=html.escape(match["support_label"]),
                p_home=html.escape(str(_prob_value(match.get("prob_home")))),
                p_draw=html.escape(str(_prob_value(match.get("prob_draw")))),
                p_away=html.escape(str(_prob_value(match.get("prob_away")))),
                xg_home=html.escape(_fmt_float(match.get("lambda_h"))),
                xg_away=html.escape(_fmt_float(match.get("lambda_a"))),
                xg_total_disp=html.escape(_fmt_expected_goals(match.get("lambda_h"), match.get("lambda_a"))),
                over_25=html.escape(_fmt_pct(over_prob)),
                total_goals=html.escape(str(total_goals) if total_goals is not None else "N/D"),
                market=html.escape(match.get("best_market") or "N/D"),
                market_prob=html.escape(_fmt_pct(match.get("best_market_probability"))),
                prediction=html.escape(match.get("primary_score") or "N/D"),
                score_prob=html.escape(_fmt_pct(match.get("primary_score_probability"))),
                ev_score=html.escape(safe_text(match.get("ev_optimal_score"), "N/D")),
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
                candidates=html.escape(candidate_text),
                committee=html.escape(committee_note),
                reliability_class=_reliability_class((match.get("reliability") or {}).get("label", "")),
                reliability_label=html.escape((match.get("reliability") or {}).get("label", "N/D")),
                reliability_score=html.escape(
                    f"({(match.get('reliability') or {}).get('score', 0)}/100)"
                    if (match.get("reliability") or {}).get("label") not in (None, "Sin soporte confiable")
                    else ""
                ),
                reliability_reasons=html.escape(
                    " ".join((match.get("reliability") or {}).get("reasons", []) or ["Sin detalle."])
                ),
                risk=html.escape(risk_note),
            )
        )

    alert_rows = []
    for item in alerts["items"][:8]:
        alert_rows.append(
            """
            <tr>
              <td><span class="tag lvl-{level_low}">{level}</span></td>
              <td>{title}</td>
              <td>{message}</td>
              <td class="nowrap">{channel}</td>
              <td class="nowrap">{delivery}</td>
            </tr>
            """.format(
                level=html.escape(safe_text(item.get("alert_level")).upper()),
                level_low=html.escape(safe_text(item.get("alert_level"), "info").lower()),
                title=html.escape(safe_text(item.get("title"))),
                message=html.escape(safe_text(item.get("message"))),
                channel=html.escape(safe_text(item.get("delivery_channel"), "dashboard")),
                delivery=html.escape(safe_text(item.get("delivery_status"), "skipped")),
            )
        )

    score_chart_json = json.dumps(score_chart_payload, ensure_ascii=False)
    result_chart_json = json.dumps(result_chart_payload, ensure_ascii=False)
    fallback_banner = (
        '<div class="fallback-banner">Brief servido por modelo de respaldo (Poisson): '
        f"{html.escape(safe_text(payload.get('fallback_reason'), 'modelo activo no disponible'))}</div>"
        if payload.get("used_fallback")
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="color-scheme" content="light dark">
  <title>ALTTEZ Daily Decision Brief | {html.escape(payload['target_date'])}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --graphite: #0f1215;
      --ink: #151a1f;
      --ivory: #f5f7f8;
      --copper: #c27a42;
      --blue: #3f6db5;
      --teal: #2bb3a3;
      --success: #16a34a;
      --warning: #d97706;
      --danger: #dc2626;
      --muted: #6b7682;
      --line: rgba(17, 19, 21, 0.10);
      --card: rgba(255,255,255,0.94);
      --shadow: 0 10px 30px rgba(17, 19, 21, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ -webkit-text-size-adjust: 100%; }}
    body {{
      margin: 0;
      font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
      background:
        radial-gradient(1200px 400px at 100% -10%, rgba(194, 116, 66, 0.10), transparent 60%),
        radial-gradient(900px 380px at -10% 0%, rgba(63, 109, 181, 0.10), transparent 55%),
        linear-gradient(180deg, #f8f8f6 0%, #eef1f3 100%);
      color: var(--graphite);
      -webkit-font-smoothing: antialiased;
    }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: clamp(14px, 4vw, 34px); }}
    a {{ color: var(--blue); }}

    .hero {{
      position: relative; overflow: hidden;
      background: linear-gradient(150deg, #0d1013 0%, #181c20 60%, #1d2024 100%);
      color: #fff; border-radius: 24px; padding: clamp(20px, 4vw, 32px);
      box-shadow: 0 24px 60px rgba(17, 19, 21, 0.22);
    }}
    .hero::after {{
      content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px;
      background: radial-gradient(circle, rgba(194,116,66,0.28), transparent 70%); pointer-events: none;
    }}
    .brand-lockup {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }}
    .brand-lockup svg {{ width: 30px; height: 30px; }}
    .brand-name {{ font-family: "Sora","Inter",sans-serif; font-weight: 700; letter-spacing: 0.14em; font-size: 18px; }}
    .eyebrow {{ color: rgba(255,255,255,0.66); text-transform: uppercase; letter-spacing: 0.18em; font-size: 11px; }}
    h1 {{ margin: 10px 0 8px; font-size: clamp(26px, 6vw, 40px); line-height: 1.05; font-weight: 700; }}
    .subcopy {{ color: rgba(255,255,255,0.74); max-width: 760px; font-size: clamp(13px, 2.6vw, 15px); line-height: 1.55; }}
    .fallback-banner {{
      margin-top: 16px; padding: 10px 14px; border-radius: 12px; font-size: 13px; font-weight: 600;
      background: rgba(217, 119, 6, 0.18); border: 1px solid rgba(217,119,6,0.5); color: #ffd9a3;
    }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-top: 22px; }}
    .stat {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 13px 14px; }}
    .stat span {{ display: block; color: rgba(255,255,255,0.6); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat strong {{ display: block; margin-top: 6px; font-size: clamp(20px, 4vw, 26px); font-variant-numeric: tabular-nums; }}

    .layout {{ display: grid; gap: 16px; margin-top: 20px; }}
    .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: clamp(16px, 3vw, 22px); box-shadow: var(--shadow); }}
    h2 {{ margin: 0 0 14px; font-size: clamp(18px, 4vw, 23px); }}
    h3 {{ margin: 0; font-size: clamp(17px, 4vw, 21px); }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .summary-grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; }}

    .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 12px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 560px; }}
    th, td {{ text-align: left; padding: 11px 10px; border-bottom: 1px solid var(--line); vertical-align: top; font-size: 13.5px; }}
    th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); position: sticky; top: 0; background: var(--card); }}
    tbody tr:hover {{ background: rgba(63, 109, 181, 0.05); }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    td.nowrap, .nowrap {{ white-space: nowrap; }}
    .match-cell strong {{ font-weight: 650; }}
    .vs {{ color: var(--muted); font-weight: 400; font-size: 0.85em; padding: 0 2px; }}

    .tag {{ display: inline-block; padding: 4px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.02em; white-space: nowrap; }}
    .tag.is-high {{ background: rgba(22,163,74,0.14); color: #15803d; }}
    .tag.is-watch {{ background: rgba(217,119,6,0.16); color: #b45309; }}
    .tag.is-low {{ background: rgba(107,118,130,0.14); color: #51606e; }}
    .tag.rel-high {{ background: rgba(22,163,74,0.16); color: #15803d; }}
    .tag.rel-medium {{ background: rgba(217,119,6,0.16); color: #b45309; }}
    .tag.rel-low {{ background: rgba(220,38,38,0.13); color: #b91c1c; }}
    .tag.rel-none {{ background: rgba(107,118,130,0.16); color: #51606e; }}
    .tag.lvl-critical {{ background: rgba(220,38,38,0.14); color: #b91c1c; }}
    .tag.lvl-warning {{ background: rgba(217,119,6,0.16); color: #b45309; }}
    .tag.lvl-info {{ background: rgba(63,109,181,0.14); color: #1d4e80; }}

    .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 6px; }}
    .pill {{ display: inline-flex; align-items: center; padding: 5px 11px; border-radius: 999px; background: rgba(63, 109, 181, 0.10); color: #12355d; font-size: 12px; font-weight: 600; }}
    .goals-strip {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: baseline; margin: 10px 0 4px; padding: 10px 13px; border-radius: 13px; background: rgba(43, 179, 163, 0.10); border: 1px solid rgba(43, 179, 163, 0.28); font-size: 14px; }}
    .goals-strip strong {{ font-variant-numeric: tabular-nums; font-size: 16px; color: var(--teal); }}

    .match-card {{ background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: clamp(15px, 3vw, 20px); margin-top: 14px; box-shadow: var(--shadow); }}
    .match-card:first-child {{ margin-top: 0; }}
    .match-topline {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; }}
    .match-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin: 6px 0 2px; }}
    .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 4px; }}

    .chart-block {{ margin: 14px 0; }}
    .chart-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin-bottom: 8px; }}
    .result-chart-wrap {{ padding: 6px 4px; }}
    .score-chart-wrap {{ padding: 10px; border: 1px solid var(--line); border-radius: 14px; background: rgba(247,249,250,0.7); }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; font-size: 12.5px; color: #2a3138; font-weight: 600; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .dot {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
    .dot.dh {{ background: var(--blue); }}
    .dot.dd {{ background: #94a3b8; }}
    .dot.da {{ background: var(--copper); }}

    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 12px 0; }}
    .metric {{ background: rgba(17, 19, 21, 0.025); border: 1px solid rgba(17, 19, 21, 0.07); border-radius: 13px; padding: 11px 12px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 11.5px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 17px; font-variant-numeric: tabular-nums; }}

    .notes p {{ margin: 6px 0; font-size: 13.5px; line-height: 1.5; }}
    .footer {{ margin-top: 22px; color: var(--muted); font-size: 12.5px; text-align: center; }}

    @media (max-width: 980px) {{
      .summary-grid {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    }}
    @media (max-width: 640px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .match-head {{ align-items: flex-start; }}
      .notes p {{ font-size: 13px; }}
    }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: linear-gradient(180deg, #0e1113 0%, #121518 100%); color: #e8ecef; }}
      .panel, .match-card {{ background: rgba(28,32,36,0.92); border-color: rgba(255,255,255,0.08); }}
      .metric {{ background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.07); }}
      .score-chart-wrap {{ background: rgba(255,255,255,0.03); border-color: rgba(255,255,255,0.08); }}
      th {{ background: rgba(28,32,36,0.92); }}
      .muted, .metric span, .chart-title, .match-topline, .meta {{ color: #9aa6b1; }}
      .notes strong {{ color: #f1f4f6; }}
      .legend {{ color: #cdd5dc; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="brand-lockup">
        {ISOTYPE_SVG}
        <span class="brand-name">ALT<span style="color:var(--copper);">TEZ</span></span>
      </div>
      <div class="eyebrow">ALTTEZ Daily Decision Brief</div>
      <h1>Agenda analitica del {html.escape(payload['target_date'])}</h1>
      <p class="subcopy">
        Informe automatico generado tras la ingesta, certificacion y evaluacion matematica del dia.
        Mesa de decisiones lista para analistas, optimizada para movil.
      </p>
      {fallback_banner}
      <div class="stats">
        <div class="stat"><span>Partidos</span><strong>{summary['total_matches']}</strong></div>
        <div class="stat"><span>Analizables</span><strong>{summary['analyzable_matches']}</strong></div>
        <div class="stat"><span>Alta prioridad</span><strong>{summary['high_priority_matches']}</strong></div>
        <div class="stat"><span>Estado lote</span><strong>{html.escape(safe_text(summary['certification_status'], 'N/D'))}</strong></div>
        <div class="stat"><span>Cobertura</span><strong>{html.escape(_fmt_float(summary['coverage_pct']))}%</strong></div>
        <div class="stat"><span>Alertas C/W/I</span><strong>{alerts['critical']}/{alerts['warning']}/{alerts['info']}</strong></div>
      </div>
    </section>

    <section class="layout">
      <div class="summary-grid">
        <div class="panel">
          <h2>Agenda priorizada</h2>
          <div class="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Hora</th>
                  <th>Tipo</th>
                  <th>Partido</th>
                  <th>Soporte</th>
                  <th>Fiabilidad</th>
                  <th>Decision</th>
                  <th>xG</th>
                  <th>Mercado</th>
                  <th>Prob.</th>
                  <th>Pronostico</th>
                  <th>EV</th>
                </tr>
              </thead>
              <tbody>
                {''.join(priority_rows) or '<tr><td colspan="11">No hay partidos analizables para esta fecha.</td></tr>'}
              </tbody>
            </table>
          </div>
        </div>
        <div class="panel">
          <h2>Estado operativo</h2>
          <p><strong>Proveedor activo:</strong> {html.escape(safe_text(batch.get('provider_used'), 'N/D'))}</p>
          <p><strong>Certificacion:</strong> {html.escape(safe_text(batch.get('certification_status'), 'unknown'))}</p>
          <p><strong>Modelo activo:</strong> {html.escape(safe_text(payload.get('model_family'), 'poisson'))} / {html.escape(safe_text(payload.get('model_version'), 'ALTTEZ'))}</p>
          <p><strong>Fallback:</strong> {'Si' if payload.get('used_fallback') else 'No'}{(' | ' + html.escape(safe_text(payload.get('fallback_reason')))) if payload.get('fallback_reason') else ''}</p>
          <p><strong>Registros del lote:</strong> {batch.get('total_records', 0)} total | {batch.get('certified_records', 0)} certificados | {batch.get('rejected_records', 0)} rechazados</p>
          <p><strong>Cobertura:</strong> {html.escape(_fmt_float(batch.get('coverage_pct')))}%</p>
          <p><strong>Quarantine reason:</strong> {html.escape(safe_text(batch.get('quarantine_reason'), 'Sin motivo activo'))}</p>
          <h3 style="margin-top:18px;">Alertas recientes</h3>
          <div class="table-scroll">
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
      </div>

      <div class="panel">
        <h2>Detalle por partido</h2>
        {''.join(detail_cards) or '<p>No hay detalle disponible.</p>'}
      </div>
    </section>

    <div class="footer">
      Generado en {html.escape(payload['generated_at'])} &middot; Fuente: Supabase canonical matches &middot; Motor: {html.escape(safe_text(payload.get('model_family'), 'poisson'))} / {html.escape(safe_text(payload.get('model_version'), 'ALTTEZ'))}
    </div>
  </div>
<script>
  const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const gridColor = dark ? "rgba(255,255,255,0.08)" : "rgba(17,19,21,0.08)";
  const tickColor = dark ? "#9aa6b1" : "#6b7682";
  if (typeof Chart !== "undefined") {{
    Chart.defaults.font.family = "Inter, Segoe UI, system-ui, sans-serif";
    Chart.defaults.color = tickColor;
  }}

  const scoreCharts = {score_chart_json};
  Object.entries(scoreCharts).forEach(([id, cfg]) => {{
    const canvas = document.getElementById(id);
    const rows = (cfg && cfg.rows) || [];
    if (!canvas || !rows.length || typeof Chart === "undefined") return;
    const colors = rows.map((row) => row.score === cfg.primary ? "rgba(194,116,66,0.92)" : "rgba(63, 109, 181, 0.68)");
    new Chart(canvas, {{
      type: "bar",
      data: {{
        labels: rows.map((row) => row.score),
        datasets: [{{
          label: "Probabilidad de marcador",
          data: rows.map((row) => Math.round(row.probability * 1000) / 10),
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false,
          maxBarThickness: 46
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: (ctx) => `${{ctx.parsed.y.toFixed(1)}}%` }} }}
        }},
        scales: {{
          y: {{ beginAtZero: true, grid: {{ color: gridColor }}, ticks: {{ callback: (v) => `${{v}}%` }} }},
          x: {{ grid: {{ display: false }} }}
        }}
      }}
    }});
  }});

  const resultCharts = {result_chart_json};
  Object.entries(resultCharts).forEach(([id, cfg]) => {{
    const canvas = document.getElementById(id);
    if (!canvas || !cfg || typeof Chart === "undefined") return;
    new Chart(canvas, {{
      type: "bar",
      data: {{
        labels: [""],
        datasets: [
          {{ label: "Local " + cfg.home_team, data: [cfg.home], backgroundColor: "#3f6db5", borderRadius: 4 }},
          {{ label: "Empate", data: [cfg.draw], backgroundColor: "#94a3b8", borderRadius: 4 }},
          {{ label: "Visitante " + cfg.away_team, data: [cfg.away], backgroundColor: "#c27a42", borderRadius: 4 }}
        ]
      }},
      options: {{
        indexAxis: "y", responsive: true, maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: (ctx) => `${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(1)}}%` }} }}
        }},
        scales: {{
          x: {{ stacked: true, max: 100, grid: {{ display: false }}, ticks: {{ callback: (v) => `${{v}}%` }} }},
          y: {{ stacked: true, grid: {{ display: false }} }}
        }}
      }}
    }});
  }});
</script>
</body>
</html>
"""
