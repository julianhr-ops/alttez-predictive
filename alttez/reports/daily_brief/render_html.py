from __future__ import annotations

import html
import json
from typing import Any

from alttez.dashboard.analysis_helpers import build_recommendation_summary, safe_text
from alttez.reports.brand import ISOTYPE_SVG
from .formatting import _fmt_float, _fmt_pct, _fmt_points_profile


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

    chart_payload: dict[str, list[dict[str, Any]]] = {}
    detail_cards = []
    for match_index, match in enumerate(matches):
        chart_id = f"score-chart-{match_index}"
        chart_payload[chart_id] = match.get("score_probability_chart", [])
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
                <div class="metric"><span>Marcador 10 pts</span><strong>{prediction}</strong></div>
                <div class="metric"><span>Prob. score</span><strong>{score_prob}</strong></div>
                <div class="metric"><span>Score exacto puro</span><strong>{raw_score}</strong></div>
                <div class="metric"><span>Score estrategia EV</span><strong>{ev_score}</strong></div>
                <div class="metric"><span>EV pick exacto</span><strong>{ev}</strong></div>
                <div class="metric"><span>Confianza media</span><strong>{avg_confidence}</strong></div>
                <div class="metric"><span>Historial util</span><strong>{history}</strong></div>
              </div>
              <div class="score-chart-wrap">
                <canvas id="{chart_id}" height="160" aria-label="Distribucion de probabilidad de marcadores"></canvas>
              </div>
              <p><strong>Forma reciente:</strong> {home} {home_form} pts/j, GD {home_gd} | {away} {away_form} pts/j, GD {away_gd}</p>
              <p><strong>Pronosticos alternativos:</strong> {candidates}</p>
              <p><strong>Estrategias EV alternativas:</strong> {alternatives}</p>
              <p><strong>Comite predictivo:</strong> {committee}</p>
              <p><strong>Riesgo operativo:</strong> {risk}</p>
            </article>
            """.format(
                chart_id=chart_id,
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
                score_prob=html.escape(_fmt_pct(match.get("primary_score_probability"))),
                raw_score=html.escape(
                    f"{safe_text(match.get('most_likely_score'), 'N/D')} ({_fmt_pct(match.get('most_likely_score_probability'))})"
                ),
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

    chart_json = json.dumps(chart_payload, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ALTTEZ Daily Decision Brief | {html.escape(payload['target_date'])}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --graphite: #111315;
      --ivory: #f5f7f8;
      --copper: #c27a42;
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
    .score-chart-wrap {{
      min-height: 190px;
      margin: 16px 0;
      padding: 12px;
      border: 1px solid rgba(17, 19, 21, 0.08);
      border-radius: 14px;
      background: rgba(255,255,255,0.64);
    }}
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
      <div class="brand-lockup" style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
        {ISOTYPE_SVG}
        <span style="font-family:'Sora','Inter',sans-serif;font-weight:700;letter-spacing:0.14em;font-size:18px;">ALT<span style="color:var(--copper);">TEZ</span></span>
      </div>
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
          <p><strong>Modelo activo:</strong> {html.escape(safe_text(payload.get('model_family'), 'poisson'))} / {html.escape(safe_text(payload.get('model_version'), 'ALTTEZ'))}</p>
          <p><strong>Fallback:</strong> {'Si' if payload.get('used_fallback') else 'No'}{(' | ' + html.escape(safe_text(payload.get('fallback_reason')))) if payload.get('fallback_reason') else ''}</p>
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
      Generado en {html.escape(payload['generated_at'])} | Fuente de agenda: Supabase canonical matches | Motor: {html.escape(safe_text(payload.get('model_family'), 'poisson'))} / {html.escape(safe_text(payload.get('model_version'), 'ALTTEZ'))}
    </div>
  </div>
<script>
  const scoreCharts = {chart_json};
  Object.entries(scoreCharts).forEach(([id, rows]) => {{
    const canvas = document.getElementById(id);
    if (!canvas || !rows || !rows.length || typeof Chart === "undefined") return;
    new Chart(canvas, {{
      type: "bar",
      data: {{
        labels: rows.map((row) => row.score),
        datasets: [{{
          label: "Probabilidad de marcador",
          data: rows.map((row) => Math.round(row.probability * 1000) / 10),
          backgroundColor: "rgba(63, 109, 181, 0.72)",
          borderColor: "rgba(18, 53, 93, 0.9)",
          borderWidth: 1
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: (ctx) => `${{ctx.parsed.y.toFixed(1)}}%` }} }}
        }},
        scales: {{
          y: {{ beginAtZero: true, ticks: {{ callback: (value) => `${{value}}%` }} }},
          x: {{ grid: {{ display: false }} }}
        }}
      }}
    }});
  }});
</script>
</body>
</html>
"""


