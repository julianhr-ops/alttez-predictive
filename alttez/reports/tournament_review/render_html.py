from __future__ import annotations

import html
from typing import Any

from alttez.reports.brand import brand_footer, brand_head, brand_topbar


def _pct(value: Any) -> str:
    if value is None:
        return "N/D"
    return f"{float(value) * 100:.1f}%"


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/D"
    return f"{float(value):.{digits}f}"


def _rating(summary: dict[str, Any]) -> tuple[str, str]:
    rating = summary.get("rating") or {}
    if not rating:
        return "N/D", "Sin muestra evaluable"
    return f"{float(rating['score']):.1f}/100", str(rating["label"])


def _breakdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">Sin datos para este desglose.</p>'
    body = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['name']))}</td>"
        f"<td>{int(row['evaluated_predictions'])}</td>"
        f"<td>{_pct(row.get('accuracy_1x2'))}</td>"
        f"<td>{_number(row.get('brier_score'))}</td>"
        f"<td>{_number(row.get('rps'))}</td>"
        f"<td>{_pct(row.get('exact_score_hit_rate'))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>Segmento</th><th>Partidos</th><th>1X2</th>"
        "<th>Brier</th><th>RPS</th><th>Exacto</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _matches_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No hay partidos evaluables.</p>'
    body = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['match_date']))}</td>"
        f"<td>{html.escape(str(row['stage']))}</td>"
        f"<td>{html.escape(str(row['home_team']))} vs {html.escape(str(row['away_team']))}</td>"
        f"<td>{html.escape(str(row['predicted_scoreline']))}</td>"
        f"<td>{html.escape(str(row['actual_scoreline']))}</td>"
        f"<td>{'Si' if row['hit_1x2'] else 'No'}</td>"
        f"<td>{int(row['pool_points'])}/{int(row['pool_points_max'])}</td>"
        f"<td>{html.escape(str(row['model_version']))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<div class=\"table-scroll\"><table><thead><tr><th>Fecha</th><th>Fase</th>"
        "<th>Partido</th><th>Predicho</th><th>Real</th><th>1X2</th>"
        "<th>Puntos</th><th>Modelo</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def render_tournament_review_html(review: dict[str, Any]) -> str:
    summary = review["summary"]
    coverage = review["coverage"]
    rating_score, rating_label = _rating(summary)
    warnings = "".join(
        f"<li>{html.escape(str(warning))}</li>" for warning in review.get("warnings", [])
    )
    components = (summary.get("rating") or {}).get("components", {})
    component_cards = "".join(
        f'<div class="metric"><span>{html.escape(name.replace("_", " ").title())}</span>'
        f"<strong>{float(value):.1f}</strong></div>"
        for name, value in components.items()
    )
    extra_css = """
    .score { display:flex; align-items:end; justify-content:space-between; gap:20px;
      border:1px solid rgba(194,122,66,.45); padding:20px; margin:20px 0; }
    .score strong { font-family:"Sora",sans-serif; font-size:42px; color:var(--copper); }
    .score .label { font-size:18px; font-weight:600; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
      background:var(--line); border:1px solid var(--line); margin:14px 0; }
    .metric { background:#15181b; padding:14px; min-height:82px; }
    .metric span { color:var(--neutral); display:block; font-size:11px; text-transform:uppercase; }
    .metric strong { display:block; font-size:22px; margin-top:8px; }
    .notice { border-left:3px solid var(--warning); background:rgba(245,158,11,.08);
      padding:12px 16px; margin:16px 0; }
    .notice li { margin:4px 0; }
    .table-scroll { overflow-x:auto; }
    .empty { color:var(--neutral); }
    """
    return (
        "<!DOCTYPE html><html lang=\"es\"><head>"
        + brand_head(f"ALTTEZ | Cierre {review['tournament']}")
        + f"<style>{extra_css}</style></head><body><main class=\"wrap\">"
        + brand_topbar("Cierre de torneo")
        + '<section class="hero">'
        + f"<h1>{html.escape(str(review['tournament']))}</h1>"
        + '<div class="muted">Evaluacion retrospectiva de predicciones persistidas · '
        + f"{html.escape(str(review['generated_at']))}</div></section>"
        + '<section class="score"><div><div class="label">Valoracion ALTTEZ</div>'
        + f"<p>{html.escape(rating_label)} · confianza de muestra "
        + f"{html.escape(str(review['sample_confidence']))}</p></div>"
        + f"<strong>{html.escape(rating_score)}</strong></section>"
        + '<div class="metrics">'
        + f'<div class="metric"><span>Cobertura evaluable</span><strong>{_pct(coverage["evaluable_rate"])}</strong></div>'
        + f'<div class="metric"><span>Acierto 1X2</span><strong>{_pct(summary.get("accuracy_1x2"))}</strong></div>'
        + f'<div class="metric"><span>Brier</span><strong>{_number(summary.get("brier_score"))}</strong></div>'
        + f'<div class="metric"><span>RPS</span><strong>{_number(summary.get("rps"))}</strong></div>'
        + f'<div class="metric"><span>Calibracion ECE</span><strong>{_number(summary.get("ece"))}</strong></div>'
        + f'<div class="metric"><span>Marcador exacto</span><strong>{_pct(summary.get("exact_score_hit_rate"))}</strong></div>'
        + f'<div class="metric"><span>Puntos polla</span><strong>{summary.get("pool_points", 0)}/{summary.get("pool_points_max", 0)}</strong></div>'
        + "</div>"
        + (f'<div class="notice"><strong>Limitaciones</strong><ul>{warnings}</ul></div>' if warnings else "")
        + '<h2 class="section">Componentes de la valoracion</h2>'
        + f'<div class="metrics">{component_cards}</div>'
        + '<h2 class="section">Desempeno por modelo</h2>'
        + _breakdown_table(review.get("model_breakdown", []))
        + '<h2 class="section">Desempeno por fase</h2>'
        + _breakdown_table(review.get("stage_breakdown", []))
        + '<h2 class="section">Partido por partido</h2>'
        + _matches_table(review.get("matches", []))
        + brand_footer()
        + "</main></body></html>"
    )
