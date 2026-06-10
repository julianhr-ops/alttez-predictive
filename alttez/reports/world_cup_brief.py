"""Render a full-tournament World Cup 2026 prediction into a Telegram-ready brief.

Takes the deterministic bracket from
``alttez.inference.world_cup_2026.predict_world_cup_2026_bracket`` plus optional
Monte-Carlo odds from ``simulate_world_cup_2026`` and produces an HTML message
covering every phase: group tables, qualifiers, each knockout tie, the final and the
champion, with champion-probability context.
"""

from __future__ import annotations

import html
from typing import Any

from alttez.reports.brand import brand_footer, brand_head, brand_topbar


KNOCKOUT_STAGE_LABELS = {
    "round_of_32": "Dieciseisavos (R32)",
    "round_of_16": "Octavos de final",
    "quarter_final": "Cuartos de final",
    "semi_final": "Semifinales",
    "final": "Final",
}
KNOCKOUT_ORDER = ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/D"


def _match_line(match: dict[str, Any]) -> str:
    home = _esc(match.get("home_team"))
    away = _esc(match.get("away_team"))
    home_score = match.get("home_score")
    away_score = match.get("away_score")
    winner = _esc(match.get("winner"))
    suffix = " (pen.)" if match.get("decided_by") == "penalties" else ""
    return f"{home} {home_score}-{away_score} {away}{suffix} → <b>{winner}</b>"


def _group_block(group_name: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"<b>Grupo {_esc(group_name)}</b>"]
    for row in rows:
        position = row.get("position")
        qualifies = position in (1, 2) or (
            position == 3 and row.get("third_rank") is not None and int(row["third_rank"]) <= 8
        )
        mark = "✅" if qualifies else "▫️"
        gd = row.get("goal_difference", 0)
        gd_str = f"+{gd}" if isinstance(gd, int) and gd > 0 else str(gd)
        lines.append(
            f"{mark} {position}. {_esc(row.get('team'))} — {row.get('points', 0)} pts ({gd_str})"
        )
    return "\n".join(lines)


def build_world_cup_brief_text(
    bracket: dict[str, Any],
    *,
    odds: list[dict[str, Any]] | None = None,
    top_n: int = 8,
) -> str:
    """Build the full HTML brief text (single string; chunk before sending)."""

    champion = _esc(bracket.get("champion"))
    runner_up = _esc(bracket.get("runner_up"))
    model_family = _esc(bracket.get("model_family"))
    model_version = _esc(bracket.get("model_version"))
    fallback = " | fallback" if bracket.get("used_fallback") else ""

    champion_pct = ""
    if odds:
        for entry in odds:
            if entry.get("team") == bracket.get("champion"):
                champion_pct = f" ({_fmt_pct(entry.get('champion_probability'))} prob.)"
                break

    sections: list[str] = []

    # Header + verdict.
    sections.append(
        "\n".join(
            [
                "🏆 <b>ALTTEZ | Predicción Mundial 2026</b>",
                f"Modelo {model_family} / {model_version}{fallback}",
                "",
                f"🥇 <b>Campeón pronosticado: {champion}</b>{champion_pct}",
                f"🥈 Subcampeón: {runner_up}",
            ]
        )
    )

    # Group stage.
    group_blocks = ["📋 <b>FASE DE GRUPOS</b>"]
    for group_name in sorted(bracket.get("groups", {}).keys()):
        group_blocks.append(_group_block(group_name, bracket["groups"][group_name]))
    sections.append("\n\n".join(group_blocks))

    # Knockout rounds.
    knockout = bracket.get("knockout", {})
    for stage in KNOCKOUT_ORDER:
        matches = knockout.get(stage) or []
        if not matches:
            continue
        block = [f"🔻 <b>{_esc(KNOCKOUT_STAGE_LABELS.get(stage, stage))}</b>"]
        block.extend(_match_line(match) for match in matches)
        sections.append("\n".join(block))

    # Odds context.
    if odds:
        ranked = sorted(odds, key=lambda item: float(item.get("champion_probability") or 0.0), reverse=True)
        odds_block = ["📊 <b>Probabilidades de título (Monte-Carlo)</b>"]
        for index, entry in enumerate(ranked[:top_n], start=1):
            odds_block.append(
                f"{index}. {_esc(entry.get('team'))} — {_fmt_pct(entry.get('champion_probability'))}"
                f" | final {_fmt_pct(entry.get('final_probability'))}"
            )
        sections.append("\n".join(odds_block))

    return "\n\n".join(sections)


def build_world_cup_summary_text(bracket: dict[str, Any], *, odds: list[dict[str, Any]] | None = None) -> str:
    """Short Telegram caption: champion + top contenders; full detail in the HTML."""

    champion = _esc(bracket.get("champion"))
    runner_up = _esc(bracket.get("runner_up"))
    lines = [
        "🏆 <b>ALTTEZ | Predicción Mundial 2026</b>",
        f"🥇 <b>Campeón: {champion}</b>",
        f"🥈 Subcampeón: {runner_up}",
    ]
    if odds:
        ranked = sorted(odds, key=lambda item: float(item.get("champion_probability") or 0.0), reverse=True)
        lines.append("")
        lines.append("<b>Favoritos (prob. título)</b>")
        for index, entry in enumerate(ranked[:3], start=1):
            lines.append(f"{index}. {_esc(entry.get('team'))} — {_fmt_pct(entry.get('champion_probability'))}")
    lines.append("")
    lines.append("📎 Cuadro completo (grupos → final) en el HTML adjunto.")
    return "\n".join(lines)


def _group_table_html(group_name: str, rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        position = row.get("position")
        qualifies = position in (1, 2) or (
            position == 3 and row.get("third_rank") is not None and int(row["third_rank"]) <= 8
        )
        mark = '<span class="qual">✔</span>' if qualifies else "·"
        gd = row.get("goal_difference", 0)
        gd_str = f"+{gd}" if isinstance(gd, int) and gd > 0 else str(gd)
        body.append(
            f"<tr><td>{mark} {position}</td><td>{_esc(row.get('team'))}</td>"
            f"<td>{row.get('points', 0)}</td><td>{gd_str}</td></tr>"
        )
    return (
        f'<div class="panel"><h3>Grupo {_esc(group_name)}</h3>'
        "<table><thead><tr><th>#</th><th>Equipo</th><th>Pts</th><th>DG</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _knockout_html(stage: str, matches: list[dict[str, Any]]) -> str:
    rows = []
    for match in matches:
        suffix = ' <span class="pen">(pen.)</span>' if match.get("decided_by") == "penalties" else ""
        rows.append(
            f'<div class="match"><span>{_esc(match.get("home_team"))} '
            f'{match.get("home_score")}-{match.get("away_score")} {_esc(match.get("away_team"))}{suffix}</span>'
            f'<span class="win">{_esc(match.get("winner"))}</span></div>'
        )
    return f'<div class="panel"><h3>{_esc(KNOCKOUT_STAGE_LABELS.get(stage, stage))}</h3>{"".join(rows)}</div>'


def render_world_cup_brief_html(bracket: dict[str, Any], *, odds: list[dict[str, Any]] | None = None, top_n: int = 12) -> str:
    """Branded standalone HTML report for the full-tournament prediction."""

    champion = _esc(bracket.get("champion"))
    runner_up = _esc(bracket.get("runner_up"))
    model = f"{_esc(bracket.get('model_family'))} / {_esc(bracket.get('model_version'))}"
    fallback = " · fallback" if bracket.get("used_fallback") else ""

    champion_pct = ""
    if odds:
        for entry in odds:
            if entry.get("team") == bracket.get("champion"):
                champion_pct = f" · {_fmt_pct(entry.get('champion_probability'))} prob. título"
                break

    groups_html = "".join(
        _group_table_html(name, bracket["groups"][name]) for name in sorted(bracket.get("groups", {}))
    )
    knockout = bracket.get("knockout", {})
    knockout_html = "".join(
        _knockout_html(stage, knockout[stage]) for stage in KNOCKOUT_ORDER if knockout.get(stage)
    )

    odds_html = ""
    if odds:
        ranked = sorted(odds, key=lambda item: float(item.get("champion_probability") or 0.0), reverse=True)
        body = "".join(
            f"<tr><td>{index}</td><td>{_esc(e.get('team'))}</td>"
            f"<td>{_fmt_pct(e.get('champion_probability'))}</td><td>{_fmt_pct(e.get('final_probability'))}</td></tr>"
            for index, e in enumerate(ranked[:top_n], start=1)
        )
        odds_html = (
            '<h2 class="section">Probabilidades (Monte-Carlo)</h2>'
            "<table><thead><tr><th>#</th><th>Equipo</th><th>Campeón</th><th>Final</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    return (
        "<!DOCTYPE html><html lang=\"es\"><head>"
        + brand_head("ALTTEZ | Predicción Mundial 2026")
        + "</head><body><div class=\"wrap\">"
        + brand_topbar("Predicción Mundial 2026")
        + f'<div class="hero"><h1>Cuadro completo del torneo</h1>'
        + f'<div class="muted">Modelo {model}{fallback}</div></div>'
        + f'<div class="champion"><div class="label">Campeón pronosticado</div>'
        + f'<div class="team">🏆 {champion}{champion_pct}</div>'
        + f'<div class="runner">Subcampeón: {runner_up}</div></div>'
        + '<h2 class="section">Fase de grupos</h2>'
        + f'<div class="grid">{groups_html}</div>'
        + '<h2 class="section">Eliminatorias</h2>'
        + f'<div class="grid">{knockout_html}</div>'
        + odds_html
        + brand_footer()
        + "</div></body></html>"
    )
