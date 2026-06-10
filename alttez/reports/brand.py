"""ALTTEZ brand system for HTML reports.

Single source of brand tokens (palette, typography, logo) so every artifact that
reaches Telegram shares the same identity. Based on "Manual de Marca ALTTEZ v1.0".
"""

from __future__ import annotations

# Palette (Manual de Marca ALTTEZ).
GRAPHITE = "#111315"
IVORY = "#F5F7F8"
COPPER = "#C27A42"
BLUE = "#3F6DB5"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#EF4444"
NEUTRAL = "#667280"

TAGLINE = "UN SOLO SISTEMA. TODO TU CLUB."
SITE = "alttez.com"

# Isotype: stylised interlocking double-T mark in copper (inline so reports are
# fully self-contained, no external asset needed).
ISOTYPE_SVG = (
    '<svg width="34" height="34" viewBox="0 0 48 48" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-label="ALTTEZ">'
    '<path d="M4 6 H22 V13 H16 V40 L9 34 V13 H4 Z" fill="#C27A42"/>'
    '<path d="M26 6 H44 V13 H39 V34 L32 40 V13 H26 Z" fill="#C27A42"/>'
    "</svg>"
)


def brand_css() -> str:
    """Shared CSS variables + base typography/components for ALTTEZ reports (dark)."""

    return f"""
    :root {{
      --graphite: {GRAPHITE};
      --ivory: {IVORY};
      --copper: {COPPER};
      --blue: {BLUE};
      --success: {SUCCESS};
      --warning: {WARNING};
      --danger: {DANGER};
      --neutral: {NEUTRAL};
      --line: rgba(245, 247, 248, 0.10);
      --panel: rgba(245, 247, 248, 0.04);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(194, 122, 66, 0.16), transparent 30%),
        linear-gradient(180deg, #0d0f11 0%, #15181b 100%);
      color: var(--ivory);
      -webkit-font-smoothing: antialiased;
    }}
    .wrap {{ max-width: 1040px; margin: 0 auto; padding: 28px 22px 56px; }}
    h1, h2, h3 {{ font-family: "Sora", "Inter", sans-serif; font-weight: 700; letter-spacing: -0.01em; }}
    .topbar {{ display: flex; align-items: center; gap: 12px; padding-bottom: 14px; border-bottom: 1px solid var(--line); }}
    .topbar .brand {{ font-family: "Sora", sans-serif; font-weight: 700; font-size: 20px; letter-spacing: 0.14em; }}
    .topbar .brand b {{ color: var(--copper); }}
    .topbar .subtitle {{ margin-left: auto; color: var(--neutral); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .hero {{ margin: 22px 0 10px; }}
    .hero h1 {{ font-size: 30px; margin: 0 0 4px; }}
    .hero .muted {{ color: var(--neutral); font-size: 14px; }}
    .champion {{ margin: 18px 0; padding: 18px 20px; border-radius: 14px;
      background: linear-gradient(120deg, rgba(194,122,66,0.20), rgba(63,109,181,0.10));
      border: 1px solid rgba(194,122,66,0.40); }}
    .champion .label {{ color: var(--copper); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; }}
    .champion .team {{ font-family: "Sora", sans-serif; font-weight: 700; font-size: 26px; }}
    .champion .runner {{ color: var(--neutral); margin-top: 2px; }}
    h2.section {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--copper);
      margin: 30px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--neutral); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }}
    .panel h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .qual {{ color: var(--success); }}
    .win {{ color: var(--copper); font-weight: 600; }}
    .pen {{ color: var(--warning); font-size: 11px; }}
    .match {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid var(--line); }}
    .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line);
      color: var(--neutral); font-size: 12px; display: flex; justify-content: space-between; }}
    .footer .tag {{ color: var(--copper); letter-spacing: 0.10em; }}
    """


def brand_head(title: str) -> str:
    """Returns the <head> block with fonts + brand CSS."""

    return (
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
        f"<style>{brand_css()}</style>"
    )


def brand_topbar(subtitle: str = "") -> str:
    """Logo lockup (isotype + wordmark) for the top of every report."""

    sub = f'<span class="subtitle">{subtitle}</span>' if subtitle else ""
    return (
        '<div class="topbar">'
        f"{ISOTYPE_SVG}"
        '<span class="brand">ALT<b>TEZ</b></span>'
        f"{sub}"
        "</div>"
    )


def brand_footer() -> str:
    return (
        '<div class="footer">'
        f'<span class="tag">{TAGLINE}</span>'
        f"<span>{SITE}</span>"
        "</div>"
    )
