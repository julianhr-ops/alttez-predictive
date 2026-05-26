from __future__ import annotations

from textwrap import dedent


def build_brand_css() -> str:
    return dedent(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@400;500;600;700;800&display=swap');

        :root {
            --alttez-grafito: #111315;
            --alttez-grafito-2: #171a1d;
            --alttez-marfil: #f5f7f8;
            --alttez-marfil-soft: #dde3e7;
            --alttez-cobre: #c27a42;
            --alttez-cobre-soft: rgba(194, 122, 66, 0.18);
            --alttez-azul: #3f6db5;
            --alttez-azul-soft: rgba(63, 109, 181, 0.18);
            --alttez-line: rgba(245, 247, 248, 0.08);
            --alttez-panel: rgba(23, 26, 29, 0.92);
            --alttez-panel-soft: rgba(19, 22, 24, 0.88);
            --alttez-muted: rgba(245, 247, 248, 0.72);
        }

        .stApp {
            background:
                radial-gradient(circle at 14% 10%, rgba(194, 122, 66, 0.10), transparent 18%),
                radial-gradient(circle at 84% 18%, rgba(63, 109, 181, 0.12), transparent 17%),
                linear-gradient(180deg, #0f1113 0%, var(--alttez-grafito) 46%, #0d0f11 100%);
            color: var(--alttez-marfil);
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px) 0 0 / 100% 72px,
                linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px) 0 0 / 124px 100%;
            opacity: 0.45;
            z-index: 0;
        }

        .stApp .block-container,
        .stApp [data-testid="stSidebar"] > div {
            position: relative;
            z-index: 1;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 17, 19, 0.97), rgba(17, 19, 21, 0.97));
            border-right: 1px solid var(--alttez-line);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        h1, h2, h3, h4, h5 {
            font-family: "Sora", "Inter", system-ui, sans-serif;
            letter-spacing: -0.02em;
            color: var(--alttez-marfil);
        }

        p, span, div, label, input, textarea, button, select {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        }

        .alttez-shell {
            display: grid;
            gap: 1.15rem;
        }

        .alttez-hero {
            border: 1px solid rgba(245, 247, 248, 0.09);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            background:
                linear-gradient(180deg, rgba(23, 26, 29, 0.95), rgba(17, 19, 21, 0.96)),
                linear-gradient(135deg, rgba(194, 122, 66, 0.10), transparent 38%);
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.24);
        }

        .alttez-hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr);
            gap: 1rem;
            align-items: start;
        }

        .alttez-kicker {
            color: var(--alttez-cobre);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            margin-bottom: 0.5rem;
        }

        .alttez-headline {
            font-size: clamp(2.2rem, 3vw, 3.4rem);
            line-height: 0.97;
            font-weight: 700;
            color: var(--alttez-marfil);
            margin: 0;
            max-width: 14ch;
        }

        .alttez-subtle {
            color: var(--alttez-muted);
            font-size: 0.98rem;
            margin-top: 0.75rem;
            max-width: 68ch;
        }

        .alttez-hero-aside {
            border: 1px solid rgba(245, 247, 248, 0.08);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.02);
            padding: 0.9rem 0.95rem;
            display: grid;
            gap: 0.7rem;
        }

        .alttez-hero-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
            border-bottom: 1px solid rgba(245, 247, 248, 0.06);
            padding-bottom: 0.55rem;
        }

        .alttez-hero-row:last-child {
            border-bottom: 0;
            padding-bottom: 0;
        }

        .alttez-hero-label {
            color: rgba(245, 247, 248, 0.58);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
        }

        .alttez-hero-value {
            color: var(--alttez-marfil);
            font-weight: 600;
            text-align: right;
        }

        .alttez-panel {
            border: 1px solid rgba(245, 247, 248, 0.08);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            background: var(--alttez-panel);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.16);
        }

        .alttez-section-title {
            font-family: "Sora", "Inter", system-ui, sans-serif;
            font-size: 1.06rem;
            font-weight: 700;
            color: var(--alttez-marfil);
            margin-bottom: 0.35rem;
        }

        .alttez-section-copy {
            color: var(--alttez-muted);
            font-size: 0.93rem;
            line-height: 1.55;
        }

        .alttez-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            font-size: 0.76rem;
            line-height: 1;
            border: 1px solid rgba(245, 247, 248, 0.10);
            background: rgba(255, 255, 255, 0.03);
            color: var(--alttez-marfil-soft);
        }

        .alttez-badge.ok {
            color: #8ee7c9;
            border-color: rgba(56, 218, 167, 0.18);
            background: rgba(8, 54, 43, 0.38);
        }

        .alttez-badge.warn {
            color: #f7d08a;
            border-color: rgba(247, 197, 74, 0.18);
            background: rgba(67, 52, 14, 0.38);
        }

        .alttez-badge.error {
            color: #ffb1a7;
            border-color: rgba(255, 116, 116, 0.18);
            background: rgba(68, 19, 19, 0.40);
        }

        .alttez-match-card {
            border: 1px solid rgba(245, 247, 248, 0.08);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.02);
            padding: 0.9rem 0.98rem;
            margin-bottom: 0.7rem;
        }

        .alttez-match-meta {
            color: rgba(245, 247, 248, 0.58);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .alttez-match-title {
            color: var(--alttez-marfil);
            font-size: 1.03rem;
            font-weight: 650;
            margin-bottom: 0.28rem;
        }

        .alttez-match-subtitle {
            color: rgba(245, 247, 248, 0.74);
            font-size: 0.88rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(245, 247, 248, 0.08);
            border-radius: 14px;
            padding: 0.78rem 0.88rem;
            background: rgba(255, 255, 255, 0.025);
        }

        div[data-testid="stMetricLabel"] {
            color: rgba(245, 247, 248, 0.62);
            font-size: 0.76rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        div[data-testid="stMetricValue"] {
            color: var(--alttez-marfil);
            font-family: "Sora", "Inter", system-ui, sans-serif;
            font-weight: 700;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--alttez-cobre), #d38a4f);
            color: #111315;
            border: 0;
            padding: 0.72rem 0.98rem;
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(194, 122, 66, 0.22);
        }

        .stButton > button:hover {
            color: #111315;
            background: linear-gradient(135deg, #d18b53, #e1a15e);
        }

        .stButton > button:focus {
            outline: 2px solid rgba(63, 109, 181, 0.55);
            outline-offset: 2px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: 1px solid rgba(245, 247, 248, 0.08);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.025);
            color: rgba(245, 247, 248, 0.68);
            border: 1px solid rgba(245, 247, 248, 0.08);
            padding: 0.45rem 0.85rem;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(194, 122, 66, 0.16);
            color: var(--alttez-marfil);
            border-color: rgba(194, 122, 66, 0.26);
        }

        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 14px;
        }

        @media (max-width: 900px) {
            .alttez-hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """
    )
