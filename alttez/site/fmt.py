"""Number and time formatting for the static site.

Every figure the pages show goes through this module so the same quantity has
the same digits on every page.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

MISSING = "n/a"


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def probability(p: float | None, digits: int = 1) -> str:
    """0.6234 -> '62.3%'."""
    if p is None:
        return MISSING
    return f"{p * 100:.{digits}f}%"


def brier(x: float | None) -> str:
    return MISSING if x is None else f"{x:.4f}"


def log_loss(x: float | None) -> str:
    return MISSING if x is None else f"{x:.4f}"


def accuracy(x: float | None) -> str:
    """Accuracy as a three-digit share, the way the validation reports print it."""
    return MISSING if x is None else f"{x:.3f}"


def ece(x: float | None) -> str:
    return MISSING if x is None else f"{x:.3f}"


def signed(x: float, digits: int = 4) -> str:
    return f"{x:+.{digits}f}"


def interval(mean: float, lo: float, hi: float, digits: int = 4) -> str:
    return f"{signed(mean, digits)} [{signed(lo, digits)}, {signed(hi, digits)}]"


def count(n: int | None) -> str:
    return MISSING if n is None else f"{n:,}"


def rating(x: float) -> str:
    return f"{x:.0f}"


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def utc_minute(value: str | None) -> str:
    if not value:
        return MISSING
    return parse_utc(value).strftime("%Y-%m-%d %H:%M UTC")


def utc_day(value: str | None) -> str:
    if not value:
        return MISSING
    return parse_utc(value).strftime("%Y-%m-%d")


def home_line(abbr: str, spread: float | None, source: str | None = None) -> str:
    """Sportsbook style home line. A positive ``market_spread`` means the home team is favoured."""
    if spread is None:
        return MISSING
    value = -spread + 0.0
    line = f"{abbr} {value:+.1f}"
    return f"{line} ({source})" if source else line


def pct_width(p: float) -> str:
    """CSS width for a probability bar, clamped to [0, 100]."""
    return f"{min(max(p, 0.0), 1.0) * 100:.1f}%"
