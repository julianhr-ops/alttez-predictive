"""Plain-text NBA digest: today's predictions plus a live scoreboard of graded games.

Delivery reuses the hardened Telegram sender of the soccer brief; the caller decides
whether to deliver (the same ``ALTTEZ_DAILY_BRIEF_DELIVER`` env flag gates it).
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from alttez.evaluation.paired import paired_bootstrap_diff
from alttez.nba.data import ATTRIBUTION
from alttez.nba.predict import PredictionRun
from alttez.nba.store import GradedSummary, ServedModel
from alttez.reports.telegram_delivery import chunk_messages, send_message, telegram_config

MIN_GAMES_FOR_INTERVAL = 30
BOOTSTRAP_DRAWS = 2000


@dataclass(frozen=True, slots=True)
class DigestInput:
    today: date
    run: PredictionRun | None
    graded_this_run: int
    unresolved: int
    summary: GradedSummary | None
    notes: Sequence[str] = ()
    mode: str = "write"


def _percent(p: float) -> str:
    return f"{100 * p:.1f}%"


def _prediction_lines(run: PredictionRun) -> list[str]:
    lines = [f"Predictions ({len(run.predictions)})"]
    for planned in run.predictions:
        row = planned.row
        market = ""
        if row.p_home_market is not None and row.market_spread is not None:
            market = f" | market {_percent(row.p_home_market)}, line {planned.home_abbreviation} {0.0 - row.market_spread:+.1f}"
        served = "blend" if row.served_model is ServedModel.BLEND else "elo+platt"
        lines.append(
            f"- {planned.matchup} {row.tipoff_utc:%m-%d %H:%MZ}: home {_percent(row.p_home_served)} [{served}] "
            f"margin {row.expected_margin:+.1f}{market}"
        )
    if not run.predictions:
        lines.append("- none: no scheduled games in the window")
    if run.skipped:
        lines.append(f"Skipped ({len(run.skipped)})")
        lines.extend(f"- {s.matchup} (id {s.game_id}): {s.reason}" for s in run.skipped)
    return lines


def _scoreboard_lines(summary: GradedSummary) -> list[str]:
    lines = ["Live scoreboard (graded games so far)"]
    if summary.n == 0:
        return lines + ["- no graded predictions yet"]
    lines.append(
        f"- served: n={summary.n}, Brier {summary.served_brier():.4f}, accuracy {_percent(summary.served_accuracy())}"
    )
    if summary.n_market == 0:
        return lines + ["- no graded game had a stored market line"]
    lines.append(
        f"- same games with a market line (n={summary.n_market}): "
        f"served Brier {summary.served_brier(market_games_only=True):.4f}, accuracy "
        f"{_percent(summary.served_accuracy(market_games_only=True))}; "
        f"market Brier {summary.market_brier():.4f}, accuracy {_percent(summary.market_accuracy())}"
    )
    if summary.n_market < MIN_GAMES_FOR_INTERVAL:
        lines.append(f"- served minus market Brier: no interval until n >= {MIN_GAMES_FOR_INTERVAL}")
        return lines
    diff = paired_bootstrap_diff(
        summary.served_losses(),
        summary.market_losses(),
        n_boot=BOOTSTRAP_DRAWS,
        seed=0,
        block_ids=np.array([r.game_date.isoformat() for r in summary.market_rows]),
    )
    verdict = "served better" if diff.a_better else "market better" if diff.b_better else "no significant difference"
    lines.append(
        f"- served minus market Brier {diff.mean_diff:+.4f}, 95% CI [{diff.ci_low:+.4f}, {diff.ci_high:+.4f}] "
        f"(paired bootstrap by game date): {verdict}"
    )
    return lines


def build_nba_digest(digest: DigestInput) -> str:
    out = [f"NBA digest {digest.today.isoformat()} (mode: {digest.mode})"]
    if digest.run is not None:
        run = digest.run
        out.append(
            f"Data: newest completed game {run.newest_completed}; fitted on seasons {run.train_seasons[0]}-{run.train_seasons[1]} "
            f"({run.n_lined_train} lined games)"
        )
    out.extend(f"Note: {note}" for note in digest.notes)
    out.append(f"Graded this run: {digest.graded_this_run}; scheduled predictions still unresolved: {digest.unresolved}")
    if digest.run is not None:
        out += ["", *_prediction_lines(digest.run)]
    if digest.summary is not None:
        out += ["", *_scoreboard_lines(digest.summary)]
    out += ["", ATTRIBUTION]
    return "\n".join(out)


def deliver_digest(text: str) -> tuple[bool, str | None]:
    """Send the digest through the shared Telegram sender. Returns (delivered, error)."""
    token, chat_id = telegram_config()
    if not token or not chat_id:
        return False, "Telegram bot token or chat id is not configured"
    for chunk in chunk_messages(html.escape(text)):
        sent, error = send_message(token, chat_id, chunk)
        if not sent:
            return False, error
    return True, None
