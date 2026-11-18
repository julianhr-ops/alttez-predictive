"""Render the NBA ledger into static JSON data files."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alttez.evaluation.paired import paired_bootstrap_diff
from alttez.nba import data, store
from alttez.site import ledger

ROOT = Path(__file__).resolve().parents[2]


def number(value):
    return round(float(value), 6)


def newest_completed():
    cache = ROOT / "artifacts/nba/schedule.parquet"
    if not cache.exists():
        return None
    games = data.normalise_schedule(pd.read_parquet(cache)).games
    return games["game_date"].max().date().isoformat() if not games.empty else None


def build_data(rows, now, newest=None):
    generated = ledger.timestamp(now)
    summary = store.summarise_graded(rows)
    upcoming = []
    for row in store.served_rows(r for r in rows if r.status is store.PredictionStatus.SCHEDULED and r.tipoff_utc > now):
        upcoming.append({
            "game_id": row.game_id, "tipoff_utc": ledger.timestamp(row.tipoff_utc),
            "home_abbr": ledger.TEAMS[str(row.home_team_id)], "away_abbr": ledger.TEAMS[str(row.away_team_id)],
            "served_model": row.served_model.value, "p_home_served": number(row.p_home_served),
            "p_home_elo": number(row.p_home_elo),
            "p_home_market": number(row.p_home_market) if row.p_home_market is not None else None,
            "expected_margin": number(row.expected_margin),
            "market_spread": number(row.market_spread) if row.market_spread is not None else None,
            "market_source": row.market_source, "created_at": ledger.timestamp(row.created_at),
        })
    paired = None
    if summary.n_market >= 30:
        diff = paired_bootstrap_diff(summary.served_losses(), summary.market_losses(),
                                     block_ids=[row.game_date for row in summary.market_rows])
        paired = {"mean": number(diff.mean_diff), "lo": number(diff.ci_low), "hi": number(diff.ci_high)}
    graded_times = [r.graded_at for r in rows if r.status is store.PredictionStatus.GRADED and r.graded_at]
    scoreboard = {
        "generated_at": generated, "n_graded": summary.n, "n_market": summary.n_market,
        "served": {"brier": number(summary.served_brier()) if summary.n else None,
                   "accuracy": number(summary.served_accuracy()) if summary.n else None},
        "market": {"brier": number(summary.market_brier()) if summary.n_market else None,
                   "accuracy": number(summary.market_accuracy()) if summary.n_market else None},
        "paired_brier_diff": paired,
        "predicted_before_tipoff": {"n": sum(r.created_at <= r.tipoff_utc for r in rows), "of": len(rows)},
        "first_prediction_at": ledger.timestamp(min(r.created_at for r in rows)) if rows else None,
        "last_graded_at": ledger.timestamp(max(graded_times)) if graded_times else None,
    }
    recent = []
    for row in reversed(summary.rows[-60:]):
        recent.append({"game_id": row.game_id, "game_date": row.game_date.isoformat(),
                       "home_abbr": ledger.TEAMS[str(row.home_team_id)], "away_abbr": ledger.TEAMS[str(row.away_team_id)],
                       "served_model": row.served_model.value, "p_home_served": number(row.p_home_served),
                       "p_home_market": number(row.p_home_market) if row.p_home_market is not None else None,
                       "home_score": row.home_score, "away_score": row.away_score, "home_win": row.home_win,
                       "brier": number(row.brier)})
    return {"nba_upcoming.json": {"generated_at": generated, "newest_completed": newest, "games": upcoming},
            "nba_scoreboard.json": scoreboard, "nba_recent.json": {"generated_at": generated, "games": recent}}


def write_data(engine, out):
    target = Path(out) / "data"
    target.mkdir(parents=True, exist_ok=True)
    for name, contents in build_data(ledger.load_rows(engine), datetime.now(UTC), newest_completed()).items():
        (target / name).write_text(json.dumps(contents, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "site/results/nba_validation.json", target / "nba_validation.json")
