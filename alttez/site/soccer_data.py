"""Data files behind the public soccer page: the Elo ratings and the 1X2 grids.

The grids come from ``ServingPolicy.target_1x2``, the call the pipeline serves, so the page and
the pipeline cannot diverge. Ratings and the draw model are built by ``build_serving_policy``
from the public results CSV, with the same 4 year window the pipeline uses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from alttez.inference.serving_policy import SERVING_WINDOW_YEARS, ServingPolicy, build_serving_policy

N_TEAMS = 100
ACTIVE_MONTHS = 24
SCALE = 10_000
RATING_DECIMALS = 6
RATINGS_FILENAME = "soccer_ratings.json"
VALIDATION_FILENAME = "soccer_validation.json"
VALIDATION_SNAPSHOT = Path(__file__).resolve().parents[2] / "site" / "results" / VALIDATION_FILENAME
_VALIDATION_KEYS = ("source_command", "report", "stages")
_STAGE_KEYS = ("name", "years", "n_matches", "rows", "notes")
_ROW_KEYS = ("model", "brier", "rps", "log_loss", "accuracy")


def round_triplet(probabilities: Sequence[float]) -> tuple[int, int, int]:
    """Integers summing to ``SCALE``: each probability rounded, the largest component absorbs the rounding."""
    scaled = [round(float(p) * SCALE) for p in probabilities]
    largest = max(range(3), key=lambda k: probabilities[k])
    scaled[largest] += SCALE - sum(scaled)
    return scaled[0], scaled[1], scaled[2]


def active_teams(history: pd.DataFrame, as_of: date, *, months: int = ACTIVE_MONTHS) -> set[str]:
    """Teams with at least one match in the ``months`` before ``as_of``."""
    cutoff = pd.Timestamp(as_of)
    dates = pd.to_datetime(history["date"])
    recent = history.loc[(dates >= cutoff - pd.DateOffset(months=months)) & (dates < cutoff)]
    return set(recent["home_team"]) | set(recent["away_team"])


def top_teams(policy: ServingPolicy, history: pd.DataFrame, as_of: date, n: int = N_TEAMS) -> list[dict[str, Any]]:
    """The ``n`` best rated active teams, best first, ties broken by name."""
    active = active_teams(history, as_of)
    ranked = sorted(((name, rating) for name, rating in policy.ratings.items() if name in active), key=lambda item: (-item[1], item[0]))
    return [{"name": name, "rating": round(float(rating), RATING_DECIMALS), "rank": rank} for rank, (name, rating) in enumerate(ranked[:n], start=1)]


def probability_grid(policy: ServingPolicy, names: Sequence[str], *, neutral: bool) -> list[int]:
    """Flat ``n * n * 3`` array, index ``((i * n) + j) * 3 + k``; k is 0 win of i, 1 draw, 2 win of j.

    Team ``i`` is the home side unless ``neutral``. The diagonal is zeros.
    """
    n = len(names)
    flat = [0] * (n * n * 3)
    for i, home in enumerate(names):
        for j, away in enumerate(names):
            if i != j:
                flat[(i * n + j) * 3 : (i * n + j) * 3 + 3] = round_triplet(policy.target_1x2(home, away, neutral))
    return flat


def build_ratings_payload(
    history: pd.DataFrame,
    as_of: date,
    *,
    source: str,
    n: int = N_TEAMS,
    window_years: int = SERVING_WINDOW_YEARS,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    policy = build_serving_policy(history, pd.Timestamp(as_of), window_years=window_years)
    teams = top_teams(policy, history, as_of, n)
    names = [team["name"] for team in teams]
    stamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "generated_at": stamp,
        "as_of": as_of.isoformat(),
        "window_years": window_years,
        "source": source,
        "n": len(teams),
        "teams": teams,
        "neutral": probability_grid(policy, names, neutral=True),
        "home": probability_grid(policy, names, neutral=False),
    }


def load_validation_snapshot(path: Path = VALIDATION_SNAPSHOT) -> dict[str, Any]:
    """The committed validation snapshot, checked for the keys the page reads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in _VALIDATION_KEYS if key not in payload]
    for stage in payload.get("stages", []):
        missing += [f"stage.{key}" for key in _STAGE_KEYS if key not in stage]
        for row in stage.get("rows", []):
            missing += [f"row.{key}" for key in _ROW_KEYS if key not in row]
    if missing:
        raise ValueError(f"{path} lacks {sorted(set(missing))}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
