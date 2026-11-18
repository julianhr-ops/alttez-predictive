"""Shared synthetic league for the NBA tests (no network, deterministic)."""

import numpy as np
import pandas as pd

from alttez.nba.model import EloParams

SMALL_GRID = (EloParams(k=12.0), EloParams(k=24.0, b2b_penalty=30.0))
SMALL_VANILLA = (EloParams(k=12.0, use_mov=False), EloParams(k=24.0, use_mov=False))
N_TEAMS = 8


def synthetic_games(*, first_season=2010, seasons=8, lined_from=2012, seed=3) -> pd.DataFrame:
    """A small league with persistent team strength, a home edge and closing lines for later seasons."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 4, N_TEAMS)
    rows = []
    game_id = 1
    for season in range(first_season, first_season + seasons):
        day = pd.Timestamp(year=season - 1, month=10, day=20)
        strength = 0.8 * strength + rng.normal(0, 1.5, N_TEAMS)
        for _ in range(28):
            day += pd.Timedelta(days=int(rng.integers(1, 4)))
            order = rng.permutation(N_TEAMS)
            for i in range(0, N_TEAMS, 2):
                home, away = int(order[i]) + 1, int(order[i + 1]) + 1
                true_margin = strength[home - 1] - strength[away - 1] + 3.0
                margin = int(np.rint(true_margin + rng.normal(0, 12)))
                margin = margin or 1
                line = float(round((true_margin + rng.normal(0, 3)) * 2) / 2) if season >= lined_from else np.nan
                rows.append(
                    {
                        "game_id": game_id, "start_time": day + pd.Timedelta(hours=20), "game_date": day,
                        "season": season, "season_type": 2, "home_id": home, "away_id": away,
                        "home_score": 100 + (margin + 1) // 2, "away_score": 100 - margin // 2,
                        "neutral": False, "home_margin_line": line,
                    }
                )
                game_id += 1
    frame = pd.DataFrame(rows)
    frame.loc[frame.groupby("season").tail(3).index, "season_type"] = 3
    return frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)
