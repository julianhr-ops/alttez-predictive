"""Serving policy for international soccer 1X2: an Elo ordinal 1X2 imposed on the Poisson score matrix.

Walk-forward validation (docs/validation_soccer.md) shows the Poisson 1X2 is clearly worse than an
Elo-based 1X2 on years held out from the choice of rule. The policy keeps the Poisson score matrix
for scorelines and rescales it by class ratios so its 1X2 marginals equal the Elo ordinal 1X2.

The ratings and the draw model are built from stored matches dated before the target date, with
the same ``rate_matches`` walk and the same 4-year training window the backtest uses.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alttez.features.elo import rate_matches
from alttez.features.elo_1x2 import DrawModel, fit_draw_model
from alttez.inference.calibration import rescale_matrix_to_triplet

SERVING_POLICY_ENV = "ALTTEZ_SERVING_POLICY"
POLICY_POISSON = "poisson"
POLICY_ELO_ORDINAL = "elo_ordinal_rescale"
SERVING_WINDOW_YEARS = 4

_OFF = {"off", "0", "false", "no", "none", POLICY_POISSON}
_ON = {"on", "1", "true", "yes", POLICY_ELO_ORDINAL}


class EloUnavailableError(LookupError):
    """A team has no Elo rating, so the policy cannot produce a 1X2 for the match."""


def serving_policy_enabled(override: bool | None = None, *, env: Mapping[str, str] | None = None) -> bool:
    """Whether the Elo serving policy is on: the explicit ``override``, else the env var, else ON."""
    if override is not None:
        return override
    raw = (env if env is not None else os.environ).get(SERVING_POLICY_ENV)
    if raw is None or not raw.strip():
        return True
    value = raw.strip().lower()
    if value in _OFF:
        return False
    if value in _ON:
        return True
    raise ValueError(f"{SERVING_POLICY_ENV}={raw!r} is not one of {sorted(_ON | _OFF)}")


@dataclass(frozen=True, slots=True)
class ServingPolicy:
    name: str
    draw_model: DrawModel
    ratings: Mapping[str, float]

    def target_1x2(self, home_team: str, away_team: str, is_neutral: bool) -> tuple[float, float, float]:
        missing = [team for team in (home_team, away_team) if team not in self.ratings]
        if missing:
            raise EloUnavailableError(f"no Elo rating for {', '.join(missing)}")
        diff = self.ratings[home_team] - self.ratings[away_team]
        home, draw, away = self.draw_model.probabilities([diff], [bool(is_neutral)])[0]
        return float(home), float(draw), float(away)

    def serve(self, matrix: np.ndarray, home_team: str, away_team: str, is_neutral: bool) -> np.ndarray:
        """The score matrix whose 1X2 marginals are the policy's 1X2. Raises ``EloUnavailableError``."""
        return rescale_matrix_to_triplet(matrix, self.target_1x2(home_team, away_team, is_neutral))


def build_serving_policy(
    historical_df: pd.DataFrame, target_date: str | pd.Timestamp, *, window_years: int = SERVING_WINDOW_YEARS
) -> ServingPolicy:
    """Rate every stored match before ``target_date`` and fit the ordinal draw model on the trailing window.

    The window starts on 1 January ``window_years`` calendar years before the target year, as the
    backtest's does. Needs ``date``, ``home_team``, ``away_team``, ``home_score``, ``away_score``,
    ``tournament`` and ``neutral``.
    """
    cutoff = pd.Timestamp(target_date)
    history = historical_df.loc[pd.to_datetime(historical_df["date"]) < cutoff].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date", kind="stable").reset_index(drop=True)
    run = rate_matches(history)
    in_window = (history["date"] >= pd.Timestamp(year=cutoff.year - window_years, month=1, day=1)).to_numpy()
    if not in_window.any():
        raise ValueError(f"no matches in the {window_years}-year window before {cutoff.date()} to fit the draw model")
    scored = history.loc[in_window]
    outcomes = np.where(scored["home_score"] > scored["away_score"], "H", np.where(scored["home_score"] < scored["away_score"], "A", "D"))
    draw_model = fit_draw_model(
        "ordinal",
        (run.home_before - run.away_before)[in_window],
        scored["neutral"].fillna(False).to_numpy(bool),
        outcomes.tolist(),
    )
    return ServingPolicy(POLICY_ELO_ORDINAL, draw_model, run.ratings)
