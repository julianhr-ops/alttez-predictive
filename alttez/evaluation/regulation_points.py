from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def regulation_points(
    pred_home: int,
    pred_away: int,
    actual_home: int,
    actual_away: int,
    *,
    is_knockout: bool = False,
) -> int:
    """Score a prediction using the pool regulation."""

    multiplier = 2 if is_knockout else 1
    points = 0

    if np.sign(pred_home - pred_away) == np.sign(actual_home - actual_away):
        points += 5 * multiplier
    if pred_home == actual_home:
        points += 2 * multiplier
    if pred_away == actual_away:
        points += 2 * multiplier
    if (pred_home - pred_away) == (actual_home - actual_away):
        points += 1 * multiplier

    return points


def max_regulation_points(*, is_knockout: bool = False) -> int:
    return 20 if is_knockout else 10


def scoreline_text(home_goals: int, away_goals: int) -> str:
    return f"{int(home_goals)} - {int(away_goals)}"


def expected_regulation_points(
    prob_matrix: np.ndarray,
    pred_home: int,
    pred_away: int,
    *,
    is_knockout: bool = False,
) -> float:
    max_goals = prob_matrix.shape[0] - 1
    expected_points = 0.0
    for actual_home in range(max_goals + 1):
        for actual_away in range(max_goals + 1):
            expected_points += float(prob_matrix[actual_home, actual_away]) * regulation_points(
                pred_home,
                pred_away,
                actual_home,
                actual_away,
                is_knockout=is_knockout,
            )
    return float(expected_points)


def rank_regulation_predictions(prob_matrix: np.ndarray, *, is_knockout: bool = False) -> pd.DataFrame:
    max_goals = prob_matrix.shape[0] - 1
    rows = []
    for pred_home in range(max_goals + 1):
        for pred_away in range(max_goals + 1):
            rows.append(
                {
                    "Pronostico": scoreline_text(pred_home, pred_away),
                    "EV": expected_regulation_points(
                        prob_matrix,
                        pred_home,
                        pred_away,
                        is_knockout=is_knockout,
                    ),
                    "ProbabilidadExacta": float(prob_matrix[pred_home, pred_away]),
                }
            )
    return pd.DataFrame(rows).sort_values(by="EV", ascending=False).reset_index(drop=True)


def points_optimal_prediction(prob_matrix: np.ndarray, *, is_knockout: bool = False) -> dict[str, Any]:
    ranking = rank_regulation_predictions(prob_matrix, is_knockout=is_knockout)
    if ranking.empty:
        return {"home_goals": None, "away_goals": None, "scoreline": "N/D", "ev": None, "exact_probability": None}

    row = ranking.iloc[0]
    home_goals, away_goals = [int(value.strip()) for value in str(row["Pronostico"]).split("-", 1)]
    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "scoreline": scoreline_text(home_goals, away_goals),
        "ev": float(row["EV"]),
        "exact_probability": float(row["ProbabilidadExacta"]),
    }


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    home_goals, away_goals = str(scoreline).split("-", 1)
    return int(home_goals.strip()), int(away_goals.strip())


def scoreline_scenarios_for_prediction(
    prob_matrix: np.ndarray,
    pred_home: int,
    pred_away: int,
    *,
    is_knockout: bool = False,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    rows = []
    max_goals = prob_matrix.shape[0] - 1
    for actual_home in range(max_goals + 1):
        for actual_away in range(max_goals + 1):
            rows.append(
                {
                    "actual_score": scoreline_text(actual_home, actual_away),
                    "probability": float(prob_matrix[actual_home, actual_away]),
                    "points_if_actual": regulation_points(
                        pred_home,
                        pred_away,
                        actual_home,
                        actual_away,
                        is_knockout=is_knockout,
                    ),
                }
            )

    return sorted(rows, key=lambda row: row["probability"], reverse=True)[:top_n]


def scoreline_candidate_profile(
    prob_matrix: np.ndarray,
    pred_home: int,
    pred_away: int,
    *,
    is_knockout: bool = False,
) -> dict[str, Any]:
    points_probability: dict[int, float] = {}
    max_goals = prob_matrix.shape[0] - 1
    for actual_home in range(max_goals + 1):
        for actual_away in range(max_goals + 1):
            points = regulation_points(
                pred_home,
                pred_away,
                actual_home,
                actual_away,
                is_knockout=is_knockout,
            )
            points_probability[points] = points_probability.get(points, 0.0) + float(
                prob_matrix[actual_home, actual_away]
            )

    max_points = max_regulation_points(is_knockout=is_knockout)
    return {
        "score": scoreline_text(pred_home, pred_away),
        "exact_probability": float(prob_matrix[pred_home, pred_away]),
        "expected_points": expected_regulation_points(
            prob_matrix,
            pred_home,
            pred_away,
            is_knockout=is_knockout,
        ),
        "full_points_probability": float(points_probability.get(max_points, 0.0)),
        "zero_points_probability": float(points_probability.get(0, 0.0)),
        "points_probability": dict(sorted(points_probability.items(), reverse=True)),
    }


def scoreline_candidate_profiles(
    prob_matrix: np.ndarray,
    *,
    is_knockout: bool = False,
    top_probability: int = 5,
    top_ev: int = 5,
) -> list[dict[str, Any]]:
    probability_rows = []
    max_goals = prob_matrix.shape[0] - 1
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability_rows.append((float(prob_matrix[home_goals, away_goals]), home_goals, away_goals))

    selected: dict[str, dict[str, Any]] = {}
    for _, home_goals, away_goals in sorted(probability_rows, reverse=True)[:top_probability]:
        profile = scoreline_candidate_profile(
            prob_matrix,
            home_goals,
            away_goals,
            is_knockout=is_knockout,
        )
        profile["source"] = "probabilidad_exacta"
        selected[profile["score"]] = profile

    ev_ranking = rank_regulation_predictions(prob_matrix, is_knockout=is_knockout)
    for row in ev_ranking.head(top_ev).to_dict(orient="records"):
        home_goals, away_goals = parse_scoreline(str(row["Pronostico"]))
        profile = scoreline_candidate_profile(
            prob_matrix,
            home_goals,
            away_goals,
            is_knockout=is_knockout,
        )
        if profile["score"] in selected:
            selected[profile["score"]]["source"] = "probabilidad_exacta+ev"
            continue
        profile["source"] = "ev"
        selected[profile["score"]] = profile

    return sorted(
        selected.values(),
        key=lambda row: (-float(row["exact_probability"]), -float(row["expected_points"]), row["score"]),
    )


def summarize_regulation_points(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "regulation_points_total": 0.0,
            "regulation_points_avg": 0.0,
            "regulation_points_pct_of_max": 0.0,
            "regulation_full_hit_rate": 0.0,
            "regulation_zero_rate": 0.0,
            "regulation_points_distribution": {},
        }

    points = [float(row.get("regulation_points", 0.0)) for row in rows]
    max_points = [float(row.get("regulation_max_points", 10.0)) for row in rows]
    total_max = sum(max_points)
    distribution = Counter(int(value) if float(value).is_integer() else value for value in points)

    return {
        "regulation_points_total": float(sum(points)),
        "regulation_points_avg": float(sum(points) / len(points)),
        "regulation_points_pct_of_max": float(sum(points) / total_max) if total_max > 0 else 0.0,
        "regulation_full_hit_rate": float(
            sum(1 for point_value, max_value in zip(points, max_points, strict=False) if point_value == max_value) / len(points)
        ),
        "regulation_zero_rate": float(sum(1 for point_value in points if point_value == 0) / len(points)),
        "regulation_points_distribution": dict(sorted(distribution.items())),
    }
