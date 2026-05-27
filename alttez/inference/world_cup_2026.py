from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alttez.inference.router import BaseMatchPredictor


DEFAULT_REPORT_DIR = Path("artifacts/world-cup-2026")
DEFAULT_CONFIG_PATH = Path("storage/world_cup_2026/world_cup_2026_config.json")
STAGE_ORDER = [
    "group_stage",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
    "champion",
]


@dataclass(slots=True)
class SimulatedMatch:
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    winner: str | None
    decided_by: str


def load_world_cup_2026_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la configuracion del Mundial 2026 en `{path}`. "
            "Carga el draw oficial antes de correr el simulador."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    groups = config.get("groups") or {}
    if not groups:
        raise ValueError("La configuracion del Mundial 2026 no tiene grupos cargados.")
    return config


def _round_robin_group_fixtures(groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for group_name, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(f"El grupo {group_name} debe tener exactamente 4 equipos.")
        home_away_pairs = [
            (teams[0], teams[1]),
            (teams[2], teams[3]),
            (teams[0], teams[2]),
            (teams[1], teams[3]),
            (teams[0], teams[3]),
            (teams[1], teams[2]),
        ]
        for match_index, (home_team, away_team) in enumerate(home_away_pairs, start=1):
            fixtures.append(
                {
                    "stage": "group_stage",
                    "group": group_name,
                    "match_key": f"{group_name}-{match_index}",
                    "home_team": home_team,
                    "away_team": away_team,
                    "is_neutral": True,
                }
            )
    return fixtures


def _sample_match(
    predictor: BaseMatchPredictor,
    home_team: str,
    away_team: str,
    *,
    rng: np.random.Generator,
    is_neutral: bool,
    is_knockout: bool,
    prediction_cache: dict[tuple[str, str, bool, bool], dict[str, Any]] | None = None,
) -> SimulatedMatch:
    cache_key = (home_team, away_team, is_neutral, is_knockout)
    if prediction_cache is not None and cache_key in prediction_cache:
        prediction = prediction_cache[cache_key]
    else:
        prediction = predictor.predict_match(home_team, away_team, is_neutral=is_neutral, is_knockout=is_knockout)
        if prediction_cache is not None:
            prediction_cache[cache_key] = prediction
    prob_matrix = prediction.get("prob_matrix")
    if isinstance(prob_matrix, np.ndarray) and prob_matrix.size:
        probabilities = prob_matrix.astype(float).ravel()
        probabilities = probabilities / probabilities.sum()
        flat_index = int(rng.choice(np.arange(len(probabilities)), p=probabilities))
        home_score, away_score = divmod(flat_index, prob_matrix.shape[1])
    else:
        outcome_probs = np.array(
            [
                float(prediction.get("prob_home") or 0.0),
                float(prediction.get("prob_draw") or 0.0),
                float(prediction.get("prob_away") or 0.0),
            ],
            dtype=float,
        )
        if outcome_probs.sum() <= 0:
            outcome_probs = np.array([0.4, 0.2, 0.4], dtype=float)
        outcome_probs = outcome_probs / outcome_probs.sum()
        outcome = rng.choice(["H", "D", "A"], p=outcome_probs)
        if outcome == "H":
            home_score, away_score = 1, 0
        elif outcome == "A":
            home_score, away_score = 0, 1
        else:
            home_score, away_score = 1, 1

    if home_score > away_score:
        return SimulatedMatch(home_team, away_team, home_score, away_score, home_team, "regular_time")
    if away_score > home_score:
        return SimulatedMatch(home_team, away_team, home_score, away_score, away_team, "regular_time")
    if not is_knockout:
        return SimulatedMatch(home_team, away_team, home_score, away_score, None, "regular_time")

    home_win = float(prediction.get("prob_home") or 0.0)
    away_win = float(prediction.get("prob_away") or 0.0)
    if home_win + away_win <= 0:
        winner = home_team if rng.random() < 0.5 else away_team
    else:
        winner = home_team if rng.random() < (home_win / (home_win + away_win)) else away_team
    return SimulatedMatch(home_team, away_team, home_score, away_score, winner, "penalties")


def _rank_group_table(table_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        table_rows.values(),
        key=lambda row: (
            -row["points"],
            -row["goal_difference"],
            -row["goals_for"],
            row["team"],
        ),
    )
    for position, row in enumerate(ranked, start=1):
        row["position"] = position
    return ranked


def _simulate_group_stage(
    groups: dict[str, list[str]],
    predictor: BaseMatchPredictor,
    *,
    rng: np.random.Generator,
    prediction_cache: dict[tuple[str, str, bool, bool], dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    standings: dict[str, list[dict[str, Any]]] = {}
    top_two: list[dict[str, Any]] = []
    third_placed: list[dict[str, Any]] = []
    match_log: dict[str, list[dict[str, Any]]] = {}

    for group_name, teams in groups.items():
        table_rows = {
            team: {
                "team": team,
                "group": group_name,
                "points": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
            }
            for team in teams
        }
        match_log[group_name] = []
        for fixture in [item for item in _round_robin_group_fixtures({group_name: teams})]:
            match = _sample_match(
                predictor,
                fixture["home_team"],
                fixture["away_team"],
                rng=rng,
                is_neutral=True,
                is_knockout=False,
                prediction_cache=prediction_cache,
            )
            home_row = table_rows[match.home_team]
            away_row = table_rows[match.away_team]
            home_row["goals_for"] += match.home_score
            home_row["goals_against"] += match.away_score
            away_row["goals_for"] += match.away_score
            away_row["goals_against"] += match.home_score
            home_row["goal_difference"] = home_row["goals_for"] - home_row["goals_against"]
            away_row["goal_difference"] = away_row["goals_for"] - away_row["goals_against"]

            if match.home_score > match.away_score:
                home_row["points"] += 3
                home_row["wins"] += 1
                away_row["losses"] += 1
            elif match.home_score < match.away_score:
                away_row["points"] += 3
                away_row["wins"] += 1
                home_row["losses"] += 1
            else:
                home_row["points"] += 1
                away_row["points"] += 1
                home_row["draws"] += 1
                away_row["draws"] += 1

            match_log[group_name].append(
                {
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_score": match.home_score,
                    "away_score": match.away_score,
                }
            )

        ranked = _rank_group_table(table_rows)
        standings[group_name] = ranked
        top_two.extend(ranked[:2])
        third_placed.append(ranked[2])

    ranked_thirds = sorted(
        third_placed,
        key=lambda row: (-row["points"], -row["goal_difference"], -row["goals_for"], row["group"]),
    )
    for index, row in enumerate(ranked_thirds, start=1):
        row["third_rank"] = index

    return standings, top_two + ranked_thirds[:8], match_log


def _qualified_lookup(qualified_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in qualified_rows:
        label = "winner" if row.get("position") == 1 else "runner_up" if row.get("position") == 2 else "third"
        lookup[f"{label}:{row['group']}"] = row
        if row.get("position") == 3 and row.get("third_rank") is not None:
            lookup[f"third_rank:{row['third_rank']}"] = row
    return lookup


def _resolve_slot(
    slot: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    slot_type = slot["type"]
    if slot_type == "winner":
        return lookup[f"winner:{slot['group']}"]
    if slot_type == "runner_up":
        return lookup[f"runner_up:{slot['group']}"]
    if slot_type == "third_rank":
        return lookup[f"third_rank:{int(slot['rank'])}"]
    if slot_type == "winner_key":
        return lookup[slot["key"]]
    raise ValueError(f"Tipo de slot no soportado: {slot_type}")


def _resolve_third_pool_assignments(
    fixtures: list[dict[str, Any]],
    third_pool_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    slot_candidates: dict[str, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        for side in ("home_slot", "away_slot"):
            slot = fixture[side]
            if slot["type"] != "third_pool":
                continue
            slot_id = f"{fixture['match_key']}:{side}"
            allowed_groups = set(slot.get("groups") or [])
            candidates = [
                row
                for row in sorted(third_pool_rows, key=lambda item: int(item.get("third_rank") or 999))
                if row["group"] in allowed_groups
            ]
            if not candidates:
                raise ValueError(
                    f"No hay terceros elegibles para el slot {slot_id} con grupos {sorted(allowed_groups)}"
                )
            slot_candidates[slot_id] = candidates

    if not slot_candidates:
        return {}

    ordered_slot_ids = sorted(slot_candidates, key=lambda slot_id: len(slot_candidates[slot_id]))
    assignments: dict[str, dict[str, Any]] = {}
    used_groups: set[str] = set()

    def backtrack(index: int) -> bool:
        if index >= len(ordered_slot_ids):
            return True
        slot_id = ordered_slot_ids[index]
        for candidate in slot_candidates[slot_id]:
            candidate_group = str(candidate["group"])
            if candidate_group in used_groups:
                continue
            assignments[slot_id] = candidate
            used_groups.add(candidate_group)
            if backtrack(index + 1):
                return True
            used_groups.remove(candidate_group)
            assignments.pop(slot_id, None)
        return False

    if not backtrack(0):
        unresolved = {
            slot_id: sorted({str(candidate["group"]) for candidate in candidates})
            for slot_id, candidates in slot_candidates.items()
        }
        raise ValueError(
            "No se pudo resolver una asignacion consistente de mejores terceros para el cuadro: "
            f"{unresolved}"
        )
    return assignments


def _play_knockout_round(
    fixtures: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
    predictor: BaseMatchPredictor,
    *,
    rng: np.random.Generator,
    third_pool_rows: list[dict[str, Any]] | None = None,
    prediction_cache: dict[tuple[str, str, bool, bool], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    winners: list[dict[str, Any]] = []
    played_matches: list[dict[str, Any]] = []
    third_pool_assignments = (
        _resolve_third_pool_assignments(fixtures, third_pool_rows)
        if third_pool_rows
        else {}
    )
    for fixture in fixtures:
        if fixture["home_slot"]["type"] == "third_pool":
            home_slot = third_pool_assignments[f"{fixture['match_key']}:home_slot"]
        else:
            home_slot = _resolve_slot(fixture["home_slot"], lookup)
        if fixture["away_slot"]["type"] == "third_pool":
            away_slot = third_pool_assignments[f"{fixture['match_key']}:away_slot"]
        else:
            away_slot = _resolve_slot(fixture["away_slot"], lookup)
        match = _sample_match(
            predictor,
            home_slot["team"],
            away_slot["team"],
            rng=rng,
            is_neutral=True,
            is_knockout=True,
            prediction_cache=prediction_cache,
        )
        winner_team = match.winner or match.home_team
        winner_row = {"team": winner_team}
        winners.append(winner_row)
        played_matches.append(
            {
                "stage": fixture["stage"],
                "match_key": fixture["match_key"],
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "winner": winner_team,
                "decided_by": match.decided_by,
            }
        )
        lookup[fixture["winner_key"]] = winner_row
    return winners, played_matches


def simulate_world_cup_2026(
    predictor: BaseMatchPredictor,
    *,
    config: dict[str, Any],
    iterations: int = 1000,
    random_seed: int = 7,
) -> dict[str, Any]:
    groups = config.get("groups") or {}
    knockout_fixtures = config.get("knockout_fixtures") or []
    if not knockout_fixtures:
        raise ValueError("La configuracion del Mundial 2026 no declara `knockout_fixtures`.")

    teams = sorted({team for group_teams in groups.values() for team in group_teams})
    stage_counts: dict[str, dict[str, int]] = {
        team: {stage: 0 for stage in STAGE_ORDER}
        for team in teams
    }
    position_counts: dict[str, dict[str, int]] = {
        team: defaultdict(int)
        for team in teams
    }

    rng = np.random.default_rng(random_seed)
    latest_standings: dict[str, list[dict[str, Any]]] = {}
    prediction_cache: dict[tuple[str, str, bool, bool], dict[str, Any]] = {}

    for _ in range(iterations):
        standings, qualified_rows, _ = _simulate_group_stage(
            groups,
            predictor,
            rng=rng,
            prediction_cache=prediction_cache,
        )
        latest_standings = standings
        lookup = _qualified_lookup(qualified_rows)
        qualified_thirds = [
            row for row in qualified_rows
            if row.get("position") == 3 and row.get("third_rank") is not None and int(row["third_rank"]) <= 8
        ]

        for group_name, rows in standings.items():
            for row in rows:
                stage_counts[row["team"]]["group_stage"] += 1
                position_counts[row["team"]][f"group_{group_name}_pos_{row['position']}"] += 1
                if row["position"] <= 2 or (row["position"] == 3 and row in qualified_rows):
                    stage_counts[row["team"]]["round_of_32"] += 1

        match_log: list[dict[str, Any]] = []
        for stage_name in ("round_of_32", "round_of_16", "quarter_final", "semi_final", "final"):
            stage_fixtures = [fixture for fixture in knockout_fixtures if fixture["stage"] == stage_name]
            if stage_name == "round_of_32":
                winners, played_matches = _play_knockout_round(
                    stage_fixtures,
                    lookup,
                    predictor,
                    rng=rng,
                    third_pool_rows=qualified_thirds,
                    prediction_cache=prediction_cache,
                )
            else:
                winners, played_matches = _play_knockout_round(
                    stage_fixtures,
                    lookup,
                    predictor,
                    rng=rng,
                    prediction_cache=prediction_cache,
                )
            match_log.extend(played_matches)
            next_stage = {
                "round_of_32": "round_of_16",
                "round_of_16": "quarter_final",
                "quarter_final": "semi_final",
                "semi_final": "final",
                "final": "champion",
            }[stage_name]
            for winner in winners:
                stage_counts[winner["team"]][next_stage] += 1

    rows = []
    for team in teams:
        row = {"team": team}
        for stage in STAGE_ORDER:
            row[f"{stage}_probability"] = stage_counts[team][stage] / float(iterations)
        row["group_winner_probability"] = sum(
            count for key, count in position_counts[team].items() if key.endswith("_pos_1")
        ) / float(iterations)
        row["group_runner_up_probability"] = sum(
            count for key, count in position_counts[team].items() if key.endswith("_pos_2")
        ) / float(iterations)
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(
        by=["champion_probability", "final_probability", "semi_final_probability"],
        ascending=False,
    ).reset_index(drop=True)

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "prediction_cache_size": len(prediction_cache),
        "iterations": iterations,
        "model_version": predictor.model_version,
        "model_family": predictor.model_family,
        "used_fallback": predictor.used_fallback,
        "fallback_reason": predictor.fallback_reason,
        "groups": groups,
        "standings_snapshot": latest_standings,
        "team_probabilities": summary_df.to_dict(orient="records"),
    }


def write_world_cup_2026_report(
    report: dict[str, Any],
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> Path:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"world-cup-2026-simulation-{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    latest_path = output_dir / "world-cup-2026-latest.json"
    latest_path.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
    return output_path


def load_latest_world_cup_2026_report(report_dir: str | Path = DEFAULT_REPORT_DIR) -> dict[str, Any] | None:
    latest_path = Path(report_dir) / "world-cup-2026-latest.json"
    if not latest_path.exists():
        return None
    return json.loads(latest_path.read_text(encoding="utf-8"))
