from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alttez.api.db import read_dataframe
from alttez.features.elo import EloEngine
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.seed_historical import normalize_name


WORLD_CUP_2026_NAME = "FIFA World Cup 2026"
KNOCKOUT_COMPETITION_CLASSES = {"world_cup", "continental"}
# Single source of truth for the model feature contract. Both the challenger
# trainer and the inference router import this so the two cannot drift apart.
MODEL_FEATURE_COLUMNS = [
    "elo_difference",
    "elo_home_before",
    "elo_away_before",
    "is_neutral",
    "home_recent_points",
    "away_recent_points",
    "home_recent_goal_diff",
    "away_recent_goal_diff",
    "home_elo_delta_30",
    "away_elo_delta_30",
    "elo_delta_diff_30",
    "home_historical_knockout_weight",
    "away_historical_knockout_weight",
    "knockout_weight_diff",
    "home_continent_advantage",
    "away_continent_advantage",
    "continent_advantage_diff",
    "is_knockout",
]
TOURNAMENT_HOST_CONFEDERATIONS = {
    normalize_name("FIFA World Cup"): {
        "1962": {"conmebol"},
        "1970": {"concacaf"},
        "1978": {"conmebol"},
        "1986": {"concacaf"},
        "1994": {"concacaf"},
        "2026": {"concacaf"},
    },
    normalize_name(WORLD_CUP_2026_NAME): {
        "2026": {"concacaf"},
        "*": {"concacaf"},
    },
    normalize_name("Copa America"): {
        "*": {"conmebol"},
    },
    normalize_name("UEFA Euro"): {
        "*": {"uefa"},
    },
    normalize_name("Concacaf Gold Cup"): {
        "*": {"concacaf"},
    },
    normalize_name("AFC Asian Cup"): {
        "*": {"afc"},
    },
    normalize_name("Africa Cup of Nations"): {
        "*": {"caf"},
    },
}


@dataclass(slots=True)
class KlementFeatureState:
    final_ratings: dict[str, float]
    post_match_history: dict[str, list[tuple[pd.Timestamp, float]]]
    knockout_event_history: dict[str, list[tuple[pd.Timestamp, str]]]
    team_macro: dict[str, dict[str, Any]]


def _classify_competition(tournament_name: str | None, stage: str | None) -> str:
    haystack = f"{tournament_name or ''} {stage or ''}".casefold()
    if any(keyword in haystack for keyword in ("friendly", "friendlies", "international friendly")):
        return "friendly"
    if any(keyword in haystack for keyword in ("fifa world cup", "world cup")):
        return "world_cup"
    if any(keyword in haystack for keyword in ("qualification", "qualifying", "qualifier")):
        return "qualifiers"
    if any(
        keyword in haystack
        for keyword in (
            "uefa euro",
            "euro championship",
            "copa america",
            "copa amÃ©rica",
            "nations league",
            "gold cup",
            "africa cup",
            "nations cup",
            "asian cup",
            "aff championship",
            "olympic",
            "confederations",
        )
    ):
        return "continental"
    return "other_international"


def _is_knockout_stage(stage: str | None) -> bool:
    lowered = str(stage or "").strip().lower()
    if not lowered:
        return False
    if any(keyword in lowered for keyword in ("group", "grupo", "league phase", "friendly", "qualification")):
        return False
    keywords = (
        "quarter-final",
        "quarterfinal",
        "semi-final",
        "semifinal",
        "final",
        "round of 32",
        "round of 16",
        "octavos",
        "cuartos",
        "semifinal",
        "elimin",
        "playoff",
        "knockout",
    )
    return any(keyword in lowered for keyword in keywords)


def _elo_k_for_tournament(tournament: str | None, base_k: float) -> float:
    value = tournament or ""
    if "FIFA World Cup" in value:
        return 60
    if "Copa America" in value or "UEFA Euro" in value:
        return 50
    if "qualification" in value.lower():
        return 40
    return base_k


def _latest_rating_at_or_before(history: list[tuple[pd.Timestamp, float]], anchor: pd.Timestamp) -> float | None:
    latest_rating: float | None = None
    for rating_date, rating_value in history:
        if rating_date <= anchor:
            latest_rating = float(rating_value)
        else:
            break
    return latest_rating


def _resolve_tournament_name(context: dict[str, Any] | None, fallback_tournament: str | None = None) -> str:
    if not context:
        return str(fallback_tournament or "")
    return str(context.get("tournament") or fallback_tournament or "")


def _resolve_tournament_edition(
    context: dict[str, Any] | None = None,
    fallback_edition: str | None = None,
) -> str:
    if context:
        edition = context.get("edition")
        if edition is not None and str(edition).strip():
            return str(edition).strip()
    return str(fallback_edition or "").strip()


def _host_confederations_for_context(
    tournament_context: dict[str, Any] | None = None,
    fallback_tournament: str | None = None,
    fallback_edition: str | None = None,
) -> set[str]:
    tournament_name = _resolve_tournament_name(tournament_context, fallback_tournament)
    normalized_tournament = normalize_name(tournament_name) if tournament_name else ""
    edition = _resolve_tournament_edition(tournament_context, fallback_edition)
    host_mapping = TOURNAMENT_HOST_CONFEDERATIONS.get(normalized_tournament)
    if not host_mapping:
        return set()

    if edition and edition in host_mapping:
        return set(host_mapping[edition])
    if "*" in host_mapping:
        return set(host_mapping["*"])
    return set()


def compute_continent_advantage_flags(
    home_confederation: str | None,
    away_confederation: str | None,
    *,
    tournament_context: dict[str, Any] | None = None,
    fallback_tournament: str | None = None,
    fallback_edition: str | None = None,
) -> tuple[float, float, float]:
    host_confederations = _host_confederations_for_context(
        tournament_context=tournament_context,
        fallback_tournament=fallback_tournament,
        fallback_edition=fallback_edition,
    )
    if not host_confederations:
        return 0.0, 0.0, 0.0

    home_flag = 1.0 if str(home_confederation or "").strip().lower() in host_confederations else 0.0
    away_flag = 1.0 if str(away_confederation or "").strip().lower() in host_confederations else 0.0
    return home_flag, away_flag, home_flag - away_flag


def load_team_macro_context(*, database_url: str | None = None) -> dict[str, dict[str, Any]]:
    try:
        df = read_dataframe(
            """
            select
                name,
                normalized_name,
                gdp_per_capita,
                confederation_code
            from public.teams
            """,
            database_url=database_url,
        )
    except Exception:
        return {}

    context: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        normalized = str(row.get("normalized_name") or "").strip()
        if not normalized:
            continue
        context[normalized] = {
            "name": row.get("name"),
            "gdp_per_capita": float(row.get("gdp_per_capita") or 0.0),
            "confederation_code": str(row.get("confederation_code") or "").strip().upper(),
        }
    return context


def load_enriched_historical_matches(*, database_url: str | None = None) -> pd.DataFrame:
    try:
        df = read_dataframe(
            """
            select
                m.match_date as date,
                ht.name as home_team,
                at.name as away_team,
                m.home_score,
                m.away_score,
                coalesce(t.name, m.competition_normalized_name, 'International') as tournament,
                m.is_neutral as neutral,
                coalesce(m.competition_class, 'other_international') as competition_class,
                m.stage,
                m.season_label
            from public.matches as m
            join public.teams as ht
                on ht.id = m.home_team_id
            join public.teams as at
                on at.id = m.away_team_id
            left join public.tournaments as t
                on t.id = m.tournament_id
            where m.status = 'finished'
            order by m.match_date asc, m.created_at asc
            """,
            database_url=database_url,
        )
    except Exception:
        df = InternationalLoader().fetch_data()
        df["competition_class"] = df.apply(
            lambda row: _classify_competition(row.get("tournament"), row.get("stage")),
            axis=1,
        )
        df["stage"] = None
        df["season_label"] = df["date"].dt.year.astype(str)
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["home_result"] = 0.5
    df.loc[df["home_score"] > df["away_score"], "home_result"] = 1.0
    df.loc[df["home_score"] < df["away_score"], "home_result"] = 0.0
    return df


def build_klement_feature_state(
    historical_df: pd.DataFrame,
    *,
    database_url: str | None = None,
    macro_context: dict[str, dict[str, Any]] | None = None,
    elo_window_days: int = 30,
    knockout_window_years: int = 20,
) -> tuple[pd.DataFrame, KlementFeatureState]:
    working_df = historical_df.copy()
    if working_df.empty:
        empty_state = KlementFeatureState(
            final_ratings={},
            post_match_history={},
            knockout_event_history={},
            team_macro={},
        )
        return working_df, empty_state

    working_df["date"] = pd.to_datetime(working_df["date"])
    working_df = working_df.sort_values("date").reset_index(drop=True)
    if "home_result" not in working_df.columns:
        working_df["home_result"] = 0.5
        working_df.loc[working_df["home_score"] > working_df["away_score"], "home_result"] = 1.0
        working_df.loc[working_df["home_score"] < working_df["away_score"], "home_result"] = 0.0
    if "tournament" not in working_df.columns:
        working_df["tournament"] = "International"
    if "competition_class" not in working_df.columns:
        working_df["competition_class"] = working_df["tournament"].map(lambda value: _classify_competition(value, None))
    if "stage" not in working_df.columns:
        working_df["stage"] = None
    if "season_label" not in working_df.columns:
        working_df["season_label"] = working_df["date"].dt.year.astype(str)

    if macro_context is None:
        macro_context = load_team_macro_context(database_url=database_url)
    elo_engine = EloEngine()
    ratings: dict[str, float] = {}
    post_match_history: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    knockout_events: dict[str, list[tuple[pd.Timestamp, str]]] = defaultdict(list)
    seen_knockout_events: set[tuple[str, str]] = set()

    elo_home_before: list[float] = []
    elo_away_before: list[float] = []
    elo_difference: list[float] = []
    home_elo_delta_30: list[float] = []
    away_elo_delta_30: list[float] = []
    home_knockout_weight: list[float] = []
    away_knockout_weight: list[float] = []
    home_confederations: list[str] = []
    away_confederations: list[str] = []
    is_knockout_values: list[int] = []

    for row in working_df.itertuples(index=False):
        match_date = pd.Timestamp(row.date)
        home_team = row.home_team
        away_team = row.away_team

        current_home_elo = float(ratings.get(home_team, elo_engine.base_rating))
        current_away_elo = float(ratings.get(away_team, elo_engine.base_rating))
        elo_home_before.append(current_home_elo)
        elo_away_before.append(current_away_elo)
        elo_difference.append(current_home_elo - current_away_elo)

        anchor_date = match_date - pd.Timedelta(days=elo_window_days)
        previous_home_rating = _latest_rating_at_or_before(post_match_history.get(home_team, []), anchor_date)
        previous_away_rating = _latest_rating_at_or_before(post_match_history.get(away_team, []), anchor_date)
        home_elo_delta_30.append(current_home_elo - previous_home_rating if previous_home_rating is not None else 0.0)
        away_elo_delta_30.append(current_away_elo - previous_away_rating if previous_away_rating is not None else 0.0)

        window_start = match_date - pd.Timedelta(days=365 * knockout_window_years)

        def _count_knockout_events(team_name: str) -> float:
            event_count = sum(1 for event_date, _ in knockout_events.get(team_name, []) if window_start <= event_date < match_date)
            return float(math.log1p(event_count))

        home_knockout_weight.append(_count_knockout_events(home_team))
        away_knockout_weight.append(_count_knockout_events(away_team))

        home_macro = macro_context.get(normalize_name(home_team), {})
        away_macro = macro_context.get(normalize_name(away_team), {})
        home_confederations.append(str(home_macro.get("confederation_code") or "").strip().upper())
        away_confederations.append(str(away_macro.get("confederation_code") or "").strip().upper())

        is_knockout = int(_is_knockout_stage(getattr(row, "stage", None)))
        is_knockout_values.append(is_knockout)

        home_advantage = 100 if not bool(row.neutral) else 0
        expected_home = elo_engine.get_expected_score(current_home_elo + home_advantage, current_away_elo)
        expected_away = 1 - expected_home
        k = _elo_k_for_tournament(getattr(row, "tournament", None), elo_engine.k_factor)
        new_home_elo = elo_engine.update_rating(current_home_elo, expected_home, float(row.home_result), k)
        new_away_elo = elo_engine.update_rating(current_away_elo, expected_away, 1 - float(row.home_result), k)
        ratings[home_team] = new_home_elo
        ratings[away_team] = new_away_elo
        post_match_history[home_team].append((match_date, new_home_elo))
        post_match_history[away_team].append((match_date, new_away_elo))

        competition_class = str(getattr(row, "competition_class", "") or "").strip().lower()
        stage = getattr(row, "stage", None)
        season_label = str(getattr(row, "season_label", "") or match_date.year)
        tournament = str(getattr(row, "tournament", "") or "International")
        if competition_class in KNOCKOUT_COMPETITION_CLASSES and _is_knockout_stage(stage):
            event_key = f"{competition_class}|{normalize_name(tournament)}|{season_label}"
            for team_name in (home_team, away_team):
                dedupe_key = (team_name, event_key)
                if dedupe_key in seen_knockout_events:
                    continue
                seen_knockout_events.add(dedupe_key)
                knockout_events[team_name].append((match_date, event_key))

    working_df["elo_home_before"] = elo_home_before
    working_df["elo_away_before"] = elo_away_before
    working_df["elo_difference"] = elo_difference
    working_df["home_elo_delta_30"] = home_elo_delta_30
    working_df["away_elo_delta_30"] = away_elo_delta_30
    working_df["elo_delta_diff_30"] = working_df["home_elo_delta_30"] - working_df["away_elo_delta_30"]
    working_df["home_historical_knockout_weight"] = home_knockout_weight
    working_df["away_historical_knockout_weight"] = away_knockout_weight
    working_df["knockout_weight_diff"] = (
        working_df["home_historical_knockout_weight"] - working_df["away_historical_knockout_weight"]
    )
    working_df["home_confederation_code"] = home_confederations
    working_df["away_confederation_code"] = away_confederations
    working_df["is_knockout"] = is_knockout_values

    continent_home_values: list[float] = []
    continent_away_values: list[float] = []
    continent_diff_values: list[float] = []
    for row in working_df.itertuples(index=False):
        home_value, away_value, diff_value = compute_continent_advantage_flags(
            getattr(row, "home_confederation_code", None),
            getattr(row, "away_confederation_code", None),
            tournament_context={
                "tournament": getattr(row, "tournament", None),
                "edition": getattr(row, "season_label", None),
            },
            fallback_tournament=getattr(row, "tournament", None),
            fallback_edition=getattr(row, "season_label", None),
        )
        continent_home_values.append(home_value)
        continent_away_values.append(away_value)
        continent_diff_values.append(diff_value)

    working_df["home_continent_advantage"] = continent_home_values
    working_df["away_continent_advantage"] = continent_away_values
    working_df["continent_advantage_diff"] = continent_diff_values

    state = KlementFeatureState(
        final_ratings={team_name: float(rating) for team_name, rating in ratings.items()},
        post_match_history={team_name: list(history) for team_name, history in post_match_history.items()},
        knockout_event_history={team_name: list(history) for team_name, history in knockout_events.items()},
        team_macro=macro_context,
    )
    return working_df, state


def build_match_feature_payload(
    *,
    home_team: str,
    away_team: str,
    is_neutral: bool,
    is_knockout: bool,
    recent_state: dict[str, dict[str, float]],
    klement_state: KlementFeatureState,
    reference_date: pd.Timestamp,
    tournament_context: dict[str, Any] | None = None,
) -> dict[str, float]:
    home_elo = float(klement_state.final_ratings.get(home_team, 1500.0))
    away_elo = float(klement_state.final_ratings.get(away_team, 1500.0))
    anchor_date = pd.Timestamp(reference_date) - pd.Timedelta(days=30)
    home_anchor_rating = _latest_rating_at_or_before(klement_state.post_match_history.get(home_team, []), anchor_date)
    away_anchor_rating = _latest_rating_at_or_before(klement_state.post_match_history.get(away_team, []), anchor_date)

    home_macro = klement_state.team_macro.get(normalize_name(home_team), {})
    away_macro = klement_state.team_macro.get(normalize_name(away_team), {})
    home_confederation = str(home_macro.get("confederation_code") or "").strip().upper()
    away_confederation = str(away_macro.get("confederation_code") or "").strip().upper()
    home_continent_advantage, away_continent_advantage, continent_advantage_diff = compute_continent_advantage_flags(
        home_confederation,
        away_confederation,
        tournament_context=tournament_context,
    )
    home_form = recent_state.get(home_team, {})
    away_form = recent_state.get(away_team, {})

    window_start = pd.Timestamp(reference_date) - pd.Timedelta(days=365 * 20)
    home_knockout_events = sum(
        1 for event_date, _ in klement_state.knockout_event_history.get(home_team, [])
        if window_start <= event_date < pd.Timestamp(reference_date)
    )
    away_knockout_events = sum(
        1 for event_date, _ in klement_state.knockout_event_history.get(away_team, [])
        if window_start <= event_date < pd.Timestamp(reference_date)
    )
    home_knockout_weight = float(math.log1p(home_knockout_events))
    away_knockout_weight = float(math.log1p(away_knockout_events))

    return {
        "elo_difference": home_elo - away_elo,
        "elo_home_before": home_elo,
        "elo_away_before": away_elo,
        "is_neutral": int(is_neutral),
        "home_recent_points": float(home_form.get("recent_points_per_match", 0.0)),
        "away_recent_points": float(away_form.get("recent_points_per_match", 0.0)),
        "home_recent_goal_diff": float(home_form.get("recent_goal_diff_per_match", 0.0)),
        "away_recent_goal_diff": float(away_form.get("recent_goal_diff_per_match", 0.0)),
        "home_elo_delta_30": home_elo - home_anchor_rating if home_anchor_rating is not None else 0.0,
        "away_elo_delta_30": away_elo - away_anchor_rating if away_anchor_rating is not None else 0.0,
        "elo_delta_diff_30": (home_elo - home_anchor_rating if home_anchor_rating is not None else 0.0)
        - (away_elo - away_anchor_rating if away_anchor_rating is not None else 0.0),
        "home_historical_knockout_weight": home_knockout_weight,
        "away_historical_knockout_weight": away_knockout_weight,
        "knockout_weight_diff": home_knockout_weight - away_knockout_weight,
        "home_continent_advantage": home_continent_advantage,
        "away_continent_advantage": away_continent_advantage,
        "continent_advantage_diff": continent_advantage_diff,
        "is_knockout": int(is_knockout),
    }
