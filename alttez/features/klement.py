from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
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
    # Tier-1 derived variables (scores + dates), all causal.
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_matches_last90",
    "away_matches_last90",
    "home_goals_for_rate",
    "home_goals_against_rate",
    "away_goals_for_rate",
    "away_goals_against_rate",
    "goals_for_rate_diff",
    "home_win_streak",
    "away_win_streak",
    "h2h_goal_diff",
    "h2h_matches",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_win_balance",
    "h2h_sample_size",
    # Tier-2 macro (static prior from public.teams; sparse → coverage flag).
    "gdp_per_capita_home",
    "gdp_per_capita_away",
    "gdp_per_capita_diff",
    "gdp_coverage",
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
    # Tier-1 causal state for rest/congestion/form/H2H features.
    team_match_dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    team_goals_history: dict[str, list[tuple[pd.Timestamp, int, int]]] = field(default_factory=dict)
    h2h_history: dict[tuple[str, str], list[tuple[pd.Timestamp, str, int, int]]] = field(default_factory=dict)


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


DEFAULT_REST_DAYS = 30.0
REST_CAP_DAYS = 180.0
ROLLING_GOALS_WINDOW = 10
H2H_WINDOW = 5
CONGESTION_WINDOW_DAYS = 90


def _h2h_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted((team_a, team_b)))  # type: ignore[return-value]


def _rest_days(dates: list[pd.Timestamp], ref: pd.Timestamp) -> float:
    past = [d for d in dates if d < ref]
    if not past:
        return DEFAULT_REST_DAYS
    return min(float((ref - max(past)).days), REST_CAP_DAYS)


def _congestion(dates: list[pd.Timestamp], ref: pd.Timestamp, window: int = CONGESTION_WINDOW_DAYS) -> float:
    start = ref - pd.Timedelta(days=window)
    return float(sum(1 for d in dates if start <= d < ref))


def _goal_rates(goals: list[tuple[pd.Timestamp, int, int]], ref: pd.Timestamp, n: int = ROLLING_GOALS_WINDOW) -> tuple[float, float]:
    past = [(gf, ga) for (d, gf, ga) in goals if d < ref][-n:]
    if not past:
        return 0.0, 0.0
    gf = sum(item[0] for item in past) / len(past)
    ga = sum(item[1] for item in past) / len(past)
    return float(gf), float(ga)


def _win_streak(goals: list[tuple[pd.Timestamp, int, int]], ref: pd.Timestamp) -> float:
    streak = 0
    for d, gf, ga in sorted((g for g in goals if g[0] < ref), key=lambda g: g[0], reverse=True):
        if gf > ga:
            streak += 1
        else:
            break
    return float(streak)


def _h2h_goal_diff(h2h_list: list[tuple[pd.Timestamp, str, int, int]], ref: pd.Timestamp, home_team: str, k: int = H2H_WINDOW) -> tuple[float, float]:
    past = [(match_home, home_goals - away_goals) for (d, match_home, home_goals, away_goals) in h2h_list if d < ref][-k:]
    if not past:
        return 0.0, 0.0
    # Goal difference from the CURRENT home team's perspective.
    values = [gd if match_home == home_team else -gd for (match_home, gd) in past]
    return float(sum(values) / len(values)), float(len(past))


def _h2h_result_summary(
    h2h_list: list[tuple[pd.Timestamp, str, int, int]],
    ref: pd.Timestamp,
    home_team: str,
    k: int = H2H_WINDOW,
) -> tuple[float, float, float, float, float]:
    past = [(match_home, home_goals, away_goals) for (d, match_home, home_goals, away_goals) in h2h_list if d < ref][-k:]
    if not past:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    home_wins = 0.0
    away_wins = 0.0
    draws = 0.0
    for match_home, match_home_goals, match_away_goals in past:
        if match_home_goals == match_away_goals:
            draws += 1.0
            continue
        historical_home_won = match_home_goals > match_away_goals
        current_home_won = historical_home_won if match_home == home_team else not historical_home_won
        if current_home_won:
            home_wins += 1.0
        else:
            away_wins += 1.0
    sample_size = float(len(past))
    return home_wins, away_wins, draws, home_wins - away_wins, sample_size


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
    team_match_dates: dict[str, list[pd.Timestamp]] = defaultdict(list)
    team_goals_history: dict[str, list[tuple[pd.Timestamp, int, int]]] = defaultdict(list)
    h2h_history: dict[tuple[str, str], list[tuple[pd.Timestamp, str, int, int]]] = defaultdict(list)

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
    home_rest: list[float] = []
    away_rest: list[float] = []
    home_cong: list[float] = []
    away_cong: list[float] = []
    home_gf_rate: list[float] = []
    home_ga_rate: list[float] = []
    away_gf_rate: list[float] = []
    away_ga_rate: list[float] = []
    home_streak: list[float] = []
    away_streak: list[float] = []
    h2h_gd: list[float] = []
    h2h_n: list[float] = []
    h2h_home_wins: list[float] = []
    h2h_away_wins: list[float] = []
    h2h_draws: list[float] = []
    h2h_win_balance: list[float] = []
    h2h_sample_size: list[float] = []
    home_gdp_values: list[float] = []
    away_gdp_values: list[float] = []
    gdp_cov_values: list[float] = []

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

        # Tier-1 causal features (computed from state holding only earlier matches).
        home_rest.append(_rest_days(team_match_dates.get(home_team, []), match_date))
        away_rest.append(_rest_days(team_match_dates.get(away_team, []), match_date))
        home_cong.append(_congestion(team_match_dates.get(home_team, []), match_date))
        away_cong.append(_congestion(team_match_dates.get(away_team, []), match_date))
        h_gf, h_ga = _goal_rates(team_goals_history.get(home_team, []), match_date)
        a_gf, a_ga = _goal_rates(team_goals_history.get(away_team, []), match_date)
        home_gf_rate.append(h_gf)
        home_ga_rate.append(h_ga)
        away_gf_rate.append(a_gf)
        away_ga_rate.append(a_ga)
        home_streak.append(_win_streak(team_goals_history.get(home_team, []), match_date))
        away_streak.append(_win_streak(team_goals_history.get(away_team, []), match_date))
        gd_value, n_value = _h2h_goal_diff(h2h_history.get(_h2h_key(home_team, away_team), []), match_date, home_team)
        h2h_gd.append(gd_value)
        h2h_n.append(n_value)
        h_wins, a_wins, draws, win_balance, sample_size = _h2h_result_summary(
            h2h_history.get(_h2h_key(home_team, away_team), []),
            match_date,
            home_team,
        )
        h2h_home_wins.append(h_wins)
        h2h_away_wins.append(a_wins)
        h2h_draws.append(draws)
        h2h_win_balance.append(win_balance)
        h2h_sample_size.append(sample_size)

        home_macro = macro_context.get(normalize_name(home_team), {})
        away_macro = macro_context.get(normalize_name(away_team), {})
        home_confederations.append(str(home_macro.get("confederation_code") or "").strip().upper())
        away_confederations.append(str(away_macro.get("confederation_code") or "").strip().upper())
        home_gdp = float(home_macro.get("gdp_per_capita", 0.0) or 0.0)
        away_gdp = float(away_macro.get("gdp_per_capita", 0.0) or 0.0)
        home_gdp_values.append(home_gdp)
        away_gdp_values.append(away_gdp)
        gdp_cov_values.append(1.0 if home_gdp > 0.0 and away_gdp > 0.0 else 0.0)

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

        # Update Tier-1 state AFTER computing this row's features (preserves causality).
        home_goals = int(row.home_score)
        away_goals = int(row.away_score)
        team_match_dates[home_team].append(match_date)
        team_match_dates[away_team].append(match_date)
        team_goals_history[home_team].append((match_date, home_goals, away_goals))
        team_goals_history[away_team].append((match_date, away_goals, home_goals))
        h2h_history[_h2h_key(home_team, away_team)].append((match_date, home_team, home_goals, away_goals))

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
    working_df["home_rest_days"] = home_rest
    working_df["away_rest_days"] = away_rest
    working_df["rest_days_diff"] = working_df["home_rest_days"] - working_df["away_rest_days"]
    working_df["home_matches_last90"] = home_cong
    working_df["away_matches_last90"] = away_cong
    working_df["home_goals_for_rate"] = home_gf_rate
    working_df["home_goals_against_rate"] = home_ga_rate
    working_df["away_goals_for_rate"] = away_gf_rate
    working_df["away_goals_against_rate"] = away_ga_rate
    working_df["goals_for_rate_diff"] = working_df["home_goals_for_rate"] - working_df["away_goals_for_rate"]
    working_df["home_win_streak"] = home_streak
    working_df["away_win_streak"] = away_streak
    working_df["h2h_goal_diff"] = h2h_gd
    working_df["h2h_matches"] = h2h_n
    working_df["h2h_home_wins"] = h2h_home_wins
    working_df["h2h_away_wins"] = h2h_away_wins
    working_df["h2h_draws"] = h2h_draws
    working_df["h2h_win_balance"] = h2h_win_balance
    working_df["h2h_sample_size"] = h2h_sample_size
    working_df["gdp_per_capita_home"] = home_gdp_values
    working_df["gdp_per_capita_away"] = away_gdp_values
    working_df["gdp_per_capita_diff"] = working_df["gdp_per_capita_home"] - working_df["gdp_per_capita_away"]
    working_df["gdp_coverage"] = gdp_cov_values

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
        team_match_dates={team_name: list(dates) for team_name, dates in team_match_dates.items()},
        team_goals_history={team_name: list(goals) for team_name, goals in team_goals_history.items()},
        h2h_history={key: list(history) for key, history in h2h_history.items()},
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

    ref = pd.Timestamp(reference_date)
    home_rest = _rest_days(klement_state.team_match_dates.get(home_team, []), ref)
    away_rest = _rest_days(klement_state.team_match_dates.get(away_team, []), ref)
    home_cong = _congestion(klement_state.team_match_dates.get(home_team, []), ref)
    away_cong = _congestion(klement_state.team_match_dates.get(away_team, []), ref)
    home_gf, home_ga = _goal_rates(klement_state.team_goals_history.get(home_team, []), ref)
    away_gf, away_ga = _goal_rates(klement_state.team_goals_history.get(away_team, []), ref)
    home_streak = _win_streak(klement_state.team_goals_history.get(home_team, []), ref)
    away_streak = _win_streak(klement_state.team_goals_history.get(away_team, []), ref)
    h2h_gd, h2h_n = _h2h_goal_diff(klement_state.h2h_history.get(_h2h_key(home_team, away_team), []), ref, home_team)
    h2h_home_wins, h2h_away_wins, h2h_draws, h2h_win_balance, h2h_sample_size = _h2h_result_summary(
        klement_state.h2h_history.get(_h2h_key(home_team, away_team), []),
        ref,
        home_team,
    )

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
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_days_diff": home_rest - away_rest,
        "home_matches_last90": home_cong,
        "away_matches_last90": away_cong,
        "home_goals_for_rate": home_gf,
        "home_goals_against_rate": home_ga,
        "away_goals_for_rate": away_gf,
        "away_goals_against_rate": away_ga,
        "goals_for_rate_diff": home_gf - away_gf,
        "home_win_streak": home_streak,
        "away_win_streak": away_streak,
        "h2h_goal_diff": h2h_gd,
        "h2h_matches": h2h_n,
        "h2h_home_wins": h2h_home_wins,
        "h2h_away_wins": h2h_away_wins,
        "h2h_draws": h2h_draws,
        "h2h_win_balance": h2h_win_balance,
        "h2h_sample_size": h2h_sample_size,
        "gdp_per_capita_home": float(home_macro.get("gdp_per_capita", 0.0) or 0.0),
        "gdp_per_capita_away": float(away_macro.get("gdp_per_capita", 0.0) or 0.0),
        "gdp_per_capita_diff": float(home_macro.get("gdp_per_capita", 0.0) or 0.0)
        - float(away_macro.get("gdp_per_capita", 0.0) or 0.0),
        "gdp_coverage": 1.0
        if float(home_macro.get("gdp_per_capita", 0.0) or 0.0) > 0.0
        and float(away_macro.get("gdp_per_capita", 0.0) or 0.0) > 0.0
        else 0.0,
    }
