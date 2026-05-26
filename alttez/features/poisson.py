from typing import Dict, Tuple

import pandas as pd


class PoissonEngine:
    """Calcula fortalezas ofensivas y defensivas usando historial reciente."""

    def __init__(
        self,
        years_window: int = 4,
        prior_matches: int = 8,
        min_history_matches: int = 5,
        recency_half_life_days: int = 365,
        recent_form_matches: int = 6,
    ):
        self.years_window = years_window
        self.prior_matches = prior_matches
        self.min_history_matches = min_history_matches
        self.recency_half_life_days = recency_half_life_days
        self.recent_form_matches = recent_form_matches
        self.team_stats: Dict[str, Dict[str, float]] = {}
        self.global_avg_goals = 0.0
        self.global_home_avg_goals = 0.0
        self.global_away_avg_goals = 0.0
        self.fitted_until = None

    def fit(self, df: pd.DataFrame, target_date: str) -> None:
        """Analiza la historia reciente y calcula fortalezas por equipo."""

        required_columns = {
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "neutral",
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas para PoissonEngine: {sorted(missing_columns)}")

        self.team_stats = {}
        self.global_avg_goals = 0.0
        self.global_home_avg_goals = 0.0
        self.global_away_avg_goals = 0.0
        self.fitted_until = pd.to_datetime(target_date)

        start_dt = self.fitted_until - pd.DateOffset(years=self.years_window)
        recent_df = df[(df["date"] >= start_dt) & (df["date"] < self.fitted_until)].copy()
        recent_df["date"] = pd.to_datetime(recent_df["date"])
        recent_df["neutral"] = recent_df["neutral"].fillna(False).astype(bool)
        recent_df["match_weight"] = self._compute_match_weights(recent_df["date"])

        if recent_df.empty:
            raise ValueError(
                f"No hay partidos entre {start_dt.date()} y {self.fitted_until.date()} para entrenar el motor Poisson."
            )

        total_matches = recent_df["match_weight"].sum()
        total_home_goals = self._weighted_total(recent_df["home_score"], recent_df["match_weight"])
        total_away_goals = self._weighted_total(recent_df["away_score"], recent_df["match_weight"])
        total_goals = total_home_goals + total_away_goals

        self.global_home_avg_goals = total_home_goals / total_matches
        self.global_away_avg_goals = total_away_goals / total_matches
        self.global_avg_goals = total_goals / (2 * total_matches)

        teams = set(recent_df["home_team"]).union(set(recent_df["away_team"]))
        for team in teams:
            home_matches = recent_df[(recent_df["home_team"] == team) & (~recent_df["neutral"])]
            away_matches = recent_df[(recent_df["away_team"] == team) & (~recent_df["neutral"])]
            all_matches = recent_df[(recent_df["home_team"] == team) | (recent_df["away_team"] == team)]

            home_weight = home_matches["match_weight"].sum()
            away_weight = away_matches["match_weight"].sum()
            total_weight = all_matches["match_weight"].sum()
            total_count = len(all_matches)
            if total_weight <= 0:
                continue

            home_scored = self._weighted_total(home_matches["home_score"], home_matches["match_weight"])
            away_scored = self._weighted_total(away_matches["away_score"], away_matches["match_weight"])
            home_conceded = self._weighted_total(home_matches["away_score"], home_matches["match_weight"])
            away_conceded = self._weighted_total(away_matches["home_score"], away_matches["match_weight"])

            home_attack_avg = self._blend_rate(home_scored, home_weight, self.global_home_avg_goals)
            away_attack_avg = self._blend_rate(away_scored, away_weight, self.global_away_avg_goals)
            home_defense_avg = self._blend_rate(home_conceded, home_weight, self.global_away_avg_goals)
            away_defense_avg = self._blend_rate(away_conceded, away_weight, self.global_home_avg_goals)

            total_scored = home_scored + away_scored
            total_conceded = home_conceded + away_conceded
            overall_attack_avg = self._blend_rate(total_scored, total_weight, self.global_avg_goals)
            overall_defense_avg = self._blend_rate(total_conceded, total_weight, self.global_avg_goals)
            recent_form = self._compute_recent_form(all_matches, team)
            confidence = self._estimate_confidence(total_count, total_weight)

            self.team_stats[team] = {
                "home_attack": float(home_attack_avg / self.global_home_avg_goals) if self.global_home_avg_goals else 1.0,
                "away_attack": float(away_attack_avg / self.global_away_avg_goals) if self.global_away_avg_goals else 1.0,
                "home_defense": float(home_defense_avg / self.global_away_avg_goals) if self.global_away_avg_goals else 1.0,
                "away_defense": float(away_defense_avg / self.global_home_avg_goals) if self.global_home_avg_goals else 1.0,
                "attack": float(overall_attack_avg / self.global_avg_goals) if self.global_avg_goals else 1.0,
                "defense": float(overall_defense_avg / self.global_avg_goals) if self.global_avg_goals else 1.0,
                "matches": float(total_count),
                "effective_matches": round(float(total_weight), 2),
                "confidence": round(confidence, 3),
                **recent_form,
            }

    def predict_xg(self, team_a: str, team_b: str, is_neutral: bool = False) -> Tuple[float, float]:
        """Cruza estadisticas y devuelve los goles esperados de ambos equipos."""

        if self.global_avg_goals <= 0:
            raise RuntimeError("PoissonEngine no ha sido entrenado todavia. Ejecuta fit() primero.")

        stats_a = self._resolve_stats(team_a)
        stats_b = self._resolve_stats(team_b)

        if is_neutral:
            lambda_a = stats_a["attack"] * stats_b["defense"] * self.global_avg_goals
            lambda_b = stats_b["attack"] * stats_a["defense"] * self.global_avg_goals
        else:
            lambda_a = stats_a["home_attack"] * stats_b["away_defense"] * self.global_home_avg_goals
            lambda_b = stats_b["away_attack"] * stats_a["home_defense"] * self.global_away_avg_goals

        return float(round(max(lambda_a, 0.05), 2)), float(round(max(lambda_b, 0.05), 2))

    def get_team_profile(self, team: str) -> Dict[str, float]:
        """Expone el perfil de un equipo sin depender de metodos privados."""

        return self._resolve_stats(team)

    def _blend_rate(self, total_goals: float, matches_played: float, baseline: float) -> float:
        """Suaviza el promedio del equipo contra el promedio global."""

        if matches_played <= 0:
            return baseline

        return (total_goals + baseline * self.prior_matches) / (matches_played + self.prior_matches)

    def _compute_match_weights(self, match_dates: pd.Series) -> pd.Series:
        if self.recency_half_life_days <= 0:
            return pd.Series(1.0, index=match_dates.index, dtype=float)

        days_ago = (self.fitted_until - match_dates).dt.days.clip(lower=0)
        return 0.5 ** (days_ago / float(self.recency_half_life_days))

    @staticmethod
    def _weighted_total(values: pd.Series, weights: pd.Series) -> float:
        if values.empty:
            return 0.0
        return float((values * weights).sum())

    def _compute_recent_form(self, all_matches: pd.DataFrame, team: str) -> Dict[str, float]:
        recent_matches = all_matches.sort_values("date", ascending=False).head(self.recent_form_matches)
        if recent_matches.empty:
            return {
                "recent_points_per_match": 0.0,
                "recent_goal_diff_per_match": 0.0,
            }

        points_total = 0
        goal_diff_total = 0
        for row in recent_matches.itertuples(index=False):
            is_home = row.home_team == team
            goals_for = row.home_score if is_home else row.away_score
            goals_against = row.away_score if is_home else row.home_score
            goal_diff_total += goals_for - goals_against
            if goals_for > goals_against:
                points_total += 3
            elif goals_for == goals_against:
                points_total += 1

        matches_count = len(recent_matches)
        return {
            "recent_points_per_match": round(points_total / matches_count, 3),
            "recent_goal_diff_per_match": round(goal_diff_total / matches_count, 3),
        }

    def _estimate_confidence(self, raw_matches: int, weighted_matches: float) -> float:
        if raw_matches <= 0:
            return 0.0

        maturity = min(raw_matches / max(self.min_history_matches * 2, 1), 1.0)
        recency_support = min(weighted_matches / max(self.min_history_matches, 1), 1.0)
        return 0.5 * maturity + 0.5 * recency_support

    def _resolve_stats(self, team: str) -> Dict[str, float]:
        stats = self.team_stats.get(team)
        if stats is None:
            return self._get_default_stats()

        if stats.get("matches", 0.0) < float(self.min_history_matches):
            neutral_stats = self._get_default_stats()
            neutral_stats["matches"] = stats.get("matches", 0.0)
            neutral_stats["effective_matches"] = stats.get("effective_matches", 0.0)
            neutral_stats["confidence"] = stats.get("confidence", 0.0)
            neutral_stats["recent_points_per_match"] = stats.get("recent_points_per_match", 0.0)
            neutral_stats["recent_goal_diff_per_match"] = stats.get("recent_goal_diff_per_match", 0.0)
            return neutral_stats

        return stats

    @staticmethod
    def _get_default_stats() -> Dict[str, float]:
        return {
            "home_attack": 1.0,
            "away_attack": 1.0,
            "home_defense": 1.0,
            "away_defense": 1.0,
            "attack": 1.0,
            "defense": 1.0,
            "matches": 0.0,
            "effective_matches": 0.0,
            "confidence": 0.0,
            "recent_points_per_match": 0.0,
            "recent_goal_diff_per_match": 0.0,
        }
