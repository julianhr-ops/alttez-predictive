from typing import Dict, Tuple

import pandas as pd


class PoissonEngine:
    """Calcula fortalezas ofensivas y defensivas usando historial reciente."""

    def __init__(self, years_window: int = 4, prior_matches: int = 8):
        self.years_window = years_window
        self.prior_matches = prior_matches
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

        if recent_df.empty:
            raise ValueError(
                f"No hay partidos entre {start_dt.date()} y {self.fitted_until.date()} para entrenar el motor Poisson."
            )

        total_matches = len(recent_df)
        total_home_goals = recent_df["home_score"].sum()
        total_away_goals = recent_df["away_score"].sum()
        total_goals = total_home_goals + total_away_goals

        self.global_home_avg_goals = total_home_goals / total_matches
        self.global_away_avg_goals = total_away_goals / total_matches
        self.global_avg_goals = total_goals / (2 * total_matches)

        teams = set(recent_df["home_team"]).union(set(recent_df["away_team"]))
        for team in teams:
            home_matches = recent_df[recent_df["home_team"] == team]
            away_matches = recent_df[recent_df["away_team"] == team]

            home_count = len(home_matches)
            away_count = len(away_matches)
            total_count = home_count + away_count
            if total_count == 0:
                continue

            home_scored = home_matches["home_score"].sum()
            away_scored = away_matches["away_score"].sum()
            home_conceded = home_matches["away_score"].sum()
            away_conceded = away_matches["home_score"].sum()

            home_attack_avg = self._blend_rate(home_scored, home_count, self.global_home_avg_goals)
            away_attack_avg = self._blend_rate(away_scored, away_count, self.global_away_avg_goals)
            home_defense_avg = self._blend_rate(home_conceded, home_count, self.global_away_avg_goals)
            away_defense_avg = self._blend_rate(away_conceded, away_count, self.global_home_avg_goals)

            total_scored = home_scored + away_scored
            total_conceded = home_conceded + away_conceded
            overall_attack_avg = self._blend_rate(total_scored, total_count, self.global_avg_goals)
            overall_defense_avg = self._blend_rate(total_conceded, total_count, self.global_avg_goals)

            self.team_stats[team] = {
                "home_attack": home_attack_avg / self.global_home_avg_goals if self.global_home_avg_goals else 1.0,
                "away_attack": away_attack_avg / self.global_away_avg_goals if self.global_away_avg_goals else 1.0,
                "home_defense": home_defense_avg / self.global_away_avg_goals if self.global_away_avg_goals else 1.0,
                "away_defense": away_defense_avg / self.global_home_avg_goals if self.global_home_avg_goals else 1.0,
                "attack": overall_attack_avg / self.global_avg_goals if self.global_avg_goals else 1.0,
                "defense": overall_defense_avg / self.global_avg_goals if self.global_avg_goals else 1.0,
                "matches": float(total_count),
            }

    def predict_xg(self, team_a: str, team_b: str, is_neutral: bool = False) -> Tuple[float, float]:
        """Cruza estadisticas y devuelve los goles esperados de ambos equipos."""

        if self.global_avg_goals <= 0:
            raise RuntimeError("PoissonEngine no ha sido entrenado todavia. Ejecuta fit() primero.")

        stats_a = self.team_stats.get(team_a, self._get_default_stats())
        stats_b = self.team_stats.get(team_b, self._get_default_stats())

        if is_neutral:
            lambda_a = stats_a["attack"] * stats_b["defense"] * self.global_avg_goals
            lambda_b = stats_b["attack"] * stats_a["defense"] * self.global_avg_goals
        else:
            lambda_a = stats_a["home_attack"] * stats_b["away_defense"] * self.global_home_avg_goals
            lambda_b = stats_b["away_attack"] * stats_a["home_defense"] * self.global_away_avg_goals

        return float(round(max(lambda_a, 0.05), 2)), float(round(max(lambda_b, 0.05), 2))

    def _blend_rate(self, total_goals: float, matches_played: int, baseline: float) -> float:
        """Suaviza el promedio del equipo contra el promedio global."""

        if matches_played <= 0:
            return baseline

        return (total_goals + baseline * self.prior_matches) / (matches_played + self.prior_matches)

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
        }
