from typing import Dict

import pandas as pd
from tqdm import tqdm


class EloEngine:
    """Calcula el ELO historico de las selecciones nacionales."""

    def __init__(self, base_rating: float = 1500, k_factor: float = 30):
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}

    def get_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calcula la probabilidad de que A le gane a B."""

        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_rating(self, rating: float, expected: float, actual: float, k: float) -> float:
        """Aplica la formula ELO."""

        return rating + k * (actual - expected)

    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recorre la historia y calcula el ELO de cada equipo antes del partido."""

        elo_home = []
        elo_away = []
        df = df.copy()

        def get_k(tournament: str) -> float:
            """Pondera la importancia del torneo."""

            tournament = tournament or ""
            if "FIFA World Cup" in tournament:
                return 60
            if "Copa America" in tournament or "UEFA Euro" in tournament:
                return 50
            if "qualification" in tournament.lower():
                return 40
            return self.k_factor

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculando ELO historico"):
            home = row["home_team"]
            away = row["away_team"]

            if home not in self.ratings:
                self.ratings[home] = self.base_rating
            if away not in self.ratings:
                self.ratings[away] = self.base_rating

            # Guardar el ELO antes del partido para evitar data leakage.
            current_home_elo = self.ratings[home]
            current_away_elo = self.ratings[away]

            home_advantage = 100 if not row["neutral"] else 0

            expected_home = self.get_expected_score(current_home_elo + home_advantage, current_away_elo)
            expected_away = 1 - expected_home

            k = get_k(row["tournament"])
            self.ratings[home] = self.update_rating(current_home_elo, expected_home, row["home_result"], k)
            self.ratings[away] = self.update_rating(current_away_elo, expected_away, 1 - row["home_result"], k)

            elo_home.append(current_home_elo)
            elo_away.append(current_away_elo)

        df["elo_home_before"] = elo_home
        df["elo_away_before"] = elo_away
        df["elo_difference"] = df["elo_home_before"] - df["elo_away_before"]

        return df
