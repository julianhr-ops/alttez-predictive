import math
import os
from typing import Dict

import pandas as pd
from tqdm import tqdm

ELO_MOV_ENABLED_ENV = "ALTTEZ_ELO_MOV_ENABLED"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class EloEngine:
    """Calcula el ELO historico de las selecciones nacionales."""

    def __init__(self, base_rating: float = 1500, k_factor: float = 30, mov_enabled: bool | None = None):
        self.base_rating = base_rating
        self.k_factor = k_factor
        # Multiplicador margin-of-victory (estilo 538). Default OFF para no mover el
        # campeon durante fase de grupos; se activa via env para retrain del retador.
        self.mov_enabled = _env_flag(ELO_MOV_ENABLED_ENV) if mov_enabled is None else bool(mov_enabled)
        self.ratings: Dict[str, float] = {}

    def get_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calcula la probabilidad de que A le gane a B."""

        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def margin_of_victory_multiplier(
        self,
        home_goals: float,
        away_goals: float,
        home_eff_rating: float,
        away_eff_rating: float,
    ) -> float:
        """Multiplicador MOV estilo FiveThirtyEight para el factor K.

        K_eff = K * ln(|GD|+1) * 2.2/(elo_diff_ganador*0.001 + 2.2).

        - Goleadas pesan mas que victorias ajustadas (ln(|GD|+1)).
        - El termino anti-runaway usa la diferencia de rating *desde la perspectiva
          del ganador*: si el favorito gana, amortigua (denominador mayor); si gana
          el underdog (diff negativa), amplifica. Evita inflar al favorito que golea.
        - Empates (|GD|=0): devuelve 1.0 -> actualizacion ELO estandar. ln(1)=0
          anularia el update y se perderia la senal del empate; el piso lo evita.
        """
        goal_diff = abs(int(home_goals) - int(away_goals))
        if goal_diff == 0:
            return 1.0
        if home_goals > away_goals:
            winner_rating_diff = home_eff_rating - away_eff_rating
        else:
            winner_rating_diff = away_eff_rating - home_eff_rating
        return math.log(goal_diff + 1.0) * (2.2 / (winner_rating_diff * 0.001 + 2.2))

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
            home_eff = current_home_elo + home_advantage

            expected_home = self.get_expected_score(home_eff, current_away_elo)
            expected_away = 1 - expected_home

            k = get_k(row["tournament"])
            if self.mov_enabled and "home_score" in row and "away_score" in row:
                k *= self.margin_of_victory_multiplier(
                    row["home_score"], row["away_score"], home_eff, current_away_elo
                )
            self.ratings[home] = self.update_rating(current_home_elo, expected_home, row["home_result"], k)
            self.ratings[away] = self.update_rating(current_away_elo, expected_away, 1 - row["home_result"], k)

            elo_home.append(current_home_elo)
            elo_away.append(current_away_elo)

        df["elo_home_before"] = elo_home
        df["elo_away_before"] = elo_away
        df["elo_difference"] = df["elo_home_before"] - df["elo_away_before"]

        return df
