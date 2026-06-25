import math
import unittest
from unittest.mock import patch

import pandas as pd

from alttez.features.elo import ELO_MOV_ENABLED_ENV, EloEngine


class MarginOfVictoryMultiplierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = EloEngine(mov_enabled=True)

    def test_draw_returns_neutral_multiplier(self):
        # |GD|=0 -> 1.0 (no anular el update con ln(1)=0).
        self.assertEqual(self.engine.margin_of_victory_multiplier(1, 1, 1500, 1500), 1.0)

    def test_bigger_margin_yields_bigger_multiplier(self):
        # Mismos ratings: la goleada pesa mas que la victoria ajustada.
        narrow = self.engine.margin_of_victory_multiplier(1, 0, 1500, 1500)
        blowout = self.engine.margin_of_victory_multiplier(5, 0, 1500, 1500)
        self.assertLess(narrow, blowout)

    def test_upset_amplifies_versus_favorite_winning(self):
        # Mismo marcador 3-0. Favorito (local +300) ganando -> amortiguado.
        # Underdog (local -300) ganando -> amplificado.
        favorite_wins = self.engine.margin_of_victory_multiplier(3, 0, 1800, 1500)
        underdog_wins = self.engine.margin_of_victory_multiplier(3, 0, 1200, 1500)
        self.assertLess(favorite_wins, underdog_wins)

    def test_formula_matches_538_for_even_match(self):
        # Ratings iguales -> termino anti-runaway = 2.2/2.2 = 1 -> mult = ln(|GD|+1).
        self.assertAlmostEqual(
            self.engine.margin_of_victory_multiplier(2, 0, 1500, 1500),
            math.log(3.0),
        )

    def test_away_winner_uses_away_perspective(self):
        # Visitante gana 0-2: la diff del ganador es away_eff - home_eff.
        mult = self.engine.margin_of_victory_multiplier(0, 2, 1500, 1800)
        expected = math.log(3.0) * (2.2 / ((1800 - 1500) * 0.001 + 2.2))
        self.assertAlmostEqual(mult, expected)


class EloMovFlagTest(unittest.TestCase):
    def test_flag_off_by_default(self):
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop(ELO_MOV_ENABLED_ENV, None)
            self.assertFalse(EloEngine().mov_enabled)

    def test_flag_on_via_env(self):
        with patch.dict("os.environ", {ELO_MOV_ENABLED_ENV: "true"}):
            self.assertTrue(EloEngine().mov_enabled)

    def test_explicit_param_overrides_env(self):
        with patch.dict("os.environ", {ELO_MOV_ENABLED_ENV: "true"}):
            self.assertFalse(EloEngine(mov_enabled=False).mov_enabled)


def _match_frame() -> pd.DataFrame:
    # Dos partidos: el segundo es una goleada que MOV debe amplificar.
    return pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "neutral": True, "tournament": "Friendly",
             "home_score": 1, "away_score": 0, "home_result": 1.0},
            {"home_team": "A", "away_team": "B", "neutral": True, "tournament": "Friendly",
             "home_score": 5, "away_score": 0, "home_result": 1.0},
        ]
    )


class EloMovIntegrationTest(unittest.TestCase):
    def test_mov_changes_ratings_versus_baseline(self):
        baseline = EloEngine(mov_enabled=False).compute_elo_features(_match_frame())
        with_mov = EloEngine(mov_enabled=True).compute_elo_features(_match_frame())
        # Las features 'before' del primer partido coinciden (base rating);
        # el rating diverge tras la goleada del segundo partido.
        self.assertNotAlmostEqual(
            float(baseline["elo_away_before"].iloc[1]),
            float(with_mov["elo_away_before"].iloc[1]),
        )

    def test_mov_off_is_unchanged(self):
        # Sin MOV el comportamiento es identico al ELO clasico (sin columnas extra rotas).
        result = EloEngine(mov_enabled=False).compute_elo_features(_match_frame())
        self.assertIn("elo_difference", result.columns)


if __name__ == "__main__":
    unittest.main()
