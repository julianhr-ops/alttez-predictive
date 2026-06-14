from __future__ import annotations

import unittest

import numpy as np

from alttez.experiments.tournament_oracle import (
    DEFAULT_DIXON_COLES_RHO,
    build_probability_matrix,
    calcular_puntos,
    optimize_prediction,
    validate_points_optimal_recommendations,
)


class TestDixonColesCorrection(unittest.TestCase):
    def test_matrix_without_rho_is_normalized(self) -> None:
        matrix = build_probability_matrix(1.5, 1.1, max_goals=7)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=9)
        self.assertTrue((matrix >= 0).all())

    def test_negative_rho_lifts_draw_mass(self) -> None:
        lambda_home, lambda_away = 1.4, 1.2
        base = build_probability_matrix(lambda_home, lambda_away, max_goals=7, rho=0.0)
        corrected = build_probability_matrix(
            lambda_home, lambda_away, max_goals=7, rho=DEFAULT_DIXON_COLES_RHO
        )

        # Still a valid distribution.
        self.assertAlmostEqual(float(corrected.sum()), 1.0, places=9)
        self.assertTrue((corrected >= 0).all())

        base_draw = float(base.diagonal().sum())
        corrected_draw = float(corrected.diagonal().sum())
        # Negative rho shifts probability mass toward draws / low scores.
        self.assertGreater(corrected_draw, base_draw)

    def test_rho_only_touches_low_score_cells(self) -> None:
        lambda_home, lambda_away = 1.6, 1.0
        base = build_probability_matrix(lambda_home, lambda_away, max_goals=7, rho=0.0)
        corrected = build_probability_matrix(
            lambda_home, lambda_away, max_goals=7, rho=DEFAULT_DIXON_COLES_RHO
        )

        # A high-score cell only changes through renormalization (relative ordering
        # within the non-corrected cells is preserved).
        low_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
        base_high = np.array(
            [base[h, a] for h in range(8) for a in range(8) if (h, a) not in low_cells]
        )
        corrected_high = np.array(
            [corrected[h, a] for h in range(8) for a in range(8) if (h, a) not in low_cells]
        )
        ratios = corrected_high / base_high
        # All non-corrected cells were scaled by the same renormalization factor.
        self.assertTrue(np.allclose(ratios, ratios[0]))


class TestTournamentScoringRules(unittest.TestCase):
    def test_group_stage_examples_match_regulation(self) -> None:
        self.assertEqual(calcular_puntos(1, 0, 1, 0, is_knockout=False), 10)
        self.assertEqual(calcular_puntos(2, 0, 3, 1, is_knockout=False), 6)
        self.assertEqual(calcular_puntos(0, 1, 2, 1, is_knockout=False), 2)

    def test_knockout_stage_doubles_group_stage_points(self) -> None:
        group_points = calcular_puntos(2, 0, 3, 1, is_knockout=False)
        knockout_points = calcular_puntos(2, 0, 3, 1, is_knockout=True)

        self.assertEqual(knockout_points, group_points * 2)

    def test_recommendation_validation_accepts_points_optimal_top_pick(self) -> None:
        matrix = build_probability_matrix(1.4, 0.9, max_goals=5, rho=DEFAULT_DIXON_COLES_RHO)
        recommendations = optimize_prediction(matrix, is_knockout=False).head(5)

        validate_points_optimal_recommendations(matrix, recommendations, is_knockout=False)

    def test_recommendation_validation_rejects_non_optimal_top_pick(self) -> None:
        matrix = build_probability_matrix(1.4, 0.9, max_goals=5, rho=DEFAULT_DIXON_COLES_RHO)
        recommendations = optimize_prediction(matrix, is_knockout=False).head(5).copy()
        recommendations.iloc[0, recommendations.columns.get_loc("Pronostico")] = "5 - 5"

        with self.assertRaises(ValueError):
            validate_points_optimal_recommendations(matrix, recommendations, is_knockout=False)


if __name__ == "__main__":
    unittest.main()
