from __future__ import annotations

import unittest

import numpy as np

from alttez.experiments.tournament_oracle import (
    DEFAULT_DIXON_COLES_RHO,
    build_probability_matrix,
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


if __name__ == "__main__":
    unittest.main()
