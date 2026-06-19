import unittest

from alttez.evaluation.metrics import ranked_probability_score


class RankedProbabilityScoreTest(unittest.TestCase):
    def test_perfect_forecast_is_zero(self):
        self.assertAlmostEqual(ranked_probability_score(1.0, 0.0, 0.0, "H"), 0.0)
        self.assertAlmostEqual(ranked_probability_score(0.0, 1.0, 0.0, "D"), 0.0)
        self.assertAlmostEqual(ranked_probability_score(0.0, 0.0, 1.0, "A"), 0.0)

    def test_worst_forecast_is_one(self):
        # Toda la masa en el extremo opuesto al resultado real.
        self.assertAlmostEqual(ranked_probability_score(0.0, 0.0, 1.0, "H"), 1.0)
        self.assertAlmostEqual(ranked_probability_score(1.0, 0.0, 0.0, "A"), 1.0)

    def test_ordinality_adjacent_error_beats_distant_error(self):
        # Mismo "miss" pero a distinta distancia ordinal: H predicho cuando ocurre D
        # debe penalizar menos que H predicho cuando ocurre A.
        adjacent = ranked_probability_score(1.0, 0.0, 0.0, "D")
        distant = ranked_probability_score(1.0, 0.0, 0.0, "A")
        self.assertLess(adjacent, distant)

    def test_uniform_forecast(self):
        # F = (1/3, 2/3); resultado H -> O = (1, 1).
        # RPS = ((1/3-1)^2 + (2/3-1)^2)/2 = (4/9 + 1/9)/2 = 5/18.
        self.assertAlmostEqual(
            ranked_probability_score(1 / 3, 1 / 3, 1 / 3, "H"),
            5 / 18,
        )

    def test_normalizes_unnormalized_input(self):
        # Probabilidades sin normalizar dan el mismo RPS que su version normalizada.
        raw = ranked_probability_score(2.0, 1.0, 1.0, "H")
        norm = ranked_probability_score(0.5, 0.25, 0.25, "H")
        self.assertAlmostEqual(raw, norm)

    def test_invalid_outcome_raises(self):
        with self.assertRaises(ValueError):
            ranked_probability_score(0.4, 0.3, 0.3, "X")

    def test_zero_total_raises(self):
        with self.assertRaises(ValueError):
            ranked_probability_score(0.0, 0.0, 0.0, "H")


if __name__ == "__main__":
    unittest.main()
