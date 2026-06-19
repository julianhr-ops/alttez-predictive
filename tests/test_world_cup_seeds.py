from __future__ import annotations

import unittest

import pandas as pd

from scripts.generate_world_cup_seeds import (
    build_seed_rows,
    compute_strength_table,
)


def _history() -> pd.DataFrame:
    # Strong vs weak: strong scores a lot and concedes little.
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2021-01-01", "2021-02-01"],
            "home_team": ["Strong", "Strong", "Weak", "Weak"],
            "away_team": ["Weak", "Weak", "Strong", "Strong"],
            "home_score": [3.0, 4.0, 0.0, 1.0],
            "away_score": [0.0, 1.0, 3.0, 2.0],
        }
    )


class TestComputeStrengthTable(unittest.TestCase):
    def test_neutral_scale_centred_on_global_average(self) -> None:
        table = compute_strength_table(_history())
        # Strong attack must exceed neutral 1.0; its defense (few conceded) must be below 1.0.
        self.assertGreater(table.loc["Strong", "fuerza_ataque"], 1.0)
        self.assertLess(table.loc["Strong", "fuerza_defensa"], 1.0)
        self.assertLess(table.loc["Weak", "fuerza_ataque"], 1.0)

    def test_since_year_filters_history(self) -> None:
        table = compute_strength_table(_history(), since_year=2021)
        # Only 2021 rows survive: each team has 2 matches.
        self.assertEqual(int(table.loc["Strong", "partidos"]), 2)

    def test_empty_history_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            compute_strength_table(_history(), since_year=3000)


class TestBuildSeedRows(unittest.TestCase):
    def test_canonical_alias_resolves_to_history(self) -> None:
        table = compute_strength_table(_history())
        table = table.rename(index={"Strong": "DR Congo"})
        rows, unresolved = build_seed_rows(
            ["Congo DR"], known_names={"DR Congo"}, strength_table=table, min_matches=1
        )
        self.assertEqual(unresolved, [])
        self.assertEqual(rows[0]["equipo"], "Congo DR")  # fixture spelling preserved
        self.assertEqual(rows[0]["equipo_canonico"], "DR Congo")

    def test_low_history_team_falls_to_neutral(self) -> None:
        table = compute_strength_table(_history())
        rows, unresolved = build_seed_rows(
            ["Strong"], known_names={"Strong"}, strength_table=table, min_matches=10
        )
        self.assertEqual(rows, [])
        self.assertTrue(any("Strong" in item for item in unresolved))

    def test_unknown_team_is_unresolved(self) -> None:
        table = compute_strength_table(_history())
        rows, unresolved = build_seed_rows(
            ["Atlantis"], known_names=set(), strength_table=table, min_matches=1
        )
        self.assertEqual(rows, [])
        self.assertIn("Atlantis", unresolved)


if __name__ == "__main__":
    unittest.main()
