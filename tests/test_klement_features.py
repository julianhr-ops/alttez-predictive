from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from alttez.api.migrations import MIGRATION_PROBES
from alttez.features.klement import build_klement_feature_state, compute_continent_advantage_flags
from scripts.seed_macro_klement import build_seed_summary, load_macro_dataframe


class TestKlementFeatures(unittest.TestCase):
    def test_migration_probe_registers_klement_schema_change(self) -> None:
        self.assertIn("010_klement_macro_features.sql", MIGRATION_PROBES)

    def test_load_macro_dataframe_normalizes_team_names(self) -> None:
        csv_dir = Path("artifacts/test-klement")
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "macro.csv"
        csv_path.write_text(
            "team_name,gdp_per_capita,confederation_code\nMexico,12000,concacaf\nGermany,50000,uefa\n",
            encoding="utf-8",
        )
        df = load_macro_dataframe(csv_path)

        self.assertEqual(df.loc[0, "normalized_name"], "mexico")
        self.assertEqual(df.loc[1, "confederation_code"], "UEFA")

    def test_build_seed_summary_reports_unmatched_and_null_values(self) -> None:
        df = pd.DataFrame(
            [
                {"team_name": "Mexico", "normalized_name": "mexico", "gdp_per_capita": 12000.0, "confederation_code": "CONCACAF"},
                {"team_name": "Atlantis", "normalized_name": "atlantis", "gdp_per_capita": None, "confederation_code": ""},
            ]
        )

        summary = build_seed_summary(df, {"mexico"})

        self.assertEqual(summary["rows_read"], 2)
        self.assertEqual(summary["teams_to_update"], 1)
        self.assertEqual(summary["rows_with_null_values"], 1)
        self.assertEqual(summary["unmatched_teams"], ["Atlantis"])

    def test_build_klement_feature_state_adds_expected_columns(self) -> None:
        historical_df = pd.DataFrame(
            [
                ("2025-01-10", "Mexico", "USA", 1, 0, True, "Friendly Cup", "friendly", "Friendly", "2025"),
                ("2026-02-15", "Mexico", "Germany", 1, 2, True, "FIFA World Cup 2026", "world_cup", "Round of 16", "2026"),
            ],
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "neutral",
                "tournament",
                "competition_class",
                "stage",
                "season_label",
            ],
        )
        macro_context = {
            "mexico": {"gdp_per_capita": 12000.0, "confederation_code": "CONCACAF"},
            "usa": {"gdp_per_capita": 70000.0, "confederation_code": "CONCACAF"},
            "germany": {"gdp_per_capita": 52000.0, "confederation_code": "UEFA"},
        }

        with patch("alttez.features.klement.load_team_macro_context", return_value=macro_context):
            feature_df, state = build_klement_feature_state(historical_df)

        # GDP dropped from the model feature engineering (dead look-ahead macro).
        self.assertNotIn("gdp_per_capita_diff", feature_df.columns)
        self.assertNotIn("gdp_per_capita_home", feature_df.columns)
        self.assertNotIn("gdp_per_capita_away", feature_df.columns)
        self.assertIn("elo_delta_diff_30", feature_df.columns)
        self.assertIn("knockout_weight_diff", feature_df.columns)
        self.assertIn("continent_advantage_diff", feature_df.columns)
        self.assertEqual(feature_df.iloc[1]["home_continent_advantage"], 1.0)
        self.assertEqual(feature_df.iloc[1]["away_continent_advantage"], 0.0)
        self.assertGreaterEqual(state.final_ratings["Mexico"], 0.0)

    def test_world_cup_2026_context_helper_removed(self) -> None:
        import alttez.features.klement as klement

        self.assertFalse(hasattr(klement, "is_world_cup_2026_context"))
        self.assertFalse(hasattr(klement, "WORLD_CUP_2026_CONFEDERATIONS"))

    def test_continent_advantage_uses_same_host_mapping_for_inference_context(self) -> None:
        home_flag, away_flag, diff = compute_continent_advantage_flags(
            "CONCACAF",
            "UEFA",
            tournament_context={"tournament": "FIFA World Cup 2026", "edition": 2026},
        )

        self.assertEqual(home_flag, 1.0)
        self.assertEqual(away_flag, 0.0)
        self.assertEqual(diff, 1.0)
