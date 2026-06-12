from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from alttez.api.migrations import MIGRATION_PROBES
from alttez.features.klement import (
    DEFAULT_REST_DAYS,
    build_klement_feature_state,
    build_match_feature_payload,
    compute_continent_advantage_flags,
)
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

        # GDP restored as a static macro prior with a coverage flag.
        self.assertIn("gdp_per_capita_diff", feature_df.columns)
        self.assertIn("gdp_coverage", feature_df.columns)
        # Both teams have macro GDP in this fixture -> coverage flagged.
        self.assertEqual(feature_df.iloc[1]["gdp_coverage"], 1.0)
        self.assertIn("elo_delta_diff_30", feature_df.columns)
        self.assertIn("knockout_weight_diff", feature_df.columns)
        self.assertIn("continent_advantage_diff", feature_df.columns)
        self.assertIn("h2h_win_balance", feature_df.columns)
        self.assertIn("h2h_sample_size", feature_df.columns)
        self.assertNotIn("weather_temperature_c", feature_df.columns)
        self.assertEqual(feature_df.iloc[1]["home_continent_advantage"], 1.0)
        self.assertEqual(feature_df.iloc[1]["away_continent_advantage"], 0.0)
        self.assertGreaterEqual(state.final_ratings["Mexico"], 0.0)

    def test_world_cup_2026_context_helper_removed(self) -> None:
        import alttez.features.klement as klement

        self.assertFalse(hasattr(klement, "is_world_cup_2026_context"))
        self.assertFalse(hasattr(klement, "WORLD_CUP_2026_CONFEDERATIONS"))

    def test_tier1_features_are_causal_and_have_train_serve_parity(self) -> None:
        # Argentina wins three in a row vs different rivals.
        rows = [
            ("2025-01-01", "Argentina", "Chile", 2, 0, True),
            ("2025-01-11", "Argentina", "Peru", 1, 0, True),
            ("2025-01-21", "Argentina", "Bolivia", 3, 1, True),
        ]
        df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"])
        df["date"] = pd.to_datetime(df["date"])

        with patch("alttez.features.klement.load_team_macro_context", return_value={}):
            feature_df, state = build_klement_feature_state(df)

        # Causality: first match has no prior history.
        first = feature_df.iloc[0]
        self.assertEqual(first["home_rest_days"], DEFAULT_REST_DAYS)
        self.assertEqual(first["home_win_streak"], 0.0)
        self.assertEqual(first["home_goals_for_rate"], 0.0)
        self.assertEqual(first["h2h_matches"], 0.0)

        # Third match: Argentina enters on a 2-win streak; rest = 10 days since match 2.
        third = feature_df.iloc[2]
        self.assertEqual(third["home_win_streak"], 2.0)
        self.assertEqual(third["home_rest_days"], 10.0)
        self.assertAlmostEqual(third["home_goals_for_rate"], 1.5)  # (2 + 1) / 2

        # Train/serve parity: a future Argentina match must see a 3-win streak post-history.
        with patch("alttez.features.klement.load_team_macro_context", return_value={}):
            payload = build_match_feature_payload(
                home_team="Argentina",
                away_team="Chile",
                is_neutral=True,
                is_knockout=False,
                recent_state={},
                klement_state=state,
                reference_date=pd.Timestamp("2025-02-01"),
            )
        self.assertEqual(payload["home_win_streak"], 3.0)
        self.assertAlmostEqual(payload["home_goals_for_rate"], 2.0)  # (2 + 1 + 3) / 3
        self.assertEqual(payload["home_rest_days"], 11.0)  # 2025-02-01 minus 2025-01-21

    def test_h2h_result_features_are_causal_and_sparse(self) -> None:
        rows = [
            ("2025-01-01", "Mexico", "Brazil", 1, 0, True),
            ("2025-06-01", "Brazil", "Mexico", 2, 2, True),
            ("2025-12-01", "Brazil", "Mexico", 3, 1, True),
        ]
        df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"])
        df["date"] = pd.to_datetime(df["date"])

        with patch("alttez.features.klement.load_team_macro_context", return_value={}):
            feature_df, state = build_klement_feature_state(df)

        self.assertEqual(feature_df.iloc[0]["h2h_sample_size"], 0.0)
        self.assertEqual(feature_df.iloc[1]["h2h_away_wins"], 1.0)
        self.assertEqual(feature_df.iloc[1]["h2h_win_balance"], -1.0)

        payload = build_match_feature_payload(
            home_team="Mexico",
            away_team="Brazil",
            is_neutral=True,
            is_knockout=False,
            recent_state={},
            klement_state=state,
            reference_date=pd.Timestamp("2026-01-01"),
        )

        self.assertEqual(payload["h2h_sample_size"], 3.0)
        self.assertEqual(payload["h2h_home_wins"], 1.0)
        self.assertEqual(payload["h2h_away_wins"], 1.0)
        self.assertEqual(payload["h2h_draws"], 1.0)
        self.assertEqual(payload["h2h_win_balance"], 0.0)

    def test_continent_advantage_uses_same_host_mapping_for_inference_context(self) -> None:
        home_flag, away_flag, diff = compute_continent_advantage_flags(
            "CONCACAF",
            "UEFA",
            tournament_context={"tournament": "FIFA World Cup 2026", "edition": 2026},
        )

        self.assertEqual(home_flag, 1.0)
        self.assertEqual(away_flag, 0.0)
        self.assertEqual(diff, 1.0)
