from __future__ import annotations

import unittest

import pandas as pd

from alttez.dashboard.analysis_helpers import (
    build_recommendation_summary,
    filter_noise_matches,
    safe_key,
)


class DashboardAnalysisHelpersTest(unittest.TestCase):
    def test_filter_noise_matches_drops_unsupported_youth_fixture(self) -> None:
        agenda_df = pd.DataFrame(
            [
                {
                    "home_team": "Colombia U17",
                    "away_team": "Brazil U17",
                    "tournament": "U17 World Cup",
                    "stage": "Group Stage",
                    "group_label": "Group A",
                },
                {
                    "home_team": "France U17",
                    "away_team": "Spain U17",
                    "tournament": "U17 Friendly",
                    "stage": "Friendly",
                    "group_label": "",
                },
                {
                    "home_team": "Argentina",
                    "away_team": "Uruguay",
                    "tournament": "International Friendly",
                    "stage": "Friendly",
                    "group_label": "",
                },
            ]
        )

        supported = {
            "Colombia U17": (False, False),
            "France U17": (True, False),
            "Argentina": (False, False),
        }

        filtered = filter_noise_matches(
            agenda_df,
            lambda row_dict: supported[row_dict["home_team"]],
        )

        self.assertEqual(list(filtered["home_team"]), ["France U17", "Argentina"])

    def test_build_recommendation_summary_uses_primary_prediction_and_best_market(self) -> None:
        result = {
            "recomendaciones": pd.DataFrame(
                [
                    {"Pronostico": "1 - 0", "EV": 1.812},
                    {"Pronostico": "1 - 1", "EV": 1.102},
                    {"Pronostico": "2 - 0", "EV": 0.964},
                ]
            ),
            "result_summary": pd.DataFrame(
                {
                    "Mercado": ["Victoria Local", "Empate", "Victoria Visitante"],
                    "Probabilidad": ["52.0%", "28.0%", "20.0%"],
                }
            ),
        }

        summary = build_recommendation_summary(result)

        self.assertEqual(summary["primary_score"], "1 - 0")
        self.assertEqual(summary["primary_ev"], 1.812)
        self.assertEqual(list(summary["alternatives"]["Pronostico"]), ["1 - 1", "2 - 0"])
        self.assertEqual(summary["best_market"]["Mercado"], "Victoria Local")

    def test_safe_key_normalizes_free_text(self) -> None:
        self.assertEqual(safe_key("World Cup: Group A!"), "world-cup-group-a")


if __name__ == "__main__":
    unittest.main()
