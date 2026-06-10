from __future__ import annotations

import unittest

from alttez.reports.world_cup_brief import (
    build_world_cup_brief_text,
    build_world_cup_summary_text,
    render_world_cup_brief_html,
)


def _bracket() -> dict:
    return {
        "model_family": "poisson",
        "model_version": "poisson-champion-v2",
        "used_fallback": False,
        "champion": "Brazil",
        "runner_up": "France",
        "groups": {
            "A": [
                {"team": "Brazil", "points": 9, "goal_difference": 5, "position": 1},
                {"team": "Serbia", "points": 6, "goal_difference": 1, "position": 2},
                {"team": "Cameroon", "points": 3, "goal_difference": -2, "position": 3, "third_rank": 5},
                {"team": "Panama", "points": 0, "goal_difference": -4, "position": 4},
            ],
        },
        "knockout": {
            "round_of_32": [
                {"home_team": "Brazil", "away_team": "Serbia", "home_score": 2, "away_score": 0, "winner": "Brazil", "decided_by": "regular_time"},
            ],
            "round_of_16": [
                {"home_team": "Brazil", "away_team": "Spain", "home_score": 1, "away_score": 1, "winner": "Brazil", "decided_by": "penalties"},
            ],
            "quarter_final": [
                {"home_team": "Brazil", "away_team": "Germany", "home_score": 2, "away_score": 1, "winner": "Brazil", "decided_by": "regular_time"},
            ],
            "semi_final": [
                {"home_team": "Brazil", "away_team": "Argentina", "home_score": 1, "away_score": 0, "winner": "Brazil", "decided_by": "regular_time"},
            ],
            "final": [
                {"home_team": "Brazil", "away_team": "France", "home_score": 2, "away_score": 1, "winner": "Brazil", "decided_by": "regular_time"},
            ],
        },
        "final": {"home_team": "Brazil", "away_team": "France", "home_score": 2, "away_score": 1, "winner": "Brazil", "decided_by": "regular_time"},
    }


def _odds() -> list[dict]:
    return [
        {"team": "Brazil", "champion_probability": 0.22, "final_probability": 0.35},
        {"team": "France", "champion_probability": 0.18, "final_probability": 0.31},
    ]


class TestWorldCupBrief(unittest.TestCase):
    def test_brief_covers_every_phase(self) -> None:
        text = build_world_cup_brief_text(_bracket(), odds=_odds())
        self.assertIn("Campeón pronosticado: Brazil", text)
        self.assertIn("Subcampeón: France", text)
        self.assertIn("FASE DE GRUPOS", text)
        self.assertIn("Grupo A", text)
        for label in ("Dieciseisavos", "Octavos", "Cuartos", "Semifinales", "Final"):
            self.assertIn(label, text)
        self.assertIn("Probabilidades de título", text)
        self.assertIn("22.0%", text)  # champion odds

    def test_penalties_are_flagged(self) -> None:
        text = build_world_cup_brief_text(_bracket())
        self.assertIn("(pen.)", text)

    def test_qualified_marks_top_two_and_best_third(self) -> None:
        text = build_world_cup_brief_text(_bracket())
        # Brazil (1), Serbia (2) and Cameroon (3rd, rank 5 <= 8) qualify; Panama does not.
        self.assertIn("✅ 1. Brazil", text)
        self.assertIn("✅ 3. Cameroon", text)
        self.assertIn("▫️ 4. Panama", text)


class TestWorldCupBriefHtml(unittest.TestCase):
    def test_html_is_branded_and_covers_phases(self) -> None:
        html = render_world_cup_brief_html(_bracket(), odds=_odds())
        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        # Brand identity present.
        self.assertIn("ALT", html)
        self.assertIn("Sora", html)  # brand font
        self.assertIn("#C27A42", html)  # copper accent
        self.assertIn("UN SOLO SISTEMA", html)  # tagline
        # Content: champion + every phase.
        self.assertIn("Brazil", html)
        self.assertIn("Fase de grupos", html)
        self.assertIn("Eliminatorias", html)
        self.assertIn("Probabilidades", html)

    def test_summary_text_is_short_and_points_to_html(self) -> None:
        summary = build_world_cup_summary_text(_bracket(), odds=_odds())
        self.assertIn("Campeón: Brazil", summary)
        self.assertIn("HTML adjunto", summary)
        # Short: a caption, not the full bracket.
        self.assertLess(len(summary), 600)


if __name__ == "__main__":
    unittest.main()
