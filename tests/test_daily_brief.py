from __future__ import annotations

import unittest

from alttez.reports.daily_brief import (
    _build_telegram_summary_text,
    _decision_tier,
    _executive_decision_label,
    exit_code_for_brief_result,
    render_daily_brief_html,
)


class TestDailyBrief(unittest.TestCase):
    def test_decision_tier_prioritizes_supported_confident_matches(self) -> None:
        self.assertEqual(_decision_tier("Analizable", 0.72, 0.49), "Alta prioridad")
        self.assertEqual(_decision_tier("Soporte parcial", 0.42, 0.37), "Seguimiento")
        self.assertEqual(_decision_tier("Sin historial suficiente", 0.10, 0.20), "Baja confianza")

    def test_render_daily_brief_html_contains_core_sections(self) -> None:
        payload = {
            "generated_at": "2026-05-26T02:00:00+00:00",
            "target_date": "2026-05-26",
            "batch_summary": {
                "provider_used": "api_football",
                "certification_status": "certified",
                "total_records": 4,
                "certified_records": 4,
                "rejected_records": 0,
                "coverage_pct": 93.5,
                "quarantine_reason": None,
            },
            "alert_summary": {
                "total": 1,
                "critical": 0,
                "warning": 1,
                "info": 0,
                "items": [
                    {
                        "alert_level": "warning",
                        "title": "Provider degraded",
                        "message": "football_data_org timeout",
                        "delivery_channel": "dashboard",
                        "delivery_status": "skipped",
                    }
                ],
            },
            "summary": {
                "total_matches": 1,
                "analyzable_matches": 1,
                "partial_matches": 0,
                "unsupported_matches": 0,
                "high_priority_matches": 1,
                "critical_alerts": 0,
                "warning_alerts": 1,
                "certification_status": "certified",
                "coverage_pct": 93.5,
                "ranked_matches": [
                    {
                        "kickoff_at": "2026-05-26 18:00:00+00:00",
                        "competition_label": "Amistoso",
                        "home_team": "Nigeria",
                        "away_team": "India",
                        "tournament": "Friendly Cup",
                        "support_label": "Analizable",
                        "decision_tier": "Alta prioridad",
                        "xg_total": 2.2,
                        "best_market": "Victoria Nigeria",
                        "best_market_probability": 0.53,
                        "primary_score": "1 - 0",
                        "primary_ev": 5.337,
                    }
                ],
            },
            "matches": [
                {
                    "competition_label": "Amistoso",
                    "tournament": "Friendly Cup",
                    "stage": "Group Stage",
                    "group_label": "Group A",
                    "home_team": "Nigeria",
                    "away_team": "India",
                    "kickoff_at": "2026-05-26 18:00:00+00:00",
                    "city": "Lagos",
                    "status": "scheduled",
                    "support_label": "Analizable",
                    "decision_tier": "Alta prioridad",
                    "lambda_h": 1.4,
                    "lambda_a": 0.8,
                    "best_market": "Victoria Nigeria",
                    "best_market_probability": 0.53,
                    "primary_score": "1 - 0",
                    "primary_ev": 5.337,
                    "confidence_local": 0.7,
                    "confidence_visitante": 0.58,
                    "matches_local": 20,
                    "matches_visitante": 18,
                    "effective_matches_local": 11.8,
                    "effective_matches_visitante": 10.6,
                    "recent_points_local": 2.1,
                    "recent_goal_diff_local": 0.9,
                    "recent_points_visitante": 1.4,
                    "recent_goal_diff_visitante": -0.2,
                    "alternatives": [{"Pronostico": "2 - 0", "EV": 5.01}],
                    "support_message": None,
                }
            ],
        }

        html = render_daily_brief_html(payload)

        self.assertIn("ALTTEZ Daily Decision Brief", html)
        self.assertIn("Agenda priorizada", html)
        self.assertIn("Nigeria", html)
        self.assertIn("Provider degraded", html)

    def test_build_telegram_summary_text_is_executive_and_compact(self) -> None:
        payload = {
            "target_date": "2026-05-26",
            "summary": {
                "total_matches": 2,
                "analyzable_matches": 1,
                "high_priority_matches": 1,
                "critical_alerts": 0,
                "warning_alerts": 1,
                "coverage_pct": 91.2,
                "ranked_matches": [
                    {
                        "home_team": "Nigeria",
                        "away_team": "India",
                        "competition_label": "Amistoso",
                        "kickoff_at": "2026-05-26 18:00:00+00:00",
                        "best_market": "Victoria Nigeria",
                        "best_market_probability": 0.53,
                        "primary_score": "1 - 0",
                        "primary_ev": 5.337,
                        "support_label": "Analizable",
                    }
                ],
            },
            "batch_summary": {
                "provider_used": "api_football",
                "certification_status": "certified",
                "quarantine_reason": None,
            },
        }

        message = _build_telegram_summary_text(payload)

        self.assertIn("ALTTEZ | Agenda Ejecutiva", message)
        self.assertIn("Top decisiones", message)
        self.assertIn("Nigeria", message)
        self.assertIn("Victoria Nigeria", message)
        self.assertIn("Decision principal", message)
        self.assertIn("Adjuntos", message)

    def test_executive_decision_label_maps_match_priority(self) -> None:
        self.assertEqual(
            _executive_decision_label({"decision_tier": "Alta prioridad", "support_label": "Analizable"}),
            "Decision principal",
        )
        self.assertEqual(
            _executive_decision_label({"decision_tier": "Seguimiento", "support_label": "Soporte parcial"}),
            "Vigilancia",
        )
        self.assertEqual(
            _executive_decision_label({"decision_tier": "Baja confianza", "support_label": "Sin historial suficiente"}),
            "Descarte",
        )

    def test_exit_code_for_brief_result_fails_when_delivery_requested_but_not_delivered(self) -> None:
        self.assertEqual(
            exit_code_for_brief_result({"delivery_status": "failed"}, deliver_requested=True),
            1,
        )
        self.assertEqual(
            exit_code_for_brief_result({"delivery_status": "skipped"}, deliver_requested=False),
            0,
        )


if __name__ == "__main__":
    unittest.main()
