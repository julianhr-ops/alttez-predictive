from __future__ import annotations

import unittest

from alttez.reports.daily_brief import (
    _build_telegram_summary_text,
    _decision_tier,
    _executive_decision_label,
    exit_code_for_brief_result,
    localize_prediction_text,
    prob_over_2_5,
    render_daily_brief_html,
    team_display_name,
)


class TestDailyBrief(unittest.TestCase):
    def test_team_and_market_names_are_localized_for_spanish_reports(self) -> None:
        self.assertEqual(team_display_name("Germany"), "Alemania")
        self.assertEqual(team_display_name("Netherlands"), "Países Bajos")
        self.assertEqual(localize_prediction_text("Victoria Germany"), "Victoria Alemania")
        self.assertEqual(localize_prediction_text("Victoria Ivory Coast"), "Victoria Costa de Marfil")

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
                    "primary_score_probability": 0.142,
                    "primary_ev": 5.337,
                    "ev_optimal_score": "1 - 0",
                    "ev_optimal_value": 5.337,
                    "score_probability_chart": [
                        {"score": "1 - 0", "probability": 0.142},
                        {"score": "2 - 0", "probability": 0.104},
                    ],
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
        self.assertIn("https://cdn.jsdelivr.net/npm/chart.js", html)
        self.assertIn("score-chart-0", html)
        self.assertIn("Prob. score", html)
        self.assertIn("Score estrategia EV", html)
        self.assertIn("Pronosticos alternativos", html)

    def test_build_telegram_summary_text_is_executive_and_compact(self) -> None:
        payload = {
            "target_date": "2026-05-26",
            "model_family": "xgboost",
            "model_version": "xgboost-challenger-v1",
            "used_fallback": False,
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
        self.assertIn("xgboost / xgboost-challenger-v1", message)
        self.assertIn("Adjuntos", message)

    def test_build_telegram_summary_text_includes_full_ranked_agenda(self) -> None:
        payload = {
            "target_date": "2026-06-13",
            "model_family": "committee",
            "model_version": "committee-consensus-v1",
            "used_fallback": False,
            "summary": {
                "total_matches": 4,
                "analyzable_matches": 4,
                "high_priority_matches": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
                "coverage_pct": 0.0,
                "ranked_matches": [
                    {
                        "home_team": "Qatar",
                        "away_team": "Switzerland",
                        "best_market": "Victoria Switzerland",
                        "best_market_probability": 0.53,
                        "primary_score": "1 - 1",
                        "primary_ev": 2.921,
                        "support_label": "Analizable",
                    },
                    {
                        "home_team": "Brazil",
                        "away_team": "Morocco",
                        "best_market": "Victoria Brazil",
                        "best_market_probability": 0.52,
                        "primary_score": "1 - 1",
                        "primary_ev": 3.04,
                        "support_label": "Analizable",
                    },
                    {
                        "home_team": "Haiti",
                        "away_team": "Scotland",
                        "best_market": "Victoria Haiti",
                        "best_market_probability": 0.40,
                        "primary_score": "1 - 1",
                        "primary_ev": 3.143,
                        "support_label": "Analizable",
                    },
                    {
                        "home_team": "Australia",
                        "away_team": "Turkey",
                        "best_market": "Victoria Australia",
                        "best_market_probability": 0.38,
                        "primary_score": "1 - 1",
                        "primary_ev": 3.163,
                        "support_label": "Analizable",
                    },
                ],
            },
            "batch_summary": {
                "provider_used": "football_data_org",
                "certification_status": "skipped",
                "quarantine_reason": None,
            },
        }

        message = _build_telegram_summary_text(payload)

        self.assertIn("Australia", message)
        self.assertIn("Turkey", message)

    def test_prob_over_2_5_matches_poisson_sum(self) -> None:
        # lambda total 2.7 -> P(>2.5) = 1 - e^-2.7 (1 + 2.7 + 2.7^2/2) ~ 0.506
        self.assertAlmostEqual(prob_over_2_5(1.5, 1.2), 0.5064, places=3)
        # Lambdas invalidas devuelven None (no rompen el resumen).
        self.assertIsNone(prob_over_2_5(None, 1.0))
        self.assertIsNone(prob_over_2_5(0.0, 0.0))

    def test_telegram_summary_surfaces_expected_goals_not_only_modal_score(self) -> None:
        payload = {
            "target_date": "2026-06-19",
            "model_family": "committee",
            "model_version": "committee-consensus-v1",
            "used_fallback": False,
            "summary": {
                "total_matches": 1,
                "analyzable_matches": 1,
                "high_priority_matches": 1,
                "critical_alerts": 0,
                "warning_alerts": 0,
                "coverage_pct": 0.0,
                "ranked_matches": [
                    {
                        "home_team": "Brazil",
                        "away_team": "Haiti",
                        "best_market": "Victoria Brazil",
                        "best_market_probability": 0.83,
                        "primary_score": "2 - 0",
                        "primary_ev": 5.9,
                        "support_label": "Analizable",
                        "lambda_h": 2.72,
                        "lambda_a": 0.58,
                    }
                ],
            },
            "batch_summary": {
                "provider_used": "espn",
                "certification_status": "skipped",
                "quarantine_reason": None,
            },
        }

        message = _build_telegram_summary_text(payload)

        # El nivel de goles esperado se comunica explicitamente (no solo el modal).
        self.assertIn("Goles esperados", message)
        self.assertIn("Over 2.5", message)
        self.assertIn("Marcador mas probable", message)

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
