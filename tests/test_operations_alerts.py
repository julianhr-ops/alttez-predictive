from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

from alttez.operations.alerts import (
    build_operational_alerts,
    dispatch_alert_telegram,
    record_operational_alerts,
)


class TestOperationalAlerts(unittest.TestCase):
    def test_build_operational_alerts_marks_quarantine_and_degradation(self) -> None:
        sync_result = {
            "status": "success",
            "provider_used": "api_football",
            "certification_batch_id": "2026-05-26_api_football",
            "certification_status": "quarantined",
            "certification_certified_records": 1,
            "certification_rejected_records": 1,
            "certification_coverage_pct": 82.5,
            "certification_quarantine_reason": "Coverage below threshold.",
            "attempted_providers": [
                {"provider": "api_football", "status": "ok", "detail": "ok"},
                {"provider": "football_data_org", "status": "error", "detail": "timeout"},
            ],
            "certification_audit_persisted": True,
        }

        alerts = build_operational_alerts(
            sync_result,
            target_date=date(2026, 5, 26),
            provider_used="api_football",
        )

        self.assertGreaterEqual(len(alerts), 2)
        self.assertEqual(alerts[0]["alert_code"], "batch_quarantined")
        self.assertEqual(alerts[0]["alert_level"], "warning")
        self.assertEqual(alerts[1]["alert_code"], "provider_degraded")

    def test_record_operational_alerts_returns_dashboard_only_without_table(self) -> None:
        sync_result = {
            "status": "error",
            "provider_used": "api_football",
            "error_category": "provider_http_error",
            "error": "timeout",
        }

        result = record_operational_alerts(
            None,
            None,
            sync_result,
            target_date=date(2026, 5, 26),
            provider_used="api_football",
        )

        self.assertEqual(result["alerts_total"], 1)
        self.assertFalse(result["alerts_persisted"])
        self.assertEqual(result["alerts_storage_status"], "skipped")
        self.assertEqual(result["alerts_delivery_channel"], "dashboard")
        self.assertEqual(result["alerts_delivery_status"], "skipped")

    @patch("alttez.operations.alerts._send_telegram_message", return_value=(True, None))
    def test_dispatch_alert_telegram_uses_configured_destination(self, mock_send) -> None:
        with patch.dict(
            os.environ,
            {
                "ALTTEZ_ALERTS_TELEGRAM_ENABLED": "true",
                "ALTTEZ_ALERTS_TELEGRAM_BOT_TOKEN": "bot-token",
                "ALTTEZ_ALERTS_TELEGRAM_CHAT_ID": "123",
            },
            clear=False,
        ):
            sent, error_message = dispatch_alert_telegram(
                [
                    {
                        "alert_level": "critical",
                        "title": "Pipeline fallo",
                        "message": "Smoke pipeline failed.",
                        "target_date": "2026-05-26",
                        "provider_used": "pipeline",
                    }
                ]
            )
        self.assertTrue(sent)
        self.assertIsNone(error_message)
        mock_send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
