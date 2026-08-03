from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from alttez.reports.daily_brief.cli import generate_daily_brief


class TestDailyBriefIdle(unittest.TestCase):
    @patch("alttez.reports.daily_brief.cli.dispatch_daily_brief")
    @patch("alttez.reports.daily_brief.cli.persist_daily_predictions")
    @patch("alttez.reports.daily_brief.cli.write_daily_brief")
    @patch("alttez.reports.daily_brief.cli.build_daily_brief_payload")
    def test_empty_agenda_skips_persistence_delivery_and_incident_path(
        self,
        build_payload,
        write_brief,
        persist_predictions,
        dispatch_brief,
    ) -> None:
        build_payload.return_value = {
            "matches": [],
            "used_fallback": True,
            "fallback_reason": "model unavailable",
        }
        write_brief.return_value = (Path("brief.html"), Path("brief.json"))

        result = generate_daily_brief(date(2026, 8, 3), deliver=True)

        self.assertEqual(result["delivery_status"], "no_schedule")
        self.assertEqual(result["payload"]["delivery_gate"]["status"], "no_schedule")
        self.assertFalse(result["prediction_persisted"])
        persist_predictions.assert_not_called()
        dispatch_brief.assert_not_called()


if __name__ == "__main__":
    unittest.main()
