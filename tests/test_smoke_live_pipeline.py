from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from scripts.smoke_live_pipeline import EXIT_FAILURE, EXIT_NO_SCHEDULE, run_smoke


class TestSmokeLivePipeline(unittest.TestCase):
    @patch("scripts.smoke_live_pipeline.run_healthcheck", return_value=[])
    @patch("scripts.smoke_live_pipeline.load_local_env")
    @patch("scripts.smoke_live_pipeline.fetch_live_fixtures_from_stack")
    def test_empty_successful_provider_is_no_schedule(
        self,
        fetch,
        _load_env,
        _healthcheck,
    ) -> None:
        fetch.return_value = (
            [],
            [],
            "espn",
            [{"provider": "espn", "status": "ok", "detail": "0 fixtures"}],
        )

        result = run_smoke(date(2026, 8, 3), None, None, "public", False, True)

        self.assertEqual(result, EXIT_NO_SCHEDULE)

    @patch("scripts.smoke_live_pipeline.run_healthcheck", return_value=[])
    @patch("scripts.smoke_live_pipeline.load_local_env")
    @patch("scripts.smoke_live_pipeline.fetch_live_fixtures_from_stack")
    def test_provider_failure_remains_failure(
        self,
        fetch,
        _load_env,
        _healthcheck,
    ) -> None:
        fetch.return_value = (
            [],
            [],
            None,
            [{"provider": "espn", "status": "error", "detail": "timeout"}],
        )

        result = run_smoke(date(2026, 8, 3), None, None, "public", False, True)

        self.assertEqual(result, EXIT_FAILURE)


if __name__ == "__main__":
    unittest.main()
