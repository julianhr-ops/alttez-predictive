from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from alttez.operations.prediction_memory import auto_promotion_enabled, production_supported_model_families
from alttez.training.challenger_cycle import _governance_decision, champion_pointer_status


class TestModelGovernance(unittest.TestCase):
    def test_supported_production_model_families_reads_env(self) -> None:
        with patch.dict(os.environ, {"ALTTEZ_SUPPORTED_PRODUCTION_MODEL_FAMILIES": "poisson,xgboost"}, clear=False):
            self.assertEqual(production_supported_model_families(), {"poisson", "xgboost"})

    def test_governance_decision_marks_eligible_when_family_not_supported(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ALTTEZ_SUPPORTED_PRODUCTION_MODEL_FAMILIES": "poisson",
                "ALTTEZ_AUTO_PROMOTE_MODELS": "true",
            },
            clear=False,
        ):
            production_status, activation_mode, reason = _governance_decision(
                promote=True,
                challenger_family="xgboost",
            )
        self.assertEqual(production_status, "eligible")
        self.assertEqual(activation_mode, "blocked")
        self.assertIn("habilitada", reason.lower())

    def test_auto_promotion_enabled_flag(self) -> None:
        with patch.dict(os.environ, {"ALTTEZ_AUTO_PROMOTE_MODELS": "true"}, clear=False):
            self.assertTrue(auto_promotion_enabled())

    def test_champion_pointer_does_not_clobber_existing_active(self) -> None:
        # Challenger not promoted (auto-promote OFF) and a model is already active:
        # the champion baseline must NOT reclaim the active slot, or it would silently
        # demote a manually-promoted model on the nightly cycle (the oscillation bug).
        self.assertEqual(
            champion_pointer_status(challenger_production_status="shadow", active_model_exists=True),
            "shadow",
        )

    def test_champion_pointer_bootstraps_empty_registry(self) -> None:
        self.assertEqual(
            champion_pointer_status(challenger_production_status="shadow", active_model_exists=False),
            "active",
        )

    def test_champion_pointer_archives_when_challenger_promoted(self) -> None:
        self.assertEqual(
            champion_pointer_status(challenger_production_status="active", active_model_exists=True),
            "archived",
        )


if __name__ == "__main__":
    unittest.main()
