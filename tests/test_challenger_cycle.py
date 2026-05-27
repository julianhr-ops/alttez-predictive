from __future__ import annotations

import unittest

from alttez.training.challenger_cycle import _promotion_decision


class TestChallengerCycle(unittest.TestCase):
    def test_promotion_requires_better_brier(self) -> None:
        champion = {"brier_score": 0.6200, "log_loss": 0.9800}
        challenger = {"brier_score": 0.6100, "log_loss": 0.9810}

        promote, reason = _promotion_decision(champion, challenger)

        self.assertTrue(promote)
        self.assertIn("Brier", reason)

    def test_promotion_rejects_material_log_loss_decay(self) -> None:
        champion = {"brier_score": 0.6200, "log_loss": 0.9800}
        challenger = {"brier_score": 0.6100, "log_loss": 1.0500}

        promote, reason = _promotion_decision(champion, challenger)

        self.assertFalse(promote)
        self.assertIn("Log Loss", reason)


if __name__ == "__main__":
    unittest.main()
