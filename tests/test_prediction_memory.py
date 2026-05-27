from __future__ import annotations

import unittest

from alttez.operations.prediction_memory import brier_score_1x2, log_loss_1x2


class TestPredictionMemory(unittest.TestCase):
    def test_brier_score_rewards_calibrated_prediction(self) -> None:
        confident_correct = brier_score_1x2(0.75, 0.15, 0.10, "H")
        flat_prediction = brier_score_1x2(0.34, 0.33, 0.33, "H")

        self.assertLess(confident_correct, flat_prediction)

    def test_log_loss_penalizes_wrong_overconfidence(self) -> None:
        calibrated = log_loss_1x2(0.60, 0.25, 0.15, "H")
        wrong_overconfident = log_loss_1x2(0.05, 0.10, 0.85, "H")

        self.assertLess(calibrated, wrong_overconfident)


if __name__ == "__main__":
    unittest.main()
