from __future__ import annotations

import unittest

from alttez.inference.committee_weighting import (
    resolve_committee_weights,
    _enforce_champion_floor,
)

DEFAULTS = {"champion": 0.5, "challenger": 0.3, "poisson_baseline": 0.2}


def _audit(*rows: tuple[str, int, float]) -> list[dict]:
    return [
        {"model_version": v, "evaluated_predictions": n, "brier_score": b}
        for v, n, b in rows
    ]


class TestConsensusGuard(unittest.TestCase):
    def test_defers_to_champion_when_consensus_worse(self) -> None:
        decision = resolve_committee_weights(
            {"champion": "poisson-champion-v2", "challenger": "xgb-v2", "poisson_baseline": "poisson-baseline-committee"},
            _audit(("poisson-champion-v2", 182, 0.608), ("xgb-v2", 40, 0.616), ("committee-consensus-v1", 85, 0.696)),
            DEFAULTS,
            committee_version="committee-consensus-v1",
        )
        self.assertTrue(decision.deferred_to_champion)
        self.assertEqual(decision.strategy, "champion_guard")
        self.assertEqual(decision.weights["champion"], 1.0)
        self.assertEqual(decision.weights["challenger"], 0.0)

    def test_no_defer_when_consensus_better(self) -> None:
        decision = resolve_committee_weights(
            {"champion": "poisson-champion-v2", "poisson_baseline": "poisson-baseline-committee"},
            _audit(("poisson-champion-v2", 182, 0.640), ("committee-consensus-v1", 85, 0.600)),
            DEFAULTS,
            committee_version="committee-consensus-v1",
        )
        self.assertFalse(decision.deferred_to_champion)


class TestInverseBrier(unittest.TestCase):
    def test_lower_brier_member_gets_more_weight(self) -> None:
        decision = resolve_committee_weights(
            {"champion": "champ", "challenger": "chall"},
            _audit(("champ", 100, 0.50), ("chall", 100, 0.70)),
            {"champion": 0.5, "challenger": 0.5},
            committee_version="committee-consensus-v1",
            champion_floor=0.0,  # isolate inverse-brier effect
        )
        self.assertEqual(decision.strategy, "inverse_brier")
        self.assertGreater(decision.weights["champion"], decision.weights["challenger"])
        self.assertAlmostEqual(sum(decision.weights.values()), 1.0, places=6)

    def test_champion_floor_enforced(self) -> None:
        decision = resolve_committee_weights(
            {"champion": "champ", "challenger": "chall"},
            _audit(("champ", 100, 0.90), ("chall", 100, 0.30)),  # challenger much better
            DEFAULTS,
            committee_version="committee-consensus-v1",
            champion_floor=0.5,
        )
        self.assertGreaterEqual(decision.weights["champion"], 0.5 - 1e-9)

    def test_insufficient_samples_falls_to_default(self) -> None:
        decision = resolve_committee_weights(
            {"champion": "champ", "challenger": "chall"},
            _audit(("champ", 5, 0.50), ("chall", 3, 0.70)),  # below min_samples
            DEFAULTS,
            committee_version="committee-consensus-v1",
        )
        self.assertEqual(decision.strategy, "default")
        self.assertEqual(decision.weights, DEFAULTS)


class TestChampionFloor(unittest.TestCase):
    def test_redistributes_excess_to_others(self) -> None:
        out = _enforce_champion_floor({"champion": 0.2, "challenger": 0.5, "poisson_baseline": 0.3}, champion_floor=0.5)
        self.assertAlmostEqual(out["champion"], 0.5, places=6)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)
        # Others keep their relative proportions within the remaining 0.5.
        self.assertAlmostEqual(out["challenger"] / out["poisson_baseline"], 0.5 / 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
