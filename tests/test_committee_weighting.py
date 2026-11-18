from __future__ import annotations

import unittest

import numpy as np

from alttez.inference.committee_weighting import (
    _enforce_champion_floor,
    resolve_committee_weights,
)

DEFAULTS = {"champion": 0.5, "challenger": 0.3, "poisson_baseline": 0.2}


def _audit(*rows: tuple[str, int, float]) -> list[dict]:
    return [
        {"model_version": v, "evaluated_predictions": n, "brier_score": b}
        for v, n, b in rows
    ]


MEMBERS = {"champion": "poisson-champion-v2", "challenger": "xgb-v2", "poisson_baseline": "poisson-baseline-committee"}


def _noisy(center: float, n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(center, 0.2, n)


class TestConsensusGuard(unittest.TestCase):
    def test_aggregate_briers_over_different_match_sets_never_trigger_the_guard(self) -> None:
        decision = resolve_committee_weights(
            MEMBERS,
            _audit(("poisson-champion-v2", 182, 0.608), ("xgb-v2", 40, 0.616), ("committee-consensus-v1", 85, 0.696)),
            DEFAULTS,
            committee_version="committee-consensus-v1",
        )
        self.assertFalse(decision.deferred_to_champion)
        self.assertNotEqual(decision.strategy, "champion_guard")
        self.assertIn("Guardia de consenso no evaluada", decision.reason)

    def test_defers_to_champion_when_consensus_clearly_worse_on_identical_matches(self) -> None:
        champion_losses = _noisy(0.60, 80)
        decision = resolve_committee_weights(
            MEMBERS,
            [],
            DEFAULTS,
            committee_version="committee-consensus-v1",
            paired_losses=(champion_losses + 0.08, champion_losses),
        )
        self.assertTrue(decision.deferred_to_champion)
        self.assertEqual(decision.strategy, "champion_guard")
        self.assertEqual(decision.weights["champion"], 1.0)
        self.assertEqual(decision.weights["challenger"], 0.0)

    def test_no_defer_when_consensus_better_on_identical_matches(self) -> None:
        champion_losses = _noisy(0.64, 80)
        decision = resolve_committee_weights(
            MEMBERS,
            [],
            DEFAULTS,
            committee_version="committee-consensus-v1",
            paired_losses=(champion_losses - 0.04, champion_losses),
        )
        self.assertFalse(decision.deferred_to_champion)

    def test_no_defer_when_worse_by_less_than_the_noise(self) -> None:
        noise = np.random.default_rng(5).normal(0.0, 0.2, 80)
        champion_losses = _noisy(0.60, 80)
        consensus_losses = champion_losses + (noise - noise.mean()) + 0.002
        decision = resolve_committee_weights(
            MEMBERS,
            [],
            DEFAULTS,
            committee_version="committee-consensus-v1",
            paired_losses=(consensus_losses, champion_losses),
        )
        self.assertFalse(decision.deferred_to_champion)

    def test_no_defer_below_minimum_sample(self) -> None:
        champion_losses = _noisy(0.60, 10)
        decision = resolve_committee_weights(
            MEMBERS,
            [],
            DEFAULTS,
            committee_version="committee-consensus-v1",
            paired_losses=(champion_losses + 0.08, champion_losses),
        )
        self.assertFalse(decision.deferred_to_champion)

    def test_rejects_paired_series_of_different_length(self) -> None:
        with self.assertRaises(ValueError):
            resolve_committee_weights(
                MEMBERS,
                [],
                DEFAULTS,
                committee_version="committee-consensus-v1",
                paired_losses=(_noisy(0.6, 80), _noisy(0.6, 60)),
            )


class TestInverseBrier(unittest.TestCase):
    def test_lower_brier_member_gets_more_weight(self) -> None:
        decision = resolve_committee_weights(
            {"champion": "champ", "challenger": "chall"},
            _audit(("champ", 100, 0.50), ("chall", 100, 0.70)),
            {"champion": 0.5, "challenger": 0.5},
            committee_version="committee-consensus-v1",
            champion_floor=0.0,
        )
        self.assertEqual(decision.strategy, "inverse_brier")
        self.assertGreater(decision.weights["champion"], decision.weights["challenger"])
        self.assertAlmostEqual(sum(decision.weights.values()), 1.0, places=6)

    def test_champion_floor_enforced(self) -> None:
        decision = resolve_committee_weights(
            {"champion": "champ", "challenger": "chall"},
            _audit(("champ", 100, 0.90), ("chall", 100, 0.30)),
            DEFAULTS,
            committee_version="committee-consensus-v1",
            champion_floor=0.5,
        )
        self.assertGreaterEqual(decision.weights["champion"], 0.5 - 1e-9)

    def test_insufficient_samples_falls_to_default(self) -> None:
        decision = resolve_committee_weights(
            {"champion": "champ", "challenger": "chall"},
            _audit(("champ", 5, 0.50), ("chall", 3, 0.70)),
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
        self.assertAlmostEqual(out["challenger"] / out["poisson_baseline"], 0.5 / 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
