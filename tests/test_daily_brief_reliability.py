from __future__ import annotations

import unittest

from alttez.reports.daily_brief.reliability import (
    RELIABILITY_HIGH,
    RELIABILITY_NONE,
    compute_match_reliability,
    evaluate_delivery_gate,
    summarize_reliability,
)


def _strong_match() -> dict:
    return {
        "prediction_supported": True,
        "used_fallback": False,
        "discordance_warning": False,
        "effective_matches_local": 40.0,
        "effective_matches_visitante": 35.0,
        "confidence_local": 0.8,
        "confidence_visitante": 0.75,
        "best_market_probability": 0.62,
    }


class TestComputeMatchReliability(unittest.TestCase):
    def test_strong_support_yields_high_and_model_backed(self) -> None:
        marker = compute_match_reliability(_strong_match())

        self.assertEqual(marker["label"], RELIABILITY_HIGH)
        self.assertGreaterEqual(marker["score"], 70)
        self.assertTrue(marker["is_model_backed"])

    def test_unsupported_match_still_gets_marker(self) -> None:
        marker = compute_match_reliability(
            {"prediction_supported": False, "support_message": "Sin historial."}
        )

        self.assertEqual(marker["label"], RELIABILITY_NONE)
        self.assertEqual(marker["score"], 0)
        self.assertFalse(marker["is_model_backed"])
        self.assertTrue(marker["reasons"])  # always explains why

    def test_fallback_halves_score_and_drops_model_backed(self) -> None:
        baseline = compute_match_reliability(_strong_match())
        fallback = compute_match_reliability(_strong_match(), run_used_fallback=True)

        self.assertLess(fallback["score"], baseline["score"])
        self.assertFalse(fallback["is_model_backed"])
        self.assertTrue(any("respaldo" in reason.lower() for reason in fallback["reasons"]))

    def test_discordance_is_penalized_and_noted(self) -> None:
        match = _strong_match()
        match["discordance_warning"] = True
        marker = compute_match_reliability(match)

        self.assertTrue(any("discrepan" in reason.lower() for reason in marker["reasons"]))


class TestDeliveryGate(unittest.TestCase):
    def _payload(self, **overrides) -> dict:
        matches = [dict(_strong_match(), reliability=compute_match_reliability(_strong_match()))]
        payload = {"used_fallback": False, "matches": matches}
        payload.update(overrides)
        return payload

    def test_gate_allows_model_backed_run(self) -> None:
        gate = evaluate_delivery_gate(self._payload())

        self.assertTrue(gate["should_deliver"])
        self.assertIsNone(gate["block_reason"])
        self.assertEqual(gate["model_backed_matches"], 1)

    def test_gate_blocks_empty_agenda(self) -> None:
        gate = evaluate_delivery_gate({"used_fallback": False, "matches": []})

        self.assertFalse(gate["should_deliver"])
        self.assertIn("agenda", gate["block_reason"].lower())

    def test_gate_blocks_global_fallback(self) -> None:
        gate = evaluate_delivery_gate(self._payload(used_fallback=True, fallback_reason="artifact missing"))

        self.assertFalse(gate["should_deliver"])
        self.assertIn("respaldo", gate["block_reason"].lower())

    def test_gate_blocks_when_no_match_is_model_backed(self) -> None:
        unsupported = {"prediction_supported": False, "support_message": "Sin historial."}
        unsupported["reliability"] = compute_match_reliability(unsupported)
        gate = evaluate_delivery_gate({"used_fallback": False, "matches": [unsupported]})

        self.assertFalse(gate["should_deliver"])
        self.assertIn("genérico", gate["block_reason"].lower())

    def test_gate_blocks_when_marker_missing(self) -> None:
        gate = evaluate_delivery_gate({"used_fallback": False, "matches": [_strong_match()]})

        self.assertFalse(gate["should_deliver"])
        self.assertIn("marcador", gate["block_reason"].lower())


class TestSummarize(unittest.TestCase):
    def test_summary_counts_model_backed(self) -> None:
        matches = [dict(_strong_match(), reliability=compute_match_reliability(_strong_match())) for _ in range(3)]
        summary = summarize_reliability(matches)

        self.assertEqual(summary["total_matches"], 3)
        self.assertEqual(summary["model_backed_matches"], 3)


if __name__ == "__main__":
    unittest.main()
