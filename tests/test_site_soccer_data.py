from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alttez.inference.serving_policy import build_serving_policy
from alttez.site import soccer_data

REPO_ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 6, 1)
ACTIVE = [f"Team {chr(65 + k)}" for k in range(12)]
STALE = "Retired Giants"


def _history() -> pd.DataFrame:
    """Twelve active teams with spread strengths, plus one strong team that stopped playing long ago."""
    rng = np.random.default_rng(11)
    strength = {name: 0.6 + 0.12 * k for k, name in enumerate(ACTIVE)}
    rows = []
    for day, home_idx, away_idx in zip(
        pd.date_range("2022-01-01", "2026-05-20", periods=900),
        rng.integers(0, len(ACTIVE), 900),
        rng.integers(0, len(ACTIVE), 900),
        strict=True,
    ):
        if home_idx == away_idx:
            continue
        home, away = ACTIVE[home_idx], ACTIVE[away_idx]
        rows.append(
            {
                "date": day.normalize(),
                "home_team": home,
                "away_team": away,
                "home_score": int(rng.poisson(1.3 * strength[home] / strength[away] ** 0.5)),
                "away_score": int(rng.poisson(1.1 * strength[away] / strength[home] ** 0.5)),
                "tournament": "Friendly" if rng.random() < 0.5 else "FIFA World Cup qualification",
                "neutral": bool(rng.random() < 0.3),
            }
        )
    for day, score in ((pd.Timestamp("2019-03-01"), 5), (pd.Timestamp("2019-06-01"), 4), (pd.Timestamp("2019-09-01"), 3)):
        rows.append(
            {
                "date": day,
                "home_team": STALE,
                "away_team": ACTIVE[0],
                "home_score": score,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)


class RatingsPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = _history()
        cls.payload = soccer_data.build_ratings_payload(
            cls.history, AS_OF, source="test", n=8, generated_at=datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
        )
        cls.policy = build_serving_policy(cls.history, pd.Timestamp(AS_OF))
        cls.n = cls.payload["n"]
        cls.names = [team["name"] for team in cls.payload["teams"]]

    def triple(self, key: str, i: int, j: int) -> list[int]:
        start = ((i * self.n) + j) * 3
        return self.payload[key][start : start + 3]

    def test_envelope(self) -> None:
        self.assertEqual(self.payload["generated_at"], "2026-06-01T12:00:00Z")
        self.assertEqual(self.payload["as_of"], "2026-06-01")
        self.assertEqual(self.payload["window_years"], 4)
        self.assertEqual(self.n, 8)
        self.assertEqual(len(self.payload["neutral"]), 8 * 8 * 3)
        self.assertEqual(len(self.payload["home"]), 8 * 8 * 3)
        self.assertEqual(json.loads(json.dumps(self.payload)), self.payload)

    def test_teams_are_the_best_active_ones_in_rank_order(self) -> None:
        ratings = [team["rating"] for team in self.payload["teams"]]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        self.assertEqual([team["rank"] for team in self.payload["teams"]], list(range(1, 9)))
        self.assertNotIn(STALE, self.names)
        self.assertGreater(self.policy.ratings[STALE], self.policy.ratings[self.names[-1]])
        best_active = sorted((r for name, r in self.policy.ratings.items() if name in ACTIVE), reverse=True)[:8]
        np.testing.assert_allclose(ratings, best_active, atol=1e-6)

    def test_page_value_is_the_pipelines_target_1x2_rounded(self) -> None:
        rng = np.random.default_rng(5)
        for _ in range(60):
            i, j = (int(v) for v in rng.choice(self.n, size=2, replace=False))
            for key, neutral in (("neutral", True), ("home", False)):
                with self.subTest(i=i, j=j, venue=key):
                    target = self.policy.target_1x2(self.names[i], self.names[j], neutral)
                    page = self.triple(key, i, j)
                    exact = [p * soccer_data.SCALE for p in target]
                    largest = max(range(3), key=lambda k: target[k])
                    for k in range(3):
                        if k != largest:
                            self.assertEqual(page[k], round(exact[k]))
                    self.assertLessEqual(abs(page[largest] - exact[largest]), 1.0)

    def test_every_triple_sums_to_10000(self) -> None:
        for key in ("neutral", "home"):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.assertEqual(sum(self.triple(key, i, j)), 10000, (key, i, j))

    def test_neutral_venue_is_symmetric(self) -> None:
        for i in range(self.n):
            for j in range(self.n):
                win, draw, loss = self.triple("neutral", i, j)
                self.assertEqual(self.triple("neutral", j, i), [loss, draw, win], (i, j))

    def test_diagonal_is_zeros(self) -> None:
        for key in ("neutral", "home"):
            for i in range(self.n):
                self.assertEqual(self.triple(key, i, i), [0, 0, 0])

    def test_home_side_gains_over_neutral(self) -> None:
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.assertGreater(self.triple("home", i, j)[0], self.triple("neutral", i, j)[0])


class RoundTripletTests(unittest.TestCase):
    def test_sum_survives_rounding_that_would_lose_a_unit(self) -> None:
        for probabilities in ((0.33335, 0.33335, 0.3333), (0.00005, 0.00005, 0.9999), (1 / 3, 1 / 3, 1 / 3)):
            self.assertEqual(sum(soccer_data.round_triplet(probabilities)), soccer_data.SCALE)


def _parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if not line.startswith("|"):
            break
        cells = [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("- "):
            continue
        rows.append(cells)
    return rows[1:]


def _first_table_after(text: str, marker: str) -> list[list[str]]:
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(marker))
    first = next(index for index in range(start, len(lines)) if lines[index].startswith("|"))
    return _parse_table(lines[first:])


class ValidationSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = soccer_data.load_validation_snapshot()
        cls.docs = (REPO_ROOT / "docs" / "validation_soccer.md").read_text(encoding="utf-8")
        cls.readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        cls.stages = {stage["years"]: stage for stage in cls.snapshot["stages"]}

    def assert_rows_equal(self, rows: list[dict], table: list[list[str]]) -> None:
        self.assertEqual(len(rows), len(table))
        for row, cells in zip(rows, table, strict=True):
            expected = [float(cell) for cell in cells[1:5]]
            actual = [row["brier"], row["rps"], row["log_loss"], row["accuracy"]]
            self.assertEqual(actual, expected, row["model"])

    def test_stage_1_equals_the_report_table(self) -> None:
        stage = self.stages["2018 to 2025"]
        table = _first_table_after(self.docs, "## Pooled results")
        self.assert_rows_equal(stage["rows"], table)
        self.assertEqual([row["model"] for row in stage["rows"]], [cells[0] for cells in table])
        self.assertIn(f"{stage['n_matches']:,}", self.docs)

    def test_stage_2_equals_the_report_table(self) -> None:
        stage = self.stages["2023 to 2025"]
        table = _first_table_after(self.docs, "## Held-out final test")
        self.assert_rows_equal(stage["rows"], table)
        for row, cells in zip(stage["rows"], table, strict=True):
            self.assertTrue(row["model"].startswith(re.sub(r"\s*\(.*\)$", "", cells[0])), (row["model"], cells[0]))
        self.assertIn(f"{stage['n_matches']:,}", self.docs)

    def test_readme_tables_agree_with_the_snapshot(self) -> None:
        for marker, years in (("**Stage 1, diagnosis.**", "2018 to 2025"), ("**Stage 2, selection", "2023 to 2025")):
            with self.subTest(stage=years):
                stage = self.stages[years]
                table = _first_table_after(self.readme, marker)
                self.assert_rows_equal(stage["rows"], table)
                self.assertEqual([row["model"] for row in stage["rows"]], [cells[0] for cells in table])
                self.assertIn(f"{stage['n_matches']:,}", self.readme)

    def test_notes_are_sentences_of_the_readme(self) -> None:
        for stage in self.snapshot["stages"]:
            for note in stage["notes"]:
                self.assertIn(note, self.readme)

    def test_source_commands_are_the_ones_the_report_gives(self) -> None:
        for command in self.snapshot["source_command"]:
            self.assertIn(command, self.docs)

    def test_loader_rejects_a_snapshot_missing_keys(self) -> None:
        broken = {"source_command": [], "report": "x", "stages": [{"name": "s", "rows": [{"model": "m"}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(ValueError):
                soccer_data.load_validation_snapshot(path)


if __name__ == "__main__":
    unittest.main()
