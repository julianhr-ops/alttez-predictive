import json
import unittest

import numpy as np
import pandas as pd

from alttez.nba import data
from alttez.nba.evaluate import (
    compare,
    point_metrics,
    render_markdown,
    run_validation,
    walk_forward,
)
from tests.nba_synthetic import SMALL_GRID, SMALL_VANILLA, synthetic_games


def _schedule_and_lines(games: pd.DataFrame) -> tuple[data.NbaSchedule, pd.DataFrame]:
    seasons = tuple(sorted(int(s) for s in games["season"].unique()))
    schedule = data.NbaSchedule(games=games[list(data.GAME_COLUMNS)], complete_seasons=seasons, in_progress_season=None)
    lined = games[games["home_margin_line"].notna()]
    lines = pd.DataFrame(
        {
            "game_id": lined["game_id"], "home_margin_line": lined["home_margin_line"],
            "total_line": 220.0, "n_books": 5,
        }
    )
    return schedule, lines.reset_index(drop=True)


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.games = synthetic_games()

    def test_every_fold_trains_strictly_before_its_test_season(self):
        folds = walk_forward(self.games, [2015, 2016, 2017], elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)
        for fold in folds:
            self.assertLess(max(fold.models.train_seasons), fold.season)
            self.assertEqual(set(fold.predictions["season"]), {fold.season})
            self.assertEqual(len(fold.predictions), int((self.games["season"] == fold.season).sum()))

    def test_corrupting_the_test_season_and_later_cannot_change_the_fit(self):
        corrupted = self.games.copy()
        later = corrupted["season"] >= 2016
        corrupted.loc[later, ["home_score", "away_score"]] = corrupted.loc[later, ["away_score", "home_score"]].to_numpy()
        corrupted.loc[later, "home_margin_line"] = -corrupted.loc[later, "home_margin_line"]
        clean = walk_forward(self.games, [2016], elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)[0]
        dirty = walk_forward(corrupted, [2016], elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)[0]
        for field in ("train_seasons", "tune_seasons", "elo_params", "vanilla_params", "margin_slope", "home_rate", "platt", "market", "blend", "n_lined_train"):
            self.assertEqual(getattr(clean.models, field), getattr(dirty.models, field), field)
        np.testing.assert_array_equal(clean.models.isotonic.values, dirty.models.isotonic.values)

    def test_a_season_without_history_is_rejected(self):
        with self.assertRaises(ValueError):
            walk_forward(self.games, [2010], elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA)


class MetricsAndComparisonTests(unittest.TestCase):
    def test_point_metrics_on_known_values(self):
        metrics = point_metrics(np.array([0.9, 0.2, 0.5]), np.array([1, 0, 1]))
        self.assertAlmostEqual(metrics["brier"], (0.01 + 0.04 + 0.25) / 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["n"], 3)

    def test_compare_flags_a_clearly_better_model_and_is_deterministic(self):
        rng = np.random.default_rng(0)
        n = 600
        truth = rng.uniform(0.2, 0.8, n)
        y = (rng.uniform(size=n) < truth).astype(int)
        frame = pd.DataFrame(
            {"y": y, "good": truth, "flat": np.full(n, 0.5), "game_date": np.repeat(np.arange(n // 3), 3)}
        )
        first = compare(frame, "good", "flat", "log_loss", n_boot=300, seed=1)
        second = compare(frame, "good", "flat", "log_loss", n_boot=300, seed=1)
        self.assertEqual(first, second)
        self.assertEqual(first["verdict"], "model better")
        self.assertLess(first["ci_high"], 0)
        self.assertEqual(compare(frame, "flat", "good", "log_loss", n_boot=300, seed=1)["verdict"], "baseline better")


class RunValidationTests(unittest.TestCase):
    def setUp(self):
        games = synthetic_games()
        self.schedule, self.lines = _schedule_and_lines(games)
        self.report = run_validation(
            self.schedule, self.lines, first_test_season=2015, n_boot=100, seed=3,
            elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA,
        )

    def test_report_is_json_serialisable_and_complete(self):
        encoded = json.dumps(self.report)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["test_seasons"], [2015, 2016, 2017])
        self.assertEqual([f["season"] for f in decoded["folds"]], [2015, 2016, 2017])
        for fold in decoded["folds"]:
            self.assertLess(fold["train_seasons"][1], fold["season"])
        self.assertEqual(set(decoded["by_season"]), {"2015", "2016", "2017"})
        lined = decoded["overall"]["lined_games"]
        self.assertEqual(set(lined["metrics"]), {"p_const", "p_vanilla", "p_market", "p_elo", "p_elo_platt", "p_elo_isotonic", "p_blend"})
        self.assertEqual(len(lined["comparisons"]), 18)

    def test_overall_sample_is_the_union_of_the_test_seasons(self):
        per_season = sum(s["all_games"]["n"] for s in self.report["by_season"].values())
        self.assertEqual(self.report["overall"]["all_games"]["n"], per_season)

    def test_same_seed_gives_the_same_report(self):
        again = run_validation(
            self.schedule, self.lines, first_test_season=2015, n_boot=100, seed=3,
            elo_candidates=SMALL_GRID, vanilla_candidates=SMALL_VANILLA,
        )
        self.assertEqual(self.report["overall"], again["overall"])

    def test_markdown_lists_models_and_verdicts(self):
        text = render_markdown(self.report)
        for label in ("market close", "vanilla Elo", "Elo + market blend", "95% CI"):
            self.assertIn(label, text)

    def test_no_completed_test_season_is_an_error(self):
        with self.assertRaises(ValueError):
            run_validation(self.schedule, self.lines, first_test_season=3000)


if __name__ == "__main__":
    unittest.main()
