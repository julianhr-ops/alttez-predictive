"""Walk-forward validation of the NBA models against baselines, with paired uncertainty.

For each test season S the models are fitted on games from seasons strictly before S
(hyperparameters, calibrators, market sd, blend) and scored on season S. Nothing is ever
tuned on the test season.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from alttez.evaluation.paired import (
    PairedDiff,
    brier_binary_per_game,
    expected_calibration_error,
    log_loss_binary_per_game,
    paired_bootstrap_diff,
    reliability_table,
)
from alttez.nba import data
from alttez.nba.model import FittedModels, fit_models, predict_games

FIRST_TEST_SEASON = 2021
MODELS = ("p_elo", "p_elo_platt", "p_blend")
BASELINES = ("p_const", "p_vanilla", "p_market")
REPORTED = ("p_const", "p_vanilla", "p_market", "p_elo", "p_elo_platt", "p_elo_isotonic", "p_blend")
LABELS = {
    "p_const": "constant home rate",
    "p_vanilla": "vanilla Elo",
    "p_market": "market close",
    "p_elo": "MOV Elo",
    "p_elo_platt": "MOV Elo + Platt",
    "p_elo_isotonic": "MOV Elo + isotonic",
    "p_blend": "Elo + market blend",
}
METRICS = ("brier", "log_loss")
LOSSES = {"brier": brier_binary_per_game, "log_loss": log_loss_binary_per_game}


@dataclass(frozen=True, slots=True)
class Fold:
    season: int
    models: FittedModels
    predictions: pd.DataFrame


def walk_forward(
    games: pd.DataFrame,
    test_seasons: Sequence[int],
    *,
    elo_candidates=None,
    vanilla_candidates=None,
    tune_seasons: int = 5,
) -> list[Fold]:
    """``games`` is completed games sorted by date, optionally with ``home_margin_line``."""
    folds = []
    for season in test_seasons:
        history = games[games["season"] < season].reset_index(drop=True)
        target = games[games["season"] == season].reset_index(drop=True)
        if history.empty or target.empty:
            raise ValueError(f"season {season} needs earlier history and its own games")
        models = fit_models(
            history, elo_candidates=elo_candidates, vanilla_candidates=vanilla_candidates, tune_seasons=tune_seasons
        )
        if max(models.train_seasons) >= season:
            raise AssertionError(f"fold {season} was fitted on test-season rows")
        predictions = predict_games(models, history, target)
        outcome = pd.DataFrame(
            {
                "game_date": target["game_date"],
                "season": target["season"],
                "season_type": target["season_type"],
                "margin": target["home_score"] - target["away_score"],
            }
        )
        outcome["y"] = (outcome["margin"] > 0).astype(int)
        folds.append(Fold(season, models, pd.concat([predictions, outcome], axis=1)))
    return folds


def point_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "n": len(y),
        "brier": float(brier_binary_per_game(p, y).mean()),
        "log_loss": float(log_loss_binary_per_game(p, y).mean()),
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
        "ece": expected_calibration_error(p, y),
    }


def metrics_by_model(frame: pd.DataFrame, models: Sequence[str] = REPORTED) -> dict[str, dict[str, float]]:
    y = frame["y"].to_numpy()
    return {m: point_metrics(frame[m].to_numpy(float), y) for m in models if frame[m].notna().all() and len(frame)}


def margin_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for name, column in (("elo", "elo_margin"), ("market", "market_margin")):
        error = (frame[column] - frame["margin"]).to_numpy(float)
        if len(error) and np.isfinite(error).all():
            out[name] = {"n": len(error), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error**2).mean()))}
    return out


def verdict(diff: PairedDiff) -> str:
    if diff.a_better:
        return "model better"
    if diff.b_better:
        return "baseline better"
    return "no significant difference"


def compare(frame: pd.DataFrame, model: str, baseline: str, metric: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    both = frame[frame[model].notna() & frame[baseline].notna()]
    y = both["y"].to_numpy()
    loss = LOSSES[metric]
    diff = paired_bootstrap_diff(
        loss(both[model].to_numpy(float), y),
        loss(both[baseline].to_numpy(float), y),
        n_boot=n_boot,
        seed=seed,
        block_ids=both["game_date"].to_numpy(),
    )
    return {"model": model, "baseline": baseline, "metric": metric, **asdict(diff), "verdict": verdict(diff)}


def comparisons(frame: pd.DataFrame, *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    return [
        compare(frame, model, baseline, metric, n_boot=n_boot, seed=seed)
        for model in MODELS
        for baseline in BASELINES
        for metric in METRICS
        if model != baseline and frame[model].notna().any() and frame[baseline].notna().any()
    ]


def scope_report(frame: pd.DataFrame, *, n_boot: int, seed: int) -> dict[str, Any]:
    lined = frame[frame["p_market"].notna()]
    return {
        "all_games": {"n": len(frame), "metrics": metrics_by_model(frame, ("p_const", "p_vanilla", "p_elo", "p_elo_platt", "p_elo_isotonic"))},
        "lined_games": {
            "n": len(lined),
            "metrics": metrics_by_model(lined),
            "margin": margin_metrics(lined),
            "comparisons": comparisons(lined, n_boot=n_boot, seed=seed),
        },
    }


def fold_summary(fold: Fold) -> dict[str, Any]:
    m = fold.models
    return {
        "season": fold.season,
        "train_seasons": [m.train_seasons[0], m.train_seasons[-1]],
        "tune_seasons": list(m.tune_seasons),
        "elo_params": asdict(m.elo_params),
        "vanilla_params": asdict(m.vanilla_params),
        "margin_points_per_elo": m.margin_slope,
        "home_rate": m.home_rate,
        "platt": asdict(m.platt),
        "market_sd": m.market.sd if m.market else None,
        "blend": asdict(m.blend) if m.blend else None,
        "n_lined_train": m.n_lined_train,
    }


def run_validation(
    schedule: data.NbaSchedule,
    lines: pd.DataFrame,
    *,
    first_test_season: int = FIRST_TEST_SEASON,
    n_boot: int = 2000,
    seed: int = 0,
    elo_candidates=None,
    vanilla_candidates=None,
) -> dict[str, Any]:
    games = data.attach_lines(schedule.games, lines)
    test_seasons = [s for s in schedule.complete_seasons if s >= first_test_season]
    if not test_seasons:
        raise ValueError("no completed test season available")
    folds = walk_forward(games, test_seasons, elo_candidates=elo_candidates, vanilla_candidates=vanilla_candidates)
    pooled = pd.concat([f.predictions for f in folds], ignore_index=True)
    by_type = {
        data.SeasonType(t).name.lower(): metrics_by_model(group[group["p_market"].notna()])
        for t, group in pooled.groupby("season_type")
    }
    reliability = {
        m: reliability_table(pooled[m].to_numpy(float), pooled["y"].to_numpy()).round(4).to_dict("records")
        for m in ("p_elo", "p_elo_platt")
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "attribution": data.ATTRIBUTION,
        "labels": LABELS,
        "test_seasons": test_seasons,
        "in_progress_season": schedule.in_progress_season,
        "folds": [fold_summary(f) for f in folds],
        "overall": scope_report(pooled, n_boot=n_boot, seed=seed),
        "by_season_type_lined": by_type,
        "by_season": {str(f.season): scope_report(f.predictions, n_boot=n_boot, seed=seed) for f in folds},
        "reliability_overall": reliability,
    }


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _metrics_table(metrics: dict[str, dict[str, float]]) -> list[str]:
    lines = [_row(["model", "n", "Brier", "log loss", "accuracy", "ECE"]), _row(["---"] * 6)]
    for name, m in metrics.items():
        lines.append(_row([LABELS[name], str(m["n"]), f"{m['brier']:.4f}", f"{m['log_loss']:.4f}", f"{m['accuracy']:.3f}", f"{m['ece']:.3f}"]))
    return lines


def _comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [_row(["model", "vs baseline", "metric", "mean diff (model - baseline)", "95% CI", "verdict"]), _row(["---"] * 6)]
    for r in rows:
        lines.append(
            _row([LABELS[r["model"]], LABELS[r["baseline"]], r["metric"], f"{r['mean_diff']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", r["verdict"]])
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    out = [
        f"# NBA walk-forward validation, test seasons {report['test_seasons'][0]} to {report['test_seasons'][-1]} (season = ending year)",
        "",
        report["attribution"],
        "",
        f"## Overall, all test games (n={overall['all_games']['n']})",
        "",
        *_metrics_table(overall["all_games"]["metrics"]),
        "",
        f"## Overall, games with a closing line (n={overall['lined_games']['n']})",
        "",
        *_metrics_table(overall["lined_games"]["metrics"]),
        "",
        "Margin error on lined games (points):",
        "",
        *[f"- {k}: MAE {v['mae']:.2f}, RMSE {v['rmse']:.2f}" for k, v in overall["lined_games"]["margin"].items()],
        "",
        "## Paired bootstrap, lined games (negative mean diff = model better; blocks = game dates)",
        "",
        *_comparison_table(overall["lined_games"]["comparisons"]),
        "",
        "## By season, lined games (Brier / log loss)",
        "",
    ]
    names = [m for m in REPORTED if m in overall["lined_games"]["metrics"]]
    out += [_row(["season", "n", *[LABELS[m] for m in names]]), _row(["---"] * (2 + len(names)))]
    for season, scope in report["by_season"].items():
        lined = scope["lined_games"]
        out.append(_row([season, str(lined["n"]), *[f"{lined['metrics'][m]['brier']:.4f} / {lined['metrics'][m]['log_loss']:.4f}" for m in names]]))
    out += ["", "## Per-season verdicts vs market close (log loss)", "", _row(["season", "model", "mean diff", "95% CI", "verdict"]), _row(["---"] * 5)]
    for season, scope in report["by_season"].items():
        for r in scope["lined_games"]["comparisons"]:
            if r["baseline"] == "p_market" and r["metric"] == "log_loss":
                out.append(_row([season, LABELS[r["model"]], f"{r['mean_diff']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", r["verdict"]]))
    out += ["", "## Selected hyperparameters per fold", "", _row(["season", "K", "home adv", "regression", "b2b", "playoff K x", "market sd", "blend (elo, market)", "lined train n"]), _row(["---"] * 9)]
    for f in report["folds"]:
        p, b = f["elo_params"], f["blend"]
        out.append(
            _row([
                str(f["season"]), f"{p['k']:g}", f"{p['home_advantage']:g}", f"{p['season_regression']:g}", f"{p['b2b_penalty']:g}",
                f"{p['playoff_k_multiplier']:g}", "n/a" if f["market_sd"] is None else f"{f['market_sd']:.2f}",
                "n/a" if b is None else f"{b['elo_coef']:.2f}, {b['market_coef']:.2f}", str(f["n_lined_train"]),
            ])
        )
    return "\n".join(out) + "\n"
