# NBA game predictions (free data only)

Package `alttez/nba/`: `data.py` (loaders, ESPN parser), `model.py` (Elo, market, blend, calibration),
`evaluate.py` (walk-forward validation). Run `python scripts/nba_validate.py`; it writes
`artifacts/nba/validation.json` (gitignored) and prints a markdown report. Network is used only by the
script and by `fetch_espn_scoreboard`; tests are offline.

## Data

- Schedule and closing lines: [sportsdataverse/hoopR-nba-data](https://github.com/sportsdataverse/hoopR-nba-data), CC BY 4.0, attribution required (`data.ATTRIBUTION`). Cached under `artifacts/nba/`, refreshed after 12 h, atomic write, explicit `NbaDataError` on any failure.
- Upcoming games and the DraftKings line: ESPN public scoreboard JSON, one date per call. Odds exist only for games a day or so away and are dropped once a game finishes.
- Join key: ESPN game id. Teams: ESPN franchise id 1..30, stable across relocations (3 Hornets/Pelicans, 17 Nets, 25 SuperSonics/Thunder, 27 Bullets/Wizards, 29 Grizzlies). Ids outside 1..30 (all-star and exhibition teams) are dropped, as are duplicate ids and the 0-0 rows of the cancelled 1998-99 lockout games.
- `season` is the ending year (2026 = 2025-26). `season_type`: 2 regular, 3 playoffs, 5 play-in. `game_date` is the US date of tip-off (UTC minus 10 h) and drives rest days.
- Spread sign, verified on 7,936 closing lines joined to final scores (corr +0.4725 with the home margin): `home_margin_line > 0` means the home team is favoured (hoopR `home_team_spread`, the negation of `home_point`). ESPN's `spread` is the sportsbook home line, so the parser negates it and cross-checks the favourite flags.

## Models

- MOV Elo: home advantage, back-to-back penalty, margin-of-victory multiplier, playoff K multiplier, regression to the mean between seasons. Each game is scored from ratings before its date.
- Market: `Phi(home_margin_line / sd)` with `sd` fitted by maximum likelihood on training seasons.
- Blend: logistic regression on the Elo logit and the market logit, only where a line exists.
- Platt and isotonic calibration of the Elo-only output.
- All hyperparameters, calibrators, `sd` and blend coefficients are fitted per test season on earlier seasons only (grid scored on the last five training seasons).

## Result (walk-forward, test seasons 2020-21 to 2025-26, 7,776 games, 7,764 with a closing line)

| model | Brier | log loss | accuracy | ECE |
| --- | --- | --- | --- | --- |
| constant home rate | 0.2473 | 0.6878 | 0.554 | 0.011 |
| vanilla Elo | 0.2199 | 0.6295 | 0.646 | 0.013 |
| MOV Elo | 0.2174 | 0.6240 | 0.651 | 0.012 |
| MOV Elo + Platt | 0.2173 | 0.6238 | 0.652 | 0.012 |
| market close | 0.2067 | 0.5995 | 0.678 | 0.012 |
| Elo + market blend | 0.2070 | 0.6004 | 0.679 | 0.010 |

Paired bootstrap by game date, 95% CI, pooled over the six test seasons: MOV Elo (with or without Platt) beats the constant rate (also in every season individually) and vanilla Elo (pooled; season by season the log-loss CI excludes 0 in 3 of 6 seasons for Elo and 4 of 6 for Elo + Platt, and the gain is about 0.005). It does not beat the market close in any season (pooled log loss +0.025 worse, CI excludes 0). The blend gives the Elo logit a coefficient of about zero and the market logit about 1.03, so it reproduces the market: Brier difference vs market +0.0004 [-0.0000, +0.0008], log loss +0.0009 [+0.0000, +0.0017]. Elo adds no measurable information to the closing line; its value is covering games that have no line yet.

## Operation (daily)

`python scripts/nba_daily.py [--date YYYY-MM-DD] [--database-url URL] [--dry-run] [--deliver]` runs after the soccer stages in `scripts/run_daily_process.ps1` (stage `nba predictions`, also on the soccer `no_schedule` path). The stage is non-fatal: a failure prints `PROCESS: nba predictions degraded (non-fatal)` and sends a warning alert. `--deliver` is passed only when `ALTTEZ_DAILY_BRIEF_DELIVER` is truthy, like the soccer brief.

Order inside one run: refresh hoopR (a failed refresh falls back to the cache and says so in the digest), grade finished games from the ESPN scoreboard (last 3 days plus today, idempotent), freshness gate, predict the games on the ESPN scoreboard for `--date` and the next day, store, print the digest. Grading does not depend on hoopR, so it still runs when the gate refuses.

- **Served model.** The blend serves when ESPN has a line for the game; Elo + Platt serves otherwise. `served_model` stores which one. Both `p_home_elo` (Platt-calibrated Elo) and `p_home_market` are kept, so the other number is never lost. This is the measured policy above: Elo does not beat the closing line, and the blend only reproduces it, so it adds nothing over the line but Elo covers games that have no line.
- **One write per game.** The table `public.nba_predictions` (migration `015_nba_predictions.sql`, key `(game_id, model_version)`) is insert-if-absent. A later run never rewrites a prediction, so a game first predicted without a line stays `elo_platt` even if a line appears later. Nothing is written for a game whose tip-off has passed. The only update is grading, which fills the score, `home_win` and `brier` once (`WHERE status = 'scheduled'`).
- **Model version.** `predict.MODEL_VERSION` (`nba-elo-v1`). Change it when the model or its fitting changes so old and new rows coexist.
- **Freshness gate.** Ratings replay every completed game, so missing results corrupt every prediction. The run refuses with `stale_data` (exit 3, nothing predicted or stored) when the newest completed game is more than 3 days old and the hoopR schedule lists game dates between it and today. The offseason and the All-Star break pass because the schedule has no such dates.
- **Fitting.** Each run refits on complete seasons (about 30 s) with the same code path as the validation, instead of caching a model. Each upcoming date is predicted from results strictly before it. A team that plays on two upcoming dates is treated as on a back-to-back on the second one.
- **Live scoreboard.** Graded rows give served vs market Brier and accuracy on the same games. The paired bootstrap interval (by game date) is shown only from 30 lined games.
- **Known limit.** The blend coefficients are fitted on odds-API closing lines in hoopR; live lines are ESPN's provider (usually DraftKings), recorded in `market_source`. Compare the live scoreboard with the validation table before trusting the blend more than the line.
- **Exit codes.** 0 ok, 1 digest delivery failed, 2 data or database error, 3 `stale_data`. With `--dry-run` and no `--database-url` the database is never opened; with no URL and no `--dry-run` the run prints only. The agent never applies migration 015; the engineer runs it.

## Attribution and disclaimer

- Schedule and closing-line data come from [sportsdataverse/hoopR-nba-data](https://github.com/sportsdataverse/hoopR-nba-data), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The repository does not redistribute the files. The scripts download them at run time and cache them under `artifacts/nba/`. ALTTEZ changes the data by dropping non-NBA franchise ids and duplicate rows, converting tip-off times to US dates, and normalising the sign of the spread (see Data above).
- Upcoming games and the DraftKings line come from ESPN's unofficial public scoreboard JSON, used for personal and educational purposes. The line belongs to a third party. ALTTEZ has no affiliation with the NBA, ESPN, DraftKings, or any bookmaker.
- The output is a statistical research result. It is not betting or financial advice.
