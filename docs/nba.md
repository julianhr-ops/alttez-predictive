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
