# Soccer 1X2 validation

Walk-forward backtest of the production soccer stack against baselines, with paired uncertainty.
Code: `alttez/evaluation/soccer_backtest.py`, run with `scripts/validate_soccer.py`.

## Verdict

- **The production Poisson champion does not beat the Elo-only baseline.** It is worse on Brier, RPS and log loss, pooled over 2018 to 2025 and in 7 of the 8 individual years, each with a 95% CI that excludes 0. In 2020 (355 matches) the difference is inconclusive.
- It does beat the uniform and base-rate baselines, with CIs that exclude 0 on all three metrics.
- **Temperature calibration changes nothing measurable.** Calibrated minus raw is 0.0000 Brier pooled, CI [-0.0007, +0.0007]. It helps in 2019 and 2021 (CI excludes 0), is inconclusive in the other six years, and never significantly hurts. The calibrated model is still significantly worse than Elo-only.
- The dominant defect of the Poisson model is a class bias that a single temperature cannot fix. It predicts draws too often (mean 0.281 against an observed 0.233) and home and away wins too rarely.

## Setup (what was actually run)

| item | value |
| --- | --- |
| data | `martj42/international_results` `results.csv`, CC0, 49,940 scored matches, 1872-11-30 to 2026-10-06, parsed by `seed_historical.load_historical_dataframe` |
| test years | 2018 to 2025 (2026 is incomplete). 7,829 matches, 7,611 scored |
| walk-forward | every test year is predicted from data before 1 January of that year |
| Elo | `alttez.features.elo` rules, base K 30, `tournament_k_factor` (qualifiers K 40), home advantage 100 unless neutral. Ratings use matches on strictly earlier dates only |
| goal model | `PoissonEngine(**DEFAULT_POISSON_KWARGS)` from the router, refit from matches strictly before each match date, `build_probability_matrix` with max goals 7 |
| rho | Dixon-Coles MLE by the production procedure (`scripts/fit_dixon_coles_rho.py`) on all matches before the test year. Fitted -0.0350 (2018) to -0.0301 (2024) |
| temperature | `fit_temperature` on the RAW 1X2 of the 4 preceding years (each predicted walk-forward), applied with `ProbabilityCalibration.calibrate_matrix` (class-ratio transform). Fitted 0.860 to 0.943 |
| baselines | uniform; base rate and Elo-only draw parameter (`alttez.evaluation.baselines`) from the 4 preceding years |
| bootstrap | `paired_bootstrap_diff`, 2,000 resamples, seed 0, blocks = match dates. Negative mean difference means the first model is better |

All numbers below come from one run: `python scripts/validate_soccer.py --refit-days 1 --workers 10` (33 minutes). The raw output is `artifacts/soccer/validation.json` (git-ignored; rerun to regenerate).

## Pooled results, 7,611 identical matches

| model | Brier | RPS | log loss | accuracy | ECE H | ECE D | ECE A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0.6667 | 0.2390 | 1.0986 | 0.475 | 0.142 | 0.101 | 0.041 |
| base rate | 0.6350 | 0.2282 | 1.0527 | 0.475 | 0.002 | 0.001 | 0.001 |
| Elo only | 0.5148 | 0.1707 | 0.8741 | 0.597 | 0.014 | 0.014 | 0.010 |
| Poisson raw | 0.5811 | 0.1998 | 0.9760 | 0.526 | 0.023 | 0.048 | 0.034 |
| Poisson + temperature | 0.5811 | 0.1996 | 0.9761 | 0.526 | 0.018 | 0.042 | 0.031 |

Accuracy of the uniform and base-rate rows is the home-win share, because both always pick H.

Mean predicted against observed frequency, pooled:

| model | H | D | A |
| --- | --- | --- | --- |
| observed | 0.475 | 0.233 | 0.292 |
| Elo only | 0.485 | 0.228 | 0.287 |
| Poisson raw | 0.452 | 0.281 | 0.267 |
| Poisson + temperature | 0.463 | 0.275 | 0.262 |

The Poisson draw reliability curve is flat where it should rise. Matches given a draw probability of 0.44 drew 29% of the time, and those given 0.53 drew 23%.

## Paired bootstrap, pooled (model minus baseline, lower is better)

| model | vs | Brier | RPS | log loss |
| --- | --- | --- | --- | --- |
| Poisson raw | uniform | -0.0856 [-0.0936, -0.0784] | -0.0392 [-0.0426, -0.0360] | -0.1226 [-0.1340, -0.1122] |
| Poisson raw | base rate | -0.0539 [-0.0605, -0.0476] | -0.0284 [-0.0313, -0.0256] | -0.0767 [-0.0864, -0.0673] |
| Poisson raw | Elo only | +0.0663 [+0.0585, +0.0735] | +0.0291 [+0.0258, +0.0322] | +0.1019 [+0.0906, +0.1129] |
| Poisson + temperature | uniform | -0.0856 [-0.0942, -0.0778] | -0.0394 [-0.0430, -0.0359] | -0.1226 [-0.1350, -0.1111] |
| Poisson + temperature | base rate | -0.0539 [-0.0609, -0.0472] | -0.0285 [-0.0317, -0.0255] | -0.0767 [-0.0872, -0.0666] |
| Poisson + temperature | Elo only | +0.0663 [+0.0585, +0.0736] | +0.0289 [+0.0256, +0.0321] | +0.1019 [+0.0904, +0.1129] |
| Poisson + temperature | Poisson raw | +0.0000 [-0.0007, +0.0007] | -0.0002 [-0.0005, +0.0001] | +0.0000 [-0.0010, +0.0011] |
| Elo only (context) | base rate | -0.1202 [-0.1289, -0.1113] | -0.0575 [-0.0617, -0.0533] | -0.1786 [-0.1922, -0.1651] |

## By year (Brier)

| year | n | Elo only | Poisson raw | Poisson + temp | raw vs Elo (CI) | temp vs raw (CI) |
| --- | --- | --- | --- | --- | --- | --- |
| 2018 | 882 | 0.5550 | 0.6060 | 0.6063 | +0.0510 [+0.0326, +0.0682] | +0.0003 [-0.0012, +0.0018] |
| 2019 | 1118 | 0.4873 | 0.5723 | 0.5709 | +0.0850 [+0.0649, +0.1059] | -0.0014 [-0.0027, -0.0001] |
| 2020 | 355 | 0.5845 | 0.6042 | 0.6048 | +0.0197 [-0.0067, +0.0458] | +0.0006 [-0.0026, +0.0036] |
| 2021 | 1113 | 0.4711 | 0.5396 | 0.5374 | +0.0685 [+0.0517, +0.0848] | -0.0021 [-0.0033, -0.0009] |
| 2022 | 952 | 0.5374 | 0.5935 | 0.5951 | +0.0561 [+0.0338, +0.0801] | +0.0016 [-0.0008, +0.0039] |
| 2023 | 986 | 0.5178 | 0.5811 | 0.5817 | +0.0633 [+0.0437, +0.0817] | +0.0006 [-0.0020, +0.0031] |
| 2024 | 1222 | 0.5347 | 0.6015 | 0.6031 | +0.0667 [+0.0472, +0.0852] | +0.0016 [-0.0003, +0.0035] |
| 2025 | 983 | 0.4849 | 0.5702 | 0.5696 | +0.0852 [+0.0672, +0.1037] | -0.0005 [-0.0016, +0.0007] |

RPS and log loss per year, and every per-year comparison, are in `artifacts/soccer/validation.json` and in the printed report.

## Why the Poisson model loses (inferred, not tested)

- `PoissonEngine` and the router's Poisson predictor take no Elo input. `git grep -i elo` finds no match in `features/poisson.py` or in the Poisson path of `inference/router.py`. So there is no "Elo + Poisson" model in production to validate; the Poisson champion is scored as it is.
- Its strengths are average goal rates over 4 years with an 8-match prior, with no opponent adjustment. Elo carries rating history across all opponents and years.
- The prior shrinks every team toward the mean, which makes matches look closer than they are. That fits the draw over-prediction and the under-prediction of both win classes.
- A temperature rescales all classes together. Fitted values below 1 (0.86 to 0.94) sharpen the output slightly, which also contradicts the "overconfident" premise in the `calibration.py` header for this model. It cannot remove a draw-specific bias.

## Exclusions and differences from production

- 218 of 7,829 test-year matches are not scored because a team had fewer than 5 matches in the last 4 years. The router abstains on these. The rule is a reimplementation of `_is_prediction_supported` with its default threshold. All models are scored on the same 7,611 matches. Elo-only could have predicted the 218.
- No match in the CSV lacks a score (0 rows dropped by the loader). Fixtures with empty scores would be dropped and counted.
- Neutral venues: Elo drops the home bonus, and `predict_xg(is_neutral=True)` uses overall attack and defence. The home label at a neutral venue is nominal. 2,494 of 7,829 test-year matches are neutral.
- All tournaments are included, friendlies as well. Scores are as recorded by the source, which counts extra time and excludes shoot-outs.
- Team names are used as given in the CSV. The router applies `canonical_team_name` to live fixtures.
- Production fits `rho` once from the whole history into `artifacts/dixon_coles/active.json`. The backtest refits it per test year on earlier data. The fitted -0.03 is far from the fallback `DEFAULT_DIXON_COLES_RHO = -0.13`. On the pre-2018 training data the Dixon-Coles log-likelihood at -0.13 is minus infinity (checked with a vectorised copy of the likelihood); a likely cause is rate-based lambdas that reach about 28 for rare teams.
- Calibration is off by default for the single-predictor path and gated by `ALTTEZ_CALIBRATE_SINGLE_PREDICTOR`. Row (e) is what the calibration step would do if enabled, fitted on a clean walk-forward series rather than on stored predictions.
- The Poisson engine is refit at every match date. A 30-day cadence on 2025 only gave the same picture in a smoke run (Brier 0.5712 raw, 0.5708 calibrated, Elo 0.4835 on 976 matches).
- Not evaluated: XGBoost, hierarchical, bivariate Poisson, the committee consensus, the tournament simulation.
- Elo constants (K 30, tournament factors, home advantage 100) were taken as they are in the repo. Whether they were tuned on these years is UNKNOWN. They are not tuned here.
- No multiple-comparison correction across the 8 comparisons, 3 metrics and 8 years. The pooled conclusions have CIs far from 0, so they do not depend on one.

## What this implies

- Anything that replaces or ranks against the Poisson champion should be compared with Elo-only on identical matches, with this paired CI, not only against uniform.
- Single-temperature calibration is not worth enabling for the Poisson champion on this evidence. A class-specific correction, such as the existing isotonic option, targets the draw bias but is untested here.
- A Poisson model whose lambdas use Elo, or an Elo-based 1X2 for the unsupported matches, are the obvious candidates to test next with this harness.
