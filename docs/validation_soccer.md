# Soccer 1X2 validation

Walk-forward backtest of the production soccer stack against baselines, with paired uncertainty.
Code: `alttez/evaluation/soccer_backtest.py`, run with `scripts/validate_soccer.py`.

## Verdict

- **The production Poisson champion does not beat the Elo-only baseline.** It is worse on Brier, RPS and log loss, pooled over 2018 to 2025 and in 7 of the 8 individual years, each with a 95% CI that excludes 0. In 2020 (355 matches) the difference is inconclusive.
- It does beat the uniform and base-rate baselines, with CIs that exclude 0 on all three metrics.
- **Temperature calibration changes nothing measurable.** Calibrated minus raw is 0.0000 Brier pooled, CI [-0.0007, +0.0007]. It helps in 2019 and 2021 (CI excludes 0), is inconclusive in the other six years, and never significantly hurts. The calibrated model is still significantly worse than Elo-only.
- The dominant defect of the Poisson model is a class bias that a single temperature cannot fix. It predicts draws too often (mean 0.281 against an observed 0.233) and home and away wins too rarely.
- **An Elo ordinal 1X2 with a fitted draw model, imposed on the Poisson score matrix, beats the champion on years held out from the choice** (2023 to 2025, Brier -0.0728 [-0.0836, -0.0623]). See the last two sections. It is statistically equivalent to the plain Elo-only baseline.

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

## Candidate serving rules: selection on 2018 to 2022

The Poisson champion loses to Elo-only, so a second stage tests rules that could replace its 1X2 while keeping a coherent score matrix. Code: `alttez/evaluation/soccer_candidates.py` (rules and selection), `alttez/features/elo_1x2.py` (fitted draw models and blend). Everything is fitted on the 4 years before each test year only, with the same walk-forward protocol (`--refit-days 1`) and the same scored matches as above. Raw Poisson output of the first run is reused from a data-keyed cache, so every number in this document comes from one refit-1 Poisson run.

Candidates (8, all declared before any run):

- Elo-only with a fitted draw model: ordinal (cumulative logit with symmetric cut points) or Davidson, home advantage and draw width fitted on the training window, with the Elo logit scale fixed at 400 points (`ln10/400`) or fitted (`_scaled`).
- A log-linear blend of that model and the raw Poisson 1X2, weight in [0, 1] fitted by log loss on the supported training rows (Poisson output of the 4 preceding years, each predicted walk-forward).
- Each candidate's 1X2 is then imposed on the Poisson score matrix by class-ratio rescaling (`rescale_matrix_to_triplet`), so scorelines agree with the served 1X2.

Protocol, fixed before the held-out years were scored: only 2018 to 2022 are scored and read in this stage (`--report-years 2018 2022 --select`; the code refuses to select on a later year). The candidate with the lowest pooled Brier is the best. Among candidates whose paired Brier difference to the best has a CI that includes 0, the simplest wins (Elo-only before blend, fixed scale before fitted, ordinal before Davidson). 2023 to 2025 are not used for any choice.

Selection result (4,420 matches, 2018 to 2022; negative = better than the best):

| candidate | pooled Brier | Brier minus best (95% CI) |
| --- | --- | --- |
| Elo ordinal, fitted scale + Poisson blend | 0.5154 | best |
| Elo ordinal, fitted scale | 0.5156 | +0.0002 [-0.0003, +0.0007] |
| Elo ordinal + Poisson blend | 0.5158 | +0.0003 [+0.0000, +0.0007] |
| **Elo ordinal** | 0.5158 | +0.0004 [-0.0000, +0.0007] |
| Elo Davidson, fitted scale + Poisson blend | 0.5158 | +0.0004 [-0.0003, +0.0010] |
| Elo Davidson, fitted scale | 0.5159 | +0.0005 [-0.0004, +0.0013] |
| Elo Davidson | 0.5223 | +0.0069 [+0.0048, +0.0091] |
| Elo Davidson + Poisson blend | 0.5223 | +0.0069 [+0.0048, +0.0091] |

**Declared primary candidate: Elo ordinal** (`PRIMARY_CANDIDATE = "elo_ordinal"`). The blend and the fitted scale buy at most 0.0004 Brier, inside the noise, and the fitted blend weight on Elo is 0.92 to 1.00 in the 2018 to 2022 folds, so the Poisson 1X2 adds close to nothing. Selection-year context: Elo only (the existing baseline) 0.5153, Poisson raw 0.5779 on the same rows. Fitted Elo ordinal parameters in those folds: home advantage 92 to 100 Elo points, draw cut point 0.63 to 0.65.

## Held-out final test, 2023 to 2025 (3,191 matches)

Elo ordinal was declared in the commit before these years were scored (`git log` shows the order). Command: `python scripts/validate_soccer.py --refit-days 1 --raw-cache-dir artifacts/soccer/raw --report-years 2023 2025 --candidates elo_ordinal`.

| model | Brier | RPS | log loss | accuracy | ECE H | ECE D | ECE A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0.6667 | 0.2389 | 1.0986 | 0.467 | 0.134 | 0.100 | 0.034 |
| base rate | 0.6381 | 0.2295 | 1.0569 | 0.467 | 0.014 | 0.006 | 0.007 |
| Elo only (existing baseline) | 0.5142 | 0.1700 | 0.8714 | 0.599 | 0.025 | 0.024 | 0.021 |
| Poisson raw (current champion) | 0.5855 | 0.2014 | 0.9812 | 0.522 | 0.020 | 0.050 | 0.039 |
| Poisson + temperature | 0.5862 | 0.2015 | 0.9821 | 0.522 | 0.024 | 0.044 | 0.039 |
| **Elo ordinal draw (primary)** | **0.5127** | **0.1694** | **0.8700** | 0.602 | 0.021 | 0.020 | 0.019 |

Mean predicted against observed: observed 0.467 / 0.233 / 0.300 (H / D / A); Poisson raw 0.451 / 0.283 / 0.266; Elo ordinal 0.476 / 0.224 / 0.300. The draw over-prediction is gone (Elo ordinal slightly under-predicts draws, 0.224).

Paired bootstrap, pooled 2023 to 2025 (primary minus baseline, lower is better, blocks = match dates):

| vs | Brier | RPS | log loss |
| --- | --- | --- | --- |
| Poisson raw (champion) | -0.0728 [-0.0836, -0.0623] | -0.0320 [-0.0364, -0.0278] | -0.1113 [-0.1271, -0.0955] |
| Elo only | -0.0015 [-0.0025, -0.0004] | -0.0006 [-0.0011, -0.0001] | -0.0014 [-0.0035, +0.0006] |
| base rate | -0.1254 [-0.1377, -0.1134] | -0.0602 [-0.0659, -0.0545] | -0.1870 [-0.2056, -0.1688] |
| uniform | -0.1540 [-0.1691, -0.1396] | -0.0695 [-0.0760, -0.0633] | -0.2287 [-0.2509, -0.2072] |

By year, Brier of Elo ordinal against Poisson raw (CI): 2023 -0.0649 [-0.0829, -0.0458], 2024 -0.0685 [-0.0863, -0.0495], 2025 -0.0861 [-0.1036, -0.0685]. Against Elo only: 2023 -0.0016 [-0.0031, -0.0001], 2024 -0.0018 [-0.0038, +0.0003], 2025 -0.0009 [-0.0025, +0.0008].

Scoreline coherence (3,168 matches whose score fits the 8 x 8 matrix, matrix rescaled to the served 1X2 by class ratios): scoreline log loss 2.9940 against 3.1046 for the raw Poisson matrix, difference -0.1106 [-0.1263, -0.0947]. The served matrix reproduces the target 1X2 to 2.2e-16 on every match.

Fitted Elo ordinal parameters in the final folds: home advantage 88 to 95 Elo points, draw cut point 0.631 to 0.645.

### Reading it

- **The primary candidate beats the champion.** On held-out years the Brier CI against Poisson raw is entirely below 0 pooled and in each of 2023, 2024 and 2025. RPS and log loss are entirely below 0 pooled (per-year RPS and log loss are in the JSON report). This clears the bar set before the run.
- **Almost all of the gain comes from using Elo, not from the fitted draw model.** Against the existing Elo-only baseline the held-out edge is 0.0015 Brier. It is significant pooled on Brier and RPS and not on log loss, not significant in 2024 or 2025, and over all of 2018 to 2025 (which includes the selection years) it is -0.0003 [-0.0010, +0.0003]. Treat the two Elo rules as equivalent in accuracy. The fitted rule is preferred for serving because it is an explicit likelihood model whose draw rate is calibrated (0.230 against an observed 0.233 over 2018 to 2025), and because its probabilities come with a coherent score matrix.
- **The Poisson blend adds nothing.** On the selection years the blend and the fitted scale were within 0.0004 Brier of each other and of the plain rule.
- Pooled 2018 to 2025 numbers for the primary (Brier 0.5145, RPS 0.1705, log loss 0.8743 against Poisson raw 0.5811 / 0.1998 / 0.9760) mix selection and held-out years, so only the 2023 to 2025 table above is an untouched test.
- Cadence: all numbers use `--refit-days 1`; development used the same cached refit-1 Poisson output, so there is no cadence gap between selection and final.
- Not covered: matches the router's support gate rejects (a team with fewer than 5 matches in 4 years), and any team absent from the Elo history. Those keep the Poisson abstain/fallback path.

## Live serving

The held-out result cleared the bar, so the single predictor serves it (`alttez/inference/serving_policy.py`, wired in `PoissonMatchPredictor`). Default ON; `ALTTEZ_SERVING_POLICY=off` restores plain Poisson.

- Ratings come from `rate_matches` over stored matches dated strictly before the target date, the same walk as the backtest. The ordinal draw model is fitted on the matches from 1 January four calendar years before the target year up to the target date. A test pins that this equals the backtest fit at the same date.
- The served matrix is the Poisson score matrix rescaled by class ratios to the Elo 1X2, so `prob_home/draw/away`, the recommendations and the scoreline all come from it. Marginals match the target 1X2 to 1e-9 (tested).
- The payload carries `serving_policy` (`elo_ordinal_rescale`, `poisson`, or `none` for an active model that is not Poisson) and keeps `raw_prob_home/draw/away` as the raw Poisson 1X2, so the calibration pipeline still reads raw Poisson. The single-predictor temperature calibration is not applied on top of the Elo 1X2 (it was fitted on Poisson output).
- Loud fallback: a team without an Elo rating serves Poisson for that match with `used_fallback=True` and a `fallback_reason`; a failure building the ratings sets the run-level flags the same way.
- The committee serves the same Elo 1X2: it blends its member score matrices, then applies the champion's serving policy to the blend, so `prob_home/draw/away` equal the Elo target to 1e-9 and the scoreline stays coherent with them (tested). Before this, the blend of the Elo champion with the raw Poisson and bivariate members brought the Poisson draw inflation back: on the test fixture the Elo draw probability 0.263 was served as 0.332. The `poisson_baseline`, `bivariate` and XGBoost baseline members stay raw Poisson and are reported as diagnostics. The committee payload carries `serving_policy` and, per match, `used_fallback` and `fallback_reason` when any member fell back. No gain from blending the members' scorelines has been measured; the committee consensus is still not evaluated in the backtest.
- `challenger_cycle` promotes a challenger only if it also beats the Elo-only baseline on the same holdout under the same rule as the Poisson champion (paired Brier interval below 0, at least 200 identical matches, no material log-loss deterioration). The Elo-only baseline stands in for the served rule; the report carries it as `elo_comparison`. An active model that is not Poisson replaces the Elo policy and reports `serving_policy` as `none`.
- The support gate is unchanged: matches the Poisson gate abstains on still abstain.
