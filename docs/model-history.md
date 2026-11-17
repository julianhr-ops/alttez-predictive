# Model history: what we measured, what we changed

This page records how the soccer stack was diagnosed, which defects the measurements exposed, and what changed in the code as a result. It is a history, not a manual. The current numbers live in [validation_soccer.md](validation_soccer.md) and [nba.md](nba.md).

## 1. First look at the live predictions (2026-05-26 to 2026-06-16)

The first diagnosis used the point-in-time prediction store in Supabase (`prediction_runs`, `match_predictions`, `prediction_evaluations`). The headline metrics at the start looked terrible. Two measurement artifacts caused that.

- Only 20 predictions had a closed evaluation.
- 45% of the rows were duplicates (626 rows for 346 matches), because every run inserted a new row per match. Old forecasts were over-weighted.

Closing every evaluation that already had a result (20 to 58) and keeping the latest prediction per match changed the picture.

| Set | n | Brier 1X2 | Accuracy 1X2 | Goals MAE |
| --- | --- | --- | --- | --- |
| Initial, with duplicates | 20 | 0.93 | 15% | 2.15 |
| All evaluated | 58 | 0.769 | 36% | 2.14 |
| One per match, latest | 29 | 0.648 | 55% | 2.07 |

On the 29 deduplicated matches the model had a clear class bias. It predicted 27 home wins, 0 draws and 2 away wins, against 14, 6 and 9 observed. The mean home expected goals were 1.64 against 1.06 for the away team, a fixed home tilt that does not hold at neutral tournament venues.

The larger limit was the number of labelled matches. Tournament results were not being ingested, so only 29 matches could be scored. A multi-source results backfill (football-data.org for the World Cup, the ESPN scoreboard for friendlies) lifted the evaluated predictions from 20 to 287, and from 34 to 137 with a stored probability. The backfill now runs as a non-fatal stage of the daily process (`scripts/backfill_match_results.py`).

With 137 labelled matches, three post-hoc corrections were tried with leave-one-out validation: a three-weight calibration, temperature scaling and a draw boost. All three made the Brier score worse (0.566 before, 0.569, 0.589 and 0.583 after). The stored probabilities were already close to the base rates, so the deficit was resolution, not calibration. Overall Brier was 0.634 and accuracy 46.7%. World Cup matches scored 0.630, close to the 0.667 of a uniform guess.

That sample was small, had no baselines and no uncertainty estimates. The next step was to measure properly.

## 2. Audit and fixes

An audit of the validation and calibration code found defects that no metric on the stored predictions could have shown.

| Defect found | Change |
| --- | --- |
| Temperature was fitted on 1X2 triplets but applied cell by cell to the 64-cell score matrix. | `alttez/inference/calibration.py` applies the 1X2 temperature to the 1X2 aggregate and rescales the score matrix by class. |
| Stored probabilities were already calibrated, and the calibration was refit on them, so corrections compounded. | Migration `014_prediction_raw_probabilities.sql` stores the raw 1X2 next to the calibrated one. The fit uses the raw values. |
| Each run inserted a new row per match, so learning and calibration queries counted a match several times. | `select_latest_pre_kickoff` in `alttez/operations/prediction_memory.py` keeps one prediction per match and model version. Closing evaluations ignores rows created at or after kickoff. |
| The Elo K factor tested "FIFA World Cup" before "qualification", so qualifiers got the finals K. | `tournament_k_factor` in `alttez/features/elo.py` matches qualification first (K 40). |
| A registry failure silently served the Poisson fallback with `used_fallback=False`. | The router logs a warning and reports `used_fallback` and `fallback_reason`. |
| The challenger cycle compared against a default-parameter engine with a fixed rho, not the production configuration. | The champion is rebuilt from the production configuration. If that fails, the cycle reports `production_config_error` and promotes nothing. |
| No baselines, and promotion compared raw Brier scores with no uncertainty. | `alttez/evaluation/baselines.py` adds uniform, base-rate and Elo-only 1X2 baselines. `alttez/evaluation/paired.py` adds the paired bootstrap. A challenger is eligible only if the whole 95% interval of the Brier difference is below 0 on at least 200 identical matches. |

## 3. Measured results after the audit

**Soccer.** A walk-forward backtest of the production stack on 7,611 international matches (2018 to 2025) found that the Poisson champion loses to a plain Elo baseline on Brier, RPS and log loss, pooled and in 7 of 8 years. It beats the uniform and base-rate baselines. Temperature calibration changes nothing measurable. The Poisson model over-predicts draws (mean 0.281 against 0.233 observed). Tables, setup and caveats are in [validation_soccer.md](validation_soccer.md).

**NBA.** The MOV Elo model beats the constant home rate and vanilla Elo, and does not beat the closing line. The blend of Elo and market reproduces the market, so the served model is the blend when a line exists and calibrated Elo otherwise. See [nba.md](nba.md) and the [architecture overview](architecture.md).

## 4. Open questions

These follow from the validation report and are not tested yet.

- A Poisson model whose expected goals use Elo, or an Elo-based 1X2 for the matches the Poisson model abstains on, are the first candidates to run through the same harness.
- A class-specific correction such as the isotonic option targets the draw bias. A single temperature cannot.
- Neutral venues. The home label is nominal there, and 2,494 of the 7,829 test-year matches were neutral.
- Not evaluated: XGBoost, the hierarchical and bivariate Poisson models, the committee consensus and the tournament simulation.
