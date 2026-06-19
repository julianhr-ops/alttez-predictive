# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ALTTEZ is a football match-prediction engine (a reshaping of the legacy ProphitBet codebase) focused on national-team / tournament forecasting. It ingests matches from a multi-provider stack into Supabase (Postgres), engineers ELO/Poisson/Klement features, predicts 1X2 + exact scorelines, optimizes scoreline picks against "polla" regulation-point rules, and delivers a daily brief. A Streamlit dashboard surfaces analysis and model monitoring.

The active branch is `desarrollo`. The old `src/` layout was retired in favor of the `alttez/` package.

## Environment & commands

Windows-first. The interpreter is the project venv: `venv\Scripts\python.exe`. CI uses Python 3.13 on Linux (`python`). Dependencies: `pip install -r requirements.txt`. `.env` is auto-loaded at runtime via `alttez.api.settings.load_local_env` — do not assume shell-exported vars.

```powershell
# Run all tests (this repo uses unittest, NOT pytest)
venv\Scripts\python.exe -m unittest discover -s tests

# Run a single test module / case
venv\Scripts\python.exe -m unittest tests.test_inference_router
venv\Scripts\python.exe -m unittest tests.test_model_governance.SomeTestCase.test_method

# Dashboard
venv\Scripts\python.exe -m streamlit run alttez/dashboard/app.py

# Provider stack preflight (which providers are configured)
venv\Scripts\python.exe -m alttez.ingestion.provider_healthcheck --json

# Live pipeline smoke (fetch -> train -> recommend), --write persists to Supabase
venv\Scripts\python.exe scripts\smoke_live_pipeline.py --date 2026-05-26 [--write]

# Migration chain: dry-run plan (no flag) vs apply
venv\Scripts\python.exe scripts\apply_supabase_migrations.py [--apply]

# Champion-vs-challenger governance cycle
venv\Scripts\python.exe scripts\run_challenger_cycle.py --date 2026-05-26

# Full daily process locally (mirrors the scheduled job)
.\scripts\run_daily_process.ps1 -ApplyMigrations -WriteSmoke
```

There is no pytest config, no linter config, and no build step — `unittest discover -s tests` is the test contract. CI (`.github/workflows/ci.yml`) runs the tests + validates the migration chain on every push/PR. The `pre-push` hook (`.githooks/pre-push`, install via `git config core.hooksPath .githooks`) additionally runs `scripts/shadow_readiness_check.py`. `daily_process.yml` runs the whole pipeline on a 02:00 UTC schedule.

## Architecture (big picture)

Data flows: **ingestion → features → inference (router) → reports/dashboard**, with a separate **training/governance** loop that decides which model the router serves.

- **`alttez/ingestion/`** — `provider_stack.py` fetches from an ordered, fall-through provider chain (`espn → football_data_org → statsbomb_open → local_json`, override via `ALTTEZ_PROVIDER_STACK`). ESPN (`providers/espn.py`) is the keyless primary live feed (covers `fifa.world` + `fifa.friendly`); api_football was retired (dead key). Each provider in `providers/` implements the `MatchProvider` base; `is_configured()` gates whether it runs. Raw rows land in `raw_matches_api`. The live-agenda read path (`live_matches.list_matches_for_date`) dedupes the same fixture across providers by canonical team name. `international.InternationalLoader` reads historical data (prefers Supabase, local fallback only when allowed). `certification.py` writes data-quality reports to `artifacts/certification/`. Provider team names are reconciled to historical canonicals via `TEAM_NAME_ALIASES` (e.g. ESPN "Congo DR"→"DR Congo").

- **`alttez/features/`** — `klement.py` is the **single source of truth for the model feature schema** (`MODEL_FEATURE_COLUMNS`); `inference/router.py` and `training/challenger_cycle.py` both alias it, so changing features means changing it here. `poisson.py` (`PoissonEngine`) and `elo.py` are the core engines; `selection.py`/`sampling.py`/`normalization.py` are preprocessing.

- **`alttez/inference/router.py`** — the production read path. `build_active_predictor(historical_df, target_date)` resolves the active model from `public.model_versions` (via `operations.prediction_memory`) and returns either an `XGBoostMatchPredictor` or `PoissonMatchPredictor`, both implementing `BaseMatchPredictor`. **Key invariant: if the active artifact can't load, it falls back to the Poisson champion** (`DEFAULT_POISSON_MODEL_VERSION`) and records `used_fallback`/`fallback_reason` so the brief and dashboard can show it. XGBoost predictors wrap a Poisson baseline and overlay learned λ (schema v2) or class probabilities. `_is_prediction_supported` gates low-history matchups (both teams must clear `ALTTEZ_MIN_SUPPORTED_MATCHES`). Scorelines come from `experiments/tournament_oracle.py` (Dixon-Coles probability matrix) optimized against `evaluation/regulation_points.py`.

- **`alttez/inference/prediction_committee.py`** — El núcleo del ensamble de inferencia. Expone `build_prediction_committee(historical_df, target_date)` que combina las señales del modelo Campeón (ej. XGBoost), el Retador (Challenger) y el baseline de Poisson. Las predicciones del comité unifican métricas clave (`lambda_h`, `lambda_a`, `prob_home`, `prob_draw`, `prob_away`, `primary_score`, `model_version`, `committee_members`). Todo desarrollo de inferencia debe integrarse a través de este comité o del enrutador activo; nunca crees motores de predicción paralelos o aislados. El consenso pasa por una **calibración post-hoc** (`inference/calibration.py`): temperature scaling + piso por clase ajustados offline (`scripts/fit_probability_calibration.py` → `artifacts/calibration/active.json`, regenerado cada corrida diaria). Es NEUTRA si el artefacto no existe, preservando el arranque sin dependencias. Mídela con `scripts/evaluate_calibration.py` (Brier/LogLoss/Accuracy/ECE). Nota: el `log_loss` histórico almacenado usaba un piso `1e-9` que inflaba la métrica (~3.85 vs ~0.98 real); el piso de evaluación ahora es `1e-3`. Los pesos del comité se resuelven por métricas recientes (`inference/committee_weighting.py`): una **guardia de consenso** que defiere al campeón cuando el consenso histórico tiene peor Brier (el promedio de matrices NO garantiza mejorar a sus miembros), y pesos por error inverso (1/Brier con piso de campeón) cuando no se defiere. Al deferir, la calibración del consenso NO se aplica (el campeón ya está calibrado). Expone `committee_weighting_strategy`/`committee_deferred_to_champion` en el payload.

- **`alttez/training/`** — `challenger_cycle.py` trains a challenger, evaluates against the incumbent on recent matches, and applies a **governance rule: Brier score decides eligibility, Log Loss is a guardrail, auto-promotion stays OFF unless `ALTTEZ_AUTO_PROMOTE_MODELS=true`** with incumbent-regression protection. Status is tracked as `eligible` / `shadow` / `active` in `model_versions`.

- **`alttez/operations/prediction_memory.py`** — the model-registry + point-in-time prediction store. Reflects/writes the Supabase tables `prediction_runs`, `match_predictions`, `prediction_evaluations`, `model_versions`. This is the integration seam between training (writes versions), inference (reads active), and reports (writes/closes predictions).

- **`alttez/reports/daily_brief/`** — packaged daily brief (payload → reliability → formatting → render_html → delivery → writer, driven by `cli.py`). `reliability.py` attaches an honest per-match reliability marker (0-100 score + Spanish label + reasons; always present) and a **delivery gate**: `evaluate_delivery_gate` blocks the *send* (never drops a match) when the run would be generic — global Poisson fallback, no model-backed match, empty agenda, or a missing marker — raising a critical incident instead. Persists the day's exact probabilities to prediction memory; the live sync later closes evaluations once matches finish. `telegram_delivery.py` + `operations/alerts.py` handle Telegram delivery and incident alerts (both opt-in via env). `world_cup_brief.py` + `inference/world_cup_2026.py` power the Mundial 2026 simulator (`scripts/run_world_cup_2026_simulation.py`).

- **`alttez/dashboard/`** — Streamlit app; the `Rendimiento del modelo` view reads prediction-memory metrics (Brier, Log Loss, 1X2 accuracy, exact-score hit rate, calibration, champion-vs-challenger).

- **`alttez/api/`** — `db.py` (`get_engine`, SQLAlchemy/psycopg to `SUPABASE_DB_URL`), `settings.py` (`.env` loader), `migrations.py`.

- **`sql/supabase/migrations/`** — numbered, forward-only migration chain (`001…009`). Adding schema means appending the next numbered file; `apply_supabase_migrations.py` validates ordering in CI.

- **`scripts/`** — operational entry points (PowerShell `.ps1` orchestrators + Python CLIs for migrations, challenger cycle, model activation/rollback, briefs, smoke tests).

## Conventions & gotchas

- Tests are `unittest`, discovered from `tests/`. Match that style for new tests.
- User-facing strings (dashboard, briefs, prediction payloads) are in **Spanish** — keep new ones consistent.
- The feature schema lives only in `features/klement.py`; never redefine feature column lists elsewhere — alias `MODEL_FEATURE_COLUMNS`.
- Inference must stay bootable without the model registry: registry/DB lookups in the router are wrapped to fall back to Poisson rather than raise.
- Supabase access needs `SUPABASE_DB_URL`. Provider keys (`ALTTEZ_API_FOOTBALL_KEY`, `ALTTEZ_FOOTBALL_DATA_KEY`, etc.), data-source/fallback flags (`ALTTEZ_DATA_SOURCE`, `ALTTEZ_ALLOW_LOCAL_FALLBACK`), and delivery/alert toggles are all read from `.env`. See `.env.example`.
- Migrations are forward-only and numbered — never edit an applied migration; add a new one.
- **Modular Design Policy (Anti-Monolithic Constraint)**: Queda estrictamente prohibido escribir funciones monolíticas o scripts gigantes con múltiples responsabilidades. Las tuberías de datos, transformaciones estadísticas (como `PoissonEngine`) y la lógica de inferencia deben fracturarse en funciones puras, pequeñas, acotadas y unitariamente testeables bajo el principio de Responsabilidad Única (SRP). Las capas de datos se comunican mediante interfaces limpias (DataFrames limpios o diccionarios tipados), nunca mediante acoplamiento directo de lógica de negocio.
- **Cold Start Handling (Estrategia de Contingencia)**: Para mitigar el problema del arranque en frío cuando el torneo tiene pocos partidos jugados, el sistema implementa una arquitectura de fallback en cadena: Data del Mundial Actual ➔ Semillas Históricas (`world_cup_historical_seeds.csv`) ➔ Valor Neutro Estricto (`1.0`). Al modificar la lógica de extracción de fuerzas, asegúrate de mantener búsquedas optimizadas en complejidad $O(1)$ mediante diccionarios mapeados por equipo para proteger el tiempo de ejecución.
- **Unicode Resilience**: Al interactuar con los flujos de impresión o logs de consola, evita el uso de emojis o caracteres Unicode complejos en los métodos principales. La terminal de Windows (PowerShell/CMD) bajo ciertos entornos locales puede levantar un `UnicodeEncodeError`. Mantén los logs limpios y pragmáticos en texto plano.
