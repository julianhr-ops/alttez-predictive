# ALTTEZ Predictive Engine

ALTTEZ is the active development branch of the former ProphitBet codebase, reshaped into a football prediction platform focused on:

- international match ingestion
- Supabase-backed historical storage
- ELO and Poisson feature engineering
- tournament prediction optimization
- a Streamlit dashboard for interactive analysis
- a multi-provider ingestion stack for operational data

This branch is currently centered on national-team workflows and tournament forecasting.

## Status

The `desarrollo` branch already includes:

- migration from the legacy `src/` layout to the `alttez/` package
- a Supabase schema and compatibility view for historical international results
- a historical seed pipeline from `international_results.csv`
- a Streamlit dashboard powered by the new ingestion layer
- separate historical and live provider stacks with ESPN, API-Football, football-data.org, StatsBomb Open Data, and local JSON support

What is still in progress:

- completing a local StatsBomb Open Data mirror in this environment
- building the transformation layer from `raw_matches_api` into canonical match tables for daily ingestion

## Repository layout

```text
alttez/
  api/                Database and environment helpers
  dashboard/          Streamlit application
  experiments/        Research and validation scripts
  features/           ELO, Poisson, statistics, preprocessing
  ingestion/          Historical seed, loaders, provider stack
  models/             Legacy model wrappers adapted to alttez/
  training/           Training helpers

sql/supabase/         Schema and compatibility SQL
sql/supabase/migrations/  Versioned Supabase migration chain
scripts/              Operational PowerShell helpers
storage/network/      Local cached datasets
docs/                 Operational notes
```

## Core workflows

### 1. Dashboard

Run the Streamlit app:

```powershell
venv\Scripts\python.exe -m streamlit run alttez/dashboard/app.py
```

Optional helper script:

```powershell
.\scripts\start_streamlit_supabase.ps1
```

That helper now runs the shadow readiness preflight before launching Streamlit.

One-click launcher for Windows:

```powershell
.\open_alttez_dashboard.cmd
```

Or, if you prefer PowerShell directly:

```powershell
.\scripts\open_alttez_dashboard.ps1
```

The dashboard reads historical data through `InternationalLoader`, which now prefers Supabase and falls back locally only when needed.

### 2. Historical seed into Supabase

Dry run:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.seed_historical --dry-run
```

Real load:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.seed_historical
```

### 3. Provider stack healthcheck

Check which providers are configured:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_healthcheck
```

JSON output:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_healthcheck --json
```

### 4. Provider stack dry-run

Fetch from the configured provider stack without writing to Supabase:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_stack --dry-run
```

Provider override example:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_stack --provider-order api_football,football_data_org --dry-run
```

### 5. Live pipeline smoke test

Validate provider fetch, historical training, and recommendation generation in one pass:

```powershell
venv\Scripts\python.exe scripts\smoke_live_pipeline.py --date 2026-05-26
```

Persist the live agenda and verify readback from Supabase:

```powershell
venv\Scripts\python.exe scripts\smoke_live_pipeline.py --date 2026-05-26 --write
```

When the live batch includes finished matches with enough metric coverage, ALTTEZ also writes a certification report under `artifacts/certification/` and the dashboard surfaces that status alongside the sync result.

A successful provider query with no international fixtures exits with code 3 (idle/no_schedule), distinct from code 1 for provider or operational failures.

Daily process runner:

```powershell
.\scripts\run_daily_process.ps1 -ApplyMigrations -WriteSmoke
```

The same sequence is scheduled in GitHub Actions at 02:00 UTC through `.github/workflows/daily_process.yml`, where the workflow materializes a temporary `.env`, applies migrations, runs the live smoke pipeline, generates the daily brief, runs the champion-vs-challenger batch, and uploads certification plus model-evaluation artifacts. Daily brief delivery is enabled automatically only when the required Telegram or webhook secrets are present.

### 6. Daily brief to Telegram

The brief can be delivered automatically to a Telegram chat after the daily process.

Local setup:

```powershell
ALTTEZ_DAILY_BRIEF_DELIVER=true
ALTTEZ_DAILY_BRIEF_DELIVERY_CHANNEL=telegram
ALTTEZ_DAILY_BRIEF_TELEGRAM_BOT_TOKEN=<bot-token>
ALTTEZ_DAILY_BRIEF_TELEGRAM_CHAT_ID=<chat-id>
```

Manual delivery test:

```powershell
.\scripts\send_daily_brief.ps1 -Date 2026-05-26
```

GitHub Actions setup:

- `ALTTEZ_DAILY_BRIEF_TELEGRAM_BOT_TOKEN`
- `ALTTEZ_DAILY_BRIEF_TELEGRAM_CHAT_ID`

Once those secrets exist, the scheduled workflow sends the brief every day at `02:00 UTC`.

### 7. Prediction memory and model monitoring

ALTTEZ now keeps point-in-time prediction memory in Supabase:

- `prediction_runs`
- `match_predictions`
- `prediction_evaluations`
- `model_versions`

The daily brief persists the exact probabilities emitted for the day, and the live sync closes evaluations once those matches finish. The Streamlit dashboard now includes a dedicated `Rendimiento del modelo` view with:

- historical `Brier Score`
- `Log Loss`
- `1X2 accuracy`
- exact score hit rate
- metrics by `model_version`
- confidence-band calibration
- champion vs challenger registry

Manual challenger cycle:

```powershell
venv\Scripts\python.exe scripts\run_challenger_cycle.py --date 2026-05-26
```

The challenger cycle applies a formal governance rule:

- the champion is rebuilt with the production configuration (active model record, the router's Poisson settings and the fitted Dixon-Coles rho); if that configuration cannot be loaded the cycle reports `production_config_error` and promotes nothing
- champion and challenger are compared on the identical holdout matches with a paired bootstrap of the per-match `Brier Score` difference; the challenger is eligible only if the whole confidence interval is below 0 and the holdout has at least 200 matches
- `Log Loss` and `RPS` act as guardrails
- the report also scores champion and challenger against uniform, base-rate and Elo-only baselines on the same matches
- `production_status` shows whether the challenger is only `eligible`, still `shadow`, or truly `active`
- auto-promotion stays off by default unless `ALTTEZ_AUTO_PROMOTE_MODELS=true`

The production router now resolves the active model directly from `public.model_versions`. If the active artifact is not loadable, ALTTEZ falls back automatically to the Poisson champion and leaves that trace in the brief and dashboard payload.

Manual activation and rollback:

```powershell
venv\Scripts\python.exe scripts\activate_model_version.py --version-key <version-key>
venv\Scripts\python.exe scripts\rollback_active_model.py
```

To allow XGBoost as a production family, keep:

```powershell
ALTTEZ_SUPPORTED_PRODUCTION_MODEL_FAMILIES=poisson,xgboost
```

### 7.1. Mundial 2026 simulator

ALTTEZ now includes a dedicated `World Cup 2026` simulator that runs on the same active predictor used by the daily brief.

Run it with:

```powershell
venv\Scripts\python.exe scripts\run_world_cup_2026_simulation.py --date 2026-05-27 --iterations 500
```

The simulator expects a local JSON configuration at:

```text
storage/world_cup_2026/world_cup_2026_config.json
```

That file should declare:

- `groups`
- `knockout_fixtures`

Artifacts are written under:

```text
artifacts/world-cup-2026/
```

For a fast validation pass, use `--iterations 100`.

The `Rendimiento del modelo` panel now shows:

- the current `active` model
- challengers `eligible` or `blocked`
- and the latest `Mundial 2026` simulation artifact when available

### 8. Telegram incident alerts

If you want critical pipeline failures to reach Telegram too, configure:

```powershell
ALTTEZ_ALERTS_TELEGRAM_ENABLED=true
ALTTEZ_ALERTS_TELEGRAM_BOT_TOKEN=<bot-token>
ALTTEZ_ALERTS_TELEGRAM_CHAT_ID=<chat-id>
```

The daily process now sends a short incident message when a critical stage fails, including:

- readiness
- migrations
- smoke pipeline
- daily brief
- challenger cycle

## Environment

Copy the template and fill in the real credentials:

```powershell
Copy-Item .env.example .env
```

Important variables:

- `SUPABASE_DB_URL`
- `ALTTEZ_DATA_SOURCE`
- `ALTTEZ_SUPABASE_VIEW`
- `ALTTEZ_ALLOW_LOCAL_FALLBACK`
- `ALTTEZ_PROVIDER_STACK`
- `ALTTEZ_API_FOOTBALL_KEY`
- `ALTTEZ_FOOTBALL_DATA_KEY`
- `ALTTEZ_STATSBOMB_OPEN_ROOT`
- `ALTTEZ_LOCAL_JSON_PATH`

The runtime now auto-loads `.env` through `alttez.api.settings`, so the app no longer depends on shell-exported variables.

If you change `.env` while Streamlit is already running, restart the process before retrying a live sync. The launcher scripts already do the right thing in the right order, so you usually do not need to remember the long command anymore.
Recommended pre-flight check:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_healthcheck --json
```

## Supabase

The canonical migration chain for fresh installs lives in:

- [001_base_catalogs_and_match_tables.sql](sql/supabase/migrations/001_base_catalogs_and_match_tables.sql)
- [002_match_enrichment.sql](sql/supabase/migrations/002_match_enrichment.sql)
- [003_indexes.sql](sql/supabase/migrations/003_indexes.sql)
- [004_updated_at_triggers.sql](sql/supabase/migrations/004_updated_at_triggers.sql)
- [005_international_results_compat_view.sql](sql/supabase/migrations/005_international_results_compat_view.sql)
- [006_certification_batches.sql](sql/supabase/migrations/006_certification_batches.sql)
- [007_alert_events.sql](sql/supabase/migrations/007_alert_events.sql)
- [008_prediction_memory.sql](sql/supabase/migrations/008_prediction_memory.sql)
- [009_model_governance.sql](sql/supabase/migrations/009_model_governance.sql)

The compatibility view keeps the current analytical code working while the canonical schema evolves underneath.
Dry-run the migration chain:

```powershell
venv\Scripts\python.exe scripts\apply_supabase_migrations.py
```

Apply the migration chain to Supabase:

```powershell
venv\Scripts\python.exe scripts\apply_supabase_migrations.py --apply
```

The migration chain is also validated automatically in CI on push and pull request.

Optional local hook install:

```powershell
.\scripts\install_git_hooks.ps1
```

That installs the `pre-push` hook from `.githooks/pre-push` into `.git/hooks/pre-push`, which runs tests, readiness, and the migration plan before a push.
If your checkout blocks writes into `.git`, use `git config core.hooksPath .githooks` or copy the hook manually into `.git/hooks/pre-push`.

## Modeling stack

### Historical international pipeline

- `alttez.ingestion.international.InternationalLoader`
- `alttez.features.elo.EloEngine`
- `alttez.features.poisson.PoissonEngine`
- `alttez.experiments.world_cup_prep`
- `alttez.experiments.train_world_cup`
- `alttez.experiments.tournament_oracle`  

### Tournament optimizer

The current tournament oracle:

- estimates expected goals from recent form
- builds a Poisson score matrix
- computes expected value using the polla rules
- returns the scoreline that maximizes long-run points

## Provider strategy

ALTTEZ does not rely on a single API anymore.

The general ingestion stack defaults to:

1. `api_football`
2. `football_data_org`
3. `statsbomb_open`
4. `local_json`

The live stack defaults to `espn`, `football_data_org`, and `local_json`. The scheduled workflow sets its own provider order. API-Football and football-data.org require keys; ESPN is keyless.

More details:

- [docs/provider-stack.md](docs/provider-stack.md)

## Notes for this branch

- `desarrollo` is the active branch for the ALTTEZ migration work
- the legacy top-level launcher files were removed in favor of the package-first structure
- the old `src/` tree is being retired from this branch

## Practical next steps

1. finish wiring a local StatsBomb Open Data mirror
2. transform `raw_matches_api` into canonical `matches` and `match_team_stats`
3. schedule the daily ingestion job
4. commit the branch in logical blocks as the migration stabilizes
