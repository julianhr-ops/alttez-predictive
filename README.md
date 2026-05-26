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
- a provider stack with `API-Football`, `football-data.org`, `StatsBomb Open Data`, and local JSON fallback support

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

Daily process runner:

```powershell
.\scripts\run_daily_process.ps1 -ApplyMigrations -WriteSmoke
```

The same sequence is scheduled in GitHub Actions at 02:00 UTC through `.github/workflows/daily_process.yml`, where the workflow materializes a temporary `.env`, applies migrations, runs the live smoke pipeline, and uploads certification reports as build artifacts. Daily brief delivery is enabled automatically only when the required Telegram or webhook secrets are present.

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

Default stack:

1. `api_football`
2. `football_data_org`
3. `statsbomb_open`
4. `local_json`

This lets us combine operational coverage, structured fallback, analytical richness, and manual recovery paths.

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
- a provider stack with `API-Football`, `football-data.org`, `StatsBomb Open Data`, and local JSON fallback support

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

Daily process runner:

```powershell
.\scripts\run_daily_process.ps1 -ApplyMigrations -WriteSmoke
```

The same sequence is scheduled in GitHub Actions at 02:00 UTC through `.github/workflows/daily_process.yml`, where the workflow materializes a temporary `.env`, applies migrations, runs the live smoke pipeline, and uploads certification reports as build artifacts.

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

Default stack:

1. `api_football`
2. `football_data_org`
3. `statsbomb_open`
4. `local_json`

This lets us combine operational coverage, structured fallback, analytical richness, and manual recovery paths.

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
