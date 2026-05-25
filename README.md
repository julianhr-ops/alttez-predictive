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

## Supabase

The current schema lives in:

- [001_schema_v2.sql](sql/supabase/001_schema_v2.sql)
- [002_international_results_compat_view.sql](sql/supabase/002_international_results_compat_view.sql)

The compatibility view keeps the current analytical code working while the canonical schema evolves underneath.

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
