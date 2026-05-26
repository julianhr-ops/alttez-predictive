# ALTTEZ Provider Stack

ALTTEZ now uses a provider stack instead of relying on a single football data source.

## Priority

The default order is controlled by `ALTTEZ_PROVIDER_STACK`:

1. `api_football`
2. `football_data_org`
3. `statsbomb_open`
4. `local_json`

## Purpose of each provider

- `api_football`: primary operational fixture feed
- `football_data_org`: structured fallback for fixtures and scores
- `statsbomb_open`: richer historical analytical layer from a local dataset mirror
- `local_json`: manual fallback for testing and emergency ingestion

## Environment variables

See `.env.example` for the full template.

Core keys:

- `ALTTEZ_PROVIDER_STACK`
- `ALTTEZ_API_FOOTBALL_KEY`
- `ALTTEZ_FOOTBALL_DATA_KEY`
- `ALTTEZ_STATSBOMB_OPEN_ROOT`
- `ALTTEZ_LOCAL_JSON_PATH`

## Healthcheck

To see which providers are configured:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_healthcheck
```

JSON output:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_healthcheck --json
```

If you update `.env`, restart Streamlit or any long-lived process before trusting the next sync attempt. The live dashboard now surfaces provider diagnostics, but a stale process can still hold old environment values until restart.

## Dry-run fetch

Run the provider stack without writing to Supabase:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_stack --dry-run
```

Provider override example:

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_stack --provider-order api_football,football_data_org --dry-run
```

## Notes on StatsBomb Open Data

The StatsBomb provider expects a local folder compatible with the public repository structure:

```text
<root>
  data/
    matches/
      <competition_id>/
        <season_id>.json
```

Configure it with:

```text
ALTTEZ_STATSBOMB_OPEN_ROOT=C:\path\to\statsbomb-open-data
```
