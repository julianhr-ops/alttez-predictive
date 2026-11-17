# Provider stack

ALTTEZ reads soccer fixtures and results through an ordered chain of providers. A provider that is not configured is skipped, and the next one is tried. No single API is required.

## Provider order

Two stacks exist, each with its own environment variable.

| Stack | Variable | Default in code | Value in `.env.example` and the daily workflow |
| --- | --- | --- | --- |
| Raw ingestion (`provider_stack.py`) | `ALTTEZ_PROVIDER_STACK` | `api_football`, `football_data_org`, `statsbomb_open`, `local_json` | `espn`, `football_data_org`, `statsbomb_open`, `local_json` |
| Live daily agenda (`live_matches.py`) | `ALTTEZ_LIVE_PROVIDER_STACK` | `espn`, `football_data_org`, `local_json` | `espn`, `football_data_org`, `local_json` |

The code default for the raw stack lists `api_football` first. Set `ALTTEZ_PROVIDER_STACK` to the `.env.example` value to put the keyless ESPN feed first.

## Providers

| Provider | Needs | Role |
| --- | --- | --- |
| `espn` | no key | Unofficial public scoreboard JSON. Primary live agenda and results, including friendlies. Enabled unless `ALTTEZ_ESPN_ENABLED=false`. |
| `football_data_org` | `ALTTEZ_FOOTBALL_DATA_KEY` | Structured fixtures and scores for the competitions in `ALTTEZ_FOOTBALL_DATA_LIVE_COMPETITIONS`. |
| `api_football` | `ALTTEZ_API_FOOTBALL_KEY` | Optional paid feed. The scheduled workflow does not include it in its stacks. |
| `statsbomb_open` | `ALTTEZ_STATSBOMB_OPEN_ROOT` | Event-level historical data from a local clone of StatsBomb Open Data. |
| `local_json` | `ALTTEZ_LOCAL_JSON_PATH` | Manual fallback for tests and emergency ingestion. |

ESPN and the DraftKings odds it relays are third-party data. ALTTEZ has no affiliation with ESPN, DraftKings, or any league. The ESPN endpoints are unofficial and can change or disappear without notice.

## Environment variables

`.env.example` is the full template. The variables for providers are `ALTTEZ_PROVIDER_STACK`, `ALTTEZ_LIVE_PROVIDER_STACK`, `ALTTEZ_ESPN_ENABLED`, `ALTTEZ_ESPN_LEAGUES`, `ALTTEZ_ESPN_BASE_URL`, `ALTTEZ_API_FOOTBALL_KEY`, `ALTTEZ_FOOTBALL_DATA_KEY`, `ALTTEZ_STATSBOMB_OPEN_ROOT`, and `ALTTEZ_LOCAL_JSON_PATH`. Never commit a real key.

## Check which providers are configured

```bash
python -m alttez.ingestion.provider_healthcheck --json
```

The output lists each provider with `configured`, `detail`, and its position in the stack. With `ALTTEZ_PROVIDER_STACK` unset, the report marks `espn` as an unknown provider because it checks the code default for the raw stack, which has no `espn` entry.

If you change `.env`, restart Streamlit and any other long-running process. A running process keeps the old values.

## Fetch without writing

```bash
python -m alttez.ingestion.provider_stack --dry-run
python -m alttez.ingestion.provider_stack --provider-order espn --dry-run
```

The first command uses the configured order and skips every provider that has no configuration. The second overrides the order and queries only ESPN. Neither writes to the database.

## StatsBomb Open Data layout

The `statsbomb_open` provider expects a local clone that keeps the public repository layout.

```text
<root>
  data/
    matches/
      <competition_id>/
        <season_id>.json
```

Set the clone path with `ALTTEZ_STATSBOMB_OPEN_ROOT`. StatsBomb Open Data requires attribution when you use it. Download scripts are in `scripts/download_statsbomb_open_data.ps1` and `scripts/check_statsbomb_download.ps1`.
