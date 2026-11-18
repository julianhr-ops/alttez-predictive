# Showcase site

A static site on GitHub Pages at https://julianhr-ops.github.io/alttez-predictive/. It has no backend, no secrets and makes no external requests. All numbers come from JSON data files generated at build time, so the pages and the pipeline cannot disagree.

## How it is built

`.github/workflows/pages.yml` runs twice a day (14:00 and 22:30 UTC), on manual dispatch, and on pushes and pull requests that touch the site code.

1. **refresh-ledger.** Loads `nba/ledger.jsonl` from the `data` branch into SQLite, runs `scripts/nba_daily.py` to grade finished games and predict upcoming ones, exports the ledger and verifies it. When the file changed and the run is on `main`, the bot commits it back to the `data` branch. It never force-pushes. This is the only job with `contents: write`.
2. **build-site.** Writes the data files with `scripts/nba_site_data.py` and `scripts/soccer_site_data.py`, then renders the pages with `scripts/build_site.py`. The soccer results are cached under `artifacts/soccer` by ISO week.
3. **deploy.** Publishes the artifact with `actions/deploy-pages`. It runs only on `main` and never on pull requests.

A stale-data stop in `nba_daily` (exit code 3) is a warning, not a failure. The site then shows the previous state.

The pages are plain `string.Template` files in `site/templates`, filled by `alttez/site/render.py`. Every dynamic value is escaped, and every number goes through `alttez/site/fmt.py`. Styles and the matchup calculator live in `site/assets`. There is no framework, no CDN and no external font.

## Data files

`scripts/build_site.py` reads these from `--data-dir` and copies them to `<out>/data/`.

| File | Written by | Content |
| --- | --- | --- |
| `nba_upcoming.json` | `scripts/nba_site_data.py` | Upcoming games with the served probability |
| `nba_scoreboard.json` | `scripts/nba_site_data.py` | Served against market Brier and accuracy on graded games |
| `nba_recent.json` | `scripts/nba_site_data.py` | The last 60 graded games |
| `nba_validation.json` | `site/results/`, committed | NBA walk-forward table |
| `soccer_ratings.json` | `scripts/soccer_site_data.py` | Top 100 Elo teams and the 1X2 grids |
| `soccer_validation.json` | `site/results/`, committed | Soccer validation tables |

## The data branch

The ledger lives on the branch `data` at `nba/ledger.jsonl`, one JSON object per prediction. The owner creates the branch once with a first commit. The workflow fails with a clear message when it is missing.

## Run it locally

```
python scripts/nba_ledger.py import --ledger ledger/nba/ledger.jsonl --database-url sqlite:///nba.db
python scripts/nba_daily.py --date 2026-11-18 --database-url sqlite:///nba.db
python scripts/nba_site_data.py --database-url sqlite:///nba.db --out site_out
python scripts/soccer_site_data.py --out site_out
python scripts/build_site.py --data-dir site_out/data --out site_out
python -m http.server 8000 --bind 127.0.0.1 --directory site_out
```

Then open http://127.0.0.1:8000/. `site_out/` is git-ignored.

To render only the pages from existing data files, run the last two commands.

## The matchup calculator

`soccer_ratings.json` holds, for 100 teams, a flat grid of win, draw and loss probabilities in units of 1/10000, once for a neutral venue and once with the first team at home. The page looks a pair up in the grid. For "team B at home" it reads the cell with the two teams swapped and flips win and loss. The grid is produced by the same `ServingPolicy` the pipeline serves.

The build embeds `soccer_ratings.json` in `soccer.html` as a JSON script block when the file is under 1.5 MB. The file is about 0.4 MB, so the page needs no second request and also works when opened from disk, where browsers block `fetch` of local files. Above the limit the build skips the embedding and the page loads `data/soccer_ratings.json` with `fetch`. In that case view the site over HTTP with `python -m http.server`.

## Data attribution

- Schedule and betting-line data: sportsdataverse/hoopR-nba-data (CC BY 4.0).
- Soccer results: martj42/international_results (CC0).
- Lines shown for upcoming games come from ESPN, usually DraftKings, and belong to a third party. Nothing here is affiliated with the NBA, ESPN, DraftKings or any bookmaker.
