# ALTTEZ data branch

This branch holds the public NBA prediction ledger of [alttez-predictive](https://github.com/julianhr-ops/alttez-predictive). It has no code.

- `nba/ledger.jsonl` has one JSON object per prediction, sorted by tip-off, game and model.
- A scheduled workflow on `main` appends rows through the bot account. Each row is written before tip-off and graded after the game. The commit history of this file is the audit trail: a prediction cannot be inserted after its game started without the commit time showing it.
- The ledger is checked by `python scripts/nba_ledger.py verify` on every run.
