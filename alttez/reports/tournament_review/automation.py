from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .analysis import build_tournament_review
from .delivery import (
    deliver_tournament_review,
    load_delivery_record,
    record_delivery,
)
from .repository import load_tournament_predictions
from .writer import write_tournament_review


DEFAULT_REGISTRY_PATH = Path("storage/tournament_reviews.json")


def load_tournament_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tournaments = payload.get("tournaments")
    if not isinstance(tournaments, list):
        raise ValueError("Tournament review registry must contain a tournaments list.")
    required = {"key", "name", "query", "date_from", "date_to", "deliver_after"}
    for tournament in tournaments:
        missing = required - set(tournament)
        if missing:
            raise ValueError(
                f"Tournament {tournament.get('key', '<unknown>')} is missing {sorted(missing)}."
            )
    return tournaments


def _is_ready(
    tournament: dict[str, Any],
    review: dict[str, Any],
    *,
    as_of: date,
) -> tuple[bool, str | None]:
    if not bool(tournament.get("enabled", True)):
        return False, "disabled"
    if as_of < date.fromisoformat(str(tournament["deliver_after"])):
        return False, "before_delivery_date"
    coverage = review["coverage"]
    minimum_matches = int(tournament.get("minimum_unique_matches", 1))
    if int(coverage.get("unique_matches") or 0) < minimum_matches:
        return False, "insufficient_matches"
    minimum_rate = float(tournament.get("minimum_evaluable_rate", 0.0))
    if float(coverage.get("evaluable_rate") or 0.0) < minimum_rate:
        return False, "insufficient_coverage"
    return True, None


def process_tournament_reviews(
    *,
    as_of: date,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    output_dir: str | Path = "artifacts/tournament-reviews",
    database_url: str | None = None,
    deliver: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    results = []
    for tournament in load_tournament_registry(registry_path):
        if not bool(tournament.get("enabled", True)):
            results.append({"key": tournament["key"], "status": "skipped", "reason": "disabled"})
            continue

        existing = load_delivery_record(tournament["key"], database_url=database_url)
        if existing and existing.get("delivery_status") == "delivered" and not force:
            results.append(
                {"key": tournament["key"], "status": "skipped", "reason": "already_delivered"}
            )
            continue

        predictions = load_tournament_predictions(
            tournament["query"],
            date_from=tournament["date_from"],
            date_to=tournament["date_to"],
            database_url=database_url,
        )
        review = build_tournament_review(predictions, tournament=tournament["name"])
        paths = write_tournament_review(review, output_dir=output_dir)
        ready, reason = _is_ready(tournament, review, as_of=as_of)
        if not ready:
            results.append(
                {
                    "key": tournament["key"],
                    "status": "skipped",
                    "reason": reason,
                    "coverage": review["coverage"],
                    "paths": paths,
                }
            )
            continue

        if not deliver:
            results.append(
                {
                    "key": tournament["key"],
                    "status": "ready",
                    "coverage": review["coverage"],
                    "rating": review["summary"].get("rating"),
                    "paths": paths,
                }
            )
            continue

        ok, error = deliver_tournament_review(review, paths["html"])
        status = "delivered" if ok else "failed"
        record_delivery(
            tournament,
            review,
            status=status,
            error_message=error,
            database_url=database_url,
        )
        results.append(
            {
                "key": tournament["key"],
                "status": status,
                "error": error,
                "coverage": review["coverage"],
                "rating": review["summary"].get("rating"),
                "paths": paths,
            }
        )
    return results
