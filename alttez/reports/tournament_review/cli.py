from __future__ import annotations

import argparse
from datetime import date

from alttez.api.settings import load_local_env

from .analysis import build_tournament_review
from .repository import load_tournament_predictions
from .writer import write_tournament_review


def _date_value(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera el cierre auditable de un torneo usando predicciones persistidas."
    )
    parser.add_argument("--tournament", required=True, help="Nombre o fragmento del torneo.")
    parser.add_argument("--date-from", type=_date_value)
    parser.add_argument("--date-to", type=_date_value)
    parser.add_argument("--database-url")
    parser.add_argument("--output-dir", default="artifacts/tournament-reviews")
    return parser.parse_args()


def main() -> int:
    load_local_env()
    args = parse_args()
    predictions = load_tournament_predictions(
        args.tournament,
        date_from=args.date_from,
        date_to=args.date_to,
        database_url=args.database_url,
    )
    review = build_tournament_review(predictions, tournament=args.tournament)
    paths = write_tournament_review(review, output_dir=args.output_dir)
    print(f"tournament={review['tournament']}")
    print(f"persisted_predictions={review['coverage']['persisted_predictions']}")
    print(f"evaluated_predictions={review['coverage']['evaluated_predictions']}")
    print(f"sample_confidence={review['sample_confidence']}")
    if review["summary"].get("rating"):
        print(f"rating={review['summary']['rating']['score']}")
        print(f"rating_label={review['summary']['rating']['label']}")
    print(f"json_path={paths['json']}")
    print(f"html_path={paths['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
