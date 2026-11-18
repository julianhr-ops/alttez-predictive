"""Import, export, or verify an NBA prediction ledger."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.db import get_engine
from alttez.site import ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("import", "export", "verify"))
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--database-url")
    args = parser.parse_args()
    try:
        if args.action == "verify":
            ledger.verify_records(ledger.read_records(args.ledger))
        else:
            if not args.database_url:
                parser.error("--database-url is required for import and export")
            engine = get_engine(args.database_url)
            try:
                if args.action == "import":
                    ledger.import_records(engine, ledger.read_records(args.ledger))
                else:
                    ledger.export_rows(engine, args.ledger)
            finally:
                engine.dispose()
    except (ValueError, KeyError, TypeError) as exc:
        print(f"ledger error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
