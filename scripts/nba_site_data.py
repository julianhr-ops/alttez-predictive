"""Write static NBA site data from the prediction database."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttez.api.db import get_engine
from alttez.site.nba_data import write_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    engine = get_engine(args.database_url)
    try:
        write_data(engine, args.out)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
