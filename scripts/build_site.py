"""Render the static ALTTEZ showcase site.

    python scripts/build_site.py --data-dir site_out/data --out site_out

Reads the six JSON data files from --data-dir and writes the HTML pages, the
assets and a copy of the data into --out. See docs/site.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttez.site.render import SiteDataError, render_site


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        written = render_site(args.data_dir, args.out)
    except SiteDataError as exc:
        print(f"build_site: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
