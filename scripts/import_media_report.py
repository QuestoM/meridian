#!/usr/bin/env python3
"""Import one real media QC CSV into Meridian's canonical asset store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos_api.media_ingest import import_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--source", default="")
    args = parser.parse_args()
    print(json.dumps(import_report(args.report, args.store, args.source), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
