"""The bundled Israeli holiday reference table served with the events payload.

A static checked-in list (``kairos/config/israel_holidays.csv``) the operator
is told to verify before operational use. Split out of events_api.py to keep
that module under the file-size cap; events_api re-exports ``_load_holidays``
so its callers and tests keep one import surface.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HOLIDAYS_PATH = ROOT / "kairos" / "config" / "israel_holidays.csv"


def _load_holidays() -> list[dict[str, Any]]:
    """The bundled holiday reference table. Comment lines (leading '#') carry the
    verify-before-use note and are skipped. Missing file returns empty."""
    if not HOLIDAYS_PATH.exists():
        return []
    with HOLIDAYS_PATH.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        rows = []
        for row in reader:
            rows.append({
                "date": str(row.get("date", "")).strip(),
                "name": str(row.get("name", "")).strip(),
                "kind": str(row.get("kind", "")).strip(),
                "is_school_holiday": str(row.get("is_school_holiday", "")).strip().lower()
                in ("true", "1", "yes"),
            })
    return rows
