"""Generate the real optimized weekly break schedule the dashboard reads.

Runs the engine over the real Programmes.xlsx for every channel-day and writes
output/weekly_break_schedule.csv, so the dashboard's schedule canvas, break
operations board and recommendations show computed revenue and retention rather
than the placeholder fallback. Console output is ASCII-only (Windows consoles
default to cp1252 and cannot print Hebrew).

Usage:
    python scripts/export_schedule.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos.export.schedule import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_weekly_schedule,
    write_weekly_schedule,
)


def main() -> int:
    print("Building the real weekly break schedule from the reference data ...")
    frame = build_weekly_schedule()
    if frame.empty:
        print("ERROR: no segments produced (is data/reference/Programmes.xlsx present?)")
        return 1

    path = write_weekly_schedule(frame=frame)
    breaks = int(frame["num_breaks"].sum())
    revenue = float(frame["predicted_revenue"].sum())
    channels = frame["channel"].nunique()
    days = frame["date"].nunique()
    print(f"Wrote {len(frame)} segment rows to {path}")
    print(f"  channels: {channels}, days: {days}")
    print(f"  total breaks placed: {breaks}")
    print(f"  total projected revenue: {revenue:,.0f} ILS")
    print(f"  (relative to {DEFAULT_OUTPUT_PATH.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
