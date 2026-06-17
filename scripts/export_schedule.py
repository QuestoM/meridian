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

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos.export.schedule import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_weekly_schedule,
    write_weekly_schedule,
)

# The dashboard persists the operator's controls here (same file the API reads);
# the weekly CSV must be built with those controls so the risk-aware decision the
# operator selected actually reaches the screens that render this schedule.
SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"
# Scoped placement constraints the operator saved (same file the API CRUD writes);
# the weekly CSV honors them so a pinned offset / forced count / forbid actually
# reaches the exported schedule. Absent file -> no constraints, unchanged output.
CONSTRAINTS_PATH = ROOT / "data" / "kairos_constraints.csv"


def _load_settings() -> dict[str, Any]:
    """Read the saved KairosSettings, falling back to {} (engine defaults)."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def main() -> int:
    print("Building the real weekly break schedule from the reference data ...")
    settings = _load_settings()
    risk_lambda = float(settings.get("risk_lambda", 0.0) or 0.0)
    # revenue_weight is stored 0..100 in KairosSettings; the optimizer takes 0..1.
    # Thread the operator's saved choice through so the exported schedule reflects
    # the revenue-vs-retention balance they selected, not the engine default (0.5).
    raw_weight = settings.get("revenue_weight")
    revenue_weight = float(raw_weight) / 100.0 if raw_weight is not None else None
    operator_channel = str(settings.get("operator_channel", "") or "")
    print(f"  using saved settings (revenue_weight={raw_weight}, risk_lambda={risk_lambda}, "
          f"retention_floor={settings.get('min_retention_floor', 'default')}, "
          f"operator_channel={operator_channel or '(any)'})")
    constraints_path = str(CONSTRAINTS_PATH) if CONSTRAINTS_PATH.exists() else None
    if constraints_path:
        print(f"  honoring placement constraints from {CONSTRAINTS_PATH.name}")
    if operator_channel:
        print(f"  constraints scoped to operator channel: {operator_channel}")
    frame = build_weekly_schedule(
        settings=settings or None,
        revenue_weight=revenue_weight,
        risk_lambda=risk_lambda,
        constraints_path=constraints_path,
        operator_channel=operator_channel,
    )
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
