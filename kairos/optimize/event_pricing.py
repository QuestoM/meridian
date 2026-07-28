"""Event-date price multipliers read from the operator's calendar events store.

The operator stores calendar events (holidays, wars, sport, other) in
``data/calendar_events.csv`` (writes are owned by ``kairos_api/events_api.py``).
Each event may carry a ``price_multiplier``: an OPERATOR-ASSERTED price factor
for the dates the event covers, exactly like the rate-card day-of-week premiums
(Friday 1.15). It is an assertion, never a measurement, and it touches pricing
only; retention coefficients are not affected in any way.

This module reads that store and builds the date -> multiplier map the pricing
engine consumes (:func:`kairos.optimize.pricing.pricing_from_settings` injects
it into the :class:`~kairos.optimize.pricing.PricingModel` only when the
operator has activated ``pricing_activation.events``, which ships OFF).
Overlapping events compose multiplicatively; an open-ended event covers from
its start to a documented forward horizon.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The operator's calendar-events store (kairos_api/events_api.py owns writes).
DEFAULT_EVENTS_PATH = ROOT / "data" / "calendar_events.csv"
# How far forward (days from today) an OPEN-ENDED event's date map extends. An
# open-ended event (a war without a declared end) covers every date from its
# start onward; the enumerated map is bounded at this documented horizon so it
# stays finite while covering any plannable schedule date.
EVENT_OPEN_HORIZON_DAYS = 366


def load_price_events(
    path: str | Path | None = None,
) -> list[tuple[date, "date | None", float]]:
    """The ACTIVE stored calendar events carrying a non-1.0 price multiplier.

    Reads the operator's events store and returns ``(start, end, multiplier)``
    tuples; ``end`` is None for an open-ended event. Rows that are inactive,
    carry the neutral 1.0 (or no) multiplier, or do not parse are skipped, so a
    legacy store without the ``price_multiplier`` column contributes nothing.
    These multipliers are operator assertions (like the rate-card day premiums),
    not measurements.
    """
    events_path = Path(path) if path is not None else DEFAULT_EVENTS_PATH
    if not events_path.exists():
        return []
    events: list[tuple[date, "date | None", float]] = []
    with open(events_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("active", "")).strip().lower() != "true":
                continue
            try:
                multiplier = float(str(row.get("price_multiplier") or "").strip() or 1.0)
            except ValueError:
                continue
            if multiplier == 1.0:
                continue
            try:
                start = date.fromisoformat(str(row.get("start_date", "")).strip())
            except ValueError:
                continue
            end: "date | None" = None
            end_raw = str(row.get("end_date") or "").strip()
            if end_raw:
                try:
                    end = date.fromisoformat(end_raw)
                except ValueError:
                    continue
                if end < start:
                    continue
            events.append((start, end, multiplier))
    return events


def load_event_day_multipliers(
    path: str | Path | None = None, today: "date | None" = None
) -> dict[str, float]:
    """The date -> price multiplier map from the active stored events.

    Overlapping events compose MULTIPLICATIVELY (two events covering the same
    date multiply their multipliers, mirroring how every other premium layer
    stacks). A bounded event covers its inclusive start..end span; an open-ended
    event covers from its start to ``EVENT_OPEN_HORIZON_DAYS`` past today. The
    values are operator assertions, not measurements.
    """
    today = today or date.today()
    horizon = today + timedelta(days=EVENT_OPEN_HORIZON_DAYS)
    combined: dict[str, float] = {}
    for start, end, multiplier in load_price_events(path):
        last = end if end is not None else max(horizon, start)
        day = start
        while day <= last:
            iso = day.isoformat()
            combined[iso] = combined.get(iso, 1.0) * multiplier
            day += timedelta(days=1)
    return combined
