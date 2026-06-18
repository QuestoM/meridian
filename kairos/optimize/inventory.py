"""Inventory-aware demand weights for the optimizer (Signal 1).

The break-count optimizer (:mod:`kairos.optimize.optimizer`) decides how many
commercial breaks each programme segment carries from CPP economics and the
retention model. It never sees the real spot pool: the channel's booked and
available inventory, the actual advertiser demand pressing on each slot. This
module is the missing link. It reads the booked/available spots per
channel-day-hour from the Spots inventory CSV (or the daily Wally export) and
turns that observed demand into a per-segment placement-preference weight.

Honesty contract (identical to the advertiser_conditions header-only seed)
--------------------------------------------------------------------------
The weights enter ONLY through ``optimize_breaks(demand_weights=...)``, which
multiplies a segment's apparent ranking gain by ``max(1.0, weight)`` and never
touches the charged revenue. So:

  * With no inventory data present the loader returns an empty pool and
    :func:`build_inventory_weights` returns ``{}`` -> every weight defaults to
    1.0 -> the schedule and total_revenue are BYTE-IDENTICAL to today.
  * A segment can never be charged more or less because of this signal; it only
    changes WHERE the next break prefers to go when two segments tie.

Phase 1 is a SOFT steer: higher booked demand on a slot raises its weight so the
optimizer leans breaks toward where advertisers actually want to buy. A segment
cannot usefully carry more break-seconds than there is inventory to fill, but
enforcing that as a HARD cap belongs to the guardrail layer; this module leaves
a clearly marked hook (:func:`inventory_hard_cap`) for that future work and does
not enforce it yet.

The math is pure and deterministic: no datetime.now, no random. Any reference
"today" a caller needs is passed in; this signal does not need one at all.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from kairos.optimize.optimizer import ProgramSegment

ROOT = Path(__file__).resolve().parents[2]
# The booked-spot inventory the channel already sold. Same file the per-spot
# pricing path reads; here we only count demand, never re-price.
DEFAULT_INVENTORY_PATH = ROOT / "data" / "Spots - inventory.csv"

_SECONDS_PER_HOUR = 3600.0

# How sharply booked demand lifts a segment's weight. The steer is a saturating
# curve: weight = 1 + STEER_GAIN * (booked / (booked + STEER_HALF)). A slot with
# STEER_HALF booked spots sits at half the maximum lift; the curve never exceeds
# 1 + STEER_GAIN, so a wildly oversubscribed slot cannot dominate the ranking.
# Both are explicit, documented, editable knobs.
STEER_GAIN = 0.5
STEER_HALF = 8.0

# Seconds of inventory one booked spot represents, used only by the future hard
# cap hook (a spot is, on the channel, a ~20-30s unit). Documented and editable.
SECONDS_PER_SPOT = 30.0


@dataclass(frozen=True)
class SlotDemand:
    """Observed booked/available demand for one channel-day-hour slot.

    ``booked`` is how many spots are already sold into this slot; ``available``
    is how many the channel still has open (0 when unknown). The slot key is
    ``(channel, day, hour)`` where ``day`` is the YYYY-MM-DD broadcast date and
    ``hour`` is the clock hour 0..23, the same coordinates a ProgramSegment
    exposes through ``segment.channel`` / ``segment.day`` / ``segment.hour``.
    """

    channel: str
    day: str
    hour: int
    booked: int = 0
    available: int = 0


def _parse_iso_day(value: object) -> Optional[str]:
    """Return a YYYY-MM-DD day string from a few common date encodings, or None.

    The inventory CSV carries dates as ``Date_dt`` (already ISO-ish) or as a
    ``02/11/2024`` style ``Last_dt``. We accept ISO directly and fall back to
    day-first DD/MM/YYYY, the Israeli convention in these files. Anything we
    cannot parse yields None so the row is skipped rather than misdated.
    """
    text = str(value or "").strip()
    if not text:
        return None
    head = text.split(" ")[0].split("T")[0]
    if len(head) >= 10 and head[4] == "-" and head[7] == "-":
        return head[:10]
    parts = head.replace("-", "/").split("/")
    if len(parts) == 3:
        a, b, c = parts
        if len(c) == 4 and a.isdigit() and b.isdigit():
            return f"{c}-{int(b):02d}-{int(a):02d}"
    return None


def _parse_hour(row: Mapping[str, str]) -> Optional[int]:
    """Clock hour 0..23 from the inventory row, or None when absent.

    Prefers an explicit ``hour_of_day`` column; falls back to the ``Start_dt``
    timestamp's hour. Returns None when neither is usable so the row is dropped.
    """
    raw = str(row.get("hour_of_day", "") or "").strip()
    if raw:
        try:
            hour = int(float(raw))
            if 0 <= hour <= 23:
                return hour
        except (TypeError, ValueError):
            pass
    start = str(row.get("Start_dt", "") or "").strip()
    if " " in start:
        clock = start.split(" ")[-1]
        bits = clock.split(":")
        if bits and bits[0].isdigit():
            hour = int(bits[0]) % 24
            return hour
    return None


def load_inventory(path: Optional[str | Path] = None) -> dict[tuple[str, str, int], SlotDemand]:
    """Read booked spot inventory into per (channel, day, hour) demand counts.

    Each CSV row is one booked spot; we count spots per slot. ``available`` is
    read from a ``Spots Block 1`` style open-inventory column when present, else
    left at 0 (unknown, not zero-meaning-full). A missing file returns ``{}`` so
    the whole signal is an identity no-op when the owner has not uploaded data.

    Pure and deterministic: the file is the only input, no clock is read.
    """
    target = Path(path) if path is not None else DEFAULT_INVENTORY_PATH
    if not target.exists():
        return {}
    slots: dict[tuple[str, str, int], SlotDemand] = {}
    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            channel = str(row.get("channel", "") or row.get("Channel", "") or "").strip()
            day = _parse_iso_day(row.get("Date_dt")) or _parse_iso_day(row.get("Last_dt"))
            hour = _parse_hour(row)
            if not channel or day is None or hour is None:
                continue
            key = (channel, day, hour)
            current = slots.get(key)
            booked = (current.booked if current else 0) + 1
            available = current.available if current else 0
            open_raw = str(row.get("Spots Block 1", "") or "").strip()
            if open_raw.isdigit():
                available = max(available, int(open_raw))
            slots[key] = SlotDemand(
                channel=channel, day=day, hour=hour,
                booked=booked, available=available,
            )
    return slots


def _booked_for_segment(
    segment: ProgramSegment,
    pool: Mapping[tuple[str, str, int], SlotDemand],
) -> int:
    """Booked spot count for the slot a segment occupies (its channel-day-hour)."""
    slot = pool.get((segment.channel, segment.day, segment.hour))
    return slot.booked if slot is not None else 0


def _demand_to_weight(booked: int) -> float:
    """Map a booked-spot count onto a placement weight >= 1.0 (saturating soft steer).

    weight = 1 + STEER_GAIN * booked / (booked + STEER_HALF)

    Zero booked demand gives exactly 1.0 (identity). The lift rises with demand
    but saturates below ``1 + STEER_GAIN``, so a hot slot is preferred without a
    single oversubscribed slot swamping the ranking.
    """
    if booked <= 0:
        return 1.0
    return 1.0 + STEER_GAIN * (booked / (booked + STEER_HALF))


def build_inventory_weights(
    segments: Iterable[ProgramSegment],
    pool: Optional[Mapping[tuple[str, str, int], SlotDemand]] = None,
    *,
    inventory_path: Optional[str | Path] = None,
) -> dict[str, float]:
    """Per-segment inventory-demand weights (>= 1.0) keyed by segment_id.

    ``pool`` is a pre-loaded demand map (from :func:`load_inventory`); when None
    it is loaded from ``inventory_path`` (default ``data/Spots - inventory.csv``).
    With no inventory file the pool is empty and EVERY weight is 1.0, so the
    returned dict is a pure no-op and the optimizer output is byte-identical to a
    run with no inventory signal at all.

    The weight steers placement toward slots advertisers are actually booking; it
    never enters the charged revenue.
    """
    if pool is None:
        pool = load_inventory(inventory_path)
    weights: dict[str, float] = {}
    for segment in segments:
        booked = _booked_for_segment(segment, pool)
        weights[segment.segment_id] = _demand_to_weight(booked)
    return weights


def inventory_hard_cap(
    segment: ProgramSegment,
    pool: Mapping[tuple[str, str, int], SlotDemand],
    *,
    seconds_per_spot: float = SECONDS_PER_SPOT,
) -> Optional[int]:
    """HOOK (not yet enforced): max breaks a slot's inventory can actually fill.

    A segment cannot usefully carry more break-seconds than there are booked (or
    available) spots to fill. This returns that ceiling in break units so a
    future guardrail can clamp the optimizer hard:

        fillable_seconds = (booked + available) * seconds_per_spot
        cap = floor(fillable_seconds / break_length_seconds)

    Returns None when the slot is unknown (no inventory data), meaning "no cap",
    which is why this is a soft-steer phase: the optimizer is biased, never
    clamped, until this hook is wired into :mod:`kairos.optimize.guardrails`.
    This function is intentionally side-effect-free and unused by the live path.
    """
    slot = pool.get((segment.channel, segment.day, segment.hour))
    if slot is None:
        return None
    fillable = (slot.booked + slot.available) * max(0.0, seconds_per_spot)
    length = segment.break_length_seconds
    if length <= 0:
        return None
    return int(fillable // length)
