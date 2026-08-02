"""The break as a first-class object: its identity, its day, and what it is worth.

Until now the product had no object below the programme segment. The saved plan
carries one row per segment with a break COUNT, and the board a person looked at
spread those counts evenly across the programme as a display heuristic. So there
was nothing to click, nothing to address, and nothing to attach money to.

This module is that object.

**Identity.** A break is ``<segment_id>~<ordinal>``, for example
``2024-11-01|רשת 13|008~1``. The segment id is the plan's own build-order key and
the ordinal is the 1-based break within the segment, so the id is stable, it is
derivable from the plan alone, and it needs nothing from the daily ad file. That
is the plan-side identity the owner ruling reserved for this piece: the explicit
per-ad identifier the ruling approved belongs to the traffic file and lands with
the break's contents, not with the break's existence.

**Where the placement comes from.** The saved weekly CSV stores counts, not
positions, so a real break position exists only inside the optimizer's own
result. This module therefore runs the commit path's own core
(:func:`kairos.optimize.day_core._optimize_one_day`) for one operator channel-day
through :func:`kairos_api.preview_inputs.preview_inputs`, and reads the placements
it returns. Measured on ``רשת 13 / 2024-11-01``: 82 segments, 80 placed breaks,
1.13 s. Every break's clock time, duration, ordinal and gold flag is the plan's
own, never a spread.

**Why re-scoring is cheap.** The day is held as a
:class:`~kairos.optimize.evaluate.Basis`, so scoring a rearrangement is
:func:`kairos.optimize.evaluate.score`, measured at 105 microseconds for the same
82 segments against 1.13 s for a re-optimization, and the compliance verdict is
:func:`kairos.optimize.guardrails.evaluate` at 53 microseconds. A person dragging
a break gets the plan's own numbers, not an approximation of them.

**The competitor boundary.** A day plan is built for
``settings.operator_channel`` and for no other channel. There is no filter to
forget: the segments for another channel are never constructed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from kairos_api import read_cache

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# One namespace for built day plans, bounded like every other read-cache
# namespace. A person works one day at a time and steps between a handful.
CACHE_NAMESPACE = "plan_day"
read_cache.configure(CACHE_NAMESPACE, capacity=8)

# Serializes the optimizer leg so two people opening the same cold day do not
# each pay 1.1 s of CPU for the same answer.
_BUILD_LOCK = threading.Lock()

BREAK_ID_SEPARATOR = "~"


def break_id(segment_id: str, ordinal: int) -> str:
    """The addressable id of one break: its segment plus its 1-based ordinal.

    The separator is a tilde rather than a hash because a hash in a URL is a
    fragment delimiter: it never reaches a server unencoded, and one caller that
    forgets to encode it turns a break id into a segment id silently. A tilde is
    unreserved in RFC 3986, so it survives every hop untouched.
    """
    return f"{segment_id}{BREAK_ID_SEPARATOR}{int(ordinal)}"


def parse_break_id(value: str) -> tuple[str, int]:
    """Split a break id back into (segment_id, ordinal), or raise ValueError."""
    text = str(value or "").strip()
    segment, separator, ordinal = text.rpartition(BREAK_ID_SEPARATOR)
    if not separator or not segment:
        raise ValueError("a break id reads <segment_id>~<ordinal>")
    try:
        index = int(ordinal)
    except (TypeError, ValueError):
        raise ValueError("the ordinal after the tilde must be a whole number") from None
    if index < 1:
        raise ValueError("the ordinal after the tilde is 1-based")
    return segment, index


@dataclass(frozen=True)
class DayPlan:
    """One operator channel-day, built once and scored many times."""

    channel: str
    day: str
    segments: tuple[Any, ...]
    result: Any
    basis: Any
    engine_kwargs: dict[str, Any]

    @property
    def guardrails(self) -> Any:
        return self.engine_kwargs["guardrails"]

    @property
    def revenue_weight(self) -> float:
        return float(self.engine_kwargs["revenue_weight"])

    def segment(self, segment_id: str) -> Optional[Any]:
        for item in self.segments:
            if item.segment_id == segment_id:
                return item
        return None


def _fingerprint() -> tuple:
    """Everything that can change a day plan, in one comparable value.

    The two operator stores are in it because both feed the optimizer leg: a
    saved pin or a saved override must invalidate the day the moment it lands.
    """
    return (
        read_cache.file_signatures([
            DATA_DIR / "kairos_settings.json",
            DATA_DIR / "manual_overrides.csv",
            DATA_DIR / "kairos_constraints.csv",
            DATA_DIR / "Programmes.csv",
            DATA_DIR / "reference" / "Programmes.xlsx",
            DATA_DIR / "calendar_events.csv",
            OUTPUT_DIR / "weekly_break_schedule.csv",
        ]),
        read_cache.directory_signatures(ROOT / "config", "*.yaml"),
        read_cache.directory_signatures(ROOT / "models", "*"),
    )


def operator_channel() -> str:
    """The one channel this operator owns, read from settings. Never guessed."""
    from kairos_api.core import _load_settings

    return str(_load_settings().operator_channel or "").strip()


def plan_days() -> list[str]:
    """The ISO dates the saved plan covers on the operator's own channel.

    Empty when no plan is saved or no channel is configured, which the caller
    renders as an honest empty state naming the missing input.
    """
    from kairos_api.core import _load_break_schedule

    owned = operator_channel()
    schedule = _load_break_schedule()
    if schedule.empty or not owned:
        return []
    if "channel" not in schedule.columns or "date" not in schedule.columns:
        return []
    mine = schedule[schedule["channel"].astype(str).str.strip() == owned]
    days = sorted({str(value).strip() for value in mine["date"].tolist() if str(value).strip()})
    return days


def _build_day(channel: str, day: str) -> DayPlan:
    from kairos.optimize.day_core import _optimize_one_day
    from kairos.optimize.evaluate import evaluation_basis
    from kairos_api.overrides import _resolved_store_overrides, _stored_constraints
    from kairos_api.preview_inputs import preview_inputs

    segments, engine_kwargs = preview_inputs(channel, day, None)
    if not segments:
        raise LookupError(f"no segments were built for {channel} on {day}")
    active, _stale = _resolved_store_overrides(segments)
    stored = active if active.overrides else None
    result = _optimize_one_day(
        segments,
        constraints=_stored_constraints(),
        overrides=stored,
        **engine_kwargs,
    )
    basis = evaluation_basis(segments, risk_lambda=float(engine_kwargs["risk_lambda"]))
    return DayPlan(
        channel=channel,
        day=day,
        segments=tuple(segments),
        result=result,
        basis=basis,
        engine_kwargs=engine_kwargs,
    )


def day_plan(day: str) -> DayPlan:
    """The operator's own plan for one broadcast day, built once and cached.

    Raises :class:`LookupError` when the channel is unset or the day carries no
    segments, so a route answers an honest 404 or 409 rather than an empty board
    that looks like a finished day with nothing in it.
    """
    channel = operator_channel()
    if not channel:
        raise LookupError("no operator channel is configured in settings")
    key = (channel, str(day or "").strip())
    if not key[1]:
        raise LookupError("a broadcast day is required")
    with _BUILD_LOCK:
        return read_cache.cached(
            CACHE_NAMESPACE,
            key=key,
            fingerprint=_fingerprint(),
            build=lambda: _build_day(channel, key[1]),
        )


def invalidate() -> None:
    """Drop every cached day. Called after a write that changes the plan."""
    read_cache.invalidate(CACHE_NAMESPACE)


def placements_by_segment(result: Any) -> dict[str, list[Any]]:
    """Every placed break grouped by its segment, in time order within each."""
    grouped: dict[str, list[Any]] = {}
    for placement in result.placements:
        grouped.setdefault(placement.segment_id, []).append(placement)
    for items in grouped.values():
        items.sort(key=lambda placement: placement.start_seconds)
    return grouped


def break_records(plan: DayPlan) -> list[dict[str, Any]]:
    """Every break in the day, addressable, with the plan's own figures.

    ``revenue`` is the optimizer's own marginal revenue credited to this break at
    insertion, so the day's breaks sum back to the day's revenue exactly. Nothing
    here is spread, split or re-priced.
    """
    grouped = placements_by_segment(plan.result)
    segments = {segment.segment_id: segment for segment in plan.segments}
    records: list[dict[str, Any]] = []
    for segment_id in sorted(grouped):
        segment = segments.get(segment_id)
        for ordinal, placement in enumerate(grouped[segment_id], start=1):
            start = float(placement.start_seconds)
            duration = float(placement.duration_seconds)
            segment_start = float(segment.start_seconds) if segment is not None else start
            records.append({
                "break_id": break_id(segment_id, ordinal),
                "segment_id": segment_id,
                "ordinal": ordinal,
                "breaks_in_segment": len(grouped[segment_id]),
                "channel": placement.channel,
                "day": placement.day,
                "hour": int(placement.hour),
                "start_seconds": round(start, 1),
                "end_seconds": round(start + duration, 1),
                "duration_seconds": round(duration, 1),
                "offset_seconds": round(max(0.0, start - segment_start), 1),
                "programme": segment.program_title if segment is not None else "",
                "genre": placement.program_type,
                "is_gold": bool(placement.is_gold),
                "projected_revenue": round(float(placement.revenue), 2),
                "segment_retention": round(float(placement.retention), 6),
            })
    return records


def arrangement(plan: DayPlan) -> tuple[dict[str, int], dict[str, tuple[Any, ...]]]:
    """The plan's own arrangement: break counts and pins, ready to re-score."""
    from kairos.optimize.evaluate import counts_from_result, pins_from_result

    counts = counts_from_result(plan.result)
    pins = pins_from_result(plan.result, plan.segments)
    for segment in plan.segments:
        counts.setdefault(segment.segment_id, 0)
    return counts, pins
