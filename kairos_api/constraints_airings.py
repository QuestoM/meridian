"""Airings: the occurrence a restriction can name, from the plan of record.

A restriction is written about a thing that happens once. "The finale airs
Sunday" is an occurrence, not a programme, and until this module there was no
object for it: the store scoped by programme title, which is every airing of it
forever, and the composer had no way to show a representative which nights they
were actually about to change.

An airing here is a segment of the saved weekly plan, joined to the engine's own
rebuilt segment so it carries the two facts the restriction language needs: how
long the programme runs and how many breaks the plan of record gives it. Both
come from the seams the commit path uses, so the composer, the preview and the
optimizer are looking at the same objects.

Everything is scoped to the operator's own channel, read from settings. A
restriction is about the operator's inventory and no competitor's programme can
appear in the list a representative picks from.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional, Sequence

from kairos_api.constraints_language import Airing

logger = logging.getLogger(__name__)

# Enough to fill a composer's programme picker without turning a type-ahead into
# a scroll. The count of matches is always reported beside the truncated list.
TITLE_SUGGESTIONS = 40


def operator_channel() -> str:
    from kairos_api._constraint_options import load_operator_channel

    return load_operator_channel()


def _plan_counts() -> dict[str, int]:
    """Planned break count per segment id, from the saved weekly plan."""
    from kairos_api.core import _load_break_schedule

    frame = _load_break_schedule()
    if frame.empty or "segment_id" not in frame.columns:
        return {}
    counts: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        try:
            counts[str(getattr(row, "segment_id", "")).strip()] = int(float(getattr(row, "num_breaks", 0) or 0))
        except (TypeError, ValueError):
            continue
    return counts


def broadcast_days(channel: str) -> list[str]:
    """Every broadcast day the programme schedule holds for one channel."""
    from kairos.data.loaders import load_programmes

    frame = load_programmes()
    if channel:
        frame = frame[frame["Channel"] == channel]
    frame = frame[frame["start_dt"].notna()]
    return sorted(frame["start_dt"].dt.strftime("%Y-%m-%d").unique().tolist())


@lru_cache(maxsize=4)
def _airings_cached(signature: tuple, channel: str) -> tuple[tuple[Airing, ...], tuple[Any, ...]]:
    """Every airing of the operator's channel in the plan window, with its segment.

    Built one broadcast day at a time, which is what makes the join to the plan of
    record real. A segment id is ``day|channel|index`` where the index is the row
    position within the built frame (``kairos.data.transform:255``), so a
    whole-window build numbers segments across the window while the plan of record
    was written a day at a time. Measured on 2026-08-01: the whole-window build
    matched 82 of 2,540 ids, all of them on the first day, so every airing after
    2024-11-01 reported a planned count of zero that no file held. Day by day the
    join is 2,540 of 2,540 in both directions, and it is also four times faster
    (0.59 s against 4.87 s), because the classifier runs on a day at a time.

    Cached on the real inputs, the same way the yield band is, so a composer that
    previews five drafts in a row pays the build once.
    """
    del signature  # cache key only
    from kairos_api.preview_inputs import preview_inputs

    counts = _plan_counts()
    airings: list[Airing] = []
    kept: list[Any] = []
    for day in broadcast_days(channel):
        segments, _kwargs = preview_inputs(channel or None, day, None)
        for segment in segments:
            if channel and str(segment.channel).strip() != channel:
                continue
            segment_id = str(segment.segment_id)
            planned = counts.get(segment_id)
            airings.append(Airing(
                segment_id=segment_id,
                channel=str(segment.channel),
                day=str(segment.day),
                title=str(segment.program_title),
                start_seconds=float(segment.start_seconds),
                duration_seconds=float(segment.duration_seconds),
                break_length_seconds=float(segment.break_length_seconds),
                planned_breaks=None if planned is None else int(planned),
            ))
            kept.append(segment)
    return tuple(airings), tuple(kept)


def _signature() -> tuple:
    from kairos_api.core import DATA_DIR, MODELS_DIR, OUTPUT_DIR, SETTINGS_PATH, _signature as sig

    return sig([
        OUTPUT_DIR / "weekly_break_schedule.csv",
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        SETTINGS_PATH,
        MODELS_DIR / "tv_break_coefficients.json",
    ])


def all_airings() -> tuple[tuple[Airing, ...], tuple[Any, ...]]:
    """The operator channel's airings and their engine segments, in plan order."""
    return _airings_cached(_signature(), operator_channel())


def matching(where: Optional[dict[str, Any]]) -> list[Airing]:
    """Airings the predicate selects, judged by the engine's own matcher.

    The composer and the optimizer must agree about which nights a restriction
    touches, so this calls :func:`kairos.optimize.predicate.evaluate_predicate`
    rather than reimplementing the match. With no predicate every airing of the
    operator's channel matches, which is what a scope-free restriction means.
    """
    airings, segments = all_airings()
    if not where:
        return list(airings)
    from kairos.optimize.predicate import evaluate_predicate

    channel = operator_channel()
    out: list[Airing] = []
    for airing, segment in zip(airings, segments):
        try:
            if evaluate_predicate(where, segment, operator_channel=channel or None):
                out.append(airing)
        except Exception:  # pragma: no cover - a malformed tree is rejected at create time
            logger.debug("predicate evaluation failed for %s", airing.segment_id, exc_info=True)
    return out


def titles(query: str = "") -> dict[str, Any]:
    """Programme titles on the operator's channel, for the composer's picker.

    Returns the matching titles with how many airings each has and the days it
    runs, so a representative choosing a programme can see straight away whether
    they are about to change one night or thirty.
    """
    airings, _ = all_airings()
    needle = str(query or "").strip().lower()
    buckets: dict[str, list[Airing]] = {}
    for airing in airings:
        if needle and needle not in airing.title.lower():
            continue
        buckets.setdefault(airing.title, []).append(airing)
    rows = [
        {
            "title": title,
            "airings": len(group),
            "first_day": min(a.day for a in group),
            "last_day": max(a.day for a in group),
            "planned_breaks": sum(a.planned_breaks or 0 for a in group),
            "airings_without_a_plan": sum(1 for a in group if a.planned_breaks is None),
        }
        for title, group in buckets.items()
    ]
    rows.sort(key=lambda row: (-row["airings"], row["title"]))
    return {
        "channel": operator_channel(),
        "match_count": len(rows),
        "titles": rows[:TITLE_SUGGESTIONS],
        "truncated": len(rows) > TITLE_SUGGESTIONS,
    }


def segments_for(airings: Sequence[Airing]) -> list[Any]:
    """The engine segments behind a set of airings, in plan order."""
    every, segments = all_airings()
    by_id = {airing.segment_id: segment for airing, segment in zip(every, segments)}
    return [by_id[airing.segment_id] for airing in airings if airing.segment_id in by_id]


def resolved_changes(rows: Sequence[Any], matched: Sequence[Airing]) -> dict[str, Any]:
    """What the engine's own resolver does to each matched airing, before any save.

    The composer and the optimizer have to agree about which nights a rule moves,
    so this asks the engine rather than reading the compiled rows a second way:
    :func:`kairos.optimize.constraints_store.resolve_constraints` is the exact
    function the commit path runs, and its answer for a break count is
    ``forbid`` to zero, ``pin_count`` to the count, and a placement pin fixing the
    count at the number of pins (``optimizer.py`` docstring, "count forced to
    ``len(pins)``").

    Resolving over the matched airings alone is exact rather than a shortcut: a
    compiled row's predicate is the author's scope narrowed, never widened, so no
    airing outside the matched set can match a row.

    Returns the per-airing changes, the airings the rule binds without moving
    their count, the airings whose planned count is unknown, and every constraint
    the resolver itself refused, with its own reason. ``bound_ids`` names the
    bound airings rather than only counting them, because a caller comparing what
    the rule holds against what its sentence asked for needs the identities and
    not the total.
    """
    from kairos.optimize.constraints_store import resolve_constraints
    from kairos_api.constraints_cost import placement_constraints

    segments = segments_for(matched)
    if not segments:
        return {
            "changes": [], "bound": 0, "bound_ids": [], "unknown": 0,
            "skipped": [], "bound_days": [],
        }
    pins, counts, forbids, skipped = resolve_constraints(
        segments, placement_constraints(rows, "preview"), operator_channel=operator_channel(),
    )
    by_id = {airing.segment_id: airing for airing in matched}
    changes: list[dict[str, Any]] = []
    bound_days: set[tuple[str, str]] = set()
    bound_ids: list[str] = []
    bound = unknown = 0
    for segment_id in sorted(set(counts) | set(pins)):
        airing = by_id.get(segment_id)
        if airing is None:
            continue
        bound += 1
        bound_ids.append(segment_id)
        bound_days.add((airing.channel, airing.day))
        if airing.planned_breaks is None:
            unknown += 1
            continue
        after = 0 if segment_id in forbids else counts.get(segment_id)
        if after is None:
            after = len(pins.get(segment_id) or ())
        effect = "forbid" if segment_id in forbids else ("pin_count" if segment_id in counts else "fix_offset")
        if int(after) == int(airing.planned_breaks):
            continue
        changes.append({
            "segment_id": segment_id,
            "day": airing.day,
            "channel": airing.channel,
            "title": airing.title,
            # Two airings of one programme on one night differ only by when they
            # start, so without the time the list reads as a duplicated row.
            "start_seconds": airing.start_seconds,
            "duration_seconds": airing.duration_seconds,
            "before_breaks": int(airing.planned_breaks),
            "after_breaks": int(after),
            "effect": effect,
        })
    changes.sort(key=lambda row: (row["day"], row["segment_id"]))
    return {
        "changes": changes,
        "bound": bound,
        "bound_ids": bound_ids,
        "bound_days": sorted(bound_days),
        "unknown": unknown,
        "skipped": [
            {"segment_id": item.segment_id, "reason": item.reason} for item in skipped
        ],
    }


def airing_records(airings: Sequence[Airing], limit: int = 200) -> list[dict[str, Any]]:
    """Airings as payload records, newest facts first and never a guessed one."""
    out: list[dict[str, Any]] = []
    for airing in airings[:limit]:
        out.append({
            "segment_id": airing.segment_id,
            "day": airing.day,
            "title": airing.title,
            "start_seconds": airing.start_seconds,
            "duration_seconds": airing.duration_seconds,
            "break_length_seconds": airing.break_length_seconds,
            "planned_breaks": airing.planned_breaks,
        })
    return out
