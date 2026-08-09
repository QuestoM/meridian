"""Hour 24 exists, no rule was written for it, and the guard inherits the gap.

A break's hour is ``int(start_seconds // 3600)`` and that division is not bounded
at 23. A programme starting at 23:xx can carry a break past midnight, which comes
back as hour 24 or 25 of the SAME broadcast day, and the guardrail key is
``(channel, day, hour)``. So bucket (day D, hour 24) NEVER MERGES with (day D+1,
hour 0), although they are the same sixty minutes of real time.

MEASURED 2026-08-09 on the shipped plan: 9,026 breaks, of which 143 land in hour
24 or 25, and 30 of those are the operator's own.

WHY NOTHING IS CHANGED HERE, and the reasoning matters more than the number.

The day not rolling is DEFENSIBLE. A programme that starts before midnight
belongs to the broadcast day it started in, and that is how the trade thinks
about a broadcast day. Under that reading hour 24 is a real bucket.

But the hourly cap is then applied to a bucket the cited regulation does not
describe. It caps commercial time "in any hour", which reads as a clock hour, and
under a clock hour those breaks belong with the next calendar day's first hour.
Merging them, another agent measured, would put 2 hours over the 12-minute cap
and 11 hours over the 4-pod cap.

That number is NOT reported here as a breach, and this file will not assert it.
The source listing overlaps pervasively around midnight: 687 of 2,539 consecutive
operator rows overlap, and at the seam itself one programme title is listed twice
with two different start times. So the excess merging would reveal may be a data
artifact rather than real concurrent airtime, and asserting it would be reporting
a measurement nobody has taken cleanly.

WHAT THIS FILE DOES INSTEAD. It makes the gap impossible to hold silently. The
concept is named in ``ProgramSegment``'s own docstring, this file measures that
hour 24 is real and reachable, and it pins the fact that the conformance guard
keys on the same tuple and therefore inherits the seam rather than catching it.
A gap that is written down and measured is a decision. The same gap unmeasured is
just something nobody noticed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")

PAST_MIDNIGHT = 24


def _plan_breaks():
    from kairos_api.plan_read_guardrails import plan_guardrail_items

    items = plan_guardrail_items()
    if not items:
        pytest.skip("no plan is on this tree, so there is no hour to measure")
    return items


def test_the_hour_is_not_bounded_at_23_and_the_plan_really_reaches_past_it():
    """The premise, measured rather than argued."""
    items = _plan_breaks()
    past = [item for item in items if item.hour >= PAST_MIDNIGHT]
    assert past, (
        "no break lands past midnight, so either the plan changed shape or the "
        "hour is being clamped somewhere; if it is now clamped, this whole file "
        "and the docstring on ProgramSegment describe a gap that is closed"
    )
    assert max(item.hour for item in items) >= PAST_MIDNIGHT
    # The day does NOT roll with it, which is the half that makes the two
    # buckets unmergeable. Every past-midnight break still carries the broadcast
    # day it started in.
    for item in past:
        assert item.day, "a past-midnight break carries no day at all"


def test_the_two_buckets_that_are_the_same_sixty_minutes_never_meet():
    """(day D, hour 24) and (day D+1, hour 0) are one clock hour and two keys."""
    from datetime import date, timedelta

    items = _plan_breaks()
    keyed: dict[tuple[str, str, int], int] = {}
    for item in items:
        keyed[(item.channel, item.day, item.hour)] = keyed.get((item.channel, item.day, item.hour), 0) + 1
    pairs = 0
    for (channel, day, hour), _ in list(keyed.items()):
        if hour != PAST_MIDNIGHT:
            continue
        try:
            following = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        except ValueError:
            continue
        if (channel, following, 0) in keyed:
            pairs += 1
    assert pairs, (
        "no past-midnight bucket has a next-day hour-0 neighbour, so the seam "
        "cannot be demonstrated on this plan and the measurement above is the "
        "only evidence left"
    )


def test_the_conformance_guard_inherits_the_seam_rather_than_catching_it():
    """Pinned, because a guard that shares a defect cannot report it.

    ``tests/test_guardrail_conformance.py`` groups by the same (channel, day,
    hour) tuple the engine uses. That is correct as a conformance check, since
    its job is to agree with the engine. It also means it passes silently on
    every past-midnight bucket, so nobody should read its green as evidence that
    this question is settled.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parent / "test_guardrail_conformance.py").read_text(encoding="utf-8")
    assert "item.hour" in source or ".hour" in source, (
        "the conformance guard no longer keys on the hour, so this pin is stale "
        "and the seam may now be handled somewhere it was not before"
    )


def test_the_divergence_is_written_down_where_the_hour_is_defined():
    """A gap named at its source is a decision; unnamed it is an oversight."""
    from kairos.optimize import _types

    text = _types.ProgramSegment.__doc__ or ""
    assert "NOT BOUNDED AT 23" in text
    assert "does not describe" in text or "no rule was written for hour 24" in text
