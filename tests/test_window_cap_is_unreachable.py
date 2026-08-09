"""The evening-window cap cannot fire on today's engine, and this pins why.

The product already ships one cap that cannot bind: ``max_ad_minutes_per_hour``
is 12 minutes, and 4 pods an hour at a hardcoded 120-second break length is a
hard ceiling of 8. The window cap inherits that ceiling. Four evening hours
cannot hold more than 32 minutes, so a 40-minute cap on 20:00-24:00 is not
merely slack, it is STRUCTURALLY UNREACHABLE: no schedule this engine can emit
will trip it.

This file exists so that fact is measured rather than asserted in prose, and so
it BREAKS THE DAY IT STOPS BEING TRUE. The ceiling is a product of two numbers,
``max_breaks_per_hour`` and ``_types.DEFAULT_BREAK_LENGTH_SECONDS``. Raise either
and the window cap becomes a live constraint; the tests below are the ones that
will fail and say so, rather than leaving a switched-on cap silently doing
nothing.

The day-fraction cap is deliberately NOT in this file. It has teeth on the same
plan, which is the asymmetry the pair must keep visible: one is a real constraint
held in reserve, the other is a lever that cannot currently fire.

Nothing here is a regulatory figure. The 40 minutes and the 20:00-24:00 bounds
are the figures under discussion, used as inputs to show they cannot be reached.
"""

from __future__ import annotations

from kairos.optimize._types import DEFAULT_BREAK_LENGTH_SECONDS
from kairos.optimize.guardrails import (
    AirtimeCaps,
    Guardrails,
    WindowAdCap,
    check_window_ad_load,
)
from kairos.optimize.optimizer import optimize_breaks
from tests.test_airtime_caps import ad_seconds_in, hour_segment

EVENING_HOURS = (20, 21, 22, 23)
DISCUSSED_LIMIT_SECONDS = 40 * 60.0     # the figure under discussion, as an input


def test_the_ceiling_is_the_product_of_two_numbers_and_sits_below_the_limit() -> None:
    """The arithmetic, pinned. This fails if either input moves."""
    guardrails = Guardrails()
    per_hour = guardrails.max_breaks_per_hour * DEFAULT_BREAK_LENGTH_SECONDS
    assert DEFAULT_BREAK_LENGTH_SECONDS == 120.0
    assert guardrails.max_breaks_per_hour == 4
    assert per_hour == 480.0                                  # 8 minutes an hour
    window_ceiling = per_hour * len(EVENING_HOURS)
    assert window_ceiling == 1920.0                           # 32 minutes
    assert window_ceiling < DISCUSSED_LIMIT_SECONDS, (
        "the window cap has become reachable; it is now a live constraint and "
        "the unreachability finding in WindowAdCap's docstring is out of date"
    )


def test_a_revenue_hungry_optimizer_cannot_reach_the_limit() -> None:
    """Not just arithmetic: the engine pushed as hard as it goes stops short."""
    segments = [hour_segment(h) for h in EVENING_HOURS]
    # Revenue-only, no retention penalty: the most ad time this engine will place.
    result = optimize_breaks(segments, Guardrails(), revenue_weight=1.0)
    placed = ad_seconds_in(result, 20, 24)
    assert placed == 1920.0, f"engine placed {placed}s, ceiling arithmetic says 1920"
    assert placed < DISCUSSED_LIMIT_SECONDS


def test_the_cap_switched_on_at_the_discussed_limit_removes_nothing() -> None:
    """Turn it on at 40 minutes and the plan is byte-for-byte the same plan."""
    segments = [hour_segment(h) for h in EVENING_HOURS]
    off = optimize_breaks(segments, Guardrails(), revenue_weight=1.0)
    on = optimize_breaks(
        segments,
        Guardrails(airtime_caps=AirtimeCaps(window=WindowAdCap(
            start_hour=20, end_hour=24,
            max_ad_seconds=DISCUSSED_LIMIT_SECONDS, enabled=True,
        ))),
        revenue_weight=1.0,
    )
    assert on.total_breaks == off.total_breaks
    assert ad_seconds_in(on, 20, 24) == ad_seconds_in(off, 20, 24)
    assert [(p.segment_id, p.start_seconds, p.duration_seconds) for p in on.placements] == \
           [(p.segment_id, p.start_seconds, p.duration_seconds) for p in off.placements]


def test_it_does_bite_once_set_below_the_ceiling() -> None:
    """It is unreachable, NOT broken. Below the ceiling it removes real breaks."""
    segments = [hour_segment(h) for h in EVENING_HOURS]
    off = optimize_breaks(segments, Guardrails(), revenue_weight=1.0)
    on = optimize_breaks(
        segments,
        Guardrails(airtime_caps=AirtimeCaps(window=WindowAdCap(
            start_hour=20, end_hour=24, max_ad_seconds=1200.0, enabled=True,
        ))),
        revenue_weight=1.0,
    )
    removed = off.total_breaks - on.total_breaks
    assert removed == 6, f"expected six breaks removed, removed {removed}"
    assert ad_seconds_in(on, 20, 24) == 1200.0


def test_raising_the_ceiling_makes_the_discussed_limit_fire() -> None:
    """The contingency, proven. "Removes nothing" is caused by the ceiling.

    Same 40-minute cap, same four hours, but a pod cap that lets an hour carry
    more than 8 minutes. The cap now removes breaks, which shows the null result
    above is a property of the CEILING and not of a cap that does nothing.
    """
    from dataclasses import replace

    raised = Guardrails(max_breaks_per_hour=8, min_break_spacing_seconds=0.0)
    segments = [replace(hour_segment(h), max_breaks=8) for h in EVENING_HOURS]

    off = optimize_breaks(segments, raised, revenue_weight=1.0)
    assert ad_seconds_in(off, 20, 24) > DISCUSSED_LIMIT_SECONDS, (
        "the raised ceiling must clear the discussed limit for this to mean anything"
    )
    on = optimize_breaks(
        segments,
        replace(raised, airtime_caps=AirtimeCaps(window=WindowAdCap(
            start_hour=20, end_hour=24,
            max_ad_seconds=DISCUSSED_LIMIT_SECONDS, enabled=True,
        ))),
        revenue_weight=1.0,
    )
    assert ad_seconds_in(on, 20, 24) <= DISCUSSED_LIMIT_SECONDS
    assert on.total_breaks < off.total_breaks


def test_the_real_plans_worst_evening_is_below_the_limit() -> None:
    """Measured on the shipped plan rather than on constructed segments.

    Skips rather than lies when the geometry cannot be rebuilt, and scopes to the
    operator's own channel so no rival figure is computed.
    """
    import pytest

    from kairos_api.channel_scope import operator_channel
    from kairos_api.core import _load_settings
    from kairos_api.plan_read_guardrails import plan_guardrail_items

    channel = str(operator_channel(_load_settings()) or "").strip()
    if not channel:
        pytest.skip("no operator channel is declared, so there is nothing to scope")
    items = [b for b in plan_guardrail_items() if str(b.channel).strip() == channel]
    if not items:
        pytest.skip("the break geometry does not join the saved plan")

    per_day: dict[str, float] = {}
    for item in items:
        if 20 <= item.hour < 24:
            per_day[item.day] = per_day.get(item.day, 0.0) + item.duration_seconds
    worst = max(per_day.values(), default=0.0)
    assert worst <= 1920.0, "an evening exceeded the ceiling this file pins"
    assert worst < DISCUSSED_LIMIT_SECONDS

    enforced = Guardrails(airtime_caps=AirtimeCaps(window=WindowAdCap(
        start_hour=20, end_hour=24,
        max_ad_seconds=DISCUSSED_LIMIT_SECONDS, enabled=True,
    )))
    assert check_window_ad_load(items, enforced) == [], (
        "the discussed evening limit now breaches on the real plan"
    )
