"""The two optional caps, registered against the INERT LEVER guard.

tests/lever_probe.py names the defect class and already lists the hourly
ad-minutes cap as instance 2. These two caps are the same shape of thing, so
they are probed with the same guard rather than with a private assert, and the
asymmetry between them is recorded as two different verdicts:

  - the EVENING WINDOW cap is PINNED INERT at any limit at or above the ceiling.
    Four pods an hour at a hardcoded 120-second break length is 8 minutes, so a
    four-hour window cannot exceed 32; the 40-minute figure under discussion can
    never be reached. The pin fails the day that stops being true.
  - the DAY-FRACTION cap BITES. It is the one of the pair that is a real
    constraint held in reserve, and the guard proves it reaches the decision.

The probe REFUSES to rule on an unconstrained fixture, which is the trap this
piece hit independently: an evening measured on a day whose hours were empty
gave a comfortable pass that proved nothing. Every fixture below is sized so a
guardrail actually binds, and ``binds`` says which one.

Nothing here is a regulatory figure; the limits are inputs chosen to sit either
side of the measured ceiling.
"""

from __future__ import annotations

from kairos.optimize.guardrails import (
    AirtimeCaps,
    DayFractionAdCap,
    Guardrails,
    WindowAdCap,
)
from kairos.optimize.optimizer import optimize_breaks
from tests.lever_probe import (
    assert_lever_bites,
    assert_lever_is_inert,
    probe_lever,
)
from tests.test_airtime_caps import ad_seconds_in, hour_segment, total_ad_seconds

EVENING_HOURS = (20, 21, 22, 23)
EVENING_CEILING_SECONDS = 1920.0        # 4 hours x 4 pods x 120s = 32 minutes
DAY_ABSOLUTE_CAP_SECONDS = 9600.0       # the shipped max_daily_ad_minutes, 160


def _evening_ad_seconds(max_ad_minutes: float) -> float:
    """Run the evening with a window cap at ``max_ad_minutes`` and return its load."""
    result = optimize_breaks(
        [hour_segment(h) for h in EVENING_HOURS],
        Guardrails(airtime_caps=AirtimeCaps(window=WindowAdCap(
            start_hour=20, end_hour=24,
            max_ad_seconds=max_ad_minutes * 60.0, enabled=True,
        ))),
        revenue_weight=1.0,
    )
    return ad_seconds_in(result, 20, 24)


def _whole_day_ad_seconds(fraction: float) -> float:
    """Run a full day under a day-fraction cap and return the ad seconds placed."""
    result = optimize_breaks(
        [hour_segment(h) for h in range(24)],
        Guardrails(airtime_caps=AirtimeCaps(day_fraction=DayFractionAdCap(
            max_fraction_of_calendar_day=fraction, enabled=True,
        ))),
        revenue_weight=1.0,
    )
    return total_ad_seconds(result)


def test_the_evening_window_cap_is_pinned_inert_above_the_ceiling() -> None:
    """At or above 32 minutes the lever cannot reach the decision.

    The fixture binds on the pods-per-hour guardrail: every probed run sits
    exactly at the 32-minute ceiling, which is what makes this a measurement of
    the lever rather than of an empty evening.
    """
    probe = probe_lever(
        name="window ad cap at or above the pod-count ceiling",
        run=_evening_ad_seconds,
        settings=(33.0, 40.0, 55.0, 90.0),
        binds=lambda seconds: seconds == EVENING_CEILING_SECONDS,
    )
    assert_lever_is_inert(probe, because=(
        "max_breaks_per_hour=4 at a hardcoded 120s break length caps an hour at "
        "8 minutes, so four evening hours cannot exceed 32; measured 2026-08-09 "
        "on the shipped plan the worst operator evening is 32.00 minutes and a "
        "40-minute cap breaches on 0 of 30 channel-days"
    ))


def test_the_evening_window_cap_does_reach_the_decision_below_the_ceiling() -> None:
    """Inert is a property of the setting, not of the wiring.

    The same lever, probed below the ceiling, moves the output. Without this the
    pin above would be indistinguishable from a cap that was never wired up.
    """
    probe = probe_lever(
        name="window ad cap below the pod-count ceiling",
        run=_evening_ad_seconds,
        settings=(10.0, 16.0, 24.0, 30.0),
        binds=lambda seconds: seconds < EVENING_CEILING_SECONDS,
    )
    assert_lever_bites(probe)


def test_the_day_fraction_cap_reaches_the_decision() -> None:
    """The half of the pair with teeth.

    Every probed fraction is below the shipped absolute daily cap, so the new
    rule is provably the one binding rather than the old one.
    """
    probe = probe_lever(
        name="day fraction ad cap",
        run=_whole_day_ad_seconds,
        settings=(0.02, 0.05, 0.08, 0.10),
        binds=lambda seconds: seconds < DAY_ABSOLUTE_CAP_SECONDS,
    )
    assert_lever_bites(probe)


def test_the_probe_refuses_a_verdict_on_an_empty_evening() -> None:
    """The vacuity guard, exercised on the exact fixture that fooled this piece.

    A whole day handed to the greedy exhausts the absolute daily cap on the early
    hours, leaving 20:00-23:59 EMPTY. A window cap measured there looks inert and
    is telling you nothing about the cap. The probe must refuse rather than pass.
    """
    import pytest

    def empty_evening(max_ad_minutes: float) -> float:
        result = optimize_breaks(
            [hour_segment(h) for h in range(24)],
            Guardrails(airtime_caps=AirtimeCaps(window=WindowAdCap(
                start_hour=20, end_hour=24,
                max_ad_seconds=max_ad_minutes * 60.0, enabled=True,
            ))),
            revenue_weight=1.0,
        )
        return ad_seconds_in(result, 20, 24)

    assert empty_evening(40.0) == 0.0, "the fixture must be the empty one to matter"
    probe = probe_lever(
        name="window ad cap on an empty evening",
        run=empty_evening,
        settings=(10.0, 40.0),
        binds=lambda seconds: seconds == EVENING_CEILING_SECONDS,
    )
    with pytest.raises(AssertionError, match="UNCONSTRAINED fixture"):
        assert_lever_is_inert(probe, because="it would look inert here, wrongly")
