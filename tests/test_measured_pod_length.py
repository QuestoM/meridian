"""Planning on the pod length the channel airs, instead of a round number.

The optimizer has always planned in two-minute breaks. That figure was never
measured; it entered as "a common unit" and then decided what every regulatory
cap in the engine could mean. Measured against the broadcaster's own numbering
of its own log, the operator channel's pods average 190.7 seconds.

The gap is a currency error rather than a rounding error. Four pods an hour is
480 seconds to the engine and 763 on air, so the twelve-minute cap the plan
clears is breached, and the eight-minute cap protecting news and children's
programming is exceeded by half again: the plan is compliant in a unit that does
not exist. Measured on the real month, activating the measured length cuts the
pod count from 2,239 to 1,500, keeps the hourly cap clear in the broadcaster's
own seconds, and raises revenue 25.2 percent -- of which 6.5 points is the extra
airtime and the rest is better placement, because a constraint that finally
binds forces the optimizer to choose.

What these tests pin: that the measurement refuses to speak from too small or
implausible a sample, that the cap arithmetic is stated rather than implied, and
above all that the OFF state is untouched, because this ships off and every
existing number has to stay exactly where it was.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos import service
from kairos.model import pod_length
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.pricing import OptimizerAssumptions


def _spots(n_pods: int, spots_per_pod: int = 4, seconds_each: float = 45.0) -> pd.DataFrame:
    rows = []
    for pod in range(n_pods):
        for position in range(1, spots_per_pod + 1):
            rows.append({
                "Channel": "A",
                "air_dt": pd.Timestamp("2024-11-01") + pd.Timedelta(hours=pod, seconds=position * 60),
                "Duration": seconds_each,
                "Pos. Block 1": position,
                "Spots Block 1": spots_per_pod,
                "Spot type": "פרסומת",
                "TVR": 1.0,
            })
    return pd.DataFrame(rows)


# --- the measurement itself -----------------------------------------------------

def test_the_planning_value_is_the_mean_because_a_total_has_to_reproduce():
    """N pods times L has to equal the airtime a schedule really consumes, and
    only the mean satisfies that by construction. The median is the better
    description of a typical pod and the wrong basis for a cap."""
    frame = _spots(120, spots_per_pod=4, seconds_each=45.0)
    measured = pod_length.measure_pod_length(frame, channel="A")
    assert measured["usable"] is True
    assert measured["seconds"] == pytest.approx(180.0)
    assert measured["pods"] == 120
    assert measured["total_seconds"] == pytest.approx(measured["seconds"] * measured["pods"])


def test_too_few_pods_is_not_a_measurement():
    """One thin week of a minor channel must not be allowed to restate a plan's
    airtime. Refusing is the honest answer; a number is not."""
    measured = pod_length.measure_pod_length(_spots(10), channel="A")
    assert measured["usable"] is False
    assert measured["seconds"] is None
    assert "fewer than" in measured["reason"]


def test_an_implausible_mean_is_refused_rather_than_planned_on():
    """A pod of an hour is a parsing failure, not a broadcast."""
    measured = pod_length.measure_pod_length(
        _spots(120, spots_per_pod=4, seconds_each=900.0), channel="A")
    assert measured["usable"] is False
    assert measured["seconds"] is None
    assert "plausible" in measured["reason"]


def test_a_source_with_no_pods_answers_unusable_rather_than_raising():
    measured = pod_length.measure_pod_length(pd.DataFrame(), channel="A")
    assert measured["usable"] is False and measured["pods"] == 0


# --- the cap arithmetic, written down -------------------------------------------

def test_the_declared_length_is_why_the_seconds_cap_never_bit():
    """At 120 seconds a twelve-minute cap admits six pods, so the four-pod rule
    binds first and the seconds cap is inert. That is the whole reason a breach
    could hide."""
    reading = pod_length.cap_reading(120.0, Guardrails())
    assert reading["pods_under_the_general_cap"] == pytest.approx(6.0)
    assert reading["binding_rule"] == "pod count"
    assert reading["general_cap_breached_at_the_pod_ceiling"] is False
    assert reading["protected_cap_breached_at_the_pod_ceiling"] is False


def test_at_the_measured_length_the_seconds_cap_becomes_the_binding_rule():
    """And the pod ceiling the engine enforces turns out to breach both caps."""
    reading = pod_length.cap_reading(190.7, Guardrails())
    assert reading["seconds_at_the_pod_ceiling"] == pytest.approx(762.8)
    assert reading["pods_under_the_general_cap"] < 4
    assert reading["binding_rule"] == "ad seconds"
    assert reading["general_cap_breached_at_the_pod_ceiling"] is True
    assert reading["protected_cap_breached_at_the_pod_ceiling"] is True


def test_a_zero_length_pod_raises_rather_than_dividing():
    with pytest.raises(ValueError):
        pod_length.pods_per_hour_under_cap(0.0, 720.0)


# --- the gate: off is off -------------------------------------------------------

@pytest.mark.parametrize("settings", [
    None,
    {},
    {"operator_channel": "A"},
    {"operator_channel": "A", pod_length.ACTIVATION_SETTINGS_KEY: False},
    {"operator_channel": "A", pod_length.ACTIVATION_SETTINGS_KEY: "true"},
    {"operator_channel": "A", pod_length.ACTIVATION_SETTINGS_KEY: 1},
])
def test_the_off_state_returns_the_assumptions_object_untouched(settings):
    """Object identity, not merely equal values: nothing on this path may read
    the spots file, rebuild the assumptions, or move a number. The feature ships
    off, so every figure the operator already has must stay exactly where it is.
    A truthy-but-not-True value reads off, because activation that moves money is
    never inferred from a string."""
    assumptions = OptimizerAssumptions()
    assert service._apply_measured_pod_length(assumptions, settings) is assumptions


def test_activation_is_refused_when_the_measurement_is_not_usable():
    """The key alone is not enough; the data has to earn it too."""
    assert pod_length.measured_length_from_settings(
        {pod_length.ACTIVATION_SETTINGS_KEY: True},
        {"usable": False, "seconds": None},
    ) is None
    assert pod_length.measured_length_from_settings(
        {pod_length.ACTIVATION_SETTINGS_KEY: True},
        {"usable": True, "seconds": 190.7},
    ) == pytest.approx(190.7)
    # And the key must be present, whatever the measurement says.
    assert pod_length.measured_length_from_settings(
        {}, {"usable": True, "seconds": 190.7},
    ) is None


def test_an_unreadable_source_keeps_the_declared_default_rather_than_failing(monkeypatch):
    """A plan must not fail because a measurement could not be taken. It falls
    back to the declared number, which is what it used before this existed."""
    def boom(*_args, **_kwargs):
        raise OSError("the spots file is not readable here")

    monkeypatch.setattr(service, "load_spots", boom)
    assumptions = OptimizerAssumptions()
    out = service._apply_measured_pod_length(
        assumptions, {"operator_channel": "A", pod_length.ACTIVATION_SETTINGS_KEY: True})
    assert out is assumptions


def test_the_export_and_the_live_service_plan_on_one_length():
    """The saved plan and the live plan must never price different breaks.

    This is the failure ``_apply_first_break_multiplier`` was written to prevent
    and it applies identically here: the service folds the measured length into
    its assumptions, so an export that did not would keep writing two-minute
    breaks while the dashboard priced the measured ones, and the same day would
    carry two different revenues depending on which surface you asked.
    """
    from kairos.export import schedule as export

    settings_off = {"operator_channel": "רשת 13"}
    settings_on = {**settings_off, pod_length.ACTIVATION_SETTINGS_KEY: True}
    for settings in (settings_off, settings_on):
        assumptions = OptimizerAssumptions()
        from_service = service._apply_measured_pod_length(assumptions, settings)
        from_export = export._apply_measured_pod_length(assumptions, settings)
        assert (
            from_service.default_break_length_seconds
            == from_export.default_break_length_seconds
        ), f"the two surfaces disagreed on pod length for {settings}"
