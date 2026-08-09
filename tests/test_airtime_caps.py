"""The two optional airtime caps: absent by default, and provably biting when on.

The engine already carries one cap that cannot bind (``max_ad_minutes_per_hour``
is unreachable behind ``max_breaks_per_hour`` at the shipped break length), so a
new cap that merely exists is worth nothing. Every "it bites" test here is
therefore paired with a removal proof: the same property is re-asserted with the
rule neutered, and it must FAIL. A test whose failure mode is silence is not a
check.

Nothing in this file is a regulatory figure. Every window bound, minute count
and fraction is an input chosen to make the arithmetic legible.
"""

from __future__ import annotations

import pytest

from kairos.optimize import guardrails as guardrails_module
from kairos.optimize.guardrails import (
    CAP_ABSENT,
    CAP_AVAILABLE,
    CAP_ENFORCED,
    AirtimeCaps,
    Break,
    DayFractionAdCap,
    Guardrails,
    WindowAdCap,
    airtime_caps_from_mapping,
    cap_state,
    check_day_fraction_ad_load,
    check_window_ad_load,
    evaluate,
    is_compliant,
)
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks

CHANNEL = "רשת 13"
DAY = "2024-11-01"


def make_break(**overrides) -> Break:
    base = dict(
        channel=CHANNEL,
        day=DAY,
        hour=21,
        start_seconds=21 * 3600.0,
        duration_seconds=120.0,
        program_type="Drama",
        retention=0.85,
        is_gold=False,
    )
    base.update(overrides)
    return Break(**base)


def hour_segment(hour: int) -> ProgramSegment:
    """One hour of drama, able to hold four breaks."""
    return ProgramSegment(
        segment_id=f"s{hour:02d}",
        channel=CHANNEL,
        day=DAY,
        start_seconds=hour * 3600.0,
        duration_seconds=3600.0,
        program_type="Drama",
        baseline_tvr=10.0,
        cpp=1000.0,
        impact_coefficient=0.0,
        retention_baseline=1.0,
        premium=1.0,
        is_gold=False,
        max_breaks=4,
        break_length_seconds=120.0,
    )


def ad_seconds_in(result, low: int, high: int) -> float:
    """Ad seconds whose break starts fall in the half-open hour range."""
    return sum(
        p.duration_seconds for p in result.placements
        if low <= int(p.start_seconds // 3600) < high
    )


def total_ad_seconds(result) -> float:
    return sum(p.duration_seconds for p in result.placements)


# --------------------------------------------------------------------------
# Absent is absent: the default cannot move anything.
# --------------------------------------------------------------------------

def test_default_guardrails_have_both_caps_absent() -> None:
    caps = Guardrails().airtime_caps
    assert caps.window is None and caps.day_fraction is None
    assert caps.states() == {
        "window_ad_load": CAP_ABSENT,
        "day_fraction_ad_load": CAP_ABSENT,
    }


def test_absent_caps_contribute_no_violations() -> None:
    # Sixteen breaks in one evening, far past anything either cap would allow.
    breaks = [
        make_break(hour=h, start_seconds=h * 3600.0 + i * 720.0)
        for h in (20, 21, 22, 23) for i in range(4)
    ]
    assert check_window_ad_load(breaks, Guardrails()) == []
    assert check_day_fraction_ad_load(breaks, Guardrails()) == []
    assert [v.code for v in evaluate(breaks, Guardrails())
            if v.code in ("window_ad_load", "day_fraction_ad_load")] == []


def test_unset_is_not_a_cap_of_zero() -> None:
    """An absent cap permits everything; a zero cap forbids everything."""
    breaks = [make_break()]
    assert check_window_ad_load(breaks, Guardrails()) == []
    zero = Guardrails(airtime_caps=AirtimeCaps(
        window=WindowAdCap(start_hour=0, end_hour=24, max_ad_seconds=0.0, enabled=True),
    ))
    assert [v.code for v in check_window_ad_load(breaks, zero)] == ["window_ad_load"]


def test_configured_but_disabled_cap_is_still_a_no_op() -> None:
    """Available is not enforced: switching a cap off must fully disarm it."""
    breaks = [make_break(hour=20, start_seconds=20 * 3600.0 + i * 700.0) for i in range(8)]
    off = Guardrails(airtime_caps=AirtimeCaps(
        window=WindowAdCap(start_hour=20, end_hour=24, max_ad_seconds=120.0, enabled=False),
        day_fraction=DayFractionAdCap(max_fraction=0.001, enabled=False),
    ))
    assert check_window_ad_load(breaks, off) == []
    assert check_day_fraction_ad_load(breaks, off) == []
    assert off.airtime_caps.states() == {
        "window_ad_load": CAP_AVAILABLE,
        "day_fraction_ad_load": CAP_AVAILABLE,
    }


def test_three_states_are_distinguishable() -> None:
    cap = WindowAdCap(start_hour=20, end_hour=24, max_ad_seconds=600.0, enabled=False)
    assert cap_state(None) == CAP_ABSENT
    assert cap_state(cap) == CAP_AVAILABLE
    assert cap_state(WindowAdCap(20, 24, 600.0, enabled=True)) == CAP_ENFORCED


# --------------------------------------------------------------------------
# The window cap bites, and the bite is attributable to the rule.
# --------------------------------------------------------------------------

EVENING = (20, 21, 22, 23)
WINDOW_LIMIT_SECONDS = 1200.0   # ten breaks' worth, chosen to be legible


def evening_plan(caps: AirtimeCaps):
    return optimize_breaks(
        [hour_segment(h) for h in EVENING],
        Guardrails(airtime_caps=caps),
        revenue_weight=1.0,
    )


def test_window_cap_changes_a_plan_and_the_count_is_exact() -> None:
    baseline = evening_plan(AirtimeCaps())
    capped = evening_plan(AirtimeCaps(window=WindowAdCap(
        start_hour=20, end_hour=24, max_ad_seconds=WINDOW_LIMIT_SECONDS, enabled=True,
    )))

    baseline_seconds = ad_seconds_in(baseline, 20, 24)
    capped_seconds = ad_seconds_in(capped, 20, 24)

    # Baseline fills to four breaks in each of the four hours.
    assert baseline_seconds == 4 * 4 * 120.0
    # The cap binds exactly, and removes the difference in whole breaks.
    assert capped_seconds <= WINDOW_LIMIT_SECONDS
    assert capped_seconds < baseline_seconds
    removed = int((baseline_seconds - capped_seconds) / 120.0)
    assert removed == 6, f"expected six breaks removed, removed {removed}"
    assert baseline.total_breaks - capped.total_breaks == removed
    assert capped.is_compliant


def test_window_cap_bite_disappears_when_the_rule_is_removed(monkeypatch) -> None:
    """The removal proof: neuter the check and the capped plan breaches the cap.

    ``is_compliant`` resolves its check functions from module globals at call
    time, so patching the module genuinely removes the rule from the engine
    rather than only from this test's view of it.
    """
    monkeypatch.setattr(
        guardrails_module, "check_window_ad_load", lambda breaks, guardrails: [],
    )
    capped = evening_plan(AirtimeCaps(window=WindowAdCap(
        start_hour=20, end_hour=24, max_ad_seconds=WINDOW_LIMIT_SECONDS, enabled=True,
    )))
    # With the rule gone the optimizer fills the evening again, so the very
    # assertion the bite test relies on now fails. That is the proof it bites.
    assert ad_seconds_in(capped, 20, 24) > WINDOW_LIMIT_SECONDS


def test_window_cap_counts_a_break_by_its_own_start() -> None:
    """A break outside the window is not charged to it, and vice versa."""
    cap = Guardrails(airtime_caps=AirtimeCaps(window=WindowAdCap(
        start_hour=20, end_hour=24, max_ad_seconds=120.0, enabled=True,
    )))
    inside = [make_break(hour=20, start_seconds=20 * 3600.0),
              make_break(hour=23, start_seconds=23 * 3600.0)]
    assert [v.code for v in check_window_ad_load(inside, cap)] == ["window_ad_load"]
    # 19:59 is before the window and hour 24 (past midnight) is after it.
    outside = [make_break(hour=19, start_seconds=19 * 3600.0 + 3540.0),
               make_break(hour=24, start_seconds=24 * 3600.0)]
    assert check_window_ad_load(outside, cap) == []


def test_window_cap_is_scoped_per_channel_day() -> None:
    """Two days each at the limit are compliant; the cap is not a total."""
    cap = Guardrails(airtime_caps=AirtimeCaps(window=WindowAdCap(
        start_hour=20, end_hour=24, max_ad_seconds=240.0, enabled=True,
    )))
    breaks = [
        make_break(day="2024-11-01", hour=20, start_seconds=20 * 3600.0),
        make_break(day="2024-11-01", hour=21, start_seconds=21 * 3600.0),
        make_break(day="2024-11-02", hour=20, start_seconds=20 * 3600.0),
        make_break(day="2024-11-02", hour=21, start_seconds=21 * 3600.0),
    ]
    assert check_window_ad_load(breaks, cap) == []


# --------------------------------------------------------------------------
# The day-fraction cap bites, and the bite is attributable to the rule.
# --------------------------------------------------------------------------

DAY_FRACTION = 0.05   # 4320s of a calendar day: below the 9600s absolute cap,
                      # so the new rule is provably the one that binds.


def whole_day_plan(caps: AirtimeCaps):
    return optimize_breaks(
        [hour_segment(h) for h in range(24)],
        Guardrails(airtime_caps=caps),
        revenue_weight=1.0,
    )


def test_day_fraction_cap_changes_a_plan_and_the_count_is_exact() -> None:
    baseline = whole_day_plan(AirtimeCaps())
    capped = whole_day_plan(AirtimeCaps(day_fraction=DayFractionAdCap(
        max_fraction=DAY_FRACTION, enabled=True,
    )))

    baseline_seconds = total_ad_seconds(baseline)
    capped_seconds = total_ad_seconds(capped)
    limit = DAY_FRACTION * 86400.0

    # The absolute daily cap is what holds the baseline back, at 9600s.
    assert baseline_seconds == Guardrails().max_daily_ad_seconds
    assert limit < baseline_seconds, "the fraction must bind before the absolute cap"

    assert capped_seconds <= limit
    removed = int((baseline_seconds - capped_seconds) / 120.0)
    assert removed == 44, f"expected forty-four breaks removed, removed {removed}"
    assert baseline.total_breaks - capped.total_breaks == removed
    assert capped.is_compliant


def test_day_fraction_bite_disappears_when_the_rule_is_removed(monkeypatch) -> None:
    """The removal proof for the day-fraction cap."""
    monkeypatch.setattr(
        guardrails_module, "check_day_fraction_ad_load", lambda breaks, guardrails: [],
    )
    capped = whole_day_plan(AirtimeCaps(day_fraction=DayFractionAdCap(
        max_fraction=DAY_FRACTION, enabled=True,
    )))
    assert total_ad_seconds(capped) > DAY_FRACTION * 86400.0


def test_day_fraction_denominator_is_configurable() -> None:
    """A shorter broadcast day makes the same fraction a tighter cap."""
    breaks = [make_break(hour=20, start_seconds=20 * 3600.0 + i * 700.0) for i in range(5)]
    full = Guardrails(airtime_caps=AirtimeCaps(day_fraction=DayFractionAdCap(
        max_fraction=0.01, day_seconds=86400.0, enabled=True,
    )))
    short = Guardrails(airtime_caps=AirtimeCaps(day_fraction=DayFractionAdCap(
        max_fraction=0.01, day_seconds=43200.0, enabled=True,
    )))
    assert check_day_fraction_ad_load(breaks, full) == []          # 600s <= 864s
    assert [v.code for v in check_day_fraction_ad_load(breaks, short)] == [
        "day_fraction_ad_load"]                                     # 600s > 432s


def test_both_caps_reach_is_compliant() -> None:
    """The caps are wired into the gate the optimizer actually consults."""
    breaks = [make_break(hour=20, start_seconds=20 * 3600.0 + i * 700.0) for i in range(3)]
    assert is_compliant(breaks, Guardrails())
    window_on = Guardrails(airtime_caps=AirtimeCaps(window=WindowAdCap(
        start_hour=20, end_hour=24, max_ad_seconds=120.0, enabled=True)))
    day_on = Guardrails(airtime_caps=AirtimeCaps(day_fraction=DayFractionAdCap(
        max_fraction=0.001, enabled=True)))
    assert not is_compliant(breaks, window_on)
    assert not is_compliant(breaks, day_on)


# --------------------------------------------------------------------------
# Settings translation.
# --------------------------------------------------------------------------

def test_mapping_translation_leaves_unmentioned_caps_absent() -> None:
    assert airtime_caps_from_mapping(None).states() == {
        "window_ad_load": CAP_ABSENT, "day_fraction_ad_load": CAP_ABSENT}
    assert airtime_caps_from_mapping({}).states() == {
        "window_ad_load": CAP_ABSENT, "day_fraction_ad_load": CAP_ABSENT}
    assert airtime_caps_from_mapping({"max_ad_minutes_per_hour": 12.0}).states() == {
        "window_ad_load": CAP_ABSENT, "day_fraction_ad_load": CAP_ABSENT}


def test_mapping_translation_converts_minutes_to_seconds() -> None:
    caps = airtime_caps_from_mapping({
        "window_ad_cap": {
            "enabled": True, "start_hour": 20, "end_hour": 24, "max_ad_minutes": 40.0,
        },
    })
    assert caps.window is not None
    assert caps.window.max_ad_seconds == 2400.0
    assert caps.window.enabled is True


def test_settings_model_defaults_to_absent_on_every_path() -> None:
    from kairos.service import guardrails_from_settings
    from kairos_api.core import KairosSettings, _settings_to_guardrails

    absent = {"window_ad_load": CAP_ABSENT, "day_fraction_ad_load": CAP_ABSENT}
    assert _settings_to_guardrails(KairosSettings()).airtime_caps.states() == absent
    assert guardrails_from_settings({}).airtime_caps.states() == absent


def test_an_operator_can_turn_a_cap_on_from_stored_json() -> None:
    """The ruling's actual requirement: the ability to switch one on when asked.

    Exercises the whole path a stored settings file takes, from raw JSON through
    the settings model to a guardrail set the optimizer would enforce, without
    touching the shipped settings file.
    """
    import json

    from kairos_api.core import KairosSettings, _settings_to_guardrails

    stored = json.loads(json.dumps({
        "window_ad_cap": {
            "enabled": True, "start_hour": 20, "end_hour": 24, "max_ad_minutes": 40.0,
        },
        "day_fraction_ad_cap": {"enabled": True, "max_fraction_of_day": 0.1},
    }))
    guardrails = _settings_to_guardrails(KairosSettings(**stored))
    assert guardrails.airtime_caps.states() == {
        "window_ad_load": CAP_ENFORCED, "day_fraction_ad_load": CAP_ENFORCED}
    assert guardrails.airtime_caps.window.max_ad_seconds == 2400.0
    assert guardrails.airtime_caps.day_fraction.max_ad_seconds == 8640.0
    # The hourly cap is untouched by any of this.
    assert guardrails.max_ad_seconds_per_hour == 720.0

    # And the enforced cap really gates. Four pods an hour across four hours is
    # 32 minutes, which is UNDER this 40-minute window, so it must pass; a load
    # above the window must not. Asserting through the window check alone keeps
    # the verdict attributable to this rule rather than to the pods-per-hour cap.
    evening = [make_break(hour=h, start_seconds=h * 3600.0 + i * 720.0)
               for h in (20, 21, 22, 23) for i in range(4)]
    assert sum(b.duration_seconds for b in evening) == 1920.0
    assert check_window_ad_load(evening, guardrails) == []

    heavy = [make_break(hour=20 + i // 6, start_seconds=(20 + i // 6) * 3600.0 + (i % 6) * 500.0)
             for i in range(21)]                                  # 42 minutes
    assert sum(b.duration_seconds for b in heavy) == 2520.0
    assert [v.code for v in check_window_ad_load(heavy, guardrails)] == ["window_ad_load"]


def test_a_half_written_cap_is_refused_not_silently_completed() -> None:
    from kairos_api.airtime_cap_settings import WindowAdCapSettings

    with pytest.raises(Exception):
        WindowAdCapSettings(enabled=True, start_hour=20)          # no end, no limit
    with pytest.raises(Exception):
        WindowAdCapSettings(enabled=True, start_hour=22, end_hour=20, max_ad_minutes=40)
