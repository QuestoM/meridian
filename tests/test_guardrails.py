"""Tests proving each Kairos guardrail actually fires (and stays quiet when safe)."""

from __future__ import annotations

from kairos.optimize.guardrails import (
    Break,
    Guardrails,
    check_break_spacing,
    check_breaks_per_hour,
    check_daily_ad_load,
    check_gold_breaks,
    check_hourly_ad_load,
    check_retention_floor,
    evaluate,
    is_compliant,
)

GR = Guardrails()


def make_break(**overrides) -> Break:
    base = dict(
        channel="קשת 12",
        day="Mon",
        hour=21,
        start_seconds=21 * 3600.0,
        duration_seconds=120.0,
        program_type="Drama",
        retention=0.85,
        is_gold=False,
    )
    base.update(overrides)
    return Break(**base)


def test_compliant_schedule_has_no_violations() -> None:
    breaks = [
        make_break(start_seconds=21 * 3600, duration_seconds=120),
        make_break(start_seconds=21 * 3600 + 900, duration_seconds=120),  # 15 min later
    ]
    assert evaluate(breaks, GR) == []
    assert is_compliant(breaks, GR)


def test_retention_floor_fires() -> None:
    breaks = [make_break(retention=0.6)]
    violations = check_retention_floor(breaks, GR)
    assert [v.code for v in violations] == ["retention_floor"]


def test_breaks_per_hour_fires_above_limit() -> None:
    breaks = [make_break(start_seconds=21 * 3600 + i * 60, duration_seconds=10) for i in range(5)]
    violations = check_breaks_per_hour(breaks, GR)
    assert violations and violations[0].code == "breaks_per_hour"
    assert violations[0].observed == 5


def test_hourly_ad_load_fires() -> None:
    # 7 breaks x 120s = 840s > 720s limit (also exceeds breaks/hour, but we test the load check).
    breaks = [make_break(start_seconds=21 * 3600 + i * 30, duration_seconds=120) for i in range(7)]
    violations = check_hourly_ad_load(breaks, GR)
    assert violations and violations[0].code == "hourly_ad_load"


def test_protected_program_has_tighter_hourly_limit() -> None:
    # 5 x 120s = 600s: fine for Drama (<=720) but over the 480s protected (News) limit.
    breaks = [make_break(program_type="News", start_seconds=21 * 3600 + i * 30, duration_seconds=120)
              for i in range(5)]
    violations = check_hourly_ad_load(breaks, GR)
    assert violations and violations[0].code == "hourly_ad_load"
    assert violations[0].limit == GR.protected_max_ad_seconds_per_hour


def test_break_spacing_fires_when_too_close() -> None:
    breaks = [
        make_break(start_seconds=21 * 3600, duration_seconds=120),
        make_break(start_seconds=21 * 3600 + 200, duration_seconds=120),  # 80s gap < 420s
    ]
    violations = check_break_spacing(breaks, GR)
    assert violations and violations[0].code == "break_spacing"


def test_daily_ad_load_fires() -> None:
    # 90 breaks x 120s = 10800s > 9600s daily limit, spread across hours.
    breaks = [make_break(hour=h, start_seconds=h * 3600 + (i % 2) * 1800, duration_seconds=120)
              for h in range(6, 24) for i in range(5)]
    violations = check_daily_ad_load(breaks, GR)
    assert violations and violations[0].code == "daily_ad_load"


def test_gold_breaks_fires_above_daily_max() -> None:
    breaks = [make_break(hour=20 + i, start_seconds=(20 + i) * 3600, is_gold=True) for i in range(4)]
    violations = check_gold_breaks(breaks, GR)
    assert violations and violations[0].code == "gold_breaks"
    assert violations[0].observed == 4


def test_evaluate_aggregates_multiple_violations() -> None:
    breaks = [
        make_break(retention=0.5),                                  # retention
        make_break(start_seconds=21 * 3600 + 30, duration_seconds=120),
    ]
    codes = {v.code for v in evaluate(breaks, GR)}
    assert "retention_floor" in codes
    assert "break_spacing" in codes  # second break only 30s-ish after the first
