"""Tests for the manual override layer.

These prove the four properties that matter: the OverrideSet reads the CSV into
honest segment and spot constraints; the optimizer treats segment overrides as
hard constraints (pin / force / forbid / gold); an infeasible override lands in
rejected_overrides instead of breaching a guardrail; and the daily pricing path
honors spot lock and move.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from kairos.optimize.advertiser_rules import AdvertiserRuleEngine, Baseline, Condition
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks
from kairos.optimize.overrides import (
    Override,
    OverrideSet,
)


# --- OverrideSet parsing ----------------------------------------------------


def _write_overrides(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["override_id", "scope", "target_id", "kind", "value", "gold", "notes", "created_at"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def test_from_csv_empty_file_yields_no_constraints(tmp_path: Path) -> None:
    path = tmp_path / "manual_overrides.csv"
    _write_overrides(path, [])
    overrides = OverrideSet.from_csv(path)
    assert overrides.segment_constraints() == {}
    assert overrides.spot_overrides() == {}


def test_from_csv_missing_file_yields_no_constraints(tmp_path: Path) -> None:
    overrides = OverrideSet.from_csv(tmp_path / "does_not_exist.csv")
    assert overrides.segment_constraints() == {}


def test_segment_constraints_parses_each_kind(tmp_path: Path) -> None:
    path = tmp_path / "manual_overrides.csv"
    _write_overrides(path, [
        {"override_id": "o1", "scope": "segment", "target_id": "s1", "kind": "pin", "value": "2"},
        {"override_id": "o2", "scope": "segment", "target_id": "s2", "kind": "force", "value": "3"},
        {"override_id": "o3", "scope": "segment", "target_id": "s3", "kind": "forbid"},
        {"override_id": "o4", "scope": "segment", "target_id": "s4", "kind": "gold"},
        {"override_id": "o5", "scope": "segment", "target_id": "s5", "kind": "force", "value": "1", "gold": "true"},
    ])
    constraints = OverrideSet.from_csv(path).segment_constraints()
    assert constraints["s1"] == {"pin": 2}
    assert constraints["s2"] == {"min": 3}
    assert constraints["s3"] == {"forbid": True}
    assert constraints["s4"] == {"gold": True}
    assert constraints["s5"] == {"min": 1, "gold": True}


def test_spot_overrides_parses_lock_and_move(tmp_path: Path) -> None:
    path = tmp_path / "manual_overrides.csv"
    _write_overrides(path, [
        {"override_id": "o1", "scope": "spot", "target_id": "ADV|CMP|2026-06-14|3", "kind": "lock"},
        {"override_id": "o2", "scope": "spot", "target_id": "ADV|CMP|2026-06-14|2", "kind": "move", "value": "position=1;daypart=prime"},
    ])
    spots = OverrideSet.from_csv(path).spot_overrides()
    assert spots["ADV|CMP|2026-06-14|3"] == {"lock": True}
    assert spots["ADV|CMP|2026-06-14|2"]["move"] == {"position": "1", "daypart": "prime"}


def test_unknown_kind_is_dropped() -> None:
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id="s1", kind="bogus", value="2"),
    ])
    assert overrides.segment_constraints() == {}


# --- optimizer honors segment overrides -------------------------------------


def _segment(segment_id: str, **overrides) -> ProgramSegment:
    base = dict(
        segment_id=segment_id,
        channel="קשת 12",
        day="Mon",
        start_seconds=21 * 3600.0,
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
    base.update(overrides)
    return ProgramSegment(**base)


def test_forbid_holds_segment_at_zero() -> None:
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id="s1", kind="forbid"),
    ])
    # revenue_weight 1.0 would normally fill the segment; forbid keeps it at 0.
    result = optimize_breaks([_segment("s1")], Guardrails(), revenue_weight=1.0, overrides=overrides)
    assert result.segments[0].num_breaks == 0
    assert not result.rejected_overrides


def test_pin_fixes_the_break_count() -> None:
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id="s1", kind="pin", value="2"),
    ])
    # revenue_weight 0.0 protects retention and would place nothing; the pin
    # forces exactly 2 regardless.
    result = optimize_breaks(
        [_segment("s1", impact_coefficient=-0.05)], Guardrails(),
        revenue_weight=0.0, overrides=overrides,
    )
    assert result.segments[0].num_breaks == 2


def test_force_floors_the_break_count() -> None:
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id="s1", kind="force", value="2"),
    ])
    result = optimize_breaks(
        [_segment("s1", impact_coefficient=-0.05)], Guardrails(),
        revenue_weight=0.0, overrides=overrides,
    )
    # At least 2 breaks, even though retention-only would place none.
    assert result.segments[0].num_breaks >= 2


def test_gold_marks_the_placements() -> None:
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id="s1", kind="force", value="1", gold=True),
    ])
    result = optimize_breaks([_segment("s1")], Guardrails(), revenue_weight=1.0, overrides=overrides)
    assert result.segments[0].num_breaks >= 1
    assert all(placement.is_gold for placement in result.placements)


def test_infeasible_force_is_rejected_not_applied() -> None:
    # force 6 breaks on a segment whose max_breaks is 4 is infeasible.
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id="s1", kind="force", value="6"),
    ])
    result = optimize_breaks([_segment("s1", max_breaks=4)], Guardrails(), revenue_weight=1.0, overrides=overrides)
    assert result.segments[0].num_breaks <= 4
    assert len(result.rejected_overrides) == 1
    rejected = result.rejected_overrides[0]
    assert rejected.segment_id == "s1"
    assert rejected.kind == "force"
    assert "max_breaks" in rejected.reason
    # The plan still respects every guardrail.
    assert result.is_compliant


def test_pin_breaching_spacing_is_rejected() -> None:
    # A 6-minute programme cannot hold 3 spaced breaks under the 7-minute
    # spacing guardrail; pinning 3 must be rejected, not shipped out-of-policy.
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id="s1", kind="pin", value="3"),
    ])
    segment = _segment("s1", duration_seconds=360.0, max_breaks=5)
    result = optimize_breaks([segment], Guardrails(), revenue_weight=1.0, overrides=overrides)
    assert result.is_compliant
    assert any(r.kind == "pin" for r in result.rejected_overrides)


# --- daily pricing honors spot lock and move --------------------------------


def _daily_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "advertiser": "ADV", "campaign": "CMP", "program": "News at Nine",
            "position_in_break": 3, "planned_tvr": 5.0, "duration_sec": 30.0,
            "pricing_type": "CPP", "price": None, "spot_time": "21:00",
            "date": "2026-06-14",
        },
    ])


def _classifier_stub():
    from dataclasses import dataclass

    @dataclass
    class _Genre:
        category: str

    class _Stub:
        def classify(self, program):
            return _Genre(category="News")

    return _Stub()


def _pricing_stub():
    from kairos.optimize.pricing import PricingModel

    return PricingModel(base_price_per_second_per_tvr_point=1000.0)


def test_lock_keeps_a_forbidden_spot(tmp_path: Path) -> None:
    from kairos.export.spots import price_daily_spots

    # An advertiser rule that forbids the spot.
    engine = AdvertiserRuleEngine(
        conditions={"ADV": [Condition("ADV", "r1", "forbid")]},
    )
    daily = _daily_frame()
    classifier = _classifier_stub()
    pricing = _pricing_stub()

    # Without the lock, the spot is dropped.
    plain = price_daily_spots(daily, engine=engine, pricing=pricing, classifier=classifier)
    assert len(plain.priced) == 0
    assert len(plain.dropped) == 1

    # With a lock override, the spot passes through untouched.
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="spot", target_id="ADV|CMP|2026-06-14|3", kind="lock"),
    ])
    locked = price_daily_spots(
        daily, engine=engine, pricing=pricing, classifier=classifier, overrides=overrides,
    )
    assert len(locked.priced) == 1
    assert len(locked.dropped) == 0


def test_move_retags_position_before_pricing(tmp_path: Path) -> None:
    from kairos.export.spots import price_daily_spots

    # Premium only on position 1; the spot is at position 3 and would not get it.
    engine = AdvertiserRuleEngine(
        baselines={"ADV": Baseline("ADV", default_premium=1.0)},
        conditions={"ADV": [
            Condition("ADV", "r1", "premium", value=2.0, scope_positions=frozenset({"1"})),
        ]},
    )
    daily = _daily_frame()
    classifier = _classifier_stub()
    pricing = _pricing_stub()

    plain = price_daily_spots(daily, engine=engine, pricing=pricing, classifier=classifier)
    assert plain.priced[0].position == 3
    assert plain.priced[0].premium == pytest.approx(1.0)

    # Move it to position 1; now the position-1 premium applies.
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="spot", target_id="ADV|CMP|2026-06-14|3", kind="move", value="position=1"),
    ])
    moved = price_daily_spots(
        daily, engine=engine, pricing=pricing, classifier=classifier, overrides=overrides,
    )
    assert moved.priced[0].position == 1
    assert moved.priced[0].premium == pytest.approx(2.0)
