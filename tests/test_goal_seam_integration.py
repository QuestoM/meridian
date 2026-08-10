"""Focused proofs that real goal orders reach the shared optimizer call site."""

from __future__ import annotations

from datetime import date

import pytest

from kairos.optimize import day_core, goal_seam
from kairos.optimize.goal_seam_integration import prepare_goal_inputs
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks
from kairos.optimize.revenue_net import segment_net_revenue

TODAY = date(2025, 5, 1)


def _segment(segment_id: str, rating: float) -> ProgramSegment:
    return ProgramSegment(
        segment_id=segment_id,
        channel="רשת 13",
        day="2025-05-01",
        start_seconds=20 * 3600 + (0 if segment_id == "high" else 1800),
        duration_seconds=1800,
        program_type="Drama",
        baseline_tvr=rating,
        cpp=100.0,
        impact_coefficient=-0.01,
    )


def _order() -> goal_seam.GoalOrder:
    return goal_seam.GoalOrder(
        campaign_id="CMP_REAL",
        channel="רשת 13",
        audience=goal_seam.ALL_VIEWERS,
        goal_points=20.0,
        starts_on="2025-05-01",
        ends_on="2025-05-05",
    )


def test_demo_only_store_keeps_the_shared_call_exactly_inert(monkeypatch) -> None:
    segments = [_segment("high", 6.0), _segment("low", 2.0)]
    captured = {}
    sentinel = object()

    monkeypatch.setattr(
        "kairos.service._assemble_demand_weights",
        lambda *_args, **_kwargs: {"high": 1.25, "low": 0.75},
    )
    monkeypatch.setattr(
        "kairos.service._constraint_inputs",
        lambda *_args, **_kwargs: (None, None),
    )

    def fake_optimize(*_args, **kwargs):
        captured.update(kwargs)
        return sentinel

    result = day_core._optimize_one_day(
        segments,
        guardrails=Guardrails(),
        revenue_weight=0.5,
        risk_lambda=0.0,
        pacing_today=TODAY,
        goal_orders=[],
        goal_delivered={},
        optimize_fn=fake_optimize,
    )

    assert result is sentinel
    assert captured["objective_mode"] == "blend"
    assert captured["demand_weights"] == {"high": 1.25, "low": 0.75}
    assert "net_of" not in captured


def test_real_goal_reaches_ranking_and_every_objective_tier(monkeypatch) -> None:
    segments = [_segment("high", 6.0), _segment("low", 2.0)]
    captured = {}

    monkeypatch.setattr(
        "kairos.service._assemble_demand_weights",
        lambda *_args, **_kwargs: {"high": 1.0, "low": 1.0},
    )
    monkeypatch.setattr(
        "kairos.service._constraint_inputs",
        lambda *_args, **_kwargs: (None, None),
    )

    def fake_optimize(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    day_core._optimize_one_day(
        segments,
        guardrails=Guardrails(),
        revenue_weight=0.5,
        risk_lambda=0.0,
        pacing_today=TODAY,
        goal_orders=[_order()],
        goal_delivered={},
        optimize_fn=fake_optimize,
    )

    assert captured["objective_mode"] == "revenue_net"
    assert captured["demand_weights"]["high"] > 1.0
    assert captured["demand_weights"]["low"] < 1.0
    goal_net = captured["net_of"]
    assert goal_net(_segment("high", 6.0), 1) > segment_net_revenue(
        _segment("high", 6.0), 1
    )


def test_unapplicable_goal_does_not_force_net_mode() -> None:
    segments = [_segment("high", 6.0), _segment("low", 2.0)]
    wrong_channel = goal_seam.GoalOrder(
        campaign_id="CMP_OTHER",
        channel="קשת 12",
        audience=goal_seam.ALL_VIEWERS,
        goal_points=20.0,
        starts_on="2025-05-01",
        ends_on="2025-05-05",
    )
    weights, mode, net_of = prepare_goal_inputs(
        segments,
        {"high": 1.0, "low": 1.0},
        TODAY,
        "blend",
        orders=[wrong_channel],
        delivered_of={},
    )
    assert weights == {"high": 1.0, "low": 1.0}
    assert mode == "blend"
    assert net_of is None


def test_optimizer_uses_the_supplied_net_scalar() -> None:
    segments = [_segment("high", 4.0), _segment("low", 4.0)]

    def only_high_has_value(segment: ProgramSegment, count: int) -> float:
        return count * (10.0 if segment.segment_id == "high" else -10.0)

    result = optimize_breaks(
        segments,
        Guardrails(),
        objective_mode="revenue_net",
        net_of=only_high_has_value,
    )
    counts = {row.segment_id: row.num_breaks for row in result.segments}
    assert counts["high"] > 0
    assert counts["low"] == 0


def test_invalid_objective_mode_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="objective_mode"):
        optimize_breaks([_segment("high", 4.0)], objective_mode="not-a-mode")
