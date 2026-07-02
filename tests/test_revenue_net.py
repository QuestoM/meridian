"""Unit tests for the ILS retention-cost monetization (kairos.optimize.revenue_net).

Every expected value here is hand-computed from the documented formula so the
test pins the economics, not just the code path:

    lost_tvr           = baseline_tvr * (1 - retention_share)
    retention_cost_ils = base_rate * lost_tvr * (ad_seconds / unit_seconds)
    revenue_net_ils    = revenue_ils - retention_cost_ils

The synthetic fixtures use round numbers (unit_seconds = 1.0, as the real
per-second rate uses) so the arithmetic is checkable by eye. The golden weekly
schedule (tests/golden_weekly_schedule.py) proves the blend default is unchanged;
this file proves the monetization and the opt-in net objective.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # pytest when it is installed (CI).
    import pytest
except ImportError:  # pragma: no cover - lets this file run under the plain venv
    import contextlib

    class _Approx:
        def __init__(self, expected: float, rel: float | None = None, abs: float | None = None):
            self.expected = float(expected)
            self.rel = rel
            self.abs = abs if abs is not None else (None if rel is not None else 1e-6)

        def __eq__(self, other: object) -> bool:
            other = float(other)  # type: ignore[arg-type]
            tol = 0.0
            if self.abs is not None:
                tol = max(tol, self.abs)
            if self.rel is not None:
                tol = max(tol, self.rel * max(abs(self.expected), abs(other)))
            return abs(other - self.expected) <= tol

    class _Pytest:
        @staticmethod
        def approx(expected: float, rel: float | None = None, abs: float | None = None) -> _Approx:
            return _Approx(expected, rel=rel, abs=abs)

        @staticmethod
        @contextlib.contextmanager
        def raises(exc, match: str | None = None):
            raised: dict[str, BaseException] = {}
            try:
                yield raised
            except exc as caught:  # noqa: B902
                raised["value"] = caught
                if match is not None and match not in str(caught):
                    raise AssertionError(f"'{match}' not in '{caught}'") from None
                return
            raise AssertionError(f"{exc.__name__} was not raised")

    pytest = _Pytest()  # type: ignore

from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks
from kairos.optimize.revenue_net import (
    RETENTION_COST_FORMULA,
    compare_objectives,
    frame_revenue_net,
    plan_revenue_net,
    retention_cost_ils,
    segment_net_revenue,
)


def _seg(seg_id: str, *, baseline_tvr: float, cpp: float, premium: float, coef: float) -> ProgramSegment:
    """A minimal real-data-shaped segment (per-second rate: unit_seconds = 1.0)."""
    return ProgramSegment(
        segment_id=seg_id,
        channel="c",
        day="2024-01-01",
        start_seconds=3600.0,
        duration_seconds=1800.0,
        program_type="Movie",
        baseline_tvr=baseline_tvr,
        cpp=cpp,
        premium=premium,
        unit_seconds=1.0,
        impact_coefficient=coef,
        retention_baseline=1.0,
        max_breaks=4,
        break_length_seconds=120.0,
    )


# 1. retention_cost_ils: the core formula, hand-computed -----------------------
def test_retention_cost_hand_computed() -> None:
    # baseline_tvr = 2.0, retention = 0.75 -> lost = 0.5 rating points.
    # base_rate = 50 ILS/point-second, ad_seconds = 120 -> cost = 50 * 0.5 * 120.
    cost = retention_cost_ils(
        baseline_tvr=2.0, retention_share=0.75, base_rate=50.0, ad_seconds=120.0
    )
    assert cost == pytest.approx(3000.0)


def test_retention_cost_zero_when_audience_fully_kept() -> None:
    # retention_share = 1.0 sheds nothing, so the cost is exactly zero.
    assert retention_cost_ils(
        baseline_tvr=5.0, retention_share=1.0, base_rate=99.0, ad_seconds=480.0
    ) == 0.0


def test_retention_cost_zero_without_ad_seconds() -> None:
    # No ad seconds -> no break aired -> nothing lost, nothing earned.
    assert retention_cost_ils(
        baseline_tvr=5.0, retention_share=0.5, base_rate=99.0, ad_seconds=0.0
    ) == 0.0


def test_retention_cost_share_clamped() -> None:
    # A share above 1.0 is clamped (never a negative "cost" that would add revenue).
    assert retention_cost_ils(
        baseline_tvr=2.0, retention_share=1.2, base_rate=50.0, ad_seconds=120.0
    ) == 0.0


def test_retention_cost_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        retention_cost_ils(baseline_tvr=-1.0, retention_share=0.5, base_rate=50.0, ad_seconds=120.0)
    with pytest.raises(ValueError):
        retention_cost_ils(baseline_tvr=1.0, retention_share=0.5, base_rate=50.0, ad_seconds=120.0, unit_seconds=0.0)


# 2. segment_net_revenue: revenue minus cost at k breaks, hand-computed --------
def test_segment_net_revenue_hand_computed() -> None:
    # coef = -0.05, so retention at k=1 is 1 + (-0.05)*1 = 0.95.
    # base_rate = 50 * 1.0 = 50; ad_seconds = 1 * 120 = 120.
    # revenue = 50 * (2.0 * 0.95) * 120 = 11400.
    # cost    = 50 * (2.0 * 0.05) * 120 = 600.
    # net     = 11400 - 600 = 10800.
    seg = _seg("s", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05)
    assert segment_net_revenue(seg, 1) == pytest.approx(10800.0)
    assert segment_net_revenue(seg, 0) == 0.0


# 3. plan_revenue_net on a real optimizer plan --------------------------------
def test_plan_revenue_net_reconciles_and_discloses_basis() -> None:
    segs = [
        _seg("d|c|000", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05),
        _seg("d|c|001", baseline_tvr=1.0, cpp=50.0, premium=1.2, coef=-0.10),
    ]
    plan = optimize_breaks(segs, Guardrails(), revenue_weight=1.0, refine=False)
    money = plan_revenue_net(plan, segments=segs)

    assert money["available"] is True
    # Revenue reconciles to the plan's own reported total (same inputs).
    assert money["revenue_ils"] == pytest.approx(round(plan.total_revenue, 2), abs=0.01)
    # Net is revenue minus a non-negative cost, so net <= revenue.
    assert money["retention_cost_ils"] >= 0.0
    assert money["revenue_net_ils"] == pytest.approx(
        round(money["revenue_ils"] - money["retention_cost_ils"], 2), abs=0.01
    )
    # Basis disclosure ships with the number (Law 9).
    basis = money["basis"]
    assert basis["source"] == "modeled"
    assert basis["formula"] == RETENTION_COST_FORMULA
    assert {"baseline_tvr", "retention_share", "base_rate", "ad_seconds"} <= set(basis["inputs"])


def test_plan_revenue_net_manual_total() -> None:
    # A single revenue-only segment, hand-computed per break (each break is valued
    # at the retention that holds once it is present, exactly as the optimizer does).
    seg = _seg("d|c|000", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05)
    plan = optimize_breaks([seg], Guardrails(), revenue_weight=1.0, refine=False)
    k = plan.segments[0].num_breaks
    # revenue = sum over breaks j of base_rate * tvr * retention(j) * break_length.
    # cost    = gross (full audience) minus that delivered revenue.
    expected_revenue = sum(
        50.0 * 2.0 * (1.0 + (-0.05) * j) * 120.0 for j in range(1, k + 1)
    )
    gross = 50.0 * 2.0 * (k * 120.0)
    expected_cost = gross - expected_revenue
    money = plan_revenue_net(plan, segments=[seg])
    assert money["revenue_ils"] == pytest.approx(round(expected_revenue, 2), abs=0.02)
    assert money["retention_cost_ils"] == pytest.approx(round(expected_cost, 2), abs=0.02)
    assert money["revenue_net_ils"] == pytest.approx(
        round(expected_revenue - expected_cost, 2), abs=0.02
    )


def test_plan_revenue_net_without_segments_is_honestly_unavailable() -> None:
    seg = _seg("d|c|000", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05)
    plan = optimize_breaks([seg], Guardrails(), revenue_weight=1.0, refine=False)
    money = plan_revenue_net(plan)  # no segments -> cannot value audience
    assert money["available"] is False
    assert "baseline_tvr" in money["reason"]
    assert money["retention_cost_ils"] is None
    # Revenue is still reported honestly from the plan.
    assert money["revenue_ils"] == pytest.approx(round(plan.total_revenue, 2), abs=0.01)


# 4. frame_revenue_net: exact via baseline_tvr, honest-unavailable without it --
def test_frame_revenue_net_matches_hand_computation() -> None:
    # Rows carrying baseline_tvr. cost = gross - revenue, gross = rate*tvr*secs.
    # Row A: tvr 2.0, base_rate 50, secs 120 -> gross 12000, rev 11400, cost 600.
    # Row B: tvr 1.0, base_rate 60, secs 240 -> gross 14400, rev 12960, cost 1440.
    frame = pd.DataFrame(
        {
            "predicted_revenue": [11400.0, 12960.0],
            "baseline_tvr": [2.0, 1.0],
            "base_rate": [50.0, 60.0],
            "total_break_time": [120.0, 240.0],
        }
    )
    money = frame_revenue_net(frame)
    assert money["available"] is True
    assert money["revenue_ils"] == pytest.approx(11400.0 + 12960.0, abs=0.01)
    assert money["retention_cost_ils"] == pytest.approx(600.0 + 1440.0, abs=0.02)
    assert money["revenue_net_ils"] == pytest.approx(
        (11400.0 + 12960.0) - (600.0 + 1440.0), abs=0.02
    )
    assert money["basis"]["source"] == "modeled"


def test_frame_revenue_net_without_baseline_tvr_is_unavailable() -> None:
    # The current saved CSV shape: has retention_used but not baseline_tvr. The
    # final share cannot recover the cost without bias, so it must refuse honestly.
    frame = pd.DataFrame(
        {
            "predicted_revenue": [11400.0],
            "retention_used": [0.95],
            "base_rate": [50.0],
            "total_break_time": [120.0],
        }
    )
    money = frame_revenue_net(frame)
    assert money["available"] is False
    assert "baseline_tvr" in money["reason"]


def test_frame_revenue_net_missing_rate_column_is_unavailable() -> None:
    frame = pd.DataFrame({"predicted_revenue": [100.0], "baseline_tvr": [2.0]})
    money = frame_revenue_net(frame)  # no base_rate / total_break_time
    assert money["available"] is False
    assert "base_rate" in money["reason"]


def test_frame_and_plan_agree_on_same_inputs() -> None:
    # The frame twin and the plan path must report the SAME money on the same day,
    # to the cent: both compute cost as gross potential minus delivered revenue.
    segs = [
        _seg("d|c|000", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05),
        _seg("d|c|001", baseline_tvr=3.0, cpp=50.0, premium=1.0, coef=-0.02),
    ]
    plan = optimize_breaks(segs, Guardrails(), revenue_weight=1.0, refine=False)
    plan_money = plan_revenue_net(plan, segments=segs)

    # Build the saved rows as an export that persisted baseline_tvr would.
    rows = []
    by_id = {s.segment_id: s for s in segs}
    for sp in plan.segments:
        seg = by_id[sp.segment_id]
        rows.append(
            {
                "predicted_revenue": sp.revenue,
                "baseline_tvr": seg.baseline_tvr,
                "base_rate": seg.cpp * seg.premium,
                "total_break_time": sp.num_breaks * seg.break_length_seconds,
            }
        )
    frame_money = frame_revenue_net(pd.DataFrame(rows))

    assert frame_money["available"] and plan_money["available"]
    assert frame_money["revenue_ils"] == pytest.approx(plan_money["revenue_ils"], abs=0.02)
    assert frame_money["retention_cost_ils"] == pytest.approx(
        plan_money["retention_cost_ils"], abs=0.02
    )
    assert frame_money["revenue_net_ils"] == pytest.approx(
        plan_money["revenue_net_ils"], abs=0.02
    )


# 5. objective mode: default byte-identical, net opt-in ------------------------
def test_blend_mode_is_the_default_and_unchanged() -> None:
    # No objective_mode == explicit 'blend' == identical placements.
    segs = [
        _seg("d|c|000", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05),
        _seg("d|c|001", baseline_tvr=1.0, cpp=50.0, premium=1.2, coef=-0.10),
    ]
    default = optimize_breaks(segs, Guardrails(), revenue_weight=0.5)
    blend = optimize_breaks(segs, Guardrails(), revenue_weight=0.5, objective_mode="blend")
    assert [(s.segment_id, s.num_breaks) for s in default.segments] == [
        (s.segment_id, s.num_breaks) for s in blend.segments
    ]
    assert default.total_revenue == blend.total_revenue


def test_net_mode_maximizes_net_at_least_as_well_as_blend() -> None:
    segs = [
        _seg("d|c|000", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05),
        _seg("d|c|001", baseline_tvr=1.0, cpp=50.0, premium=1.2, coef=-0.10),
        _seg("d|c|002", baseline_tvr=3.0, cpp=50.0, premium=1.0, coef=-0.02),
    ]
    blend = optimize_breaks(segs, Guardrails(), revenue_weight=0.5, refine=False)
    net = optimize_breaks(segs, Guardrails(), revenue_weight=0.5, refine=False, objective_mode="revenue_net")
    blend_net = plan_revenue_net(blend, segments=segs)["revenue_net_ils"]
    net_net = plan_revenue_net(net, segments=segs)["revenue_net_ils"]
    # The net objective must not do WORSE on net ILS than the blend does.
    assert net_net >= blend_net - 0.01


def test_net_mode_refuses_without_monetizable_audience() -> None:
    # Every segment has zero rating: no lost audience to value -> honest refusal.
    segs = [_seg("d|c|000", baseline_tvr=0.0, cpp=50.0, premium=1.0, coef=-0.05)]
    with pytest.raises(ValueError, match="revenue_net"):
        optimize_breaks(segs, Guardrails(), objective_mode="revenue_net")


def test_invalid_objective_mode_raises() -> None:
    seg = _seg("d|c|000", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05)
    with pytest.raises(ValueError, match="objective_mode"):
        optimize_breaks([seg], Guardrails(), objective_mode="nonsense")


# 6. compare_objectives: two real plans, honest two-sided money ---------------
def test_compare_objectives_reports_both_plans() -> None:
    segs = [
        _seg("d|c|000", baseline_tvr=2.0, cpp=50.0, premium=1.0, coef=-0.05),
        _seg("d|c|001", baseline_tvr=1.0, cpp=50.0, premium=1.2, coef=-0.10),
        _seg("d|c|002", baseline_tvr=3.0, cpp=50.0, premium=1.0, coef=-0.02),
    ]
    report = compare_objectives(segs, guardrails=Guardrails(), revenue_weight=0.5)
    for side in ("blend", "net"):
        leg = report[side]
        assert leg["revenue_ils"] is not None
        assert leg["retention_cost_ils"] is not None
        assert leg["revenue_net_ils"] is not None
        assert leg["basis"]["source"] == "modeled"
    # The net leg maximises net ILS, so it is at least as good as blend on net.
    assert report["net"]["revenue_net_ils"] >= report["blend"]["revenue_net_ils"] - 0.01
    # Deltas are net minus blend on each money field.
    assert report["delta"]["revenue_net_ils"] == pytest.approx(
        round(report["net"]["revenue_net_ils"] - report["blend"]["revenue_net_ils"], 2), abs=0.02
    )


def _main() -> int:
    """Run every ``test_*`` in this module as a plain script (no pytest needed)."""
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures: list[str] = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - a test harness reports every failure
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
