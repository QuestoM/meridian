"""Contract tests for the money-story additions: the net-focused comparison
(/api/optimizer/net-comparison plus the overview's additive net_point) and the
retention-cost uncertainty band on /api/yield-per-second.

These boot the real FastAPI app and exercise the genuine optimizer and loaders,
like the other engine-backed API suites. The frontier machinery is a single
background thread; the helpers below wait for it honestly (no fabricated ready
state) and the computing-state test simulates an in-flight sweep through the
machine's own lock and state, restoring it afterwards.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from kairos_api.dashboard_api import (
    NET_POINT_ID,
    _frontier_bg_lock,
    _frontier_bg_state,
    _frontier_state,
)
from kairos_api.server import _ENGINE_AVAILABLE, _load_settings, app

client = TestClient(app)

# The sweep is a handful of refined single-day optimizations (seconds); the
# deadline is generous so a cold cache on a slow machine still finishes.
_FRONTIER_DEADLINE_SECONDS = 300.0

_MONEY_FIELDS = {"gross", "retention_cost", "net", "breaks"}
_POINT_FIELDS = {"retention", "revenue", "retention_floor", "num_breaks", "selected"}


def _engine_scope_ready() -> bool:
    """Whether the engine and an owned channel exist (else nothing to test)."""
    return bool(_ENGINE_AVAILABLE and _load_settings().operator_channel)


def _wait_frontier_ready():
    """Drive the shared background machine to a settled state and return it."""
    settings = _load_settings()
    deadline = time.time() + _FRONTIER_DEADLINE_SECONDS
    while time.time() < deadline:
        points, bundle, status = _frontier_state(settings, None)
        if status != "computing":
            return points, bundle, status
        time.sleep(0.2)
    raise AssertionError("frontier background sweep did not finish in time")


# 1. Computing state is honest: no numbers while the sweep is in flight -------
def test_net_comparison_computing_state_has_no_numbers() -> None:
    if not _engine_scope_ready():
        pytest.skip("engine or owned channel unavailable")
    _wait_frontier_ready()  # ensure no real compute thread is in flight
    with _frontier_bg_lock:
        saved = {
            "key": _frontier_bg_state["key"],
            "status": _frontier_bg_state["status"],
            "points": _frontier_bg_state["points"],
            "net_bundle": _frontier_bg_state["net_bundle"],
        }
        _frontier_bg_state["status"] = "computing"
        _frontier_bg_state["points"] = ()
        _frontier_bg_state["net_bundle"] = None
    try:
        body = client.get("/api/optimizer/net-comparison").json()
        assert body["status"] == "computing"
        assert body["current"] is None
        assert body["net_focused"] is None
        assert body["delta"] is None
        assert isinstance(body["basis"], str) and body["basis"].strip()
        # The overview's additive point is null too, never a fabricated point.
        overview = client.get("/api/overview").json()
        assert overview["frontier_net_point"] is None
        assert overview["frontier_status"] == "computing"
    finally:
        with _frontier_bg_lock:
            _frontier_bg_state.update(saved)


# 2. Ready state: internally consistent money on one shared basis -------------
def test_net_comparison_ready_state_is_internally_consistent() -> None:
    if not _engine_scope_ready():
        pytest.skip("engine or owned channel unavailable")
    points, bundle, status = _wait_frontier_ready()
    assert status == "ready"
    assert bundle is not None and bundle.get("comparison_available"), bundle

    body = client.get("/api/optimizer/net-comparison").json()
    assert body["status"] == "ready"
    assert isinstance(body["basis"], str) and "scenario-runner" in body["basis"]

    for side in ("current", "net_focused"):
        block = body[side]
        assert set(block) == _MONEY_FIELDS
        assert block["gross"] >= 0 and block["retention_cost"] >= 0
        assert isinstance(block["breaks"], int) and block["breaks"] >= 0
        # net = gross - retention_cost within rounding (each side rounds to 2dp).
        assert abs(block["net"] - (block["gross"] - block["retention_cost"])) <= 0.02

    delta = body["delta"]
    assert set(delta) == _MONEY_FIELDS
    for key in ("gross", "retention_cost", "net"):
        expected = round(body["net_focused"][key] - body["current"][key], 2)
        assert abs(delta[key] - expected) <= 0.02
    assert delta["breaks"] == body["net_focused"]["breaks"] - body["current"]["breaks"]

    # One shared basis: the current side is the same runner the sweep anchored at
    # the saved floor/weight, so its gross and break count reproduce the sweep's
    # selected point exactly.
    anchors = [point for point in points if point.get("selected")]
    if anchors:
        assert abs(anchors[0]["revenue"] - body["current"]["gross"]) < 0.5
        assert anchors[0]["num_breaks"] == body["current"]["breaks"]


# 3. The overview's net_point: additive, labelled, frontier-point shaped ------
def test_overview_net_point_shape_and_frontier_points_unchanged() -> None:
    if not _engine_scope_ready():
        pytest.skip("engine or owned channel unavailable")
    _wait_frontier_ready()
    overview = client.get("/api/overview").json()
    net_point = overview["frontier_net_point"]
    assert net_point is not None
    assert net_point["id"] == NET_POINT_ID
    assert set(net_point) == _POINT_FIELDS | {"id"}
    # The saved objective_mode is what makes the net point "selected"; it is only
    # ever True when the operator actually saved revenue_net.
    saved_mode = str(getattr(_load_settings(), "objective_mode", "blend"))
    assert net_point["selected"] == (saved_mode == "revenue_net")
    # The existing sweep points keep their exact shape (frontend compatibility).
    assert isinstance(overview["frontier"], list) and overview["frontier"]
    for point in overview["frontier"]:
        assert set(point) == _POINT_FIELDS

    # The net point agrees with the comparison's net-focused leg: same run.
    body = client.get("/api/optimizer/net-comparison").json()
    if body["status"] == "ready":
        assert net_point["num_breaks"] == body["net_focused"]["breaks"]
        assert abs(net_point["revenue"] - body["net_focused"]["gross"]) < 0.5


# 4. Yield band: brackets the point with the documented sign convention -------
def test_yield_retention_cost_band_brackets_point() -> None:
    body = client.get("/api/yield-per-second").json()
    if not body.get("available") or not body.get("revenue_net_available"):
        return  # honest empty states are covered by the phase B suite
    assert "retention_cost_low" in body
    assert "retention_cost_high" in body
    assert isinstance(body["retention_cost_basis"], str)
    low = body["retention_cost_low"]
    high = body["retention_cost_high"]
    point = body["retention_cost_ils"]
    assert low is not None and high is not None, body["retention_cost_basis"]
    assert "95" in body["retention_cost_basis"]
    # Sign convention: ci_low (more damage) -> high cost, ci_high -> low cost,
    # so low <= point <= high, all non-negative.
    assert 0 <= low <= point <= high
    # The calibrated intervals are real (non-degenerate), so the band has width.
    assert low < high
    # The band is additive: the existing identity still holds beside it.
    expected_net = round(body["revenue_ils"] - point, 2)
    assert abs(body["revenue_net_ils"] - expected_net) < 1.0


# 5. Route hygiene: the new endpoint exists exactly once ----------------------
def test_net_comparison_route_is_unique() -> None:
    paths = [
        (route.path, method)
        for route in app.routes
        if hasattr(route, "methods") and route.methods
        for method in route.methods
    ]
    assert paths.count(("/api/optimizer/net-comparison", "GET")) == 1
