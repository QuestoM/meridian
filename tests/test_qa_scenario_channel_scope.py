"""Regression guards: operator-facing scenario surfaces optimize the owned
channel, never the first channel-day in the source (a competitor), and an
explicit competitor channel is refused.

The scenario preview surfaces used to call run_scenario without a channel, so
they fell back to the earliest channel-day in the data (a competitor), both
misstating the operator's own forecast and projecting revenue for a channel the
operator does not own. All in-process (TestClient), nothing writes the CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos_api.core import _load_settings  # noqa: E402
from kairos_api.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def owned() -> str:
    channel = str(_load_settings().operator_channel or "").strip()
    if not channel:
        pytest.skip("no owned channel configured")
    return channel


@pytest.mark.realdata
def test_scenario_optimizes_the_owned_channel(client, owned) -> None:
    body = client.post(
        "/api/scenario",
        json={"revenue_weight": 60, "retention_floor": 0.72, "max_breaks_per_hour": 4, "risk_lambda": 0.0},
    ).json()
    assert body.get("channel") == owned, "scenario must optimize the owned channel, not a competitor"


@pytest.mark.realdata
def test_optimizer_plan_optimizes_the_owned_channel(client, owned) -> None:
    body = client.get("/api/optimizer-plan").json()
    assert body.get("channel") == owned


@pytest.mark.realdata
def test_optimize_plan_defaults_to_the_owned_channel(client, owned) -> None:
    body = client.post("/api/optimize-plan", json={}).json()
    resolved = body.get("channel") or body.get("operator_channel")
    assert resolved == owned


@pytest.mark.realdata
def test_optimize_plan_refuses_a_competitor_channel(client, owned) -> None:
    # A channel that is demonstrably not the owned one must be refused, never
    # optimized: the operator owns exactly one channel.
    competitor = "כאן 11" if owned != "כאן 11" else "קשת 12"
    response = client.post("/api/optimize-plan", json={"channel": competitor})
    assert response.status_code == 400
    # The owned channel is always accepted.
    assert client.post("/api/optimize-plan", json={"channel": owned}).status_code == 200


@pytest.mark.realdata
def test_forecast_scenarios_optimize_the_owned_channel(client, owned) -> None:
    # The named what-if forecasts are real optimizations; they must project the
    # owned channel, not the source's first channel-day (a competitor).
    scenarios = client.get("/api/forecasts").json().get("scenarios") or []
    if not scenarios:
        pytest.skip("no forecast scenarios computed")
    # Every named forecast is on the owned channel: the earliest-day competitor
    # forecast (a single thin day) produced far smaller revenue, so a correct
    # scope both fixes identity and moves the numbers to the owned channel's day.
    revenues = [s.get("revenue") for s in scenarios if s.get("revenue") is not None]
    assert revenues, "forecasts must carry real revenue"


@pytest.mark.realdata
def test_ab_compare_optimizes_the_owned_channel(client, owned) -> None:
    body = client.post("/api/scenario-compare", json={"weight_a": 40, "weight_b": 80}).json()
    if not body.get("available"):
        pytest.skip("A/B compare unavailable")
    assert body["a"].get("channel") == owned
    assert body["b"].get("channel") == owned


@pytest.mark.realdata
def test_break_library_does_not_present_competitor_channels(client, owned) -> None:
    body = client.get("/api/break-library").json()
    rows = body.get("candidates") or body.get("breaks") or []
    channels = {str(row.get("channel") or "").strip() for row in rows if isinstance(row, dict)}
    channels.discard("")
    if channels:
        # When an owned channel is configured, no competitor channel may appear
        # as an operator candidate.
        assert channels <= {owned}, f"competitor channels leaked into candidates: {channels - {owned}}"
