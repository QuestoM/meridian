"""Integration tests for the engine-backed API endpoints.

These boot the FastAPI app and call the real endpoints, which read the reference
xlsx, so they are slower than the unit tests and are excluded from the fast gate
(run them explicitly, like test_loaders.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kairos_api.server import _ENGINE_AVAILABLE, app

client = TestClient(app)


def test_parameters_exposes_every_adjustable_knob() -> None:
    response = client.get("/api/parameters")
    assert response.status_code == 200
    body = response.json()
    assert "settings" in body
    assert _ENGINE_AVAILABLE, "engine should import in the test environment"
    assert "guardrails" in body and "assumptions" in body
    assert body["pricing"]["base_price_per_second_per_tvr_point"] == 60.0
    assert "קשת 12" in body["channels"]


def test_scenario_runs_the_real_engine() -> None:
    low = client.post(
        "/api/scenario",
        json={"revenue_weight": 0, "retention_floor": 0.72, "max_breaks_per_hour": 4},
    ).json()
    high = client.post(
        "/api/scenario",
        json={"revenue_weight": 100, "retention_floor": 0.72, "max_breaks_per_hour": 4},
    ).json()
    assert low["engine"] == "kairos"
    assert low["summary"]["total_breaks"] == 0
    assert high["summary"]["total_breaks"] > 0
    assert "risk_score" in high["summary"]


def test_optimize_plan_returns_a_compliant_plan() -> None:
    response = client.post("/api/optimize-plan", json={"revenue_weight": 0.6})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["compliant"] is True
    assert "guardrails" in body and "assumptions" in body


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
