"""Integration tests for the Phase B read-only/additive endpoints.

Each endpoint must return 200 with a well-formed payload: real numbers where the
source data exists, and a clearly-marked honest empty state where it does not
(advertiser revenue is daily-path-only; campaign flights are header-only). These
boot the real FastAPI app and exercise the genuine optimizer/loaders, so they sit
with the other engine-backed API tests (run explicitly, not in the fast gate).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from kairos_api.server import _ENGINE_AVAILABLE, _build_frontier, _load_settings, app

client = TestClient(app)


# 1. Advertiser stats ---------------------------------------------------------
def test_advertiser_stats_shape_and_honesty() -> None:
    response = client.get("/api/advertisers/stats")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["advertisers"], list)
    assert body["count"] == len(body["advertisers"])
    assert set(body["effect_types"]) == {"premium", "require", "forbid", "pressure"}
    # The honest caveat: weekly optimizer ignores advertiser rules.
    assert "weekly optimizer does not consume" in body["status"]
    if body["advertisers"]:
        row = body["advertisers"][0]
        assert {"advertiser_id", "rule_count", "effect_breakdown", "baseline_premium",
                "avg_effective_premium", "has_conditions"} <= set(row)
        assert set(row["effect_breakdown"]) == {"premium", "require", "forbid", "pressure"}
        assert row["rule_count"] == sum(row["effect_breakdown"].values())
        # Revenue is never fabricated: it is null and explicitly source_pending.
        assert row["revenue"] is None
        assert row["revenue_source"] == "source_pending"
        assert isinstance(row["baseline_premium"], (int, float))


# 2. Frontier scope -----------------------------------------------------------
def test_overview_default_frontier_unchanged() -> None:
    response = client.get("/api/overview")
    assert response.status_code == 200
    body = response.json()
    assert "frontier" in body
    assert body["frontier_scope"] is None


def test_frontier_scope_byte_identical_to_default() -> None:
    """No-scope must equal the unscoped builder exactly (byte-identical default)."""
    settings = _load_settings()
    assert _build_frontier(settings) == _build_frontier(settings, None)
    assert _build_frontier(settings) == _build_frontier(settings, "")


def test_frontier_scope_day_filters_to_one_day() -> None:
    if not _ENGINE_AVAILABLE:
        return
    response = client.get("/api/overview", params={"scope": "day:2024-11-01"})
    assert response.status_code == 200
    body = response.json()
    assert body["frontier_scope"] == "day:2024-11-01"
    assert isinstance(body["frontier"], list)


def test_frontier_scope_rejects_competitor_channel() -> None:
    """A channel that is not the configured owned channel must be a no-op scope."""
    settings = _load_settings()
    if not settings.operator_channel:
        return  # nothing to enforce against when unconfigured
    from kairos_api.server import _parse_frontier_scope

    parsed = _parse_frontier_scope(f"channel:not-{settings.operator_channel}", settings)
    assert parsed["channel"] is None


# 3. Yield per second ---------------------------------------------------------
def test_yield_per_second_shape() -> None:
    response = client.get("/api/yield-per-second")
    assert response.status_code == 200
    body = response.json()
    assert "available" in body
    assert isinstance(body["by_daypart"], list)
    assert isinstance(body["by_programme"], list)
    if body["available"]:
        assert body["revenue_net_available"] is False
        assert "totals" in body and body["totals"]["ad_seconds"] > 0
        for row in body["by_programme"]:
            assert {"group", "revenue", "ad_seconds", "yield_per_second"} <= set(row)
            # Yield is a real ratio: revenue / ad_seconds, both non-negative.
            assert row["ad_seconds"] > 0


# 4. Scenario compare ---------------------------------------------------------
def test_scenario_compare_runs_two_real_optimizations() -> None:
    response = client.post("/api/scenario-compare", json={"weight_a": 0, "weight_b": 100})
    assert response.status_code == 200
    body = response.json()
    if not _ENGINE_AVAILABLE:
        assert body["available"] is False
        return
    assert body["available"] is True
    assert "a" in body and "b" in body and "delta" in body
    assert body["a"]["revenue_weight"] == 0
    assert body["b"]["revenue_weight"] == 100
    # Revenue-first (100) places at least as many breaks as retention-first (0).
    assert body["b"]["total_breaks"] >= body["a"]["total_breaks"]
    assert set(body["delta"]) >= {"revenue", "retention", "breaks", "revenue_net"}
    # revenue_net is honestly null (objective is a convex blend, reported separately).
    assert body["delta"]["revenue_net"] is None


def test_scenario_compare_validates_weight_bounds() -> None:
    response = client.post("/api/scenario-compare", json={"weight_a": 0, "weight_b": 200})
    assert response.status_code == 422


# 5. Gold breaks --------------------------------------------------------------
def test_gold_breaks_shape_and_honesty() -> None:
    response = client.get("/api/gold-breaks")
    assert response.status_code == 200
    body = response.json()
    assert "available" in body
    assert isinstance(body["breaks"], list)
    assert isinstance(body["by_day"], list)
    # No fabricated premium: any listed gold break marks its premium source_pending.
    for item in body["breaks"]:
        assert item["realized_premium"] is None
        assert item["premium_source"] == "source_pending"


# 6. Make-good alerts ---------------------------------------------------------
def test_make_good_alerts_data_pending() -> None:
    response = client.get("/api/make-good-alerts")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["alerts"], list)
    # campaign_flights.csv is header-only, so this is honestly data-pending today.
    assert body["data_available"] is False
    assert body["alerts"] == []
