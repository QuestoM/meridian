"""Integration tests for the additive read-only insights endpoints.

Each endpoint must return 200 with a well-formed payload: real numbers where the
source data exists, and a clearly-marked honest empty state where it does not
(advertiser revenue is daily-path-only; campaign flights are header-only). These
boot the real FastAPI app and exercise the genuine optimizer/loaders, so they sit
with the other engine-backed API tests (run explicitly, not in the fast gate).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kairos_api.server import _ENGINE_AVAILABLE, _frontier_async, _load_settings, app

client = TestClient(app)


# 1. Advertiser stats ---------------------------------------------------------
def test_advertiser_stats_shape_and_honesty() -> None:
    response = client.get("/api/advertisers/stats")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["advertisers"], list)
    assert body["count"] == len(body["advertisers"])
    assert set(body["effect_types"]) == {"premium", "require", "forbid", "pressure"}
    # THE CAVEAT GOT MORE HONEST AND THIS ASSERTION HAD TO FOLLOW IT.
    #
    # It required the sentence "the weekly optimizer does not consume advertiser
    # rules". That was true only by accident: the advertiser conditions file was
    # EMPTY, so nothing was consumed because nothing was there. Measured
    # 2026-08-09, the preference does reach the weekly plan's first pass. What
    # happens next is that the refinement step optimises it straight back out,
    # which is why the schedule does not move. Same outcome, different reason,
    # and the reason is the part an operator needs: the first sentence would have
    # started lying the moment anybody added a pressure row.
    #
    # So the assertion checks the MECHANISM rather than the old wording, because
    # the mechanism is what must not quietly disappear.
    status = body["status"]
    assert "refinement" in status or "refined" in status, (
        "the caveat no longer names the refinement step, so it has gone back to "
        "explaining the absence of data instead of the behaviour of the engine"
    )
    assert "does not change" in status
    # And the daily path is a different answer to a different question, which is
    # the distinction the original sentence collapsed.
    assert "daily" in status
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
    """No-scope must equal the unscoped builder exactly (byte-identical default).

    ``_frontier_async`` returns a ``(points, status)`` tuple; the frontier points
    are the byte-identity contract (an empty scope must resolve to the unscoped
    forecast), so the points element is what we compare.
    """
    settings = _load_settings()
    assert _frontier_async(settings)[0] == _frontier_async(settings, None)[0]
    assert _frontier_async(settings)[0] == _frontier_async(settings, "")[0]


def test_frontier_scope_day_filters_to_one_day() -> None:
    if not _ENGINE_AVAILABLE:
        pytest.skip("engine or owned channel unavailable")
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
        assert "totals" in body and body["totals"]["ad_seconds"] > 0
        # Revenue net of retention is now materialized: the saved CSV carries
        # baseline_tvr, so the endpoint prices retention loss in ILS. When
        # available it must carry the net, the cost and a basis disclosure, and
        # the net must equal revenue minus the priced retention cost.
        if body["revenue_net_available"]:
            assert body["retention_cost_ils"] >= 0
            assert body["basis"] and body["basis"].get("formula")
            expected_net = round(body["revenue_ils"] - body["retention_cost_ils"], 2)
            assert abs(body["revenue_net_ils"] - expected_net) < 1.0
        else:
            # Honest-unavailable path (an older CSV without baseline_tvr).
            assert "revenue_net_reason" in body
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
    # revenue_net used to be pinned to null because run_scenario's summary carries
    # no such field and relabelling the convex-blend objective as a net would have
    # been a lie. The subtraction is now genuinely computed, by the frozen read
    # layer's own pricer on the per-break basis the committed plan's money uses, so
    # what this pins is the arithmetic rather than the absence: either both legs
    # priced and every money figure reconciles on one basis, or the pricer refused
    # and every money figure is null with the reason named. Nothing is proxied.
    assert "not revenue minus retention cost" in body["objective_note"]
    if not body["money_available"]:
        assert body["delta"]["revenue_net"] is None
        assert body["money_reason"]
        return
    assert body["money_basis"]
    for leg in (body["a"], body["b"]):
        assert leg["money_available"] is True
        assert leg["gross"] - leg["retention_cost"] == pytest.approx(leg["revenue_net"], abs=0.01)
        # The blended score keeps its own name and never stands in for the net.
        assert leg["objective"] != leg["revenue_net"]
    assert body["delta"]["revenue_net"] == pytest.approx(
        body["b"]["revenue_net"] - body["a"]["revenue_net"], abs=0.01
    )
    # A week total is its own days added up, so the rows a reader can check add
    # to the figure the panel prints.
    if body["scope"]["mode"] == "week":
        for leg in ("a", "b"):
            assert sum(row[leg]["revenue_net"] for row in body["by_day"]) == pytest.approx(
                body[leg]["revenue_net"], abs=0.05
            )


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
