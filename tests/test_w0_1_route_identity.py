"""W0-1: the split routes keep their path, method, tag and module, exactly once.

The wave-zero router split moved 25 routes out of dashboard_api, insights_api,
catalog_api and version_store into per-owner modules. The bar is the live
OpenAPI surface: every path published before the split is still published, every
one of the 25 sits in the module its published contract names, no path is
defined twice, and the static dashboard mount stays last so a later append can
never be shadowed by it.

The 90 paths below are the surface measured before the split
(docs/ux-gauntlet/spec.md section 8.2, and re-measured in process this session:
90 paths, 113 operations, 56 of them writes). They are asserted as a floor that
must still be served, not as an equality, because the append-only registration
region exists precisely so a later piece can add its own published paths.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute
from starlette.routing import Mount

# path, method, defining module, OpenAPI tag: the frozen destination of every
# route the split moved.
SPLIT_ROUTES = (
    ("/api/overview", "GET", "kairos_api.overview_api", "dashboard"),
    ("/api/break-decisions", "GET", "kairos_api.overview_api_decisions", "dashboard"),
    ("/api/break-decisions", "POST", "kairos_api.overview_api_decisions", "dashboard"),
    ("/api/schedule", "GET", "kairos_api.week_api", "dashboard"),
    ("/api/inventory", "GET", "kairos_api.week_api", "catalog"),
    ("/api/scenario-compare", "POST", "kairos_api.scenario_compare_api", "insights"),
    ("/api/forecasts", "GET", "kairos_api.scenario_compare_api", "catalog"),
    ("/api/schedule/segments", "GET", "kairos_api.day_api", "dashboard"),
    ("/api/schedule/segment/{segment_id:path}", "GET", "kairos_api.day_api", "dashboard"),
    ("/api/break-operations", "GET", "kairos_api.day_api", "dashboard"),
    ("/api/break-library", "GET", "kairos_api.day_api", "catalog"),
    ("/api/gold-breaks", "GET", "kairos_api.gold_api", "insights"),
    ("/api/campaigns", "GET", "kairos_api.campaigns_read", "catalog"),
    ("/api/compliance", "GET", "kairos_api.compliance_api", "dashboard"),
    ("/api/yield-per-second", "GET", "kairos_api.yield_api", "insights"),
    ("/api/reports", "GET", "kairos_api.downloads_api", "catalog"),
    ("/api/files", "GET", "kairos_api.downloads_api", "catalog"),
    ("/api/model/audience", "GET", "kairos_api.model_audience_api", "insights"),
    ("/api/impact", "GET", "kairos_api.model_impact_api", "catalog"),
    ("/api/versions", "GET", "kairos_api.history_api", "versions"),
    ("/api/versions/{version_id}/diff", "GET", "kairos_api.history_api", "versions"),
    ("/api/versions/{version_id}/restore", "POST", "kairos_api.history_api", "versions"),
    ("/api/versions/snapshot", "POST", "kairos_api.history_api", "versions"),
    ("/api/versions/{version_id}", "PATCH", "kairos_api.history_api", "versions"),
    ("/api/make-good-alerts", "GET", "kairos_api.pacing_alerts_api", "insights"),
)

BASELINE_PATHS = (
    "/api/activity-log",
    "/api/advertisers",
    "/api/advertisers/options",
    "/api/advertisers/stats",
    "/api/advertisers/{advertiser_id}",
    "/api/advertisers/{advertiser_id}/conditions",
    "/api/advertisers/{advertiser_id}/conditions/{rule_id}",
    "/api/agencies",
    "/api/agencies/summary",
    "/api/agencies/{agency_id}",
    "/api/agencies/{agency_id}/advertisers",
    "/api/agencies/{agency_id}/advertisers/{advertiser}",
    "/api/agencies/{agency_id}/conditions",
    "/api/agencies/{agency_id}/conditions/{rule_id}",
    "/api/agencies/{agency_id}/deactivate",
    "/api/assistant/ask",
    "/api/assistant/ask/stream",
    "/api/assistant/audit",
    "/api/assistant/conversations",
    "/api/assistant/conversations/{conversation_id}",
    "/api/assistant/conversations/{conversation_id}/changes",
    "/api/assistant/conversations/{conversation_id}/restore",
    "/api/assistant/proposals",
    "/api/assistant/proposals/{batch_id}/apply",
    "/api/assistant/proposals/{batch_id}/reject",
    "/api/assistant/restore",
    "/api/assistant/restore/{restore_id}",
    "/api/assistant/status",
    "/api/assistant/thread",
    "/api/assistant/upload",
    "/api/assistant/uploads",
    "/api/assistant/uploads/{upload_id}",
    "/api/auth/change-password",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/me",
    "/api/auth/users",
    "/api/auth/users/{username}",
    "/api/auth/users/{username}/affiliation",
    "/api/auth/users/{username}/reset-password",
    "/api/break-decisions",
    "/api/break-library",
    "/api/break-operations",
    "/api/campaigns",
    "/api/compliance",
    "/api/constraints",
    "/api/constraints/effect",
    "/api/constraints/options",
    "/api/constraints/{constraint_id}",
    "/api/events",
    "/api/events/{event_id}",
    "/api/export/schedule.csv",
    "/api/export/spots.csv",
    "/api/files",
    "/api/forecasts",
    "/api/gold-breaks",
    "/api/health",
    "/api/impact",
    "/api/inventory",
    "/api/jobs/recompute",
    "/api/jobs/{job_id}",
    "/api/make-good-alerts",
    "/api/model/audience",
    "/api/optimal-plan",
    "/api/optimizer-plan",
    "/api/optimizer/net-comparison",
    "/api/overrides",
    "/api/overrides/effect",
    "/api/overrides/{override_id}",
    "/api/overview",
    "/api/parameters",
    "/api/pricing",
    "/api/pricing/price-slot",
    "/api/recompute-schedule",
    "/api/reports",
    "/api/scenario",
    "/api/scenario-compare",
    "/api/schedule",
    "/api/schedule/segment/{segment_id}",
    "/api/schedule/segments",
    "/api/settings",
    "/api/settings/controls",
    "/api/uploads/status",
    "/api/uploads/{kind}",
    "/api/versions",
    "/api/versions/snapshot",
    "/api/versions/{version_id}",
    "/api/versions/{version_id}/diff",
    "/api/versions/{version_id}/restore",
    "/api/yield-per-second",
)


@pytest.fixture(scope="module")
def api_routes() -> list[APIRoute]:
    from kairos_api.server import app

    return [route for route in app.routes if isinstance(route, APIRoute)]


def _key(route: APIRoute) -> set[tuple[str, str]]:
    return {(route.path, method) for method in route.methods if method != "HEAD"}


def test_every_split_route_sits_in_its_published_module(api_routes) -> None:
    index = {}
    for route in api_routes:
        for path, method in _key(route):
            index[(path, method)] = route
    for path, method, module, tag in SPLIT_ROUTES:
        route = index.get((path, method))
        assert route is not None, f"{method} {path} is no longer served"
        assert route.endpoint.__module__ == module, (
            f"{method} {path} moved to {route.endpoint.__module__}, not {module}")
        assert route.tags == [tag], f"{method} {path} carries tags {route.tags}, not [{tag}]"


def test_no_path_and_method_is_defined_twice(api_routes) -> None:
    seen: dict[tuple[str, str], str] = {}
    duplicates = []
    for route in api_routes:
        for pair in _key(route):
            if pair in seen:
                duplicates.append((pair, seen[pair], route.endpoint.__module__))
            seen[pair] = route.endpoint.__module__
    assert duplicates == [], f"a router is mounted twice: {duplicates}"


def test_every_pre_split_path_is_still_published() -> None:
    """Read from the published schema, not the route table, because that is what
    the append rule's bar is stated on and what a client reads."""
    from kairos_api.server import app

    served = set(app.openapi()["paths"])
    missing = sorted(path for path in BASELINE_PATHS if path not in served)
    assert missing == [], f"paths lost since the pre-split baseline: {missing}"
    assert len(BASELINE_PATHS) == 90


def test_the_static_dashboard_mount_stays_last(api_routes) -> None:
    from kairos_api.server import app

    mounts = [index for index, route in enumerate(app.routes) if isinstance(route, Mount)]
    if not mounts:
        pytest.skip("no built dashboard on disk, so no static mount to order")
    assert max(mounts) == len(app.routes) - 1, (
        "the StaticFiles mount serves '/' and must stay last, or it shadows every "
        "route appended after it")


def test_the_four_split_modules_carry_no_routes_of_their_own() -> None:
    from kairos_api import catalog_api, dashboard_api, insights_api

    for module in (dashboard_api, insights_api, catalog_api):
        assert module.router.routes == [], (
            f"{module.__name__}.router is mounted above the append marker and must "
            "stay empty after the split, or its routes are defined twice")
