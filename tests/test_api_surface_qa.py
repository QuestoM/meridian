"""API surface gate: every read endpoint answers 200 with a sane payload.

Runs the real FastAPI app in process (no live server) and sweeps every GET
route that takes no path parameter, asserting:

  * status 200 and a parseable JSON body (CSV endpoints excepted),
  * no raw NaN / Infinity tokens leaking into the wire format,
  * no Unicode replacement characters (Hebrew mojibake canary),
  * the two safe POST bodies (scenario, price-slot) answer with real numbers.

It then ties the headline surfaces to the committed plan: the overview summary
totals must equal the CSV's own sums, and the compliance payload must carry the
full seven-rule check set with observed/limit pairs. These are the standing
tri-state honesty checks: empty data must surface as null or an empty list,
never as a fabricated zero-shaped plan.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"

# Raw JSON text must never carry bare NaN / Infinity tokens (invalid JSON that
# json.loads on some clients still accepts, so it can hide in manual testing).
BAD_TOKENS = re.compile(r"(?<![\w\"])(NaN|Infinity|-Infinity)(?![\w\"])")
REPLACEMENT_CHAR = "�"

# GET routes that need a path parameter or heavy implicit compute are exercised
# by the journey suites instead of the blind sweep.
SWEEP_EXCLUDED = {
    "/api/schedule/segment/{segment_id:path}",
    "/api/jobs/{job_id}",
    "/api/advertisers/{advertiser_id}/conditions",
    # Without a target the effect preview optimizes the ENTIRE grid twice;
    # the journey suite covers it with a scoped candidate instead.
    "/api/overrides/effect",
    "/api/constraints/effect",
}

# Session-gated by design (kairos_api/auth.py): /api/auth/* answers 401 without
# a signed-in session, which is the contract, not a failure. The public-path
# promise (health stays open, /api/auth/me reports the honest auth state) has
# its own test below.
SWEEP_EXCLUDED_PREFIXES = ("/api/auth/",)


@pytest.fixture(scope="module")
def client() -> TestClient:
    from kairos_api.server import app

    return TestClient(app, raise_server_exceptions=False)


def _swept_routes() -> "tuple[list[str], list[tuple[str, list[str]]]]":
    """``(bare, parameterised)``: routes callable with nothing, and routes that
    declare a REQUIRED query parameter.

    A route that requires a query parameter cannot be swept blind -- there is no
    honest default for "which programme" or "which day", and inventing one would
    make the gate assert against a fabricated target. Such a route is not
    excluded, it is held to a different and equally strict contract below: a bare
    call must be a clean 422 naming the missing field, never a 500 and never a
    200 that answered a question nobody asked.
    """
    from kairos_api.server import app

    bare: list[str] = []
    parameterised: list[tuple[str, list[str]]] = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", "")
        if "GET" not in methods or not path.startswith("/api") or "{" in path:
            continue
        if path in SWEEP_EXCLUDED or path.startswith(SWEEP_EXCLUDED_PREFIXES):
            continue
        dependant = getattr(route, "dependant", None)
        required = sorted(
            field.name
            for field in (getattr(dependant, "query_params", None) or ())
            if getattr(field, "required", False)
        )
        if required:
            parameterised.append((path, required))
        else:
            bare.append(path)
    return sorted(bare), sorted(parameterised)


@pytest.fixture(scope="module")
def get_routes() -> list[str]:
    routes, _parameterised = _swept_routes()
    assert len(routes) >= 25, f"route surface shrank unexpectedly: {sorted(routes)}"
    return routes


@pytest.fixture(scope="module")
def parameterised_routes() -> "list[tuple[str, list[str]]]":
    _bare, parameterised = _swept_routes()
    return parameterised


def test_every_get_endpoint_is_healthy(client, get_routes):
    failures = []
    for path in get_routes:
        response = client.get(path)
        body = response.text
        if response.status_code != 200:
            failures.append(f"{path}: status {response.status_code} {body[:120]}")
            continue
        if BAD_TOKENS.search(body):
            failures.append(f"{path}: raw NaN/Infinity token in body")
        if REPLACEMENT_CHAR in body:
            failures.append(f"{path}: unicode replacement character (mojibake)")
        if not path.endswith(".csv"):
            try:
                json.loads(body)
            except ValueError as exc:
                failures.append(f"{path}: body is not JSON ({exc})")
    assert not failures, "\n".join(failures)


def test_a_route_that_needs_a_parameter_refuses_cleanly_without_one(
    client, parameterised_routes
):
    """The other half of the sweep: every GET that requires a query parameter.

    These cannot be called blind, but they can still be wrong in the two ways
    that matter. A 500 means the route trusted an input it never received; a 200
    means it answered without being told what to answer about. Both are caught
    here, so no route escapes the gate merely by taking an argument."""
    assert parameterised_routes, (
        "no GET route declares a required query parameter; if that is genuinely "
        "true this test can go, but more likely the introspection stopped working"
    )
    failures = []
    for path, required in parameterised_routes:
        response = client.get(path)
        if response.status_code != 422:
            failures.append(
                f"{path}: bare call returned {response.status_code}, expected 422 "
                f"naming the missing {', '.join(required)}"
            )
            continue
        body = response.text
        if REPLACEMENT_CHAR in body:
            failures.append(f"{path}: unicode replacement character (mojibake)")
        try:
            detail = json.loads(body)["detail"]
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(f"{path}: refusal body is not a JSON detail ({exc})")
            continue
        named = {
            str(item.get("loc", ["", ""])[-1])
            for item in detail if isinstance(item, dict)
        }
        missing = [name for name in required if name not in named]
        if missing:
            failures.append(f"{path}: refusal does not name {missing}")
    assert not failures, "\n".join(failures)


def test_hebrew_round_trips_uncorrupted(client):
    """The owned channel name and Hebrew labels must survive the wire intact."""
    settings_file = json.loads(
        (ROOT / "data" / "kairos_settings.json").read_text(encoding="utf-8")
    )
    api_settings = client.get("/api/settings").json()
    assert api_settings["operator_channel"] == settings_file["operator_channel"]
    compliance = client.get("/api/compliance").json()
    labels = [check.get("label_he", "") for check in compliance.get("checks", [])]
    assert labels and all(labels), "compliance checks must carry Hebrew labels"
    assert any("א" <= ch <= "ת" for label in labels for ch in label), (
        "Hebrew labels contain no Hebrew letters (encoding corruption)"
    )


def test_scenario_post_returns_real_summary(client):
    response = client.post("/api/scenario", json={
        "revenue_weight": 60, "retention_floor": 0.72,
        "max_breaks_per_hour": 4, "risk_lambda": 0.0,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("engine") == "kairos", f"engine fell back: {body.get('detail')}"
    summary = body["summary"]
    for key in ("total_breaks", "total_ad_seconds", "projected_revenue", "average_retention"):
        assert key in summary
    assert summary["total_breaks"] >= 0
    assert isinstance(body["compliant"], bool)


def test_price_slot_post_returns_traceable_breakdown(client):
    response = client.post("/api/pricing/price-slot", json={
        "pricing_class": "Other", "weekday_iso": 1,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["base_cpp"] > 0
    assert body["final_cpp"] > 0
    assert isinstance(body["layers"], list)


def test_overview_summary_equals_committed_plan(client):
    """The morning-check headline tiles must be the committed plan's own sums
    for the operator's channel, never whole-market money quoted as ours, and
    the payload must disclose that basis."""
    plan = pd.read_csv(CSV_PATH, encoding="utf-8")
    summary = client.get("/api/overview").json()["summary"]
    channel = summary["scope_channel"]
    assert channel, "the overview summary must disclose its channel basis"
    owned = plan[plan["channel"].astype(str).str.strip() == channel]
    assert len(owned) > 0
    assert summary["total_breaks"] == int(owned["num_breaks"].sum())
    assert summary["total_ad_seconds"] == int(owned["total_break_time"].sum())
    assert summary["projected_revenue"] == pytest.approx(
        owned["predicted_revenue"].sum(), abs=0.05
    )
    assert summary["n_channels_total"] == plan["channel"].nunique()
    assert summary["n_dates"] == owned["date"].nunique()
    assert summary["average_retention"] is not None
    assert 0 <= summary["risk_score"] <= 100


def test_overview_source_counts_are_real(client):
    body = client.get("/api/overview").json()
    counts = body["source_counts"]
    plan = pd.read_csv(CSV_PATH, encoding="utf-8")
    assert counts["planned_break_rows"] == len(plan)
    assert counts["programmes"] > 0
    assert counts["spots"] > 0


def test_compliance_payload_carries_the_full_rule_set(client):
    """/api/compliance is the per-rule promise surface: all seven rules must be
    present with observed/limit pairs and a tri-state status."""
    body = client.get("/api/compliance").json()
    check_ids = {check["id"] for check in body["checks"]}
    assert check_ids == {
        "hourly_ad_load", "break_density", "retention_floor", "protected_programs",
        "break_spacing", "daily_ad_load", "gold_breaks",
    }, f"rule set drifted: {sorted(check_ids)}"
    for check in body["checks"]:
        assert check["status"] in {"compliant", "at_risk"}
        assert "observed" in check and "limit" in check
    assert body["status"] in {"compliant", "at_risk"}
    assert isinstance(body["violations"], list)


def test_recommendations_are_individually_identifiable(client):
    """BUG-5 regression: the five recommendations must carry five DISTINCT real
    identities (programme type + clock + weekday + date) built from the plan,
    each bound to a real owned segment with its semantic anchor, and grouped on
    real distinguishing fields (daypart plus risk), never on stamped constants."""
    from kairos.data.dayparts import daypart_keys

    body = client.get("/api/overview").json()
    recs = body["recommendations"]
    assert len(recs) == 5
    for key in ("title", "title_he"):
        labels = [rec[key] for rec in recs]
        assert len(set(labels)) == len(labels), f"duplicate {key}: {labels}"
    for rec in recs:
        assert rec["segment_id"], "top recommendations must bind to a real segment"
        anchor = rec["anchor"]
        assert anchor["date"] and anchor["start_clock"] and anchor["program"]
        # The displayed identity is built from the same fields the anchor carries.
        assert anchor["start_clock"] in rec["title"] and anchor["date"] in rec["title"]
        assert rec["program_type"] in rec["title"]
        assert anchor["start_clock"] in rec["title_he"] and anchor["date"] in rec["title_he"]
        # Grouping keys on real fields, resolved to real owned candidate segments.
        assert rec["daypart"] in daypart_keys()
        assert rec["risk"] in {"High", "Medium", "Low"}
        assert rec["candidates"], "grouping must resolve to real candidate segments"
        for cand in rec["candidates"][:3]:
            assert cand["segment_id"] and cand["anchor"]["date"]


def test_recommendations_bind_only_to_owned_channel(client):
    """The competitor boundary on the decision surface: an actionable
    recommendation may only carry an owned-channel segment_id."""
    body = client.get("/api/overview").json()
    settings = json.loads((ROOT / "data" / "kairos_settings.json").read_text(encoding="utf-8"))
    owned = str(settings["operator_channel"]).strip()
    plan = pd.read_csv(CSV_PATH, encoding="utf-8")
    owned_ids = set(plan[plan["channel"] == owned]["segment_id"])
    for rec in body["recommendations"]:
        if rec.get("actionable"):
            assert rec["channel"] == owned
            assert rec["segment_id"] in owned_ids, (
                f"recommendation {rec['id']} binds outside the owned channel"
            )
            assert rec.get("proposed_kind") in {"pin", "force", "forbid", "gold",
                                                "lower_count"}
        else:
            assert rec.get("segment_id") is None


def test_empty_schedule_summary_is_honest(client):
    """Tri-state honesty: with no plan rows the summary must report null for
    revenue / retention / risk (unknown), not confident zeros. Exercises the
    builder directly with an empty frame (the fresh-deploy state)."""
    from kairos_api.server import _summarize_schedule

    summary = _summarize_schedule(pd.DataFrame())
    assert summary["projected_revenue"] is None
    assert summary["average_retention"] is None
    assert summary["risk_score"] is None
    assert summary["total_breaks"] == 0


def test_jobs_endpoint_unknown_id_is_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_auth_public_paths_stay_open(client):
    """The auth middleware must keep /api/health public in every auth state,
    and /api/auth/me must report the honest auth state instead of walling the
    dashboard when no user store has been seeded."""
    assert client.get("/api/health").status_code == 200
    me = client.get("/api/auth/me")
    assert me.status_code == 200, me.text
    body = me.json()
    users_store = ROOT / "data" / "auth" / "users.json"
    if not users_store.exists():
        assert body.get("auth_disabled") is True, (
            "with no seeded user store, auth must report itself disabled"
        )
