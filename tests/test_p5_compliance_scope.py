"""P5: the licence verdict is the operator's own channel, and says so.

The measured defect. The frozen builder grades the FULL committed plan, and the
committed plan is the whole market, because the retention model is measured
against the competitive lineup. Measured on the reference plan: 9,026 breaks
graded, of which 6,635, 73.5 percent, belong to three channels this operator does
not own. The break that set the printed retention figure was ``כאן 11``,
2024-11-01, hour 20. No wrong number was on screen, because the operator's own
worst break is numerically identical today, but the figure was drawn from the
wrong population, and 136 of the 200 violations in the payload the operator's
browser held named a rival channel. The seven figures also rendered with no scope
line at all while every other figure on this workspace carries one.

The population is the whole point of a compliance verdict, so the tests here are
about the population rather than about the wording: the graded set is derived
from the frozen geometry rather than asserted, the rival set is derived from the
data rather than listed, and the last test shows the unscoped builder still
naming rivals so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.compliance_api as compliance_api

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(compliance_api.router)
    return TestClient(app)


@pytest.fixture(scope="module")
def geometry() -> tuple:
    """The frozen break geometry the verdict is computed over."""
    from kairos_api import plan_read_guardrails

    items = plan_read_guardrails.plan_guardrail_items()
    if not items:
        pytest.skip("the break geometry does not join the saved plan, so there is nothing to scope")
    return tuple(items)


@pytest.fixture(scope="module")
def channel() -> str:
    from kairos_api.core import _load_settings

    declared = str(getattr(_load_settings(), "operator_channel", "") or "").strip()
    if not declared:
        pytest.skip("no operator channel is declared on this tree, so the scoped path is unreachable")
    return declared


@pytest.fixture(scope="module")
def rivals(geometry, channel) -> list[str]:
    """Derived from the data, so this keeps holding when the lineup changes."""
    names = sorted({str(item.channel).strip() for item in geometry} - {channel, ""})
    if not names:
        pytest.skip("the plan carries one channel, so there is no boundary to measure")
    return names


@pytest.fixture(scope="module")
def verdict(client) -> dict:
    response = client.get("/api/compliance")
    assert response.status_code == 200, response.text
    return response.json()


def test_the_seven_checks_still_return_with_their_licence(verdict):
    """The regression row: nothing this closes may take the verdict away."""
    assert len(verdict["checks"]) == 7
    assert verdict["profile"]
    assert verdict["effective_date"]
    assert verdict["source_url"]
    assert verdict["disclaimer"]
    assert verdict["status"] in ("compliant", "at_risk")
    for check in verdict["checks"]:
        assert check["label_en"] and check["label_he"]
        assert check["observed"] is not None and check["limit"] is not None
        assert check["unit"]


def test_the_verdict_is_computed_over_the_operators_own_breaks(verdict, geometry, channel):
    own = [item for item in geometry if str(item.channel).strip() == channel]
    assert verdict["graded_breaks"] == len(own)
    assert verdict["graded_breaks"] < len(geometry), "the plan carries more than this operator's channel"
    scope = verdict["scope"]
    assert scope["scoped"] is True
    assert scope["scope_channel"] == channel
    assert scope["rows_in"] == len(geometry)
    assert scope["rows_out"] == len(own)
    assert scope["competitor_rows_excluded"] == len(geometry) - len(own)
    assert scope["competitor_channels_excluded"] == len({str(item.channel).strip() for item in geometry}) - 1


def test_the_printed_retention_figure_is_the_operators_own_worst_break(verdict, geometry, channel):
    """The figure that was drawn from the wrong population, recomputed here."""
    own = [item for item in geometry if str(item.channel).strip() == channel]
    floor = next(check for check in verdict["checks"] if check["id"] == "retention_floor")
    assert floor["observed"] == round(min(item.retention for item in own) * 100, 1)


def test_no_rival_channel_reaches_the_payload_the_browser_holds(verdict, rivals):
    body = json.dumps(verdict, ensure_ascii=False)
    for name in rivals:
        assert name not in body, f"{name} is not this operator's channel and may not reach its screen"


def test_the_violations_served_are_the_frozen_verdict_over_the_owned_breaks(verdict, geometry, channel):
    """Whatever the plan breaches today, the list served is one channel's."""
    from kairos_api.core import _load_settings
    from kairos_api.plan_read_guardrails import guardrail_compliance_from_breaks

    own = [item for item in geometry if str(item.channel).strip() == channel]
    expected = guardrail_compliance_from_breaks(own, _load_settings())["violations"]
    assert verdict["violations"] == expected
    for violation in verdict["violations"]:
        assert str(violation["scope"]).startswith(channel), violation["scope"]


def test_under_a_limit_that_bites_the_market_names_rivals_and_the_operators_own_does_not(
    geometry, channel, rivals,
):
    """The defect reproduced from the data rather than from today's limits.

    The plan on disk breaches nothing as it stands: every channel's worst break
    retains 85.1 percent against a floor of 82, so both populations return an
    empty violation list and an assertion about their contents would prove
    nothing. When it did breach, 200 violations reached the operator's browser
    and 136 of them named a rival.

    So the floor is raised here, in memory, to the point where both populations
    breach, and the same raised floor grades each of them. Nothing on disk moves
    and no figure from this probe reaches a screen. What it measures is whose
    breaks each population is able to name.
    """
    import heapq

    from kairos_api.core import _load_settings
    from kairos_api.plan_read_guardrails import guardrail_compliance_from_breaks

    own_items = [item for item in geometry if str(item.channel).strip() == channel]
    rival_items = [item for item in geometry if str(item.channel).strip() != channel]
    depth = 25
    floor = max(
        heapq.nsmallest(depth, [item.retention for item in own_items])[-1],
        heapq.nsmallest(depth, [item.retention for item in rival_items])[-1],
    )
    probe = _load_settings().model_copy(update={"min_retention_floor": min(floor + 1e-9, 1.0)})

    market = guardrail_compliance_from_breaks(list(geometry), probe)["violations"]
    own = guardrail_compliance_from_breaks(own_items, probe)["violations"]
    assert market and own, "the raised floor has to make both populations breach, or this proves nothing"

    named = sorted({name for row in market for name in rivals if str(row["scope"]).startswith(name)})
    assert named, "grading the market names rivals in the payload, which is what the scoped route stops"
    assert all(str(row["scope"]).startswith(channel) for row in own), (
        "the same limit over the operator's own breaks can only name the operator's own channel"
    )


def test_the_attestation_signs_the_same_scoped_verdict(client, verdict):
    """One set of seven checks in the product, and it is the scoped one."""
    response = client.get("/api/rules/attestation")
    assert response.status_code == 200, response.text
    signed = response.json()["compliance"]
    assert signed["scope"] == verdict["scope"]
    assert signed["graded_breaks"] == verdict["graded_breaks"]
    assert [check["observed"] for check in signed["checks"]] == [
        check["observed"] for check in verdict["checks"]
    ]


def test_with_no_channel_declared_nothing_is_judged_and_the_route_is_named(client, tmp_path, monkeypatch):
    """The market total is never served as if it were the operator's."""
    import kairos_api.core as core

    copy = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", copy)
    document = json.loads(copy.read_text(encoding="utf-8"))
    document["operator_channel"] = ""
    copy.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_PATH", copy)

    body = client.get("/api/compliance").json()
    assert body["checks"] == []
    assert body["violations"] == []
    assert body["status"] == "unknown"
    assert body["graded_breaks"] == 0
    assert body["scope"]["scoped"] is False
    assert body["scope"]["supply_route"] == "PUT /api/rules/operator-channel"
    assert body["scope"]["reason_en"] and body["scope"]["reason_he"]
    # The licence half survives, because it is a fact about the record and not
    # about the plan, and the regression row asks for it either way.
    assert body["profile"] and body["effective_date"] and body["source_url"]


# Each figure is an extreme over its population, so grading a superset can only
# move it one way. A maximum cannot fall when breaks are added and a minimum
# cannot rise, which is what makes the direction below a real invariant rather
# than a restatement of today's numbers.
MARKET_SIDE = {
    "hourly_ad_load": "at_least",
    "break_density": "at_least",
    "retention_floor": "at_most",
    "protected_programs": "at_least",
    "break_spacing": "at_most",
    "daily_ad_load": "at_least",
    "gold_breaks": "at_least",
}


def test_the_market_verdict_is_a_different_population_and_the_route_serves_the_smaller_one(
    verdict, geometry, channel,
):
    """The finding was the population, so the population is what this pins.

    Measured on the plan as it stands: the market grades 8,449 breaks over four
    channels and the route grades 2,344 over one, and two of the seven figures
    move with the population, protected programme ad load 8.0 against 4.0 and
    minimum break spacing 7.0 against 7.01. The direction of every figure is
    asserted rather than its value, because a maximum over a superset cannot be
    smaller and a minimum cannot be larger, so a scoping that silently became a
    no-op would put a market figure on the operator's card and fail here. What
    is asserted exactly is the other end: every figure served equals the frozen
    verdict over the operator's own breaks, which is true whatever the plan says.
    """
    from kairos_api.core import _load_settings
    from kairos_api.plan_read_guardrails import guardrail_compliance_from_breaks

    settings = _load_settings()
    market = guardrail_compliance_from_breaks(list(geometry), settings)
    own = guardrail_compliance_from_breaks(
        [item for item in geometry if str(item.channel).strip() == channel], settings,
    )
    assert len(geometry) > verdict["graded_breaks"], "the plan carries more than this operator's channel"

    served = {check["id"]: check["observed"] for check in verdict["checks"]}
    whole = {check["id"]: check["observed"] for check in market["checks"]}
    assert set(served) == set(whole) == set(MARKET_SIDE)
    assert served == {check["id"]: check["observed"] for check in own["checks"]}, (
        "every figure on the card is the verdict over the operator's own breaks and nobody else's"
    )
    for check_id, side in MARKET_SIDE.items():
        if side == "at_least":
            assert whole[check_id] >= served[check_id], check_id
        else:
            assert whole[check_id] <= served[check_id], check_id
