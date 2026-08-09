"""The optional caps as the compliance surface reports them: off must be honest.

Split from :mod:`tests.test_airtime_caps` under the 450-line law; that file
proves the caps bite in the engine, this one proves the read-out never claims a
rule ran when it did not.

The rule these all defend: a cap that is not enforced must never render as
though it were, and a plan produced without one must not carry a badge earned by
a rule nobody ran. Every window bound and fraction below is a chosen input, not a
regulatory figure.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos.optimize.guardrails import CAP_ABSENT, CAP_AVAILABLE, CAP_ENFORCED
from kairos_api import compliance_api
from kairos_api.airtime_cap_settings import WindowAdCapSettings
from kairos_api.core import KairosSettings
from kairos_api.plan_read_guardrails import guardrail_compliance_from_breaks
from tests.test_airtime_caps import make_break

ABSENT_BOTH = {"window_ad_load": CAP_ABSENT, "day_fraction_ad_load": CAP_ABSENT}


def evening_breaks():
    """Sixteen breaks across 20:00-23:59, thirty-two minutes in all."""
    return [make_break(hour=h, start_seconds=h * 3600.0 + i * 720.0)
            for h in (20, 21, 22, 23) for i in range(4)]


def compliance_for(window=None, day_fraction=None) -> dict:
    settings = KairosSettings(window_ad_cap=window, day_fraction_ad_cap=day_fraction)
    return guardrail_compliance_from_breaks(evening_breaks(), settings)


def test_an_absent_cap_never_grades_compliant_and_never_reads_as_zero() -> None:
    verdict = compliance_for()
    assert verdict["cap_states"] == ABSENT_BOTH
    rows = {row["id"]: row for row in verdict["optional_caps"]}
    assert set(rows) == {"window_ad_load", "day_fraction_ad_load"}
    for row in rows.values():
        assert row["status"] == CAP_ABSENT
        assert row["status"] not in ("compliant", "at_risk")
        # Unavailable, not zero: there is no window and no fraction to measure.
        assert row["observed"] is None and row["limit"] is None
    # An absent cap is not one of the rules that graded this plan.
    assert not any(check["id"] in rows for check in verdict["checks"])
    assert len(verdict["checks"]) == 7


def test_an_available_cap_reports_what_it_would_have_found_without_grading() -> None:
    verdict = compliance_for(window=WindowAdCapSettings(
        enabled=False, start_hour=20, end_hour=24, max_ad_minutes=10.0))
    row = next(r for r in verdict["optional_caps"] if r["id"] == "window_ad_load")
    assert verdict["cap_states"]["window_ad_load"] == CAP_AVAILABLE
    assert row["status"] == CAP_AVAILABLE and row["status"] not in ("compliant", "at_risk")
    assert row["observed"] == 32.0 and row["limit"] == 10.0
    assert row["would_breach"] is True
    # It reported, but it did not grade: the plan carries no verdict from it.
    assert row["violations"] == 0
    assert not any(check["id"] == "window_ad_load" for check in verdict["checks"])
    assert len(verdict["checks"]) == 7
    assert verdict["status"] == "compliant"


def test_an_enforced_cap_grades_and_joins_the_checks_that_ran() -> None:
    verdict = compliance_for(window=WindowAdCapSettings(
        enabled=True, start_hour=20, end_hour=24, max_ad_minutes=10.0))
    assert verdict["cap_states"]["window_ad_load"] == CAP_ENFORCED
    row = next(r for r in verdict["checks"] if r["id"] == "window_ad_load")
    assert row["status"] == "at_risk" and row["violations"] == 1
    assert row["observed"] == 32.0 and row["limit"] == 10.0
    assert verdict["status"] == "at_risk"
    assert len(verdict["checks"]) == 8


def test_every_check_that_ran_declares_that_it_ran() -> None:
    verdict = compliance_for()
    assert all(check["cap_state"] == CAP_ENFORCED for check in verdict["checks"])


def test_the_enforced_row_still_satisfies_the_licence_shape() -> None:
    """An enforced cap joins ``checks``, so it must carry what a check carries."""
    verdict = compliance_for(window=WindowAdCapSettings(
        enabled=True, start_hour=20, end_hour=24, max_ad_minutes=10.0))
    for check in verdict["checks"]:
        assert check["label_en"] and check["label_he"]
        assert check["observed"] is not None and check["limit"] is not None
        assert check["unit"]


# --------------------------------------------------------------------------
# The route itself must disclose. A builder that knows and a route that drops
# it on the floor is the same silence.
# --------------------------------------------------------------------------

def test_the_compliance_route_says_which_caps_ran() -> None:
    app = FastAPI()
    app.include_router(compliance_api.router)
    response = TestClient(app).get("/api/compliance")
    assert response.status_code == 200, response.text
    payload = response.json()
    # Shipped settings configure neither cap, so both must read absent here.
    assert payload["cap_states"] == ABSENT_BOTH
    assert {row["id"] for row in payload["optional_caps"]} == set(ABSENT_BOTH)
    for row in payload["optional_caps"]:
        assert row["status"] == CAP_ABSENT
        assert row["observed"] is None and row["limit"] is None
    # And the disclosure did not disturb the seven the licence already had.
    assert len(payload["checks"]) == 7
