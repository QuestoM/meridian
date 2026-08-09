"""The number two parties audit each other with, and nobody could compute it.

``kairos.optimize.positions.preferred_position_rate`` shipped built, tested,
bilingual and with ZERO CALLERS outside its own test file. The adversarial
re-audit of 2026-08-09 ranked it first of three findings, and its point is that
this is worse than a missing feature: the code, the tests and the docstring all
say the work is done, so nobody looks again.

This file measures that it is reachable, and measures the two things a reachable
version must not do: guess the preferred set, and pick a method.

WHAT WAS MEASURED when this was written, on the shipped traffic file
``Wally_Prime_Reshet_Example_2025-04-27.csv``:

    41 campaigns carry at least one placeable broadcast
    51 rows carry no campaign or no position and are counted as dropped
    preferred_state is "unavailable", because no set is configured on this tree

and with a set of {"1", "L"} configured for the test:

    11 of 41 campaigns obtain a real percentage
    the two counting methods DISAGREE on 2 of those 11

That last number is the whole reason the positions module refuses to answer
without naming its method. The largest disagreement is a campaign whose single
broadcast holds BOTH position 1 and Last, in a break where it is the only spot:
the agency method counts the positions obtained and answers 200 percent, the
channel method counts the broadcasts that obtained at least one and answers 100.
Neither is wrong. An unlabelled percentage would be.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import break_api_pod as pod
from kairos_api import preferred_rate

pytest.importorskip("pandas")


@pytest.fixture()
def client():
    from kairos_api.break_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def day():
    days = pod.covered_days()
    if not days:
        pytest.skip("no traffic file is loaded on this tree")
    return days[0]


def test_the_function_now_has_a_caller_that_is_not_its_own_test():
    """The premise. If this ever passes trivially the seam has been deleted."""
    import inspect

    source = inspect.getsource(preferred_rate)
    assert "preferred_position_rate(" in source
    assert "AGENCY_METHOD" in source and "CHANNEL_METHOD" in source


def test_the_route_answers_and_names_the_day_it_read(client, day):
    body = client.get("/api/preferred-position-rate").json()
    assert body["day"] == day
    assert day in body["covered_days"]
    assert isinstance(body["campaigns"], list) and body["campaigns"]


def test_with_no_configured_set_it_computes_nothing_and_says_so(client):
    """The honest state on this tree, and the one it must not paper over.

    An unset preferred set is not an occasion to fall back on the trade default.
    Both methods answer basis "unset" with percent None, which a surface renders
    as unavailable rather than as a zero that looks like a result.
    """
    body = client.get("/api/preferred-position-rate").json()
    assert body["preferred_state"] in {"unavailable", "unreadable"}
    assert body["preferred_set"] is None
    for row in body["campaigns"]:
        for method in ("agency", "channel"):
            assert row[method]["basis"] == "unset"
            assert row[method]["percent"] is None


def test_both_methods_always_travel_and_both_carry_their_own_bilingual_label(client):
    """One number without its method is what the positions module exists to stop."""
    body = client.get("/api/preferred-position-rate").json()
    for row in body["campaigns"]:
        assert row["agency"]["method"] != row["channel"]["method"]
        for method in ("agency", "channel"):
            assert row[method]["method_label_en"] and row[method]["method_label_he"]
            # The Hebrew label is Hebrew, not the English one copied across.
            assert any("֐" <= character <= "׿" for character in row[method]["method_label_he"])


def test_a_configured_set_produces_real_percentages_and_the_methods_disagree(monkeypatch, day):
    """The branch that never runs on this tree, driven so the seam is proven.

    Every assertion above holds while the set is unset, which means all of them
    would still pass if this seam computed nothing at all. This one configures a
    set and requires real numbers out of it.
    """
    monkeypatch.setattr(
        preferred_rate, "preferred_reading",
        lambda: {"codes": frozenset({"1", "L"}), "state": "real"},
    )
    body = preferred_rate.rates_for_day(day)
    assert body["preferred_set"] == ["1", "L"]
    real = [row for row in body["campaigns"] if row["agency"]["percent"] is not None]
    assert real, "a configured set produced no percentage anywhere, so nothing is being counted"
    for row in real:
        assert row["agency"]["basis"] == "real"
        assert list(row["agency"]["preferred"]) == ["1", "L"]
    disagree = [row for row in real if row["agency"]["percent"] != row["channel"]["percent"]]
    assert disagree, (
        "the two counting methods agreed on every campaign, which means the "
        "distinction the trade audits with is not being computed"
    )
    # A broadcast can hold two preferred positions at once, which is why the
    # agency method can exceed 100. It is the trade's own arithmetic, not a bug.
    assert max(row["agency"]["percent"] for row in disagree) > 100.0


def test_rows_it_could_not_place_are_counted_rather_than_dropped_quietly(client):
    """A denominator that quietly shrank is a lie about coverage."""
    body = client.get("/api/preferred-position-rate").json()
    dropped = body["rows_without_a_campaign_or_position"]
    assert isinstance(dropped, int) and dropped >= 0
    # The shipped file really does carry unpositioned billboards, so a zero here
    # would mean the count is not being taken rather than that nothing was
    # dropped. Measured at 51 when this was written.
    assert dropped > 0, "no row was dropped, so this count is not being taken"


def test_the_break_identity_is_the_pods_own_and_not_a_second_answer(day):
    """Two answers to "what is one break" is the defect the pod module prevents."""
    by_campaign, _ = preferred_rate._appearances(day)
    pod_ids = {record["pod_id"] for record in pod.pods_for_day(day)}
    used = {appearance.break_id for rows in by_campaign.values() for appearance in rows}
    assert used, "no appearance was placed in any break"
    assert used <= pod_ids, "an appearance names a break the pod board does not have"
