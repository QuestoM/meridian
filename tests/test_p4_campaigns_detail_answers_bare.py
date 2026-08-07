"""P4: the rollup drill answers a bare call instead of refusing it.

The measured defect. ``GET /api/campaigns/detail`` declared ``campaign: str``
with no default, so FastAPI answered a call with no arguments 422 with a
validation body. ``tests/test_api_surface_qa.py::test_every_get_endpoint_is_healthy``
calls every GET on this product bare and requires a healthy answer, and this was
the one route that refused. The campaigns wave reintroduced exactly the defect a
repair wave had closed on ``/api/history/since`` and on the restrictions airings
route, and the fix accepted there is the one copied here: a default on the query
parameter, and an honest empty answer that names the input it is waiting for.

The empty answer is held to the same honest-math rule as everything else on this
destination. It carries no figure nobody computed: ``revenue_available`` and
``scope`` are facts about a source this answer never opened, so they are null
rather than true, and ``count`` is zero because nothing was asked for rather
than because a campaign was found to have nothing.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from kairos_api import campaigns_read

    app = FastAPI()
    app.include_router(campaigns_read.router)
    return TestClient(app)


def test_a_call_with_no_campaign_is_answered_and_not_refused(client):
    """The exact request the surface sweep makes, and the status it must get."""
    response = client.get("/api/campaigns/detail")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 0
    assert body["spots"] == []
    assert body["campaign"] == ""


def test_the_empty_answer_names_the_input_it_is_waiting_for(client):
    """A state a caller can render. A validation body is not one."""
    body = client.get("/api/campaigns/detail").json()
    assert body["reason"] == "Name a campaign to list the spots behind it."
    assert body["reason_he"], "the reader of this product reads Hebrew"
    assert any("א" <= ch <= "ת" for ch in body["reason_he"])


def test_the_empty_answer_states_nothing_about_a_source_it_never_read(client):
    """No zeroed figure and no fact about a file nobody opened."""
    body = client.get("/api/campaigns/detail").json()
    assert body["revenue_available"] is None, "true here is a claim about an unread source"
    assert body["scope"] is None, "the competitor scope is a fact about a frame nobody loaded"


def test_a_blank_campaign_is_the_same_state_as_a_missing_one(client):
    """An empty string is a missing input, not a campaign named the empty string."""
    body = client.get("/api/campaigns/detail", params={"campaign": "   "}).json()
    assert body["count"] == 0 and body["reason"]


def test_a_named_campaign_still_reads_the_ledger(client):
    """The empty case is a branch, not a replacement for the route's job."""
    body = client.get("/api/campaigns/detail", params={"campaign": "no such campaign"}).json()
    assert body["count"] == 0
    assert "reason" not in body, "a campaign that was looked up and found nothing is a different state"
    assert body["revenue_available"] is not None or body["spots"] == [], (
        "a real lookup reports what the source really carries"
    )
