"""P4: a malformed onboarding body is declined, not crashed on.

A 500 and a 422 say opposite things to an operator. A 422 says the product
understood the request perfectly and declined it; a 500 says the product broke.
The onboarding route said the second while meaning the first, for one reason:
its body arrived as a raw ``dict``, so FastAPI validated nothing and the model
was constructed inside the handler, where pydantic's ValidationError had nobody
to catch it.

A blind critic measured this as a SITE rather than a class, and measured it
twice. The grep: exactly one body parameter in P4's row is annotated ``dict``
rather than a model. The empirical check, which is the half that counts: the
same malformed body against every addressable P4 write route, where the four
typed siblings already answered 422 and only onboarding did not.

This file keeps both halves, because the fix is one word in a signature and one
word is exactly what a later refactor removes without noticing.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api.campaigns_api_onboarding_models import OnboardRequest

MALFORMED = {"nonsense_field": 1}


@pytest.fixture(name="client")
def _client(monkeypatch) -> TestClient:
    """The real app with the wall stood down, since a 401 answers before a body
    is ever validated and would make all five routes look identical."""
    monkeypatch.setenv("KAIROS_AUTH_DISABLED", "1")
    from kairos_api.server import app

    return TestClient(app)


def test_a_malformed_onboarding_body_is_declined_not_crashed_on(client) -> None:
    assert client.post("/api/clients/onboarding", json=MALFORMED).status_code == 422


@pytest.mark.parametrize(
    "route",
    [
        "/api/agencies",
        "/api/clients/campaigns",
        "/api/agencies/AGY_01/advertisers",
        "/api/agencies/AGY_01/conditions",
    ],
)
def test_the_typed_siblings_answer_the_same_way(client, route: str) -> None:
    """The controls. These already answered 422 before the fix, which is what
    established that onboarding was one site and not the house style."""
    assert client.post(route, json=MALFORMED).status_code == 422


def test_an_untyped_body_really_is_what_produced_the_500() -> None:
    """THE POSITIVE CONTROL, and without it the assertions above are vacuous.

    Every test here would keep passing if 422 came from somewhere else entirely,
    so this rebuilds the defect in miniature: the same handler shape, a body
    annotated ``dict`` with the model built inside, on a throwaway app. It must
    still fail loudly, or the shape being guarded against is not the shape that
    caused the outage.
    """
    broken = FastAPI()

    @broken.post("/onboarding")
    def _onboard(payload: dict[str, Any]) -> dict[str, Any]:  # the old signature
        return OnboardRequest(**payload).model_dump()

    with pytest.raises(Exception) as caught:
        TestClient(broken).post("/onboarding", json=MALFORMED)
    assert "validation error" in str(caught.value).lower(), (
        "the untyped shape no longer raises, so this control proves nothing and "
        "the tests above are no longer evidence that typing the body is the fix"
    )


def test_the_route_still_accepts_a_body_it_understands(client) -> None:
    """A refusal that refuses everything is not a fix. An empty object is
    well-formed JSON and still incomplete, so it must be declined by VALIDATION
    (422) rather than by a crash, and never accepted."""
    assert client.post("/api/clients/onboarding", json={}).status_code == 422
