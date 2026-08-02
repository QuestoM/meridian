"""P2: the fourth open read, closed by narrowing it rather than by walling it.

Section 4.5 of the specification lists ``GET /api/parameters`` beside
``/api/impact``, ``/api/model/audience`` and the calendar's ``model_context`` as
a read that leaks the training side to a channel-affiliated account. W0-4's own
contract says it did not close this one, because the route is served from a
module that piece does not own; W0-1's contract reassigns the route to P2.

Walling the whole route would be the wrong fix. The parameters are the
operator's own settings and their own rate card, and an operator who cannot read
them cannot work. What leaks is exactly three keys, and section 4.2's lexicon
test is what names them: ``coefficient_freshness`` carries the lexicon word
itself and names the training inputs that changed, and ``first_break_active``
with ``first_break_multiplier`` are a gate verdict and its coefficient under
other names.

So the fix is the one the calendar takes: the three keys are company-only, and
every account gets ``model_version`` instead, which is the two facts section 4.4
says a run surface needs.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kairos_api import auth_store
from kairos_api.server import app

# Section 4.2's lexicon, verbatim.
TRAINING_LEXICON = (
    "gate", "held_out", "tau", "drift", "coefficient",
    "pooling", "p_value", "training_window", "wartime",
)

ADMIN_PASSWORD = "rootpass-1234"
CHANNEL_PASSWORD = "channelpass-123"
COMPANY_PASSWORD = "companypass-123"


@pytest.fixture()
def sessions(tmp_path, monkeypatch):
    """A real channel account and a real company account, on live sessions."""
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)

    admin = TestClient(app)
    signed = admin.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD})
    assert signed.status_code == 200, signed.text
    for username, password, affiliation in (
        ("chan2", CHANNEL_PASSWORD, "channel"),
        ("comp2", COMPANY_PASSWORD, "company"),
    ):
        created = admin.post("/api/auth/users", json={
            "username": username, "password": password, "role": "operator",
            "display_name": username, "must_change_password": False,
            "affiliation": affiliation,
        })
        assert created.status_code == 201, created.text

    clients = {}
    for username, password in (("chan2", CHANNEL_PASSWORD), ("comp2", COMPANY_PASSWORD)):
        client = TestClient(app)
        response = client.post("/api/auth/login", json={"username": username, "password": password})
        assert response.status_code == 200, response.text
        clients[username] = client
    yield clients
    auth_store.reset_runtime_state()


def test_a_channel_account_reads_the_parameters_it_needs(sessions):
    """The route is not walled: an operator still gets their own settings."""
    response = sessions["chan2"].get("/api/parameters")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "settings" in body
    assert "pricing" in body
    assert body["training_visible"] is False


def test_a_channel_account_gets_no_training_content_at_all(sessions):
    """Section 4.2's lexicon test, run on the payload a run surface receives."""
    body = sessions["chan2"].get("/api/parameters").json()
    text = json.dumps(body, ensure_ascii=False).lower()
    hits = [word for word in TRAINING_LEXICON if word in text]
    assert hits == [], hits
    assert "coefficient_freshness" not in body
    assert "first_break_active" not in body
    assert "first_break_multiplier" not in body


def test_a_channel_account_still_gets_the_two_model_facts_a_run_surface_may_show(sessions):
    """Section 4.4: which model version this plan used, and whether it is current."""
    body = sessions["chan2"].get("/api/parameters").json()
    version = body["model_version"]
    assert set(version) == {"trained_at", "current", "status"}
    assert isinstance(version["current"], bool)
    # No verdict, no coverage, no coefficient, no p-value.
    assert json.dumps(version).lower().count("gate") == 0


def test_a_company_account_still_reads_the_freshness_verdict(sessions):
    """The training side keeps what it always had, so nothing was lost."""
    body = sessions["comp2"].get("/api/parameters").json()
    assert body["training_visible"] is True
    assert "coefficient_freshness" in body
    assert set(body["coefficient_freshness"]) >= {"status", "computed_at"}
    assert "first_break_active" in body
    assert "first_break_multiplier" in body


def test_the_two_copies_agree_on_everything_that_is_not_training(sessions):
    channel = sessions["chan2"].get("/api/parameters").json()
    company = sessions["comp2"].get("/api/parameters").json()
    shared = set(channel) & set(company)
    for key in sorted(shared):
        assert channel[key] == company[key] or key == "training_visible", key
    # The company copy is the channel copy plus exactly the three training keys.
    assert set(company) - set(channel) == {
        "coefficient_freshness", "first_break_active", "first_break_multiplier",
    }
