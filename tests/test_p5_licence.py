"""P5: the licence, the attestation, the two declarations and the rate-card delta.

The compliance verdict already worked. What did not exist was the second half of
the compliance owner's job: proving the limits it was judged against are the
current ones and that none of them moved. These tests hold the join between the
licence store and the engine, because a licence that says one thing while the
optimizer runs another is worse than no licence at all.

They also hold the two writes that had no permission on the surface that threw
them, and the rate-card delta, whose only real claim is that re-pricing the plan
under the card as saved reproduces the plan's own money exactly.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.core as core
import kairos_api.guardrail_store as guardrail_store
import kairos_api.version_store as vs

ROOT = Path(__file__).resolve().parents[1]

# The channel the golden plan is scoped to, pinned into the copy rather than
# read out of it. The deployed document is writable by any client of
# `PUT /api/settings` and was measured empty on 2026-08-01, which is the defect
# this file tests the fix for and not a fixture it can stand on.
OPERATOR_CHANNEL = "רשת 13"


@pytest.fixture(autouse=True)
def relocated(tmp_path, monkeypatch):
    """Every store this surface writes moves into tmp, including settings."""
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setenv(guardrail_store.PATH_ENV, str(tmp_path / "regulatory_guardrails.json"))
    settings_copy = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", settings_copy)
    document = json.loads(settings_copy.read_text(encoding="utf-8"))
    document["operator_channel"] = OPERATOR_CHANNEL
    settings_copy.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_PATH", settings_copy)
    core._load_settings.cache_clear() if hasattr(core._load_settings, "cache_clear") else None
    return tmp_path


@pytest.fixture()
def client() -> TestClient:
    import kairos_api.compliance_api as compliance_api

    app = FastAPI()
    app.include_router(compliance_api.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Bar 3: the verdict itself does not move.


def test_the_seven_checks_still_carry_their_profile_date_and_source(client):
    body = client.get("/api/compliance").json()
    assert body["profile"] == "Israel commercial TV"
    assert body["effective_date"] == "2026-06-14"
    assert body["source_url"] == "https://www.rashut2.org.il/"
    assert len(body["checks"]) == 7
    assert body["status"] == "compliant"
    assert body["disclaimer"]


def test_the_attestation_serves_the_same_seven_checks_and_not_a_second_set(client):
    verdict = client.get("/api/compliance").json()
    attestation = client.get("/api/rules/attestation").json()
    assert attestation["compliance"] == verdict, (
        "one set of checks in the product, or two surfaces can disagree about compliance"
    )


# ---------------------------------------------------------------------------
# The attestation answer, and the divergence it refuses to hide.


def test_an_untouched_store_answers_the_attestation_with_evidence_not_a_blank(client):
    body = client.get("/api/rules/attestation").json()
    assert body["unchanged"] is True
    assert body["changes_since"] == []
    assert body["since_is_whole_log"] is True
    assert body["engine_matches_licence"] is True
    assert body["licence"]["effective_date"] == "2026-06-14"
    assert body["licence"]["values"]["max_ad_minutes_per_hour"] == 12.0


def test_a_recorded_change_shows_up_in_the_attestation_with_who_and_why(client):
    response = client.post("/api/rules/guardrails", json={
        "values": {"max_breaks_per_hour": 3},
        "effective_date": date.today().isoformat(),
        "reason": "regulator circular",
    })
    assert response.status_code == 200, response.text
    body = client.get("/api/rules/attestation").json()
    assert body["unchanged"] is False
    assert len(body["changes_since"]) == 1
    change = body["changes_since"][0]
    assert change["values"] == {"max_breaks_per_hour": 3}
    assert change["before"] == {"max_breaks_per_hour": 4}
    assert change["reason"] == "regulator circular"


def test_a_change_in_force_is_written_through_so_the_engine_agrees(client):
    client.post("/api/rules/guardrails", json={
        "values": {"max_breaks_per_hour": 3},
        "effective_date": date.today().isoformat(),
        "reason": "regulator circular",
    })
    body = client.get("/api/rules/attestation").json()
    assert body["engine_values"]["max_breaks_per_hour"] == 3, (
        "a licence in force that the optimizer does not read is a licence in name only"
    )
    assert body["engine_matches_licence"] is True


def test_a_future_dated_change_moves_no_number_today_and_is_named_as_pending(client):
    later = (date.today() + timedelta(days=30)).isoformat()
    client.post("/api/rules/guardrails", json={
        "values": {"max_breaks_per_hour": 2},
        "effective_date": later,
        "reason": "next quarter",
    })
    body = client.get("/api/rules/attestation").json()
    assert body["licence"]["values"]["max_breaks_per_hour"] == 4
    assert body["engine_values"]["max_breaks_per_hour"] == 4
    assert body["engine_matches_licence"] is True
    assert [entry["effective_date"] for entry in body["scheduled_changes"]] == [later]


def test_a_value_the_licence_cannot_hold_is_refused_and_leaves_no_trace(client):
    before = client.get("/api/rules/guardrails").json()
    response = client.post("/api/rules/guardrails", json={
        "values": {"max_breaks_per_hour": 900},
        "effective_date": date.today().isoformat(),
    })
    assert response.status_code == 400
    assert client.get("/api/rules/guardrails").json()["changes"] == before["changes"]


def test_a_bad_date_is_refused_with_the_reason(client):
    response = client.post("/api/rules/guardrails", json={
        "values": {"max_breaks_per_hour": 3}, "effective_date": "next tuesday",
    })
    assert response.status_code == 400
    assert client.get("/api/rules/attestation", params={"since": "not-a-date"}).status_code == 400


# ---------------------------------------------------------------------------
# The two declarations, and the permission each now carries on its own surface.


def test_the_guardrail_read_carries_can_edit_so_the_refusal_is_legible(client):
    body = client.get("/api/rules/guardrails").json()
    assert "can_edit" in body, "a walled control has to say so before the click"


def test_the_operator_channel_read_names_the_schedule_it_validates_against(client):
    body = client.get("/api/rules/operator-channel").json()
    assert body["operator_channel"] == "רשת 13"
    assert body["is_declared"] is True
    assert body["is_in_schedule"] is True
    assert "can_edit" in body
    assert set(body["available_channels"]) >= {"רשת 13"}


def test_a_channel_the_schedule_does_not_carry_is_refused(client):
    response = client.put("/api/rules/operator-channel", json={"operator_channel": "Channel 9"})
    assert response.status_code == 400
    assert "not a channel in the loaded schedule" in response.json()["detail"]
    assert client.get("/api/rules/operator-channel").json()["operator_channel"] == "רשת 13"


def test_a_declared_channel_is_persisted_through_the_settings_seam(client):
    options = client.get("/api/rules/operator-channel").json()["available_channels"]
    other = next(name for name in options if name != "רשת 13")
    response = client.put("/api/rules/operator-channel", json={"operator_channel": other})
    assert response.status_code == 200
    assert response.json()["operator_channel"] == other
    assert core._load_settings().operator_channel == other


def test_the_activation_switch_reads_thin_and_carries_no_training_word(client):
    body = client.get("/api/rules/model-activation").json()
    assert body["field"] == "audience_model_activation"
    assert body["state"] in ("off", "on", "on_no_artifact")
    assert "can_edit" in body
    lexicon = ("gate", "held_out", "tau", "drift", "coefficient", "pooling",
               "p_value", "training_window", "wartime")
    text = json.dumps(body, ensure_ascii=False).lower()
    assert not [word for word in lexicon if word in text], (
        "the activation payload rides a run surface and must carry no training word"
    )


def test_throwing_the_switch_saves_it_and_the_read_agrees(client):
    assert client.get("/api/rules/model-activation").json()["active"] is False
    response = client.put("/api/rules/model-activation", json={"active": True})
    assert response.status_code == 200
    assert response.json()["active"] is True
    assert client.get("/api/rules/model-activation").json()["active"] is True
    assert core._load_settings().audience_model_activation is True


# ---------------------------------------------------------------------------
# The owner's ruling of 2026-08-01: changing a licence number is company staff
# only. These drive the real routes with real sessions, because the wall's own
# unit tests prove the lock works and not that this door is locked.


def _logged_in(app, username, password, role, affiliation, admin):
    """A second client, logged in as an account the admin creates."""
    from fastapi.testclient import TestClient

    assert admin.post("/api/auth/users", json={
        "username": username, "password": password, "role": role,
        "display_name": username, "must_change_password": False,
        "affiliation": affiliation,
    }).status_code == 201
    client = TestClient(app)
    assert client.post("/api/auth/login", json={
        "username": username, "password": password}).status_code == 200
    return client


@pytest.fixture()
def walled(tmp_path, monkeypatch):
    """The real app with real auth, an admin, and a channel-affiliated admin."""
    from fastapi.testclient import TestClient

    from kairos_api import auth_store
    from kairos_api.server import app

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    try:
        auth_store.seed_initial_admin(password="Company-Admin-1")
        admin = TestClient(app)
        assert admin.post("/api/auth/login", json={
            "username": "admin", "password": "Company-Admin-1"}).status_code == 200
        channel = _logged_in(app, "chan1", "Channel-Admin-1", "admin", "channel", admin)
        yield {"admin": admin, "channel": channel}
    finally:
        auth_store.reset_runtime_state()


def test_a_channel_administrator_is_refused_the_licence_write_and_keeps_the_read(walled):
    """Affiliation is the outer gate, so an administrator of the channel is still
    refused. The read is deliberately open: the licence is the broadcaster's own
    and the person who attests to it works for the broadcaster."""
    change = {"values": {"max_breaks_per_hour": 3}, "effective_date": "2026-12-01", "reason": "test"}
    refused = walled["channel"].post("/api/rules/guardrails", json=change)
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"] == guardrail_store.GUARDRAIL_COMPANY_ONLY_DETAIL

    read = walled["channel"].get("/api/rules/guardrails")
    assert read.status_code == 200
    assert read.json()["can_edit"] is False
    assert read.json()["can_edit_reason"] == guardrail_store.GUARDRAIL_COMPANY_ONLY_DETAIL
    assert read.json()["values"]["max_breaks_per_hour"] == 4, "the refusal changed nothing"

    attested = walled["channel"].get("/api/rules/attestation")
    assert attested.status_code == 200, "the compliance owner reads their own licence"
    assert attested.json()["can_edit"] is False


def test_the_same_refusal_covers_the_apply_route_and_the_company_admin_passes_both(walled):
    assert walled["channel"].post("/api/rules/guardrails/apply").status_code == 403
    assert walled["admin"].get("/api/rules/guardrails").json()["can_edit"] is True
    assert walled["admin"].post("/api/rules/guardrails/apply").status_code == 200


def test_a_reader_who_cannot_declare_the_channel_is_not_shown_the_others(walled):
    """The declaration is the one act performed from outside the boundary, so the
    person performing it sees every name in the loaded schedule. Everybody else
    sees the declaration alone, which keeps the exception as narrow as the act."""
    from kairos_api.server import app

    planner = _logged_in(app, "plan1", "Planner-Pass-1", "operator", "channel", walled["admin"])
    body = planner.get("/api/rules/operator-channel")
    assert body.status_code == 200, "a reader still reads which channel is theirs"
    payload = body.json()
    assert payload["available_channels"] == [OPERATOR_CHANNEL]
    assert payload["lists_every_channel"] is False
    assert payload["can_edit"] is False

    every = walled["admin"].get("/api/rules/operator-channel").json()
    assert every["lists_every_channel"] is True
    rivals = set(every["available_channels"]) - {OPERATOR_CHANNEL}
    assert rivals, "the loaded schedule has to carry another channel or this proves nothing"
    text = json.dumps(payload, ensure_ascii=False)
    assert [name for name in rivals if name in text] == [], (
        "a rival channel name reached an account that cannot declare one"
    )


def test_a_channel_administrator_may_not_throw_the_audience_model_switch(walled):
    """The other company-only control on this surface, through its own route."""
    from kairos_api import model_activation

    refused = walled["channel"].put("/api/rules/model-activation", json={"active": True})
    assert refused.status_code == 403
    assert refused.json()["detail"] == model_activation.AUDIENCE_MODEL_COMPANY_ONLY_DETAIL
    state = walled["channel"].get("/api/rules/model-activation")
    assert state.status_code == 200 and state.json()["can_edit"] is False
