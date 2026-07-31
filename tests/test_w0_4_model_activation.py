"""The audience model activation switch: run-side, company-gated, honest read.

The switch decides where every forward-dated rating comes from, so it moves
money. It is configuration, not training, because throwing it writes data/ and
marks the saved plan out of date. These tests pin the gate, the payload the
operator surface may render, and the rule that the payload never carries a word
from the training lexicon.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import kairos_api.core as core
from kairos_api import auth_store, model_activation

ROOT = Path(__file__).resolve().parents[1]

ADMIN_PASSWORD = "rootpass-1234"
COMPANY_PASSWORD = "companypass-123"
CHANNEL_PASSWORD = "channelpass-123"

# Section 4.2's lexicon test: a run surface returns zero hits for any of these.
TRAINING_LEXICON = (
    "gate",
    "held_out",
    "tau",
    "drift",
    "coefficient",
    "pooling",
    "p_value",
    "training_window",
    "wartime",
)


@pytest.fixture()
def settings_copy(tmp_path, monkeypatch):
    """A private settings file, so no test can move the deployed switch."""
    path = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", path)
    monkeypatch.setattr(core, "SETTINGS_PATH", path)
    return path


@pytest.fixture()
def signed_in(tmp_path, monkeypatch, settings_copy):
    from kairos_api.server import app

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    admin = TestClient(app)
    assert admin.post("/api/auth/login", json={
        "username": "admin", "password": ADMIN_PASSWORD}).status_code == 200
    for username, password, affiliation in (
        ("comp1", COMPANY_PASSWORD, "company"),
        ("chan1", CHANNEL_PASSWORD, "channel"),
    ):
        assert admin.post("/api/auth/users", json={
            "username": username, "password": password, "role": "operator",
            "display_name": username, "must_change_password": False,
            "affiliation": affiliation,
        }).status_code == 201

    class _Req:
        def __init__(self, token):
            self.cookies = {auth_store.COOKIE_NAME: token}

    def request_for(username: str, password: str):
        client = TestClient(app)
        assert client.post("/api/auth/login", json={
            "username": username, "password": password}).status_code == 200
        return _Req(client.cookies[auth_store.COOKIE_NAME])

    yield request_for
    auth_store.reset_runtime_state()


def test_the_payload_carries_the_switch_its_basis_and_can_edit(settings_copy):
    body = model_activation.payload(None)
    assert body["field"] == "audience_model_activation"
    assert body["active"] is False
    assert body["state"] in ("off", "on", "on_no_artifact", "unknown")
    assert body["can_edit"] is True
    assert body["consequence_he"]
    assert body["consequence_en"]
    assert model_activation.is_active() is False


def test_the_payload_passes_the_lexicon_test(settings_copy):
    """A run surface returns zero hits for the training lexicon."""
    rendered = json.dumps(model_activation.payload(None), ensure_ascii=False).lower()
    for word in TRAINING_LEXICON:
        assert word not in rendered, f"{word} is a training word and may not ride a run payload"


def test_a_channel_account_reads_the_state_and_may_not_throw_it(signed_in, settings_copy):
    channel = signed_in("chan1", CHANNEL_PASSWORD)
    body = model_activation.payload(channel)
    assert body["can_edit"] is False
    assert body["can_edit_reason"] == model_activation.AUDIENCE_MODEL_COMPANY_ONLY_DETAIL
    assert body["state"] is not None, "the state is readable, only the control is walled"

    with pytest.raises(HTTPException) as caught:
        model_activation.require_activation_editor(channel)
    assert caught.value.status_code == 403
    assert caught.value.detail == model_activation.AUDIENCE_MODEL_COMPANY_ONLY_DETAIL

    with pytest.raises(HTTPException):
        model_activation.set_active(True, channel)
    saved = json.loads(settings_copy.read_text(encoding="utf-8"))
    assert saved["audience_model_activation"] is False, "a refused write leaves no trace"


def test_the_denial_is_worded_like_the_pricing_switch_beside_it():
    from kairos_api.events_access import EVENT_PRICING_COMPANY_ONLY_DETAIL

    assert model_activation.AUDIENCE_MODEL_COMPANY_ONLY_DETAIL == "הפעלת מודל הקהל שמורה לצוות החברה"
    assert EVENT_PRICING_COMPANY_ONLY_DETAIL == "הפעלת תמחור אירועים שמורה לצוות החברה"


def test_a_company_account_throws_it_and_the_saved_settings_move(signed_in, settings_copy):
    company = signed_in("comp1", COMPANY_PASSWORD)
    after = model_activation.set_active(True, company)
    assert after["active"] is True
    assert after["can_edit"] is True
    assert json.loads(settings_copy.read_text(encoding="utf-8"))["audience_model_activation"] is True
    assert model_activation.payload(company)["state"] in ("on", "on_no_artifact")

    model_activation.set_active(False, company)
    assert json.loads(settings_copy.read_text(encoding="utf-8"))["audience_model_activation"] is False
