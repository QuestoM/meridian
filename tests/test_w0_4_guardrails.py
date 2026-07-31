"""The regulatory guardrails in their own store: date, record, permission.

Two things are pinned here. The first is that nothing moved: the store holds
exactly the four values the settings model ships, and the overlay that will one
day replace them is an exact identity today. The second is that the three
things the settings document could not give them now exist and work.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from kairos_api import auth_store, guardrail_store
from kairos_api.core import KairosSettings, _load_settings, _model_dump

ROOT = Path(__file__).resolve().parents[1]

ADMIN_PASSWORD = "rootpass-1234"
OPERATOR_PASSWORD = "operatorpass-123"


@pytest.fixture()
def store_path(tmp_path, monkeypatch):
    """Relocate the store so no test can touch data/regulatory_guardrails.json."""
    path = tmp_path / "regulatory_guardrails.json"
    monkeypatch.setenv(guardrail_store.PATH_ENV, str(path))
    path.write_text(
        json.dumps(guardrail_store._seed_record(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Nothing moved
# ---------------------------------------------------------------------------

def test_the_shipped_store_holds_exactly_the_settings_values():
    """The file on disk is the licence as the product ships it, to the digit."""
    shipped = json.loads((ROOT / "data" / "regulatory_guardrails.json").read_text(encoding="utf-8"))
    defaults = KairosSettings()
    assert shipped["baseline"]["values"] == {
        "max_ad_minutes_per_hour": 12.0,
        "max_breaks_per_hour": 4,
        "min_break_spacing_minutes": 7,
        "protected_program_max_ad_minutes_per_hour": 8.0,
    }
    for key, value in shipped["baseline"]["values"].items():
        assert getattr(defaults, key) == value, f"{key} drifted from the settings model"
    assert shipped["baseline"]["effective_date"] == defaults.effective_date
    assert shipped["source_url"] == defaults.regulatory_source_url
    assert shipped["changes"] == []


def test_the_overlay_is_an_exact_identity_on_the_saved_settings(store_path):
    """The one-line cutover moves no number while the two stores agree."""
    saved = _load_settings()
    assert _model_dump(guardrail_store.settings_overlay(saved)) == _model_dump(saved)

    defaults = KairosSettings()
    assert _model_dump(guardrail_store.settings_overlay(defaults)) == _model_dump(defaults)
    as_dict = _model_dump(saved)
    assert guardrail_store.settings_overlay(as_dict) == as_dict


def test_an_absent_or_unreadable_store_serves_the_shipped_baseline(tmp_path, monkeypatch):
    missing = tmp_path / "not-written-yet.json"
    monkeypatch.setenv(guardrail_store.PATH_ENV, str(missing))
    assert guardrail_store.current_values() == {
        key: getattr(KairosSettings(), key) for key in guardrail_store.GUARDRAIL_KEYS
    }
    missing.write_text("{ this is not json", encoding="utf-8")
    assert guardrail_store.current_values()["max_ad_minutes_per_hour"] == 12.0


# ---------------------------------------------------------------------------
# The effective date and the change record
# ---------------------------------------------------------------------------

def test_a_change_carries_a_date_and_does_not_apply_before_it(store_path):
    change = guardrail_store.record_change(
        {"max_ad_minutes_per_hour": 11.0},
        effective="2026-09-01",
        actor="admin",
        reason="regulator circular",
    )
    assert change["effective_date"] == "2026-09-01"
    assert change["before"] == {"max_ad_minutes_per_hour": 12.0}
    assert change["values"] == {"max_ad_minutes_per_hour": 11.0}
    assert change["actor"] == "admin"
    assert change["reason"] == "regulator circular"
    assert change["recorded_at"]

    day_before = date(2026, 8, 31)
    assert guardrail_store.values_on(day_before)["max_ad_minutes_per_hour"] == 12.0
    assert guardrail_store.effective_date(day_before) == "2026-06-14"
    pending = guardrail_store.scheduled_changes(day_before)
    assert [item["effective_date"] for item in pending] == ["2026-09-01"]

    on_the_day = date(2026, 9, 1)
    assert guardrail_store.values_on(on_the_day)["max_ad_minutes_per_hour"] == 11.0
    assert guardrail_store.effective_date(on_the_day) == "2026-09-01"
    assert guardrail_store.scheduled_changes(on_the_day) == []
    # The other three limits are untouched by a change that named one.
    later = guardrail_store.values_on(on_the_day)
    assert later["max_breaks_per_hour"] == 4
    assert later["min_break_spacing_minutes"] == 7
    assert later["protected_program_max_ad_minutes_per_hour"] == 8.0


def test_the_log_is_append_only_and_answers_the_attestation(store_path):
    guardrail_store.record_change({"max_breaks_per_hour": 3}, effective="2026-06-20", actor="a")
    guardrail_store.record_change({"min_break_spacing_minutes": 8}, effective="2026-07-20", actor="b")
    log = guardrail_store.changes()
    assert [item["actor"] for item in log] == ["a", "b"]
    assert len(guardrail_store.changed_since(date(1970, 1, 1))) == 2
    assert guardrail_store.changed_since(date.today().replace(year=date.today().year + 1)) == []
    # Both changes are in force by the later date, in effective order.
    values = guardrail_store.values_on(date(2026, 8, 1))
    assert values["max_breaks_per_hour"] == 3
    assert values["min_break_spacing_minutes"] == 8


def test_a_value_outside_the_licence_or_a_bad_date_is_refused(store_path):
    with pytest.raises(guardrail_store.GuardrailError):
        guardrail_store.record_change({"max_breaks_per_hour": 99}, effective="2026-09-01")
    with pytest.raises(guardrail_store.GuardrailError):
        guardrail_store.record_change({"revenue_weight": 90}, effective="2026-09-01")
    with pytest.raises(guardrail_store.GuardrailError):
        guardrail_store.record_change({}, effective="2026-09-01")
    with pytest.raises(guardrail_store.GuardrailError):
        guardrail_store.record_change({"max_breaks_per_hour": 3}, effective="soon")
    assert guardrail_store.changes() == [], "a refused change leaves no trace"


# ---------------------------------------------------------------------------
# The distinct permission
# ---------------------------------------------------------------------------

def test_the_permission_is_distinct_from_the_revenue_slider(tmp_path, monkeypatch, store_path):
    """An operator may move the revenue weight and may not move the licence."""
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from kairos_api.server import app

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    try:
        auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
        admin = TestClient(app)
        assert admin.post("/api/auth/login", json={
            "username": "admin", "password": ADMIN_PASSWORD}).status_code == 200
        assert admin.post("/api/auth/users", json={
            "username": "op1", "password": OPERATOR_PASSWORD, "role": "operator",
            "display_name": "op1", "must_change_password": False, "affiliation": "company",
        }).status_code == 201
        operator = TestClient(app)
        assert operator.post("/api/auth/login", json={
            "username": "op1", "password": OPERATOR_PASSWORD}).status_code == 200

        class _Req:
            def __init__(self, client):
                self.cookies = {auth_store.COOKIE_NAME: client.cookies[auth_store.COOKIE_NAME]}

        operator_request = _Req(operator)
        admin_request = _Req(admin)

        # The operator role is a write role, which is what moves the slider.
        assert operator_request.cookies[auth_store.COOKIE_NAME]
        assert guardrail_store.GUARDRAIL_WALL.allows(operator_request) is False
        assert guardrail_store.GUARDRAIL_WALL.reason(operator_request) == (
            guardrail_store.GUARDRAIL_ADMIN_ONLY_DETAIL
        )
        with pytest.raises(HTTPException) as caught:
            guardrail_store.require_guardrail_editor(operator_request)
        assert caught.value.status_code == 403

        assert guardrail_store.GUARDRAIL_WALL.allows(admin_request) is True
        guardrail_store.require_guardrail_editor(admin_request)

        # The read payload says so before the click, in the same words.
        refused = guardrail_store.payload(operator_request)
        assert refused["can_edit"] is False
        assert refused["can_edit_reason"] == guardrail_store.GUARDRAIL_ADMIN_ONLY_DETAIL
        allowed = guardrail_store.payload(admin_request)
        assert allowed["can_edit"] is True
        assert "can_edit_reason" not in allowed
        assert allowed["values"]["max_ad_minutes_per_hour"] == 12.0
        assert allowed["effective_date"] == "2026-06-14"
        assert allowed["source_url"] == "https://www.rashut2.org.il/"
    finally:
        auth_store.reset_runtime_state()


def test_the_payload_is_readable_with_login_off(store_path):
    body = guardrail_store.payload(None)
    assert body["can_edit"] is True
    assert set(body["values"]) == set(guardrail_store.GUARDRAIL_KEYS)
    assert body["changes"] == []
    assert body["scheduled_changes"] == []
