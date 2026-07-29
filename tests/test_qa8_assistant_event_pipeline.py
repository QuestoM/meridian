"""Assistant event-pipeline coverage: the get_event_pipeline snapshot, the
new-war keyword grounding, the company-only write gate in the propose path,
and the four-stage operational order in the system prompt.

Everything runs on seeded tmp stores or the repo's real read-only data, with
the affiliation helper mocked at the auth_store seam (the parallel permissions
build lands auth_store.is_company_user; these tests cover both its presence
and the tolerant fallback while it is absent).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import kairos_api.assistant as assistant
import kairos_api.assistant_event_pipeline as pipeline
import kairos_api.assistant_tools as tools
import kairos_api.events_api as events_api
from kairos_api import assistant_keywords, auth, auth_store
from kairos_api.assistant_read_tools import execute_read_tool

STAGES = {"events_store", "pricing_layer", "freshness", "training_gate", "permissions"}


# --- fixtures ---------------------------------------------------------------------
@pytest.fixture()
def tmp_events(tmp_path, monkeypatch):
    """The events store and the coefficients artifact on throwaway paths."""
    monkeypatch.setattr(events_api, "EVENTS_PATH", tmp_path / "calendar_events.csv")
    monkeypatch.setattr(events_api, "COEFFICIENTS_PATH", tmp_path / "coefficients.json")
    return tmp_path


def _seed_event(**overrides: Any) -> dict[str, Any]:
    fields = {"name": "מלחמת בדיקה", "type": "war", "start_date": "2025-07-01",
              "end_date": "", "intensity": 5, "price_multiplier": 0.8}
    fields.update(overrides)
    return events_api.create_event(events_api.EventCreate(**fields), request=None)


def _block(name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id="tu1", name=name, input=args)


def _event_args() -> dict[str, Any]:
    return {"action": "create", "reason": "מלחמה חדשה",
            "event": {"name": "אירוע חדש", "type": "war", "start_date": "2025-08-01",
                      "intensity": 4, "price_multiplier": 0.9}}


# --- registry ---------------------------------------------------------------------
def test_get_event_pipeline_registered_as_read_tool() -> None:
    assert "get_event_pipeline" in tools.READ_TOOL_NAMES
    assert "get_event_pipeline" not in tools.PROPOSE_TOOL_NAMES
    read_only = {schema["name"] for schema in tools.anthropic_tools(include_propose=False)}
    assert "get_event_pipeline" in read_only


# --- the snapshot: every stage on seeded stores -----------------------------------
def test_pipeline_snapshot_returns_every_stage(tmp_events) -> None:
    _seed_event()  # open-ended active war, multiplier 0.8
    _seed_event(name="חג כבוי", type="holiday", start_date="2025-04-01",
                end_date="2025-04-02", intensity=2, price_multiplier=1.0, active=False)
    _seed_event(name="ספורט", type="sport", start_date="2025-05-01",
                end_date="2025-05-03", intensity=1, price_multiplier=1.0)
    payload = execute_read_tool("get_event_pipeline", {}, "someone")
    assert STAGES <= set(payload)
    store = payload["events_store"]
    assert store["events_total"] == 3
    assert store["active_count"] == 2
    assert store["active_by_type"] == {"sport": 1, "war": 1}
    assert store["open_ended_active_count"] == 1
    assert store["open_ended_active"][0]["name"] == "מלחמת בדיקה"
    assert store["nonneutral_multiplier_active_count"] == 1
    layer = payload["pricing_layer"]
    assert layer["activation_flag"] == "pricing_activation.events"
    assert isinstance(layer["enabled"], bool)
    assert layer["nonneutral_active_count"] == 1
    assert layer["nonneutral_active_events"][0]["price_multiplier"] == pytest.approx(0.8)
    fresh = payload["freshness"]
    assert fresh["schedule_status"] in {"fresh", "stale", "unknown"}
    assert isinstance(fresh["stale_from_events"], bool)
    assert fresh["events_group_tracked"] == layer["enabled"]
    assert payload["source"] == pipeline.PIPELINE_SOURCE
    assert len(payload["operational_order"]) == 4


def test_training_gate_tristate_absent_and_present(tmp_events) -> None:
    # No coefficients artifact at all: honest unknown, in Hebrew.
    absent = execute_read_tool("get_event_pipeline", {}, None)["training_gate"]
    assert absent["verdict"] == "unknown"
    assert absent["reason"] == pipeline.GATE_UNKNOWN_REASON_HE
    assert absent["coefficients_available"] is False
    # An artifact whose metadata predates the event layer: still unknown.
    events_api.COEFFICIENTS_PATH.write_text(
        json.dumps({"metadata": {"computed_at": "2026-07-01T00:00:00Z"}}), encoding="utf-8")
    stale = execute_read_tool("get_event_pipeline", {}, None)["training_gate"]
    assert stale["verdict"] == "unknown"
    assert stale["reason"] == pipeline.GATE_UNKNOWN_REASON_HE
    assert stale["coefficients_available"] is True
    # The rebuilt artifact carries the gate verdict: passed through untouched.
    gate = {"verdict": "off", "reason": "no contrast in the training window",
            "held_out_delta_pct": 0.4, "measured_at": "2026-07-20T00:00:00Z"}
    events_api.COEFFICIENTS_PATH.write_text(
        json.dumps({"metadata": {"event_layer_gate": gate}}), encoding="utf-8")
    carried = execute_read_tool("get_event_pipeline", {}, None)["training_gate"]
    assert carried["verdict"] == "off"
    assert carried["held_out_delta_pct"] == pytest.approx(0.4)
    assert carried["measured_at"] == "2026-07-20T00:00:00Z"


def test_permissions_stage_reflects_affiliation(tmp_events, monkeypatch) -> None:
    monkeypatch.setattr(auth_store, "is_company_user",
                        lambda username: username == "company_user", raising=False)
    company = execute_read_tool("get_event_pipeline", {}, "company_user")["permissions"]
    assert company["can_propose_event_writes"] is True
    assert company["actor"] == "company_user"
    channel = execute_read_tool("get_event_pipeline", {}, "channel_user")["permissions"]
    assert channel["can_propose_event_writes"] is False
    assert channel["policy"] == pipeline.PERMISSION_POLICY_HE


# --- keyword grounding ------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "פרצה מלחמה חדשה, מה צריך לעדכן במערכת",
    "יש אירוע חדש בשבוע הבא",
    "how do I handle a new war in the system",
])
def test_event_pipeline_section_attaches_on_triggers(tmp_events, question: str) -> None:
    _seed_event()
    context: dict[str, Any] = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, question)
    assert "event_pipeline" in context, f"event_pipeline did not attach for {question!r}"
    section = context["event_pipeline"]
    assert len(section["operational_order"]) == 4
    assert "training_gate" in section
    # Compose time has no actor, so the per-actor permissions stage stays out.
    assert "permissions" not in section


@pytest.mark.parametrize("question", [
    "כמה ברייקים יש מחר",
    "מה קורה עם המלחמה",
    "מה מצב האירועים בלוח שנה",
])
def test_event_pipeline_section_stays_off_unprompted(question: str) -> None:
    context: dict[str, Any] = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, question)
    assert "event_pipeline" not in context
    assert "event_pipeline" not in sources


# --- the company-only write gate in the propose path ------------------------------
def test_channel_actor_event_proposal_refused_without_pending_item(tmp_events, monkeypatch) -> None:
    monkeypatch.setattr(auth_store, "is_company_user",
                        lambda username: username == "company_user", raising=False)
    trace: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    result = tools.handle_tool_use(_block("propose_event_change", _event_args()),
                                   trace, items, user="channel_user")
    assert items == [], "a refused proposal must capture nothing"
    assert trace[-1] == {"tool": "propose_event_change", "ok": False}
    content = json.loads(result["content"])
    assert "שמורה לצוות החברה" in content["error"]
    assert not events_api.EVENTS_PATH.exists()


def test_company_actor_event_proposal_captures_pending(tmp_events, monkeypatch) -> None:
    monkeypatch.setattr(auth_store, "is_company_user",
                        lambda username: username == "company_user", raising=False)
    trace: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    tools.handle_tool_use(_block("propose_event_change", _event_args()),
                          trace, items, user="company_user")
    assert len(items) == 1
    assert items[0]["status"] == "pending"
    assert items[0]["kind"] == "event_change"
    assert not events_api.EVENTS_PATH.exists(), "capture must not write the store"


def test_gate_covers_agency_and_events_activation_but_not_other_proposals(monkeypatch) -> None:
    monkeypatch.setattr(auth_store, "is_company_user", lambda username: False, raising=False)
    assert pipeline.company_refusal("propose_agency_change", {}, "channel_user") is not None
    assert pipeline.company_refusal(
        "propose_pricing_change",
        {"changes": {"pricing_activation": {"events": True}}}, "channel_user") is not None
    assert pipeline.company_refusal(
        "propose_pricing_change",
        {"changes": {"premiums": {"day_of_week": {"5": 1.2}}}}, "channel_user") is None
    assert pipeline.company_refusal("propose_settings_change", {}, "channel_user") is None
    trace: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    tools.handle_tool_use(_block("propose_recompute", {"scope": "full", "reason": "רענון"}),
                          trace, items, user="channel_user")
    assert len(items) == 1 and items[0]["status"] == "pending"


def test_actor_is_company_fallback_without_helper(monkeypatch) -> None:
    monkeypatch.delattr(auth_store, "is_company_user", raising=False)
    monkeypatch.setattr(auth, "auth_active", lambda: False)
    assert pipeline.actor_is_company("anyone") is True
    monkeypatch.setattr(auth, "auth_active", lambda: True)
    assert pipeline.actor_is_company("anyone") is False


# --- the system prompt carries the playbook ---------------------------------------
def test_system_prompt_states_the_four_stage_order() -> None:
    prompt = assistant.SYSTEM_PROMPT
    record = prompt.index("(a) Record")
    pricing = prompt.index("(b) Pricing")
    recompute = prompt.index("(c) Recompute")
    training = prompt.index("(d) Training")
    assert record < pricing < recompute < training
    assert "changes NOTHING" in prompt
    assert "pricing_activation.events" in prompt
    assert "event_layer_gate" in prompt
    assert "ASSERTION" in prompt and "MEASURED" in prompt
    assert "SEPARATE approval" in prompt
    assert "reserved for company staff" in prompt
