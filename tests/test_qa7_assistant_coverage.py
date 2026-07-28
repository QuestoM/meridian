"""Assistant coverage of the new structures: agencies, calendar events, event
pricing, custom pricing, the daily ledger, and the page-context grounding.

Everything runs against seeded tmp stores or the repo's real read-only data;
the Claude call, where needed, is mocked at the module seam exactly like the
sibling suites. The contracts under test: each new read tool returns real
store data with an explicit basis, the two propose tools only capture pending
items (never writes) and apply through the existing engine with restore points
and version-timeline coverage, the pricing tool and keyword section keep the
events block, and page_context attaches a current_location section when sent
while degrading to exactly today's behavior when absent or invalid.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.advertiser_conditions as advc
import kairos_api.advertisers as adv
import kairos_api.agencies as ag
import kairos_api.agency_conditions as agc
import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_page_context as page_context
import kairos_api.assistant_tools as tools
import kairos_api.events_api as events_api
from kairos_api import assistant_keywords, version_store
from kairos_api.assistant_read_tools import execute_read_tool

NEW_READ_TOOLS = {"get_agencies", "get_agency_detail", "get_calendar_events",
                  "get_event_pricing", "get_advertiser_pricing", "get_top_advertisers"}
NEW_PROPOSE_TOOLS = {"propose_event_change", "propose_agency_change"}

ASK_BODY_KEYS = {"available", "answer", "model", "grounding", "context_disclosure",
                 "truncated", "error", "proposals", "tool_trace", "conversation_id"}


# --- fixtures ---------------------------------------------------------------------
@pytest.fixture()
def tmp_stores(tmp_path, monkeypatch):
    """Every touched store on throwaway CSVs; the action plane on a tmp dir."""
    monkeypatch.setattr(events_api, "EVENTS_PATH", tmp_path / "calendar_events.csv")
    monkeypatch.setattr(ag, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(ag, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agc, "LINKS_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(agc, "CONDITIONS_PATH", tmp_path / "agency_conditions.csv")
    monkeypatch.setattr(agc, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agc, "_latest_daily_pairs", lambda: ([], None))
    monkeypatch.setattr(adv, "RULES_PATH", tmp_path / "advertiser_rules.csv")
    monkeypatch.setattr(adv, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(advc, "CONDITIONS_PATH", tmp_path / "advertiser_conditions.csv")
    monkeypatch.setattr(advc, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    return tmp_path


def _seed_event(**overrides: Any) -> dict[str, Any]:
    fields = {"name": "מבצע בדיקה", "type": "war", "start_date": "2024-11-03",
              "end_date": "2024-11-05", "intensity": 4, "price_multiplier": 0.8}
    fields.update(overrides)
    return events_api.create_event(events_api.EventCreate(**fields), request=None)


def _seed_agency(**overrides: Any) -> dict[str, Any]:
    fields = {"agency_id": "AGY_Q7", "name": "סוכנות שבע", "rebate_percent": 5.0,
              "payment_terms_days": 60, "agency_type": "בוטיק"}
    fields.update(overrides)
    return ag.create_agency(ag.AgencyCreate(**fields))


def _seed_advertiser_with_condition() -> None:
    adv.create_advertiser(adv.AdvertiserCreate(advertiser_id="שופרסל", default_premium=1.1))
    advc.create_condition("שופרסל", advc.ConditionCreate(
        rule_id="R_SAT", effect="premium", value=12.0, mode="premium_discount",
        scope_weekdays="6", notes="הנחת שבת"))


# --- registry ---------------------------------------------------------------------
def test_new_tools_registered_with_read_propose_split() -> None:
    assert NEW_READ_TOOLS <= tools.READ_TOOL_NAMES
    assert NEW_PROPOSE_TOOLS <= tools.PROPOSE_TOOL_NAMES
    read_only = {schema["name"] for schema in tools.anthropic_tools(include_propose=False)}
    assert NEW_READ_TOOLS <= read_only
    assert not read_only & NEW_PROPOSE_TOOLS
    offered = {schema["name"] for schema in tools.anthropic_tools(include_propose=True)}
    assert NEW_READ_TOOLS | NEW_PROPOSE_TOOLS <= offered


def test_net_descriptions_disambiguate_the_two_nets() -> None:
    by_name = {schema["name"]: schema["description"] for schema in tools.READ_TOOL_SCHEMAS}
    assert "RETENTION" in by_name["get_net_comparison"]
    assert "get_top_advertisers" in by_name["get_net_comparison"]
    assert "agency rebates" in by_name["get_top_advertisers"]


# --- read tools on seeded stores --------------------------------------------------
def test_get_agencies_returns_seeded_terms(tmp_stores) -> None:
    _seed_agency()
    payload = execute_read_tool("get_agencies", {}, None)
    assert payload["count"] == 1
    row = payload["agencies"][0]
    assert row["agency_id"] == "AGY_Q7"
    assert row["rebate_percent"] == pytest.approx(5.0)
    assert row["payment_terms_days"] == 60
    assert "daily per-spot ledger" in payload["basis"]
    assert payload["source"] == "agencies store"


def test_get_agency_detail_by_id_and_by_name(tmp_stores) -> None:
    _seed_agency()
    agc.create_condition("AGY_Q7", agc.ConditionCreate(
        rule_id="AR1", effect="premium", value=1.2, scope_dayparts="prime"))
    by_id = execute_read_tool("get_agency_detail", {"agency_id": "AGY_Q7"}, None)
    assert by_id["name"] == "סוכנות שבע"
    assert by_id["conditions"][0]["rule_id"] == "AR1"
    assert "links" in by_id and by_id["links"]["effective_count"] == 0
    by_name = execute_read_tool("get_agency_detail", {"name": "סוכנות שבע"}, None)
    assert by_name["agency_id"] == "AGY_Q7"
    missing = execute_read_tool("get_agency_detail", {"agency_id": "NOPE"}, None)
    assert "error" in missing


def test_get_calendar_events_filters_and_multiplier(tmp_stores) -> None:
    _seed_event()
    _seed_event(name="חג בדיקה", type="holiday", start_date="2025-04-01",
                end_date="2025-04-02", price_multiplier=1.0, intensity=2)
    everything = execute_read_tool("get_calendar_events", {}, None)
    assert everything["count"] == 2 and everything["events_total_stored"] == 2
    war_only = execute_read_tool("get_calendar_events", {"type": "war"}, None)
    assert war_only["count"] == 1
    assert war_only["events"][0]["price_multiplier"] == pytest.approx(0.8)
    assert "plan_overlap_dates" in war_only["events"][0]
    ranged = execute_read_tool(
        "get_calendar_events", {"date_from": "2025-01-01", "date_to": "2025-12-31"}, None)
    assert [event["name"] for event in ranged["events"]] == ["חג בדיקה"]
    bad = execute_read_tool("get_calendar_events", {"date_from": "not-a-date"}, None)
    assert "error" in bad


def test_get_event_pricing_reports_layer_state_and_nonneutral_events(tmp_stores) -> None:
    _seed_event()  # multiplier 0.8, active
    _seed_event(name="נייטרלי", price_multiplier=1.0)
    payload = execute_read_tool("get_event_pricing", {}, None)
    assert payload["activation_flag"] == "pricing_activation.events"
    assert isinstance(payload["enabled"], bool)
    assert payload["count"] == 1
    assert payload["nonneutral_active_events"][0]["price_multiplier"] == pytest.approx(0.8)
    assert "operator assertion" in payload["basis"]
    if not payload["enabled"]:
        assert "OFF" in payload["note"]


def test_get_advertiser_pricing_returns_structured_rows(tmp_stores) -> None:
    _seed_advertiser_with_condition()
    payload = execute_read_tool("get_advertiser_pricing", {"advertiser": "שופרסל"}, None)
    assert payload["baseline"]["advertiser_id"] == "שופרסל"
    assert payload["conditions_count"] == 1
    condition = payload["conditions"][0]
    assert condition["mode"] == "premium_discount"
    assert condition["scope_weekdays"] == "6"
    assert "daily per-spot" in payload["basis"]
    missing = execute_read_tool("get_advertiser_pricing", {"advertiser": "לא קיים"}, None)
    assert "error" in missing and "find_advertiser" in missing["error"]


def test_get_top_advertisers_ledger_basis_and_ranking() -> None:
    payload = execute_read_tool("get_top_advertisers", {"limit": 5}, None)
    if payload.get("status") == "unavailable":
        pytest.skip("no daily spot file in this checkout")
    rows = payload["advertisers"]
    assert 0 < len(rows) <= 5
    gross = [row["gross_revenue_ils"] for row in rows]
    assert gross == sorted(gross, reverse=True)
    for row in rows:
        assert row["net_revenue_ils"] <= row["gross_revenue_ils"] + 1e-6
    assert "agency rebates" in payload["basis"]
    assert "retention-net" in payload["basis"]
    assert payload["source_file"]


def test_get_top_advertisers_honest_without_daily_file(monkeypatch) -> None:
    from kairos_api import uploads

    monkeypatch.setattr(uploads, "_newest_daily", lambda: None)
    payload = execute_read_tool("get_top_advertisers", {}, None)
    assert payload["status"] == "unavailable"
    assert "daily" in payload["reason"]


# --- the events block stays visible ----------------------------------------------
def test_get_pricing_keeps_events_block() -> None:
    payload = execute_read_tool("get_pricing", {}, None)
    assert isinstance(payload.get("events"), dict)
    assert set(payload["events"]) >= {"enabled", "active_event_count", "basis"}


def test_pricing_state_keyword_section_keeps_events_block() -> None:
    context: dict[str, Any] = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, "מה מצב התמחור")
    assert "pricing_state" in context
    assert isinstance(context["pricing_state"].get("events"), dict)


@pytest.mark.parametrize("question,section", [
    ("אילו סוכנויות עובדות איתנו", "agencies_state"),
    ("מה יש בלוח שנה החודש", "calendar_events"),
    ("מה המכפיל של החג", "event_pricing"),
    ("יש הנחה מיוחדת בשבת", "custom_pricing"),
])
def test_new_keyword_sections_attach(tmp_stores, question: str, section: str) -> None:
    _seed_event()
    _seed_agency()
    _seed_advertiser_with_condition()
    context: dict[str, Any] = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, question)
    assert section in context, f"{section} did not attach for {question!r}"
    assert section in sources


def test_keyword_sections_do_not_attach_unprompted() -> None:
    context: dict[str, Any] = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, "כמה ברייקים יש מחר")
    for section in ("agencies_state", "calendar_events", "event_pricing", "custom_pricing"):
        assert section not in context


# --- propose tools: capture only, then apply through the real engine --------------
def test_propose_event_change_captures_pending_without_writing(tmp_stores) -> None:
    item = tools.build_proposal_item("propose_event_change", {
        "action": "create", "reason": "חג חדש",
        "event": {"name": "אירוע הצעה", "type": "holiday", "start_date": "2025-06-01",
                  "end_date": "2025-06-02", "intensity": 2, "price_multiplier": 1.3},
    })
    assert item["status"] == "pending"
    assert item["kind"] == "event_change"
    assert "אירוע" in item["summary"] and "1.3" in item["summary"]
    assert not events_api.EVENTS_PATH.exists(), "capture must not write the store"
    bad = tools.build_proposal_item("propose_event_change", {
        "action": "create", "reason": "רע",
        "event": {"name": "x", "type": "nope", "start_date": "2025-06-01"},
    })
    assert bad["status"] == "rejected" and "type" in bad["error"]
    unknown_field = tools.build_proposal_item("propose_event_change", {
        "action": "create", "reason": "רע",
        "event": {"name": "x", "type": "holiday", "start_date": "2025-06-01", "made_up": 1},
    })
    assert unknown_field["status"] == "rejected" and "made_up" in unknown_field["error"]


def test_event_apply_writes_store_with_restore_and_version(tmp_stores) -> None:
    _seed_event(name="קיים", price_multiplier=1.0)
    before = events_api.EVENTS_PATH.read_bytes()
    item = tools.build_proposal_item("propose_event_change", {
        "action": "create", "reason": "מלחמה חדשה",
        "event": {"name": "אירוע חדש", "type": "war", "start_date": "2025-07-01",
                  "intensity": 5, "price_multiplier": 0.7},
    })
    batch = actions.create_batch("שאלה", [item], "auth-disabled", "test-model")
    result = actions.apply_proposals(batch["batch_id"], actions.ItemIdsRequest(item_ids=[item["id"]]), None)
    assert result["results"][0]["status"] == "applied"
    names = {record["name"] for _, row in events_api._load_frame().iterrows()
             for record in [events_api._record(row)]}
    assert "אירוע חדש" in names
    # The restore point covers the events store byte-for-byte.
    restore_id = result["restore_id"]
    manifest = json.loads((actions._restore_root() / restore_id / "manifest.json").read_text(encoding="utf-8"))
    assert any(entry["name"] == "calendar_events.csv" for entry in manifest["files"])
    actions.restore_state(restore_id, None)
    assert events_api.EVENTS_PATH.read_bytes() == before
    # The unified version timeline recorded the pre-apply events state.
    manifests = version_store._all_manifests()
    assert any(entry.get("logical") == "events"
               for m in manifests for entry in m.get("files", []))


def test_agency_proposals_refine_kind_and_apply(tmp_stores) -> None:
    _seed_agency()
    terms = tools.build_proposal_item("propose_agency_change", {
        "agency_id": "AGY_Q7", "action": "update", "reason": "עדכון עמלה",
        "changes": {"rebate_percent": 7.5},
    })
    assert terms["status"] == "pending" and terms["kind"] == "agency_change"
    assert "סוכנות" in terms["summary"] and "7.5" in terms["summary"]
    link = tools.build_proposal_item("propose_agency_change", {
        "agency_id": "AGY_Q7", "action": "link_advertiser", "reason": "קישור",
        "advertiser": "שופרסל",
    })
    assert link["status"] == "pending" and link["kind"] == "agency_link_change"
    condition = tools.build_proposal_item("propose_agency_change", {
        "agency_id": "AGY_Q7", "action": "add_condition", "reason": "תנאי פריים",
        "condition": {"rule_id": "AC1", "effect": "premium", "value": 1.15,
                      "scope_dayparts": "prime"},
    })
    assert condition["status"] == "pending" and condition["kind"] == "agency_condition_change"
    missing = tools.build_proposal_item("propose_agency_change", {
        "agency_id": "NOPE", "action": "update", "reason": "x", "changes": {"rebate_percent": 1},
    })
    assert missing["status"] == "rejected"

    batch = actions.create_batch("שאלה", [terms, link, condition], "auth-disabled", "test-model")
    result = actions.apply_proposals(
        batch["batch_id"],
        actions.ItemIdsRequest(item_ids=[terms["id"], link["id"], condition["id"]]), None)
    assert {entry["status"] for entry in result["results"]} == {"applied"}
    record = ag._row_to_record(ag._load_frame().iloc[0])
    assert record["rebate_percent"] == pytest.approx(7.5)
    assert agc.links_for("AGY_Q7")["manual"] == ["שופרסל"]
    assert agc.conditions_for("AGY_Q7")[0]["rule_id"] == "AC1"
    # The restore point copied all three touched agency stores.
    manifest = json.loads((actions._restore_root() / result["restore_id"] / "manifest.json")
                          .read_text(encoding="utf-8"))
    copied = {entry["name"] for entry in manifest["files"]}
    assert {"agencies.csv", "agency_advertisers.csv", "agency_conditions.csv"} <= copied


def test_pricing_activation_events_proposal_states_event_day_forecast() -> None:
    item = tools.build_proposal_item("propose_pricing_change", {
        "changes": {"pricing_activation": {"events": True}},
        "reason": "להפעיל תמחור אירועים",
    })
    assert item["status"] == "pending"
    assert item["kind"] == "pricing"
    assert item["summary"].startswith("pricing: edit")
    assert "אירוע" in item["summary"], "the events activation summary must state the event-day forecast"


# --- page context -----------------------------------------------------------------
def test_page_context_parse_rejects_invalid_shapes() -> None:
    assert page_context.parse_page_context(None) is None
    assert page_context.parse_page_context(42) is None
    assert page_context.parse_page_context({}) is None
    assert page_context.parse_page_context({"entity": {"type": "bogus", "id": "x"}}) is None
    parsed = page_context.parse_page_context(
        {"view": "Advertisers", "label": "מפרסמים",
         "entity": {"type": "advertiser", "id": "שופרסל", "label": "שופרסל"}})
    assert parsed == {"view": "Advertisers", "label": "מפרסמים",
                      "entity": {"type": "advertiser", "id": "שופרסל", "label": "שופרסל"}}


def test_page_context_section_present_when_sent_absent_when_not(tmp_stores) -> None:
    _seed_advertiser_with_condition()
    context: dict[str, Any] = {}
    sources: list[str] = []
    page_context.extend_with_current_location(context, sources, {
        "view": "Advertisers", "label": "מפרסמים",
        "entity": {"type": "advertiser", "id": "שופרסל", "label": "שופרסל"}})
    assert "current_location" in context and "current_location" in sources
    data = context["current_location"]["entity_data"]
    assert data["record"]["advertiser_id"] == "שופרסל"
    assert data["conditions"][0]["rule_id"] == "R_SAT"

    untouched: dict[str, Any] = {}
    untouched_sources: list[str] = []
    page_context.extend_with_current_location(untouched, untouched_sources, None)
    page_context.extend_with_current_location(untouched, untouched_sources, {"entity": {"type": "zz", "id": ""}})
    assert untouched == {} and untouched_sources == []


def test_page_context_entity_types_resolve_from_real_stores(tmp_stores) -> None:
    record = _seed_event()
    _seed_agency()
    event_data = page_context._entity_event(record["event_id"])
    assert event_data["name"] == "מבצע בדיקה"
    assert "plan_overlap_count" in event_data
    agency_data = page_context._entity_agency("AGY_Q7")
    assert agency_data["record"]["agency_id"] == "AGY_Q7"
    assert page_context._entity_agency("NOPE")["status"] == "not_found"
    assert page_context._entity_event("nope")["status"] == "not_found"


# --- the ask pipeline: shape unchanged, page context flows ------------------------
def _scripted_client(answer: str):
    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=answer)], stop_reason="end_turn")
        return SimpleNamespace(messages=SimpleNamespace(create=create))
    return factory


@pytest.fixture()
def ask_client(tmp_path, monkeypatch):
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    assistant._reset_rate_limit()
    seen: dict[str, Any] = {}

    def fake_compose(question: str, page_ctx: dict[str, Any] | None = None):
        seen["page_context"] = page_ctx
        return {}, ["settings"]

    monkeypatch.setattr(assistant, "_compose_context", fake_compose)
    monkeypatch.setattr(assistant, "_client_factory", _scripted_client("תשובה"))
    app = FastAPI()
    app.include_router(assistant.router)
    yield TestClient(app), seen
    assistant._reset_rate_limit()


def test_ask_shape_unchanged_and_page_context_reaches_composer(ask_client) -> None:
    client, seen = ask_client
    sent = {"view": "Agencies", "label": "סוכנויות",
            "entity": {"type": "agency", "id": "AGY_Q7", "label": "סוכנות שבע"}}
    body = client.post("/api/assistant/ask",
                       json={"question": "מה התנאים שלו", "page_context": sent}).json()
    assert set(body) == ASK_BODY_KEYS
    assert seen["page_context"] == sent


def test_ask_without_page_context_degrades_to_none(ask_client) -> None:
    client, seen = ask_client
    body = client.post("/api/assistant/ask", json={"question": "שאלה רגילה"}).json()
    assert set(body) == ASK_BODY_KEYS
    assert seen["page_context"] is None


def test_stream_accepts_page_context_and_final_shape_unchanged(ask_client) -> None:
    client, seen = ask_client
    sent = {"view": "Events", "label": "לוח שנה", "entity": None}
    with client.stream("POST", "/api/assistant/ask/stream",
                       json={"question": "מה יש בלוח", "page_context": sent}) as response:
        text = "".join(chunk for chunk in response.iter_text())
    frames = [line for line in text.splitlines() if line.startswith("data: ")]
    final = json.loads(frames[-1][len("data: "):])
    assert set(final) == ASK_BODY_KEYS
    assert seen["page_context"] == sent
