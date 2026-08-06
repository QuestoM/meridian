"""P9: the undo is an object that outlives the tab that made it.

The reversal Kai hands back was reachable for exactly as long as one browser
tab: the apply response carried a restore id once, nothing stored it, and a
reload lost the only handle to the snapshot. Cursor's model is the reference,
where a checkpoint sits on the request that created it and can be opened at any
later time, so the batch now carries every restore point it produced.

Also asserted here, because both are new contracts a later round could break
without noticing:

* the scope Kai grounded on is streamed as a fact and is copied, never guessed
* a settings change that would move one of the four licence limits says so,
  with the effective date, before the approval
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import (
    assistant,
    assistant_actions as actions,
    assistant_permissions,
    assistant_pipeline,
    assistant_tools as tools,
)


@pytest.fixture()
def action_store(tmp_path, monkeypatch):
    """Tmp action-plane state and a settings file nothing else shares."""
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    settings = tmp_path / "kairos_settings.json"
    settings.write_text(json.dumps({"revenue_weight": 50}), encoding="utf-8")
    from kairos_api import core

    monkeypatch.setattr(core, "SETTINGS_PATH", settings, raising=False)
    return tmp_path


def _batch(*specs: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    items = [tools.build_proposal_item(name, args) for name, args in specs]
    return actions.create_batch("q", items, "tester", "test-model")


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


# --- the restore point lives on the batch ------------------------------------
def test_the_restore_point_is_stored_on_the_batch_and_survives_a_reread(action_store) -> None:
    batch = _batch(("propose_settings_change",
                    {"changes": {"revenue_weight": 61}, "reason": "test"}))
    client = _client()

    applied = client.post(f"/api/assistant/proposals/{batch['batch_id']}/apply",
                          json={"item_ids": [batch["items"][0]["id"]]})
    assert applied.status_code == 200
    body = applied.json()
    restore_id = body["restore_id"]
    assert restore_id

    # A fresh read of the store, which is what a reloaded browser gets.
    stored = {row["batch_id"]: row for row in client.get("/api/assistant/proposals").json()["batches"]}
    points = stored[batch["batch_id"]]["restore_points"]
    assert [point["restore_id"] for point in points] == [restore_id]
    assert points[0]["applied_by"] == "auth-disabled"
    assert points[0]["item_ids"] == [batch["items"][0]["id"]]
    assert points[0]["applied_at"]

    # The History page addresses a row by its unified-timeline version id, not
    # by the restore id /api/assistant/restore applies against: the two are
    # different ids on the same apply, and a "see it in the history" control
    # that carries the wrong one lands on nothing. Both the apply response and
    # the stored restore point carry the version id the timeline actually used.
    assert body["version_id"]
    assert points[0]["version_id"] == body["version_id"]
    from kairos_api import version_store

    timeline_entry = next(m for m in version_store._all_manifests()
                          if m.get("source") == "assistant_apply" and m.get("batch_id") == batch["batch_id"])
    assert timeline_entry["version_id"] == body["version_id"]


def test_two_applies_on_one_batch_keep_two_restore_points_in_order(action_store) -> None:
    batch = _batch(
        ("propose_settings_change", {"changes": {"revenue_weight": 61}, "reason": "one"}),
        ("propose_settings_change", {"changes": {"revenue_weight": 62}, "reason": "two"}),
    )
    client = _client()
    first = client.post(f"/api/assistant/proposals/{batch['batch_id']}/apply",
                        json={"item_ids": [batch["items"][0]["id"]]}).json()["restore_id"]
    second = client.post(f"/api/assistant/proposals/{batch['batch_id']}/apply",
                         json={"item_ids": [batch["items"][1]["id"]]}).json()["restore_id"]

    stored = {row["batch_id"]: row for row in client.get("/api/assistant/proposals").json()["batches"]}
    points = stored[batch["batch_id"]]["restore_points"]
    assert [point["restore_id"] for point in points] == [first, second]
    # Two applies of the same settings file produce two distinct timeline
    # versions, each addressable in its own right.
    version_ids = [point["version_id"] for point in points]
    assert len(set(version_ids)) == 2
    assert all(version_ids)


def test_an_apply_that_touches_no_state_file_records_no_restore_point(action_store) -> None:
    """A plan run cannot be un-run, so it offers no undo rather than an empty one."""
    batch = _batch(("propose_recompute", {"scope": "full", "reason": "test"}))
    client = _client()
    response = client.post(f"/api/assistant/proposals/{batch['batch_id']}/apply",
                           json={"item_ids": [batch["items"][0]["id"]]}).json()
    assert response["restore_id"] is None

    stored = {row["batch_id"]: row for row in client.get("/api/assistant/proposals").json()["batches"]}
    assert "restore_points" not in stored[batch["batch_id"]]


# --- the scope Kai grounded on ------------------------------------------------
def test_the_grounded_facts_are_copied_from_the_context_and_nothing_else() -> None:
    context = {
        "overview_summary": {"scope_channel": "רשת 13", "date_from": "2024-11-01",
                             "date_to": "2024-11-30", "n_dates": 30, "total_breaks": 2391,
                             "projected_revenue": 40944759.33},
        "schedule_freshness": {"status": "stale", "changed": ["settings"]},
        "counts": {"segments": 2540, "breaks": 2391, "scope_channel": "רשת 13"},
    }
    facts = assistant_pipeline.grounding_facts(context)

    assert facts == {"channel": "רשת 13", "date_from": "2024-11-01", "date_to": "2024-11-30",
                     "dates": 30, "breaks": 2391, "plan_status": "stale"}
    # Money is deliberately not among them: a figure would have to carry its
    # scope, and this line is the scope.
    assert "projected_revenue" not in facts


def test_an_absent_section_contributes_no_fact_rather_than_an_empty_one() -> None:
    assert assistant_pipeline.grounding_facts({}) == {}
    facts = assistant_pipeline.grounding_facts(
        {"overview_summary": {"scope_channel": "רשת 13", "total_breaks": 7}})
    assert facts == {"channel": "רשת 13", "breaks": 7}


def test_a_count_with_no_channel_prints_the_reason_instead_of_the_number() -> None:
    """A count over every channel in the file is not the operator's count, so
    the scope is stated and the figure is withheld rather than mislabelled."""
    facts = assistant_pipeline.grounding_facts({
        "overview_summary": {"total_breaks": 9026, "date_from": "2024-11-01", "date_to": "2024-11-30"},
        "counts": {"segments": 8704, "breaks": 9026, "scope_channel": None,
                   "reason": "operator channel is not configured in settings"},
    })
    assert "breaks" not in facts
    assert facts["scope_reason"] == "operator channel is not configured in settings"
    assert facts["date_from"] == "2024-11-01"


def test_the_stream_carries_the_grounding_facts_before_the_answer(monkeypatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: (
        {"overview_summary": {"scope_channel": "רשת 13", "total_breaks": 553}}, ["overview_summary"]))
    monkeypatch.setattr(assistant, "_client_factory", lambda key: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
            content=[SimpleNamespace(type="text", text="תשובה")], stop_reason="end_turn"))))
    assistant._reset_rate_limit()

    body = _client().post("/api/assistant/ask/stream", json={"question": "מה מצב השבוע"}).text
    stages = [json.loads(chunk.split("data: ", 1)[1])
              for chunk in body.strip().split("\n\n")
              if chunk.startswith("event: stage")]
    grounded = [stage for stage in stages if stage["stage"] == "grounded"]
    assert grounded and grounded[0]["facts"] == {"channel": "רשת 13", "breaks": 553}
    # And it lands before any answer text, which is the whole point of it.
    assert body.index('"stage": "grounded"') < body.index("event: delta")
    assistant._reset_rate_limit()


# --- the licence limits, disclosed before the approval ------------------------
def test_a_settings_change_that_moves_a_licence_limit_says_so(action_store) -> None:
    item = tools.build_proposal_item(
        "propose_settings_change", {"changes": {"max_breaks_per_hour": 3}, "reason": "test"})

    permission = item["permission"]
    assert permission["fields"] == ["max_breaks_per_hour"]
    assert permission["may_change"] is True, "with authentication off there is no account to refuse"
    assert permission["effective_date"], "the licence limits carry the date they took effect"
    assert "רגולציה" in permission["basis_he"]


def test_an_ordinary_settings_change_carries_no_permission_block(action_store) -> None:
    item = tools.build_proposal_item(
        "propose_settings_change", {"changes": {"revenue_weight": 61}, "reason": "test"})
    assert "permission" not in item


def test_the_licence_permission_reads_the_stores_own_wall(tmp_path, monkeypatch) -> None:
    """Not a rule restated here: the answer comes from guardrail_store's wall,
    so Kai and the surface that owns the limits can never disagree."""
    from kairos_api import auth_store, guardrail_store

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password="rootpass-1234")
    auth_store.add_user("planner_one", "plannerpass-123", role="operator")

    allowed, reason = assistant_permissions.actor_may_change_guardrails("planner_one")
    assert allowed is False
    assert reason == guardrail_store.GUARDRAIL_ADMIN_ONLY_DETAIL
    assert assistant_permissions.actor_may_change_guardrails("admin")[0] is True

    # The disclosure carries that refusal, and the proposal is still captured:
    # the four values still live on the unguarded settings document, so refusing
    # here alone would make Kai stricter than the control beside it.
    permission = assistant_permissions.guardrail_permission({"max_breaks_per_hour": 3}, "planner_one")
    assert permission["may_change"] is False
    assert permission["reason"] == guardrail_store.GUARDRAIL_ADMIN_ONLY_DETAIL
    assert assistant_permissions.refusal(
        "propose_settings_change", {"changes": {"max_breaks_per_hour": 3}}, "planner_one")
    assert assistant_permissions.refusal(
        "propose_settings_change", {"changes": {"revenue_weight": 61}}, "planner_one") is None
    auth_store.reset_runtime_state()
