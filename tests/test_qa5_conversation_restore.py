"""Conversation-level applied-changes view and restore orchestration.

Everything below the mocked Claude seam is real: batches go through the real
proposal validators, applies write through the real settings seam (redirected
to a tmp copy of the settings file), and versions land in the real unified
version store (relocated to tmp beside KAIROS_ASSISTANT_DATA_DIR). Covered:
the per-conversation changes contract (batches with item status and their
assistant_apply version ids, plus the legacy batch_id fallback), the restore
composing only shipped primitives (oldest assistant_apply version per logical
file, nearest-older fallback when the byte-identical short-circuit elided a
batch's own version, a forced pre_restore safety version making the whole
operation undoable), the 409 honesty when nothing applied, and the audit trail
for every restore.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_conversations as conversations
import kairos_api.assistant_memory as memory
import kairos_api.assistant_tools as tools
from kairos_api import core, version_store

ROOT = Path(__file__).resolve().parents[1]
USER = "auth-disabled"


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    # The simulated effect is additive on a settings item and the apply engine
    # ignores it; stubbing it keeps these tests fast and optimizer-free.
    import kairos_api.assistant_simulate as assistant_simulate

    monkeypatch.setattr(assistant_simulate, "settings_effect",
                        lambda changes: {"status": "unavailable", "reason": "stubbed in test"})
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def tmp_settings(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", target)
    monkeypatch.setattr(core, "SETTINGS_PATH", target)
    return target


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


def _settings_batch(conversation_id: str, revenue_weight: int) -> dict[str, Any]:
    """A real stored batch (validated items) linked to one conversation."""
    items = [tools.build_proposal_item(
        "propose_settings_change",
        {"changes": {"revenue_weight": revenue_weight}, "reason": "בדיקת שחזור"})]
    return actions.create_batch(f"שינוי משקל ל-{revenue_weight}", items, USER,
                                "claude-opus-5", conversation_id=conversation_id)


def _apply(batch: dict[str, Any]) -> dict[str, Any]:
    ids = [item["id"] for item in batch["items"]]
    return actions.apply_proposals(batch["batch_id"], actions.ItemIdsRequest(item_ids=ids), None)


def _apply_versions() -> list[dict[str, Any]]:
    return [m for m in version_store._all_manifests() if m.get("source") == "assistant_apply"]


# --- changes view ----------------------------------------------------------------
def test_changes_lists_batches_with_item_status_and_version_ids(client: TestClient, tmp_settings) -> None:
    cid = conversations.create(USER, title="שיחת שינויים")["id"]
    batch = _settings_batch(cid, 85)
    _apply(batch)

    body = client.get(f"/api/assistant/conversations/{cid}/changes").json()
    assert len(body["batches"]) == 1
    entry = body["batches"][0]
    assert entry["batch_id"] == batch["batch_id"]
    assert entry["created_by"] == USER
    assert entry["status"] == "resolved"
    assert [set(item) for item in entry["items"]] == [
        {"id", "kind", "summary", "status", "resolved_by", "resolved_at"}]
    assert entry["items"][0]["kind"] == "settings"
    assert entry["items"][0]["status"] == "applied"
    assert entry["items"][0]["resolved_by"] == USER
    version_ids = [m["version_id"] for m in _apply_versions()
                   if m.get("batch_id") == batch["batch_id"]]
    assert entry["version_ids"] == version_ids and len(version_ids) == 1


def test_changes_falls_back_to_entry_batch_ids_for_legacy_batches(client: TestClient) -> None:
    # A pre-conversation batch carries no conversation_id; it displays under the
    # legacy conversation because its batch_id sits on one of its entries.
    items = [tools.build_proposal_item(
        "propose_recompute", {"scope": "full", "reason": "רענון ישן"})]
    batch = actions.create_batch("שאלה ישנה", items, USER, "claude-opus-5")
    memory._write_atomic(memory._path_for(USER), USER, [
        {"question": "שאלה ישנה", "answer": "תשובה", "at": "2026-07-01T00:00:00+00:00",
         "batch_id": batch["batch_id"]}])
    legacy_id = f"legacy-{USER}"
    body = client.get(f"/api/assistant/conversations/{legacy_id}/changes").json()
    assert [entry["batch_id"] for entry in body["batches"]] == [batch["batch_id"]]
    # And a stranger batch never leaks into an unrelated conversation.
    other = conversations.create(USER)["id"]
    assert client.get(f"/api/assistant/conversations/{other}/changes").json()["batches"] == []


def test_changes_unknown_conversation_404(client: TestClient) -> None:
    assert client.get("/api/assistant/conversations/deadbeef0000/changes").status_code == 404


# --- restore orchestration -------------------------------------------------------
def test_restore_returns_the_state_before_the_conversations_first_mutation(
        client: TestClient, tmp_settings: Path) -> None:
    original = tmp_settings.read_bytes()
    cid = conversations.create(USER, title="שיחת שחזור")["id"]
    _apply(_settings_batch(cid, 85))
    after_first = tmp_settings.read_bytes()
    _apply(_settings_batch(cid, 90))
    assert tmp_settings.read_bytes() not in (original, after_first)

    body = client.post(f"/api/assistant/conversations/{cid}/restore").json()
    assert body["restored_files"] == ["settings"]
    assert body["recompute_required"] is True
    assert isinstance(body["note"], str) and body["note"]
    # The OLDEST assistant_apply version wins: its snapshot is pre-conversation.
    assert tmp_settings.read_bytes() == original
    oldest = min(_apply_versions(), key=lambda m: (m["created_at"], m.get("seq", 0)))
    assert body["version_ids_used"] == [oldest["version_id"]]

    # The forced pre_restore safety version holds the pre-restore state, so the
    # whole operation is undoable through the shipped per-version restore.
    pre = body["pre_restore_version_id"]
    manifest = version_store._read_manifest(pre)
    assert manifest["source"] == "pre_restore"
    before_restore = version_store._version_bytes(pre, "settings")
    version_store.restore_version(pre, None, None)
    assert tmp_settings.read_bytes() == before_restore

    audits = [entry for entry in actions.read_audit(100)["entries"]
              if entry["event"] == "conversation_restore"]
    assert len(audits) == 1
    assert audits[0]["conversation_id"] == cid
    assert audits[0]["results"]["pre_restore_version_id"] == pre


def test_restore_uses_nearest_older_version_when_short_circuit_elided(
        client: TestClient, tmp_settings: Path) -> None:
    # Another conversation's apply first, so the file is in canonical serialized
    # form and carries an assistant_apply version outside the conversation.
    _apply(_settings_batch(conversations.create(USER)["id"], 85))
    canonical = tmp_settings.read_bytes()

    cid = conversations.create(USER, title="שיחת דילוג")["id"]
    # A no-op apply (same value) records a version but leaves the bytes as-is.
    noop = _settings_batch(cid, 85)
    _apply(noop)
    assert tmp_settings.read_bytes() == canonical
    # The next apply captures byte-identical state: the short-circuit elides its
    # own version, so this batch has NO version carrying its batch_id.
    real = _settings_batch(cid, 90)
    _apply(real)
    assert all(m.get("batch_id") != real["batch_id"] for m in _apply_versions())
    noop_version = next(m for m in _apply_versions() if m.get("batch_id") == noop["batch_id"])

    body = client.post(f"/api/assistant/conversations/{cid}/restore").json()
    # The elided batch mapped to the nearest older assistant_apply version, and
    # the oldest per-file choice lands on the no-op batch's own version.
    assert body["version_ids_used"] == [noop_version["version_id"]]
    assert tmp_settings.read_bytes() == canonical


def test_restore_409_when_the_conversation_applied_nothing(
        client: TestClient, tmp_settings, monkeypatch) -> None:
    import kairos_api.recompute_api as recompute_api

    monkeypatch.setattr(recompute_api, "_run_recompute", lambda **kwargs: {"ok": True})
    empty = conversations.create(USER)["id"]
    assert client.post(f"/api/assistant/conversations/{empty}/restore").status_code == 409

    pending_only = conversations.create(USER)["id"]
    _settings_batch(pending_only, 85)  # stored, never applied
    assert client.post(f"/api/assistant/conversations/{pending_only}/restore").status_code == 409

    # Recompute-only conversations snapshot no state file: honest 409, no fake restore.
    recompute_only = conversations.create(USER)["id"]
    items = [tools.build_proposal_item("propose_recompute", {"scope": "full", "reason": "ר"})]
    batch = actions.create_batch("רק חישוב", items, USER, "m", conversation_id=recompute_only)
    _apply(batch)
    response = client.post(f"/api/assistant/conversations/{recompute_only}/restore")
    assert response.status_code == 409

    assert client.post("/api/assistant/conversations/deadbeef0000/restore").status_code == 404
