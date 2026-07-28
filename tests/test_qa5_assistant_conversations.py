"""Assistant conversations: per-conversation store, CRUD, scoping, migration.

The Claude call is mocked at the module seam (assistant._client_factory); no
real API call is ever made. The store is real, relocated to tmp via
KAIROS_ASSISTANT_DATA_DIR; data/ is never touched. Covered: the CRUD contract
(list, create, rename, delete with audit), ask minting and carrying
conversation_id on both /ask and /ask/stream, history scoped to one
conversation (no cross-contamination), the no-param GET /thread backward
compatibility, the legacy flat-file migration (no data loss, flat file deleted
only after the new files exist), batches stamped with conversation_id, and the
entry and index caps with file pruning.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_conversations as conversations
import kairos_api.assistant_memory as memory

USER = "auth-disabled"


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_ACTIONS", raising=False)
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


def _recording_factory(recorder: dict[str, Any], answer: str = "the answer"):
    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            snapshot = dict(kwargs)
            snapshot["messages"] = list(kwargs.get("messages") or [])
            recorder.setdefault("calls", []).append(snapshot)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=answer)], stop_reason="end_turn"
            )

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


# --- CRUD contract ---------------------------------------------------------------
def test_conversation_crud_lifecycle_with_audited_delete(client: TestClient) -> None:
    created = client.post("/api/assistant/conversations", json={"title": "תכנון שבועי"}).json()
    assert set(created) == {"id", "title"}
    assert created["title"] == "תכנון שבועי"

    listing = client.get("/api/assistant/conversations").json()
    assert listing["user"] == USER
    assert [set(record) for record in listing["conversations"]] == [
        {"id", "title", "created_at", "updated_at", "entry_count"}]
    assert listing["conversations"][0]["id"] == created["id"]
    assert listing["conversations"][0]["entry_count"] == 0

    renamed = client.patch(f"/api/assistant/conversations/{created['id']}",
                           json={"title": "שם חדש"}).json()
    assert renamed == {"id": created["id"], "title": "שם חדש"}
    assert client.get("/api/assistant/conversations").json()["conversations"][0]["title"] == "שם חדש"

    deleted = client.delete(f"/api/assistant/conversations/{created['id']}").json()
    assert deleted == {"deleted": True, "entries_removed": 0}
    assert client.get("/api/assistant/conversations").json()["conversations"] == []
    audit = actions.read_audit(50)["entries"]
    deletes = [entry for entry in audit if entry["event"] == "conversation_delete"]
    assert len(deletes) == 1
    assert deletes[0]["conversation_id"] == created["id"]
    assert deletes[0]["results"] == {"entries_removed": 0}


def test_unknown_conversation_returns_404_on_rename_delete_thread(client: TestClient) -> None:
    assert client.patch("/api/assistant/conversations/deadbeef0000",
                        json={"title": "x"}).status_code == 404
    assert client.delete("/api/assistant/conversations/deadbeef0000").status_code == 404
    assert client.get("/api/assistant/thread",
                      params={"conversation_id": "deadbeef0000"}).status_code == 404
    # A traversal-shaped id can never reach the filesystem: it is unknown, not an error.
    assert client.delete("/api/assistant/conversations/..%2F..%2Fetc").status_code == 404


# --- ask mints, titles, and scopes ------------------------------------------------
def test_ask_mints_conversation_and_titles_it_from_the_question(client: TestClient, monkeypatch) -> None:
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    question = "כמה ברייקים יש ביום שלישי בתוכנית השבועית שנשמרה אצל המפעיל הראשי בערוץ"
    body = client.post("/api/assistant/ask", json={"question": question}).json()
    cid = body["conversation_id"]
    assert cid and conversations.valid_id(cid)

    records = client.get("/api/assistant/conversations").json()["conversations"]
    assert [record["id"] for record in records] == [cid]
    assert records[0]["title"] == question[:conversations.TITLE_MAX_CHARS]
    assert records[0]["entry_count"] == 1

    thread = client.get("/api/assistant/thread").json()
    assert thread["conversation_id"] == cid
    assert [entry["question"] for entry in thread["entries"]] == [question]
    assert thread["entries"][0]["conversation_id"] == cid

    scoped = client.get("/api/assistant/thread", params={"conversation_id": cid}).json()
    assert scoped["entries"] == thread["entries"]


def test_history_is_scoped_per_conversation_never_cross_contaminated(client: TestClient, monkeypatch) -> None:
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    first = client.post("/api/assistant/ask", json={"question": "שאלה בשיחה הראשונה"}).json()
    cid_a = first["conversation_id"]

    cid_b = client.post("/api/assistant/conversations", json={}).json()["id"]
    second = client.post("/api/assistant/ask",
                         json={"question": "שאלה בשיחה השנייה", "conversation_id": cid_b}).json()
    assert second["conversation_id"] == cid_b

    # The ask in conversation B replayed NO history from conversation A.
    messages_b = recorder["calls"][1]["messages"]
    assert len(messages_b) == 1
    assert messages_b[0]["content"].endswith("שאלה בשיחה השנייה")

    # A follow-up in conversation A replays only A's own exchange.
    third = client.post("/api/assistant/ask",
                        json={"question": "המשך בראשונה", "conversation_id": cid_a}).json()
    assert third["conversation_id"] == cid_a
    messages_a = recorder["calls"][2]["messages"]
    assert [m["role"] for m in messages_a] == ["user", "assistant", "user"]
    assert messages_a[0]["content"] == "שאלה בשיחה הראשונה"

    # Each conversation's thread holds exactly its own entries.
    entries_a = client.get("/api/assistant/thread", params={"conversation_id": cid_a}).json()["entries"]
    entries_b = client.get("/api/assistant/thread", params={"conversation_id": cid_b}).json()["entries"]
    assert [entry["question"] for entry in entries_a] == ["שאלה בשיחה הראשונה", "המשך בראשונה"]
    assert [entry["question"] for entry in entries_b] == ["שאלה בשיחה השנייה"]


def test_absent_conversation_id_uses_newest_and_unknown_mints(client: TestClient, monkeypatch) -> None:
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    first = client.post("/api/assistant/ask", json={"question": "ראשונה"}).json()
    # Absent id lands in the newest conversation: the old client keeps its thread.
    followup = client.post("/api/assistant/ask", json={"question": "שנייה"}).json()
    assert followup["conversation_id"] == first["conversation_id"]
    assert [m["role"] for m in recorder["calls"][1]["messages"]] == ["user", "assistant", "user"]

    # An unknown id mints a fresh conversation and carries the minted id back.
    minted = client.post("/api/assistant/ask",
                         json={"question": "לא מוכר", "conversation_id": "deadbeef0000"}).json()
    assert minted["conversation_id"] not in (None, "deadbeef0000", first["conversation_id"])
    ids = {r["id"] for r in client.get("/api/assistant/conversations").json()["conversations"]}
    assert minted["conversation_id"] in ids


def test_stream_ask_carries_conversation_id_and_appends_scoped(client: TestClient, monkeypatch) -> None:
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    cid = client.post("/api/assistant/conversations", json={"title": "סטרים"}).json()["id"]
    response = client.post("/api/assistant/ask/stream",
                           json={"question": "שאלה בסטרים", "conversation_id": cid})
    assert response.status_code == 200
    final = json.loads(response.text.strip().split("\n\n")[-1].split("\ndata: ")[1])
    assert final["conversation_id"] == cid
    entries = client.get("/api/assistant/thread", params={"conversation_id": cid}).json()["entries"]
    assert [entry["question"] for entry in entries] == ["שאלה בסטרים"]
    assert entries[0]["conversation_id"] == cid


def test_batches_are_stamped_with_the_conversation_id(client: TestClient, monkeypatch) -> None:
    def propose_factory(api_key: str) -> Any:
        turns = [
            SimpleNamespace(content=[SimpleNamespace(
                type="tool_use", name="propose_recompute",
                input={"scope": "full", "reason": "לרענן את התוכנית"}, id="tu_1")],
                stop_reason="tool_use"),
            SimpleNamespace(content=[SimpleNamespace(type="text", text="הצעתי חישוב מחדש")],
                            stop_reason="end_turn"),
        ]
        return SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: turns.pop(0)))

    monkeypatch.setattr(assistant, "_client_factory", propose_factory)
    body = client.post("/api/assistant/ask", json={"question": "רענון"}).json()
    assert body["proposals"] is not None
    with actions._LOCK:
        stored = actions._load_store()["batches"]
    assert stored[-1]["batch_id"] == body["proposals"]["batch_id"]
    assert stored[-1]["conversation_id"] == body["conversation_id"]


# --- legacy migration ------------------------------------------------------------
def test_legacy_flat_thread_migrates_losslessly(client: TestClient) -> None:
    flat = memory._path_for(USER)
    memory._write_atomic(flat, USER, [
        {"question": "שאלה ישנה", "answer": "תשובה ישנה", "at": "2026-07-01T00:00:00+00:00",
         "batch_id": "abc123abc123"},
        {"question": "שאלה שנייה", "answer": "תשובה שנייה", "at": "2026-07-02T00:00:00+00:00",
         "batch_id": None},
    ])
    assert flat.exists()

    thread = client.get("/api/assistant/thread").json()
    legacy_id = f"legacy-{USER}"
    assert thread["conversation_id"] == legacy_id
    assert [(e["question"], e["answer"], e["batch_id"]) for e in thread["entries"]] == [
        ("שאלה ישנה", "תשובה ישנה", "abc123abc123"), ("שאלה שנייה", "תשובה שנייה", None)]
    assert all(entry["conversation_id"] == legacy_id for entry in thread["entries"])

    # The flat file is gone only because the new files exist.
    assert not flat.exists()
    assert conversations._conversation_path(USER, legacy_id).exists()
    assert conversations._index_path(USER).exists()

    records = client.get("/api/assistant/conversations").json()["conversations"]
    assert [(r["id"], r["title"], r["entry_count"]) for r in records] == [
        (legacy_id, conversations.LEGACY_TITLE, 2)]
    # Re-running the migration is a no-op, never a duplicate.
    conversations._migrate_legacy(USER)
    assert len(client.get("/api/assistant/conversations").json()["conversations"]) == 1


def test_migration_failure_keeps_the_flat_file(client: TestClient, monkeypatch) -> None:
    flat = memory._path_for(USER)
    memory._write_atomic(flat, USER, [
        {"question": "ק", "answer": "ת", "at": "2026-07-01T00:00:00+00:00", "batch_id": None}])
    monkeypatch.setattr(conversations, "_write_json", lambda path, payload: 1 / 0)
    conversations._migrate_legacy(USER)
    # Nothing was lost: the flat file survives an incomplete migration.
    assert flat.exists()
    assert [e["question"] for e in memory._load_entries(flat)] == ["ק"]


# --- caps -----------------------------------------------------------------------
def test_entries_prune_to_newest_50_per_conversation(client: TestClient) -> None:
    cid = conversations.create(USER)["id"]
    for index in range(55):
        memory.append_entry(USER, f"q{index}", f"a{index}", None, conversation_id=cid)
    entries = client.get("/api/assistant/thread", params={"conversation_id": cid}).json()["entries"]
    assert len(entries) == memory.MAX_ENTRIES == 50
    assert [entry["question"] for entry in entries] == [f"q{i}" for i in range(5, 55)]
    stored = json.loads(conversations._conversation_path(USER, cid).read_text(encoding="utf-8"))
    assert len(stored["entries"]) == 50


def test_index_caps_at_30_conversations_pruning_oldest_with_files(client: TestClient) -> None:
    ids = [conversations.create(USER, title=f"שיחה {index}")["id"] for index in range(33)]
    records = client.get("/api/assistant/conversations").json()["conversations"]
    assert len(records) == conversations.MAX_CONVERSATIONS == 30
    kept = {record["id"] for record in records}
    pruned = [cid for cid in ids if cid not in kept]
    assert len(pruned) == 3
    for cid in pruned:
        assert not conversations._conversation_path(USER, cid).exists()


# --- clear and identity ----------------------------------------------------------
def test_delete_thread_clears_all_of_only_the_callers_conversations(client: TestClient) -> None:
    cid_a = conversations.create(USER)["id"]
    cid_b = conversations.create(USER)["id"]
    memory.append_entry(USER, "שאלה א", "תשובה א", None, conversation_id=cid_a)
    memory.append_entry(USER, "שאלה ב", "תשובה ב", None, conversation_id=cid_b)
    memory.append_entry("someone-else", "שלהם", "תשובתם", None)

    body = client.delete("/api/assistant/thread").json()
    assert body == {"cleared": True, "entries_removed": 2, "user": USER}
    assert client.get("/api/assistant/conversations").json()["conversations"] == []
    assert client.get("/api/assistant/thread").json()["entries"] == []
    # The other user's conversations are untouched.
    assert conversations.list_records("someone-else")[0]["entry_count"] == 1
    audit = actions.read_audit(50)["entries"]
    clears = [entry for entry in audit if entry["event"] == "thread_clear"]
    assert clears[0]["results"] == {"entries_removed": 2}
