"""Contract tests for the per-operator assistant thread memory.

The store is real (a tmp KAIROS_ASSISTANT_DATA_DIR); nothing is mocked below
the endpoint seam except the Claude client. Covered: append and prune to the
newest 50, newest-last GET ordering, restart persistence (a fresh read of the
same file), strictly session-derived identity with safe filenames, DELETE
clearing only the caller's thread with an audit entry, and the non-streaming
ask appending exactly one entry on success and none on failure.
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
import kairos_api.assistant_memory as memory


@pytest.fixture(autouse=True)
def memory_env(tmp_path, monkeypatch):
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


def _answer_factory(answer: str = "grounded answer"):
    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=answer)], stop_reason="end_turn"
            )

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


# --- identity: sanitized, session-derived, collision-free ------------------------
def test_sanitized_usernames_are_safe_and_distinct() -> None:
    traversal = memory._sanitize_username("../../etc/passwd")
    assert "/" not in traversal and ".." not in traversal
    assert not traversal.startswith(".")
    # A sanitized name differing from the original carries a hash of the
    # ORIGINAL, so distinct raw usernames can never share one thread file.
    assert memory._sanitize_username("a/b") != memory._sanitize_username("a_b")
    assert memory._sanitize_username("a/b") != memory._sanitize_username("a.b")
    # A clean username maps to itself, and the file stays inside the threads dir.
    assert memory._sanitize_username("auth-disabled") == "auth-disabled"
    path = memory._path_for("../../etc/passwd")
    assert path.parent == memory._threads_dir()


def test_thread_user_is_session_derived_not_a_parameter(client: TestClient) -> None:
    # Auth is disabled in tests, so the identity is the documented constant; a
    # client-supplied parameter must not select a different thread.
    body = client.get("/api/assistant/thread", params={"user": "someone-else"}).json()
    assert body["user"] == "auth-disabled"
    assert body["entries"] == []


# --- append, order, prune, persistence -------------------------------------------
def test_append_prunes_to_newest_50_newest_last(client: TestClient) -> None:
    for index in range(55):
        memory.append_entry("auth-disabled", f"q{index}", f"a{index}", None)

    body = client.get("/api/assistant/thread").json()
    assert len(body["entries"]) == 50
    assert body["entries"][0] == {"question": "q5", "answer": "a5",
                                  "at": body["entries"][0]["at"], "batch_id": None}
    assert body["entries"][-1]["question"] == "q54"
    questions = [entry["question"] for entry in body["entries"]]
    assert questions == [f"q{index}" for index in range(5, 55)]

    # The file itself holds exactly the pruned 50: the prune is at-write, not at-read.
    stored = json.loads(memory._path_for("auth-disabled").read_text(encoding="utf-8"))
    assert len(stored["entries"]) == 50


def test_thread_survives_restart_fresh_reader_sees_same_file(client: TestClient) -> None:
    memory.append_entry("auth-disabled", "before restart", "the answer", "batch123")
    # A restart holds no module state: reading through a brand-new app and a
    # direct fresh file read must both reproduce the entry.
    fresh_app = FastAPI()
    fresh_app.include_router(assistant.router)
    body = TestClient(fresh_app).get("/api/assistant/thread").json()
    assert [entry["question"] for entry in body["entries"]] == ["before restart"]
    assert body["entries"][0]["batch_id"] == "batch123"
    assert body["entries"][0]["at"]
    direct = memory._load_entries(memory._path_for("auth-disabled"))
    assert direct == body["entries"]


# --- delete: only the caller's thread, audited ------------------------------------
def test_delete_clears_only_the_callers_thread_and_audits(client: TestClient) -> None:
    memory.append_entry("auth-disabled", "mine", "my answer", None)
    memory.append_entry("someone-else", "theirs", "their answer", None)

    body = client.delete("/api/assistant/thread").json()
    assert body == {"cleared": True, "entries_removed": 1, "user": "auth-disabled"}

    assert client.get("/api/assistant/thread").json()["entries"] == []
    assert not memory._path_for("auth-disabled").exists()
    others = memory._load_entries(memory._path_for("someone-else"))
    assert [entry["question"] for entry in others] == ["theirs"]

    audit = actions.read_audit(50)["entries"]
    clears = [entry for entry in audit if entry["event"] == "thread_clear"]
    assert len(clears) == 1
    assert clears[0]["user"] == "auth-disabled"
    assert clears[0]["results"] == {"entries_removed": 1}


# --- the ask paths append on success only ------------------------------------------
def test_successful_ask_appends_one_entry(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(assistant, "_client_factory", _answer_factory())
    client.post("/api/assistant/ask", json={"question": "כמה ברייקים השבוע?"})
    entries = client.get("/api/assistant/thread").json()["entries"]
    assert len(entries) == 1
    assert entries[0]["question"] == "כמה ברייקים השבוע?"
    assert entries[0]["answer"] == "grounded answer"
    assert entries[0]["batch_id"] is None


def test_failed_ask_appends_nothing(client: TestClient, monkeypatch) -> None:
    def broken_factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            raise RuntimeError("socket exploded")

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(assistant, "_client_factory", broken_factory)
    client.post("/api/assistant/ask", json={"question": "this one fails"})
    assert client.get("/api/assistant/thread").json()["entries"] == []

    monkeypatch.delenv("KAIROS_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client.post("/api/assistant/ask", json={"question": "no key either"})
    assert client.get("/api/assistant/thread").json()["entries"] == []


def test_corrupt_thread_file_reads_as_empty_and_recovers(client: TestClient) -> None:
    path = memory._path_for("auth-disabled")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert client.get("/api/assistant/thread").json()["entries"] == []
    memory.append_entry("auth-disabled", "after corruption", "recovered", None)
    entries = client.get("/api/assistant/thread").json()["entries"]
    assert [entry["question"] for entry in entries] == ["after corruption"]
