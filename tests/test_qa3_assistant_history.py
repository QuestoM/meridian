"""Multi-turn memory: the saved thread is replayed into the model call.

The Claude call is mocked at the module seam (assistant._client_factory); the
contract under test is OURS: the exact messages array shape (history turns as
alternating user/assistant pairs, oldest first, BEFORE the current
CONTEXT+QUESTION message), the windowing caps (HISTORY_MAX_EXCHANGES within
HISTORY_CHAR_BUDGET, per-answer truncation with an explicit marker), same-user
scoping, parity between /ask and /ask/stream, and the honesty rule that a
memory read failure yields an empty history rather than a failed ask. Thread
state is relocated to tmp via KAIROS_ASSISTANT_DATA_DIR; data/ is never
touched. Runs against a mini FastAPI app; no live server, no real API call.
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
import kairos_api.assistant_history as history
import kairos_api.assistant_memory as memory


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_ACTIONS", raising=False)
    # Context composition is exercised elsewhere; keep these fast and focused.
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


def _recording_factory(recorder: dict[str, Any], answer: str = "the fresh answer"):
    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            recorder.setdefault("calls", []).append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=answer)], stop_reason="end_turn"
            )

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


# --- windowing unit contract ---------------------------------------------------
def test_window_caps_exchanges_and_orders_oldest_first() -> None:
    entries = [{"question": f"q{i}", "answer": f"a{i}", "at": "t", "batch_id": None} for i in range(9)]
    window = history._window(entries)
    assert len(window) == history.HISTORY_MAX_EXCHANGES == 6
    # The newest six exchanges, replayed oldest first.
    assert window == [(f"q{i}", f"a{i}") for i in range(3, 9)]


def test_window_respects_char_budget_newest_first() -> None:
    # Each exchange costs ~2030 chars after truncation, so only five of the
    # eight fit the 12000-char budget, and they are the five newest.
    entries = [
        {"question": f"q{i:02d}", "answer": "x" * 2500, "at": "t", "batch_id": None}
        for i in range(8)
    ]
    window = history._window(entries)
    assert 0 < len(window) < 6
    assert [q for q, _ in window] == [f"q{i:02d}" for i in range(8 - len(window), 8)]
    used = sum(len(q) + len(a) for q, a in window)
    assert used <= history.HISTORY_CHAR_BUDGET


def test_window_truncates_answers_with_explicit_marker() -> None:
    entries = [{"question": "q", "answer": "y" * 5000, "at": "t", "batch_id": None}]
    ((_, answer),) = history._window(entries)
    assert answer.endswith(history.ANSWER_TRUNCATION_MARKER)
    assert len(answer) <= history.ANSWER_REPLAY_CHARS + len(history.ANSWER_TRUNCATION_MARKER) + 1
    short = [{"question": "q", "answer": "short", "at": "t", "batch_id": None}]
    assert history._window(short) == [("q", "short")]


def test_window_skips_malformed_entries_preserving_alternation() -> None:
    entries = [
        {"question": "kept early", "answer": "a1", "at": "t", "batch_id": None},
        {"question": "", "answer": "orphan answer", "at": "t", "batch_id": None},
        {"question": "orphan question", "answer": None, "at": "t", "batch_id": None},
        {"question": "kept late", "answer": "a2", "at": "t", "batch_id": None},
    ]
    assert history._window(entries) == [("kept early", "a1"), ("kept late", "a2")]


def test_history_messages_alternate_roles_and_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    memory.append_entry("auth-disabled", "כמה ברייקים ביום שלישי?", "42 ברייקים")
    memory.append_entry("auth-disabled", "ומה ההכנסה הצפויה?", "1,000 ILS")
    messages = history.history_messages("auth-disabled")
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "כמה ברייקים ביום שלישי?"
    assert messages[3]["content"] == "1,000 ILS"
    # Same-user only: another user's thread stays invisible.
    assert history.history_messages("someone-else") == []
    # A memory read failure yields an honest empty history, never an exception.
    monkeypatch.setattr(memory, "_load_entries", lambda path: 1 / 0)
    assert history.history_messages("auth-disabled") == []


# --- the ask pipeline replays history before the current message -----------------
def test_ask_places_history_turns_before_context_question(client: TestClient, monkeypatch) -> None:
    memory.append_entry("auth-disabled", "שאלה ראשונה", "תשובה ראשונה")
    memory.append_entry("auth-disabled", "שאלה שנייה", "תשובה שנייה")
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))

    response = client.post("/api/assistant/ask", json={"question": "שאלת המשך"})
    assert response.status_code == 200
    assert response.json()["answer"] == "the fresh answer"

    kwargs = recorder["calls"][0]
    messages = kwargs["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"] == "שאלה ראשונה"
    assert messages[1]["content"] == "תשובה ראשונה"
    assert messages[2]["content"] == "שאלה שנייה"
    assert messages[3]["content"] == "תשובה שנייה"
    # History turns carry ONLY the past question text, never a CONTEXT block;
    # the current turn is the one CONTEXT+QUESTION message, placed last.
    for turn in messages[:-1]:
        assert not str(turn["content"]).startswith("CONTEXT:")
    assert messages[-1]["content"].startswith("CONTEXT:\n")
    assert messages[-1]["content"].rstrip().endswith("שאלת המשך")
    # The upgraded model and raised ceilings ride on the same call.
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] == assistant.LOOP_MAX_TOKENS == 4000


def test_first_ask_of_a_fresh_thread_has_no_history_turns(client: TestClient, monkeypatch) -> None:
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    client.post("/api/assistant/ask", json={"question": "שאלה בלי עבר"})
    messages = recorder["calls"][0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("CONTEXT:\n")


def test_memory_read_failure_never_fails_the_ask(client: TestClient, monkeypatch) -> None:
    memory.append_entry("auth-disabled", "q", "a")
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    monkeypatch.setattr(memory, "_load_entries", lambda path: 1 / 0)
    response = client.post("/api/assistant/ask", json={"question": "עדיין עובד?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "the fresh answer"
    assert body["error"] is None
    assert len(recorder["calls"][0]["messages"]) == 1


def test_consecutive_asks_accumulate_history(client: TestClient, monkeypatch) -> None:
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    client.post("/api/assistant/ask", json={"question": "ראשונה"})
    client.post("/api/assistant/ask", json={"question": "שנייה"})
    first, second = recorder["calls"][0]["messages"], recorder["calls"][1]["messages"]
    assert len(first) == 1
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[0]["content"] == "ראשונה"
    assert second[1]["content"] == "the fresh answer"


# --- stream parity: the same history flows through /ask/stream -------------------
def test_stream_ask_replays_the_same_history(client: TestClient, monkeypatch) -> None:
    memory.append_entry("auth-disabled", "שאלה קודמת", "תשובה קודמת")
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    response = client.post("/api/assistant/ask/stream", json={"question": "המשך בסטרים"})
    assert response.status_code == 200
    assert "event: final" in response.text
    final = json.loads(response.text.strip().split("\n\n")[-1].split("\ndata: ")[1])
    assert final["answer"] == "the fresh answer"
    assert final["model"] == "claude-opus-5"
    messages = recorder["calls"][0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "שאלה קודמת"
    assert messages[1]["content"] == "תשובה קודמת"
    assert messages[-1]["content"].startswith("CONTEXT:\n")
