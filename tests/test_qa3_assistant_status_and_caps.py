"""Exact-contract pins the ask-pipeline suite leaves at the edges.

Two mandate points are checked here end to end through the ask, because the
sibling suites prove them only at the unit level: the /api/assistant/status
route now names the upgraded default model (and still honours the env
override), and the history replay carried in the messages array respects the
exchange cap and the char budget with the per-answer truncation marker exactly
as specified (seed eight short exchanges, expect the newest six; a long answer
arrives cut with the explicit marker; an over-budget thread is trimmed under
HISTORY_CHAR_BUDGET). Every model call is mocked through assistant._client_factory
with a recording fake, so no real Anthropic request is made. Thread state is
relocated to a tmp dir via KAIROS_ASSISTANT_DATA_DIR; nothing under data/ is
read or written. Runs against a mini FastAPI app; the live server is never
touched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_context as assistant_context
import kairos_api.assistant_history as history
import kairos_api.assistant_memory as memory


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    # State to tmp, no ambient key/model/actions leaking in from the environment.
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_ACTIONS", raising=False)
    monkeypatch.delenv(assistant_context.BUDGET_ENV, raising=False)
    # The composition is exercised by the sibling suites; keep these focused on
    # the status payload and on the history slice of the messages array.
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


def _history_pairs(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The replayed exchanges in front of the final CONTEXT+QUESTION message.

    The last message is always the current turn; everything before it is the
    strict user/assistant history alternation, returned as (question, answer)
    pairs oldest first.
    """
    history_turns = messages[:-1]
    assert len(history_turns) % 2 == 0
    assert [turn["role"] for turn in history_turns] == ["user", "assistant"] * (len(history_turns) // 2)
    return [
        (history_turns[i]["content"], history_turns[i + 1]["content"])
        for i in range(0, len(history_turns), 2)
    ]


# --- /api/assistant/status names the upgraded default model ----------------------
def test_status_route_reports_the_upgraded_default_model(client: TestClient) -> None:
    body = client.get("/api/assistant/status").json()
    assert body["model"] == "claude-opus-5" == assistant.DEFAULT_MODEL
    # Honest availability is unchanged: no key configured, so unavailable.
    assert body["available"] is False
    assert body["action_plane"]["enabled"] is False


def test_status_route_honours_the_model_env_override(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setenv("KAIROS_ASSISTANT_MODEL", "claude-sonnet-4-6")
    body = client.get("/api/assistant/status").json()
    assert body["model"] == "claude-sonnet-4-6"
    assert body["available"] is True


# --- the exchange cap, observed in the messages array through a real ask ----------
def test_ask_replays_exactly_six_of_eight_short_exchanges_newest_first(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    # Eight short exchanges: well within the char budget, so the exchange cap is
    # the only thing that can bind. The window keeps the newest six.
    for i in range(8):
        memory.append_entry("auth-disabled", f"q{i}", f"a{i}")
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))

    response = client.post("/api/assistant/ask", json={"question": "current turn"})
    assert response.status_code == 200

    messages = recorder["calls"][0]["messages"]
    pairs = _history_pairs(messages)
    assert len(pairs) == history.HISTORY_MAX_EXCHANGES == 6
    # The newest six (q2..q7), oldest first, and the eldest two (q0, q1) dropped.
    assert pairs == [(f"q{i}", f"a{i}") for i in range(2, 8)]
    # The current turn is the last message and carries the composed CONTEXT block.
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("CONTEXT:\n")
    assert messages[-1]["content"].rstrip().endswith("current turn")


# --- the per-answer truncation marker survives into the replayed message ----------
def test_ask_replays_a_long_answer_cut_with_the_explicit_marker(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    long_answer = "y" * (history.ANSWER_REPLAY_CHARS + 500)
    memory.append_entry("auth-disabled", "a long one", long_answer)
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))

    client.post("/api/assistant/ask", json={"question": "follow up"})
    ((question, replayed),) = _history_pairs(recorder["calls"][0]["messages"])
    assert question == "a long one"
    # The stored answer is replayed shortened, with the explicit marker, never
    # silently dropped and never replayed whole.
    assert replayed.endswith(history.ANSWER_TRUNCATION_MARKER)
    assert len(replayed) < len(long_answer)
    assert len(replayed) <= history.ANSWER_REPLAY_CHARS + len(history.ANSWER_TRUNCATION_MARKER) + 1


# --- the char budget bounds the replayed history through a real ask ---------------
def test_ask_history_replay_respects_the_char_budget(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    # Each answer is capped to ANSWER_REPLAY_CHARS on replay, so a handful of
    # long exchanges already exceed HISTORY_CHAR_BUDGET; only the newest that
    # fit are replayed, and the total replayed text stays under the budget.
    for i in range(8):
        memory.append_entry("auth-disabled", f"q{i:02d}", "z" * (history.ANSWER_REPLAY_CHARS + 300))
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))

    client.post("/api/assistant/ask", json={"question": "current"})
    pairs = _history_pairs(recorder["calls"][0]["messages"])
    assert 0 < len(pairs) < history.HISTORY_MAX_EXCHANGES
    used = sum(len(question) + len(answer) for question, answer in pairs)
    assert used <= history.HISTORY_CHAR_BUDGET
    # The survivors are the newest exchanges, oldest first among them.
    kept_questions = [question for question, _ in pairs]
    assert kept_questions == sorted(kept_questions)
    assert kept_questions[-1] == "q07"
