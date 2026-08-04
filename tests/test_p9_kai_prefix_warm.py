"""P9: the model's cached prefix is written while the person is still typing.

The measured defect, round 5, named by a blind critic: "first token on an action
ask was 2,855 ms against the 2 s budget in job-stories.md (best case 2,075 ms on
a grounded question, worst 9,359 ms on the turn that skipped the tools)".

Measured here, 2026-08-04, on this machine, through the dock's own endpoints. A
fresh server, ``POST /api/assistant/context/warm`` first exactly as the panel
does on mount, then one Hebrew ask streamed: grounding at 0.28 s, first model
turn at 0.959 s, first text delta at 9.032 s. The same payload with the prefix
already in the cache returned its first text in 1.804 s and 2.717 s, and the
API's own usage record prices that prefix at 16,455 cached input tokens (39 tool
schemas and three system blocks). The 9 s was the cache WRITE, paid in front of
a person watching a cursor.

So the warm call the panel already makes now writes that prefix too. What this
file proves is the part a stopwatch cannot: that the prefix written is byte for
byte the one the next ask sends (a different one would be a different cache key
and the ask would pay the write anyway), that it is written once per lifetime
rather than once per mount, that a failure is named and backed off rather than
retried per keystroke, and that the reply is discarded and reaches no person, no
thread and no audit line.

The Claude call is mocked at the module seam; no key and no live server is used.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_pipeline as pipeline
import kairos_api.assistant_tools as assistant_tools
import kairos_api.assistant_warm as warm


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Any:
    monkeypatch.setenv("KAIROS_ASSISTANT_DATA_DIR", str(tmp_path / "assistant"))
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    warm.reset()
    assistant._reset_rate_limit()
    yield
    warm.reset()
    assistant._reset_rate_limit()


def _recording(calls: list[dict[str, Any]], error: Exception | None = None):
    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if error is not None:
                raise error
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="")],
                                   stop_reason="max_tokens")

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


# --- the prefix written is the prefix the ask sends ---------------------------

def test_the_warm_call_sends_exactly_what_the_next_ask_will_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prefix written with different tools or different system blocks is a
    different cache key, so the ask would pay the write and this would be a
    model call spent on nothing."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(calls))
    assert warm.warm_prefix(sync=True)["state"] == "warm"
    assert len(calls) == 1
    written = calls[0]
    ask_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(ask_calls))
    pipeline.run_tool_loop(assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nמה מצב השבוע",
                           [], [], actions_on=True)
    assert written["tools"] == ask_calls[0]["tools"]
    assert written["system"] == ask_calls[0]["system"]
    assert written["model"] == ask_calls[0]["model"]
    # The breakpoint that makes it one cacheable unit is on the last block.
    assert written["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # One output token: the write happens while the input is processed.
    assert written["max_tokens"] == warm.WARM_MAX_TOKENS == 1


def test_an_account_that_may_not_propose_warms_its_own_smaller_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool array sits BEFORE the system blocks in the cached prefix, so a
    viewer's 31 tools and an operator's 39 are two different keys."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(calls))
    warm.warm_prefix(can_propose=False, sync=True)
    warm.warm_prefix(can_propose=True, sync=True)
    viewer, operator = calls
    assert viewer["tools"] == assistant_tools.anthropic_tools(include_propose=False)
    assert operator["tools"] == assistant_tools.anthropic_tools(include_propose=True)
    assert len(viewer["tools"]) < len(operator["tools"])
    assert not any(tool["name"].startswith("propose") for tool in viewer["tools"])


def test_a_declared_job_warms_its_own_prefix_because_it_rides_in_the_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(calls))
    warm.warm_prefix(job="planner", sync=True)
    assert calls[0]["system"] == assistant._system_blocks(auth_mode="api_key", job="planner")


# --- it is written once, not once per mount -----------------------------------

def test_the_second_mount_inside_the_cache_lifetime_spends_no_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(calls))
    assert warm.warm_prefix(sync=True)["state"] == "warm"
    second = warm.warm_prefix(sync=True)
    assert second["state"] == "warm"
    assert second["age_seconds"] >= 0
    assert len(calls) == 1, "one write per prefix per lifetime, never one per mount"


def test_a_prefix_older_than_the_cache_is_written_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache lapses after five minutes and so does this record, or a dock
    left open all morning would report warm over an empty cache."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(calls))
    warm.warm_prefix(sync=True)
    aged = time.monotonic() - (warm.PREFIX_TTL_SECONDS + 1)
    for record in (warm._SUCCEEDED, warm._ATTEMPTED):
        for key in list(record):
            record[key] = aged
    assert warm.warm_prefix(sync=True)["state"] == "warm"
    assert len(calls) == 2


def test_a_write_still_in_flight_is_never_started_a_second_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured at 14.76 s on a throttled account, which is longer than the
    retry window. Without this guard the second mount would pay for the same
    prefix again, and the person would wait for whichever call finished last."""
    calls: list[dict[str, Any]] = []
    released = threading.Event()

    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            released.wait(5.0)
            return SimpleNamespace(content=[], stop_reason="max_tokens")

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(assistant, "_client_factory", factory)
    assert warm.warm_prefix()["state"] == "warming"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not calls:
        time.sleep(0.01)
    during = warm.warm_prefix()
    assert during["state"] == "writing"
    assert during["started_seconds_ago"] >= 0
    released.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and warm.last_attempt().get("state") != "warm":
        time.sleep(0.01)
    assert warm.last_attempt()["state"] == "warm"
    assert len(calls) == 1


# --- a failure is named, and it is not retried per keystroke ------------------

def test_a_failed_write_is_named_and_backed_off(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory",
                        _recording(calls, RuntimeError("connection reset")))
    failed = warm.warm_prefix(sync=True)
    assert failed["state"] == "failed"
    assert "connection reset" in failed["error"]
    waiting = warm.warm_prefix(sync=True)
    assert waiting["state"] == "waiting" and waiting["last"] == "failed"
    assert "connection reset" in waiting["error"]
    assert len(calls) == 1, "a rejected credential costs one call, not one per mount"


def test_no_credentials_is_reported_as_itself_and_spends_no_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(calls))
    monkeypatch.setattr(assistant, "_resolve_auth", lambda: None)
    outcome = warm.warm_prefix(sync=True)
    assert outcome == {"state": "unavailable", "reason": assistant.AUTH_MISSING_REASON,
                       "at": outcome["at"]}
    assert calls == []


# --- the route, and what the warm must not touch ------------------------------

def test_the_route_reports_the_prefix_beside_the_context_and_keeps_its_own_keys(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(calls))
    body = client.post("/api/assistant/context/warm").json()
    assert {"sections", "absent", "already_warm", "elapsed_seconds"} <= set(body)
    assert body["model_prefix"]["state"] == "warming"
    # The route does not wait for the model, so the write lands on its own
    # thread; it is still a real write and this is where that is proved.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and warm.last_attempt().get("state") != "warm":
        time.sleep(0.02)
    assert warm.last_attempt()["state"] == "warm"
    assert len(calls) == 1


def test_the_warm_reply_reaches_no_person_no_thread_and_no_audit(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """It is a cache write, not an ask. Nothing it returns is stored, shown or
    counted, and the discarded reply can never appear in a conversation."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _recording(calls))
    warm.warm_prefix(sync=True)
    thread = client.get("/api/assistant/thread").json()
    assert thread.get("entries") == []
    audit = client.get("/api/assistant/audit").json()
    assert audit["entries"] == []
    assert calls[0]["messages"] == [{"role": "user", "content": warm.WARM_QUESTION}]
