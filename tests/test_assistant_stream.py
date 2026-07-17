"""Contract tests for the streaming ask endpoint (SSE).

The Claude call is mocked at the module seam (assistant._client_factory), with
and without messages.stream support, so no key is ever needed. The contract
under test is OURS: frame order (steps right after tool results, deltas as text
is produced, exactly one terminal frame), the final frame carrying the EXACT
body the non-streaming /ask returns for the same question, and audit plus
thread-append happening exactly once per streamed ask. Runs against a mini
FastAPI app mounting only the assistant router; the live server is never
touched.
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
def stream_env(tmp_path, monkeypatch):
    """Tmp action-plane state, a fake key, a fast context, a fresh rate window."""
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


# --- scripted Anthropic mocks ---------------------------------------------------
def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use(name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=f"tu_{name}")


def model_turn(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


class _FakeStream:
    """A minimal messages.stream context manager: chunked text deltas, then the
    scripted response as the final message."""

    def __init__(self, response: SimpleNamespace, chunk: int = 7) -> None:
        self._response = response
        self._chunk = chunk

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    @property
    def text_stream(self):
        for block in self._response.content:
            if getattr(block, "type", "") == "text":
                text = block.text
                for start in range(0, len(text), self._chunk):
                    yield text[start:start + self._chunk]

    def get_final_message(self) -> SimpleNamespace:
        return self._response


def scripted_factory(turns: list[SimpleNamespace], with_stream: bool = False):
    """A fake client factory replaying scripted turns via create() and,
    optionally, via a messages.stream() that yields chunked text deltas."""

    def factory(api_key: str) -> Any:
        remaining = list(turns)

        def create(**kwargs: Any) -> Any:
            return remaining.pop(0)

        messages = SimpleNamespace(create=create)
        if with_stream:
            def stream(**kwargs: Any) -> _FakeStream:
                return _FakeStream(remaining.pop(0))

            messages.stream = stream
        return SimpleNamespace(messages=messages)

    return factory


def parse_sse(text: str) -> list[tuple[str, Any]]:
    """(event, data) frames in order. json.dumps escapes newlines, so the
    blank-line frame delimiter is unambiguous."""
    frames: list[tuple[str, Any]] = []
    for chunk in text.strip().split("\n\n"):
        lines = chunk.split("\n")
        event = next(line[len("event: "):] for line in lines if line.startswith("event: "))
        data = "".join(line[len("data: "):] for line in lines if line.startswith("data: "))
        frames.append((event, json.loads(data)))
    return frames


def _normalized(body: dict[str, Any]) -> dict[str, Any]:
    """The ask body with its per-request timestamps removed, for equivalence."""
    clone = json.loads(json.dumps(body))
    for key in ("grounding", "context_disclosure"):
        if isinstance(clone.get(key), dict):
            clone[key].pop("generated_at", None)
    return clone


TOOL_SCRIPT = lambda: [  # noqa: E731 - a fresh script per client, the loop pops it
    model_turn(text_block("checking the settings"), tool_use("get_settings", {})),
    model_turn(text_block("the final grounded answer"), stop="end_turn"),
]


# --- frame order and content ----------------------------------------------------
def test_stream_frames_step_delta_final_in_order(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(TOOL_SCRIPT()))
    response = client.post("/api/assistant/ask/stream", json={"question": "מה ההגדרות?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = parse_sse(response.text)
    kinds = [event for event, _ in frames]
    # Exactly one terminal frame, and it is last.
    assert kinds.count("final") == 1 and kinds.count("error") == 0
    assert kinds[-1] == "final"
    # The step frame lands after the first turn's delta and before the final
    # turn's delta: steps are emitted right after each tool result.
    assert kinds.index("step") > kinds.index("delta")
    steps = [data for event, data in frames if event == "step"]
    assert steps == [{"tool": "get_settings", "ok": True, "source": "saved settings"}]
    # Without messages.stream support each turn's answer arrives as ONE delta.
    deltas = [data["text"] for event, data in frames if event == "delta"]
    assert deltas == ["checking the settings", "the final grounded answer"]

    final = frames[-1][1]
    assert final["answer"] == "the final grounded answer"
    assert {"answer", "error", "model", "context_disclosure", "truncated",
            "proposals", "tool_trace"} <= set(final)
    assert final["model"] == "claude-opus-4-8"
    assert [step["tool"] for step in final["tool_trace"]] == ["get_settings"]


def test_stream_final_body_equals_nonstreaming_ask_body(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(TOOL_SCRIPT()))
    streamed = parse_sse(
        client.post("/api/assistant/ask/stream", json={"question": "same question"}).text
    )[-1]
    assert streamed[0] == "final"

    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(TOOL_SCRIPT()))
    plain = client.post("/api/assistant/ask", json={"question": "same question"}).json()

    assert _normalized(streamed[1]) == _normalized(plain)


def test_stream_with_sdk_stream_support_emits_chunked_deltas(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        assistant, "_client_factory", scripted_factory(TOOL_SCRIPT(), with_stream=True)
    )
    frames = parse_sse(
        client.post("/api/assistant/ask/stream", json={"question": "chunked?"}).text
    )
    deltas = [data["text"] for event, data in frames if event == "delta"]
    # The fake stream chunks each turn's text, so real deltas mean MORE frames
    # than turns, and their concatenation reproduces the text exactly.
    assert len(deltas) > 2
    assert "".join(deltas) == "checking the settings" + "the final grounded answer"
    final = frames[-1]
    assert final[0] == "final"
    assert final[1]["answer"] == "the final grounded answer"
    assert [step["tool"] for step in final[1]["tool_trace"]] == ["get_settings"]


# --- audit and thread append happen exactly once ---------------------------------
def test_stream_audits_and_appends_thread_exactly_once(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(TOOL_SCRIPT()))
    client.post("/api/assistant/ask/stream", json={"question": "רק פעם אחת"})

    events = [entry["event"] for entry in actions.read_audit(500)["entries"]]
    assert events.count("ask") == 1

    thread = client.get("/api/assistant/thread").json()
    assert thread["user"] == "auth-disabled"
    assert len(thread["entries"]) == 1
    entry = thread["entries"][0]
    assert entry["question"] == "רק פעם אחת"
    assert entry["answer"] == "the final grounded answer"
    assert entry["batch_id"] is None


def test_stream_proposal_batch_id_reaches_final_body_and_thread(client: TestClient, monkeypatch) -> None:
    import kairos_api.assistant_simulate as simulate

    # The simulated effect runs the real optimizer; stub it so the proposal
    # capture path stays fast and deterministic here.
    monkeypatch.setattr(
        simulate, "settings_effect",
        lambda changes: {"status": "unavailable", "reason": "stubbed in test"},
    )
    script = [
        model_turn(tool_use("propose_settings_change",
                            {"changes": {"revenue_weight": 55}, "reason": "test reason"})),
        model_turn(text_block("proposed for review"), stop="end_turn"),
    ]
    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(script))
    frames = parse_sse(
        client.post("/api/assistant/ask/stream", json={"question": "raise the weight"}).text
    )
    steps = [data for event, data in frames if event == "step"]
    assert steps == [{"tool": "propose_settings_change", "ok": True, "source": None}]
    final = frames[-1][1]
    batch_id = final["proposals"]["batch_id"]
    assert batch_id
    thread = client.get("/api/assistant/thread").json()
    assert thread["entries"][-1]["batch_id"] == batch_id


# --- protection parity with the non-streaming ask --------------------------------
def test_stream_validates_and_rate_limits_like_ask(client: TestClient, monkeypatch) -> None:
    assert client.post("/api/assistant/ask/stream", json={"question": ""}).status_code == 422
    assert client.post("/api/assistant/ask/stream", json={"question": "   "}).status_code == 422
    assert client.post("/api/assistant/ask/stream", json={"question": "x" * 2001}).status_code == 422

    # The two routes share ONE sliding window: ten asks (either route) then 429.
    monkeypatch.setattr(
        assistant, "_client_factory",
        lambda key: SimpleNamespace(messages=SimpleNamespace(
            create=lambda **kwargs: model_turn(text_block("ok"), stop="end_turn"))),
    )
    for index in range(10):
        route = "/api/assistant/ask/stream" if index % 2 else "/api/assistant/ask"
        assert client.post(route, json={"question": f"q {index}"}).status_code == 200
    blocked = client.post("/api/assistant/ask/stream", json={"question": "over budget"})
    assert blocked.status_code == 429


def test_stream_without_key_final_frame_is_honest(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("KAIROS_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frames = parse_sse(
        client.post("/api/assistant/ask/stream", json={"question": "בלי מפתח"}).text
    )
    assert [event for event, _ in frames] == ["final"]
    final = frames[0][1]
    assert final["available"] is False
    assert final["answer"] is None
    assert final["error"] == "API key not configured"
    # No answer means no thread entry.
    assert client.get("/api/assistant/thread").json()["entries"] == []
