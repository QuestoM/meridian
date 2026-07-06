"""Standing adversarial security tests for the assistant isolation invariants.

Five invariants, each attacked before it is asserted:

1. Competitor exclusion: every registered READ tool plus the composed grounding
   context is serialized and scanned for every non-owned channel name taken
   from the real data (the programmes reference and the committed weekly CSV).
2. Secret and internals containment: a poisoned read executor raising an
   exception that carries a fake API key and a filesystem path must surface
   only the exception type, never the token, the path or a traceback; and no
   real tool result or context may carry internal markers (env-var names,
   password-hash vocabulary, the auth store path).
3. Role cap end to end: with a seeded auth store and real logins, a viewer ask
   offers the model NO propose tools and a forced propose tool_use is refused
   with no pending batch; anonymous requests are walled with 401 through the
   real server middleware; an operator keeps the full toolset.
4. Thread isolation: two logged-in users each see only their own thread, no
   query parameter or path can select another user's thread, and a clear
   removes only the caller's file.
5. Stream parity and safety: the terminal SSE frame equals the non-streaming
   ask body for the same conversation, and an unauthenticated stream request
   is refused before any frame is emitted.

The Claude call is mocked at the module seam (assistant._client_factory); the
data, builders, auth store and thread storage under test are real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_context as assistant_context
import kairos_api.assistant_memory as memory
import kairos_api.assistant_tools as tools

FAKE_SECRET = "sk-fake-123"
FAKE_PATH = "/Users/home/secret.py"
INTERNAL_MARKERS = ("ANTHROPIC_API_KEY", "KAIROS_ASSISTANT", "scrypt", "data/auth")


@pytest.fixture(autouse=True)
def _assistant_state(tmp_path, monkeypatch):
    """Tmp action-plane state (audit, proposals, threads) and a fresh rate
    window on both sides of every test; no ambient model override."""
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_ACTIONS", raising=False)
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    """A seeded auth store with admin, viewer and two operator accounts, at a
    tmp KAIROS_AUTH_DIR so the real data/auth store is never touched."""
    from kairos_api import auth_store

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password="rootpass-1234")
    auth_store.add_user("viewer1", "viewerpass-123", "viewer", "V", must_change_password=False)
    auth_store.add_user("usera", "userapass-123", "operator", "A", must_change_password=False)
    auth_store.add_user("userb", "userbpass-123", "operator", "B", must_change_password=False)
    yield auth_store
    auth_store.reset_runtime_state()


# --- shared helpers ---------------------------------------------------------------
def _channel_facts() -> tuple[Any, str, list[str]]:
    """(owned weekly rows, owned channel, competitor names) from the REAL data:
    the channel universe is the union of the programmes reference and the
    committed weekly CSV, so a leak of any non-owned name is caught."""
    from kairos.data.loaders import load_programmes

    server = assistant_context._server()
    frame = pd.read_csv(server.OUTPUT_DIR / "weekly_break_schedule.csv")
    owned = str(server._load_settings().operator_channel or "").strip()
    assert not frame.empty and owned
    channels = {text for text in frame["channel"].astype(str).str.strip().unique() if text}
    programmes = load_programmes()
    if "Channel" in programmes.columns:
        channels |= {
            text for text in programmes["Channel"].astype(str).str.strip().unique() if text
        }
    competitors = sorted(channels - {owned})
    assert competitors, "the data must carry competitor channels for these tests to bite"
    own = frame[frame["channel"].astype(str).str.strip() == owned]
    return own, owned, competitors


def _read_tool_args(own_rows: Any) -> dict[str, dict[str, Any]]:
    """Real, valid arguments for the read tools that take any."""
    from kairos_api.core import _load_settings

    sample_date = sorted(own_rows["date"].astype(str).str.strip().unique())[0]
    current_weight = int(_load_settings().revenue_weight)
    return {
        "get_day_detail": {"date": sample_date},
        "simulate_settings_change": {
            "changes": {"revenue_weight": 70 if current_weight != 70 else 75}
        },
    }


def _all_read_payloads() -> dict[str, dict[str, Any]]:
    """EVERY registered read tool executed with valid arguments, so a tool
    added later is automatically covered by the leak scans below."""
    own_rows, _, _ = _channel_facts()
    args_by_tool = _read_tool_args(own_rows)
    payloads: dict[str, dict[str, Any]] = {}
    for name in sorted(tools._READ_EXECUTORS):
        payloads[name] = tools.execute_read_tool(name, args_by_tool.get(name, {}))
    return payloads


def _scripted_recording_factory(recorder: dict[str, Any], turns: list[SimpleNamespace]):
    """A fake Anthropic client factory that records every create() kwargs and
    replays the scripted turns."""

    def factory(api_key: str) -> Any:
        remaining = list(turns)

        def create(**kwargs: Any) -> Any:
            recorder.setdefault("calls", []).append(kwargs)
            return remaining.pop(0)

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=f"tu_{name}")


def _turn(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


def _answer_factory(text: str = "grounded answer"):
    """A factory whose client always answers with one plain text turn."""

    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            return _turn(_text_block(text), stop="end_turn")

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


def _mini_client(auth_store=None, username: str | None = None, role: str | None = None) -> TestClient:
    """A client on a mini app mounting only the assistant router. With a store
    and username, a REAL session cookie is attached (route-level auth seam)."""
    app = FastAPI()
    app.include_router(assistant.router)
    client = TestClient(app)
    if auth_store is not None and username is not None:
        client.cookies.set(auth_store.COOKIE_NAME, auth_store.create_session(username, role))
    return client


def _server_login(username: str, password: str) -> TestClient:
    from kairos_api.server import app as server_app

    client = TestClient(server_app)
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return client


def _parse_sse(text: str) -> list[tuple[str, Any]]:
    frames: list[tuple[str, Any]] = []
    for chunk in text.strip().split("\n\n"):
        lines = chunk.split("\n")
        event = next(line[len("event: "):] for line in lines if line.startswith("event: "))
        data = "".join(line[len("data: "):] for line in lines if line.startswith("data: "))
        frames.append((event, json.loads(data)))
    return frames


def _normalized(body: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(body))
    for key in ("grounding", "context_disclosure"):
        if isinstance(clone.get(key), dict):
            clone[key].pop("generated_at", None)
    return clone


# --- invariant 1: competitor exclusion ---------------------------------------------
def test_every_read_tool_excludes_competitor_channel_names() -> None:
    _, _, competitors = _channel_facts()
    payloads = _all_read_payloads()
    leaks: list[str] = []
    for name, payload in payloads.items():
        assert "error" not in payload, f"{name} errored, so the leak scan would not bite: {payload}"
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        for competitor in competitors:
            if competitor in serialized:
                leaks.append(f"{name} leaked competitor channel {competitor}")
    assert not leaks, leaks


def test_composed_context_excludes_competitor_channel_names() -> None:
    own_rows, owned, competitors = _channel_facts()
    sample_date = sorted(own_rows["date"].astype(str).str.strip().unique())[0]
    context, sources = assistant._compose_context(
        f"מה מתוכנן ב-{sample_date} ומה סך ההכנסות השבוע?"
    )
    serialized = json.dumps({"context": context, "sources": sources}, ensure_ascii=False, default=str)
    for competitor in competitors:
        assert competitor not in serialized, f"context leaked competitor channel {competitor}"
    assert context["per_day_plan"]["channel"] == owned


# --- invariant 2: secret and internals containment ----------------------------------
def test_poisoned_executor_crash_surfaces_type_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(args: dict[str, Any]) -> dict[str, Any]:
        raise Exception(f"ANTHROPIC_API_KEY={FAKE_SECRET} loaded at {FAKE_PATH}")

    monkeypatch.setitem(tools._READ_EXECUTORS, "get_settings", boom)
    payload = tools.execute_read_tool("get_settings", {})
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    # The result names the exception type honestly and points at the server log.
    assert "Exception" in payload["error"]
    assert "server log" in payload["error"]
    # Neither the token, the path, the env-var name nor a traceback may surface.
    for banned in (FAKE_SECRET, FAKE_PATH, "ANTHROPIC_API_KEY", "Traceback", "secret"):
        assert banned not in serialized, f"tool result leaked {banned!r}"
    # The same containment holds for the tool_result content the model sees.
    block = _tool_use("get_settings", {})
    trace: list[dict[str, Any]] = []
    result = tools.handle_tool_use(block, trace, [])
    for banned in (FAKE_SECRET, FAKE_PATH, "ANTHROPIC_API_KEY", "Traceback"):
        assert banned not in result["content"]
    assert trace == [{"tool": "get_settings", "ok": False, "source": "saved settings"}]


def test_no_internal_markers_in_any_real_tool_result_or_context() -> None:
    for name, payload in _all_read_payloads().items():
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        for marker in INTERNAL_MARKERS:
            assert marker not in serialized, f"{name} carries internal marker {marker!r}"
    context, sources = assistant._compose_context("סיכום שבועי של הערוץ")
    serialized = json.dumps({"context": context, "sources": sources}, ensure_ascii=False, default=str)
    for marker in INTERNAL_MARKERS:
        assert marker not in serialized, f"composed context carries internal marker {marker!r}"


# --- invariant 3: role cap end to end -----------------------------------------------
def test_viewer_ask_offers_no_propose_tools_and_forced_propose_is_refused(
    auth_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    recorder: dict[str, Any] = {}
    script = [
        _turn(_tool_use("propose_settings_change",
                        {"changes": {"revenue_weight": 55}, "reason": "forced by mock"})),
        _turn(_text_block("the role does not allow proposing changes"), stop="end_turn"),
    ]
    monkeypatch.setattr(assistant, "_client_factory", _scripted_recording_factory(recorder, script))

    viewer = _mini_client(auth_env, "viewer1", "viewer")
    body = viewer.post("/api/assistant/ask", json={"question": "העלה את משקל ההכנסה"}).json()

    # The model was never offered a propose tool: the read-only toolset exactly.
    offered = {tool["name"] for tool in recorder["calls"][0]["tools"]}
    assert offered == set(tools.READ_TOOL_NAMES)
    assert not offered & set(tools.PROPOSE_TOOL_NAMES)

    # The forced propose came back as an honest refusal the model can read.
    tool_results = recorder["calls"][1]["messages"][-1]["content"]
    assert "does not allow proposing" in tool_results[0]["content"]
    assert body["error"] is None
    assert body["answer"] == "the role does not allow proposing changes"
    assert body["proposals"] is None
    assert body["tool_trace"] == [{"tool": "propose_settings_change", "ok": False}]

    # NO pending batch was created anywhere.
    assert actions.list_proposals()["batches"] == []


def test_operator_keeps_the_full_toolset_through_the_real_server(
    auth_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(
        assistant, "_client_factory",
        _scripted_recording_factory(recorder, [_turn(_text_block("ok"), stop="end_turn")]),
    )
    operator = _server_login("usera", "userapass-123")
    response = operator.post("/api/assistant/ask", json={"question": "מה ההגדרות?"})
    assert response.status_code == 200
    offered = {tool["name"] for tool in recorder["calls"][0]["tools"]}
    assert offered == set(tools.READ_TOOL_NAMES) | set(tools.PROPOSE_TOOL_NAMES)


def test_anonymous_ask_stream_and_thread_are_walled_with_401(auth_env) -> None:
    from kairos_api.server import app as server_app

    anonymous = TestClient(server_app)
    assert anonymous.post("/api/assistant/ask", json={"question": "מי אני?"}).status_code == 401
    stream = anonymous.post("/api/assistant/ask/stream", json={"question": "מי אני?"})
    # Refused before any frame: a JSON denial, not an event stream.
    assert stream.status_code == 401
    assert not stream.headers["content-type"].startswith("text/event-stream")
    assert "event:" not in stream.text
    assert anonymous.get("/api/assistant/thread").status_code == 401
    assert anonymous.delete("/api/assistant/thread").status_code == 401


def test_viewer_is_walled_from_ask_by_the_server_guard(auth_env) -> None:
    # Defense in depth: through the deployed server the mutating-method guard
    # stops a viewer POST before the route runs, so the read-only toolset in
    # _can_propose is the route-level backstop, proven on the mini app above.
    viewer = _server_login("viewer1", "viewerpass-123")
    assert viewer.post("/api/assistant/ask", json={"question": "שאלה"}).status_code == 403
    assert viewer.post("/api/assistant/ask/stream", json={"question": "שאלה"}).status_code == 403
    assert viewer.delete("/api/assistant/thread").status_code == 403
    # Reading the own thread stays open to a viewer session.
    thread = viewer.get("/api/assistant/thread")
    assert thread.status_code == 200
    assert thread.json()["user"] == "viewer1"


# --- invariant 4: thread isolation ---------------------------------------------------
def test_threads_are_isolated_per_session_user_and_unaddressable(
    auth_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    monkeypatch.setattr(assistant, "_client_factory", _answer_factory())

    user_a = _server_login("usera", "userapass-123")
    user_b = _server_login("userb", "userbpass-123")
    assert user_a.post("/api/assistant/ask", json={"question": "question from usera"}).status_code == 200
    assert user_b.post("/api/assistant/ask", json={"question": "question from userb"}).status_code == 200

    thread_a = user_a.get("/api/assistant/thread").json()
    thread_b = user_b.get("/api/assistant/thread").json()
    assert thread_a["user"] == "usera"
    assert [entry["question"] for entry in thread_a["entries"]] == ["question from usera"]
    assert thread_b["user"] == "userb"
    assert [entry["question"] for entry in thread_b["entries"]] == ["question from userb"]

    # No request parameter can select another user's thread: the identity is
    # the session username only, so every probe still returns A's own thread.
    probes = (
        "?user=userb", "?username=userb", "?actor=userb", "?session=userb",
        "?user_id=userb", "?thread=userb", "?name=userb",
    )
    for probe in probes:
        probed = user_a.get(f"/api/assistant/thread{probe}")
        assert probed.status_code == 200, probe
        body = probed.json()
        assert body["user"] == "usera", probe
        assert [entry["question"] for entry in body["entries"]] == ["question from usera"], probe
    # No path variant addresses another user's thread either.
    assert user_a.get("/api/assistant/thread/userb").status_code in (404, 405)
    assert user_a.delete("/api/assistant/thread/userb").status_code in (404, 405)
    assert user_a.get("/api/assistant/threads/userb").status_code in (404, 405)

    # A's clear removes only A's file; B's thread survives on disk and over the wire.
    cleared = user_a.delete("/api/assistant/thread").json()
    assert cleared == {"cleared": True, "entries_removed": 1, "user": "usera"}
    assert user_a.get("/api/assistant/thread").json()["entries"] == []
    survivor = user_b.get("/api/assistant/thread").json()
    assert [entry["question"] for entry in survivor["entries"]] == ["question from userb"]
    assert not memory._path_for("usera").exists()
    assert memory._path_for("userb").exists()


# --- invariant 5: stream parity and safety -------------------------------------------
def test_stream_final_frame_equals_the_nonstreaming_ask_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    script = lambda: [  # noqa: E731 - a fresh script per client, the loop pops it
        _turn(_text_block("checking"), _tool_use("get_settings", {})),
        _turn(_text_block("the grounded answer"), stop="end_turn"),
    ]
    client = _mini_client()

    monkeypatch.setattr(assistant, "_client_factory", _scripted_recording_factory({}, script()))
    streamed = client.post("/api/assistant/ask/stream", json={"question": "אותה שאלה"})
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(streamed.text)
    kinds = [event for event, _ in frames]
    assert kinds.count("final") == 1 and kinds.count("error") == 0
    assert kinds[-1] == "final"
    final = frames[-1][1]
    assert {"answer", "error", "model", "context_disclosure", "truncated",
            "proposals", "tool_trace"} <= set(final)

    monkeypatch.setattr(assistant, "_client_factory", _scripted_recording_factory({}, script()))
    plain = client.post("/api/assistant/ask", json={"question": "אותה שאלה"}).json()
    assert _normalized(final) == _normalized(plain)
