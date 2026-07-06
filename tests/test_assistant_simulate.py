"""Contract tests for the assistant's in-chat simulation, provenance and goal-seek.

The simulation primitive runs the REAL owned-channel scenario runner on the
repository's saved data, so the before/after figures are genuine optimizer
output, not placeholders. The Claude call is mocked at the module seam
(assistant._client_factory) with scripted tool-use turns, exactly as the action
plane suite does, so the goal-seek loop is exercised end to end without a key.
Everything downstream of the model is real: the primitive, the read-tool
routing, the provenance stamping, and the proposal effect.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_simulate as simulate
import kairos_api.assistant_tools as tools
from kairos_api import core

ROOT = Path(__file__).resolve().parents[1]
SIM_SOURCE = "owned-channel scenario runner, representative day"
_MONEY_KEYS = {"gross", "retention_cost", "net", "breaks"}


# --- scripted Anthropic mock (mirrors the action-plane suite) ------------------
def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use(name: str, args: dict[str, Any], block_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=block_id or f"tu_{name}")


def model_turn(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


def scripted_factory(recorder: dict[str, Any], turns: list[SimpleNamespace]):
    remaining = list(turns)

    def factory(api_key: str) -> Any:
        recorder["api_key"] = api_key

        def create(**kwargs: Any) -> Any:
            snapshot = dict(kwargs)
            snapshot["messages"] = list(kwargs.get("messages") or [])
            recorder.setdefault("calls", []).append(snapshot)
            return remaining.pop(0)

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


@pytest.fixture(autouse=True)
def sim_env(tmp_path, monkeypatch):
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


@pytest.fixture()
def tmp_settings(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", target)
    monkeypatch.setattr(core, "SETTINGS_PATH", target)
    return target


def _other_floor() -> float:
    saved = core._load_settings().min_retention_floor
    return 0.85 if abs(saved - 0.85) > 1e-9 else 0.80


def _assert_priced(block: dict[str, Any]) -> None:
    assert set(block) == _MONEY_KEYS
    # net equals gross minus retention cost to the cent, the plan's own basis.
    assert abs((block["gross"] - block["retention_cost"]) - block["net"]) <= 0.01


# --- (a) the primitive: before/after, net identity, owned scope, writes nothing -
def test_primitive_before_after_identity_and_writes_nothing(tmp_settings: Path) -> None:
    output_csv = ROOT / "output" / "weekly_break_schedule.csv"
    csv_mtime = output_csv.stat().st_mtime_ns
    settings_bytes = tmp_settings.read_bytes()

    result = simulate.simulate_settings_change({"min_retention_floor": _other_floor()})
    assert result["status"] == "ok"
    assert result["channel"] == core._load_settings().operator_channel
    assert result["day"]
    _assert_priced(result["before"])
    _assert_priced(result["after"])
    # The deltas are exactly after minus before on every field.
    for key in ("gross", "retention_cost", "net"):
        assert result["delta"][key] == round(result["after"][key] - result["before"][key], 2)
    assert result["delta"]["breaks"] == result["after"]["breaks"] - result["before"]["breaks"]
    # Raising the retention floor moves a real plan: the sides genuinely differ.
    assert result["before"] != result["after"]

    # Side-effect free: neither the committed plan CSV nor the settings changed.
    assert output_csv.stat().st_mtime_ns == csv_mtime
    assert tmp_settings.read_bytes() == settings_bytes


def test_primitive_is_honest_when_no_owned_channel(tmp_path, monkeypatch) -> None:
    saved = json.loads((ROOT / "data" / "kairos_settings.json").read_text(encoding="utf-8"))
    saved["operator_channel"] = ""
    target = tmp_path / "kairos_settings.json"
    target.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_PATH", target)

    result = simulate.simulate_settings_change({"revenue_weight": 70})
    assert result["status"] == "unavailable"
    assert "channel" in result["reason"]


# --- (b) the read tool: executes in-loop, carries before/after + source ---------
def test_simulate_read_tool_executes_in_loop_with_source(
    client: TestClient, tmp_settings: Path, monkeypatch
) -> None:
    floor = _other_floor()
    recorder: dict[str, Any] = {}
    turns = [
        model_turn(tool_use("simulate_settings_change", {"changes": {"min_retention_floor": floor}}, "tu_1")),
        model_turn(text_block("this is a simulation, nothing saved"), stop="end_turn"),
    ]
    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(recorder, turns))

    body = client.post("/api/assistant/ask", json={"question": f"what if the floor were {floor}?"}).json()
    assert body["error"] is None and body["answer"]

    # The model saw a real before/after with the provenance source attached.
    result = json.loads(recorder["calls"][1]["messages"][-1]["content"][0]["content"])
    assert result["status"] == "ok"
    _assert_priced(result["before"])
    _assert_priced(result["after"])
    assert result["source"] == SIM_SOURCE
    # The response's tool_trace surfaces that source for the figure.
    assert body["tool_trace"] == [{"tool": "simulate_settings_change", "ok": True, "source": SIM_SOURCE}]
    # A read-only what-if captures no proposal.
    assert body["proposals"] is None


def test_simulate_read_tool_forbidden_field_is_honest_not_crash() -> None:
    forbidden = tools.execute_read_tool("simulate_settings_change", {"changes": {"operator_channel": "x"}})
    assert forbidden.get("source") == SIM_SOURCE
    assert forbidden["status"] == "unavailable" and "not allowed" in forbidden["reason"]
    empty = tools.execute_read_tool("simulate_settings_change", {"changes": {}})
    assert empty["status"] == "unavailable" and empty.get("source")


# --- (c) the proposal effect: settings items carry before/after/delta -----------
def test_settings_proposal_item_carries_effect(tmp_settings: Path) -> None:
    item = tools.build_proposal_item(
        "propose_settings_change", {"changes": {"min_retention_floor": _other_floor()}, "reason": "r"}
    )
    assert item["status"] == "pending"
    effect = item["effect"]
    assert set(effect) == {"before", "after", "delta"}
    _assert_priced(effect["before"])
    _assert_priced(effect["after"])
    assert effect["delta"]["net"] == round(effect["after"]["net"] - effect["before"]["net"], 2)


def test_settings_effect_unavailable_is_honest(monkeypatch) -> None:
    monkeypatch.setattr("kairos_api.dashboard_api._owned_scope", lambda settings: (None, None))
    effect = simulate.settings_effect({"revenue_weight": 70})
    assert effect == {"status": "unavailable", "reason": effect["reason"]}
    assert "channel" in effect["reason"]


# --- (d) every read tool result carries a non-empty source ----------------------
def test_every_read_tool_result_carries_source(tmp_settings: Path) -> None:
    args_by_tool = {
        "get_day_detail": {"date": "2024-11-05"},
        "simulate_settings_change": {"changes": {"max_breaks_per_hour": 4}},
    }
    for name in sorted(tools.READ_TOOL_NAMES):
        result = tools.execute_read_tool(name, args_by_tool.get(name, {}))
        assert isinstance(result, dict), name
        assert result.get("source"), name


# --- (e) goal-seek: search with simulate, then one proposal with an effect -------
def test_goal_seek_simulates_then_proposes_one_effect_bearing_change(
    client: TestClient, tmp_settings: Path, monkeypatch
) -> None:
    before_bytes = tmp_settings.read_bytes()
    winner = _other_floor()
    recorder: dict[str, Any] = {}
    turns = [
        model_turn(tool_use("simulate_settings_change", {"changes": {"min_retention_floor": winner}}, "s1")),
        model_turn(tool_use("simulate_settings_change", {"changes": {"min_retention_floor": 0.90}}, "s2")),
        model_turn(tool_use("propose_settings_change",
                            {"changes": {"min_retention_floor": winner}, "reason": "meets the goal"}, "p1")),
        model_turn(text_block("proposing the winning floor for your approval"), stop="end_turn"),
    ]
    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(recorder, turns))

    body = client.post(
        "/api/assistant/ask", json={"question": "get me higher net without dropping retention"}
    ).json()

    # Exactly one pending proposal, and it carries the simulated effect.
    assert body["error"] is None
    items = body["proposals"]["items"]
    assert len(items) == 1 and items[0]["status"] == "pending" and items[0]["kind"] == "settings"
    effect = items[0]["effect"]
    assert set(effect) == {"before", "after", "delta"}
    _assert_priced(effect["before"])
    _assert_priced(effect["after"])

    # Nothing was applied along the way: the saved settings are untouched.
    assert tmp_settings.read_bytes() == before_bytes

    # The tool_trace carries the source on every simulate step; the proposal has none.
    trace = body["tool_trace"]
    assert [(step["tool"], step["ok"]) for step in trace] == [
        ("simulate_settings_change", True), ("simulate_settings_change", True),
        ("propose_settings_change", True)]
    assert trace[0]["source"] == SIM_SOURCE and trace[1]["source"] == SIM_SOURCE
    assert "source" not in trace[2]

    # Wiring: the opening call stays a plain answer; the search turns enable adaptive
    # thinking with a medium effort, drop temperature, and cache the tools+system prefix.
    opening = recorder["calls"][0]
    assert opening["temperature"] == assistant.ANSWER_TEMPERATURE
    # Every call (opening included) sends the cache-controlled system list, so
    # the stable tools+system prefix is written once and read across turns.
    assert isinstance(opening["system"], list)
    assert opening["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in opening
    for call in recorder["calls"][1:]:
        assert call["thinking"] == {"type": "adaptive"}
        assert call["output_config"] == {"effort": assistant.LOOP_EFFORT}
        assert "temperature" not in call
        assert isinstance(call["system"], list)
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


# --- (f) backward compatibility: the /ask response contract is unchanged ---------
def test_ask_response_shape_is_backward_compatible(
    client: TestClient, tmp_settings: Path, monkeypatch
) -> None:
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(
        assistant, "_client_factory",
        scripted_factory(recorder, [model_turn(text_block("plain answer"), stop="end_turn")]),
    )
    body = client.post("/api/assistant/ask", json={"question": "how does the week look?"}).json()
    # The streaming contract extended the ask body additively with model,
    # context_disclosure and truncated; everything pre-existing is unchanged.
    assert set(body) == {"available", "answer", "model", "grounding", "context_disclosure",
                         "truncated", "error", "proposals", "tool_trace"}
    assert body["available"] is True
    assert body["answer"] == "plain answer"
    assert body["tool_trace"] == []
    assert body["proposals"] is None
    assert set(body["grounding"]) == {"sources", "generated_at"}
    assert body["model"] == "claude-sonnet-4-6"
    assert body["context_disclosure"] == body["grounding"]
    assert body["truncated"] is False
