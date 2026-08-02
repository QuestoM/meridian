"""P9: the goal-seek ask ends in an answer, never in a tool call written as text.

Measured on 2026-08-01, twice, on Kai's own headline flow and on two of the six
suggestion chips it ships: ``POST /api/assistant/ask/stream`` with
"הצע שינוי שיגדיל את הנטו באחוז אחד" returned after 71.6 s with proposals null,
error null, and an answer whose whole body was
``call\\n<invoke name="simulate_settings_change">...``, rendered verbatim in the
dock. Both leaked runs left the tool loop by exhausting its 12-turn ceiling
while the model still wanted tools, so the ask returned the text of a turn that
was protocol rather than an answer.

The two leaked bodies below are quoted from the thread the run wrote,
``data/assistant/threads/auth-disabled/86e5bff8c4cb.json``, so these assert
against what actually happened rather than against a description of it. The
Claude call is mocked at the module seam; no key and no live server are used.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_history as history
import kairos_api.assistant_pipeline as pipeline
import kairos_api.assistant_prompt as prompt
import kairos_api.assistant_protocol_text as protocol_text
import kairos_api.assistant_tools as assistant_tools

# Verbatim from the leaked thread: the whole answer was three calls.
LEAKED_WHOLE = (
    'call\n<invoke name="simulate_settings_change">\n'
    '<parameter name="changes">{"min_break_spacing_minutes": 5}</parameter>\n</invoke>\n'
    '<invoke name="simulate_settings_change">\n'
    '<parameter name="changes">{"min_break_spacing_minutes": 6}</parameter>\n</invoke>'
)
# Verbatim shape of the other leak: a real Hebrew summary, then a call.
LEAKED_TAIL = (
    "סיכום החיפוש: צמצום המרווח המינימלי בין ברייקים ל-2 דקות מעלה את הנטו ב-80,300 ש\"ח.\n\n"
    "הנה שתי ההצעות, יחד:\n\ncall\n"
    '<invoke name="propose_settings_change">\n'
    '<parameter name="changes">{"min_break_spacing_minutes": 2}</parameter>'
)

CARD = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src" / "kai" / "AssistantProposalCard.jsx"


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("KAIROS_ASSISTANT_DATA_DIR", str(tmp_path / "assistant"))
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


def _tool_use(index: int) -> Any:
    return SimpleNamespace(type="tool_use", name="get_settings", input={}, id=f"tu_{index}")


def _leaking_factory(calls: list[dict[str, Any]], recovery: str):
    """A model that keeps asking for tools and writes its calls as text.

    Exactly the measured shape: every turn carries protocol text beside a real
    tool_use, so the loop runs to its ceiling and the last turn's text is a
    call. The final call, the one with no tools on it, answers ``recovery``.
    """

    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if "tools" not in kwargs:
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=recovery)],
                                       stop_reason="end_turn")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=LEAKED_WHOLE), _tool_use(len(calls))],
                stop_reason="tool_use",
            )

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


# --- what protocol text is, measured against the two bodies that leaked -------
def test_the_two_leaked_bodies_are_recognised_and_cut() -> None:
    assert protocol_text.looks_like_tool_protocol(LEAKED_WHOLE) is True
    assert protocol_text.strip_tool_protocol(LEAKED_WHOLE) == ""
    assert protocol_text.looks_like_tool_protocol(LEAKED_TAIL) is True
    kept = protocol_text.strip_tool_protocol(LEAKED_TAIL)
    assert kept.endswith("הנה שתי ההצעות, יחד:")
    assert "invoke" not in kept and "call" not in kept


def test_an_ordinary_answer_is_returned_untouched() -> None:
    answer = 'ההכנסה הצפויה היא 10,123,070.8 ש"ח (overview_summary.week).\nהשוואה: 5 < 6.'
    assert protocol_text.looks_like_tool_protocol(answer) is False
    assert protocol_text.strip_tool_protocol(answer) == answer


# --- the live stream: prose paints, a call never does -------------------------
def test_the_gate_streams_prose_and_stops_at_the_first_tag() -> None:
    seen: list[str] = []
    gate = protocol_text.LiveTextGate(seen.append)
    for chunk in ("סיכום החיפוש: ", "נטו +80,300.\n\n", "הנה שתי ההצעות:\n\n", "call\n", '<invoke name="x">'):
        gate.feed(chunk)
    gate.flush()
    painted = "".join(seen)
    assert "invoke" not in painted
    assert "call" not in painted
    assert painted.startswith("סיכום החיפוש: נטו +80,300.")
    assert gate.blocked is True


def test_the_gate_holds_nothing_back_from_a_clean_turn() -> None:
    seen: list[str] = []
    gate = protocol_text.LiveTextGate(seen.append)
    gate.feed("checking the settings")
    assert seen == ["checking the settings"], "a delta with no tag streams as it arrives"
    gate.feed("\nand the day")
    gate.flush()
    assert "".join(seen) == "checking the settings\nand the day"
    assert gate.blocked is False


# --- the loop: one final tool-free call, and never protocol as an answer ------
def test_a_leaked_call_is_replaced_by_one_tool_free_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _leaking_factory(calls, "הנטו השבועי הוא 36,799,560.06 ש\"ח."))
    trace: list[dict] = []
    outcome: dict[str, Any] = {}
    answer, stopped = pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהצע שינוי",
        trace, [], actions_on=True, outcome=outcome,
    )
    assert stopped is False
    assert outcome["ceiling"] is True and outcome["protocol_text"] is True
    assert outcome["turns"] == pipeline.MAX_TOOL_ITERATIONS == 12
    # Exactly one call beyond the ceiling, and it is the tool-free one.
    assert len(calls) == 13
    assert "tools" not in calls[-1]
    assert "thinking" not in calls[-1]
    assert any(pipeline.FINAL_TURN_INSTRUCTION == block.get("text") for block in calls[-1]["system"])
    assert answer == "הנטו השבועי הוא 36,799,560.06 ש\"ח."
    assert outcome["recovered"] is True


def test_a_retry_that_leaks_again_returns_no_answer_rather_than_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _leaking_factory(calls, LEAKED_WHOLE))
    outcome: dict[str, Any] = {}
    answer, _stopped = pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהצע שינוי",
        [], [], actions_on=True, outcome=outcome,
    )
    assert answer == ""
    assert outcome["recovered"] is False


def test_the_ceiling_note_names_the_turn_budget_the_code_enforces() -> None:
    assert str(pipeline.MAX_TOOL_ITERATIONS) in pipeline.CEILING_NOTE
    assert assistant.CEILING_NOTE == pipeline.CEILING_NOTE
    # The model is told the same number the loop enforces, so it can converge
    # inside the budget instead of being cut off at it.
    assert str(pipeline.MAX_TOOL_ITERATIONS) in prompt.SYSTEM_PROMPT


# --- the whole ask body: no protocol, and the limit is named ------------------
def test_the_ask_body_never_carries_protocol_and_names_the_limit(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _leaking_factory(calls, LEAKED_WHOLE))
    body = client.post("/api/assistant/ask", json={"question": "הצע שינוי שיגדיל את הנטו באחוז אחד"}).json()
    assert body["available"] is True
    assert body["answer"] is None, "an unusable turn is no answer, not a printed call"
    assert body["error"] == pipeline.CEILING_NOTE
    assert "12" in body["error"]
    assert "invoke" not in str(body["error"])


def test_a_recovered_answer_still_says_the_search_hit_its_limit(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _leaking_factory(calls, "המרווח המינימלי ל-2 דקות מעלה את הנטו ב-6.3%."))
    body = client.post("/api/assistant/ask", json={"question": "הצע שינוי שיגדיל את הנטו באחוז אחד"}).json()
    assert body["answer"] == "המרווח המינימלי ל-2 דקות מעלה את הנטו ב-6.3%."
    assert body["error"] == pipeline.CEILING_NOTE
    assert set(body) == {"available", "answer", "model", "grounding", "context_disclosure",
                         "truncated", "error", "proposals", "tool_trace", "conversation_id"}


# --- the configured model: one place, and the parameters the loop sends -------
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def test_every_turn_carries_the_one_configured_model_and_no_rejected_parameter(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model name resolves from one place and the request shape is fixed.

    Every turn names whatever ``_model_name()`` returns, so changing the model
    is a one-line change and never a search through the loop. And every turn
    omits the parameters the Opus-tier family rejects: sampling controls, and a
    thinking budget. A search turn sends adaptive thinking and nothing else.
    """
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _leaking_factory(calls, "הנטו הוא 36,799,560.06 ש\"ח."))
    pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהצע שינוי",
        [], [], actions_on=True, outcome={},
    )
    configured = assistant._model_name()
    assert configured == assistant.DEFAULT_MODEL
    assert calls, "the loop must have called the model at least once"
    for call in calls:
        assert call["model"] == configured
        for rejected in ("temperature", "top_p", "top_k", "budget_tokens"):
            assert rejected not in call, rejected
        assert call.get("thinking", {"type": "adaptive"}) == {"type": "adaptive"}
    searching = [call for call in calls if "thinking" in call]
    assert searching, "the loop must reach at least one search turn"
    assert all(call["output_config"] == {"effort": pipeline.LOOP_EFFORT} for call in searching)
    assert pipeline.LOOP_EFFORT in EFFORT_LEVELS


def test_the_configured_model_is_served_and_accepts_what_the_loop_sends() -> None:
    """The configured default, checked against the live catalogue.

    This is the check that settles a model-name question in one request, and it
    exists because the name has been wrongly corrected from memory before: a
    published model table is a snapshot, and the catalogue is the fact. It
    skips rather than passing when it cannot read the catalogue, so a green run
    without credentials never reads as proof.

    Going live is opt-in on the flag ``conftest`` pins off, so the ordinary
    suite makes no network call and the outcome does not depend on which other
    file ran first. Run this one test with the flag on, and only this one:

        KAIROS_ASSISTANT_USE_CLAUDE_CODE_OAUTH=1 \\
          pytest tests/test_p9_kai_goal_seek.py -k configured_model_is_served

    Do not run the whole file that way. The two ask-route tests above mock the
    api-key seam only, so a resolvable OAuth credential sends them to the real
    model and they fail on a real answer rather than the scripted one.
    """
    import anthropic

    from kairos_api import assistant_auth

    enabled = os.environ.get(assistant_auth.OAUTH_FLAG_ENV, "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        pytest.skip(f"live catalogue check is opt-in: set {assistant_auth.OAUTH_FLAG_ENV}=1")
    auth = assistant._resolve_auth()
    if auth is None or auth.token == "test-key":
        pytest.skip("no real Anthropic credential resolved, so the catalogue cannot be read")
    client = assistant._client_from_auth(auth)
    try:
        model = client.models.retrieve(assistant.DEFAULT_MODEL)
    except anthropic.NotFoundError as exc:
        pytest.fail(f"DEFAULT_MODEL {assistant.DEFAULT_MODEL} is not served: {exc}")
    except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
        pytest.skip(f"catalogue unreachable, so this proves nothing either way: {type(exc).__name__}")

    capabilities = model.capabilities
    tree = capabilities.model_dump() if hasattr(capabilities, "model_dump") else dict(capabilities)
    assert tree["thinking"]["types"]["adaptive"]["supported"] is True
    assert tree["effort"][pipeline.LOOP_EFFORT]["supported"] is True


# --- history: a leak that did land never teaches the next turn ----------------
def test_a_stored_answer_that_leaked_is_not_replayed_to_the_model() -> None:
    assert history._replay_answer(LEAKED_WHOLE) == ""
    entries = [{"question": "הצע שינוי", "answer": LEAKED_WHOLE, "at": "t", "batch_id": None},
               {"question": "כמה ברייקים", "answer": "2,391 ברייקים.", "at": "t", "batch_id": None}]
    assert history._window(entries) == [("כמה ברייקים", "2,391 ברייקים.")]


# --- the card prints the kind the server sends, in both languages -------------
def test_every_proposal_kind_the_server_can_send_has_a_label_on_the_card() -> None:
    from kairos_api.assistant_propose_extra import _AGENCY_KIND_BY_ACTION

    source = CARD.read_text(encoding="utf-8")
    block = source.split("export const KINDS = {", 1)[1].split("};", 1)[0]
    labelled = set(re.findall(r"^\s{2}([a-z_]+):", block, re.MULTILINE))
    server_kinds = set(assistant_tools.KIND_BY_TOOL.values()) | set(_AGENCY_KIND_BY_ACTION.values())
    assert server_kinds <= labelled, f"unlabelled kinds print their raw key: {sorted(server_kinds - labelled)}"
    # And the measured before-and-after is claimed for exactly the kind that has
    # one, which is the kind the settings tool actually emits.
    assert assistant_tools.KIND_BY_TOOL["propose_settings_change"] == "settings"
    assert "MEASURED_EFFECT_KINDS = new Set(['settings'])" in source
