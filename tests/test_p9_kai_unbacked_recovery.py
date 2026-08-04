"""P9: a claim that nothing backs buys one more turn, and never stands as the answer.

The measured defect, round 5, by a blind critic in the browser. First ask of the
session, plain Hebrew ``העלה את רצפת השימור ל-82 אחוז``. The answer opened

    רשמתי שתי הצעות שממתינות לאישורך

and listed both. ``GET /api/assistant/audit`` for that turn recorded
``tools: []``, no batch was created and the proposal store did not grow: 25.4 s
spent and nothing to approve. Reproduced deliberately on the wire against a
conversation that already carried a proposal for the same field, where the same
sentence came back with a fabricated provenance. Rate that session: 2 of 9
action asks, 1 of 3 inside such a conversation against 0 of 4 in fresh ones.

Both stored answers are still on this machine, in
``data/assistant/threads/auth-disabled/697dabafa588.json`` entries 2 and 6, and
entry 6 is the shape that made the correction useless: the claim first, the
truth in the last paragraph, so the operator reads two contradictory statements
and has to trust the smaller one.

Two halves, and this file proves both.

1. ``kairos_api/assistant_claimed_action.py`` is the browser rule ported to the
   server, verb for verb, and when it fires with no successful propose step the
   loop spends ONE more turn telling the model, in the tool-result channel, that
   its claim is unbacked and it must either call the propose tool or restate.
   That is exactly the recovery the critic performed by hand, which worked first
   time.
2. ``AssistantThread.jsx`` prints the honest sentence INSTEAD of the claim, which
   is the other half and is proved next door, in
   ``test_p9_kai_retracted_claim.py``.

The two rules must agree, or the surface annotates answers the server let
through and the server corrects answers the surface would have accepted. The
first test runs the shipped Python and the shipped JavaScript over the same
corpus, including every stored exchange on this machine, and compares verdicts.

The Claude call is mocked at the module seam; no key and no live server is used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_claimed_action as claimed
import kairos_api.assistant_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
KAI = DASHBOARD / "src" / "kai"
THREADS = ROOT / "data" / "assistant" / "threads"
# The two answers measured this round, verbatim from the stored thread.
CLAIM_ONLY = "רשמתי שתי הצעות שממתינות לאישורך: שינוי ההגדרה והרצת התוכנית."
CLAIM_THEN_CORRECTION = (
    "רשמתי שתי הצעות שממתינות לאישורך.\n"
    "הבהרה חשובה: לא באמת רשמתי דבר עדיין. בהודעה זו לא הפעלתי כלי הצעה, ולכן אין הצעה שממתינה לאישורך."
)
HONEST = (
    "לא נרשמה הצעה בתשובה הזו ואין מה לאשר. "
    "השינוי שאני מציע הוא min_retention_floor מ-0.82 ל-0.84. רוצה שארשום אותו?"
)
RECORDED = "רשמתי הצעה אחת שממתינה לאישורך: min_retention_floor מ-0.82 ל-0.84."


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Any:
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


def _text(body: str, stop: str = "end_turn") -> Any:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=body)], stop_reason=stop)


def _propose_turn(preamble: str = "") -> Any:
    """A turn that calls the propose tool, optionally after one line of text."""
    blocks: list[Any] = []
    if preamble:
        blocks.append(SimpleNamespace(type="text", text=preamble))
    blocks.append(SimpleNamespace(
        type="tool_use", name="propose_settings_change", id="tu_fix",
        input={"changes": {"min_retention_floor": 0.84},
               "reason": "the operator asked for a retention floor of 84 percent"},
    ))
    return SimpleNamespace(content=blocks, stop_reason="tool_use")


def _scripted(calls: list[dict[str, Any]], turns: list[Any]):
    """A model that answers a fixed script, one entry per call, last repeating."""

    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return turns[min(len(calls) - 1, len(turns) - 1)]

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


# --- the two rules are one rule -----------------------------------------------

JS = ("const m = await import('./src/kai/kai-claimed-action.js');"
      "const cases = JSON.parse(process.env.KAI_TEST_INPUT);"
      "process.stdout.write(JSON.stringify(cases.map((c) => m.claimsRecordedProposal(c))));")


def _js_verdicts(texts: list[str]) -> list[bool]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the shipped browser module cannot be executed")
    done = subprocess.run(
        [node, "--input-type=module", "-e", JS], cwd=DASHBOARD, capture_output=True,
        text=True, timeout=120, env={**os.environ, "KAI_TEST_INPUT": json.dumps(texts, ensure_ascii=False)},
    )
    assert done.returncode == 0, f"the browser module did not load: {done.stderr[-800:]}"
    return json.loads(done.stdout)


def _corpus() -> list[str]:
    """Every phrasing this campaign has measured, plus every answer stored on
    this machine, so the comparison is against real prose and not fixtures."""
    texts = [
        CLAIM_ONLY, CLAIM_THEN_CORRECTION, HONEST, RECORDED,
        "ההצעה נרשמה (settings: min_retention_floor 0.78 ← 0.80), במצב ממתין לאישור.",
        "רשמתי הצעה להעלות את רצפת השימור ל-0.80.",
        "שתי ההצעות ממתינות לאישורכם.",
        "ההצעה נמצאת במצב pending, ולא הוחל דבר",
        "שלחתי הצעה לאישורכם", "יצרתי הצעה", "הכנתי עבורכם הצעה",
        "ההצעה מחכה לאישורכם", "ההצעה כבר בתור לאישור שלכם",
        "I created a proposal for you; it is waiting in the pending list",
        "אני יכול להכין עבורכם הצעה לשינוי רצפת השימור.",
        "רוצה שאשלח הצעה לאישור?", "לא יצרתי הצעה", "לא שלחתי שום הצעה",
        "ההצעה תישלח רק לאחר אישורכם.", "הדוח נשלח אליכם במייל.",
        "Would you like me to create a proposal for the retention floor?",
        "No proposal was created, so nothing is pending your approval.",
        "The report is pending review by the planner.",
        "The proposal was recorded and is pending your approval.",
        "I submitted a proposal to raise the retention floor to 0.80.",
        "No proposal was recorded, so there is nothing to approve yet.",
        'רצפת השימור השמורה היא 0.78 (settings בהקשר). לא שיניתי דבר.',
    ]
    for path in sorted(THREADS.glob("*/*.json")) if THREADS.exists() else []:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(saved, dict):
            texts.extend(str(entry["answer"]) for entry in saved.get("entries", [])
                         if isinstance(entry, dict) and entry.get("answer"))
    return texts


def test_the_server_rule_and_the_browser_rule_agree_on_every_measured_answer() -> None:
    """The port is a port. Two implementations of one rule that disagree are two
    rules, and the operator would meet whichever one the code path reached."""
    texts = _corpus()
    ours = [claimed.claims_recorded_proposal(text) for text in texts]
    theirs = _js_verdicts(texts)
    disagreed = [text[:70] for text, a, b in zip(texts, ours, theirs) if a != b]
    assert not disagreed, f"the two copies of the rule classified differently: {disagreed}"
    # A comparison where both sides said false about everything would pass and
    # prove nothing, so the corpus is asserted to contain both verdicts.
    assert any(ours) and not all(ours), "the corpus must contain claims and honest answers"


def test_the_claim_with_a_correction_under_it_is_still_a_claim() -> None:
    """The shape measured in entry 6: the false sentence first, the truth last.
    The first sentence claims, so the answer claims, in both copies."""
    assert claimed.claims_recorded_proposal(CLAIM_THEN_CORRECTION) is True
    assert _js_verdicts([CLAIM_THEN_CORRECTION]) == [True]


def test_a_successful_propose_step_is_proof_and_a_refused_one_is_not() -> None:
    ok = [{"tool": "propose_settings_change", "ok": True}]
    refused = [{"tool": "propose_event_change", "ok": False}]
    assert claimed.unbacked_claim(RECORDED, ok, []) is False
    assert claimed.unbacked_claim(RECORDED, [], [{"id": "x"}]) is False
    assert claimed.unbacked_claim(RECORDED, refused, []) is True
    assert claimed.unbacked_claim(RECORDED, [], []) is True


# --- the recovery turn --------------------------------------------------------

def test_an_unbacked_claim_buys_one_turn_that_records_the_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured failure, end to end. Turn 1 claims two proposals and calls
    nothing. The correction turn calls the propose tool, the item is captured,
    and the answer the person reads is the one written after the tool result."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [
        _text(CLAIM_ONLY), _propose_turn(), _text(RECORDED),
    ]))
    trace: list[dict] = []
    items: list[dict] = []
    outcome: dict[str, Any] = {}
    answer, stopped = pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהעלה את רצפת השימור ל-84 אחוז",
        trace, items, actions_on=True, outcome=outcome,
    )
    assert stopped is False
    assert outcome["claim_recovery"] == "proposed"
    # One opening turn, one correction turn, one tool-free turn for the sentence.
    assert len(calls) == 3
    assert [step["tool"] for step in trace] == ["propose_settings_change"]
    assert len(items) == 1 and items[0]["status"] == "pending"
    assert items[0]["payload"]["changes"] == {"min_retention_floor": 0.84}
    assert answer == RECORDED
    assert claimed.unbacked_claim(answer, trace, items) is False


def test_the_correction_reaches_the_model_in_its_own_channel_and_names_the_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is sent as a system verification rather than in the person's voice, it
    states the payload fact, and it forbids the shape entry 6 produced: the
    claim left standing with a correction under it."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [_text(CLAIM_ONLY), _text(HONEST)]))
    pipeline.run_tool_loop(assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהעלה", [], [],
                           actions_on=True, outcome={})
    correction = calls[-1]["messages"][-1]
    assert correction["role"] == "user"
    assert correction["content"] == claimed.CORRECTION_WITH_TOOLS
    assert correction["content"].startswith("SYSTEM VERIFICATION, not the person speaking.")
    assert "no successful propose_ call" in correction["content"]
    assert "do not leave it standing with a correction under it" in correction["content"]
    # The turn before it is the claim itself, echoed back, so the model is
    # correcting its own words rather than being told about them.
    assert calls[-1]["messages"][-2]["role"] == "assistant"
    assert calls[-1]["messages"][-2]["content"] == [{"type": "text", "text": CLAIM_ONLY}]
    # The propose tools are on the correction turn, or it could not record.
    assert any(tool["name"] == "propose_settings_change" for tool in calls[-1]["tools"])


def test_a_restatement_replaces_the_claim_as_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other acceptable outcome. The model does not propose, it says plainly
    that nothing was recorded, and that sentence is what the person reads."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [_text(CLAIM_ONLY), _text(HONEST)]))
    outcome: dict[str, Any] = {}
    answer, _stopped = pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהעלה", [], [],
        actions_on=True, outcome=outcome,
    )
    assert len(calls) == 2, "exactly one extra turn, never a loop of corrections"
    assert answer == HONEST
    assert outcome["claim_recovery"] == "restated"


def test_a_correction_that_claims_again_is_named_and_never_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model can refuse to be corrected. The run says so in its outcome, the
    answer is still the claim, and the surface's guard is what the operator
    meets, which is why that guard stays."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [_text(CLAIM_ONLY), _text(RECORDED)]))
    outcome: dict[str, Any] = {}
    answer, _stopped = pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהעלה", [], [],
        actions_on=True, outcome=outcome,
    )
    assert len(calls) == 2
    assert outcome["claim_recovery"] == "unfixed"
    assert claimed.unbacked_claim(answer, [], []) is True


def test_a_truthful_answer_buys_no_extra_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cost of this mechanism on the ordinary path is exactly zero calls."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [
        _text('ההכנסה הצפויה השבוע היא 9,960,616.55 ש"ח (overview_summary.week).'),
    ]))
    outcome: dict[str, Any] = {}
    pipeline.run_tool_loop(assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nמה מצב השבוע",
                           [], [], actions_on=True, outcome=outcome)
    assert len(calls) == 1
    assert "claim_recovery" not in outcome


def test_a_claim_that_a_propose_call_backs_buys_no_extra_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of reading the payload rather than the prose: the same
    sentence, with a tool call behind it, is true and is left alone."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [_propose_turn(), _text(RECORDED)]))
    trace: list[dict] = []
    items: list[dict] = []
    outcome: dict[str, Any] = {}
    answer, _stopped = pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהעלה", trace, items,
        actions_on=True, outcome=outcome,
    )
    assert len(calls) == 2, "the tool turn and the answer turn, and nothing more"
    assert answer == RECORDED and len(items) == 1
    assert "claim_recovery" not in outcome


def test_the_clock_still_ends_the_run_before_the_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deadline that has passed stops the recovery like it stops everything
    else. The answer comes back as it was and the surface annotates it."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [_text(CLAIM_ONLY)]))
    outcome: dict[str, Any] = {}
    answer, _stopped = pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהעלה", [], [],
        actions_on=True, deadline=0.0, outcome=outcome,
    )
    assert len(calls) == 1
    assert answer == CLAIM_ONLY
    assert outcome["claim_recovery"] == "skipped_deadline"


def test_a_failed_correction_returns_the_answer_it_was_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never worse than before. A correction that raises is named in the outcome
    and the original answer survives for the surface to annotate."""
    calls: list[dict[str, Any]] = []

    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if len(calls) > 1:
                raise RuntimeError("connection reset")
            return _text(CLAIM_ONLY)

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(assistant, "_client_factory", factory)
    outcome: dict[str, Any] = {}
    answer, _stopped = pipeline.run_tool_loop(
        assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהעלה", [], [],
        actions_on=True, outcome=outcome,
    )
    assert answer == CLAIM_ONLY
    assert outcome["claim_recovery"] == "failed"
    assert "connection reset" in outcome["claim_recovery_error"]


def test_an_account_that_may_not_propose_is_told_to_restate_not_to_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A viewer cannot record anything, so the correction offers one outcome and
    carries no propose tool at all."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [_text(CLAIM_ONLY), _text(HONEST)]))
    pipeline.run_tool_loop(assistant._client_factory("test-key"), "CONTEXT:\n{}\n\nQUESTION:\nהעלה", [], [],
                           actions_on=True, can_propose=False, outcome={})
    assert calls[-1]["messages"][-1]["content"] == claimed.CORRECTION_NO_TOOLS
    assert "cannot propose changes here" in claimed.CORRECTION_NO_TOOLS
    assert not any(tool["name"].startswith("propose") for tool in calls[-1].get("tools", []))


# --- the whole ask: the batch, the trace, the audit ---------------------------

def test_the_recovered_proposal_reaches_the_body_the_batch_and_the_audit(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """The critic's own check. That turn's audit line recorded ``tools: []`` and
    the store did not grow; now the ask returns a batch, the trace carries the
    propose step, and the audit line names it."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [
        _text(CLAIM_ONLY), _propose_turn(), _text(RECORDED),
    ]))
    body = client.post("/api/assistant/ask", json={"question": "העלה את רצפת השימור ל-84 אחוז"}).json()
    assert body["available"] is True
    assert body["answer"] == RECORDED
    assert body["error"] is None
    assert [step["tool"] for step in body["tool_trace"]] == ["propose_settings_change"]
    assert body["proposals"] and body["proposals"]["status"] == "pending"
    assert len(body["proposals"]["items"]) == 1
    # The frozen key set did not move.
    assert set(body) == {"available", "answer", "model", "grounding", "context_disclosure",
                         "truncated", "error", "proposals", "tool_trace", "conversation_id"}
    stored = client.get("/api/assistant/proposals").json()
    batches = stored.get("batches") if isinstance(stored, dict) else stored
    assert any(batch["batch_id"] == body["proposals"]["batch_id"] for batch in batches)
    audit = client.get("/api/assistant/audit").json()
    latest = [row for row in audit["entries"] if row.get("event") == "ask"][0]
    assert latest["results"]["tools"] == ["propose_settings_change"]
    assert latest["batch_id"] == body["proposals"]["batch_id"]


def test_the_stream_announces_the_check_as_its_own_stage(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """The dock cannot silently swap the answer under the operator. The extra
    turn is a named stage on P9's own channel, which is what lets the surface
    drop the false text it already painted and say what is happening."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assistant, "_client_factory", _scripted(calls, [_text(CLAIM_ONLY), _text(HONEST)]))
    with client.stream("POST", "/api/assistant/ask/stream",
                       json={"question": "העלה את רצפת השימור ל-84 אחוז"}) as response:
        frames = "".join(chunk for chunk in response.iter_text())
    stages = [json.loads(line[5:]) for line in frames.splitlines()
              if line.startswith("data: ") and '"stage"' in line]
    names = [stage["stage"] for stage in stages]
    assert "verifying" in names, names
    assert names.index("verifying") > names.index("grounded")
    verifying = stages[names.index("verifying")]
    assert verifying["note"] == claimed.VERIFYING_NOTE
