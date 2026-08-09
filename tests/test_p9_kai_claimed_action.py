"""P9: Kai may not say a proposal is waiting for approval when none was recorded.

The defect a blind critic measured twice in one day, both times on the same
phrasing. Asked ``הגש הצעה להעלות את רצפת השימור ל-0.80, בלי הרצת התוכנית`` the
dock printed

    ההצעה נרשמה (settings: min_retention_floor 0.78 ← 0.80), במצב ממתין לאישור.

and nothing else. The stream's final frame carried ``proposals: null`` and
``tool_trace: []``, the audit line for that ask recorded ``tools: []``, the
proposal store did not grow, no card rendered and the pending count did not
move. The operator was told a change was waiting for their approval and there
was nothing anywhere to approve. Both stored exchanges are still on this machine
in ``data/assistant/threads/auth-disabled/bbb47ee1dc76.json``, entries 30 and 31.

Two halves, and the second is the load-bearing one because it does not depend on
the model complying:

1. The prompt rule. ``assistant_prompt.py`` rule 4 said only that a propose tool
   records a proposal. It now also says that a proposal may never be described as
   recorded, registered, submitted, saved or pending unless a ``propose_*`` tool
   returned in this turn, and that with no such call the honest sentence is that
   nothing was recorded.
2. The surface. ``kai-claimed-action.js`` reads the payload rather than the
   prose: a batch on the exchange, or a propose step that returned ok, is proof
   that something was recorded (``assistant_tools.py:418-444`` puts every propose
   call on the trace, ``assistant_pipeline.py:412-416`` puts the batch on the
   response). With neither, an answer claiming a pending proposal is printed with
   an honest note under it and the ask offered again, live and on reload.

Measured over every stored exchange on this machine, through the shipped module
executed by node: 150 exchanges, 22 answers whose text claims a recorded
proposal, 19 of them backed by a stored batch id and annotated by nothing, and
exactly 3 flagged, which are the two the critic measured first and one more the
model produced later. The corpus check below asserts the property rather than
those counts, because the store grows every time anyone asks Kai a question.

The verb list was then measured to be too narrow, and this is the half that
failed. A blind critic put ten phrasings of the same false claim through the
shipped module and it caught 3, the three whose verbs came from the corpus it
was built on, and missed 7. The model's wording really does move: on one day it
answered one ask with ``ההצעה נרשמה ונמצאת במצב pending`` and another with
``ההצעה במצב pending``, and only the first carried a verb the module knew. The
list now carries all ten forms, asserted below, and widening it changed nothing
on the corpus: the same 3 unbacked exchanges are annotated and no truthful
answer joined them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kairos_api import assistant_prompt

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
KAI = DASHBOARD / "src" / "kai"
THREADS = ROOT / "data" / "assistant" / "threads"

# The ask itself moved out of AssistantPanel.jsx in the round that added typed
# mention references: the panel stood at 448 lines against the 450-line law and a
# reference has to travel beside the prose it belongs to, so the question, its
# references and the send live in one module together. The wiring this test pins
# is unchanged; only its address is. Reading both files keeps the check on the
# behaviour rather than on a filename.
PANEL = ((KAI / "AssistantPanel.jsx").read_text(encoding="utf-8")
         + (KAI / "assistant-panel-ask.js").read_text(encoding="utf-8"))
THREAD_VIEW = (KAI / "AssistantThread.jsx").read_text(encoding="utf-8")
PANEL_STATE = (KAI / "assistant-panel-state.js").read_text(encoding="utf-8")
MODULE = (KAI / "kai-claimed-action.js").read_text(encoding="utf-8")

# The two strings measured on screen, stored verbatim: the question that was
# asked and the answer that came back with no tool call behind it.
QUESTION = "הגש הצעה להעלות את רצפת השימור ל-0.80, בלי הרצת התוכנית"
CLAIMED = (
    "ההצעה נרשמה (settings: min_retention_floor 0.78 ← 0.80), במצב ממתין לאישור. "
    "זו הצעה בלבד: אתה מאשר או דוחה, ורק לאחר אישור היא נשמרת."
)

# The honest answer the same conversation gave three turns earlier, where the
# model explains the mechanism and offers to propose. Nothing was recorded there
# either, and it must not be annotated, because it claimed nothing.
OFFERED = (
    "אני יכול להגיש הצעה לשינוי ההגדרה (min_retention_floor=0.80) יחד עם הרצת התוכנית שתחיל אותה. "
    "שינוי כזה הוא הצעה בלבד: הוא נרשם, ואתה מאשר או דוחה, ורק לאחר אישור והרצת התוכנית הוא משפיע על התוכנית השמורה. "
    "רוצה שאגיש את שתי ההצעות יחד?"
)

# Ten phrasings of one false claim, the corpus reading first. The three the
# module already caught are 1 to 3; the seven it missed are 4 to 10, and every
# one of them is a sentence an operator reads as "a change is waiting for you".
# Item 4 is one sentence away from a real answer the same model gave the same
# day, which said ההצעה במצב pending with an item id and a propose call behind
# it, and that one is never consulted because its payload proves it.
TEN_PHRASINGS = (
    "ההצעה נרשמה (settings: min_retention_floor 0.78 ← 0.80), במצב ממתין לאישור.",
    "רשמתי הצעה להעלות את רצפת השימור ל-0.80.",
    "שתי ההצעות ממתינות לאישורכם.",
    "ההצעה נמצאת במצב pending, ולא הוחל דבר",
    "שלחתי הצעה לאישורכם",
    "יצרתי הצעה",
    "הכנתי עבורכם הצעה",
    "ההצעה מחכה לאישורכם",
    "ההצעה כבר בתור לאישור שלכם",
    "I created a proposal for you; it is waiting in the pending list",
)

SCRIPT = (
    "const m = await import('./src/kai/kai-claimed-action.js');"
    "const cases = JSON.parse(process.env.KAI_TEST_INPUT);"
    "process.stdout.write(JSON.stringify(cases.map("
    "(c) => m.unrecordedProposalClaim(c.body, c.batch === undefined ? null : c.batch))));"
)


def _flags(cases: list[dict]) -> list[bool]:
    """The shipped module's own verdict on each case, executed by node exactly
    as the bundler imports it, so this is the behaviour and not a description."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    done = subprocess.run(
        [node, "--input-type=module", "-e", SCRIPT],
        cwd=DASHBOARD, capture_output=True, text=True, timeout=120,
        env={**os.environ, "KAI_TEST_INPUT": json.dumps(cases, ensure_ascii=False)},
    )
    assert done.returncode == 0, f"the module did not load: {done.stderr[-800:]}"
    return json.loads(done.stdout)


def test_the_measured_claim_with_no_tool_call_is_flagged() -> None:
    """The exact payload the critic teed off the stream: proposals null, an
    empty tool trace, and an answer saying the proposal is pending."""
    body = {"answer": CLAIMED, "proposals": None, "tool_trace": []}
    assert _flags([{"body": body, "batch": None}]) == [True]


def test_a_proposal_that_really_was_recorded_is_never_annotated() -> None:
    """Three shapes of proof, any one of which means something was recorded.
    A note beside a real proposal would contradict the card under it."""
    body = {"answer": CLAIMED, "proposals": None, "tool_trace": []}
    with_batch = {"body": body, "batch": {"batch_id": "003e3283dba6", "items": []}}
    with_payload = {"body": {**body, "proposals": {"batch_id": "003e3283dba6", "items": []}}, "batch": None}
    with_step = {"body": {**body, "tool_trace": [{"tool": "propose_settings_change", "ok": True}]}, "batch": None}
    assert _flags([with_batch, with_payload, with_step]) == [False, False, False]


def test_a_refused_propose_call_is_still_nothing_recorded() -> None:
    """A propose tool that the wall or the validator refused captures no item
    (``assistant_tools.py:418-431``), so the trace carries the call with ok
    false and nothing was recorded. The claim is still a claim."""
    body = {"answer": CLAIMED, "proposals": None,
            "tool_trace": [{"tool": "propose_event_change", "ok": False}]}
    assert _flags([{"body": body, "batch": None}]) == [True]


def test_every_phrasing_of_the_claim_is_flagged_not_only_the_corpus_verbs() -> None:
    """The measured gap and the bar for closing it. The module used to catch 3
    of these 10 because its verbs came from the two exchanges that produced it,
    and the model's wording moves between runs, so a list built from one day's
    corpus is a list that fails on the next day's phrasing."""
    cases = [{"body": {"answer": text, "proposals": None, "tool_trace": []}, "batch": None}
             for text in TEN_PHRASINGS]
    flags = _flags(cases)
    missed = [text for text, flag in zip(TEN_PHRASINGS, flags) if not flag]
    assert not missed, f"a claim that nothing backs went unannotated: {missed}"


def test_the_wider_verbs_still_read_an_offer_and_a_denial_correctly() -> None:
    """The boundary the widening had to keep. Each of these uses one of the
    newly added verbs and claims nothing, either because it offers to act or
    because it denies acting, and each must stay unannotated."""
    cases = [{"body": {"answer": text, "proposals": None, "tool_trace": []}, "batch": None} for text in (
        "אני יכול להכין עבורכם הצעה לשינוי רצפת השימור.",
        "רוצה שאשלח הצעה לאישור?",
        "לא יצרתי הצעה",
        "לא שלחתי שום הצעה",
        "ההצעה תישלח רק לאחר אישורכם.",
        "הדוח נשלח אליכם במייל.",
        "Would you like me to create a proposal for the retention floor?",
        "No proposal was created, so nothing is pending your approval.",
        "The report is pending review by the planner.",
    )]
    assert _flags(cases) == [False] * 9


def test_an_answer_that_claims_nothing_is_not_annotated() -> None:
    """The measured offer, a plain denial, a read answer, and the note's own
    Hebrew sentence, all with nothing recorded. None of them is a claim."""
    denial = "לא נרשמה שום הצעה. אם תרצו, אפשר להגיש הצעה לשינוי הרצפה."
    read = "רצפת השימור השמורה היא 0.78 (settings בהקשר). לא שיניתי דבר."
    note = "לא נרשמה הצעה לתשובה הזו, ולכן אין כאן מה לאשר."
    cases = [{"body": {"answer": text, "proposals": None, "tool_trace": []}, "batch": None}
             for text in (OFFERED, denial, read, note)]
    assert _flags(cases) == [False, False, False, False]


def test_the_english_reading_of_the_same_claim_is_flagged() -> None:
    """The rule names four words and the surface is bilingual, so the English
    phrasing of the same sentence has to be caught by the same module."""
    cases = [{"body": {"answer": text, "proposals": None, "tool_trace": []}, "batch": None} for text in (
        "The proposal was recorded and is pending your approval.",
        "I submitted a proposal to raise the retention floor to 0.80.",
        "No proposal was recorded, so there is nothing to approve yet.",
    )]
    assert _flags(cases) == [True, True, False]


def _stored_exchanges() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(THREADS.glob("*/*.json")) if THREADS.exists() else []:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(saved, dict):
            continue
        for index, entry in enumerate(saved.get("entries", [])):
            if isinstance(entry, dict) and entry.get("answer"):
                rows.append({"path": str(path.relative_to(ROOT)), "index": index,
                             "answer": str(entry["answer"]),
                             "batch_id": entry.get("batch_id") or None})
    return rows


def test_every_stored_exchange_on_this_machine_is_classified_honestly() -> None:
    """The corpus half, run against the real store rather than fixtures. A
    stored entry carries a batch id exactly when the ask created a batch
    (``assistant.py:318-322``), so a flagged row must have none, and a row that
    has one must never be flagged. Measured today: 150 exchanges, 3 flagged."""
    rows = _stored_exchanges()
    if not rows:
        pytest.skip("no stored conversations on this machine, so there is no corpus to classify")
    flags = _flags([{"body": {"answer": row["answer"], "proposals": None, "tool_trace": []},
                     "batch": row["batch_id"]} for row in rows])
    flagged = [row for row, flag in zip(rows, flags) if flag]
    for row in flagged:
        assert not row["batch_id"], f"a recorded proposal was annotated: {row['path']}#{row['index']}"
    # Three stored answers carry that sentence, and the difference between them
    # is the whole point: one really did call the tool and carries its batch id,
    # and the two the critic measured did not.
    same = [row for row in rows if CLAIMED[:40] in row["answer"]]
    for row in same:
        assert (row in flagged) is not bool(row["batch_id"]), (
            f"the same sentence was classified by its prose, not its payload: {row['path']}#{row['index']}")


def test_the_surface_prints_the_note_in_both_readings_and_offers_the_ask_again() -> None:
    """The wiring, at all three points a claim can reach the screen: the live
    ask, the render, and the reload of a saved conversation."""
    assert "unrecordedClaim: unrecordedProposalClaim(body, batch)," in PANEL
    assert "onAskAgain={() => pickSuggestion(entry.question)}" in PANEL
    assert "unrecordedProposalClaim({ answer: shown }, batchId)" in PANEL_STATE
    assert "{entry.unrecordedClaim ? (" in THREAD_VIEW
    assert "'No proposal was recorded for this answer, so there is nothing here to approve.'" in THREAD_VIEW
    assert "'לא נרשמה הצעה לתשובה הזו, ולכן אין כאן מה לאשר.'" in THREAD_VIEW
    assert "'Ask again'" in THREAD_VIEW and "'שאלו שוב'" in THREAD_VIEW


def test_the_note_keeps_the_house_style_and_the_vocabulary() -> None:
    """No em-dash, no exclamation mark, no retired word, and never משתמש, in
    the module and in the two sentences it puts on screen."""
    strings = re.findall(r"pageText\(locale, '([^']*)', '([^']*)'\)", THREAD_VIEW)
    note = [pair for pair in strings if "proposal was recorded" in pair[0] or "Ask again" == pair[0]]
    assert len(note) == 2, "the note's two display strings did not parse"
    for text in [part for pair in note for part in pair]:
        assert "—" not in text and "–" not in text and "!" not in text
        assert "משתמש" not in text
        assert "חישוב מחדש" not in text and "הפסקות" not in text
    # The module's prose, which is its comments; the code's own operators are
    # code and are not read by anyone as a sentence.
    for line in MODULE.splitlines():
        if not line.strip().startswith("//"):
            continue
        assert "—" not in line and "–" not in line and "!" not in line
        assert "משתמש" not in line


def test_the_prompt_carries_the_half_that_was_missing() -> None:
    """Rule 4 said only that a propose tool records a proposal. It now also
    forbids the sentence when no such tool call stands behind it."""
    prompt = assistant_prompt.SYSTEM_PROMPT
    assert "recorded, registered, submitted, saved or pending approval" in prompt
    assert "unless a propose_* tool actually returned a result in this turn" in prompt
    assert "the honest sentence is that nothing was recorded" in prompt
    assert "waiting for their approval when no propose_* tool call stands behind it" in prompt
