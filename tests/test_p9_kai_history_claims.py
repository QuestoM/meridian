"""P9: a false claim in the saved thread is never replayed back as the model's voice.

Where the measured rate came from. A blind critic measured the unbacked claim on
1 of 3 action asks inside a conversation that already carried a proposal for the
same field, against 0 of 4 in fresh conversations. That conversation is on this
machine: ``data/assistant/threads/auth-disabled/697dabafa588.json``, where seven
of the first eight answers open with ``רשמתי שתי הצעות שממתינות לאישורך`` and two
of them (entries 2 and 6) carry ``batch_id`` null. The pipeline replays the
newest six exchanges as ``assistant`` turns before the current question
(``kairos_api/assistant_history.py``), so the model was reading its own voice
saying a sentence that was false, and then it wrote it again.

``assistant_protocol_text`` already established the principle for the other
thing a stored answer can teach: a leaked tool call is cut before replay, or the
leak re-teaches itself for as long as the conversation lives. This is the same
rule applied to the same shape of harm, with the stored ``batch_id`` as the
proof rather than a reading of the prose.

The tri-state is the point and it is not decoration. A batch id is proof the
turn recorded something and the claim STAYS, because four of the five claims in
that slice are true and deleting a true sentence would corrupt the operator's
own history. An explicit null is proof it did not and only the claiming
sentences go. A stored shape with no such field at all is unknown and is left
exactly as written.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import kairos_api.assistant_claimed_action as claimed
import kairos_api.assistant_conversations as conversations
import kairos_api.assistant_history as history

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
MEASURED_THREAD = ROOT / "data" / "assistant" / "threads" / "auth-disabled" / "697dabafa588.json"

CLAIM_AND_FIGURE = (
    "רשמתי שתי הצעות שממתינות לאישורך: שינוי ההגדרה והרצת התוכנית.\n"
    "רצפת השימור השמורה היא 0.82 (מתוך settings). רוצה שאריץ את התוכנית?"
)


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Any:
    monkeypatch.setenv("KAIROS_ASSISTANT_DATA_DIR", str(tmp_path / "assistant"))
    yield


# --- the tri-state -------------------------------------------------------------

def test_an_answer_with_a_batch_behind_it_is_replayed_word_for_word() -> None:
    """Four of the five claims in the measured slice were TRUE. A rule that cut
    them would be rewriting the operator's own history to flatter itself."""
    assert history._replay_answer(CLAIM_AND_FIGURE, True) == CLAIM_AND_FIGURE


def test_an_answer_with_no_batch_loses_the_claim_and_keeps_the_figures() -> None:
    replayed = history._replay_answer(CLAIM_AND_FIGURE, False)
    assert "רשמתי שתי הצעות" not in replayed
    assert "רצפת השימור השמורה היא 0.82 (מתוך settings)" in replayed
    assert "רוצה שאריץ את התוכנית?" in replayed
    assert history.CLAIM_REMOVAL_MARKER in replayed
    assert claimed.claims_recorded_proposal(replayed) is False


def test_a_stored_shape_that_never_carried_the_field_is_left_alone() -> None:
    """Unknown is not false. Nothing is cut on a guess."""
    assert history._replay_answer(CLAIM_AND_FIGURE, None) == CLAIM_AND_FIGURE


def test_an_answer_that_was_nothing_but_the_claim_still_replays() -> None:
    """The exchange is not silently dropped: an empty answer would break the
    strict alternation the API requires and the question would vanish with it."""
    replayed = history._replay_answer("רשמתי הצעה שממתינה לאישורך.", False)
    assert replayed == history.CLAIM_REMOVAL_MARKER
    assert replayed.strip(), "the exchange must survive as a fact about the turn"


def test_the_marker_itself_does_not_read_as_a_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caught by this test on the first draft, which used the words the rule
    keys on. A marker that trips its own rule would be read as a fresh claim by
    the recovery turn and struck on screen by the surface."""
    assert claimed.claims_recorded_proposal(history.CLAIM_REMOVAL_MARKER) is False
    assert claimed.without_claims(history.CLAIM_REMOVAL_MARKER) == history.CLAIM_REMOVAL_MARKER
    assert claimed.claims_recorded_proposal(history.ANSWER_TRUNCATION_MARKER) is False


def test_a_truthful_answer_is_untouched_whatever_the_batch_says() -> None:
    honest = 'ההכנסה הצפויה השבוע היא 9,090,175 ש"ח (overview_summary.week). לא נרשמה הצעה.'
    assert history._replay_answer(honest, False) == honest
    assert history._replay_answer(honest, True) == honest


# --- end to end through the stored thread --------------------------------------

def test_the_replayed_conversation_carries_no_unbacked_claim() -> None:
    """The whole path a real ask takes: store two exchanges, one that recorded a
    batch and one that recorded nothing while claiming it did, then read the
    history the next ask would send to the model."""
    conversations.append_exchange("tester", None, "העלה את רצפת השימור ל-82 אחוז",
                                  CLAIM_AND_FIGURE, "b1a2c3d4e5f6")
    newest = conversations.newest_id("tester")
    conversations.append_exchange("tester", newest, "העלה את רצפת השימור ל-84 אחוז",
                                  CLAIM_AND_FIGURE, None)
    messages = history.history_messages("tester", newest)
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    backed, unbacked = messages[1]["content"], messages[3]["content"]
    assert backed == CLAIM_AND_FIGURE
    assert claimed.claims_recorded_proposal(unbacked) is False
    assert history.CLAIM_REMOVAL_MARKER in unbacked


def test_the_measured_conversation_replays_without_the_false_sentence() -> None:
    """The critic's own conversation, read from disk rather than from a fixture.
    The slice replayed before entry 6 carries five claiming answers; four of them
    have a batch id and stay, and the one with ``batch_id`` null is cut."""
    if not MEASURED_THREAD.exists():
        pytest.skip(f"the measured thread is not on this machine ({MEASURED_THREAD})")
    entries = json.loads(MEASURED_THREAD.read_text(encoding="utf-8"))["entries"][:6]
    claiming = [entry for entry in entries if claimed.claims_recorded_proposal(entry.get("answer"))]
    unbacked = [entry for entry in claiming if not entry.get("batch_id")]
    if not unbacked:
        pytest.skip("this thread no longer carries the measured unbacked answer")
    replayed = history._window(entries)
    cut = [answer for _question, answer in replayed if history.CLAIM_REMOVAL_MARKER in answer]
    assert len(cut) == len(unbacked), "every unbacked claim in the slice, and only those"
    kept = [answer for _question, answer in replayed if history.CLAIM_REMOVAL_MARKER not in answer]
    assert sum(1 for answer in kept if claimed.claims_recorded_proposal(answer)) == len(claiming) - len(unbacked)


# --- the two copies of the segment rule stay one rule ---------------------------

JS = ("const m = await import('./src/kai/kai-claimed-action.js');"
      "const cases = JSON.parse(process.env.KAI_TEST_INPUT);"
      "process.stdout.write(JSON.stringify(cases.map((c) => m.claimSegments(c)"
      ".map((s) => [s.text, s.claim]))));")


def test_the_server_cuts_a_sentence_exactly_where_the_browser_strikes_it() -> None:
    """The browser strikes the claiming sentence on screen and the server cuts
    the same one out of the replay. Two implementations that disagreed would
    strike one sentence and cut another, and nobody would be able to tell."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the shipped browser module cannot be executed")
    texts = [CLAIM_AND_FIGURE,
             "רשמתי הצעה אחת. ההכנסה הצפויה היא 9,090,175 ש\"ח.",
             "ההצעה נרשמה (settings: min_retention_floor 0.78 ← 0.80), במצב ממתין לאישור.",
             "לא נרשמה הצעה. רוצה שארשום?",
             "The proposal was recorded and is pending your approval. Revenue is 9,090,175."]
    done = subprocess.run(
        [node, "--input-type=module", "-e", JS], cwd=DASHBOARD, capture_output=True,
        text=True, timeout=120,
        env={**os.environ, "KAI_TEST_INPUT": json.dumps(texts, ensure_ascii=False)},
    )
    assert done.returncode == 0, f"the browser module did not load: {done.stderr[-800:]}"
    theirs = json.loads(done.stdout)
    ours = [[[text, claim] for text, claim in claimed.claim_segments(case)] for case in texts]
    assert ours == theirs
    # A segmentation that loses a character would delete an operator's figure.
    for case, segments in zip(texts, ours):
        assert "".join(text for text, _claim in segments) == case
    assert any(claim for segments in ours for _text, claim in segments)
