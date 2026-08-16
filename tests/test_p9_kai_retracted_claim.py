"""P9: the surface prints the honest sentence instead of the claim.

The companion to ``test_p9_kai_unbacked_recovery.py``, which proves the server
half. Split from it because both together crossed the file-size cap.

The gap a blind critic measured in round 4, in their own words: the correction
"is a footnote under a confident paragraph, the operator reads two contradictory
statements and must trust the smaller one". So the note no longer follows the
claim. When the payload recorded nothing, the honest sentence IS the answer, and
what the model wrote is kept below it, quoted, with the claiming sentences struck
through, because an answer that lied in one line usually carries real figures in
the next and the operator still needs those.

The live half is the same rule while the answer is still streaming: the dock
painted the false claim as it arrived, and a note under it does not unpaint it.
Every stage frame that starts a new model turn now clears the text the previous
turn wrote, so what is on screen is always the turn being written.

Everything here is executed rather than described wherever it can be: the browser
module runs under node exactly as the bundler imports it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import kairos_api.assistant_claimed_action as claimed
import kairos_api.assistant_prompt as prompt

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
KAI = DASHBOARD / "src" / "kai"
THREAD_VIEW = (KAI / "AssistantThread.jsx").read_text(encoding="utf-8")
# The ask itself moved out of AssistantPanel.jsx in the round that added typed
# mention references: the panel stood at 448 lines against the 450-line law and a
# reference has to travel beside the prose it belongs to, so the question, its
# references and the send live in one module together. The wiring this test pins
# is unchanged; only its address is. Reading both files keeps the check on the
# behaviour rather than on a filename.
PANEL = ((KAI / "AssistantPanel.jsx").read_text(encoding="utf-8")
         + (KAI / "assistant-panel-ask.js").read_text(encoding="utf-8"))
LIVE_TURN = (KAI / "kai-live-turn.js").read_text(encoding="utf-8")

# The answer measured in the browser: the claim first, the truth last.
CLAIM_THEN_CORRECTION = (
    "רשמתי שתי הצעות שממתינות לאישורך.\n"
    "הבהרה חשובה: לא באמת רשמתי דבר עדיין. בהודעה זו לא הפעלתי כלי הצעה, ולכן אין הצעה שממתינה לאישורך."
)


# --- the surface: the honest line replaces the claim --------------------------

def test_the_surface_prints_the_honest_line_instead_of_the_claim() -> None:
    """The measured gap. The note used to follow the claim, so the operator read
    a confident paragraph and a footnote correcting it. The model's answer now
    renders only when nothing contradicts it."""
    assert "{entry.answer && !entry.unrecordedClaim ? <ModelText" in THREAD_VIEW
    assert "{entry.answer && entry.unrecordedClaim ? <RetractedText" in THREAD_VIEW
    # Order on screen: the honest sentence, then the quoted original.
    honest = THREAD_VIEW.index("asst-unrecorded")
    quoted = THREAD_VIEW.index("{entry.answer && entry.unrecordedClaim ?")
    plain = THREAD_VIEW.index("{entry.answer && !entry.unrecordedClaim ?")
    assert plain < honest < quoted


def test_the_retracted_answer_strikes_the_claim_and_keeps_the_rest() -> None:
    """Nothing is hidden and nothing is discarded: the sentence that lied is
    struck, the sentences carrying real figures stay readable."""
    assert "<del className=\"asst-struck\"" in THREAD_VIEW
    assert "claimSegments(line).map" in THREAD_VIEW
    assert "'What Mabat wrote, with the unbacked part struck out'" in THREAD_VIEW
    assert "'מה שמבט כתב, כשהחלק שאינו נתמך מסומן במחיקה'" in THREAD_VIEW


def test_the_segments_cover_the_original_exactly_and_mark_only_the_claim() -> None:
    """Executed through node, on the answer measured in the browser: the pieces
    rejoin into the original character for character, and exactly the sentence
    that claims is marked."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the shipped browser module cannot be executed")
    script = ("const m = await import('./src/kai/kai-claimed-action.js');"
              "process.stdout.write(JSON.stringify(m.claimSegments(process.env.KAI_TEST_INPUT)));")
    done = subprocess.run([node, "--input-type=module", "-e", script], cwd=DASHBOARD,
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ, "KAI_TEST_INPUT": CLAIM_THEN_CORRECTION})
    assert done.returncode == 0, done.stderr[-800:]
    segments = json.loads(done.stdout)
    assert "".join(segment["text"] for segment in segments) == CLAIM_THEN_CORRECTION
    claimed_text = "".join(segment["text"] for segment in segments if segment["claim"])
    assert claimed_text.startswith("רשמתי שתי הצעות שממתינות לאישורך")
    assert "לא באמת רשמתי דבר עדיין" not in claimed_text


def test_the_live_text_never_keeps_a_turn_the_answer_replaced() -> None:
    """The dock painted the false claim as it streamed, and a note under it does
    not unpaint it. Executed through node on the shipped module: a new model
    turn, and the verification turn above all, clears what the previous one
    wrote, so what is on screen is always the turn being written."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the shipped browser module cannot be executed")
    script = (
        "const m = await import('./src/kai/kai-live-turn.js');"
        "let live = { text: '', stage: null, facts: null };"
        "const seen = [];"
        "const frames = JSON.parse(process.env.KAI_TEST_INPUT);"
        "for (const f of frames) { if (f.delta) { live = { ...live, text: live.text + f.delta }; }"
        " else { live = m.applyStage(live, f); } seen.push(live.text); }"
        "process.stdout.write(JSON.stringify({ seen, final: live.text, verifying: live.verifying === true,"
        " facts: live.facts, deadline: live.deadlineSeconds }));"
    )
    frames = [
        {"stage": "grounded", "facts": {"channel": "רשת 13", "breaks": 553}},
        {"stage": "thinking", "turn": 1},
        {"delta": "רשמתי שתי הצעות שממתינות לאישורך."},
        {"stage": "verifying", "note": claimed.VERIFYING_NOTE},
        {"delta": "לא נרשמה הצעה. "},
        {"delta": "רוצה שארשום אותה?"},
    ]
    done = subprocess.run([node, "--input-type=module", "-e", script], cwd=DASHBOARD,
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ, "KAI_TEST_INPUT": json.dumps(frames, ensure_ascii=False)})
    assert done.returncode == 0, done.stderr[-800:]
    result = json.loads(done.stdout)
    assert result["final"] == "לא נרשמה הצעה. רוצה שארשום אותה?"
    assert "רשמתי" not in result["final"], "the claim was still on screen when the answer replaced it"
    assert result["verifying"] is True
    # The grounding it painted at 0.08 s is not thrown away with the text.
    assert result["facts"] == {"channel": "רשת 13", "breaks": 553}
    assert "applyStage(prev, stage)" in PANEL and "noteStageLimits(stage, measured)" in PANEL
    # The extra seconds are explained on screen for the rest of the run, rather
    # than a stage label flashing past on its way to the next one.
    trace_view = (KAI / "AssistantRunTrace.jsx").read_text(encoding="utf-8")
    assert "verifying: ['Checking that the change was really recorded', 'בודק שהשינוי אכן נרשם']" in trace_view
    assert "{live.verifying ? (" in trace_view
    assert "'הטיוטה הראשונה אמרה שנרשמה הצעה בזמן שלא נרשמה דבר, ולכן היא נכתבת מחדש.'" in trace_view


def test_the_prompt_forbids_both_measured_shapes_and_buys_the_first_token() -> None:
    """The model-side half. Rule 4 now names the two shapes measured this round,
    and rule 29 turns a silent tool-planning turn into one honest sentence the
    person can read while the tools run."""
    text = prompt.SYSTEM_PROMPT
    assert "opening with the claim and correcting it in a later paragraph" in text
    assert "was that turn's proposal and not this one's" in text
    assert "write ONE short sentence first" in text
    assert "never in the past" in text


def test_the_new_prose_keeps_the_house_style() -> None:
    """No em-dash, no exclamation mark, no retired word, in everything this
    round added that a person or a model reads. The prompt itself is checked for
    punctuation only: it names the two forbidden words in order to forbid them,
    which is the one place they are allowed to appear."""
    # The module's prose is its comments; a negation operator in the code beside
    # them is code and nobody reads it as a sentence.
    comments = "\n".join(line for line in LIVE_TURN.splitlines() if line.strip().startswith("//"))
    written = [claimed.CORRECTION_WITH_TOOLS, claimed.CORRECTION_NO_TOOLS,
               claimed.AFTER_TOOLS_INSTRUCTION, claimed.VERIFYING_NOTE,
               comments, claimed.__doc__ or ""]
    for text in [*written, prompt.SYSTEM_PROMPT]:
        assert "—" not in text and "–" not in text and "!" not in text
    for text in written:
        assert "משתמש" not in text
        assert "חישוב מחדש" not in text and "הפסקות" not in text
