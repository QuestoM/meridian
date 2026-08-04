"""A claim that a proposal is waiting for approval, checked against the payload.

Split out of kairos_api.assistant_pipeline so that module stays under the
file-size cap, and because this is one idea with two halves that belong beside
each other: reading whether an answer claims a change was recorded, and doing
something about it when nothing was.

**The measured defect.** A blind critic, browser, first ask of a session, plain
Hebrew "העלה את רצפת השימור ל-82 אחוז". The answer opened

    רשמתי שתי הצעות שממתינות לאישורך

and listed both. ``GET /api/assistant/audit`` for that turn recorded
``tools: []``, the ask body carried ``proposals: null`` and ``tool_trace: []``,
and the proposal store did not grow. The operator was told a change was waiting
for their approval and there was nothing anywhere to approve. Measured rate in
that session: 2 of 9 action asks, and 1 of 3 inside a conversation that already
carried a proposal for the same field against 0 of 4 in fresh conversations.

Two halves already existed and neither was enough on its own. The prompt forbids
the sentence, and a rule the model can ignore is not a guarantee. The surface
prints an honest note beside the claim (``tv-break-dashboard/src/kai/
kai-claimed-action.js``), and a correction the operator has to trust over the
paragraph above it costs them a whole second round trip.

**What this module adds is the recovery the critic performed by hand.** The
detection is the port of that shipped browser module, verb for verb and filter
for filter, so the two surfaces cannot disagree about what a claim is
(``tests/test_p9_kai_unbacked_recovery.py`` runs both over the same corpus and
compares). When it fires and the tool record shows no successful ``propose_*``
step, ``correct`` runs one more turn: the model is told, in the channel that
carries tool results rather than in the person's voice, that its claim is
unbacked, and it must either call the propose tool now or write the answer again
without the claim. Done by hand it worked first time.

Why the correction is a user-role message and not a ``tool_result`` block. A
``tool_result`` is only legal against a ``tool_use`` in the immediately previous
assistant turn, and the turn that produced the false claim ended with text and
no call, so there is no id to answer. The message is therefore marked as a
system verification in its own first line, never phrased as the person speaking.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from kairos_api import assistant_protocol_text as protocol_text, assistant_tools

# --- the detection, ported from kai-claimed-action.js -------------------------
# Every pattern below is the same source text as the browser module's, and the
# corpus reasoning that produced each one is documented there rather than
# repeated here. The one deliberate difference is stated at NEGATED_EN.

# Digits keep their decimal point through the sentence split, so a claim is not
# cut in half by the very number it quotes. The replacement is the same length
# as what it replaces, which is what lets a caller map a span back onto the
# original text unshifted.
DECIMAL_POINT = re.compile(r"(\d)\.(\d)")
SENTENCE = re.compile(r"[\n.?!]+")

PROPOSAL = re.compile(r"הצע|proposal", re.IGNORECASE)

RECORDED_DONE = r"נרשמה|נרשמו|הוגשה|הוגשו|נשמרה|נשמרו|נוצרה|נוצרו|נשלחה|נשלחו|רשמתי|הגשתי|שמרתי|יצרתי|הכנתי|שלחתי"
RECORDED_WAITING = r"ממתינה לאישור|ממתינות לאישור|ממתין לאישור|מחכה לאישור|מחכות לאישור|בתור לאישור|במצב pending|במצב ממתין"
RECORDED_EN = (r"recorded|registered|submitted|logged|created a proposal|pending approval|"
               r"pending your approval|awaiting approval|awaiting your approval|is pending|"
               r"are pending|waiting in the pending")
RECORDED = re.compile("|".join((RECORDED_DONE, RECORDED_WAITING, RECORDED_EN)), re.IGNORECASE)

NEGATION_WINDOW = 24
NEGATED_HE = re.compile(r"(^|[^֐-׿])(לא|אין|אינה|אינו|טרם|ללא|בלי)([^֐-׿]|$)")
# re.ASCII on purpose: JavaScript's \b is ASCII-only, so in the browser a Hebrew
# letter before "no" is a word boundary. Without this flag Python treats Hebrew
# letters as word characters and the same sentence classifies differently on the
# two surfaces, which is the one thing a ported rule may not do.
NEGATED_EN = re.compile(r"\b(no|not|nothing|never|without|cannot)\b", re.IGNORECASE | re.ASCII)

OFFER = re.compile(r"רוצה ש|תרצה ש|אם תרצה|האם|אוכל ל|אני יכול|אפשר ש|would you like|shall i|if you want",
                   re.IGNORECASE)


def _negated_before(sentence: str, index: int) -> bool:
    before = sentence[max(0, index - NEGATION_WINDOW):index]
    return bool(NEGATED_HE.search(before) or NEGATED_EN.search(before))


def sentence_claims(sentence: str) -> bool:
    """True when this one sentence asserts that a proposal is recorded."""
    if not PROPOSAL.search(sentence) or OFFER.search(sentence):
        return False
    found = RECORDED.search(sentence)
    return bool(found) and not _negated_before(sentence, found.start())


def claims_recorded_proposal(text: Any) -> bool:
    """True when the text asserts, somewhere in it, that a proposal is recorded,
    registered, submitted or waiting for approval."""
    value = DECIMAL_POINT.sub(r"\1 \2", "" if text is None else str(text))
    return any(sentence_claims(part) for part in SENTENCE.split(value))


def recorded_proposal(trace: list[dict[str, Any]] | None,
                      items: list[dict[str, Any]] | None) -> bool:
    """Did this turn actually record a proposal.

    A captured item is the direct proof; a ``propose_*`` step that returned ok is
    the same fact read off the trace, and it stands even when the item failed to
    normalize. A refused propose call captures no item and lands with ok false
    (``assistant_tools.py:418-431``), so it is correctly not proof of anything.
    """
    if items:
        return True
    for step in trace or []:
        if isinstance(step, dict) and step.get("ok") is True and str(step.get("tool") or "").startswith("propose"):
            return True
    return False


def unbacked_claim(answer: Any, trace: list[dict[str, Any]] | None,
                   items: list[dict[str, Any]] | None) -> bool:
    """The one question: was the person told a change is waiting for their
    approval when the payload of this same turn recorded nothing."""
    if recorded_proposal(trace, items):
        return False
    return claims_recorded_proposal(answer)


# --- the recovery turn --------------------------------------------------------

# What the model is told. It is deliberately blunt about the two acceptable
# outcomes and it forbids the third shape the corpus already contains: an honest
# paragraph appended UNDER the false one, which leaves the operator reading two
# contradictory statements and having to trust the smaller.
CORRECTION_WITH_TOOLS = (
    "SYSTEM VERIFICATION, not the person speaking. "
    "Your last answer says a proposal was recorded, submitted or is waiting for approval. "
    "The tool record for this turn carries no successful propose_ call, so nothing was recorded, "
    "nothing is pending and there is nothing for the person to approve. As written the answer is false "
    "and they would act on it. "
    "Do exactly one of these two things now, in the language of the question. "
    "Either call the propose_ tool for the change they asked for, in this turn, and then say what it recorded. "
    "Or write the answer again without the claim: say plainly that nothing was recorded, name the change "
    "you would propose, and offer to record it. "
    "Write the whole answer as it should have been. Do not repeat the false sentence, do not leave it standing "
    "with a correction under it, and do not spend the answer apologising."
)
CORRECTION_NO_TOOLS = (
    "SYSTEM VERIFICATION, not the person speaking. "
    "Your last answer says a proposal was recorded, submitted or is waiting for approval. "
    "No proposal tool is available in this conversation, so nothing was recorded, nothing is pending "
    "and there is nothing for the person to approve. As written the answer is false and they would act on it. "
    "Write the answer again now, in the language of the question, without the claim: say plainly that nothing "
    "was recorded and that this account cannot propose changes here. "
    "Write the whole answer as it should have been, and do not leave the false sentence standing with a "
    "correction under it."
)
# The final tool-free turn AFTER a propose call landed. The loop's own
# FINAL_TURN_INSTRUCTION says the opposite (that nothing was recorded), which is
# right where it is used and wrong here, so this turn carries its own.
AFTER_TOOLS_INSTRUCTION = (
    "This is the final turn of this answer and no tool is available on it. "
    "Answer now, in the language of the question, using only the context and the tool results already in this "
    "conversation. Never write a tool call, an invoke tag or any other protocol text in an answer. "
    "State exactly what the propose tool results in this conversation say was recorded, and nothing beyond them: "
    "if a call returned captured false, that item was not recorded and you must say so."
)
# What the stage frame carries, so the surface can say what is happening in its
# own two languages rather than printing this record.
VERIFYING_NOTE = "The answer claimed a recorded proposal and the tool record showed none, so it is being checked."


def _echoed(blocks: list[Any]) -> list[dict[str, Any]]:
    from kairos_api import assistant_pipeline as pipeline

    return [echo for echo in (pipeline.echo_block(block) for block in blocks) if echo is not None]


def correct(client: Any, messages: list[dict[str, Any]], blocks: list[Any], answer: str, *,
            trace: list[dict[str, Any]], items: list[dict[str, Any]],
            can_propose: bool = True, user: str | None = None,
            auth_mode: str | None = None, job: str | None = None,
            on_text: Callable[[str], None] | None = None,
            on_step: Callable[[dict[str, Any]], None] | None = None,
            on_stage: Callable[[str, dict[str, Any]], None] | None = None,
            deadline: float | None = None) -> tuple[str, dict[str, Any]]:
    """One more turn, because the answer claimed a proposal nothing backs.

    ``messages``, ``trace`` and ``items`` are the loop's own caller-owned lists
    and are extended in place, so a proposal recorded here reaches the batch and
    the audit trail exactly as one recorded in the loop does. Returns the answer
    to use and an outcome fragment naming how it ended, and it never returns
    something worse than what it was given: on any failure the original answer
    comes back untouched and the surface's own guard still prints the honest
    note over it.
    """
    from kairos_api import assistant, assistant_pipeline as pipeline

    if deadline is not None and time.monotonic() > deadline:
        return answer, {"claim_recovery": "skipped_deadline"}
    if on_stage is not None:
        on_stage("verifying", {"note": VERIFYING_NOTE})
    messages.append({"role": "assistant", "content": _echoed(blocks)})
    messages.append({"role": "user",
                     "content": CORRECTION_WITH_TOOLS if can_propose else CORRECTION_NO_TOOLS})
    kwargs: dict[str, Any] = {
        "model": assistant._model_name(),
        "max_tokens": pipeline.LOOP_MAX_TOKENS,
        "system": assistant._system_blocks(auth_mode=auth_mode, job=job),
        "messages": messages,
    }
    if can_propose:
        kwargs["tools"] = assistant_tools.anthropic_tools(include_propose=True)
    try:
        gate = protocol_text.LiveTextGate(on_text) if on_text is not None else None
        response = pipeline.call_model(client, kwargs, gate.feed if gate is not None else None)
        if gate is not None:
            gate.flush()
        blocks_out = list(getattr(response, "content", []) or [])
        tool_uses = [block for block in blocks_out if getattr(block, "type", "") == "tool_use"]
        if tool_uses and getattr(response, "stop_reason", None) == "tool_use":
            results = []
            for block in tool_uses:
                before = len(trace)
                results.append(assistant_tools.handle_tool_use(block, trace, items,
                                                               propose_allowed=can_propose, user=user))
                if on_step is not None:
                    for step in trace[before:]:
                        on_step(step)
            messages.append({"role": "assistant", "content": _echoed(blocks_out)})
            messages.append({"role": "user", "content": results})
            # A new turn is starting, so the surface clears the sentence the
            # correction turn wrote before it called the tool. What is painted
            # is always the turn being written, never two of them stacked.
            if on_stage is not None:
                on_stage("thinking", {"searching": False})
            text = pipeline.plain_answer(client, messages, auth_mode, job, on_text,
                                         instruction=AFTER_TOOLS_INSTRUCTION)
            outcome = "proposed" if recorded_proposal(trace, items) else "refused"
            return (text or answer), {"claim_recovery": outcome}
        text = protocol_text.strip_tool_protocol(pipeline.extract_answer(response))
        if not text:
            return answer, {"claim_recovery": "empty"}
        return text, {"claim_recovery": "restated" if not claims_recorded_proposal(text) else "unfixed"}
    except Exception as exc:  # noqa: BLE001 - a failed correction is named, never hidden
        return answer, {"claim_recovery": "failed", "claim_recovery_error": pipeline.describe_error(exc)}
