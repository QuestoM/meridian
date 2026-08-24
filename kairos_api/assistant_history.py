"""Conversation-history replay for the assistant's ask pipeline.

The per-user thread (kairos_api.assistant_memory) stores every successful ask.
This module turns the newest slice of that thread into Anthropic messages so a
follow-up question has an anchor: each past exchange is replayed as a
``{"role": "user"}`` turn carrying ONLY the past question text (never a past
CONTEXT block) followed by a ``{"role": "assistant"}`` turn carrying the stored
answer, oldest first, ready to be placed before the current CONTEXT+QUESTION
user message. The window is hard-capped at ``HISTORY_MAX_EXCHANGES`` newest
exchanges within ``HISTORY_CHAR_BUDGET`` total characters, and each stored
answer longer than ``ANSWER_REPLAY_CHARS`` is cut with an explicit marker so
nothing silently disappears. Reads happen under the memory module's lock,
strictly for the caller's own username (the same-user identity keystone) and
scoped to ONE conversation, so parallel conversations never cross-contaminate.
A memory read failure never fails the ask: the history is honestly empty
instead.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos_api import assistant_claimed_action as claimed
from kairos_api import assistant_memory
from kairos_api import assistant_protocol_text as protocol_text

HISTORY_MAX_EXCHANGES = 6
HISTORY_CHAR_BUDGET = 12000
ANSWER_REPLAY_CHARS = 2000
ANSWER_TRUNCATION_MARKER = "[earlier answer shortened]"
# What replaces a sentence that told the person a change was waiting for their
# approval on a turn that recorded none. It states the fact rather than deleting
# it silently, and it is English like every other replay marker because it is
# addressed to the model and not to a person.
#
# It is worded to carry neither word the rule keys on: a marker that trips the
# very rule it serves would be read as a fresh claim by the recovery turn and
# struck on screen by the surface. Nothing else about it is subtle, so the
# constraint is stated here rather than left for the next person to
# rediscover.
CLAIM_REMOVAL_MARKER = (
    "[a sentence was cut from this earlier answer: it told the person a change was waiting for "
    "their approval, and that turn had recorded nothing]"
)

logger = logging.getLogger(__name__)


def _replay_answer(answer: str, backed: bool | None = None) -> str:
    """The stored answer capped for replay, cut with an explicit marker.

    Any tool-call protocol an older answer carries is cut first. A stored
    answer is replayed as an assistant turn, so a leaked call in one would
    teach the next turn to write its calls the same way, and the leak would
    keep re-teaching itself for as long as the conversation lives. An answer
    that was nothing but protocol becomes empty here and the caller drops the
    exchange entirely.

    A false claim re-teaches itself exactly the same way, and this one was
    measured. In this repository's own stored thread ``697dabafa588`` seven of
    the first eight answers open with ``רשמתי שתי הצעות שממתינות לאישורך``, and
    two of them (entries 2 and 6) carry ``batch_id`` null: nothing was recorded
    on those turns. A blind critic then measured the same sentence coming back
    on 1 of 3 asks inside that conversation against 0 of 4 in fresh ones, which
    is what a model does with six examples of its own voice saying it. So an
    answer replayed with no batch behind it and a claim inside it loses the
    claiming SENTENCES only: the figures, the reasoning and the offer that
    shared the paragraph with it are the operator's context and stay.

    ``backed`` is a tri-state read of the stored entry, never a guess. True is a
    batch id, False is an explicit null, and None is a stored shape that carries
    no such field at all, which is left exactly as it was written.
    """
    answer = protocol_text.strip_tool_protocol(answer)
    if backed is False and claimed.claims_recorded_proposal(answer):
        answer = f"{claimed.without_claims(answer)} {CLAIM_REMOVAL_MARKER}".strip()
    if len(answer) <= ANSWER_REPLAY_CHARS:
        return answer
    return answer[:ANSWER_REPLAY_CHARS].rstrip() + " " + ANSWER_TRUNCATION_MARKER


def _window(entries: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The newest exchanges that fit both caps, returned oldest first.

    Walks the stored entries newest first, keeps an exchange only when its
    question and answer are non-empty strings (an empty side would break the
    strict user/assistant alternation the API requires), and stops as soon as
    the next exchange would cross ``HISTORY_CHAR_BUDGET`` total characters or
    ``HISTORY_MAX_EXCHANGES`` exchanges.
    """
    kept: list[tuple[str, str]] = []
    used = 0
    for entry in reversed(entries):
        question = entry.get("question")
        answer = entry.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        question = question.strip()
        # The stored proof, tri-state: a batch id means that turn really did
        # record a proposal, an explicit null means it did not, and an entry
        # written by a shape that never carried the field means unknown.
        backed = bool(entry.get("batch_id")) if "batch_id" in entry else None
        answer = _replay_answer(answer.strip(), backed)
        if not question or not answer:
            continue
        cost = len(question) + len(answer)
        if kept and used + cost > HISTORY_CHAR_BUDGET:
            break
        if not kept and cost > HISTORY_CHAR_BUDGET:
            break
        kept.append((question, answer))
        used += cost
        if len(kept) >= HISTORY_MAX_EXCHANGES:
            break
    kept.reverse()
    return kept


def history_messages(username: str, conversation_id: str | None = None) -> list[dict[str, str]]:
    """One conversation of the caller's own thread as alternating messages.

    Loads the conversation derived strictly from ``username`` (the
    authenticated actor, resolved by the caller) under the memory module's
    lock, windows it, and replays it oldest first. Only the named conversation
    is replayed (the newest one when ``conversation_id`` is None), so parallel
    conversations never cross-contaminate. Any failure returns an empty list:
    history is additive and must never fail or delay the ask it decorates.
    """
    from kairos_api import assistant_conversations

    try:
        with assistant_memory._LOCK:
            selected = conversation_id or assistant_conversations.newest_id(username)
            entries = (assistant_conversations.entries_for(username, selected)
                       if selected else [])
        messages: list[dict[str, str]] = []
        for question, answer in _window(entries):
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages
    except Exception:  # noqa: BLE001 - honest empty history over a failed ask
        logger.exception("assistant history load failed for user %s", username)
        return []
