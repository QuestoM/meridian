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
nothing silently disappears. Reads happen under the memory module's lock and
strictly for the caller's own username (the same-user identity keystone). A
memory read failure never fails the ask: the history is honestly empty instead.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos_api import assistant_memory

HISTORY_MAX_EXCHANGES = 6
HISTORY_CHAR_BUDGET = 12000
ANSWER_REPLAY_CHARS = 2000
ANSWER_TRUNCATION_MARKER = "[earlier answer shortened]"

logger = logging.getLogger(__name__)


def _replay_answer(answer: str) -> str:
    """The stored answer capped for replay, cut with an explicit marker."""
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
        answer = _replay_answer(answer.strip())
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


def history_messages(username: str) -> list[dict[str, str]]:
    """The caller's own thread as alternating user/assistant messages.

    Loads the thread file derived strictly from ``username`` (the authenticated
    actor, resolved by the caller) under the memory module's lock, windows it,
    and replays it oldest first. Any failure returns an empty list: history is
    additive and must never fail or delay the ask it decorates.
    """
    try:
        with assistant_memory._LOCK:
            entries = assistant_memory._load_entries(assistant_memory._path_for(username))
        messages: list[dict[str, str]] = []
        for question, answer in _window(entries):
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages
    except Exception:  # noqa: BLE001 - honest empty history over a failed ask
        logger.exception("assistant history load failed for user %s", username)
        return []
