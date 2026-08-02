"""Tool-call protocol text: recognise it, cut it, and never stream it.

Measured on 2026-08-01, twice, on the goal-seek ask that is Kai's own headline
flow: ``POST /api/assistant/ask/stream`` returned after 71.6 s with an answer
whose whole body was a tool call the model had written into the TEXT channel
instead of the tool channel, rendered verbatim in the dock, with no proposal to
approve behind it. A tool call written as text reaches nothing: it is neither an
answer a person can read nor a call the loop can run.

So this module holds the one definition of what that text looks like, used in
three places: the answer a turn returns (cut before it can be returned), the
deltas streamed to the browser while the turn is still running (held back
before they can paint), and a stored answer replayed as history (dropped before
it can teach the next turn the same habit).

Nothing here decides what to do about it. The pipeline does that.
"""

from __future__ import annotations

import re
from typing import Callable

# A tool call written in the text channel. The anchors are the tag names the
# protocol uses, with an optional namespace prefix and an optional closing
# slash, so an opening call, one parameter and a closing tag all match.
TOOL_PROTOCOL_RE = re.compile(
    r"<\s*/?\s*(?:[A-Za-z][\w.-]*:)?(?:invoke|function_calls|parameter|antml)\b",
    re.IGNORECASE,
)

# The lead-in the model writes on the line above the first tag, which is the
# tail of the opening word. It is cut with the tags, and only ever when tags
# were actually found, so a sentence that merely ends in the word call survives.
_LEAD_IN_RE = re.compile(r"(?:^|\n)[ \t]*<?[A-Za-z_:]*calls?[ \t]*\Z")

# The same lead-in seen from the streaming side: a short word alone on the last
# line of what has arrived so far, finished or not. Held back until the next
# delta says whether a tag follows it or ordinary prose does.
_LEAD_IN_TAIL_RE = re.compile(r"\n[ \t]*[A-Za-z_]{0,20}[ \t]*\n?\Z")


def looks_like_tool_protocol(text: str | None) -> bool:
    """True when this text carries a tool call instead of being an answer."""
    return bool(text) and TOOL_PROTOCOL_RE.search(text or "") is not None


def strip_tool_protocol(text: str | None) -> str:
    """The answer with any tool-call protocol cut off, or the text untouched.

    Text with no protocol in it is returned exactly as it came, so the ordinary
    answer is never reshaped. Text with protocol keeps only the prose BEFORE
    the first tag, which is the part the person can actually read, and loses
    the lead-in word above it.
    """
    value = text or ""
    match = TOOL_PROTOCOL_RE.search(value)
    if match is None:
        return value
    return _LEAD_IN_RE.sub("", value[: match.start()].rstrip()).strip()


class LiveTextGate:
    """One turn's text on its way to the browser, minus any protocol.

    Deltas go out as they arrive, because the first token has a two-second
    budget and buffering a whole turn would spend it, so this holds back only
    the shortest tail that could still turn into a tag: everything from the
    last unclosed ``<``, plus a short unfinished word on its own line. When a
    tag does arrive the gate closes for the rest of the turn and the held tail
    is dropped rather than painted.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._buffer = ""
        self._sent = 0
        self._blocked = False

    @property
    def blocked(self) -> bool:
        """True when this turn turned into protocol text and was cut off."""
        return self._blocked

    def feed(self, text: str) -> None:
        if self._blocked or not text:
            return
        self._buffer += text
        match = TOOL_PROTOCOL_RE.search(self._buffer)
        # A tag closes the gate for the rest of the turn. What goes out is the
        # prose before it, minus the lead-in line above it, which reads as
        # nothing at all once the call it announced is gone.
        hold = match.start() if match is not None else self._buffer.rfind("<")
        if hold == -1:
            hold = len(self._buffer)
        lead = _LEAD_IN_TAIL_RE.search(self._buffer, 0, hold)
        if lead is not None:
            hold = lead.start()
        self._blocked = match is not None
        self._send_to(hold)

    def flush(self) -> None:
        """End of turn: whatever is still held is prose, so it goes out now."""
        if not self._blocked:
            self._send_to(len(self._buffer))

    def _send_to(self, end: int) -> None:
        if end > self._sent:
            self._emit(self._buffer[self._sent:end])
            self._sent = end
