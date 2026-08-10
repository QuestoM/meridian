"""Streaming ask endpoint for the assistant (server-sent events).

POST /api/assistant/ask/stream runs the SAME ask pipeline as the non-streaming
/ask (same auth middleware, same rate limit, same grounding, tool loop,
proposals and honesty rules) and streams progress as SSE frames, in order:

  * ``event: stage`` frames naming what the server is doing right now, starting
    with ``accepted`` before any work begins and continuing through ``reading``,
    ``grounded`` and one ``thinking`` per model turn, each with the elapsed
    seconds since the request was accepted;
  * zero or more ``event: step`` frames, one right after each tool result,
    with data ``{"tool", "ok", "source"}`` (source null when the step has none);
  * zero or more ``event: delta`` frames with data ``{"text"}`` carrying
    assistant text as it is produced (real token deltas when the client
    supports ``messages.stream``, else each turn's answer as a single delta);
  * exactly one terminal frame: ``event: final`` whose data is the EXACT JSON
    body the non-streaming /ask would have returned for the same question, or
    ``event: error`` with ``{"error"}`` when the pipeline itself crashed.

Two properties the measured failure demanded. **The first frame does not wait
for the pipeline.** Discovery measured a browser sitting on "preparing an
answer" for 499 s with no reply, no error and nothing to cancel, while the same
question answered in 78 s on the wire, so the stream now proves it is alive
before it does any work. **A silent stretch is never silent.** A heartbeat
comment goes out every HEARTBEAT_SECONDS while a turn is in flight, which keeps
an intermediary from buffering the response into a single late block and lets
the dock show honest elapsed time rather than a spinner with no clock.

The final body is audited and appended to the caller's thread exactly once,
mirroring a non-streaming ask. The pipeline runs in a worker thread feeding a
queue so frames flush to the client as they happen, not after the loop ends.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kairos_api import assistant_actions, assistant_memory

# No prefix: kairos_api.assistant includes this router under /api/assistant.
router = APIRouter(tags=["assistant"])

HEARTBEAT_SECONDS = 5.0


def _opening_line(question: str) -> str:
    """An honest first line that needs no model and claims no result.

    The model's first network token is not a reliable two-second boundary. The
    stream can still begin the answer immediately by stating the work it is
    actually starting, in the language the question itself uses. It contains
    no figure, conclusion or promise, so later evidence cannot contradict it.
    """
    if any("\u0590" <= character <= "\u05ff" for character in question):
        return "בודק את השאלה מול נתוני המערכת. "
    return "Checking the question against the system data. "


class StreamAskRequest(BaseModel):
    """Same body contract as the non-streaming ask. Declared locally (not
    imported from kairos_api.assistant, which imports this module to mount the
    router) so the modules stay import-order safe."""

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=80)
    # The frozen page-context contract: advisory grounding for where the
    # operator is; absent or invalid degrades to exactly today's behavior.
    page_context: dict[str, Any] | None = None
    # The typed references the operator pointed at in the question. Same
    # advisory contract as the non-streaming ask.
    mentions: list[dict[str, Any]] | None = None


def _frame(event: str, data: Any) -> str:
    """One SSE frame. json.dumps escapes newlines, so the data stays one line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/ask/stream")
def assistant_ask_stream(request: StreamAskRequest, http_request: Request) -> StreamingResponse:
    from kairos_api import assistant

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    if assistant._rate_limited():
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: at most {assistant.RATE_LIMIT_ASKS} questions per minute.",
        )
    user = assistant_actions._actor(http_request)
    events: queue.Queue[tuple[str, Any] | None] = queue.Queue()
    started = time.monotonic()

    def elapsed() -> float:
        return round(time.monotonic() - started, 3)

    def on_stage(name: str, detail: dict[str, Any] | None = None) -> None:
        events.put(("stage", {"stage": name, "elapsed_seconds": elapsed(), **(detail or {})}))
        # Grounding is the last internal step before the first provider call.
        # Begin the answer here, after the visible scope fact and before any
        # external latency, so the prose and its evidence keep their order.
        if name == "grounded":
            events.put(("delta", {"text": _opening_line(question)}))

    def on_step(step: dict[str, Any]) -> None:
        events.put(
            ("step", {"tool": step.get("tool"), "ok": bool(step.get("ok")),
                      "source": step.get("source") or None, "elapsed_seconds": elapsed()})
        )

    def on_text(text: str) -> None:
        events.put(("delta", {"text": text}))

    def worker() -> None:
        try:
            body = assistant._ask_body(question, http_request, on_step=on_step, on_text=on_text,
                                       conversation_id=request.conversation_id,
                                       page_context=request.page_context,
                                       mentions=request.mentions,
                                       on_stage=on_stage)
            batch_id = body["proposals"]["batch_id"] if body.get("proposals") else None
            assistant._audit_ask(user, question, body, batch_id)
            if body.get("answer"):
                assistant_memory.append_entry(user, question, str(body["answer"]), batch_id,
                                              conversation_id=body.get("conversation_id"))
            events.put(("final", body))
        except Exception as exc:  # noqa: BLE001 - the terminal frame is honest, never absent
            events.put(("error", {"error": assistant._describe_error(exc),
                                  "elapsed_seconds": elapsed()}))
        finally:
            events.put(None)

    worker_thread = threading.Thread(target=worker, name="kairos-assistant-stream", daemon=True)

    def generate() -> Iterator[str]:
        # Proof of life before any work: the browser has a frame in hand within
        # milliseconds, so an ask can never look identical to a dead connection.
        yield _frame("stage", {"stage": "accepted", "elapsed_seconds": 0.0,
                               "deadline_seconds": assistant._deadline_seconds()})
        worker_thread.start()
        while True:
            try:
                item = events.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                # An SSE comment: no event, no data, and no client-visible frame,
                # but it keeps the connection warm and unbuffered.
                yield f": alive {elapsed()}s\n\n"
                continue
            if item is None:
                break
            yield _frame(*item)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
