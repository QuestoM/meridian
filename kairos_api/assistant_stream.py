"""Streaming ask endpoint for the assistant (server-sent events).

POST /api/assistant/ask/stream runs the SAME ask pipeline as the non-streaming
/ask (same auth middleware, same rate limit, same grounding, tool loop,
proposals and honesty rules) and streams progress as SSE frames, in order:

  * zero or more ``event: step`` frames, one right after each tool result,
    with data ``{"tool", "ok", "source"}`` (source null when the step has none);
  * zero or more ``event: delta`` frames with data ``{"text"}`` carrying
    assistant text as it is produced (real token deltas when the client
    supports ``messages.stream``, else each turn's answer as a single delta);
  * exactly one terminal frame: ``event: final`` whose data is the EXACT JSON
    body the non-streaming /ask would have returned for the same question, or
    ``event: error`` with ``{"error"}`` when the pipeline itself crashed.

The final body is audited and appended to the caller's thread exactly once,
mirroring a non-streaming ask. The pipeline runs in a worker thread feeding a
queue so frames flush to the client as they happen, not after the loop ends.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kairos_api import assistant_actions, assistant_memory

# No prefix: kairos_api.assistant includes this router under /api/assistant.
router = APIRouter(tags=["assistant"])


class StreamAskRequest(BaseModel):
    """Same body contract as the non-streaming ask. Declared locally (not
    imported from kairos_api.assistant, which imports this module to mount the
    router) so the modules stay import-order safe."""

    question: str = Field(min_length=1, max_length=2000)


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

    def on_step(step: dict[str, Any]) -> None:
        events.put(
            ("step", {"tool": step.get("tool"), "ok": bool(step.get("ok")),
                      "source": step.get("source") or None})
        )

    def on_text(text: str) -> None:
        events.put(("delta", {"text": text}))

    def worker() -> None:
        try:
            body = assistant._ask_body(question, http_request, on_step=on_step, on_text=on_text)
            batch_id = body["proposals"]["batch_id"] if body.get("proposals") else None
            assistant._audit_ask(user, question, body, batch_id)
            if body.get("answer"):
                assistant_memory.append_entry(user, question, str(body["answer"]), batch_id)
            events.put(("final", body))
        except Exception as exc:  # noqa: BLE001 - the terminal frame is honest, never absent
            events.put(("error", {"error": assistant._describe_error(exc)}))
        finally:
            events.put(None)

    threading.Thread(target=worker, name="kairos-assistant-stream", daemon=True).start()

    def generate() -> Iterator[str]:
        while True:
            item = events.get()
            if item is None:
                break
            yield _frame(*item)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
