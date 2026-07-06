"""In-product AI assistant grounded in the saved Kairos payloads.

The ask endpoint composes context from the SAME dashboard builders plus the
day-level grounding in kairos_api.assistant_context, under a character budget
with honest truncation flags. With an API key the question runs an Anthropic
tool-use loop: READ tools (including simulate_settings_change, a side-effect-free
owned-channel what-if) execute immediately and are stamped with a provenance
source; PROPOSE tools are captured as pending items
(kairos_api.assistant_actions) that only an operator's approval applies, each
settings change carrying its simulated effect. The loop searches with adaptive
thinking so a stated goal can be met by trying settings against the real
optimizer. The response carries the grounding manifest, the tool trace (with
sources) and the proposal batch; a failed section is listed absent, never
fabricated. Without a key both routes report available false honestly.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos_api import assistant_actions, assistant_context, assistant_tools

router = APIRouter(prefix="/api/assistant", tags=["assistant"])
router.include_router(assistant_actions.router)

DEFAULT_MODEL = "claude-sonnet-4-6"
MODEL_ENV = "KAIROS_ASSISTANT_MODEL"
KEY_ENVS = ("ANTHROPIC_API_KEY", "KAIROS_ASSISTANT_API_KEY")
KEY_MISSING_REASON = "API key not configured"
ACTIONS_ENV = "KAIROS_ASSISTANT_ACTIONS"
ACTIONS_DISABLED_REASON = f"disabled by {ACTIONS_ENV}"
ASK_TIMEOUT_SECONDS = 30.0
MAX_ANSWER_TOKENS = 1000  # plain Q&A call when the action plane is off
LOOP_MAX_TOKENS = 1500
# The goal-seeker searches by calling simulate_settings_change repeatedly, so the
# loop needs room to try several settings before it converges; the ceiling stays
# hard so a runaway conversation still terminates.
MAX_TOOL_ITERATIONS = 10
ANSWER_TEMPERATURE = 0.2
LOOP_EFFORT = "medium"
RATE_LIMIT_ASKS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0

# The grounding contract. Every rule is load-bearing: the assistant may only
# restate what the composed context and this turn's tool results carry, must
# name missing data instead of guessing, never crosses the competitor boundary,
# and never presents a proposal as an executed change.
SYSTEM_PROMPT = (
    "You are the Kairos schedule analyst, the in-product assistant of a TV ad-break "
    "revenue optimizer. The user message contains a CONTEXT block of JSON computed "
    "from the operator's saved data, followed by the operator's QUESTION. Tools let "
    "you read more saved state and propose changes for review. "
    "Rules, in priority order: "
    "1. Language: Hebrew first. Mirror the language of the question; when the "
    "question is in Hebrew, answer in natural Hebrew. "
    "2. Grounding: every number, count, date, currency amount and verdict in your "
    "answer must be taken from the CONTEXT block or from a tool result in this "
    "conversation, and when you state a figure, name the context section or tool "
    "it came from. Never invent, estimate, extrapolate or recall figures from "
    "memory or general knowledge. "
    "3. Missing data: when the question needs data that is in neither CONTEXT nor "
    "a read tool's result, say exactly that, name the specific missing data, and "
    "stop. A source section marked absent failed to load and is unavailable. "
    "4. Proposals: you never change anything yourself. A propose_* tool only "
    "records a proposal; the operator reviews and approves or rejects it, and only "
    "approved items are applied. Say this plainly whenever you propose. Propose "
    "related changes together in one turn (for example a settings change plus the "
    "recompute that makes it take effect), each with a concrete reason. "
    "5. Competitor boundary: the operator owns exactly one channel; never state, "
    "estimate or speculate about competitor revenue or competitor performance, and "
    "never propose or discuss actions on another channel. Competitor channels "
    "appear in CONTEXT only as aggregate counts, never by name or by figure. "
    "6. Context layout: per_day_plan is a per-day table of the operator's own "
    "channel (date, weekday, breaks, revenue in ILS, average retention percent). "
    "When the question names a date, weekday or time found in the saved plan, "
    "day_detail sections carry that day's segments ordered by revenue, highest "
    "first, and matched_full_rows carries the complete saved fields for segments "
    "matching a time or programme type named in the question. "
    "7. Truncation: when a day_detail section carries truncated true, or the "
    "context carries day_detail_truncated true, rows were cut to fit the context "
    "budget. When your answer relies on such a section, say so. "
    "8. Currency and units: monetary amounts are in ILS unless the context states "
    "otherwise; attach units to every number. "
    "9. Style: short and concrete, plain text only, no markdown formatting. Prefer "
    "two to six sentences, or a short plain list for several figures. "
    "10. Provenance: every read tool result carries a source field; name that source "
    "for each figure you state, and never give a number without a context section or "
    "tool result behind it. "
    "11. Simulation: simulate_settings_change runs the owned-channel optimizer under "
    "proposed settings and returns the before and after (gross, retention cost, net, "
    "breaks) plus deltas, changing nothing; use it for any settings what-if and say the "
    "numbers are a simulation, not the saved plan. "
    "12. Goal-seek: on a stated goal, call simulate_settings_change repeatedly to try "
    "settings against the optimizer, compare each result to the goal, and only when one "
    "meets it emit ONE propose_settings_change for it, never applying mid-search; if "
    "nothing meets it, say so and report the closest result and its settings."
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _api_key() -> str | None:
    for name in KEY_ENVS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _model_name() -> str:
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL


def _actions_enabled() -> bool:
    """The action plane (tool loop + proposals) is on unless explicitly disabled."""
    return os.environ.get(ACTIONS_ENV, "").strip().lower() not in {"0", "false", "no", "off"}


def _client_factory(api_key: str) -> Any:
    """Build the Anthropic client. Module-level seam so tests can mock it."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key, timeout=ASK_TIMEOUT_SECONDS, max_retries=1)


# Context sections. Each reuses a real dashboard builder, so the assistant reads
# exactly what the operator's own pages render, nothing else.
def _overview_body() -> dict[str, Any]:
    server = _server()
    return server._overview_cached(
        server._signature(
            [
                server.OUTPUT_DIR / "weekly_break_schedule.csv",
                server.DATA_DIR / "reference" / "Programmes.xlsx",
                server.DATA_DIR / "reference" / "Spots.xlsx",
                server.DATA_DIR / "Programmes.csv",
                server.DATA_DIR / "Spots.csv",
                server.SETTINGS_PATH,
            ]
        ),
        None,
    )


def _section_overview_summary() -> dict[str, Any]:
    return dict(_overview_body()["summary"])


def _section_schedule_freshness() -> dict[str, Any]:
    from kairos.export.schedule_freshness import schedule_freshness

    return dict(schedule_freshness(_server().ROOT))


def _section_yield_totals() -> dict[str, Any]:
    from kairos_api.phase_b import _build_yield_per_second

    payload = _build_yield_per_second(_server()._load_break_schedule())
    keys = (
        "available",
        "reason",
        "currency",
        "totals",
        "revenue_net_available",
        "revenue_net_ils",
        "retention_cost_ils",
        "revenue_ils",
        "revenue_net_reason",
    )
    return {key: payload[key] for key in keys if key in payload}


def _section_recommendations() -> list[dict[str, Any]]:
    server = _server()
    rows = server._build_recommendations(server._load_break_schedule())
    return [
        {
            "title": row.get("title"),
            "title_he": row.get("title_he"),
            "severity": row.get("risk"),
            "segment": row.get("segment_id"),
            "program_type": row.get("program_type"),
            "impact_ils": row.get("impact"),
            "retention_pct": row.get("retention"),
        }
        for row in rows[:5]
    ]


def _section_settings() -> dict[str, Any]:
    settings = _server()._load_settings()
    return {
        "revenue_weight": settings.revenue_weight,
        "min_retention_floor": settings.min_retention_floor,
        "objective_mode": settings.objective_mode,
        "operator_channel": settings.operator_channel or None,
    }


def _section_counts() -> dict[str, Any]:
    import pandas as pd

    frame = _server()._load_break_schedule()
    if frame.empty:
        return {"segments": 0, "breaks": 0}
    if "segment_id" in frame.columns:
        segments = int(frame["segment_id"].nunique())
    else:
        segments = int(len(frame))
    if "num_breaks" in frame.columns:
        breaks = int(pd.to_numeric(frame["num_breaks"], errors="coerce").fillna(1).sum())
    else:
        breaks = int(len(frame))
    return {"segments": segments, "breaks": breaks}


_SECTIONS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("overview_summary", _section_overview_summary),
    ("schedule_freshness", _section_schedule_freshness),
    ("yield_totals", _section_yield_totals),
    ("recommendations", _section_recommendations),
    ("settings", _section_settings),
    ("counts", _section_counts),
)


def _compose_context(question: str) -> tuple[dict[str, Any], list[str]]:
    """Build the grounding context from the real payload builders.

    A failing section is omitted and listed in sources with an absent marker,
    never substituted or fabricated. assistant_context then adds the always-on
    per-day owned-channel table and any day_detail sections the question's dates
    resolve to, and enforces the serialized character budget.
    """
    context: dict[str, Any] = {}
    sources: list[str] = []
    for name, build in _SECTIONS:
        try:
            context[name] = build()
            sources.append(name)
        except Exception:
            sources.append(f"{name} (absent)")
    assistant_context.extend_with_day_grounding(context, sources, question)
    assistant_context.enforce_budget(context)
    return context, sources


# Basic protection: a simple in-process sliding-window rate limit.
_RATE_LOCK = threading.Lock()
_ASK_TIMES: deque[float] = deque()


def _rate_limited() -> bool:
    """True when this ask exceeds the per-minute budget (caller returns 429)."""
    moment = time.monotonic()
    with _RATE_LOCK:
        while _ASK_TIMES and moment - _ASK_TIMES[0] >= RATE_LIMIT_WINDOW_SECONDS:
            _ASK_TIMES.popleft()
        if len(_ASK_TIMES) >= RATE_LIMIT_ASKS:
            return True
        _ASK_TIMES.append(moment)
        return False


def _reset_rate_limit() -> None:
    """Test helper: clear the sliding window between test cases."""
    with _RATE_LOCK:
        _ASK_TIMES.clear()


def _describe_error(exc: Exception) -> str:
    """Honest, operator-readable description of a failed Claude call."""
    generic = f"Assistant call failed ({type(exc).__name__}): {str(exc)[:200]}"
    try:
        import anthropic
    except Exception:
        return generic
    if isinstance(exc, anthropic.AuthenticationError):
        return "The configured API key was rejected by Anthropic."
    if isinstance(exc, anthropic.RateLimitError):
        return "Anthropic rate limit reached. Try again in a minute."
    if isinstance(exc, anthropic.APITimeoutError):
        return f"The model did not answer within {int(ASK_TIMEOUT_SECONDS)} seconds."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API. Check network access."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error {exc.status_code}: {str(getattr(exc, 'message', exc))[:200]}"
    return generic


def _extract_answer(response: Any) -> str:
    parts = [
        getattr(block, "text", "")
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", "") == "text"
    ]
    return "".join(parts).strip()


def _echo_block(block: Any) -> dict[str, Any] | None:
    """One assistant content block as a plain dict, or None to drop it.

    Preserves text, tool_use, and (unchanged, in order) thinking blocks: with
    adaptive thinking on, the API requires the thinking that preceded a tool_use
    to be echoed in the assistant turn sent back with the tool results.
    """
    kind = getattr(block, "type", "")
    if kind == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if kind == "thinking":
        return {"type": "thinking", "thinking": getattr(block, "thinking", ""),
                "signature": getattr(block, "signature", "")}
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": getattr(block, "data", "")}
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name,
                "input": dict(block.input or {})}
    return None


def _run_tool_loop(client: Any, user_content: str, trace: list[dict[str, Any]],
                   items: list[dict[str, Any]], actions_on: bool) -> str:
    """One Anthropic conversation, with the tool loop when the action plane is on.

    READ tools execute immediately and their results go back to the model;
    PROPOSE tools are captured into items and never executed here. trace and
    items are caller-owned lists, so a failure mid-loop loses nothing already
    captured. The opening call stays byte-identical to a plain answer; once the
    loop iterates to search it enters goal-seek mode (adaptive thinking at a
    medium effort, temperature dropped since thinking rejects it, and a
    cache_control breakpoint on the stable tools+system prefix). Returns the
    final text answer (empty when the model gave none).
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    response = None
    for iteration in range(MAX_TOOL_ITERATIONS):
        searching = actions_on and iteration > 0
        kwargs: dict[str, Any] = {
            "model": _model_name(),
            "max_tokens": LOOP_MAX_TOKENS if actions_on else MAX_ANSWER_TOKENS,
            "system": ([{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
                       if searching else SYSTEM_PROMPT),
            "messages": messages,
        }
        if actions_on:
            kwargs["tools"] = assistant_tools.anthropic_tools()
        if searching:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": LOOP_EFFORT}
        else:
            kwargs["temperature"] = ANSWER_TEMPERATURE
        response = client.messages.create(**kwargs)
        blocks = list(getattr(response, "content", []) or [])
        tool_uses = [block for block in blocks if getattr(block, "type", "") == "tool_use"]
        if not actions_on or not tool_uses or getattr(response, "stop_reason", None) != "tool_use":
            break
        echoed = [echo for echo in (_echo_block(block) for block in blocks) if echo is not None]
        results = [assistant_tools.handle_tool_use(block, trace, items)
                   for block in tool_uses]
        messages.append({"role": "assistant", "content": echoed})
        messages.append({"role": "user", "content": results})
    return _extract_answer(response) if response is not None else ""


def _audit_ask(user: str, question: str, body: dict[str, Any], batch_id: str | None) -> None:
    assistant_actions.audit_append(
        "ask", user, model=_model_name(), question=question, batch_id=batch_id,
        results={
            "answered": bool(body.get("answer")),
            "error": body.get("error"),
            "tools": [step.get("tool") for step in body.get("tool_trace") or []],
        },
    )


# Routes
@router.get("/status")
def assistant_status() -> dict[str, Any]:
    """Honest availability: the answer path, the model, and the action plane."""
    available = bool(_api_key())
    if not _actions_enabled():
        action_reason: str | None = ACTIONS_DISABLED_REASON
    elif not available:
        action_reason = KEY_MISSING_REASON
    else:
        action_reason = None
    return {
        "available": available,
        "reason": None if available else KEY_MISSING_REASON,
        "model": _model_name(),
        "action_plane": {"enabled": action_reason is None, "reason": action_reason},
    }


@router.post("/ask")
def assistant_ask(request: AskRequest, http_request: Request) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    if _rate_limited():
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: at most {RATE_LIMIT_ASKS} questions per minute.",
        )
    generated_at = datetime.now(timezone.utc).isoformat()
    user = assistant_actions._actor(http_request)
    api_key = _api_key()
    if not api_key:
        body = {
            "available": False,
            "answer": None,
            "grounding": {"sources": [], "generated_at": generated_at},
            "error": KEY_MISSING_REASON,
            "proposals": None,
            "tool_trace": [],
        }
        _audit_ask(user, question, body, None)
        return body

    context, sources = _compose_context(question)
    grounding = {"sources": sources, "generated_at": generated_at}
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    trace: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    answer, error = "", None
    try:
        client = _client_factory(api_key)
        answer = _run_tool_loop(
            client, f"CONTEXT:\n{context_json}\n\nQUESTION:\n{question}",
            trace, items, _actions_enabled(),
        )
    except Exception as exc:  # noqa: BLE001 - every SDK failure surfaces honestly
        error = _describe_error(exc)
    # Items captured before a mid-loop failure are still real proposals: store
    # them so the operator sees exactly what was proposed, error and all.
    proposals = None
    if items:
        batch = assistant_actions.create_batch(question, items, user, _model_name())
        proposals = {key: batch[key] for key in ("batch_id", "status", "created_at", "items")}
    if error is None and not answer:
        error = "The model returned no text answer."
    body = {
        "available": True,
        "answer": answer or None,
        "grounding": grounding,
        "error": error,
        "proposals": proposals,
        "tool_trace": trace,
    }
    _audit_ask(user, question, body, proposals["batch_id"] if proposals else None)
    return body
