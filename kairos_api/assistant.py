"""In-product AI assistant grounded in the saved Kairos payloads.

The ask endpoint composes a compact context from the SAME internal builders the
dashboard endpoints use (overview summary, schedule freshness verdict, yield
totals, top recommendations, saved settings essentials, plan counts), plus the
day-level grounding built in kairos_api.assistant_context: an always-included
per-day table of the operator's own channel and, when the question names a date
the saved plan contains, that day's segment detail, all kept under a serialized
character budget with honest truncation flags. The context goes to Claude with
a system prompt that forbids answering beyond it, and the response carries a
grounding manifest naming every section that was included. A section whose
builder fails is omitted and listed as absent, never fabricated. Without an API
key both routes stay up and report available false honestly instead of
guessing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kairos_api import assistant_context

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

DEFAULT_MODEL = "claude-sonnet-4-6"
MODEL_ENV = "KAIROS_ASSISTANT_MODEL"
KEY_ENVS = ("ANTHROPIC_API_KEY", "KAIROS_ASSISTANT_API_KEY")
KEY_MISSING_REASON = "API key not configured"
ASK_TIMEOUT_SECONDS = 30.0
MAX_ANSWER_TOKENS = 1000
ANSWER_TEMPERATURE = 0.2
RATE_LIMIT_ASKS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0

# The grounding contract. Every rule here is load-bearing: the assistant may
# only restate what the composed context carries, must name missing data
# instead of guessing, and never crosses the competitor-information boundary.
SYSTEM_PROMPT = (
    "You are the Kairos schedule analyst, the in-product assistant of a TV ad-break "
    "revenue optimizer. The user message contains a CONTEXT block of JSON computed "
    "from the operator's saved data, followed by the operator's QUESTION. "
    "Rules, in priority order: "
    "1. Grounding: every number, count, date, currency amount and verdict in your "
    "answer must be taken from the CONTEXT block. Never invent, estimate, extrapolate "
    "or recall figures from memory or general knowledge. "
    "2. Missing data: when the question needs data that is not present in CONTEXT, "
    "say exactly that, name the specific missing data, and stop. A source section "
    "marked absent failed to load and its data is unavailable. "
    "3. Competitor boundary: never state, estimate or speculate about competitor "
    "revenue or competitor performance. The context covers only the operator's own "
    "channel; say so when asked about competitors. Competitor channels appear in "
    "CONTEXT only as aggregate counts, never by name or by figure. "
    "4. Context layout: per_day_plan is a per-day table of the operator's own "
    "channel (date, weekday, breaks, revenue in ILS, average retention percent). "
    "When the question names a date, weekday or time found in the saved plan, "
    "day_detail sections carry that day's segments ordered by revenue, highest "
    "first; the start field is each segment's clock time, and matched_full_rows "
    "carries the complete saved fields for segments matching a time or programme "
    "type named in the question. "
    "5. Truncation: when a day_detail section carries truncated true, or the "
    "context carries day_detail_truncated true, rows were cut to fit the context "
    "budget. When your answer relies on such a section, state that it is based on "
    "a truncated list. "
    "6. Language: answer in the language of the question. When the question is in "
    "Hebrew, answer in natural Hebrew. "
    "7. Currency: monetary amounts are in ILS unless the context states otherwise. "
    "8. Style: keep answers short and concrete, plain text only, no markdown "
    "formatting. Prefer two to six sentences, or a short plain list when the "
    "operator asks for several figures."
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


def _client_factory(api_key: str) -> Any:
    """Build the Anthropic client. Module-level seam so tests can mock it."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key, timeout=ASK_TIMEOUT_SECONDS, max_retries=1)


# ---------------------------------------------------------------------------
# Context sections. Each one reuses a real dashboard builder, so the assistant
# reads exactly what the operator's own pages render, nothing else.
# ---------------------------------------------------------------------------
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
    body = _overview_body()
    return dict(body["summary"])


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

    A failing section is omitted from the context and listed in sources with an
    absent marker, so the model (and the operator) can see exactly which data
    was and was not available. Nothing is ever substituted or fabricated. After
    the base sections, assistant_context adds the always-on per-day table of
    the operator's own channel, any day_detail sections the question's dates
    resolve to, and finally enforces the serialized character budget.
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


# ---------------------------------------------------------------------------
# Basic protection: a simple in-process sliding-window rate limit.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/status")
def assistant_status() -> dict[str, Any]:
    """Honest availability: configured or not, and which model would answer."""
    if not _api_key():
        return {"available": False, "reason": KEY_MISSING_REASON, "model": _model_name()}
    return {"available": True, "reason": None, "model": _model_name()}


@router.post("/ask")
def assistant_ask(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    if _rate_limited():
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: at most {RATE_LIMIT_ASKS} questions per minute.",
        )
    generated_at = datetime.now(timezone.utc).isoformat()
    api_key = _api_key()
    if not api_key:
        return {
            "available": False,
            "answer": None,
            "grounding": {"sources": [], "generated_at": generated_at},
            "error": KEY_MISSING_REASON,
        }

    context, sources = _compose_context(question)
    grounding = {"sources": sources, "generated_at": generated_at}
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    try:
        client = _client_factory(api_key)
        response = client.messages.create(
            model=_model_name(),
            max_tokens=MAX_ANSWER_TOKENS,
            temperature=ANSWER_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_json}\n\nQUESTION:\n{question}",
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - every SDK failure surfaces honestly
        return {"available": True, "answer": None, "grounding": grounding, "error": _describe_error(exc)}

    answer = _extract_answer(response)
    if not answer:
        return {
            "available": True,
            "answer": None,
            "grounding": grounding,
            "error": "The model returned no text answer.",
        }
    return {"available": True, "answer": answer, "grounding": grounding, "error": None}
