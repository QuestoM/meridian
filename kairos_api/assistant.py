"""Kai, the in-product assistant, grounded in the saved Kairos payloads.

The ask endpoint composes context from the SAME dashboard builders plus the
day-level grounding in kairos_api.assistant_context, under a character budget
with honest truncation flags. With credentials the question runs an Anthropic
tool-use loop: READ tools (including simulate_settings_change, a side-effect-free
owned-channel what-if) execute immediately and are stamped with a provenance
source; PROPOSE tools are captured as pending items
(kairos_api.assistant_actions) that only an explicit approval applies, each
settings change carrying its simulated effect. The loop searches with adaptive
thinking so a stated goal can be met by trying settings against the real
optimizer. The response carries the grounding manifest, the tool trace (with
sources) and the proposal batch; a failed section is listed absent, never
fabricated. Without credentials both routes report available false honestly.

Four seams live in sibling modules so this one stays under the file-size cap:
kairos_api.assistant_prompt holds the grounding contract and the system blocks,
kairos_api.assistant_sections holds the context sections and their cache,
kairos_api.assistant_restore holds the restore-point reader, and
kairos_api.assistant_pipeline holds the tool loop and the ask body.

Every ask carries a deadline. A run that reaches it stops the loop and answers
with what it has, naming the stop, because the measured failure this replaced
was a browser that showed "preparing an answer" for 499 s with no reply, no
error and no way out.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos_api import (
    assistant_actions,
    assistant_pipeline,
    assistant_prompt,
    assistant_saved_entry,
    assistant_sections,
    assistant_warm,
)

router = APIRouter(prefix="/api/assistant", tags=["assistant"])
router.include_router(assistant_actions.router)

# The current Opus, and the name every assistant test in this repository pins.
#
# A rename here is never local: it moves seven test files, three of which this
# piece does not own, so a builder correctly refused to make it and raised it
# instead. The owner ruled on 2026-08-09: the assistant runs the newest Opus.
# That is the owner's call and not a builder's, because the assistant runs on the
# owner's credentials and a model change moves the owner's bill.
#
# The env override below still moves it without touching code or tests, which is
# how an operator pins an older model for a run without a commit.
DEFAULT_MODEL = "claude-opus-5"
MODEL_ENV = "KAIROS_ASSISTANT_MODEL"
KEY_ENVS = ("ANTHROPIC_API_KEY", "KAIROS_ASSISTANT_API_KEY")
KEY_MISSING_REASON = "API key not configured"
AUTH_MISSING_REASON = (
    "No Anthropic credentials: Claude Code OAuth (Max) not available and no API key configured"
)
ACTIONS_ENV = "KAIROS_ASSISTANT_ACTIONS"
ACTIONS_DISABLED_REASON = f"disabled by {ACTIONS_ENV}"
ASK_TIMEOUT_SECONDS = 120.0  # per-request; adaptive-thinking search calls run long
# The whole ask, end to end. A goal-seek that keeps finding one more thing to
# simulate must still end, and the person must be told it ended rather than left
# watching a spinner. Overridable so a long research question can be given room.
ASK_DEADLINE_ENV = "KAIROS_ASSISTANT_DEADLINE_SECONDS"
ASK_DEADLINE_SECONDS = 150.0
RATE_LIMIT_ASKS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0

# Re-exported so existing importers and tests keep their entry points after the
# split. The definitions live in kairos_api.assistant_prompt,
# kairos_api.assistant_sections and kairos_api.assistant_pipeline.
SYSTEM_PROMPT = assistant_prompt.SYSTEM_PROMPT
HANDBOOK_PATH = assistant_prompt.HANDBOOK_PATH
_HANDBOOK_CACHE = assistant_prompt._HANDBOOK_CACHE  # the same dict, so a test that clears it still clears it
DEADLINE_NOTE = assistant_pipeline.DEADLINE_NOTE
CEILING_NOTE = assistant_pipeline.CEILING_NOTE
MAX_ANSWER_TOKENS = assistant_pipeline.MAX_ANSWER_TOKENS
LOOP_MAX_TOKENS = assistant_pipeline.LOOP_MAX_TOKENS
SEARCH_MAX_TOKENS = assistant_pipeline.SEARCH_MAX_TOKENS
MAX_TOOL_ITERATIONS = assistant_pipeline.MAX_TOOL_ITERATIONS
ANSWER_TEMPERATURE = assistant_pipeline.ANSWER_TEMPERATURE
LOOP_EFFORT = assistant_pipeline.LOOP_EFFORT
_describe_error = assistant_pipeline.describe_error
_extract_answer = assistant_pipeline.extract_answer
_echo_block = assistant_pipeline.echo_block
_call_model = assistant_pipeline.call_model
_run_tool_loop = assistant_pipeline.run_tool_loop
_ask_body = assistant_pipeline.ask_body
_SECTIONS = assistant_sections._SECTIONS
_section_overview_summary = assistant_sections._section_overview_summary
_section_schedule_freshness = assistant_sections._section_schedule_freshness
_section_yield_totals = assistant_sections._section_yield_totals
_section_recommendations = assistant_sections._section_recommendations
_section_settings = assistant_sections._section_settings
_section_counts = assistant_sections._section_counts


def _handbook_text() -> str | None:
    """The handbook this module points at. HANDBOOK_PATH is read at call time so
    it stays the one seam a caller or a test redirects."""
    return assistant_prompt.read_handbook(HANDBOOK_PATH)


def _system_blocks(*, auth_mode: str | None = None, job: str | None = None) -> list[dict[str, Any]]:
    return assistant_prompt.system_blocks(auth_mode=auth_mode, job=job,
                                          handbook=_handbook_text())


def _compose_context(question: str, page_context: dict[str, Any] | None = None,
                     user: str | None = None) -> tuple[dict[str, Any], list[str]]:
    return assistant_sections.compose_context(question, page_context, user)


def _compose(question: str, page_context: dict[str, Any] | None,
             user: str | None) -> tuple[dict[str, Any], list[str]]:
    """Call the composer with as many arguments as it accepts.

    The acting account is always passed to the real composer, because the model
    disclosure wall reads it and a channel account that skipped it would ground
    on training content. Several suites replace this module's _compose_context
    with a one-argument stub, so the arity is inspected rather than assumed
    instead of being guessed from whether a page context happens to be present.
    """
    import inspect

    compose = _compose_context
    try:
        positional = [
            parameter for parameter in inspect.signature(compose).parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        arity = len(positional)
    except (TypeError, ValueError):
        arity = 3
    return compose(*(question, page_context, user)[: max(1, min(arity, 3))])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # The conversation this ask belongs to. Absent selects the caller's newest
    # conversation; an unknown or invalid id mints a fresh one. The response
    # carries the resolved id back either way.
    conversation_id: str | None = Field(default=None, max_length=80)
    # Where the operator is in the dashboard (frozen contract: view, label,
    # optional entity {type,id,label}). Advisory grounding only: absent or
    # invalid degrades to exactly the behavior without it, and the response
    # shape never changes.
    page_context: dict[str, Any] | None = None
    # What the operator POINTED AT in the question itself: a list of
    # {type, id, label} references from the composer's picker. Advisory in
    # exactly the sense page_context is advisory, and additive in the sense that
    # matters most here: the prose still carries the label, so an absent or
    # invalid list degrades to the same question typed by hand.
    mentions: list[dict[str, Any]] | None = None


def _resolve_auth() -> Any:
    """Module seam: credentials for the assistant (OAuth Max or API key)."""
    from kairos_api.assistant_auth import resolve_auth

    return resolve_auth()


def _api_key() -> str | None:
    """Legacy seam used by tests: any usable credential token, or None."""
    auth = _resolve_auth()
    return auth.token if auth is not None else None


def _model_name() -> str:
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL


def _deadline_seconds() -> float:
    raw = os.environ.get(ASK_DEADLINE_ENV, "").strip()
    try:
        value = float(raw)
    except ValueError:
        return ASK_DEADLINE_SECONDS
    return value if value > 0 else ASK_DEADLINE_SECONDS


def _actions_enabled() -> bool:
    """The action plane (tool loop + proposals) is on unless explicitly disabled."""
    return os.environ.get(ACTIONS_ENV, "").strip().lower() not in {"0", "false", "no", "off"}


def _client_factory(api_key: str) -> Any:
    """Build an API-key Anthropic client. Module-level seam so tests can mock it."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key, timeout=ASK_TIMEOUT_SECONDS, max_retries=1)


def _client_from_auth(auth: Any) -> Any:
    """Build a client for the resolved auth mode (OAuth Bearer or API key)."""
    from kairos_api.assistant_auth import build_client

    if getattr(auth, "mode", None) == "api_key":
        # Keep the testable seam for the pay-as-you-go path.
        return _client_factory(auth.token)
    return build_client(auth, timeout=ASK_TIMEOUT_SECONDS, max_retries=1)


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


def _session(http_request: Any) -> dict[str, Any] | None:
    from kairos_api import auth

    if not auth.auth_active() or http_request is None:
        return None
    return auth._session_from_request(http_request)


def _can_propose(http_request: Any) -> bool:
    """Whether the caller's role may generate proposals. The assistant's
    capability surface is capped at the account's own access level: a viewer
    gets the read-only toolset and is never offered a propose tool."""
    from kairos_api import auth

    if not auth.auth_active():
        return True
    session = _session(http_request)
    return bool(session) and session.get("role") in auth.WRITE_ROLES


def _caller_job(http_request: Any) -> str | None:
    """The job the account declared for itself, or None. It changes how Kai
    addresses the person and nothing else, so an unset job is silent."""
    session = _session(http_request) or {}
    job = str(session.get("job") or "").strip()
    return job or None


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
    auth = _resolve_auth()
    available = auth is not None
    if not _actions_enabled():
        action_reason: str | None = ACTIONS_DISABLED_REASON
    elif not available:
        action_reason = AUTH_MISSING_REASON
    else:
        action_reason = None
    body: dict[str, Any] = {
        "available": available,
        "reason": None if available else AUTH_MISSING_REASON,
        "model": _model_name(),
        "action_plane": {"enabled": action_reason is None, "reason": action_reason},
    }
    if auth is not None:
        body["auth"] = auth.public_status()
    return body


@router.post("/context/warm")
def assistant_warm_context(http_request: Request) -> dict[str, Any]:
    """Build the grounding context now, so the next ask does not pay for it.

    Measured in a fresh process: the base sections cost 11.13 s cold and 0.034 s
    warm. The dock calls this when it opens, which is the moment the person
    starts typing, so the wait lands where nobody is watching a cursor.

    The model's cached prefix is the other half of the same idea and is written
    the same way, on its own thread so this route stays as fast as it was.
    Measured on the first ask of a session: 9.032 s to the first token with the
    prefix cold, 1.804 s with it written (kairos_api.assistant_warm). The tool
    set and the system blocks it writes are this account's own, or it would
    write a prefix the next ask does not send.
    """
    started = time.monotonic()
    outcome = assistant_sections.warm()
    outcome["elapsed_seconds"] = round(time.monotonic() - started, 3)
    outcome["model_prefix"] = assistant_warm.warm_prefix(
        job=_caller_job(http_request), can_propose=_can_propose(http_request))
    return outcome


def _actor_name(http_request: Request | None) -> str:
    return assistant_actions._actor(http_request)


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
    user = _actor_name(http_request)
    started = time.monotonic()
    body = _ask_body(question, http_request, conversation_id=request.conversation_id,
                     page_context=request.page_context, mentions=request.mentions)
    batch_id = body["proposals"]["batch_id"] if body.get("proposals") else None
    _audit_ask(user, question, body, batch_id)
    if body.get("answer"):
        assistant_memory.append_entry(user, question, str(body["answer"]), batch_id,
                                      conversation_id=body.get("conversation_id"),
                                      metadata=assistant_saved_entry.from_ask(
                                          body, time.monotonic() - started))
    return body


# Mounted last: assistant_stream reaches back into this module at call time for
# the shared pipeline, so the includes sit below every definition it needs.
from kairos_api import (  # noqa: E402
    assistant_conversations,
    assistant_conversations_api,
    assistant_memory,
    assistant_mentions,
    assistant_mentions_children,
    assistant_restore,
    assistant_stream,
    assistant_uploads,
)

router.include_router(assistant_stream.router)
router.include_router(assistant_memory.router)
router.include_router(assistant_mentions.router)
router.include_router(assistant_mentions_children.router)
router.include_router(assistant_conversations_api.router)
router.include_router(assistant_uploads.router)
router.include_router(assistant_restore.router)

# The calendar-event and agency proposal kinds register their appliers, restore
# state files and version-timeline mapping here, once, when the assistant (and
# with it the action-plane router this module mounts) loads; every HTTP apply
# therefore runs with the full kind registry.
from kairos_api import assistant_propose_extra  # noqa: E402

assistant_propose_extra.register_action_plane()
