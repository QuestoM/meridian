"""The ask pipeline: one model conversation, one honest body, one clock.

Split out of kairos_api.assistant so that module stays under the file-size cap.
Nothing here changed on the move except its address; kairos_api.assistant
re-exports every name below, so an importer, a test and a monkeypatch all reach
the same objects they reached before.

What lives here is the part of an ask that is the same for both routes: the
tool loop (READ tools execute and their results go back to the model, PROPOSE
tools are captured and never executed), the streaming and non-streaming model
call, and the body shape both /ask and /ask/stream return. The credential seam,
the rate limit, the context composer and the routes stay in kairos_api.assistant
because tests replace them there by name.

Three properties this file is responsible for, all of them measured failures
before they existed. **Every ask carries a deadline**, checked between turns,
because the browser once sat on "preparing an answer" for 499 s with no reply,
no error and no way out. **Every stage of the run is announced as it happens**,
so a streaming caller can paint what is true right now: the conversation it
landed in, the grounding it read with the scope of that grounding, each model
turn, and the stop when a limit ends the search. **No answer is ever protocol
text**: a search that ends at the turn ceiling, or a turn that writes a tool
call into the text channel, gets one final tool-free call for a plain answer,
and when even that returns nothing the reply names the limit it stopped on
instead of printing a call nobody can read.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from kairos_api import (
    assistant_actions,
    assistant_conversations,
    assistant_history,
    assistant_protocol_text as protocol_text,
    assistant_sections,
    assistant_tools,
)

MAX_ANSWER_TOKENS = 2000  # plain Q&A call when the action plane is off
LOOP_MAX_TOKENS = 4000
SEARCH_MAX_TOKENS = 12000  # thinking tokens count inside max_tokens on search calls
# The goal-seeker searches by calling simulate_settings_change repeatedly, so the
# loop needs room to try several settings before it converges; the ceiling stays
# hard so a runaway conversation still terminates.
MAX_TOOL_ITERATIONS = 12
ANSWER_TEMPERATURE = 0.2
LOOP_EFFORT = "medium"
DEADLINE_NOTE = (
    "The answer stopped at the time limit and reports what it had reached by then."
)
# The other limit a search can end on, named as plainly as the time limit is.
CEILING_NOTE = (
    f"The search stopped at its limit of {MAX_TOOL_ITERATIONS} model turns "
    "and reports what it had reached by then."
)
PROTOCOL_NOTE = (
    "The model wrote a tool call instead of an answer, and the plain retry returned nothing. "
    "Nothing was changed."
)
# The instruction that ends a search. It rides as its own system block on the
# one call that has no tools on it, so the model cannot answer with another
# call, and it says plainly that a proposal cannot be captured on this turn:
# an answer that promises an approval nobody can see would be worse than none.
FINAL_TURN_INSTRUCTION = (
    "This is the final turn of this answer and no tool is available on it. "
    "Answer now, in the language of the question, using only the context and the tool results already in this conversation. "
    "Never write a tool call, an invoke tag or any other protocol text in an answer: it reaches nothing and the person sees it. "
    "If a change was worth proposing, say which change and say plainly that nothing was recorded for approval, so the person can ask for it in one more question."
)


def describe_error(exc: Exception) -> str:
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
        from kairos_api import assistant

        return f"The model did not answer within {int(assistant.ASK_TIMEOUT_SECONDS)} seconds."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API. Check network access."
    if isinstance(exc, (anthropic.BadRequestError, anthropic.PermissionDeniedError)):
        message = str(getattr(exc, "message", None) or exc).lower()
        if "credit" in message or "billing" in message:
            return "The Anthropic account has no credit. Top up at console.anthropic.com (Plans and Billing). אין קרדיט בחשבון Anthropic; יש לטעון יתרה ולנסות שוב."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error {exc.status_code}: {str(getattr(exc, 'message', exc))[:200]}"
    return generic


def extract_answer(response: Any) -> str:
    parts = [
        getattr(block, "text", "")
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", "") == "text"
    ]
    return "".join(parts).strip()


def echo_block(block: Any) -> dict[str, Any] | None:
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


def call_model(client: Any, kwargs: dict[str, Any],
               on_text: Callable[[str], None] | None) -> Any:
    """One model call. A non-streaming caller (on_text None) gets
    client.messages.create untouched, byte-identical to before. A streaming
    caller gets real text deltas through client.messages.stream when the client
    supports it; otherwise the turn's answer is emitted as a single delta, so
    mocked clients stay simple while production streaming stays real."""
    stream_fn = getattr(getattr(client, "messages", None), "stream", None)
    if on_text is not None and callable(stream_fn):
        with stream_fn(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    on_text(text)
            return stream.get_final_message()
    response = client.messages.create(**kwargs)
    if on_text is not None:
        text = extract_answer(response)
        if text:
            on_text(text)
    return response


def plain_answer(client: Any, messages: list[dict[str, Any]],
                 auth_mode: str | None, job: str | None,
                 on_text: Callable[[str], None] | None) -> str:
    """One last model call, with no tools on it, for a plain answer.

    Same conversation, same system contract, plus FINAL_TURN_INSTRUCTION as its
    own block: everything the search gathered is already in ``messages``, so
    this turns it into the sentence a person reads. No tools are sent, so the
    model cannot answer with another call, and whatever it does return is cut
    of protocol text before it is handed back.
    """
    from kairos_api import assistant

    kwargs: dict[str, Any] = {
        "model": assistant._model_name(),
        "max_tokens": LOOP_MAX_TOKENS,
        "system": [*assistant._system_blocks(auth_mode=auth_mode, job=job),
                   {"type": "text", "text": FINAL_TURN_INSTRUCTION}],
        "messages": messages,
    }
    gate = protocol_text.LiveTextGate(on_text) if on_text is not None else None
    response = call_model(client, kwargs, gate.feed if gate is not None else None)
    if gate is not None:
        gate.flush()
    return protocol_text.strip_tool_protocol(extract_answer(response))


def run_tool_loop(client: Any, user_content: str, trace: list[dict[str, Any]],
                  items: list[dict[str, Any]], actions_on: bool,
                  can_propose: bool = True,
                  on_step: Callable[[dict[str, Any]], None] | None = None,
                  on_text: Callable[[str], None] | None = None,
                  user: str | None = None,
                  history: list[dict[str, Any]] | None = None,
                  auth_mode: str | None = None,
                  job: str | None = None,
                  deadline: float | None = None,
                  on_stage: Callable[[str, dict[str, Any]], None] | None = None,
                  outcome: dict[str, Any] | None = None) -> tuple[str, bool]:
    """One Anthropic conversation, with the tool loop when the action plane is on.

    READ tools execute immediately and their results go back to the model;
    PROPOSE tools are captured into items and never executed here. trace and
    items are caller-owned lists, so a failure mid-loop loses nothing already
    captured. The opening call stays byte-identical to a plain answer; once the
    loop iterates to search it enters goal-seek mode (adaptive thinking at a
    medium effort and a cache_control breakpoint on the stable tools+system
    prefix). A streaming caller receives each new trace step through on_step
    right after its tool result and assistant text through on_text as it is
    produced. history is the caller's replayed thread, placed BEFORE the current
    CONTEXT+QUESTION message so a follow-up question has its anchor.

    Returns ``(answer, stopped_at_deadline)``. The deadline is checked between
    turns, which is where the loop can actually stop without discarding a call
    already paid for. ``outcome`` is a caller-owned dict, filled with how the
    run ended (turns, ceiling, protocol_text, recovered, recovery_error), the
    same way trace and items are caller-owned: the two-value return is the
    frozen shape every existing caller unpacks.
    """
    from kairos_api import assistant

    messages: list[dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": user_content})
    response = None
    stopped = False
    ceiling = False
    turns = 0
    for iteration in range(MAX_TOOL_ITERATIONS):
        if deadline is not None and time.monotonic() > deadline and response is not None:
            stopped = True
            break
        searching = actions_on and iteration > 0
        if on_stage is not None:
            on_stage("thinking", {"turn": iteration + 1,
                                  "searching": bool(searching)})
        kwargs: dict[str, Any] = {
            "model": assistant._model_name(),
            "max_tokens": (SEARCH_MAX_TOKENS if searching else LOOP_MAX_TOKENS) if actions_on else MAX_ANSWER_TOKENS,
            "system": assistant._system_blocks(auth_mode=auth_mode, job=job),
            "messages": messages,
        }
        if actions_on:
            kwargs["tools"] = assistant_tools.anthropic_tools(include_propose=can_propose)
        if searching:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": LOOP_EFFORT}
        # temperature omitted: newer Claude models reject it as deprecated, and
        # adaptive-thinking turns already forbid it.
        # The gate is per turn: text streams as it is produced, and a turn that
        # turns into a tool call written as text stops painting at the tag.
        gate = protocol_text.LiveTextGate(on_text) if on_text is not None else None
        response = call_model(client, kwargs, gate.feed if gate is not None else None)
        if gate is not None:
            gate.flush()
        turns += 1
        blocks = list(getattr(response, "content", []) or [])
        tool_uses = [block for block in blocks if getattr(block, "type", "") == "tool_use"]
        if not actions_on or not tool_uses or getattr(response, "stop_reason", None) != "tool_use":
            break
        echoed = [echo for echo in (echo_block(block) for block in blocks) if echo is not None]
        results = []
        for block in tool_uses:
            before = len(trace)
            results.append(assistant_tools.handle_tool_use(block, trace, items,
                                                           propose_allowed=can_propose, user=user))
            if on_step is not None:
                for step in trace[before:]:
                    on_step(step)
        messages.append({"role": "assistant", "content": echoed})
        messages.append({"role": "user", "content": results})
    else:
        # Left by the ceiling rather than by an answer: the model still wanted
        # another tool when its last turn was spent.
        ceiling = True
    raw = extract_answer(response) if response is not None else ""
    leaked = protocol_text.looks_like_tool_protocol(raw)
    answer = protocol_text.strip_tool_protocol(raw)
    # A turn that wrote a tool call into the text channel is not an answer, and
    # a search cut off mid-sentence at the turn ceiling is not one either. One
    # more call, with no tools on it, turns what the search already gathered
    # into the sentence a person reads. It is skipped when the deadline stopped
    # the run, whose whole job is to stop the work, and when the last turn said
    # nothing at all, because then there is no half-written turn to finish.
    recovery_error: str | None = None
    if response is not None and not stopped and (leaked or (ceiling and raw.strip())):
        try:
            answer = plain_answer(client, messages, auth_mode, job, on_text)
        except Exception as exc:  # noqa: BLE001 - a failed retry is named, never hidden
            answer, recovery_error = "", describe_error(exc)
    if outcome is not None:
        outcome.update({"turns": turns, "ceiling": ceiling, "protocol_text": leaked,
                        "recovered": bool(answer) and (leaked or ceiling),
                        "recovery_error": recovery_error})
    return answer, stopped


def grounding_facts(context: dict[str, Any]) -> dict[str, Any]:
    """The scope the answer is grounded on, for the person rather than the model.

    Every value is copied from the context that was just composed, so this
    discloses what Kai actually read and can never be a guess. A missing section
    contributes no key at all: the run trace prints what is there and says
    nothing about what is not. It is emitted with the ``grounded`` stage, which
    lands within a fifth of a second, so the browser shows a true fact about
    this question long before the model's first word.
    """
    facts: dict[str, Any] = {}
    summary = context.get("overview_summary")
    if isinstance(summary, dict):
        for source, target in (("scope_channel", "channel"), ("date_from", "date_from"),
                               ("date_to", "date_to"), ("n_dates", "dates"),
                               ("total_breaks", "breaks")):
            if summary.get(source) not in (None, ""):
                facts[target] = summary[source]
    counts = context.get("counts")
    if isinstance(counts, dict) and counts.get("scope_channel"):
        facts["channel"] = counts["scope_channel"]
    # A count with no channel is a count over every channel in the file, so the
    # scope is stated as what it is rather than left off. Section 3.4's rule
    # applies to the grounding line exactly as it applies to a figure on a page.
    if not facts.get("channel"):
        facts.pop("breaks", None)
        if isinstance(counts, dict) and counts.get("reason"):
            facts["scope_reason"] = counts["reason"]
            # The English reason is the record an API reader gets; the code
            # beside it is what the Hebrew surface says the reason from.
            code = assistant_sections.SCOPE_REASON_CODES.get(str(counts["reason"]))
            if code:
                facts["scope_reason_code"] = code
    freshness = context.get("schedule_freshness")
    if isinstance(freshness, dict) and freshness.get("status"):
        facts["plan_status"] = freshness["status"]
    return facts


def ask_body(question: str, http_request: Any,
             on_step: Callable[[dict[str, Any]], None] | None = None,
             on_text: Callable[[str], None] | None = None,
             conversation_id: str | None = None,
             page_context: dict[str, Any] | None = None,
             on_stage: Callable[[str, dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """The full ask pipeline shared by /ask and /ask/stream.

    Composes the grounding context, replays the caller's own saved thread as
    history turns before the current message (assistant_history), runs the tool
    loop (optionally emitting stage, step and text-delta events for the
    streaming route), stores any captured proposals, and shapes the response
    body. Auditing and thread-append are the caller's job, exactly once per ask.

    The body keys are the frozen ask contract and this function does not add to
    them. A run that stopped at its deadline says so through the ``error`` field
    the contract already carries, and, for a streaming caller, through a
    ``deadline`` stage frame that the dock renders as a notice rather than a
    failure.
    """
    from kairos_api import assistant

    generated_at = datetime.now(timezone.utc).isoformat()
    deadline = time.monotonic() + assistant._deadline_seconds()
    auth = assistant._resolve_auth()
    if auth is None:
        grounding = {"sources": [], "generated_at": generated_at}
        return {
            "available": False,
            "answer": None,
            "model": assistant._model_name(),
            "grounding": grounding,
            "context_disclosure": grounding,
            "truncated": False,
            "error": assistant.AUTH_MISSING_REASON,
            "proposals": None,
            "tool_trace": [],
            # No conversation is engaged or minted for an unavailable ask.
            "conversation_id": None,
        }

    user = assistant._actor_name(http_request)
    # The conversation this ask lands in: the requested one when it exists, the
    # caller's newest when none was named, a fresh mint otherwise. History is
    # scoped to it so parallel conversations never cross-contaminate.
    conversation_id = assistant_conversations.resolve_for_ask(user, conversation_id)
    if on_stage is not None:
        on_stage("reading", {"conversation_id": conversation_id})
    context, sources = assistant._compose(question, page_context, user)
    grounding = {"sources": sources, "generated_at": generated_at}
    if on_stage is not None:
        on_stage("grounded", {"sections": len(sources), "facts": grounding_facts(context)})
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    trace: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    answer, error, stopped = "", None, False
    # The caller's own saved conversation, replayed before the current turn so
    # follow-up questions have an anchor. A memory read failure yields an
    # empty history, never a failed ask.
    history = assistant_history.history_messages(user, conversation_id)
    outcome: dict[str, Any] = {}
    try:
        client = assistant._client_from_auth(auth)
        answer, stopped = run_tool_loop(
            client, f"CONTEXT:\n{context_json}\n\nQUESTION:\n{question}",
            trace, items, assistant._actions_enabled(),
            can_propose=assistant._can_propose(http_request),
            on_step=on_step, on_text=on_text,
            user=user,
            history=history,
            auth_mode=getattr(auth, "mode", None),
            job=assistant._caller_job(http_request),
            deadline=deadline,
            on_stage=on_stage,
            outcome=outcome,
        )
    except Exception as exc:  # noqa: BLE001 - every SDK failure surfaces honestly
        error = describe_error(exc)
    # Items captured before a mid-loop failure are still real proposals: store
    # them so the operator sees exactly what was proposed, error and all.
    proposals = None
    if items:
        batch = assistant_actions.create_batch(question, items, user, assistant._model_name(),
                                               conversation_id=conversation_id)
        proposals = {key: batch[key] for key in ("batch_id", "status", "created_at", "items")}
    # A run that hit a limit says which limit, in the field the contract already
    # has, rather than growing a key. The answer it did reach is kept beside it.
    # The limits come first because a limit is the reason and an empty answer is
    # only the symptom of it.
    if error is None and outcome.get("recovery_error"):
        error = str(outcome["recovery_error"])
    if error is None and outcome.get("ceiling"):
        error = CEILING_NOTE
    if error is None and stopped:
        error = DEADLINE_NOTE
    if error is None and outcome.get("protocol_text") and not answer:
        error = PROTOCOL_NOTE
    if error is None and not answer:
        error = "The model returned no text answer."
    if stopped and on_stage is not None:
        on_stage("deadline", {"note": DEADLINE_NOTE})
    if outcome.get("ceiling") and on_stage is not None:
        on_stage("ceiling", {"note": CEILING_NOTE, "turns": outcome.get("turns")})
    return {
        "available": True,
        "answer": answer or None,
        "model": assistant._model_name(),
        "grounding": grounding,
        "context_disclosure": grounding,
        "truncated": bool(context.get("day_detail_truncated")),
        "error": error,
        "proposals": proposals,
        "tool_trace": trace,
        "conversation_id": conversation_id,
    }
