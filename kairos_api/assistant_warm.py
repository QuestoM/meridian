"""The model's cached prefix, written while the person is still typing.

Every ask sends the same stable prefix before the question: 39 tool schemas
(19,186 characters) and three system blocks (29,076 characters), which the API's
own usage record prices at 16,455 cached input tokens. The cache_control
breakpoint that makes it one cacheable unit already sits on the last system
block (kairos_api.assistant_prompt), so the second ask of a session pays a cache
READ for it. The first one pays the WRITE, and it pays it while a person is
watching a cursor.

**Measured on this machine, 2026-08-04, through the dock's own endpoints.** A
fresh server, ``POST /api/assistant/context/warm`` first exactly as the panel
does on mount, then one Hebrew ask streamed: the grounding stage landed at
0.28 s, the first model turn opened at 0.959 s, and the first text delta arrived
at 9.032 s. The same payload issued again with the prefix already cached
returned its first text in 1.804 s and 2.717 s. The gap is the write, and it is
the whole distance between the 2 s first-token budget in ``job-stories.md`` and
the 9,359 ms a blind critic measured on the first ask of their session.

So the panel's existing warm call now writes that prefix too. The context warm
it already did is the same idea applied to the other half of the prompt: do the
work at the moment the dock opens, which is the moment the person starts typing,
rather than at the moment they press Enter.

Three properties this file is responsible for. **It never delays the warm
response**: the model call runs on its own thread and the route returns as soon
as the context is built. **It never fires more often than the cache needs**: one
call per prefix per PREFIX_TTL_SECONDS, and after a failure not before
RETRY_SECONDS, so a rejected credential costs one call a mount rather than one
call a keystroke. **It never fabricates readiness**: the state it reports is
what the last attempt actually did, with unavailable and failed named as
themselves, and the reply itself is discarded unread.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

# Anthropic holds a written prefix for five minutes. The refresh interval sits
# inside that, so a dock that is opened again before it lapses finds it warm.
PREFIX_TTL_SECONDS = 240.0
# How soon a failed attempt may be retried. Short, because a rate limit clears
# in under a minute, and never zero, because a rejected key would otherwise
# spend one call per mount.
RETRY_SECONDS = 20.0
# The smallest call that still makes the API read the whole prefix: the write
# happens while the input is processed, so one output token is enough. The reply
# is discarded and never reaches a person or a thread.
WARM_QUESTION = "ready"
WARM_MAX_TOKENS = 1

_LOCK = threading.Lock()
_ATTEMPTED: dict[tuple[str, bool], float] = {}
_SUCCEEDED: dict[tuple[str, bool], float] = {}
# A write already running. Measured at 14.76 s on a throttled account, which is
# longer than the retry window, so without this a second mount would start a
# second write of the same prefix and pay for it twice.
_INFLIGHT: dict[tuple[str, bool], float] = {}
_LAST: dict[str, Any] = {"state": "never"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(entry: dict[str, Any]) -> None:
    with _LOCK:
        _LAST.clear()
        _LAST.update(entry)


def last_attempt() -> dict[str, Any]:
    """What the last prefix write actually did, for a caller that wants to say."""
    with _LOCK:
        return dict(_LAST)


def reset() -> None:
    """Forget every timestamp. Tests call this; nothing in the product does."""
    with _LOCK:
        _ATTEMPTED.clear()
        _SUCCEEDED.clear()
        _INFLIGHT.clear()
        _LAST.clear()
        _LAST["state"] = "never"


def _due(key: tuple[str, bool], now: float) -> bool:
    """Whether this prefix is worth writing again right now.

    Claims the write when it returns True, under the same lock that reads the
    timestamps, so two mounts arriving together produce one call and not two.
    """
    with _LOCK:
        if key in _INFLIGHT:
            return False
        wrote = _SUCCEEDED.get(key)
        if wrote is not None and now - wrote < PREFIX_TTL_SECONDS:
            return False
        tried = _ATTEMPTED.get(key)
        if tried is not None and now - tried < RETRY_SECONDS:
            return False
        _ATTEMPTED[key] = now
        _INFLIGHT[key] = now
        return True


def _write(key: tuple[str, bool], job: str | None, can_propose: bool) -> dict[str, Any]:
    """One model call whose only purpose is to leave the prefix in the cache.

    Every input is the one the next ask will send: the same tool set for this
    account's role, the same system blocks for the resolved credential and the
    declared job. A prefix written with different inputs would be a different
    prefix and the ask would still pay the write.
    """
    from kairos_api import assistant, assistant_tools

    started = time.monotonic()
    try:
        auth = assistant._resolve_auth()
        if auth is None:
            entry = {"state": "unavailable", "reason": assistant.AUTH_MISSING_REASON,
                     "at": _now_iso()}
            _record(entry)
            return entry
        try:
            client = assistant._client_from_auth(auth)
            client.messages.create(
                model=assistant._model_name(),
                max_tokens=WARM_MAX_TOKENS,
                system=assistant._system_blocks(auth_mode=getattr(auth, "mode", None), job=job),
                tools=assistant_tools.anthropic_tools(include_propose=can_propose),
                messages=[{"role": "user", "content": WARM_QUESTION}],
            )
        except Exception as exc:  # noqa: BLE001 - a failed warm is named, never hidden
            entry = {"state": "failed", "error": assistant._describe_error(exc),
                     "seconds": round(time.monotonic() - started, 3), "at": _now_iso()}
            _record(entry)
            return entry
        with _LOCK:
            _SUCCEEDED[key] = time.monotonic()
        entry = {"state": "warm", "seconds": round(time.monotonic() - started, 3),
                 "at": _now_iso()}
        _record(entry)
        return entry
    finally:
        with _LOCK:
            _INFLIGHT.pop(key, None)


def warm_prefix(*, job: str | None = None, can_propose: bool = True,
                sync: bool = False) -> dict[str, Any]:
    """Write the cached prefix for this account, unless it is already written.

    Returns what is true right now rather than a promise: ``warm`` when a write
    inside the cache's own lifetime already covers this prefix, ``warming`` when
    this call started one, ``writing`` when one started earlier is still in
    flight, ``waiting`` when the last attempt failed too recently to repeat, and
    on the synchronous path the outcome of the write itself. The background
    thread is a daemon, so a shutdown never waits for it.
    """
    key = (str(job or ""), bool(can_propose))
    now = time.monotonic()
    with _LOCK:
        wrote = _SUCCEEDED.get(key)
        running = _INFLIGHT.get(key)
    if wrote is not None and now - wrote < PREFIX_TTL_SECONDS:
        return {"state": "warm", "age_seconds": round(now - wrote, 3)}
    if running is not None:
        return {"state": "writing", "started_seconds_ago": round(now - running, 3)}
    if not _due(key, now):
        held = last_attempt()
        entry: dict[str, Any] = {"state": "waiting"}
        if held.get("state") in ("failed", "unavailable"):
            entry["last"] = held.get("state")
            for field in ("error", "reason"):
                if held.get(field):
                    entry[field] = held[field]
        return entry
    if sync:
        return _write(key, job, can_propose)
    threading.Thread(target=_write, args=(key, job, can_propose),
                     name="kairos-assistant-warm", daemon=True).start()
    return {"state": "warming"}
