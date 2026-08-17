"""Provider seam for the extraction pipeline: routed models, forced schemas,
measured cost.

The AI-layer contract (docs/trade/engine-design.md §4): route each task to the
smallest model that does it well, constrain every result that feeds code to a
strict schema, and keep tokens/latency visible per stage. This module is the
ONLY place the pipeline touches the Anthropic SDK, mirroring the assistant's
own seam discipline (kairos_api.assistant_model_call).

Model routing, overridable per deployment without a commit:

- ``small``  (classify, transcription checks)     default Haiku 4.5
- ``mid``    (parameterise, ordinary terms)       default Sonnet 5
- ``reason`` (hard families, cross-refs, vision)  default Opus 5, the same
  model the owner ruled the assistant runs

Credentials resolve through the assistant's own auth seam so there is exactly
one way this product signs a model call. Tests never reach a provider: the
stage functions take a ``call`` callable, and StageCaller is only built by the
live runner.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MODEL_SMALL_ENV = "KAIROS_TRADE_MODEL_SMALL"
MODEL_MID_ENV = "KAIROS_TRADE_MODEL_MID"
MODEL_REASON_ENV = "KAIROS_TRADE_MODEL_REASON"
DEFAULT_SMALL = "claude-haiku-4-5-20251001"
DEFAULT_MID = "claude-sonnet-5"
DEFAULT_REASON = "claude-opus-5"

CALL_TIMEOUT_SECONDS = 180.0

TIERS = ("small", "mid", "reason")


def model_for(tier: str) -> str:
    if tier == "small":
        return os.environ.get(MODEL_SMALL_ENV, "").strip() or DEFAULT_SMALL
    if tier == "mid":
        return os.environ.get(MODEL_MID_ENV, "").strip() or DEFAULT_MID
    if tier == "reason":
        return os.environ.get(MODEL_REASON_ENV, "").strip() or DEFAULT_REASON
    raise ValueError(f"unknown model tier {tier!r}; tiers: {TIERS}")


@dataclass
class CallRecord:
    """One measured provider call."""

    stage: str
    tier: str
    model: str
    latency_seconds: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    ok: bool
    error: str = ""


@dataclass
class RunStats:
    """Accumulated cost/latency for one extraction run, published with it."""

    calls: list[CallRecord] = field(default_factory=list)

    def record(self, record: CallRecord) -> None:
        self.calls.append(record)

    def to_payload(self) -> dict[str, Any]:
        by_stage: dict[str, dict[str, Any]] = {}
        for call in self.calls:
            bucket = by_stage.setdefault(call.stage, {
                "calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
                "latency_seconds": 0.0, "models": set(),
            })
            bucket["calls"] += 1
            bucket["failed"] += 0 if call.ok else 1
            bucket["input_tokens"] += call.input_tokens or 0
            bucket["output_tokens"] += call.output_tokens or 0
            bucket["latency_seconds"] = round(
                bucket["latency_seconds"] + call.latency_seconds, 2)
            bucket["models"].add(call.model)
        return {
            stage: {**bucket, "models": sorted(bucket["models"])}
            for stage, bucket in by_stage.items()
        }


class ProviderUnavailable(RuntimeError):
    """No credentials resolve; the caller surfaces this honestly."""


def build_client() -> Any:
    """One client, through the assistant's credential seam."""
    from kairos_api import assistant

    auth = assistant._resolve_auth()
    if auth is None:
        raise ProviderUnavailable(assistant.AUTH_MISSING_REASON)
    return assistant._client_from_auth(auth)


def _usage(response: Any) -> tuple[Optional[int], Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return (getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """The provider's own instruction, when it sends one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "anthropic-ratelimit-requests-reset"):
        raw = headers.get(key) if hasattr(headers, "get") else None
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class StageCaller:
    """The live ``call`` implementation handed to the pure stage functions.

    ``call(stage, tier, system, content, tool_name, tool_schema)`` issues one
    forced-tool call and returns the tool input dict. ``content`` is either a
    string (text prompt) or a prebuilt content-block list (vision pages).

    Rate limits are the expected failure of a long extraction run, not an
    exception: a full agreement is dozens of sequential calls on the heavy
    tiers. The retry ladder is therefore exponential and patient (it honours
    the provider's own retry-after when present), and ``pace_seconds`` puts a
    floor between calls so a run does not sprint into its own ceiling.
    A terminal failure still raises; the runner turns it into an honest
    incomplete instance rather than losing the clause.
    """

    client: Any
    stats: RunStats
    max_tokens: int = 4000
    attempts: int = 5
    backoff_seconds: tuple[float, ...] = (5.0, 15.0, 40.0, 90.0)
    pace_seconds: float = 0.6
    _last_call_at: float = field(default=0.0, repr=False)

    def call(self, stage: str, tier: str, system: str, content: Any,
             tool_name: str, tool_schema: dict[str, Any]) -> dict[str, Any]:
        model = model_for(tier)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.attempts + 1):
            gap = time.monotonic() - self._last_call_at
            if gap < self.pace_seconds:
                time.sleep(self.pace_seconds - gap)
            started = time.monotonic()
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                    tools=[{
                        "name": tool_name,
                        "description": f"Structured output for the {stage} stage.",
                        "input_schema": tool_schema,
                    }],
                    tool_choice={"type": "tool", "name": tool_name},
                )
            except Exception as exc:  # noqa: BLE001 - recorded, then decided
                self._last_call_at = time.monotonic()
                self.stats.record(CallRecord(
                    stage=stage, tier=tier, model=model,
                    latency_seconds=round(time.monotonic() - started, 2),
                    input_tokens=None, output_tokens=None, ok=False,
                    error=type(exc).__name__,
                ))
                last_error = exc
                if attempt < self.attempts and _retryable(exc):
                    index = min(attempt - 1, len(self.backoff_seconds) - 1)
                    wait = _retry_after_seconds(exc) or self.backoff_seconds[index]
                    time.sleep(min(wait, 120.0))
                    continue
                raise
            self._last_call_at = time.monotonic()
            latency = round(time.monotonic() - started, 2)
            input_tokens, output_tokens = _usage(response)
            block = next(
                (b for b in getattr(response, "content", []) or []
                 if getattr(b, "type", "") == "tool_use"),
                None,
            )
            ok = block is not None
            self.stats.record(CallRecord(
                stage=stage, tier=tier, model=model, latency_seconds=latency,
                input_tokens=input_tokens, output_tokens=output_tokens, ok=ok,
                error="" if ok else "no_tool_use_block",
            ))
            if ok:
                return dict(block.input or {})
            last_error = RuntimeError("model returned no tool_use block")
        raise last_error  # pragma: no cover - loop always returns or raises


def _retryable(exc: Exception) -> bool:
    try:
        import anthropic
    except Exception:  # pragma: no cover
        return False
    return isinstance(exc, (anthropic.RateLimitError, anthropic.APITimeoutError,
                            anthropic.APIConnectionError, anthropic.InternalServerError))


def image_block(png_bytes: bytes) -> dict[str, Any]:
    import base64

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode("ascii"),
        },
    }


def describe_error(exc: Exception) -> str:
    from kairos_api.assistant_model_call import describe_error as assistant_describe

    if isinstance(exc, ProviderUnavailable):
        return str(exc)
    return assistant_describe(exc)


def dump_stats(stats: RunStats) -> str:
    return json.dumps(stats.to_payload(), ensure_ascii=False, indent=1)
