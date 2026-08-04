"""One model call: how it is issued, how its blocks are read, how it fails.

Split out of kairos_api.assistant_pipeline so that module stays under the
file-size cap. Nothing here changed on the move except its address:
kairos_api.assistant_pipeline re-exports all four names and
kairos_api.assistant re-exports them again as ``_describe_error``,
``_extract_answer``, ``_echo_block`` and ``_call_model``, so an importer, a test
and a monkeypatch all reach the same objects they reached before.

This is the seam between the product and the Anthropic SDK, and it is the only
place in the assistant that knows the SDK's exception types or its streaming
protocol. The loop above it decides what to ask for; this decides how to ask.
"""

from __future__ import annotations

from typing import Any, Callable


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
