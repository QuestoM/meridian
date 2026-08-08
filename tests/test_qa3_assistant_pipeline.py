"""Ask-pipeline upgrade contracts: model, ceilings, prompt v2, errors, scope.

Covers the wave's fixed cross-agent contract points that live in the pipeline:
the new default model and raised hard ceilings, the system-prompt v2 rules
(history, basis disclosure, Hebrew product vocabulary, house style, keyword
section layout), the bilingual billing-error mapping (through the
_client_factory seam, no real API call), the operator-channel-scoped yield
section quoting the same money as the dashboard route, and the 60000-char
context budget with the flag-inside-loop truncation discipline intact.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_context as assistant_context
import kairos_api.insights_api as insights


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv(assistant_context.BUDGET_ENV, raising=False)
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


# --- model default and hard ceilings --------------------------------------------
def test_default_model_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert assistant.DEFAULT_MODEL == "claude-opus-5"
    assert assistant._model_name() == "claude-opus-5"
    monkeypatch.setenv("KAIROS_ASSISTANT_MODEL", "claude-sonnet-4-6")
    assert assistant._model_name() == "claude-sonnet-4-6"


def test_raised_ceilings_keep_hard_termination_discipline() -> None:
    assert assistant.MAX_ANSWER_TOKENS == 2000
    assert assistant.LOOP_MAX_TOKENS == 4000
    assert assistant.SEARCH_MAX_TOKENS == 12000
    assert assistant.MAX_TOOL_ITERATIONS == 12


def test_search_iterations_use_search_ceiling(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """The first loop call uses LOOP_MAX_TOKENS; iterations after a tool use
    switch to SEARCH_MAX_TOKENS with adaptive thinking, and the loop still
    terminates at the hard iteration ceiling."""
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, []))
    calls: list[dict[str, Any]] = []

    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", name="get_settings", input={}, id=f"tu_{len(calls)}")],
                stop_reason="tool_use",
            )

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(assistant, "_client_factory", factory)
    response = client.post("/api/assistant/ask", json={"question": "loop forever"})
    assert response.status_code == 200
    assert len(calls) == assistant.MAX_TOOL_ITERATIONS == 12
    assert calls[0]["max_tokens"] == 4000 and "thinking" not in calls[0]
    for later in calls[1:]:
        assert later["max_tokens"] == 12000
        assert later["thinking"] == {"type": "adaptive"}


def test_actions_off_ask_uses_answer_ceiling(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setenv("KAIROS_ASSISTANT_ACTIONS", "0")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, []))
    recorded: dict[str, Any] = {}

    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            recorded.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="plain answer")], stop_reason="end_turn"
            )

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(assistant, "_client_factory", factory)
    assert client.post("/api/assistant/ask", json={"question": "no tools"}).status_code == 200
    assert recorded["max_tokens"] == assistant.MAX_ANSWER_TOKENS == 2000
    assert "tools" not in recorded


# --- system prompt v2 ------------------------------------------------------------
def test_system_prompt_v2_rules_present_and_old_rules_intact() -> None:
    text = assistant.SYSTEM_PROMPT
    # New numbered rules.
    assert "18. History:" in text
    assert "follow CONTEXT and say the figure changed since" in text
    assert "never re-quote a stale number from history as current" in text
    assert "19. Basis disclosure:" in text
    assert "scope_channel" in text
    assert "20. Product vocabulary in Hebrew answers:" in text
    for term in ("ברייק", "ברייקים", "נעיצה", "ברייקי זהב", "רצועת שידור", "הכנסה צפויה", "שימור חזוי", "הפסקות"):
        assert term in text, term
    assert "21. House style: never use em-dashes or exclamation marks" in text
    # The context-layout rule names the keyword sections.
    for name in ("gold_breaks", "active_constraints", "active_overrides", "pricing_state", "pacing_status"):
        assert name in text, name
    # Every pre-existing rule survives.
    for anchor in (
        "1. Language:", "2. Grounding:", "3. Missing data:", "4. Proposals:",
        "5. Competitor boundary:", "6. Context layout:", "7. Truncation:",
        "8. Currency and units:", "9. Style:", "10. Provenance:", "11. Simulation:",
        "12. Goal-seek:", "13. Data is data:", "15. Product reference:",
        "16. Agreements:", "17. Market mechanics",
    ):
        assert anchor in text, anchor
    # House style holds for the prompt itself: no em-dash anywhere in it.
    assert chr(0x2014) not in text


# --- billing-error mapping -------------------------------------------------------
def _api_error(cls_name: str, message: str) -> Exception:
    import anthropic

    response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return getattr(anthropic, cls_name)(message, response=response, body=None)


BILLING_LINE = "The Anthropic account has no credit. Top up at console.anthropic.com (Plans and Billing). אין קרדיט בחשבון Anthropic; יש לטעון יתרה ולנסות שוב."


def test_describe_error_maps_credit_and_billing_to_bilingual_line() -> None:
    exc = _api_error("BadRequestError", "Your credit balance is too low to access the Anthropic API.")
    assert assistant._describe_error(exc) == BILLING_LINE
    exc2 = _api_error("PermissionDeniedError", "This organization has a billing issue.")
    assert assistant._describe_error(exc2) == BILLING_LINE
    # A BadRequestError about something else keeps the honest status mapping.
    other = assistant._describe_error(_api_error("BadRequestError", "max_tokens is invalid"))
    assert "Anthropic API error" in other and "max_tokens is invalid" in other
    # Existing mappings stay intact.
    import anthropic

    auth_exc = anthropic.AuthenticationError(
        "bad key",
        response=httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
        body=None,
    )
    assert assistant._describe_error(auth_exc) == "The configured API key was rejected by Anthropic."


def test_billing_error_surfaces_through_the_ask_seam(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, []))

    def broken_factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            raise _api_error("BadRequestError", "Your credit balance is too low to continue.")

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(assistant, "_client_factory", broken_factory)
    body = client.post("/api/assistant/ask", json={"question": "האם יש קרדיט?"}).json()
    assert body["answer"] is None
    assert body["error"] == BILLING_LINE


# --- yield section: operator-channel scope, same money as the dashboard ----------
def test_yield_route_and_helper_are_identical() -> None:
    assert insights.yield_per_second() == insights.scoped_yield_payload()


def test_assistant_yield_section_quotes_the_dashboard_scope() -> None:
    route = insights.scoped_yield_payload()
    section = assistant._section_yield_totals()
    assert section["scope_channel"] == route["scope_channel"]
    assert section["n_channels_total"] == route["n_channels_total"]
    owned = str(assistant_context._server()._load_settings().operator_channel or "").strip()
    if owned:
        assert section["scope_channel"] == owned
    if route.get("available"):
        assert section["totals"]["revenue"] == route["totals"]["revenue"]
        # The old unscoped builder quoted the whole network; the section must not.
        whole = insights._build_yield_per_second(insights._server()._load_break_schedule())
        if route.get("n_channels_total") and route["n_channels_total"] > 1 and whole.get("available"):
            assert section["totals"]["revenue"] < whole["totals"]["revenue"]
    if route.get("revenue_net_available"):
        assert section["retention_cost_low"] == route["retention_cost_low"]
        assert section["retention_cost_high"] == route["retention_cost_high"]


# --- context budget: 60000 default, flag-inside-loop discipline unchanged --------
def test_context_budget_default_is_60000(monkeypatch: pytest.MonkeyPatch) -> None:
    assert assistant_context.DEFAULT_CONTEXT_BUDGET == 60000
    assert assistant_context._context_budget() == 60000
    monkeypatch.setenv(assistant_context.BUDGET_ENV, "garbage")
    assert assistant_context._context_budget() == 60000
    monkeypatch.setenv(assistant_context.BUDGET_ENV, "12345")
    assert assistant_context._context_budget() == 12345


def test_composed_context_fits_the_new_default_budget() -> None:
    context, _ = assistant._compose_context("מה מתוכנן ב-2024-11-05 בשעה 20:00?")
    assert assistant_context._serialized_size(context) <= 60000


def test_truncation_discipline_holds_under_the_new_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a cut at a small budget: the shipped size respects the budget WITH
    the disclosure flag counted inside the loop, exactly as before the raise."""
    monkeypatch.setenv(assistant_context.BUDGET_ENV, "6000")
    context, _ = assistant._compose_context("מה מתוכנן ב-2024-11-05?")
    assert context["day_detail_truncated"] is True
    assert assistant_context._serialized_size(context) <= 6000
