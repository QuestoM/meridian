"""Contract tests for the in-product AI assistant endpoints.

The Claude call is mocked at the module seam (assistant._client_factory):
mocking the LLM is acceptable here because the contract under test is OUR
composition and honesty layer (grounding sources, the grounding-only system
prompt, honest no-key and error paths, rate limiting), not Claude itself.
Context composition still runs the REAL kairos_api.server builders on the
repository's saved data, so the grounded sections reflect genuine payloads.

Runs against a mini FastAPI app that mounts only the assistant router, so no
live server on :8000 is ever touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch):
    """No ambient API keys, a known default model, and a fresh rate window."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


def _recording_factory(recorder: dict[str, Any], answer: str = "grounded answer"):
    """A fake Anthropic client factory that records the messages.create call."""

    def factory(api_key: str) -> Any:
        recorder["api_key"] = api_key

        def create(**kwargs: Any) -> Any:
            recorder["kwargs"] = kwargs
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=answer)])

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


# --- status ------------------------------------------------------------------
def test_status_honest_without_key(client: TestClient) -> None:
    response = client.get("/api/assistant/status")
    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "reason": "API key not configured",
        "model": "claude-sonnet-4-6",
    }


def test_status_reports_key_and_model_override(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setenv("KAIROS_ASSISTANT_MODEL", "claude-opus-4-8")
    body = client.get("/api/assistant/status").json()
    assert body == {"available": True, "reason": None, "model": "claude-opus-4-8"}


# --- ask: honest no-key path --------------------------------------------------
def test_ask_without_key_is_honest(client: TestClient) -> None:
    response = client.post("/api/assistant/ask", json={"question": "כמה ברייקים יש בתוכנית השבועית?"})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["answer"] is None
    assert body["error"] == "API key not configured"
    assert body["grounding"]["sources"] == []
    assert body["grounding"]["generated_at"]


# --- ask: input protection ----------------------------------------------------
def test_ask_rejects_empty_and_oversize_questions(client: TestClient) -> None:
    assert client.post("/api/assistant/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/assistant/ask", json={"question": "   "}).status_code == 422
    assert client.post("/api/assistant/ask", json={"question": "x" * 2001}).status_code == 422


# --- ask: grounded composition on the real repo data --------------------------
def test_ask_composes_real_sections_and_grounding_prompt(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))

    question = "מה ההכנסה הצפויה השבוע ומה רמת הסיכון?"
    response = client.post("/api/assistant/ask", json={"question": question})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["error"] is None
    assert body["answer"] == "grounded answer"

    # Every declared section is accounted for: present by name, or marked absent.
    sources = body["grounding"]["sources"]
    declared = {name for name, _ in assistant._SECTIONS}
    assert {source.replace(" (absent)", "") for source in sources} == declared
    # On this repository's real saved data the core sections must be present.
    for expected in ("overview_summary", "yield_totals", "settings", "counts"):
        assert expected in sources

    # The Claude call carries the frozen grounding contract and the real context.
    kwargs = recorder["kwargs"]
    assert recorder["api_key"] == "test-key"
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 1000
    assert kwargs["temperature"] == 0.2
    assert "must be taken from the CONTEXT block" in kwargs["system"]
    assert "Never invent" in kwargs["system"]
    assert "never state, estimate or speculate about competitor" in kwargs["system"]

    user_text = kwargs["messages"][0]["content"]
    assert user_text.startswith("CONTEXT:\n")
    assert user_text.rstrip().endswith(question)
    context_json = user_text[len("CONTEXT:\n"):user_text.index("\n\nQUESTION:")]
    context = json.loads(context_json)
    present = [source for source in sources if not source.endswith(" (absent)")]
    assert set(context) == set(present)
    # The context numbers are the real builders' outputs, not placeholders.
    saved = assistant._server()._load_settings()
    assert context["settings"]["revenue_weight"] == saved.revenue_weight
    assert context["settings"]["min_retention_floor"] == saved.min_retention_floor
    assert context["settings"]["objective_mode"] == saved.objective_mode
    assert context["counts"]["segments"] > 0
    assert context["counts"]["breaks"] > 0
    assert set(context["overview_summary"]) >= {"total_breaks", "projected_revenue", "average_retention"}
    if "recommendations" in context and context["recommendations"]:
        assert {"title", "severity", "segment"} <= set(context["recommendations"][0])
        assert len(context["recommendations"]) <= 5


# --- ask: SDK failure surfaces as an honest error ------------------------------
def test_api_exception_surfaces_as_honest_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")

    def broken_factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            raise RuntimeError("socket exploded")

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(assistant, "_client_factory", broken_factory)
    response = client.post("/api/assistant/ask", json={"question": "How many breaks are planned?"})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["answer"] is None
    assert "socket exploded" in body["error"]
    # The grounding manifest still reports what was composed; no fake answer.
    assert body["grounding"]["sources"]


# --- ask: in-process rate limit -------------------------------------------------
def test_rate_limit_fires_at_eleventh_ask(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))
    # The limiter is the contract here; skip the heavy real composition per call.
    monkeypatch.setattr(assistant, "_compose_context", lambda: ({}, []))

    for index in range(10):
        response = client.post("/api/assistant/ask", json={"question": f"question {index}"})
        assert response.status_code == 200, f"ask {index} should pass the limiter"

    blocked = client.post("/api/assistant/ask", json={"question": "one over the budget"})
    assert blocked.status_code == 429
    assert "10" in blocked.json()["detail"]
