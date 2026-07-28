"""Contract tests for the in-product AI assistant endpoints.

The Claude call is mocked at the module seam (assistant._client_factory):
mocking the LLM is acceptable here because the contract under test is OUR
composition and honesty layer (grounding sources, the grounding-only system
prompt, the day-level drill-down sections, the context budget, honest no-key
and error paths, rate limiting), not Claude itself. Context composition still
runs the REAL kairos_api.server builders on the repository's saved data, so
the grounded sections reflect genuine payloads and the day-level expectations
are derived from the weekly CSV itself.

Runs against a mini FastAPI app that mounts only the assistant router, so no
live server on :8000 is ever touched.
"""

from __future__ import annotations

import datetime
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as assistant_actions
import kairos_api.assistant_context as assistant_context


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """No ambient API keys, a known default model and budget, a fresh rate
    window, and the action-plane state (audit log) redirected to tmp."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv(assistant_context.BUDGET_ENV, raising=False)
    monkeypatch.setenv(assistant_actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
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


def _plan_facts() -> tuple[Any, str]:
    """The saved weekly CSV and the configured operator channel, read directly
    so the day-level expectations are derived from the data, not the code."""
    import pandas as pd

    server = assistant_context._server()
    frame = pd.read_csv(server.OUTPUT_DIR / "weekly_break_schedule.csv")
    owned = str(server._load_settings().operator_channel or "").strip()
    assert not frame.empty, "the repository must carry a saved weekly plan"
    assert owned, "the repository settings must configure an operator channel"
    return frame, owned


def _owned_rows(frame: Any, owned: str) -> Any:
    return frame[frame["channel"].astype(str).str.strip() == owned]


# --- status ------------------------------------------------------------------
def _system_text(kwargs):
    system = kwargs["system"]
    if isinstance(system, str):
        return system
    return "".join(block.get("text", "") for block in system)


def test_status_honest_without_key(client: TestClient) -> None:
    response = client.get("/api/assistant/status")
    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "reason": assistant.AUTH_MISSING_REASON,
        "model": "claude-opus-4-8",
        "action_plane": {"enabled": False, "reason": assistant.AUTH_MISSING_REASON},
    }


def test_status_reports_key_and_model_override(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setenv("KAIROS_ASSISTANT_MODEL", "claude-opus-4-8")
    body = client.get("/api/assistant/status").json()
    assert body == {
        "available": True,
        "reason": None,
        "model": "claude-opus-4-8",
        "action_plane": {"enabled": True, "reason": None},
        "auth": {"mode": "api_key", "source": "KAIROS_ASSISTANT_API_KEY"},
    }


# --- ask: honest no-key path --------------------------------------------------
def test_ask_without_key_is_honest(client: TestClient) -> None:
    response = client.post("/api/assistant/ask", json={"question": "כמה ברייקים יש בתוכנית השבועית?"})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["answer"] is None
    assert body["error"] == assistant.AUTH_MISSING_REASON
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
    # The question names no date, so no day_detail section may appear.
    sources = body["grounding"]["sources"]
    declared = {name for name, _ in assistant._SECTIONS} | {assistant_context.PER_DAY_SECTION}
    assert {source.replace(" (absent)", "") for source in sources} == declared
    # On this repository's real saved data the core sections must be present.
    for expected in ("overview_summary", "yield_totals", "settings", "counts", "per_day_plan"):
        assert expected in sources

    # The Claude call carries the frozen grounding contract and the real context.
    kwargs = recorder["kwargs"]
    assert recorder["api_key"] == "test-key"
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["max_tokens"] == assistant.LOOP_MAX_TOKENS  # the tool-use loop budget
    assert "temperature" not in kwargs  # omitted on every call: newer models reject it
    assert {tool["name"] for tool in kwargs["tools"]} >= {"get_settings", "propose_settings_change"}
    assert "must be taken from the CONTEXT block" in _system_text(kwargs)
    assert "Never invent" in _system_text(kwargs)
    assert "never state, estimate or speculate about competitor" in _system_text(kwargs)
    assert "per_day_plan" in _system_text(kwargs)
    assert "day_detail" in _system_text(kwargs)
    assert "truncated" in _system_text(kwargs)

    user_text = kwargs["messages"][0]["content"]
    assert user_text.startswith("CONTEXT:\n")
    assert user_text.rstrip().endswith(question)
    context_json = user_text[len("CONTEXT:\n"):user_text.index("\n\nQUESTION:")]
    context = json.loads(context_json)
    present = [source for source in sources if not source.endswith(" (absent)")]
    assert set(context) - {"day_detail_truncated"} == set(present)
    # The context numbers are the real builders' outputs, not placeholders.
    saved = assistant_context._server()._load_settings()
    assert context["settings"]["revenue_weight"] == saved.revenue_weight
    assert context["settings"]["min_retention_floor"] == saved.min_retention_floor
    assert context["settings"]["objective_mode"] == saved.objective_mode
    assert context["counts"]["segments"] > 0
    assert context["counts"]["breaks"] > 0
    assert set(context["overview_summary"]) >= {"total_breaks", "projected_revenue", "average_retention"}
    if "recommendations" in context and context["recommendations"]:
        assert {"title", "severity", "segment"} <= set(context["recommendations"][0])
        assert len(context["recommendations"]) <= 5


def test_system_prompt_carries_quarter_hour_market_mechanics() -> None:
    """The market-mechanics section is present in the constructed system prompt:
    the owner-stated quarter-hour settlement fact sourced to its doc, the
    measured status and off-by-default engine expression, the two-currencies
    caveat, the surface-when-asked instruction, and the not-contractually-
    verified hedge so the assistant does not overclaim."""
    text = "".join(block["text"] for block in assistant._system_blocks())
    assert "docs/quarter-hour-billing.md" in text
    assert "ROUND quarter hour" in text
    assert "PER SPOT" in text
    assert "straddling break" in text
    assert "Two currencies" in text
    assert "minute-level true audience" in text
    assert "round-quarter-hour averages" in text
    assert "consolidation-versus-split" in text
    # Surfaced on the four named question kinds.
    assert "placement, splitting, consolidation, or CPP revenue" in text
    # Marked owner-stated and dated, now measured and expressed off-by-default.
    assert "owner-stated market convention recorded 2026-07-07" in text
    assert "measured on the real Nov-2024 month" in text
    assert "pricing_activation.qh_settlement" in text
    assert "OFF by default" in text
    # The measured placement answer, not the open conjecture.
    assert "symmetric boundary-straddling is optimal" in text
    # Honest hedge: rule confirmed in plan data, not contractually verified.
    assert "not contractually verified" in text


# --- per-day table: always on, owned channel only ------------------------------
def test_per_day_table_present_and_owned_channel_only() -> None:
    frame, owned = _plan_facts()
    own = _owned_rows(frame, owned)
    context, sources = assistant._compose_context("How does the week look overall?")

    assert "per_day_plan" in sources
    assert not [source for source in sources if source.startswith("day_detail")]
    assert not [key for key in context if key.startswith("day_detail")]
    assert "day_detail_truncated" not in context

    table = context["per_day_plan"]
    assert table["channel"] == owned
    days = table["days"]
    assert len(days) == own["date"].astype(str).str.strip().nunique()
    assert all(set(day) == {"date", "weekday", "breaks", "revenue_ils", "avg_retention_pct"} for day in days)
    assert [day["date"] for day in days] == sorted(day["date"] for day in days)

    # A sample day's figures equal the CSV's owned-channel sums for that date.
    import pandas as pd

    sample = days[0]
    day_rows = own[own["date"].astype(str).str.strip() == sample["date"]]
    assert sample["breaks"] == int(pd.to_numeric(day_rows["num_breaks"], errors="coerce").fillna(0).sum())
    expected_revenue = int(round(float(pd.to_numeric(day_rows["predicted_revenue"], errors="coerce").fillna(0).sum())))
    assert sample["revenue_ils"] == expected_revenue
    assert table["totals"]["breaks"] == sum(day["breaks"] for day in days)


# --- day detail: an ISO date in the question pulls that day's segments ---------
def test_iso_date_question_pulls_day_detail_and_source_says_so(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(assistant, "_client_factory", _recording_factory(recorder))

    response = client.post("/api/assistant/ask", json={"question": "מה מתוכנן ב-2024-11-05?"})
    assert response.status_code == 200
    sources = response.json()["grounding"]["sources"]
    assert "day_detail 2024-11-05" in sources

    user_text = recorder["kwargs"]["messages"][0]["content"]
    context = json.loads(user_text[len("CONTEXT:\n"):user_text.index("\n\nQUESTION:")])
    section = context["day_detail 2024-11-05"]
    frame, owned = _plan_facts()
    own_day = _owned_rows(frame, owned)
    own_day = own_day[own_day["date"].astype(str).str.strip() == "2024-11-05"]
    assert section["rows_total"] == len(own_day)
    assert section["channel"] == owned
    assert len(section["segments"]) == len(own_day)
    assert all(seg["segment_id"].startswith(f"2024-11-05|{owned}|") for seg in section["segments"])
    revenues = [seg["revenue_ils"] for seg in section["segments"]]
    assert revenues == sorted(revenues, reverse=True)
    assert "day_detail_truncated" not in context


# --- conservative parsing: no date means no day section ------------------------
def test_questions_without_a_plan_date_add_no_day_detail() -> None:
    questions = (
        "כמה הכנסות צפויות השבוע?",
        "מה צפוי ב-99/99?",
        "what about 2025-11-05?",
        "compare the morning to the evening",
    )
    for question in questions:
        context, sources = assistant._compose_context(question)
        assert not [source for source in sources if source.startswith("day_detail")], question
        assert not [key for key in context if key.startswith("day_detail")], question


def test_day_month_and_hebrew_weekday_formats_resolve_to_plan_days() -> None:
    frame, owned = _plan_facts()
    own = _owned_rows(frame, owned)
    context, sources = assistant._compose_context("מה קורה ב-05/11 בערוץ שלנו?")
    assert "day_detail 2024-11-05" in sources
    assert context["day_detail 2024-11-05"]["date"] == "2024-11-05"

    tuesdays = sorted(
        {
            text
            for text in own["date"].astype(str).str.strip()
            if datetime.date.fromisoformat(text).weekday() == 1
        }
    )
    assert tuesdays, "the saved plan must contain at least one Tuesday"
    _, weekday_sources = assistant._compose_context("מה מתוכנן ביום שלישי?")
    detail_sources = [source for source in weekday_sources if source.startswith("day_detail ")]
    assert detail_sources == [f"day_detail {text}" for text in tuesdays]


# --- clock and programme-type refinement: full rows for matched segments -------
def test_clock_and_type_matches_include_full_rows() -> None:
    frame, owned = _plan_facts()
    own_day = _owned_rows(frame, owned)
    own_day = own_day[own_day["date"].astype(str).str.strip() == "2024-11-05"]
    assert len(own_day), "the saved plan must contain the owned channel on 2024-11-05"

    clock = str(own_day.iloc[0]["start_time"]).strip()
    context, _ = assistant._compose_context(f"מה משודר ב-2024-11-05 בשעה {clock}?")
    section = context["day_detail 2024-11-05"]
    assert section["match"]["clocks"] == [clock]
    full_rows = section["matched_full_rows"]
    assert full_rows
    assert all(str(row["start_time"]).strip() == clock for row in full_rows)
    # Full rows carry the complete saved record, not just the compact fields.
    assert {"base_rate", "break_type", "retention_confidence"} <= set(full_rows[0])

    types = [
        text
        for text in own_day["program_type"].dropna().astype(str).str.strip()
        if text and text.lower() != "other"
    ]
    assert types, "the owned day must carry at least one unambiguous programme type"
    target = types[0]
    context2, _ = assistant._compose_context(f"How did {target} segments perform on 2024-11-05?")
    section2 = context2["day_detail 2024-11-05"]
    assert section2["match"]["program_types"] == [target]
    assert section2["matched_full_rows"]
    assert all(str(row["program_type"]).strip() == target for row in section2["matched_full_rows"])


# --- context budget: deterministic truncation with an honest flag --------------
def test_truncation_flag_fires_on_artificially_small_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(assistant_context.BUDGET_ENV, "6000")
    context, sources = assistant._compose_context("מה מתוכנן ב-2024-11-05?")
    assert "day_detail 2024-11-05" in sources
    assert context["day_detail_truncated"] is True
    section = context["day_detail 2024-11-05"]
    assert section["truncated"] is True
    assert 0 < len(section["segments"]) < section["rows_total"]
    assert section["rows_omitted"] == section["rows_total"] - len(section["segments"])
    assert assistant_context._serialized_size(context) <= 6000
    # The survivors are the highest-revenue segments, still in descending order.
    revenues = [seg["revenue_ils"] for seg in section["segments"]]
    assert revenues == sorted(revenues, reverse=True)


def test_context_budget_env_default_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert assistant_context._context_budget() == 60000
    monkeypatch.setenv(assistant_context.BUDGET_ENV, "garbage")
    assert assistant_context._context_budget() == 60000
    monkeypatch.setenv(assistant_context.BUDGET_ENV, "-5")
    assert assistant_context._context_budget() == 60000
    monkeypatch.setenv(assistant_context.BUDGET_ENV, "12345")
    assert assistant_context._context_budget() == 12345


# --- competitor boundary: no competitor channel name in the composed context ---
def test_competitor_boundary_holds_in_composed_context() -> None:
    frame, owned = _plan_facts()
    channels = {text for text in frame["channel"].astype(str).str.strip().unique() if text}
    competitors = sorted(channels - {owned})
    assert competitors, "the saved plan must carry competitor channels for this test to bite"

    context, _ = assistant._compose_context("מה מתוכנן ב-2024-11-05 ומה סך ההכנסות?")
    serialized = json.dumps(context, ensure_ascii=False, default=str)
    for name in competitors:
        assert name not in serialized
    assert context["per_day_plan"]["channel"] == owned
    assert context["per_day_plan"]["competitor_channels_excluded"] == len(competitors)


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
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, []))

    for index in range(10):
        response = client.post("/api/assistant/ask", json={"question": f"question {index}"})
        assert response.status_code == 200, f"ask {index} should pass the limiter"

    blocked = client.post("/api/assistant/ask", json={"question": "one over the budget"})
    assert blocked.status_code == 429
    assert "10" in blocked.json()["detail"]
