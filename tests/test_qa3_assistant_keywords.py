"""Keyword-driven grounding sections in the composed assistant context.

Section builders run against the REAL repository stores (gold from the saved
schedule, constraints and overrides CSVs, the pricing hierarchy, the pacing
loader), so the assertions are about the honest shapes the real data produces:
present-on-keyword, absent-marked-on-failure, hard caps, competitor-boundary
scoping, and nothing at all without a keyword. No API key and no model call is
involved; _compose_context is pure composition.
"""

from __future__ import annotations

import json

import pytest

import kairos_api.assistant as assistant
import kairos_api.assistant_context as assistant_context
import kairos_api.assistant_keywords as keywords


# --- keyword matching: conservative, bilingual, prefix-stripped ------------------
@pytest.mark.parametrize(
    ("question", "name"),
    [
        ("כמה ברייקי זהב יש השבוע?", "gold_breaks"),
        ("where are the gold breaks placed?", "gold_breaks"),
        ("מה האילוצים הפעילים?", "active_constraints"),
        ("יש אילוץ על יום שישי?", "active_constraints"),
        ("list the active constraints", "active_constraints"),
        ("אילו עקיפות מוגדרות?", "active_overrides"),
        ("יש נעיצה בפריים טיים?", "active_overrides"),
        ("מה הנעיצות הפעילות?", "active_overrides"),
        ("show me the manual overrides", "active_overrides"),
        ("did we pin anything tonight?", "active_overrides"),
        ("מה המחירון הנוכחי?", "pricing_state"),
        ("איך עובד התמחור שלנו?", "pricing_state"),
        ("מה המחיר לשנייה?", "pricing_state"),
        ("what is the CPP pricing today?", "pricing_state"),
        ("מה מצב הפייסינג של הקמפיין?", "pacing_status"),
        ("צריך מייק גוד למישהו?", "pacing_status"),
        ("any make-good alerts for campaigns?", "pacing_status"),
    ],
)
def test_keyword_matches(question: str, name: str) -> None:
    assert keywords._matches(question, name), (question, name)


@pytest.mark.parametrize(
    ("question", "name"),
    [
        ("מה ההכנסה הצפויה השבוע?", "gold_breaks"),
        ("golden retriever is not inventory", "gold_breaks"),
        ("how is the week looking?", "active_constraints"),
        ("pinpoint accuracy", "active_overrides"),
        ("spinning wheels", "active_overrides"),
        ("מה שלומך היום?", "pricing_state"),
        ("campfire stories", "pacing_status"),
    ],
)
def test_keyword_non_matches(question: str, name: str) -> None:
    assert not keywords._matches(question, name), (question, name)


# --- composition: present on keyword, nothing without one ------------------------
def test_no_keyword_adds_no_keyword_section() -> None:
    context, sources = assistant._compose_context("מה ההכנסה הצפויה השבוע?")
    for name in keywords.SECTION_NAMES:
        assert name not in context
        assert name not in sources
        assert f"{name} (absent)" not in sources


def test_gold_question_attaches_scoped_gold_section() -> None:
    context, sources = assistant._compose_context("כמה ברייקי זהב יש בתוכנית?")
    assert "gold_breaks" in sources
    section = context["gold_breaks"]
    assert section["available"] is True
    assert isinstance(section["breaks"], list)
    assert len(section["breaks"]) <= keywords.GOLD_ROW_CAP
    owned = str(assistant_context._server()._load_settings().operator_channel or "").strip()
    if owned:
        assert section["scope_channel"] == owned
    # Zero gold is an honest empty state with a reason, never invented rows.
    if section["count"] == 0:
        assert section["breaks"] == []
        assert section.get("reason") or section.get("enabled") is False


def test_constraints_and_overrides_sections_are_capped_and_compact() -> None:
    context, _ = assistant._compose_context("מה האילוצים והנעיצות הפעילים?")
    constraints = context["active_constraints"]
    assert constraints["count"] == len(constraints["constraints"]) + constraints.get("rows_omitted", 0)
    assert len(constraints["constraints"]) <= keywords.CONSTRAINT_ROW_CAP
    for row in constraints["constraints"]:
        assert set(row) == {"constraint_id", "scope_type", "scope_value", "channel", "effect"}
    overrides = context["active_overrides"]
    assert overrides["count"] == len(overrides["overrides"]) + overrides.get("rows_omitted", 0)
    assert len(overrides["overrides"]) <= keywords.OVERRIDE_ROW_CAP
    for row in overrides["overrides"]:
        assert set(row) == {"override_id", "scope", "kind", "target_id", "value"}


def test_pricing_section_carries_base_layers_and_overrides_flag() -> None:
    context, sources = assistant._compose_context("what does our pricing look like?")
    assert "pricing_state" in sources
    section = context["pricing_state"]
    assert {"currency", "units", "base", "layers", "activation", "has_overrides"} <= set(section)
    assert {"value", "default", "overridden"} <= set(section["base"])
    assert section["layers"]
    for layer in section["layers"]:
        assert set(layer) == {"name", "live_today"}
    assert isinstance(section["has_overrides"], bool)


def test_pacing_section_reports_flights_and_make_good_honestly() -> None:
    context, sources = assistant._compose_context("מה מצב הפייסינג?")
    assert "pacing_status" in sources
    section = context["pacing_status"]
    assert isinstance(section["flights_count"], int)
    assert isinstance(section["make_good_available"], bool)
    if not section["make_good_available"]:
        assert section["make_good_reason"]
        assert "make_good_alerts_count" not in section
    else:
        assert isinstance(section.get("make_good_alerts_count"), int)


# --- honesty: a failing builder is absent-marked, never substituted --------------
def test_failed_builder_is_marked_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import kairos_api.insights_api as insights

    def boom() -> dict:
        raise RuntimeError("builder exploded")

    monkeypatch.setattr(insights, "_build_gold_breaks", boom)
    context, sources = assistant._compose_context("כמה ברייקי זהב יש?")
    assert "gold_breaks" not in context
    assert "gold_breaks (absent)" in sources


# --- competitor boundary survives the new sections -------------------------------
def test_keyword_sections_leak_no_competitor_channel_name() -> None:
    import pandas as pd

    server = assistant_context._server()
    frame = pd.read_csv(server.OUTPUT_DIR / "weekly_break_schedule.csv")
    owned = str(server._load_settings().operator_channel or "").strip()
    competitors = sorted(
        {text for text in frame["channel"].astype(str).str.strip().unique() if text} - {owned}
    )
    assert competitors, "the saved plan must carry competitor channels for this test to bite"
    context, _ = assistant._compose_context(
        "מה מצב ברייקי הזהב, האילוצים, הנעיצות, התמחור והפייסינג?"
    )
    serialized = json.dumps(context, ensure_ascii=False, default=str)
    for name in competitors:
        assert name not in serialized
