"""Audience-model disclosure coverage: the get_audience_model read tool on a
seeded and an absent artifact, the activation flag read, the keyword grounding
section's on and off triggers, and the system-prompt sentence that keeps
expected rating measured and gated, never asserted.

Everything runs on throwaway artifact paths (the module's path seam is
monkeypatched); no model call and no API key is involved.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import kairos_api.assistant as assistant
import kairos_api.assistant_audience_model as audience
import kairos_api.assistant_tools as tools
from kairos_api import assistant_keywords
from kairos_api.assistant_read_tools import execute_read_tool

FAMILIES = set(audience.AUDIENCE_FAMILIES)


# --- fixtures ---------------------------------------------------------------------
@pytest.fixture()
def artifact_path(tmp_path, monkeypatch):
    """The audience-model artifact on a throwaway path; absent until written."""
    path = tmp_path / "audience_model.json"
    monkeypatch.setattr(audience, "_audience_model_path", lambda: path)
    return path


def _gate(verdict: str = "off", reason: str = "history carries no contrast for this family",
          delta: "float | None" = None) -> dict[str, Any]:
    return {"verdict": verdict, "reason": reason, "held_out_delta_pct": delta,
            "measured_at": "2026-07-19T00:00:00Z"}


def _seed(path, gates: dict[str, Any], **top: Any) -> None:
    body: dict[str, Any] = {
        "computed_at": "2026-07-20T00:00:00Z",
        "activation_default": False,
        "base": {"kind": "pooled slot and programme base", "rows": 1234},
        "gates": gates,
        "source_fingerprints": {"spots": "abc123"},
    }
    body.update(top)
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")


# --- registry ---------------------------------------------------------------------
def test_get_audience_model_registered_as_read_tool() -> None:
    assert "get_audience_model" in tools.READ_TOOL_NAMES
    assert "get_audience_model" not in tools.PROPOSE_TOOL_NAMES
    read_only = {schema["name"] for schema in tools.anthropic_tools(include_propose=False)}
    assert "get_audience_model" in read_only


# --- absent artifact: honest tri-state, never an invented gate --------------------
def test_absent_artifact_reads_unavailable_with_reason(artifact_path) -> None:
    payload = execute_read_tool("get_audience_model", {}, None)
    assert payload["available"] is False
    assert "models/audience_model.json" in payload["reason"]
    assert "gates" not in payload
    assert payload["source"] == audience.AUDIENCE_SOURCE
    activation = payload["activation"]
    assert activation["flag"] == "audience_model_activation"
    assert isinstance(activation["enabled"], bool)
    assert activation["default_off"] is True
    assert "expected rating" in payload["basis"]
    assert "retention" in payload["basis"]


def test_unparsable_artifact_reads_unavailable(artifact_path) -> None:
    artifact_path.write_text("not json at all", encoding="utf-8")
    payload = execute_read_tool("get_audience_model", {}, None)
    assert payload["available"] is False
    assert "could not be parsed" in payload["reason"]


# --- seeded artifact: verdicts pass through, families stay complete ---------------
def test_seeded_all_off_artifact_carries_every_family_and_headline(artifact_path) -> None:
    _seed(artifact_path, {family: _gate() for family in audience.AUDIENCE_FAMILIES})
    payload = execute_read_tool("get_audience_model", {}, None)
    assert payload["available"] is True
    assert payload["computed_at"] == "2026-07-20T00:00:00Z"
    assert set(payload["gates"]) == FAMILIES
    for family, gate in payload["gates"].items():
        assert gate["verdict"] == "off"
        assert gate["label_he"] == audience.AUDIENCE_FAMILY_LABELS_HE[family]
    assert payload["gates"]["weekday_slot"]["label_he"] == "יום ורצועה"
    assert payload["gates"]["competitor_lineup"]["label_he"] == "ליינאפ מתחרים"
    assert payload["families_on"] == []
    assert payload["families_on_count"] == 0
    assert payload["all_off_headline_he"] == audience.AUDIENCE_ALL_OFF_HE
    assert payload["base_summary"] == {"kind": "pooled slot and programme base", "rows": 1234}


def test_seeded_partial_artifact_marks_on_and_unknown_families(artifact_path) -> None:
    _seed(artifact_path, {
        "weekday_slot": _gate("on", "held-out gate passed", 3.2),
        "series": _gate("off", "insufficient per-series history"),
    })
    payload = execute_read_tool("get_audience_model", {}, None)
    gates = payload["gates"]
    assert set(gates) == FAMILIES, "every contract family is reported, recorded or not"
    assert gates["weekday_slot"]["verdict"] == "on"
    assert gates["weekday_slot"]["held_out_delta_pct"] == pytest.approx(3.2)
    assert gates["series"]["verdict"] == "off"
    for family in FAMILIES - {"weekday_slot", "series"}:
        assert gates[family]["verdict"] == "unknown"
        assert gates[family]["reason"] == audience.AUDIENCE_MISSING_FAMILY_REASON
    assert payload["families_on"] == ["weekday_slot"]
    assert payload["families_on_count"] == 1
    assert "all_off_headline_he" not in payload


# --- the activation flag rides the settings seam ----------------------------------
def test_activation_flag_reflects_settings(artifact_path, monkeypatch) -> None:
    import kairos_api.core as core

    monkeypatch.setattr(core, "_load_settings",
                        lambda: SimpleNamespace(audience_model_activation=True))
    assert execute_read_tool("get_audience_model", {}, None)["activation"]["enabled"] is True
    monkeypatch.setattr(core, "_load_settings", lambda: SimpleNamespace())
    assert execute_read_tool("get_audience_model", {}, None)["activation"]["enabled"] is False


# --- keyword grounding: on for the audience phrases, off otherwise ----------------
@pytest.mark.parametrize("question", [
    "מה הרייטינג הצפוי מחר בערב",
    "איך עובד מודל הקהל",
    "תן לי תחזית צפייה לשבוע הבא",
    "what does the audience model condition on",
    "what is the expected rating for Sunday prime time",
])
def test_audience_section_attaches_on_triggers(artifact_path, question: str) -> None:
    _seed(artifact_path, {family: _gate() for family in audience.AUDIENCE_FAMILIES})
    context: dict[str, Any] = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, question)
    assert "audience_model" in context, f"audience_model did not attach for {question!r}"
    assert "audience_model" in sources
    section = context["audience_model"]
    assert section["available"] is True
    assert set(section["gates"]) == FAMILIES
    assert "base_summary" not in section, "the keyword section stays compact"


def test_audience_section_absent_artifact_is_honest_not_absent_marked(artifact_path) -> None:
    context: dict[str, Any] = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, "מה מצב מודל הקהל")
    assert "audience_model" in sources
    section = context["audience_model"]
    assert section["available"] is False
    assert "models/audience_model.json" in section["reason"]


@pytest.mark.parametrize("question", [
    "כמה ברייקים יש מחר",
    "מה ההכנסה הצפויה השבוע",
    "rating the show was fun",
    "מה מצב הקמפיין",
])
def test_audience_section_stays_off_unprompted(question: str) -> None:
    context: dict[str, Any] = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, question)
    assert "audience_model" not in context
    assert "audience_model" not in sources
    assert "audience_model (absent)" not in sources


# --- the system prompt keeps expected rating measured and gated -------------------
def test_system_prompt_carries_the_audience_sentence() -> None:
    prompt = assistant.SYSTEM_PROMPT
    sentence = prompt.index("Expected rating (the audience model) is likewise measured and gated, never asserted")
    assert "get_audience_model" in prompt
    assert prompt.index("(d) Training") < sentence < prompt.index("Never skip the honesty line")
