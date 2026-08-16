"""Regression for the advertiser-history conversation that fanned out by pod.

This suite never calls a model provider.  It replays the exact operator question
against the deterministic first-turn router, records the kwargs sent to a fake
Anthropic client, and executes the real read tool against the shipped traffic
file.  The contract is deliberately about completeness of the files on disk,
not about an advertiser's lifetime: one complete bounded corpus is useful;
calling a top-20 ranking and then opening an arbitrary subset of pods is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kairos_api import (
    assistant_actions,
    assistant_conversations,
    assistant_memory,
    assistant_saved_entry,
    assistant_tool_trace,
    assistant_tools,
)
from kairos_api.assistant_pipeline import run_tool_loop
from kairos_api.assistant_read_tools import execute_read_tool
from kairos_api.assistant_tool_routing import preferred_read_tool


QUESTION = "מי עדן מה הם פרסמו עד היום?"
HISTORY_TOOL = "get_advertiser_airings"
LEGACY_FANOUT = {"get_top_advertisers", "get_break_pods", "get_pod"}
ROOT = Path(__file__).resolve().parents[1]


def _scripted_client(calls: list[dict[str, Any]]) -> Any:
    """A provider-free model seam: one forced read, then the final answer."""

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(
                content=[SimpleNamespace(
                    type="tool_use",
                    id="airings-1",
                    name=HISTORY_TOOL,
                    input={"name": "מי עדן", "limit": 100},
                )],
                stop_reason="tool_use",
            )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_the_exact_question_forces_the_complete_read_on_the_first_model_turn() -> None:
    """The regression is selection, not merely whether a tool exists.

    A prompt recommendation still leaves the old top-20/pod path available to
    the model.  The server-side route must therefore select the history tool in
    ``tool_choice`` on the first call.  The fake client makes this deterministic
    and incurs no provider request or billing.
    """
    preferred = preferred_read_tool(QUESTION)
    assert preferred == HISTORY_TOOL
    assert preferred in assistant_tools.READ_TOOL_NAMES

    calls: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    answer, stopped = run_tool_loop(
        _scripted_client(calls),
        f"CONTEXT:\n{{}}\n\nQUESTION:\n{QUESTION}",
        trace=trace,
        items=[],
        actions_on=True,
        preferred_tool=preferred,
    )

    assert answer == "done"
    assert stopped is False
    assert len(calls) == 2
    first = calls[0]
    assert first["tool_choice"] == {"type": "tool", "name": HISTORY_TOOL}
    assert HISTORY_TOOL in {tool["name"] for tool in first["tools"]}
    assert first["tool_choice"]["name"] not in LEGACY_FANOUT
    # The complete read is one tool call.  The second model turn only writes the
    # answer from that result and returns to automatic choice; no ranking or pod
    # lookup was called before or after it.
    assert [step["tool"] for step in trace] == [HISTORY_TOOL]
    assert trace[0]["ok"] is True
    assert "tool_choice" not in calls[1]
    result_block = calls[1]["messages"][-1]["content"][0]
    assert result_block["tool_use_id"] == "airings-1"
    assert json.loads(result_block["content"])["summary"]["airings"] == 2


def test_the_forced_route_is_narrow_and_does_not_guess_an_advertiser() -> None:
    assert preferred_read_tool("מי המפרסמים המובילים היום?") is None
    assert preferred_read_tool("מי עדן מה תנאי התמחור שלהם?") is None
    assert preferred_read_tool("חברה שאינה במאגר מה היא פרסמה עד היום?") is None


def test_one_read_returns_every_sourced_mei_eden_airing_with_honest_coverage() -> None:
    payload = execute_read_tool(HISTORY_TOOL, {"name": "מי עדן", "limit": 100}, None)

    assert payload["status"] == "ok"
    assert payload["query"] == {"name": "מי עדן", "date_from": None, "date_to": None}
    assert payload["identity"]["resolved"] is True
    assert payload["identity"]["canonical_name"] == "מי עדן"
    assert payload["identity"]["raw_names_matched"] == ["מי עדן"]

    coverage = payload["coverage"]
    assert coverage["files_discovered"] == coverage["files_read"] == 1
    assert coverage["files_failed"] == []
    assert coverage["rows_read"] == coverage["authoritative_rows"] == 175
    assert coverage["rows_without_broadcast_day"] == 0
    assert coverage["selected_rows"] == coverage["authoritative_rows"]
    assert coverage["available_days"] == coverage["selected_days"] == ["2025-04-27"]
    assert coverage["source_files_used"] == ["Wally_Prime_Reshet_Example_2025-04-27.csv"]
    assert coverage["shadowed_day_versions"] == []
    # Complete for the explicitly enumerated corpus, never presented as
    # lifetime history or as continuous coverage through the current day.
    assert coverage["complete_for_available_files"] is True
    assert coverage["complete_through_today"] is False
    assert "not continuous history" in coverage["completeness_note"]

    assert payload["summary"] == {
        "airings": 2,
        "seconds": 30.0,
        "broadcast_days": 1,
        "campaigns": 1,
        "creatives": 1,
        "breaks": 2,
        "agencies": ["יוניברסל"],
        "first_airing_at": "2025-04-27T22:06:50",
        "last_airing_at": "2025-04-27T23:05:28",
    }
    assert len(payload["campaigns"]) == len(payload["creatives"]) == 1
    assert len(payload["breaks"]) == 2
    assert payload["pagination"] == {
        "offset": 0,
        "limit": 100,
        "returned": 2,
        "total": 2,
        "has_more": False,
        "next_offset": None,
    }

    airings = payload["airings"]
    assert [row["airing_at"] for row in airings] == [
        "2025-04-27T22:06:50",
        "2025-04-27T23:05:28",
    ]
    assert [row["break_start"] for row in airings] == ["22:03:06", "22:59:40"]
    assert [row["position_in_break"] for row in airings] == [5, 4]
    assert {row["programme"] for row in airings} == {
        "המקור - עונה 24",
        "המקור - עונה 24 - דיון באולפן",
    }
    for row in airings:
        assert row["advertiser"] == "מי עדן"
        assert row["campaign"] == "2025-02 - מי עדן - מי עדן סודה — מי עדן סודה חדש"
        assert row["creative"] == "מי עדן סודה קיצור 15 מחליפה 2"
        assert row["house_number"] == "CMK022702"
        assert row["agency"] == "יוניברסל"
        assert row["duration_seconds"] == 15.0
        assert row["source_file"] == coverage["source_files_used"][0]
        assert row["source_row"] > 1
    assert "not priced, invoiced" in payload["basis"]


def test_the_finished_trace_has_a_human_label_for_the_one_complete_read() -> None:
    source = (ROOT / "tv-break-dashboard" / "src" / "kai" / "AssistantRunTrace.jsx").read_text(
        encoding="utf-8"
    )
    assert "get_advertiser_airings: ['Reading advertiser airing history'" in source
    assert "קורא את היסטוריית שידורי המפרסם" in source


def test_the_complete_read_keeps_bounded_evidence_after_a_reload(
    monkeypatch, tmp_path: Path,
) -> None:
    """A saved answer remains auditable without copying the product database."""
    monkeypatch.setenv(assistant_actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    payload = execute_read_tool(HISTORY_TOOL, {"name": "מי עדן", "limit": 100}, None)
    step = assistant_tool_trace.trace_step(
        HISTORY_TOOL,
        True,
        "all authoritative raw daily traffic files on disk",
        payload,
    )
    body = {
        "grounding": {"sources": ["daily traffic"]},
        "tool_trace": [step],
        "context_disclosure": "one complete raw-file read",
    }
    metadata = assistant_saved_entry.from_ask(body, 1.2345)

    assistant_memory.append_entry("operator", QUESTION, "answer", metadata=metadata)
    conversation_id = assistant_conversations.newest_id("operator")
    stored = assistant_conversations.entries_for("operator", conversation_id)[0]

    assert stored["sources"] == ["daily traffic"]
    assert stored["elapsed_seconds"] == 1.234
    assert stored["context_disclosure"] == "one complete raw-file read"
    assert stored["coverage"]["authoritative_rows"] == 175
    result = stored["tool_trace"][0]["result"]
    assert result["kind"] == "advertiser_airings"
    assert result["summary"]["airings"] == 2
    assert len(result["airings"]) == 2
    assert result["trace_airings_omitted"] == 0

    component = (ROOT / "tv-break-dashboard" / "src" / "kai" / "AdvertiserAiringsResult.jsx").read_text(
        encoding="utf-8"
    )
    assert 'className="card asst-airings-card"' in component
    assert "זה אינו כיסוי היסטורי מלא עד היום" in component
