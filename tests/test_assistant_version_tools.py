"""Wave 4 closes the version and adoption reads without adding an act."""

from __future__ import annotations

import json

import pandas as pd

from kairos_api import assistant_model_disclosure as disclosure
from kairos_api import assistant_tools
from kairos_api.assistant_read_tools import SOURCE_BY_TOOL, _READ_EXECUTORS, execute_read_tool


def test_both_version_reads_are_registered_as_reads_only() -> None:
    for name in ("get_plan_versions", "get_model_adoption"):
        assert name in assistant_tools.READ_TOOL_NAMES
        assert name in _READ_EXECUTORS
        assert SOURCE_BY_TOOL[name]
        assert name not in assistant_tools.PROPOSE_TOOL_NAMES


def test_plan_versions_publish_owned_summaries_and_no_market_totals() -> None:
    payload = execute_read_tool("get_plan_versions", {}, None)
    assert "error" not in payload, payload
    assert payload["scope"]["scoped"] is True
    assert set(payload["scope"]) == {"scope_channel", "scoped"}
    assert payload["versions_count"] >= len(payload["versions"])
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    for forbidden in ("all_channels", "competitor_rows", "competitor_channels",
                      "input_fingerprints", "plan_sha256", "whole_plan_delta"):
        assert forbidden not in serialized
    if payload["versions"]:
        assert payload["versions"][0]["owned_summary"]["channels"] == 1


def test_model_adoption_reads_real_candidates_and_decisions_without_acting(monkeypatch) -> None:
    monkeypatch.setattr(disclosure, "actor_is_company", lambda _user: True)
    payload = execute_read_tool("get_model_adoption", {}, "company-reader")
    assert "error" not in payload, payload
    assert payload["candidates_count"] >= len(payload["candidates"])
    assert payload["decisions_count"] >= len(payload["decisions"])
    assert payload["proposing_or_adopting"] == "not available in this read tool"
    assert payload["candidates"], "the real candidate shelf is empty, so this read proves nothing"
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    assert "whole_plan_delta" not in serialized
    assert '"file"' not in serialized and '"sha256"' not in serialized


def test_channel_account_gets_the_standard_model_release_wall(monkeypatch) -> None:
    monkeypatch.setattr(disclosure, "actor_is_company", lambda _user: False)
    payload = execute_read_tool("get_model_adoption", {}, "channel-reader")
    assert payload["available"] is False
    assert payload["withheld"] == disclosure.WITHHELD_REASON
    assert "model_version" in payload and "release_note" in payload
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    for forbidden in ("candidates", "decisions", "gate_deltas", "held_out_deltas"):
        assert forbidden not in serialized


def test_plan_versions_are_rescoped_from_frozen_bytes_not_historical_labels(monkeypatch) -> None:
    """Changing operator channel cannot relabel the old channel's money as owned."""
    from kairos_api import assistant_read_tools_versions as versions
    from kairos_api import channel_scope, plan_version_store as store

    rival_manifest = {
        "version_id": "v1",
        "name": "old freeze",
        "settings_basis": {"operator_channel": "RIVAL", "revenue_weight": 60},
        "summary": {"owned": {"revenue": 999.0}},
        "owned_delta_from_previous": {"revenue": 999.0},
    }
    frozen = pd.DataFrame([
        {"channel": "RIVAL", "predicted_revenue": 999.0, "num_breaks": 9,
         "total_break_time": 1080, "date": "2024-11-01"},
        {"channel": "OWNED", "predicted_revenue": 7.0, "num_breaks": 1,
         "total_break_time": 120, "date": "2024-11-01"},
    ])
    monkeypatch.setattr(channel_scope, "operator_channel", lambda: "OWNED")
    monkeypatch.setattr(store, "all_manifests", lambda: [rival_manifest])
    monkeypatch.setattr(store, "_frame_for", lambda _version_id: frozen)
    monkeypatch.setattr(store, "live_state", lambda: {
        "exists": True,
        "summary": {"owned": store._totals(frozen[frozen["channel"] == "OWNED"])},
    })

    payload = versions._read_get_plan_versions({}, None)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert payload["scope"]["scope_channel"] == "OWNED"
    assert payload["versions"][0]["owned_summary"]["revenue"] == 7.0
    assert "RIVAL" not in serialized and "999" not in serialized
    assert "operator_channel" not in payload["versions"][0]["settings_basis"]
