"""The remaining Kai coverage gap: one complete, scoped campaign record."""

from __future__ import annotations

from kairos_api import assistant_tools
from kairos_api.assistant_read_tools import SOURCE_BY_TOOL, _READ_EXECUTORS, execute_read_tool


def test_campaign_read_is_registered_and_not_a_write() -> None:
    assert "get_campaign" in assistant_tools.READ_TOOL_NAMES
    assert "get_campaign" in _READ_EXECUTORS
    assert SOURCE_BY_TOOL["get_campaign"]
    assert "get_campaign" not in assistant_tools.PROPOSE_TOOL_NAMES


def test_one_owned_campaign_arrives_with_every_record_layer() -> None:
    board = execute_read_tool("get_pacing_board", {}, None)
    rows = list(board.get("campaigns") or board.get("rows") or [])
    if not rows:
        return
    campaign_id = str(rows[0]["campaign_id"])
    payload = execute_read_tool("get_campaign", {"campaign_id": campaign_id}, None)
    assert "error" not in payload, payload
    campaign = payload["campaign"]
    assert campaign["campaign_id"] == campaign_id
    assert payload["scope"] == {"channel": campaign["channel"], "scoped": True}
    assert {"commitment", "order", "flights", "assets", "delivery"} <= set(campaign)
    assert campaign["flights_count"] >= len(campaign["flights"])
    assert campaign["assets_count"] >= len(campaign["assets"])
    assert campaign["delivery"]["days_count"] >= len(campaign["delivery"]["days"])
    if campaign["order"]["kind"] == "goal_based":
        assert "goal_preflight" in payload


def test_rival_and_typo_are_indistinguishable(monkeypatch) -> None:
    from kairos_api import campaigns_api_store, channel_scope

    monkeypatch.setattr(channel_scope, "operator_channel", lambda: "mine")
    monkeypatch.setattr(campaigns_api_store, "load_frame", lambda: object())
    monkeypatch.setattr(
        campaigns_api_store,
        "campaigns_with_flights",
        lambda _frame: [{"campaign_id": "RIVAL", "channel": "theirs"}],
    )
    rival = execute_read_tool("get_campaign", {"campaign_id": "RIVAL"}, None)
    typo = execute_read_tool("get_campaign", {"campaign_id": "TYPO"}, None)
    rival_shape = {key: value for key, value in rival.items() if key != "campaign_id"}
    typo_shape = {key: value for key, value in typo.items() if key != "campaign_id"}
    assert rival_shape == typo_shape
    assert rival["error"] == typo["error"]


def test_missing_id_is_an_explicit_refusal() -> None:
    payload = execute_read_tool("get_campaign", {}, None)
    assert "error" in payload
