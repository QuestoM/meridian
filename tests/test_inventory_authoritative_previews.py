"""Authoritative optimizer reads refuse a present inventory file with no slots.

The inventory signal is optional when the file is absent. Once a source with
rows is present, however, silently treating a total parse failure as the same
neutral state would publish money from an optimizer whose advertised steer was
inert. These tests pin every application boundary that can publish such figures
or persist a measurement.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from kairos.optimize import inventory as inventory_module
from kairos.optimize.inventory import InventoryInputError
from kairos_api import (
    assistant_simulate,
    core,
    model_console_api,
    model_console_candidates,
    model_version_store,
    plan_read_frontier,
    scenario_api,
    scenario_compare_api,
    scenario_compare_api_week,
    server,
)


@pytest.fixture()
def invalid_inventory(tmp_path, monkeypatch) -> Any:
    target = tmp_path / "Spots - inventory.csv"
    target.write_text(
        "Channel,Date_dt,hour_of_day,Start_dt\n"
        "owned,2024-11-04,,\n"
        "owned,2024-11-05,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(inventory_module, "DEFAULT_INVENTORY_PATH", target)
    scenario_api._optimizer_plan_cached.cache_clear()
    scenario_api._scenario_cached.cache_clear()
    plan_read_frontier.frontier_points_cached.cache_clear()
    plan_read_frontier.frontier_net_bundle_cached.cache_clear()
    scenario_compare_api_week._day_leg_cached.cache_clear()
    assistant_simulate._priced_side.cache_clear()
    return target


@pytest.fixture()
def client() -> TestClient:
    # No context manager: the test is about request seams, not cache warm-up.
    return TestClient(server.app)


def _must_not_run(label: str):
    def blocked(*_args, **_kwargs):
        raise AssertionError(f"{label} ran after the inventory refusal")

    return blocked


def test_service_strict_switch_stops_before_optimizer_or_run_log(
    invalid_inventory, monkeypatch
) -> None:
    import kairos.service as service

    monkeypatch.setattr(service, "_optimize_one_day", _must_not_run("optimizer"))
    monkeypatch.setattr(service, "write_run_log", _must_not_run("run log"))

    with pytest.raises(InventoryInputError, match="produced no usable"):
        service.optimize_day_plan(require_usable_inventory=True)
    with pytest.raises(InventoryInputError, match="produced no usable"):
        service.run_scenario(
            revenue_weight=60,
            retention_floor=0.72,
            max_breaks_per_hour=4,
            require_usable_inventory=True,
        )


@pytest.mark.parametrize("method", ["get", "post"])
def test_optimizer_plan_routes_return_422_without_computing(
    invalid_inventory, client, monkeypatch, method
) -> None:
    monkeypatch.setattr(scenario_api, "run_scenario", _must_not_run("optimizer plan"))
    response = getattr(client, method)("/api/optimizer-plan", **({"json": {}} if method == "post" else {}))

    assert response.status_code == 422
    assert "produced no usable" in response.json()["detail"]


def test_scenario_returns_no_figures_and_never_enters_cached_runner(
    invalid_inventory, client, monkeypatch
) -> None:
    monkeypatch.setattr(scenario_api, "_scenario_cached", _must_not_run("scenario cache"))
    response = client.post("/api/scenario", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "unavailable"
    assert body["summary"] == {
        "total_breaks": None,
        "total_ad_seconds": None,
        "projected_revenue": None,
        "average_retention": None,
        "risk_score": None,
    }
    assert "produced no usable" in body["detail"]


def test_scenario_cache_key_moves_when_valid_inventory_moves(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "Spots - inventory.csv"
    source.write_text("Channel,Date_dt,hour_of_day\nowned,2024-11-04,9\n", encoding="utf-8")
    monkeypatch.setattr(scenario_api, "DATA_DIR", data_dir)
    before = scenario_api._scenario_data_signature()

    source.write_text(
        "Channel,Date_dt,hour_of_day\nowned,2024-11-04,10\nowned,2024-11-05,11\n",
        encoding="utf-8",
    )
    after = scenario_api._scenario_data_signature()

    assert before != after


def test_frontier_cache_key_follows_the_inventory_path_actually_loaded(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "first-inventory.csv"
    second = tmp_path / "second-inventory.csv"
    first.write_text("Channel,Date_dt,hour_of_day\nowned,2024-11-04,9\n", encoding="utf-8")
    second.write_text("Channel,Date_dt,hour_of_day\nowned,2024-11-04,10\n", encoding="utf-8")

    monkeypatch.setattr(inventory_module, "DEFAULT_INVENTORY_PATH", first)
    first_signature = plan_read_frontier.frontier_data_signature()
    monkeypatch.setattr(inventory_module, "DEFAULT_INVENTORY_PATH", second)
    second_signature = plan_read_frontier.frontier_data_signature()

    assert first_signature != second_signature
    assert any(path == str(first) for path, _stamp in first_signature)
    assert any(path == str(second) for path, _stamp in second_signature)


def test_named_forecasts_refuse_before_cached_saved_or_scenario_figures(
    invalid_inventory, client, monkeypatch
) -> None:
    monkeypatch.setattr(
        scenario_compare_api,
        "_forecasts_cached",
        _must_not_run("forecast cache"),
    )
    response = client.get("/api/forecasts")

    assert response.status_code == 422
    assert "produced no usable" in response.json()["detail"]


def test_optimal_plan_refuses_before_run_and_run_log(
    invalid_inventory, client, monkeypatch
) -> None:
    import kairos.service as service

    monkeypatch.setattr(service, "write_run_log", _must_not_run("run log"))
    response = client.post("/api/optimal-plan", json={})

    assert response.status_code == 422
    assert "produced no usable" in response.json()["detail"]


def test_frontier_and_net_comparison_are_unavailable_without_background_work(
    invalid_inventory, client, monkeypatch
) -> None:
    settings = core.KairosSettings(operator_channel="owned")
    monkeypatch.setattr(plan_read_frontier.threading, "Thread", _must_not_run("frontier thread"))
    monkeypatch.setattr(scenario_api, "_load_settings", lambda: settings)

    points, bundle, status = plan_read_frontier.frontier_state(settings)
    assert points == []
    assert status == "unavailable"
    assert bundle and bundle["comparison_available"] is False
    assert bundle["current"] is None and bundle["net_focused"] is None

    response = client.get("/api/optimizer/net-comparison")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["current"] is None and body["net_focused"] is None and body["delta"] is None
    assert "produced no usable" in body["reason"]


@pytest.mark.parametrize("scope", ["day", "week"])
def test_plain_comparison_refuses_before_either_leg(
    invalid_inventory, client, monkeypatch, scope
) -> None:
    monkeypatch.setattr(server, "_ENGINE_AVAILABLE", True)
    monkeypatch.setattr(scenario_compare_api_week, "week_events", _must_not_run("week legs"))
    response = client.post(
        "/api/scenario-compare",
        json={"weight_a": 20, "weight_b": 80, "scope": scope},
    )

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert "produced no usable" in response.json()["reason"]
    assert "a" not in response.json() and "b" not in response.json()


def test_streaming_comparison_preparation_emits_only_an_error(
    invalid_inventory, client, monkeypatch
) -> None:
    monkeypatch.setattr(server, "_ENGINE_AVAILABLE", True)
    monkeypatch.setattr(scenario_compare_api_week, "week_events", _must_not_run("stream legs"))
    response = client.post(
        "/api/scenario-compare/stream",
        json={"weight_a": 20, "weight_b": 80, "scope": "week"},
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "produced no usable" in response.text
    assert "event: window" not in response.text
    assert "event: day" not in response.text
    assert "event: final" not in response.text


def test_assistant_simulation_returns_unavailable_before_pricing(
    invalid_inventory, monkeypatch
) -> None:
    monkeypatch.setattr(core, "_ENGINE_AVAILABLE", True)
    monkeypatch.setattr(assistant_simulate, "_side_for", _must_not_run("assistant simulation"))

    result = assistant_simulate.simulate_settings_change({"revenue_weight": 61})
    assert result["status"] == "unavailable"
    assert "produced no usable" in result["reason"]
    assert not ({"before", "after", "delta"} & set(result))


def test_candidate_measurement_refuses_before_thread_or_store_write(
    invalid_inventory, client, monkeypatch
) -> None:
    paths = model_console_candidates.candidate_paths()
    if not paths:
        pytest.skip("no candidate artifact is present")
    candidate_id = model_console_candidates.candidate_id(paths[0])
    monkeypatch.setattr(model_console_api.threading, "Thread", _must_not_run("measurement thread"))
    monkeypatch.setattr(model_version_store, "save_measurement", _must_not_run("measurement store"))

    response = client.post(f"/api/model/candidates/{candidate_id}/measure")
    assert response.status_code == 422
    assert "produced no usable" in response.json()["detail"]
    assert candidate_id not in model_console_api._RUNNING

    monkeypatch.setattr(
        model_console_candidates,
        "build_plan_totals",
        _must_not_run("candidate plan totals"),
    )
    with pytest.raises(InventoryInputError, match="produced no usable"):
        model_console_candidates.measure_money_movement(candidate_id)


def test_candidate_measurement_fingerprint_includes_inventory(
    tmp_path, monkeypatch
) -> None:
    paths = model_console_candidates.candidate_paths()
    if not paths:
        pytest.skip("no candidate artifact is present")
    source = tmp_path / "inventory.csv"
    source.write_text("Channel,Date_dt,hour_of_day\nowned,2024-11-04,9\n", encoding="utf-8")
    monkeypatch.setattr(inventory_module, "DEFAULT_INVENTORY_PATH", source)
    before = model_console_candidates.measurement_inputs(paths[0])

    source.write_text("Channel,Date_dt,hour_of_day\nowned,2024-11-04,10\n", encoding="utf-8")
    after = model_console_candidates.measurement_inputs(paths[0])

    assert before["inventory"] != after["inventory"]
    assert model_console_candidates.measurement_fingerprint(paths[0], before) != (
        model_console_candidates.measurement_fingerprint(paths[0], after)
    )


def test_missing_inventory_remains_the_explicit_neutral_signal(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "not-uploaded.csv"
    monkeypatch.setattr(inventory_module, "DEFAULT_INVENTORY_PATH", missing)

    assert inventory_module.load_inventory(require_usable=True) == {}
