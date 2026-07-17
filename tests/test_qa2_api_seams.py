"""API seam fixes: cache keys, honest fallbacks, exports, and truth fields.

Covers: cache signatures gaining the settings file / rate card / reference
workbook they actually read (a settings edit must invalidate, not serve stale),
the /api/scenario failure fallback returning nulls with a reason instead of the
whole-month CSV summary dressed as a day, GET /api/optimizer-plan memoized on a
settings+data signature, CSV exports carrying a BOM plus charset and the weekly
export carrying the real freshness verdict header, the reports catalog's daily
spot ledger entry with honest row counts, /api/parameters carrying flights_count
and the live (overrides-merged) pricing, and /api/schedule disclosing the real
plan size beside its 200-row display slice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"


@pytest.fixture(scope="module")
def client() -> TestClient:
    from kairos_api.server import app

    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
# Cache signatures include what the builders actually read.
# --------------------------------------------------------------------------- #
def _touch(path: Path) -> None:
    """Change the file's signature (mtime and size both move)."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write("x\n")


def test_schedule_and_break_ops_cache_keys_include_settings_and_rate_card(
    client, tmp_path, monkeypatch
):
    import kairos_api.dashboard_api as dash

    # Trivial builders: the probe measures the cache KEY, not the payload.
    monkeypatch.setattr(dash, "_load_programmes", lambda: pd.DataFrame())
    monkeypatch.setattr(dash, "_load_break_schedule", lambda: pd.DataFrame())
    settings_file = tmp_path / "kairos_settings.json"
    settings_file.write_text("{}\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    yaml_file = tmp_path / "config" / "optimization_weights.yaml"
    yaml_file.write_text("base_price_per_second_per_tvr_point: 60.0\n", encoding="utf-8")
    monkeypatch.setattr(dash, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(dash, "ROOT", tmp_path)
    monkeypatch.setattr(dash, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(dash, "OUTPUT_DIR", tmp_path / "output")

    for cached, route in (
        (dash._schedule_cached, "/api/schedule"),
        (dash._break_operations_cached, "/api/break-operations"),
    ):
        cached.cache_clear()
        try:
            assert client.get(route).status_code == 200
            assert client.get(route).status_code == 200
            assert cached.cache_info().misses == 1, f"{route}: repeat read must hit"
            _touch(settings_file)
            assert client.get(route).status_code == 200
            assert cached.cache_info().misses == 2, f"{route}: settings edit must invalidate"
            _touch(yaml_file)
            assert client.get(route).status_code == 200
            assert cached.cache_info().misses == 3, f"{route}: rate-card edit must invalidate"
        finally:
            cached.cache_clear()


def test_forecasts_cache_key_includes_settings_and_programmes_source(
    client, tmp_path, monkeypatch
):
    import kairos_api.catalog_api as catalog

    monkeypatch.setattr(catalog, "_load_break_schedule", lambda: pd.DataFrame())
    settings_file = tmp_path / "kairos_settings.json"
    settings_file.write_text("{}\n", encoding="utf-8")
    reference = tmp_path / "data" / "reference"
    reference.mkdir(parents=True)
    programmes_file = reference / "Programmes.xlsx"
    programmes_file.write_bytes(b"probe")
    monkeypatch.setattr(catalog, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.setattr(catalog, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(catalog, "OUTPUT_DIR", tmp_path / "output")

    catalog._forecasts_cached.cache_clear()
    try:
        assert client.get("/api/forecasts").status_code == 200
        assert client.get("/api/forecasts").status_code == 200
        assert catalog._forecasts_cached.cache_info().misses == 1
        _touch(settings_file)
        assert client.get("/api/forecasts").status_code == 200
        assert catalog._forecasts_cached.cache_info().misses == 2, (
            "a settings edit must invalidate the cached forecast"
        )
        programmes_file.write_bytes(b"probe-changed")
        assert client.get("/api/forecasts").status_code == 200
        assert catalog._forecasts_cached.cache_info().misses == 3, (
            "a Programmes re-ingest must invalidate the cached forecast"
        )
    finally:
        catalog._forecasts_cached.cache_clear()


def test_inventory_and_campaigns_cache_keys_include_reference_workbook(
    client, tmp_path, monkeypatch
):
    import kairos_api.catalog_api as catalog

    monkeypatch.setattr(catalog, "_load_spots", lambda: pd.DataFrame())
    reference = tmp_path / "data" / "reference"
    reference.mkdir(parents=True)
    spots_file = reference / "Spots.xlsx"
    spots_file.write_bytes(b"probe")
    monkeypatch.setattr(catalog, "DATA_DIR", tmp_path / "data")

    for cached, route in (
        (catalog._inventory_cached, "/api/inventory"),
        (catalog._campaigns_cached, "/api/campaigns"),
    ):
        cached.cache_clear()
        try:
            assert client.get(route).status_code == 200
            assert cached.cache_info().misses == 1
            spots_file.write_bytes(b"probe-changed")
            assert client.get(route).status_code == 200
            assert cached.cache_info().misses == 2, (
                f"{route}: a reference Spots.xlsx re-ingest must invalidate"
            )
        finally:
            cached.cache_clear()


# --------------------------------------------------------------------------- #
# Honest scenario fallback and the memoized optimizer-plan GET.
# --------------------------------------------------------------------------- #
def test_scenario_failure_returns_nulls_with_reason_not_month_scale_money(
    client, monkeypatch
):
    import kairos_api.scenario_api as scenario_api

    if not scenario_api._ENGINE_AVAILABLE:
        pytest.skip("engine unavailable; the failure path needs the engine gate open")

    def boom(*args, **kwargs):
        raise RuntimeError("probe failure")

    monkeypatch.setattr(scenario_api, "_scenario_cached", boom)
    body = client.post(
        "/api/scenario",
        json={"revenue_weight": 60, "retention_floor": 0.72, "max_breaks_per_hour": 4},
    ).json()
    assert body["engine"] == "unavailable"
    assert body["reason"]
    assert "probe failure" in body.get("detail", "")
    summary = body["summary"]
    for key in ("total_breaks", "total_ad_seconds", "projected_revenue", "average_retention", "risk_score"):
        assert summary[key] is None, f"{key} must be null on failure, never the saved CSV summary"


def test_optimizer_plan_get_is_memoized_on_a_settings_data_signature(
    client, tmp_path, monkeypatch
):
    import kairos_api.scenario_api as scenario_api

    calls = {"count": 0}

    def stub_plan():
        calls["count"] += 1
        return {"summary": {"is_compliant": True}, "controls": {}, "engine": "kairos"}

    monkeypatch.setattr(scenario_api, "_build_optimizer_plan", stub_plan)
    settings_file = tmp_path / "kairos_settings.json"
    settings_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(scenario_api, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(scenario_api, "ROOT", tmp_path)
    monkeypatch.setattr(scenario_api, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(scenario_api, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(scenario_api, "MODELS_DIR", tmp_path / "models")

    scenario_api._optimizer_plan_cached.cache_clear()
    try:
        assert client.get("/api/optimizer-plan").status_code == 200
        assert client.get("/api/optimizer-plan").status_code == 200
        assert calls["count"] == 1, "the repeat GET must be a cache hit"
        _touch(settings_file)
        assert client.get("/api/optimizer-plan").status_code == 200
        assert calls["count"] == 2, "a settings edit must recompute the saved plan"
    finally:
        scenario_api._optimizer_plan_cached.cache_clear()


# --------------------------------------------------------------------------- #
# CSV exports: Excel-safe Hebrew and the freshness verdict on the download.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not CSV_PATH.exists(), reason="no committed weekly plan on disk")
def test_schedule_export_carries_bom_charset_and_real_freshness_header(client):
    response = client.get("/api/export/schedule.csv")
    assert response.status_code == 200
    assert response.content[:3] == b"\xef\xbb\xbf", "utf-8-sig BOM missing (Excel Hebrew mojibake)"
    assert response.headers["content-type"].startswith("text/csv")
    assert "charset=utf-8" in response.headers["content-type"]
    verdict = response.headers.get("x-kairos-schedule-freshness")
    assert verdict in {"fresh", "stale", "unknown"}, f"not a real verdict: {verdict}"
    # The header must agree with the engine's own read-only verdict.
    from kairos.export.schedule_freshness import schedule_freshness

    assert verdict == str(schedule_freshness(ROOT).get("status") or "unknown")


def test_spots_export_carries_bom_and_charset(client):
    response = client.get("/api/export/spots.csv")
    assert response.status_code == 200
    assert response.content[:3] == b"\xef\xbb\xbf"
    assert "charset=utf-8" in response.headers["content-type"]


# --------------------------------------------------------------------------- #
# Reports catalog: the daily spot ledger is a first-class entry.
# --------------------------------------------------------------------------- #
def test_reports_catalog_lists_the_daily_spot_ledger_with_honest_rows(client):
    body = client.get("/api/reports").json()
    by_id = {report["id"]: report for report in body["reports"]}
    assert set(by_id) == {"weekly-plan", "compliance", "revenue", "daily-spots", "data-quality"}
    ledger = by_id["daily-spots"]
    assert isinstance(ledger["rows"], int) and ledger["rows"] >= 0
    # The row count must be the exact ledger the download carries.
    export = client.get("/api/export/spots.csv")
    data_rows = max(0, len(export.text.strip().splitlines()) - 1)
    assert ledger["rows"] == data_rows
    assert ledger["status"] == ("ready" if data_rows else "empty")


# --------------------------------------------------------------------------- #
# Parameters: pacing truth and the live (overrides-merged) rate card.
# --------------------------------------------------------------------------- #
def test_parameters_reports_the_real_flights_count(client):
    from kairos.optimize.pacing import load_campaigns

    body = client.get("/api/parameters").json()
    assert body["flights_count"] == len(load_campaigns()), (
        "flights_count must be the pacing loader's own count so the UI can key "
        "the pacing-inactive note on truth"
    )


def test_parameters_pricing_reflects_saved_overrides(client, tmp_path, monkeypatch):
    import kairos_api.core as core

    settings_file = tmp_path / "kairos_settings.json"
    monkeypatch.setattr(core, "SETTINGS_PATH", settings_file)

    core._save_settings(core.KairosSettings())
    body = client.get("/api/parameters").json()
    assert body["pricing"]["has_overrides"] is False
    default_base = body["pricing"]["base_price_per_second_per_tvr_point"]

    core._save_settings(core.KairosSettings(
        pricing_overrides={"base_price_per_second_per_tvr_point": default_base + 1.5}
    ))
    body = client.get("/api/parameters").json()
    assert body["pricing"]["has_overrides"] is True
    assert body["pricing"]["base_price_per_second_per_tvr_point"] == pytest.approx(
        default_base + 1.5
    ), "the Parameters page must show what the engine actually prices with"


# --------------------------------------------------------------------------- #
# Schedule payload truth fields and dead-code removal.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not CSV_PATH.exists(), reason="no committed weekly plan on disk")
def test_schedule_payload_discloses_total_rows_beside_the_display_slice(client):
    body = client.get("/api/schedule").json()
    total = int(len(pd.read_csv(CSV_PATH, encoding="utf-8-sig")))
    assert body["break_schedule_total_rows"] == total
    assert len(body["break_schedule"]) == min(200, total)


def test_dead_day_key_helper_is_gone():
    import kairos_api.dashboard_api as dash

    assert not hasattr(dash, "_day_key")


def test_cold_overview_no_longer_computes_the_discarded_break_board(monkeypatch):
    """_build_compliance ignores its operations argument, so the cold overview
    must not compute the truncated break board at all. Probed on the unwrapped
    builder (no cache pollution, no background frontier)."""
    import kairos_api.dashboard_api as dash

    def boom(*args, **kwargs):
        raise AssertionError("dead full-board compute still runs on cold overview")

    monkeypatch.setattr(dash, "_build_break_operations", boom)
    monkeypatch.setattr(dash, "_load_programmes", lambda: pd.DataFrame())
    monkeypatch.setattr(dash, "_load_spots", lambda: pd.DataFrame())
    monkeypatch.setattr(dash, "_load_break_schedule", lambda: pd.DataFrame())
    monkeypatch.setattr(dash, "_plan_guardrail_items", lambda: [])
    body = dash._overview_cached.__wrapped__((), None)
    assert "summary" in body and "compliance" in body


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
