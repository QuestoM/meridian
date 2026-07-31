"""W0-1: the split modules keep every name they defined, against the same objects.

Fourteen production modules and twenty test files import names from
dashboard_api, insights_api, catalog_api and version_store, and none of those
files is W0-1's to edit. So the wave-zero split holds only if every name those
four modules defined before it still resolves from them, and resolves to the
SAME object: the single lru_cache instances the cache tests clear and count, the
one frontier lock, the one frontier state dict, and the version router three
fixtures mount into their own app.

The last two cases carry the assertions two frozen tests lose when their patch
target moves module. They are asserted here against the new modules while C2
rules on the test-change request in docs/ux-gauntlet/contracts/W0-1.md section 7.
"""

from __future__ import annotations

import pandas as pd
import pytest

DASHBOARD_NAMES = (
    "logger", "router", "BreakDecisionRequest", "_program_datetime_columns",
    "_build_schedule_canvas", "_plan_by_program_key", "_build_break_operations",
    "_build_schedule_segments", "_proposed_kind", "_build_recommendations",
    "_parse_frontier_scope", "_frontier_data_signature", "_frontier_points_cached",
    "_owned_representative_day", "_owned_scope", "NET_POINT_ID",
    "_scenario_plan_money", "_net_bundle_failure", "_frontier_net_bundle_cached",
    "_frontier_bg_lock", "_frontier_bg_state", "_frontier_async", "_frontier_state",
    "_infer_hourly_ad_seconds", "_infer_hourly_break_counts",
    "_plan_guardrail_items_cached", "_plan_guardrail_items",
    "_guardrail_breaks_from_operations", "_max_group_sum", "_max_group_count",
    "_min_break_spacing_seconds", "_guardrail_compliance_from_breaks",
    "_build_compliance", "_overview_cached", "_schedule_cached",
    "_schedule_segments_cached", "_break_operations_cached", "_decision_log",
    "_resolve_decision", "_segment_overrides", "compliance", "overview",
    "schedule", "schedule_segments", "schedule_segment_detail",
    "break_operations", "break_decisions", "create_break_decision",
)

INSIGHTS_NAMES = (
    "logger", "router", "ScenarioCompareRequest", "_server", "_daypart_for_start",
    "_RETENTION_BAND_BASIS", "_optimistic_impact", "_plan_cost_band_cached",
    "_plan_cost_band", "_build_yield_per_second", "scoped_yield_payload",
    "yield_per_second", "_is_gold_truthy", "_cell_or_none", "_build_gold_breaks",
    "gold_breaks", "_scenario_summary", "_delta", "_build_scenario_compare",
    "scenario_compare", "model_audience", "make_good_alerts", "_reference_today",
)

CATALOG_NAMES = (
    "logger", "router", "_segment_key", "_weighted_impact_rows", "_pooling_note",
    "_load_measured_impact_summary", "_build_inventory", "_build_campaigns",
    "_build_break_library", "_build_forecasts", "_build_forecast_scenarios",
    "_source_file_paths", "_build_reports", "_inventory_cached",
    "_break_library_cached", "_campaigns_cached", "_forecasts_cached",
    "_reports_cached", "_impact_cached", "impact", "inventory", "break_library",
    "campaigns", "forecasts", "reports", "files",
)

VERSION_STORE_NAMES = (
    "ROOT", "VERSIONS_DIR_ENV", "ASSISTANT_DIR_ENV", "MAX_VERSIONS", "router",
    "_LOGICAL_ORDER", "_VERSION_ID_RE", "_require_version_id", "_now_iso",
    "_versions_root", "_logical_path", "_ID_COLUMN", "_snapshot_name", "_actor",
    "_require_session", "_require_writer", "_hash_bytes", "_capture",
    "_manifest_path", "_read_manifest", "_all_manifests", "_next_seq",
    "_atomic_write", "_identical", "_prune", "snapshot", "_LOGICAL_FOR_KIND",
    "snapshot_assistant_apply", "_audit", "_read_json", "_read_rows",
    "_version_bytes", "_current_bytes", "_settings_diff", "_rows_diff",
    "_diff_logical", "_restore_logical", "snapshot_manual_edit", "_SCOPE_NOTE",
    "_public_entry", "list_versions", "version_diff", "RestoreRequest",
    "restore_version", "LabelRequest", "create_snapshot", "rename_version",
)


@pytest.mark.parametrize(
    "module_name,names",
    [
        ("kairos_api.dashboard_api", DASHBOARD_NAMES),
        ("kairos_api.insights_api", INSIGHTS_NAMES),
        ("kairos_api.catalog_api", CATALOG_NAMES),
        ("kairos_api.version_store", VERSION_STORE_NAMES),
    ],
)
def test_every_pre_split_name_still_resolves(module_name, names) -> None:
    import importlib

    module = importlib.import_module(module_name)
    missing = [name for name in names if not hasattr(module, name)]
    assert missing == [], f"{module_name} no longer exposes {missing}"


def test_the_shared_caches_are_one_object_not_two() -> None:
    """A re-export must not clone a cache, or clearing one leaves the other warm."""
    from kairos_api import (
        campaigns_read,
        catalog_api,
        dashboard_api,
        day_api,
        downloads_api,
        insights_api,
        model_impact_api,
        overview_api,
        plan_read_frontier,
        plan_read_guardrails,
        scenario_compare_api,
        week_api,
        yield_api,
    )

    assert dashboard_api._overview_cached is overview_api._overview_cached
    assert dashboard_api._schedule_cached is week_api._schedule_cached
    assert dashboard_api._break_operations_cached is day_api._break_operations_cached
    assert dashboard_api._schedule_segments_cached is day_api._schedule_segments_cached
    assert dashboard_api._frontier_points_cached is plan_read_frontier.frontier_points_cached
    assert dashboard_api._frontier_bg_state is plan_read_frontier.frontier_bg_state
    assert dashboard_api._frontier_bg_lock is plan_read_frontier.frontier_bg_lock
    assert dashboard_api._plan_guardrail_items_cached is plan_read_guardrails.plan_guardrail_items_cached
    assert catalog_api._forecasts_cached is scenario_compare_api._forecasts_cached
    assert catalog_api._inventory_cached is week_api._inventory_cached
    assert catalog_api._campaigns_cached is campaigns_read._campaigns_cached
    assert catalog_api._break_library_cached is day_api._break_library_cached
    assert catalog_api._reports_cached is downloads_api._reports_cached
    assert catalog_api._impact_cached is model_impact_api._impact_cached
    assert insights_api._plan_cost_band_cached is yield_api._plan_cost_band_cached


def test_the_version_router_is_the_history_router() -> None:
    """Three fixtures mount version_store.router into their own app, so the name
    must resolve to the router the routes are now defined on."""
    from kairos_api import history_api, version_store

    assert version_store.router is history_api.router
    paths = {route.path for route in version_store.router.routes}
    assert paths == {
        "/api/versions",
        "/api/versions/{version_id}",
        "/api/versions/{version_id}/diff",
        "/api/versions/{version_id}/restore",
        "/api/versions/snapshot",
    }


def test_a_substitution_on_a_compatibility_layer_reaches_the_reader() -> None:
    """The layers re-export names that four frozen probes substitute to measure a
    cache key or an unwanted call. A write must reach the module that now reads
    the name, including where the name lost its underscore on the move, and the
    restore must reach it too."""
    from kairos_api import (
        dashboard_api,
        overview_api,
        plan_read,
        plan_read_compliance,
        plan_read_guardrails,
        week_api,
    )

    cases = (
        ("_plan_guardrail_items", plan_read_guardrails, "plan_guardrail_items"),
        ("_build_break_operations", plan_read, "build_break_operations"),
        ("_build_compliance", plan_read_compliance, "build_compliance"),
        ("_load_programmes", overview_api, "_load_programmes"),
        ("_load_programmes", week_api, "_load_programmes"),
        ("SETTINGS_PATH", week_api, "SETTINGS_PATH"),
    )
    for name, target, target_name in cases:
        sentinel = object()
        original = getattr(dashboard_api, name)
        setattr(dashboard_api, name, sentinel)
        try:
            assert getattr(target, target_name) is sentinel, (
                f"substituting {name} did not reach {target.__name__}.{target_name}")
        finally:
            setattr(dashboard_api, name, original)
        assert getattr(target, target_name) is original, (
            f"restoring {name} did not reach {target.__name__}.{target_name}")


def test_cold_overview_does_not_compute_the_discarded_break_board(monkeypatch) -> None:
    """build_compliance ignores its operations argument, so the cold overview must
    not compute the truncated break board at all. Probed on the unwrapped builder,
    so no cache is polluted and no background frontier starts.

    This is the assertion tests/test_qa2_api_seams.py made before the split, when
    every name it patched was a module attribute of dashboard_api.
    """
    from kairos_api import overview_api, plan_read, plan_read_guardrails

    def boom(*args, **kwargs):
        raise AssertionError("dead full-board compute still runs on cold overview")

    monkeypatch.setattr(plan_read, "build_break_operations", boom)
    monkeypatch.setattr(overview_api, "_load_programmes", lambda: pd.DataFrame())
    monkeypatch.setattr(overview_api, "_load_spots", lambda: pd.DataFrame())
    monkeypatch.setattr(overview_api, "_load_break_schedule", lambda: pd.DataFrame())
    monkeypatch.setattr(plan_read_guardrails, "plan_guardrail_items", lambda: [])
    body = overview_api._overview_cached.__wrapped__((), None)
    assert "summary" in body and "compliance" in body


def test_compliance_fallback_survives_an_empty_schedule(monkeypatch) -> None:
    """With no committed plan the geometry is empty, so the verdict falls back to
    the schedule summary. An unknown retention is reported honestly, never
    asserted compliant, and the comparison never raises on a None.

    This is the assertion tests/test_qa_known_bugs_20260706.py made before the
    split, when the geometry was a module attribute of dashboard_api.
    """
    from kairos_api import plan_read_compliance, plan_read_guardrails
    from kairos_api.core import KairosSettings

    monkeypatch.setattr(plan_read_guardrails, "plan_guardrail_items", lambda: [])
    result = plan_read_compliance.build_compliance(pd.DataFrame(), KairosSettings())
    assert result["status"] in {"compliant", "at_risk", "unknown"}
    retention = next(check for check in result["checks"] if check["id"] == "retention_floor")
    assert retention["observed"] is None or isinstance(retention["observed"], (int, float))
    if retention["observed"] is None:
        assert retention["status"] == "unknown"


def test_the_compliance_verdict_is_one_object_for_all_three_readers() -> None:
    """Today prints it, Rules serves it and Sources counts its checks, so all
    three must reach the same builder rather than three copies of it."""
    import inspect

    from kairos_api import compliance_api, downloads_api, overview_api, plan_read_compliance

    assert "plan_read_compliance.build_compliance" in inspect.getsource(overview_api._overview_cached)
    assert "plan_read_compliance.build_compliance" in inspect.getsource(compliance_api.compliance)
    assert "build_compliance(schedule, settings)" in inspect.getsource(downloads_api._build_reports)
    assert callable(plan_read_compliance.build_compliance)
