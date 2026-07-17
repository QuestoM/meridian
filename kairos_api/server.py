"""FastAPI server for the Kairos revenue optimization dashboard.

Thin composition root for the modular monolith. This file owns only the
app-level concerns: constructing the FastAPI app, the auth session guard and
CORS middleware, the background cache warm-up, mounting every domain router, and
serving the built dashboard. All endpoint logic lives in focused routers over
the shared kernel (:mod:`kairos_api.core`).

The dashboard read endpoints and their builders, the revenue-vs-retention
frontier machinery, and the decision shortcut moved into
:mod:`kairos_api.dashboard_api`; the catalog and scenario endpoints into their
own routers. The moved names are re-exported below under their original names so
existing references (the assistant and catalog routers, the test suite, and the
warm-up here) keep resolving against the SAME objects, including the single
lru_cache instances and the one frontier background-thread state and its lock.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Load the operator's local .env (repo root) before anything reads the
# environment. override=False keeps a variable that is already exported in the
# real environment authoritative, so tests and deployments that set their own
# values are never clobbered by the file. The .env file itself is gitignored;
# .env.example documents the recognised variables without secrets.
try:
    from pathlib import Path as _Path

    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:  # pragma: no cover - dotenv stays optional at runtime
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Kernel: settings, paths, cached loaders and the small shared response helpers.
# Re-exported for the domain routers and the tests that import them from here.
from kairos_api.core import (  # noqa: F401  (re-exported for domain routers and tests)
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KAIROS_CHANNELS,
    KairosSettings,
    OptimizerAssumptions,
    PricingModel,
    _ENGINE_AVAILABLE,
    _asdict,
    _augment_segment_ids,
    _load_break_schedule,
    _load_programmes,
    _load_settings,
    _load_spots,
    _load_spots_cached,
    _model_dump,
    _money,
    _pacing_call_kwargs,
    _percent,
    _ratio,
    _read_csv,
    _read_csv_cached,
    _records,
    _reference_today,
    _risk_from_retention,
    _row_anchor,
    _safe_number,
    _save_settings,
    _series,
    _settings_to_guardrails,
    _signature,
    _summarize_schedule,
    _time_to_seconds,
    build_weekly_schedule,
    guardrails_from_settings,
    optimize_day_plan,
    run_scenario,
    write_weekly_schedule,
)

app = FastAPI(
    title="Kairos API",
    version="0.1.0",
    description="Operational API for TV ad break revenue optimization.",
)

# Login / user system: one session guard in front of every /api route. It is
# registered before CORSMiddleware is added below, which keeps CORS outermost
# so denial responses still carry CORS headers. Enforcement only activates
# once the operator seeds data/auth/users.json (scripts/init_auth.py); see the
# kairos_api.auth module docstring for the full lifecycle and the
# KAIROS_AUTH_DISABLED escape hatch.
from kairos_api.auth import enforce_request as _auth_enforce_request  # noqa: E402
from kairos_api.auth import router as _auth_router  # noqa: E402


@app.middleware("http")
async def _auth_session_guard(request, call_next):
    denial = _auth_enforce_request(request)
    if denial is not None:
        return denial
    return await call_next(request)


app.include_router(_auth_router)

# System-wide activity log. Registered after the auth guard above, which makes
# it the outer of the two (Starlette stacks later middleware outside earlier
# ones), so it observes every mutating /api request with its final status,
# including requests the guard denies (recorded as "anonymous"). CORSMiddleware
# is added below and therefore stays outermost. Login and logout are excluded
# inside the recorder and appended as dedicated events by kairos_api.auth, so
# the two requests whose bodies carry credentials never transit the recorder;
# and a recording failure is swallowed there, never failing the request.
from kairos_api.activity_log import record_api_mutation as _record_api_mutation  # noqa: E402
from kairos_api.activity_log import router as _activity_log_router  # noqa: E402


@app.middleware("http")
async def _activity_recorder(request, call_next):
    return await _record_api_mutation(request, call_next)


app.include_router(_activity_log_router)

# Default to the local dashboard origins: the Vite dev server (5173/5174) and a
# 3000 fallback, on both localhost and 127.0.0.1. Without the dev port here the
# browser blocks the cross-origin fetch and the dashboard shows its offline "demo
# data" fallback. Override with KAIROS_CORS_ORIGINS for a deployed origin.
allowed_origins = os.getenv(
    "KAIROS_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:5174,http://127.0.0.1:5174",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Operational capabilities live in focused modules to keep this file lean. The
# dashboard reads and their builders (overview + frontier, schedule canvas, break
# board, segment inspector, compliance, break-decisions) live in dashboard_api;
# the catalog and scenario endpoints in their own routers. Every moved builder and
# cache is imported back under its original name so existing references (tests, the
# assistant and catalog routers, the warm-up below) keep working against the SAME
# objects, including the single lru_cache instances and the one frontier background
# state and its lock.
from kairos_api.advertiser_conditions import router as advertiser_conditions_router  # noqa: E402
from kairos_api.advertisers import router as advertisers_router  # noqa: E402
from kairos_api.constraints import router as constraints_router  # noqa: E402
from kairos_api.exporters import router as exporters_router  # noqa: E402
from kairos_api.overrides import router as overrides_router  # noqa: E402
from kairos_api.insights_api import router as insights_router  # noqa: E402
from kairos_api.pricing_api import router as pricing_router  # noqa: E402
from kairos_api.uploads import router as uploads_router  # noqa: E402

# Dashboard read endpoints and their builders. Re-exported here (F401) so the
# assistant/catalog routers and the test suite keep importing them from
# kairos_api.server, and so the warm-up below shares the same cache objects.
from kairos_api.dashboard_api import (  # noqa: E402,F401  (re-exported for compatibility)
    BreakDecisionRequest,
    GuardrailBreak,
    _break_operations_cached,
    _build_break_operations,
    _build_compliance,
    _build_recommendations,
    _build_schedule_canvas,
    _build_schedule_segments,
    _decision_log,
    _frontier_async,
    _frontier_bg_lock,
    _frontier_bg_state,
    _frontier_data_signature,
    _frontier_points_cached,
    _guardrail_breaks_from_operations,
    _guardrail_compliance_from_breaks,
    _overview_cached,
    _owned_representative_day,
    _parse_frontier_scope,
    _plan_by_program_key,
    _plan_guardrail_items,
    _proposed_kind,
    _resolve_decision,
    _schedule_cached,
    _schedule_segments_cached,
    _segment_overrides,
)
from kairos_api.dashboard_api import router as dashboard_router  # noqa: E402

app.include_router(uploads_router)
app.include_router(advertisers_router)
app.include_router(advertiser_conditions_router)
app.include_router(exporters_router)
app.include_router(overrides_router)
app.include_router(constraints_router)
app.include_router(insights_router)
app.include_router(pricing_router)

from kairos_api.recompute_api import router as recompute_router  # noqa: E402

app.include_router(recompute_router)

from kairos_api.settings_api import router as settings_router  # noqa: E402

app.include_router(settings_router)

from kairos_api.assistant import router as assistant_router  # noqa: E402

app.include_router(assistant_router)

from kairos_api.version_store import router as version_store_router  # noqa: E402

app.include_router(version_store_router)

# Catalog and scenario endpoints live in their own domain routers. The moved
# builders are imported back under their original names so existing references
# (tests, the startup warm-up below) keep working against the SAME objects,
# including the single lru_cache instances.
from kairos_api.catalog_api import (  # noqa: E402,F401  (re-exported for tests and warm-up)
    _break_library_cached,
    _build_break_library,
    _build_campaigns,
    _build_forecast_scenarios,
    _build_forecasts,
    _build_inventory,
    _build_reports,
    _campaigns_cached,
    _forecasts_cached,
    _impact_cached,
    _inventory_cached,
    _load_measured_impact_summary,
    _pooling_note,
    _reports_cached,
    _segment_key,
    _source_file_paths,
    _weighted_impact_rows,
)
from kairos_api.catalog_api import router as catalog_router  # noqa: E402
from kairos_api.scenario_api import (  # noqa: E402,F401  (re-exported for compatibility)
    OptimizePlanRequest,
    ScenarioRequest,
    _build_optimizer_plan,
    _scenario_cached,
    _warm_scenario,
)
from kairos_api.scenario_api import router as scenario_router  # noqa: E402

app.include_router(catalog_router)
app.include_router(scenario_router)
app.include_router(dashboard_router)


@app.on_event("startup")
def _warm_overview_cache() -> None:
    """Pre-compute the expensive caches in a background thread so the first
    dashboard load is not blocked. Three endpoints are slow on a cold cache:
    /api/overview and /api/forecasts each sweep several real optimizations, and
    /api/campaigns + /api/parameters both trigger the one-time spots/EPG parse.
    All of them are GIL-bound pure-Python work, so warm them sequentially in a
    single thread (parallel warmers would just starve each other) and the
    dashboard's parallel fetch finds every cache hot. Failures are swallowed: a
    cold cache only means the first real request pays the cost.
    """

    def _run() -> None:
        steps = (
            ("overview", lambda: _overview_cached(
                _signature([
                    OUTPUT_DIR / "weekly_break_schedule.csv",
                    DATA_DIR / "reference" / "Programmes.xlsx",
                    DATA_DIR / "reference" / "Spots.xlsx",
                    DATA_DIR / "Programmes.csv",
                    DATA_DIR / "Spots.csv",
                    SETTINGS_PATH,
                ]),
                None,
            )),
            # Warm-up keys must mirror the routes' keys exactly, or the warm-up
            # populates entries the routes never read and the first real request
            # pays the cold cost anyway.
            ("forecasts", lambda: _forecasts_cached(
                _signature([
                    OUTPUT_DIR / "weekly_break_schedule.csv",
                    ROOT / "optimization_results.csv",
                    SETTINGS_PATH,
                    DATA_DIR / "reference" / "Programmes.xlsx",
                    DATA_DIR / "Programmes.csv",
                ])
            )),
            ("campaigns", lambda: _campaigns_cached(_signature([
                DATA_DIR / "reference" / "Spots.xlsx",
                DATA_DIR / "Spots.csv",
            ]))),
            ("inventory", lambda: _inventory_cached(_signature([
                DATA_DIR / "reference" / "Spots.xlsx",
                DATA_DIR / "Spots.csv",
            ]))),
            # Kick off the background frontier sweep at startup so it is "ready"
            # by the time the operator opens the dashboard (it spawns its own
            # thread and returns immediately, never blocking warm-up). The same
            # sweep also computes the net-comparison bundle, so that surface warms
            # with it and no second background thread is needed.
            ("frontier", lambda: _frontier_async(_load_settings(), None)),
            # Prime the scenario-preview cache on the owned channel-day scope so the
            # first slider read is a cache hit, reusing this single warm thread.
            ("scenario", _warm_scenario),
        )
        for name, step in steps:
            try:
                step()
            except Exception:
                logger.exception("cache warm-up failed for %s", name)

    threading.Thread(target=_run, name="kairos-cache-warm", daemon=True).start()


# Serve the built dashboard (Vite `dist/`) from the same container in production.
# Mounted last so it never shadows the `/api/*` routes above; only active when a
# build is present, so local API-only runs are unaffected.
_DASHBOARD_DIST = ROOT / "tv-break-dashboard" / "dist"
if _DASHBOARD_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles  # noqa: E402

    app.mount("/", StaticFiles(directory=str(_DASHBOARD_DIST), html=True), name="dashboard")
