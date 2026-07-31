"""Compatibility layer for the catalog reads, after the wave-zero router split.

This module no longer defines anything. Its seven routes moved to the per-owner
modules named in ``docs/ux-gauntlet/contracts/W0-1.md``: the coefficients to
:mod:`kairos_api.model_impact_api`, the supply view to
:mod:`kairos_api.week_api`, the break list to :mod:`kairos_api.day_api`, the
campaign rollup to :mod:`kairos_api.campaigns_read`, the forecast rows to
:mod:`kairos_api.scenario_compare_api`, and the report shelf and source-file
audit to :mod:`kairos_api.downloads_api`.

Two assistant modules and three test files import names from here, and none of
them is this piece's to edit, so every name this module defined before the split
still resolves from it, against the SAME objects, including the single lru_cache
instances the cache-key tests clear and count.

``router`` stays and is deliberately empty. It keeps the registration line in
server.py above the append marker unchanged, and mounting an empty router adds
nothing to the OpenAPI surface.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from kairos_api import (
    campaigns_read,
    day_api,
    downloads_api,
    model_impact_api,
    plan_read_compliance,
    scenario_compare_api,
    split_compat,
    week_api,
)
from kairos_api.campaigns_read import (  # noqa: F401
    _build_campaigns,
    _campaigns_cached,
    campaigns,
)
from kairos_api.core import (  # noqa: F401  (re-exported, the pre-split namespace)
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KairosSettings,
    _load_break_schedule,
    _load_settings,
    _load_spots,
    _model_dump,
    _money,
    _percent,
    _records,
    _reference_today,
    _safe_number,
    _series,
    _signature,
    _summarize_schedule,
    run_scenario,
)
from kairos_api.day_api import (  # noqa: F401
    _break_library_cached,
    _build_break_library,
    break_library,
)
from kairos_api.downloads_api import (  # noqa: F401
    _build_reports,
    _reports_cached,
    _source_file_paths,
    files,
    reports,
)
from kairos_api.model_impact_api import (  # noqa: F401
    _impact_cached,
    _load_measured_impact_summary,
    _pooling_note,
    _segment_key,
    _weighted_impact_rows,
    impact,
)
from kairos_api.scenario_compare_api import (  # noqa: F401
    _build_forecast_scenarios,
    _build_forecasts,
    _forecasts_cached,
    forecasts,
)
from kairos_api.week_api import (  # noqa: F401
    _build_inventory,
    _inventory_cached,
    inventory,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# A substitution on this layer must still reach the code that reads the name, or
# a probe that measures a cache key here becomes a silent no-op. See
# kairos_api/split_compat.py.
split_compat.mirror_writes(__name__, (
    campaigns_read,
    day_api,
    downloads_api,
    model_impact_api,
    plan_read_compliance,
    scenario_compare_api,
    week_api,
))
