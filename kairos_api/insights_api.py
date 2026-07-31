"""Compatibility layer for the insight reads, after the wave-zero router split.

This module no longer defines anything. Its five routes moved to the per-owner
modules named in ``docs/ux-gauntlet/contracts/W0-1.md``: yield to
:mod:`kairos_api.yield_api`, gold to :mod:`kairos_api.gold_api`, the A/B to
:mod:`kairos_api.scenario_compare_api`, the audience-model status to
:mod:`kairos_api.model_audience_api`, and the make-good alerts to
:mod:`kairos_api.pacing_alerts_api`.

Four assistant modules and five test files import names from here, and none of
them is this piece's to edit, so every name this module defined before the split
still resolves from it, against the SAME objects, including the single lru_cache
instance behind the retention-cost band.

``router`` stays and is deliberately empty. It keeps the registration line in
server.py above the append marker unchanged, and mounting an empty router adds
nothing to the OpenAPI surface.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from kairos_api import (
    gold_api,
    model_audience_api,
    pacing_alerts_api,
    scenario_compare_api,
    split_compat,
    yield_api,
)
from kairos_api.gold_api import (  # noqa: F401
    _build_gold_breaks,
    _cell_or_none,
    _is_gold_truthy,
    gold_breaks,
)
from kairos_api.model_audience_api import model_audience  # noqa: F401
from kairos_api.pacing_alerts_api import (  # noqa: F401
    _reference_today,
    make_good_alerts,
)
from kairos_api.scenario_compare_api import (  # noqa: F401
    ScenarioCompareRequest,
    _build_scenario_compare,
    _delta,
    _scenario_summary,
    scenario_compare,
)
from kairos_api.yield_api import (  # noqa: F401
    _RETENTION_BAND_BASIS,
    _build_yield_per_second,
    _daypart_for_start,
    _optimistic_impact,
    _plan_cost_band,
    _plan_cost_band_cached,
    _server,
    scoped_yield_payload,
    yield_per_second,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# A substitution on this layer must still reach the code that reads the name, or
# a probe that measures a cache key here becomes a silent no-op. See
# kairos_api/split_compat.py.
split_compat.mirror_writes(__name__, (
    gold_api,
    model_audience_api,
    pacing_alerts_api,
    scenario_compare_api,
    yield_api,
))
