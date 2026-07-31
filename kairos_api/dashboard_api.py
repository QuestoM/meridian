"""Compatibility layer for the dashboard reads, after the wave-zero router split.

This module no longer defines anything. Its eight routes moved to the per-owner
modules named in ``docs/ux-gauntlet/contracts/W0-1.md`` (the overview and the
decision plane to :mod:`kairos_api.overview_api` and
:mod:`kairos_api.overview_api_decisions`, the schedule canvas to
:mod:`kairos_api.week_api`, the board, the override targets, the inspector and
the break list to :mod:`kairos_api.day_api`, the verdict to
:mod:`kairos_api.compliance_api`), and the machinery more than one owner reads
moved to the frozen plan-read layer (:mod:`kairos_api.plan_read` and its scope,
frontier, guardrail and compliance helpers).

Fourteen modules and twenty test files import names from here, and none of them
is this piece's to edit, so every name this module defined before the split still
resolves from it, against the SAME objects: the single lru_cache instances, the
one frontier lock and the one frontier state dict.

``router`` stays and is deliberately empty. It keeps the registration line in
server.py above the append marker unchanged, and mounting an empty router adds
nothing to the OpenAPI surface.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from kairos.optimize.guardrails import Break as GuardrailBreak  # noqa: F401
from kairos.optimize.guardrails import evaluate as evaluate_guardrails  # noqa: F401

from kairos_api import (
    compliance_api,
    day_api,
    overview_api,
    overview_api_decisions,
    plan_read,
    plan_read_compliance,
    plan_read_frontier,
    plan_read_guardrails,
    plan_read_scope,
    split_compat,
    week_api,
)
from kairos_api.compliance_api import compliance  # noqa: F401
from kairos_api.core import (  # noqa: F401  (re-exported, the pre-split namespace)
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KairosSettings,
    _ENGINE_AVAILABLE,
    _augment_segment_ids,
    _load_break_schedule,
    _load_programmes,
    _load_settings,
    _load_spots,
    _model_dump,
    _money,
    _pacing_call_kwargs,
    _percent,
    _plan_segment_index,
    _ratio,
    _row_anchor,
    _safe_number,
    _settings_to_guardrails,
    _signature,
    _summarize_schedule,
    _time_to_seconds,
    run_scenario,
)
from kairos_api.day_api import (  # noqa: F401
    _break_operations_cached,
    _build_schedule_segments,
    _schedule_segments_cached,
    _segment_overrides,
    break_operations,
    schedule_segment_detail,
    schedule_segments,
)
from kairos_api.overview_api import (  # noqa: F401
    _build_recommendations,
    _overview_cached,
    _proposed_kind,
    overview,
)
from kairos_api.overview_api_decisions import (  # noqa: F401
    BreakDecisionRequest,
    _decision_log,
    _resolve_decision,
    break_decisions,
    create_break_decision,
)
from kairos_api.plan_read import (  # noqa: F401
    build_break_operations as _build_break_operations,
)
from kairos_api.plan_read import (  # noqa: F401
    plan_by_program_key as _plan_by_program_key,
)
from kairos_api.plan_read import (  # noqa: F401
    program_datetime_columns as _program_datetime_columns,
)
from kairos_api.plan_read_compliance import (  # noqa: F401
    build_compliance as _build_compliance,
)
from kairos_api.plan_read_frontier import (  # noqa: F401
    NET_POINT_ID,
    frontier_async as _frontier_async,
    frontier_bg_lock as _frontier_bg_lock,
    frontier_bg_state as _frontier_bg_state,
    frontier_net_bundle_cached as _frontier_net_bundle_cached,
    frontier_points_cached as _frontier_points_cached,
    frontier_state as _frontier_state,
    net_bundle_failure as _net_bundle_failure,
    scenario_plan_money as _scenario_plan_money,
)
from kairos_api.plan_read_guardrails import (  # noqa: F401
    _max_group_count,
    _max_group_sum,
    _min_break_spacing_seconds,
    guardrail_breaks_from_operations as _guardrail_breaks_from_operations,
    guardrail_compliance_from_breaks as _guardrail_compliance_from_breaks,
    infer_hourly_ad_seconds as _infer_hourly_ad_seconds,
    infer_hourly_break_counts as _infer_hourly_break_counts,
    plan_guardrail_items as _plan_guardrail_items,
    plan_guardrail_items_cached as _plan_guardrail_items_cached,
)
from kairos_api.plan_read_scope import (  # noqa: F401
    frontier_data_signature as _frontier_data_signature,
    owned_representative_day as _owned_representative_day,
    owned_scope as _owned_scope,
    parse_frontier_scope as _parse_frontier_scope,
)
from kairos_api.week_api import (  # noqa: F401
    _build_schedule_canvas,
    _schedule_cached,
    schedule,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# A substitution on this layer must still reach the code that reads the name, or
# a probe that measures a cache key here becomes a silent no-op. See
# kairos_api/split_compat.py.
split_compat.mirror_writes(__name__, (
    compliance_api,
    day_api,
    overview_api,
    overview_api_decisions,
    plan_read,
    plan_read_compliance,
    plan_read_frontier,
    plan_read_guardrails,
    plan_read_scope,
    week_api,
))
