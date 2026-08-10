"""Plan, week: the parameters surface.

Split out of :mod:`kairos_api.scenario_api` under the 450-line law and named by
the helper rule. It carries one route, and that route is the fourth of the four
open reads section 4.5 of the specification names, so the affiliation split it
performs is the whole reason it deserves a file a reader can hold in mind.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from kairos_api.core import (
    MODELS_DIR,
    ROOT,
    OptimizerAssumptions,
    _ENGINE_AVAILABLE,
    _asdict,
    _load_settings,
    _model_dump,
    guardrails_from_settings,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scenario"])


@router.get("/api/parameters")
def parameters(request: Request) -> dict[str, Any]:
    """Every adjustable parameter the optimizer uses, in one place.

    Surfaces the guardrails (derived from the saved settings), the declared
    optimizer assumptions, the pricing model, and the caller's declared channel,
    so the dashboard can show and edit each legitimate operator parameter.

    **This is the fourth of the four open reads, and it is closed here rather
    than walled.** Section 4.5 of the specification lists this route beside
    ``/api/impact``, ``/api/model/audience`` and the calendar's ``model_context``
    as a read that leaks the training side. Walling the whole route would be
    wrong: the parameters are the operator's own settings and their own rate
    card, and an operator who cannot read them cannot work. What leaks is
    narrower and exact, three keys measured against the section 4.2 lexicon:
    ``coefficient_freshness`` (whose name is the lexicon word ``coefficient`` and
    whose body names changed training inputs), ``first_break_active`` and
    ``first_break_multiplier``, which are a gate verdict and its coefficient in
    all but name.

    So the fix is the one the calendar's ``model_context`` takes: those three
    keys are served only to a company-affiliated caller, and every account is
    served ``model_version`` instead, which is the two facts section 4.4 says a
    run surface needs, the model's date and whether that date is current.
    ``training_visible`` states which of the two copies the caller received, so a
    surface never has to infer it from an absence.
    """
    from kairos_api.affiliation_wall import is_company

    company = is_company(request)
    settings = _load_settings()
    payload: dict[str, Any] = {"settings": _model_dump(settings)}
    # Real campaign-flight count from the same loader the pacing signal uses, so
    # the dashboard can key its pacing-inactive note on truth: pacing is an
    # exact no-op until flight rows exist. Null (unknown) when the loader itself
    # fails, never a fabricated zero.
    try:
        from kairos.optimize.pacing import load_campaigns

        payload["flights_count"] = len(load_campaigns())
    except Exception:
        logger.exception("campaign flights count unavailable")
        payload["flights_count"] = None
    if not _ENGINE_AVAILABLE:
        payload["engine"] = "unavailable"
        return payload
    payload["guardrails"] = _asdict(guardrails_from_settings(_model_dump(settings)))
    payload["assumptions"] = _asdict(OptimizerAssumptions())
    payload["operator_channel"] = settings.operator_channel
    # Honest flag: when no channel is selected the competitor-boundary filter is
    # inactive (constraints match any channel). The dashboard uses this to warn
    # the operator so they know to visit OperatorChannelPanel and pick a channel.
    payload["operator_channel_unset"] = not bool(settings.operator_channel)
    # This route is loaded by every operator workspace.  A channel operator may
    # read their own declared channel, but the names of rival channels are not a
    # parameter they need and must not ride along in the shared bootstrap
    # payload.  The admin-only declaration workflow has its own scoped endpoint
    # (/api/rules/operator-channel), which remains the legitimate place for the
    # full EPG-backed option list.
    owned_channels = [settings.operator_channel] if settings.operator_channel else []
    payload["channels"] = owned_channels
    payload["available_channels"] = owned_channels
    try:
        # The LIVE rate card: the YAML defaults with the operator's saved
        # pricing_overrides merged on top (pricing_from_settings, the same seam
        # the optimizer, forecast and spot export price with). A bare from_yaml
        # here showed the shipped defaults while the engine priced with the
        # operator's edits. has_overrides marks whether any edit is in effect.
        from kairos.optimize.pricing import pricing_from_settings

        pricing = pricing_from_settings(_model_dump(settings))
        payload["pricing"] = {
            "base_price_per_second_per_tvr_point": pricing.base_price,
            "program_type_premiums": pricing.program_type_premiums,
            "ad_type_premiums": pricing.ad_type_premiums,
            "position_premiums": {str(k): v for k, v in pricing.position_premiums.items()},
            "day_of_week_premiums": {str(k): v for k, v in pricing.day_of_week_premiums.items()},
            "has_overrides": bool(settings.pricing_overrides),
        }
    except Exception as exc:  # pragma: no cover - config dependent
        payload["pricing"] = {"error": str(exc)[:200]}
    # Honest freshness of the measured retention coefficients: re-hash the source
    # files the coefficients were computed from and report fresh/stale/unknown.
    # The verdict itself is training content, so it is folded into the two facts
    # every account may read (the model version's date, and whether it is
    # current) and served in full only to the company side.
    payload["training_visible"] = bool(company)
    try:
        from kairos.model.freshness import coefficient_freshness
        from kairos.model.measure import read_coefficients_metadata

        metadata = read_coefficients_metadata(MODELS_DIR / "tv_break_coefficients.json")
        freshness = coefficient_freshness(metadata, root=ROOT)
        payload["model_version"] = {
            "trained_at": freshness.get("computed_at"),
            "current": freshness.get("status") == "fresh",
            "status": freshness.get("status"),
        }
        if company:
            payload["coefficient_freshness"] = freshness
            # The self-activating first-break retention lever, read from the
            # measured coefficients metadata. It is a gate verdict and its
            # multiplier, so it stays on the company side by the same rule.
            payload["first_break_active"] = bool(metadata.get("first_break_active", False))
            try:
                payload["first_break_multiplier"] = float(metadata.get("first_break_multiplier", 1.0) or 1.0)
            except (TypeError, ValueError):
                payload["first_break_multiplier"] = 1.0
    except Exception as exc:  # pragma: no cover - defensive, never blocks parameters
        payload["model_version"] = {"trained_at": None, "current": False, "status": "unknown"}
        if company:
            payload["coefficient_freshness"] = {
                "status": "unknown",
                "computed_at": None,
                "changed_files": [],
                "reason": f"freshness check unavailable: {str(exc)[:160]}",
            }
            payload["first_break_active"] = False
            payload["first_break_multiplier"] = 1.0
    return payload
