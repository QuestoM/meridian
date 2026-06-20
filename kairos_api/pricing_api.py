"""Pricing hierarchy endpoints: read, tune, and test the rate card from the dashboard.

The operator owns the rate card. These endpoints let the dashboard read the full
pricing hierarchy (base CPP plus the named premium layers), edit any value or flip a
layer's activation, and test the price of any slot with a full per-layer breakdown.

Every number traces to base x named layers x named overrides (Law 9). The operator's
edits persist in KairosSettings.pricing_overrides and are deep-merged onto the YAML rate
card by PricingModel.from_config, the same constructor the optimizer, dashboard forecast
and spot export use (kairos.optimize.pricing.pricing_from_settings). So a saved edit is
genuinely live: it changes the next computed schedule and forecast, not just this view.
An empty override set is an exact identity to the shipped rate card, and the position,
ad-type and show layers ship activation-OFF, so revenue is unchanged until the operator
deliberately turns a layer on here (a visible, one-click decision, never silent).

This module keeps server.py lean: it imports the settings load/save helpers from
server.py and the pricing engine from kairos.optimize, rather than re-deriving them.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.layer_overrides import apply_overrides, resolve_layer_overrides
from kairos.optimize.price_guardrails import Guardrails
from kairos.optimize.pricing import PriceBreakdown, PricingModel, _deep_merge

router = APIRouter(tags=["pricing"])

# Which layers are multiplied into live revenue today, and a one-line description each.
# program and day are always live (they are the existing segment_premium). show, position
# and ad-type are gated on their activation flag because their configured multipliers are
# not 1.0, so turning them on moves real revenue.
_LAYER_META = [
    {"name": "base", "kind": "base", "always_live": True,
     "description": "Base price per rating point per second (channel rate). Not a premium."},
    {"name": "program", "kind": "premium", "always_live": True,
     "description": "Program-class premium (News, prime shows, other). Always applied."},
    {"name": "day", "kind": "premium", "always_live": True,
     "description": "Day-of-week premium. Always applied."},
    {"name": "show", "kind": "premium", "always_live": False, "activation_key": "show",
     "description": "Per-show premium (for example Big Brother). Stacks on the program class."},
    {"name": "position", "kind": "premium", "always_live": False, "activation_key": "position",
     "description": "Position-in-break premium (first, second, last). Off until activated."},
    {"name": "ad_type", "kind": "premium", "always_live": False, "activation_key": "ad_type",
     "description": "Ad-type premium (commercial, sponsorship, promo). Off until activated."},
]


def _table(model: PricingModel, name: str) -> dict[str, Any]:
    if name == "program":
        return {str(k): v for k, v in model.program_type_premiums.items()}
    if name == "day":
        return {str(k): v for k, v in model.day_of_week_premiums.items()}
    if name == "show":
        return {str(k): v for k, v in model.show_premiums.items()}
    if name == "position":
        return {str(k): v for k, v in model.position_premiums.items()}
    if name == "ad_type":
        return {str(k): v for k, v in model.ad_type_premiums.items()}
    return {}


def _state_payload(settings: Any) -> dict[str, Any]:
    """The full pricing hierarchy: effective values, YAML defaults, and activation.

    The dashboard renders the effective model and marks a value as operator-edited by
    diffing against ``defaults``. Activation flags say which layers move live revenue.
    """
    overrides = getattr(settings, "pricing_overrides", None) or {}
    effective = PricingModel.from_config(overrides)
    defaults = PricingModel.from_yaml()
    layers: list[dict[str, Any]] = []
    for meta in _LAYER_META:
        if meta["name"] == "base":
            layers.append({
                "name": "base", "kind": "base", "description": meta["description"],
                "value": effective.base_price, "default": defaults.base_price,
                "live_today": True,
            })
            continue
        enabled = True if meta["always_live"] else bool(
            getattr(effective, f"enable_{meta['name']}", False)
        )
        layers.append({
            "name": meta["name"], "kind": "premium",
            "description": meta["description"],
            "values": _table(effective, meta["name"]),
            "defaults": _table(defaults, meta["name"]),
            "activatable": not meta["always_live"],
            "enabled": enabled,
            "live_today": enabled,
        })
    return {
        "currency": getattr(settings, "currency", "ILS"),
        "units": "currency per second per rating point",
        "base": {"value": effective.base_price, "default": defaults.base_price,
                 "overridden": effective.base_price != defaults.base_price},
        "layers": layers,
        "activation": {
            "show": effective.enable_show,
            "position": effective.enable_position,
            "ad_type": effective.enable_ad_type,
        },
        "has_overrides": bool(overrides),
        "note": ("Rate card only. No operator edits yet." if not overrides
                 else "Operator edits applied. Every value traces to base x named layers."),
    }


class PricingUpdate(BaseModel):
    """A partial edit to the rate card, deep-merged onto the operator's saved overrides.

    ``overrides`` follows the YAML shape: ``base_price_per_second_per_tvr_point`` and the
    nested ``premiums`` / ``pricing_activation`` blocks. Only the keys present are changed;
    everything else keeps its current value. ``reset`` clears all operator overrides back
    to the shipped rate card.
    """

    overrides: dict[str, Any] = Field(default_factory=dict)
    reset: bool = False


class PriceSlotRequest(BaseModel):
    """Inputs for the price-any-slot tester. Only class and weekday are required.

    ``advertiser`` and ``campaign`` opt the slot into the per-layer override path:
    when an advertiser is given, the tester resolves and applies that advertiser's
    (and campaign's) targeted layer/final overrides on top of the rate card, so the
    operator sees exactly what a scoped rule does to the price.
    """

    pricing_class: str = "Other"
    weekday_iso: int = Field(default=1, ge=1, le=7)
    show: Optional[str] = None
    position: Optional[int] = Field(default=None, ge=1)
    break_size: Optional[int] = Field(default=None, ge=1)
    ad_type: Optional[str] = None
    advertiser_base: Optional[float] = Field(default=None, ge=0)
    advertiser: Optional[str] = None
    campaign: Optional[str] = None
    genre: Optional[str] = None
    daypart: Optional[str] = None


# These three helpers are imported lazily from server.py to avoid an import cycle at
# module load (server.py imports this router near the end of its own definition).
def _settings_io():
    from kairos_api.server import _load_settings, _save_settings
    return _load_settings, _save_settings


@router.get("/api/pricing")
def get_pricing() -> dict[str, Any]:
    """Return the full pricing hierarchy for the dashboard rate-card workspace."""
    load, _ = _settings_io()
    return _state_payload(load())


@router.put("/api/pricing")
def put_pricing(update: PricingUpdate) -> dict[str, Any]:
    """Apply an operator edit to the rate card and persist it.

    The edit is deep-merged onto the saved overrides, validated by constructing the
    PricingModel (a negative premium is rejected), then saved. The merged overrides flow
    into the next optimizer run, dashboard forecast and spot export. Returns the new state.
    """
    load, save = _settings_io()
    settings = load()
    current = dict(getattr(settings, "pricing_overrides", None) or {})
    merged: dict[str, Any] = {} if update.reset else _deep_merge(current, update.overrides)
    try:
        PricingModel.from_config(merged)  # validate before persisting
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid pricing edit: {exc}") from exc
    settings.pricing_overrides = merged
    save(settings)
    return _state_payload(settings)


def _advertiser_overrides(req: PriceSlotRequest, breakdown: PriceBreakdown) -> Any:
    """Resolve this advertiser/campaign's targeted overrides for the slot, or None.

    Returns the OverrideResolution when an advertiser is named and has at least one
    matching targeted rule, so the tester shows exactly what a scoped rule does to the
    price. With no advertiser (or no matching rule) the rate-card price stands unchanged.
    """
    if not req.advertiser:
        return None
    engine = AdvertiserRuleEngine.from_files()
    resolution = resolve_layer_overrides(
        engine, req.advertiser,
        position=req.position, genre=req.genre, daypart=req.daypart,
        programme=req.show, campaign=req.campaign, base_cpp=breakdown.base_cpp,
    )
    if not resolution.layer_overrides and not resolution.final_overrides:
        return None
    return resolution


@router.post("/api/pricing/price-slot")
def price_slot(req: PriceSlotRequest) -> dict[str, Any]:
    """Price one slot with a full per-layer breakdown (the price-any-slot tester).

    Uses the operator's saved rate card. Each layer names its multiplier and source, and
    the live total only multiplies the layers active today, so the tester never overstates
    the price. A struck-through "wired_off" line shows a configured-but-not-applied layer.

    When an advertiser (and optionally a campaign) is named, the tester resolves and
    applies that advertiser's targeted per-layer and final overrides on top of the rate
    card, so the operator sees exactly what a scoped rule does. Final-CPP guardrails run
    on the resulting price and any breach is surfaced as a named warning, never clamped.
    """
    load, _ = _settings_io()
    settings = load()
    overrides = getattr(settings, "pricing_overrides", None) or {}
    model = PricingModel.from_config(overrides)
    breakdown = model.price_slot(
        pricing_class=req.pricing_class,
        weekday_iso=req.weekday_iso,
        show=req.show,
        position=req.position,
        break_size=req.break_size,
        ad_type=req.ad_type,
        base_cpp=req.advertiser_base,
    )

    # Apply this advertiser's scoped overrides (per-layer REPLACE / final adjust), if any.
    resolution = _advertiser_overrides(req, breakdown)
    applied_overrides: list[dict[str, Any]] = []
    shadowed_overrides: list[dict[str, Any]] = []
    if resolution is not None:
        breakdown = apply_overrides(breakdown, resolution)
        applied_overrides = [
            {"rule_id": o.rule_id, "target_layer": o.target_layer, "multiplier": o.multiplier}
            for o in (*resolution.layer_overrides, *resolution.final_overrides)
        ]
        shadowed_overrides = [
            {"rule_id": s.rule_id, "target_layer": s.target_layer,
             "winner_rule_id": s.winner_rule_id, "reason": s.reason}
            for s in resolution.shadowed
        ]

    live_layers = [
        {"name": layer.name, "multiplier": layer.multiplier, "source": layer.source}
        for layer in breakdown.layers
    ]
    # Show the configured-but-off layers transparently (not multiplied into the total).
    wired_off: list[dict[str, Any]] = []
    if not model.enable_show and req.show and model.show_premium(req.show) != 1.0:
        wired_off.append({"name": "show", "multiplier": model.show_premium(req.show),
                          "source": "rate_card", "applied": False})
    if (not model.enable_position and req.position is not None
            and req.break_size is not None):
        mult = model.position_premium(req.position, req.break_size)
        if mult != 1.0:
            wired_off.append({"name": "position", "multiplier": mult,
                              "source": "rate_card", "applied": False})
    if not model.enable_ad_type and req.ad_type and model.ad_type_premium(req.ad_type) != 1.0:
        wired_off.append({"name": "ad_type", "multiplier": model.ad_type_premium(req.ad_type),
                          "source": "rate_card", "applied": False})

    # Final-CPP guardrails: surface a breach, never silently clamp the price.
    warnings = [
        {"code": w.code, "bound": w.bound, "message": w.message}
        for w in Guardrails.from_config(overrides).check(breakdown)
    ]
    return {
        "base_cpp": breakdown.base_cpp,
        "layers": live_layers,
        "wired_off_layers": wired_off,
        "applied_overrides": applied_overrides,
        "shadowed_overrides": shadowed_overrides,
        "guardrail_warnings": warnings,
        "total_premium": breakdown.total_premium,
        "final_cpp": breakdown.final_cpp,
        "currency": getattr(settings, "currency", "ILS"),
    }
