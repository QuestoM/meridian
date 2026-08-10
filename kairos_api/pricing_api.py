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

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.layer_overrides import apply_overrides, resolve_layer_overrides
from kairos.optimize.price_guardrails import Guardrails
from kairos.optimize.positions import label as position_label
from kairos.optimize.qh_billing import (
    QHSettlementConfigurationError,
    qh_settlement_enabled,
    validate_qh_settlement_provenance,
)
from kairos.optimize.pricing import (
    PriceBreakdown,
    PricingModel,
    _deep_merge,
    load_event_day_multipliers,
    load_price_events,
    pricing_from_settings,
)
from kairos_api.events_access import (
    EVENT_PRICING_COMPANY_ONLY_DETAIL,
    require_company_editor,
    requester_is_company,
)
from kairos_api.pricing_positions import position_vocabulary, preferred_payload

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
     "description": "Position-in-break premium (1 to 5 and L for last). Off until activated."},
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


def _layer_warnings(name: str, values: dict[str, Any], enabled: bool) -> list[dict[str, Any]]:
    """Honest hazards for a premium layer that is not yet live.

    A configured multiplier of 0 (or negative) is a trap: it reads as a harmless
    rate-card entry today, but the moment the layer is activated it zeroes the
    price of every slot in that category. The clearest real case is the ad-type
    ``promo`` multiplier, set to 0 because a channel promo carries no direct
    revenue, which would silently wipe promo-slot revenue if the ad-type layer
    were turned on. We surface this per layer as structured data (the dashboard
    renders it in the operator's language) so activation is an informed choice,
    never a silent revenue cut. Only flagged while the layer is off, since a live
    layer's effect is already in the numbers.
    """
    if enabled:
        return []
    zeroed = sorted(
        str(key) for key, value in values.items() if _is_number(value) and float(value) <= 0.0
    )
    if not zeroed:
        return []
    return [{"kind": "zeroes_on_activation", "categories": zeroed}]


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _qh_activation(model: PricingModel) -> dict[str, Any]:
    """Effective QH state, including proof from the currently ingested ratings."""
    state = {
        "qh_settlement": False,
        "qh_settlement_requested": model.enable_qh_settlement,
        "qh_audience_basis": model.qh_audience_basis or None,
        "qh_rating_vintage": model.qh_rating_vintage or None,
        "qh_rating_source": model.qh_rating_source or None,
        "qh_configuration_valid": (
            not model.enable_qh_settlement or qh_settlement_enabled(model)
        ),
        "qh_data_provenance_valid": None,
    }
    if not model.enable_qh_settlement:
        return state
    if not qh_settlement_enabled(model):
        state["qh_data_provenance_valid"] = False
        state["qh_blocked_reason"] = "The requested QH rating-currency configuration is incomplete."
        return state
    try:
        from kairos_api.preview_inputs import preview_inputs

        segments, _ = preview_inputs(None, None, None)
        validate_qh_settlement_provenance(model, segments)
    except Exception as exc:  # an unavailable source is a refusal, never activation
        state["qh_data_provenance_valid"] = False
        state["qh_blocked_reason"] = str(exc)[:300]
        return state
    state["qh_data_provenance_valid"] = True
    state["qh_settlement"] = True
    return state


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
        layer_values = _table(effective, meta["name"])
        entry = {
            "name": meta["name"], "kind": "premium",
            "description": meta["description"],
            "values": layer_values,
            "defaults": _table(defaults, meta["name"]),
            "activatable": not meta["always_live"],
            "enabled": enabled,
            "live_today": enabled,
            "warnings": _layer_warnings(meta["name"], layer_values, enabled),
        }
        if meta["name"] == "position":
            entry["vocabulary"] = position_vocabulary(layer_values)
        layers.append(entry)
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
            # The quarter-hour settlement restatement. It was missing here while
            # the assistant's prompt rule 17 instructs the model to "say whether
            # the settlement restatement flag is on or off", so the model was
            # told to state a fact this payload never carried. That is the same
            # shape as the compliance effective date found the same day: a
            # persona instructed to state something it is not given states it
            # anyway, and a confident wrong answer about a billing flag is worse
            # than a refusal.
            #
            # It restates measured revenue by +7.45 percent when activated and
            # ships OFF, so whether it is on is a question about money.
            **_qh_activation(effective),
        },
        # Event-date layer state: operator-asserted multipliers stored on calendar
        # events (data/calendar_events.csv), gated on pricing_activation.events.
        # active_event_count counts the ACTIVE stored events carrying a non-1.0
        # multiplier, whether or not the layer is switched on.
        "events": {
            "enabled": effective.enable_events,
            "active_event_count": len(load_price_events()),
            "basis": "operator assertion per calendar event, not measured",
        },
        "preferred_positions": preferred_payload(effective),
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
    # Broadcast date (YYYY-MM-DD) for the event-date layer. Optional; without it
    # the tester prices the slot date-blind, exactly as before.
    day: Optional[str] = None
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
def get_pricing(request: Request = None) -> dict[str, Any]:
    """Return the full pricing hierarchy for the dashboard rate-card workspace.

    The events activation is company-only on the write, and until now this read
    carried no gate at all, so the toggle rendered enabled to a channel account
    and failed after the click. It now carries ``can_edit_events`` and, when the
    answer is no, the same Hebrew reason the write would refuse with, so the
    refusal is legible before the click rather than a 403 after it.
    """
    load, _ = _settings_io()
    body = _state_payload(load())
    allowed = requester_is_company(request)
    body["can_edit_events"] = allowed
    if not allowed:
        body["can_edit_events_reason"] = EVENT_PRICING_COMPANY_ONLY_DETAIL
    return body


@router.put("/api/pricing")
def put_pricing(update: PricingUpdate, request: Request = None) -> dict[str, Any]:
    """Apply an operator edit to the rate card and persist it.

    The edit is deep-merged onto the saved overrides, validated by constructing the
    PricingModel (a negative premium is rejected), then saved. The merged overrides flow
    into the next optimizer run, dashboard forecast and spot export. Returns the new state.

    The event pricing activation switch (``pricing_activation.events``) is a
    company-only surface: a channel-affiliated session touching that key answers
    403; every other pricing edit stays open to any operator or admin session.
    """
    activation = update.overrides.get("pricing_activation")
    if isinstance(activation, dict) and "events" in activation:
        require_company_editor(request, detail=EVENT_PRICING_COMPANY_ONLY_DETAIL)
    load, save = _settings_io()
    settings = load()
    current = dict(getattr(settings, "pricing_overrides", None) or {})
    # A reset clears every override, including a live events activation, so it
    # is walled the same way whenever that activation is currently on.
    if update.reset and bool((current.get("pricing_activation") or {}).get("events")):
        require_company_editor(request, detail=EVENT_PRICING_COMPANY_ONLY_DETAIL)
    merged: dict[str, Any] = {} if update.reset else _deep_merge(current, update.overrides)
    try:
        candidate = PricingModel.from_config(merged)  # validate before persisting
        if candidate.enable_qh_settlement and not qh_settlement_enabled(candidate):
            raise ValueError(
                "qh_settlement requires qh_audience_basis='jewish_households', "
                "qh_rating_vintage='overnight_plus_1', and a non-empty qh_rating_source"
            )
        if candidate.enable_qh_settlement:
            from kairos_api.preview_inputs import preview_inputs

            try:
                segments, _ = preview_inputs(None, None, None)
            except Exception as exc:
                raise QHSettlementConfigurationError(
                    f"QH settlement rating provenance could not be validated: {exc}"
                ) from exc
            validate_qh_settlement_provenance(candidate, segments)
    except QHSettlementConfigurationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid pricing edit: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid pricing edit: {exc}") from exc
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, "settings")
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
    # pricing_from_settings is the same seam the optimizer, forecast and export
    # use; it also carries the event-date map when the events layer is active.
    model = pricing_from_settings(settings)
    breakdown = model.price_slot(
        pricing_class=req.pricing_class,
        weekday_iso=req.weekday_iso,
        show=req.show,
        position=req.position,
        break_size=req.break_size,
        ad_type=req.ad_type,
        day=req.day,
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
    # Which of the six positions the slot resolved to. A spot can hold two at
    # once (the last spot of a three-spot break is both 3 and L), so naming the
    # key that actually priced it stops an operator reading "5" where the tail
    # premium applied, or the reverse.
    position_key = None
    if req.position is not None:
        position_key = model.position_key(req.position, req.break_size)
    if not model.enable_ad_type and req.ad_type and model.ad_type_premium(req.ad_type) != 1.0:
        wired_off.append({"name": "ad_type", "multiplier": model.ad_type_premium(req.ad_type),
                          "source": "rate_card", "applied": False})
    if not model.enable_events and req.day:
        event_mult = load_event_day_multipliers().get(str(req.day)[:10], 1.0)
        if event_mult != 1.0:
            wired_off.append({"name": "event", "multiplier": event_mult,
                              "source": "operator_event", "applied": False})

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
        "position_key": position_key,
        "position_label_en": None if position_key is None else position_label(position_key, "en"),
        "position_label_he": None if position_key is None else position_label(position_key, "he"),
        "total_premium": breakdown.total_premium,
        "final_cpp": breakdown.final_cpp,
        "currency": getattr(settings, "currency", "ILS"),
    }


# The rate-card effect rides on this router: it prices an edit against the same
# saved overrides this module writes, so it belongs to the same mount and needs
# no registration of its own below the marker in server.py.
from kairos_api.pricing_api_effect import router as _effect_router  # noqa: E402

router.include_router(_effect_router)
