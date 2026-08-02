"""What a rate-card edit does to the money, before the edit is saved.

The rate card is the one surface in the product where a person changes a number
and every projected figure downstream moves. Until now it moved them silently:
the tester showed one slot, the save went through, and the plan's own worth of a
second changed the next time anybody looked.

This is the delta. It re-prices the saved weekly plan under the card as saved
and under the unsaved edit, on the operator's own channel, and reports the worth
of a second and the projected revenue under each. It is exact rather than
modelled, because break revenue is ``cpp * rating_points * units * premium``
(:func:`kairos.optimize.objective.break_revenue`) and is therefore strictly
linear in the effective rate, so re-pricing a committed row is one multiplication
by the ratio of its new rate to the rate it was priced at. The plan's own
``base_rate`` column is that rate, and the check that this is not an assumption
is that re-pricing under the card as saved reproduces the plan's own revenue and
its own yield per second to the last digit. The payload reports the row count it
reproduced, so a plan priced on a card nobody has since is visible rather than
quietly averaged in.

Two layers deliberately report no movement here, and the payload says so rather
than letting a reader assume the edit did nothing. Position and ad type price an
individual spot inside a break, not the break, so they cannot move a per-break
projection; their money shows up in the spot ledger.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# The premium layers that multiply into a break's own price. The rest price a
# spot inside the break and are named separately so the payload never implies a
# per-break projection covers them.
BREAK_LEVEL_LAYERS = ("base", "program", "day", "events")
SPOT_LEVEL_LAYERS = ("position", "ad_type", "show")


class PricingEffectRequest(BaseModel):
    """The same partial edit ``PUT /api/pricing`` takes, priced instead of saved."""

    overrides: dict[str, Any] = Field(default_factory=dict)
    reset: bool = False


def _rates(settings_map: dict[str, Any], pairs: tuple) -> dict[str, float]:
    """Effective break rate per segment id, through the engine's own seams."""
    from kairos_api.core import _plan_segment_index

    index = _plan_segment_index(pairs, settings_map)
    return {
        segment_id: float(segment.cpp) * float(segment.premium)
        for segment_id, segment in index.items()
    }


def _reprice(schedule: Any, rates: dict[str, float]) -> dict[str, Any]:
    """Revenue and ad seconds of the committed plan under one set of rates."""
    revenue = 0.0
    ad_seconds = 0.0
    repriced = 0
    unpriced = 0
    for row in schedule.itertuples(index=False):
        segment_id = str(getattr(row, "segment_id", "")).strip()
        try:
            planned = float(getattr(row, "predicted_revenue", 0.0) or 0.0)
            seconds = float(getattr(row, "total_break_time", 0.0) or 0.0)
            plan_rate = float(getattr(row, "base_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            unpriced += 1
            continue
        rate = rates.get(segment_id)
        if rate is None or plan_rate <= 0:
            unpriced += 1
            continue
        revenue += planned * (rate / plan_rate)
        ad_seconds += seconds
        repriced += 1
    return {
        "revenue": round(revenue, 2),
        "ad_seconds": round(ad_seconds, 1),
        "yield_per_second": round(revenue / ad_seconds, 4) if ad_seconds else None,
        "rows_repriced": repriced,
        "rows_unpriced": unpriced,
    }


def _changed_layers(saved: Any, draft: Any) -> list[dict[str, Any]]:
    """Which named layers the edit actually moves, and whether the plan sees them."""
    changed: list[dict[str, Any]] = []
    if saved.base_price != draft.base_price:
        changed.append({"layer": "base", "moves_plan": True,
                        "from": saved.base_price, "to": draft.base_price})
    tables = (
        ("program", "program_type_premiums", True),
        ("day", "day_of_week_premiums", True),
        ("show", "show_premiums", False),
        ("position", "position_premiums", False),
        ("ad_type", "ad_type_premiums", False),
    )
    for name, attribute, moves_plan in tables:
        before = dict(getattr(saved, attribute, {}) or {})
        after = dict(getattr(draft, attribute, {}) or {})
        keys = sorted({str(key) for key in (*before, *after)})
        moved = [
            key for key in keys
            if before.get(_coerce(key, before)) != after.get(_coerce(key, after))
        ]
        if moved:
            changed.append({"layer": name, "moves_plan": moves_plan, "keys": moved})
    for name, attribute in (("show", "enable_show"), ("position", "enable_position"),
                            ("ad_type", "enable_ad_type"), ("events", "enable_events")):
        if bool(getattr(saved, attribute, False)) != bool(getattr(draft, attribute, False)):
            changed.append({
                "layer": name,
                "moves_plan": name in BREAK_LEVEL_LAYERS,
                "activation": bool(getattr(draft, attribute, False)),
            })
    return changed


def _coerce(key: str, table: dict) -> Any:
    """Tables are keyed by int for weekday and position, by string elsewhere."""
    if key in table:
        return key
    try:
        numeric = int(key)
    except (TypeError, ValueError):
        return key
    return numeric if numeric in table else key


@router.post("/api/pricing/effect", tags=["pricing"])
def pricing_effect(request_body: PricingEffectRequest) -> dict[str, Any]:
    """The worth of a second and the projected revenue, saved card against edit.

    Nothing is written. The scope is the operator's own channel over the saved
    plan's own dates, and both figures carry it, because two money figures on one
    screen without their scope is how a contradiction gets read as a fact.
    """
    from kairos.optimize.pricing import PricingModel, _deep_merge, pricing_from_settings
    from kairos_api.core import _load_break_schedule, _load_settings, _model_dump

    settings = _load_settings()
    current = dict(getattr(settings, "pricing_overrides", None) or {})
    merged: dict[str, Any] = {} if request_body.reset else _deep_merge(current, request_body.overrides)
    try:
        PricingModel.from_config(merged)
    except ValueError as exc:
        return {"available": False, "reason": f"The edit is not a usable rate card: {exc}"}

    schedule = _load_break_schedule()
    channel = str(getattr(settings, "operator_channel", "") or "").strip()
    if schedule.empty or "segment_id" not in schedule.columns:
        return {"available": False, "reason": "No saved weekly plan with segment ids is on disk to price."}
    channels_in = (
        sorted({str(value).strip() for value in schedule["channel"].astype(str)})
        if "channel" in schedule.columns else []
    )
    if channel and "channel" in schedule.columns:
        schedule = schedule[schedule["channel"].astype(str).str.strip() == channel]
    if schedule.empty:
        return {"available": False, "reason": "The saved plan carries no rows for the declared operator channel."}

    pairs = tuple(
        (str(row_channel), str(day))
        for row_channel, day in schedule[["channel", "date"]].astype(str).drop_duplicates().itertuples(index=False, name=None)
    )
    saved_map = _model_dump(settings)
    draft_map = dict(saved_map)
    draft_map["pricing_overrides"] = merged
    try:
        saved_rates = _rates(saved_map, pairs)
        draft_rates = _rates(draft_map, pairs)
    except Exception:
        logger.exception("plan segment rebuild failed for the rate-card effect")
        return {"available": False, "reason": "Plan segment rebuild failed; see the server log."}

    saved_side = _reprice(schedule, saved_rates)
    draft_side = _reprice(schedule, draft_rates)
    plan_revenue = round(float(schedule["predicted_revenue"].fillna(0).sum()), 2)
    saved_model = pricing_from_settings(saved_map)
    draft_model = pricing_from_settings(draft_map)
    delta_revenue = round(draft_side["revenue"] - saved_side["revenue"], 2)
    return {
        "available": True,
        "currency": getattr(settings, "currency", "ILS"),
        "scope": {
            "channel": channel,
            # With no declared channel there is nothing to scope to, so this
            # prices every channel the plan carries. That is not the operator's
            # money and the payload has to say so rather than let a surface read
            # a market total as one broadcaster's. The channels are counted and
            # never named: an unnamed aggregate is the only shape a competitor
            # may reach an operator payload in.
            "scoped": bool(channel),
            "channels_priced": 1 if channel else len(channels_in),
            "unscoped_reason": "" if channel else "No operator channel is declared, so this prices every channel in the loaded plan.",
            "date_from": str(schedule["date"].min()),
            "date_to": str(schedule["date"].max()),
            "rows": int(len(schedule)),
            "days": len(pairs),
        },
        "saved": saved_side,
        "draft": draft_side,
        "delta": {
            "revenue": delta_revenue,
            "yield_per_second": _delta(draft_side["yield_per_second"], saved_side["yield_per_second"]),
            "percent": round(100.0 * delta_revenue / saved_side["revenue"], 4) if saved_side["revenue"] else None,
        },
        "plan_revenue_on_record": plan_revenue,
        "reproduces_plan": saved_side["revenue"] == plan_revenue,
        "changed_layers": _changed_layers(saved_model, draft_model),
        "spot_level_layers": list(SPOT_LEVEL_LAYERS),
        "basis": {
            "formula": "revenue = cpp * rating_points * (duration / unit) * premium, so a committed row re-prices by new_rate / plan_rate",
            "source": "the saved weekly plan, re-priced through kairos_api.core._plan_segment_index",
            "note": "Position and ad type price a spot inside a break, so they cannot move a per-break projection.",
        },
    }


def _delta(after: Optional[float], before: Optional[float]) -> Optional[float]:
    if after is None or before is None:
        return None
    return round(after - before, 4)
