"""Additive read-only scenario, yield, gold and make-good endpoints.

These endpoints feed a new operator workspace (yield, scenario A/B, gold breaks,
make-good alerts). Every number is traced to a real source: the saved weekly
schedule, a live optimizer run, the gold-break guardrail, or the pacing
make-good helper. Where the underlying data does not exist yet the payload says
so honestly (``available``/``data_available`` false with a reason) and never
fabricates figures. None of these routes mutate state.

The module keeps server.py lean: it imports the existing loaders and helpers
from server.py rather than re-deriving them, so the data contracts stay aligned.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["insights"])


class ScenarioCompareRequest(BaseModel):
    """A what-if A/B: two revenue weights under shared (optional) guardrails.

    ``weight_a``/``weight_b`` are the 0..100 revenue-vs-retention levers. The three
    guardrails are optional; when omitted they fall back to the operator's saved
    settings so the comparison reflects the real plan baseline, not an arbitrary
    default. Both legs run the genuine optimizer; nothing here is synthesized.
    """

    weight_a: int = Field(ge=0, le=100)
    weight_b: int = Field(ge=0, le=100)
    retention_floor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_breaks_per_hour: Optional[int] = Field(default=None, ge=1, le=20)
    risk_lambda: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _daypart_for_start(start_time: object) -> Optional[str]:
    """Map a 'HH:MM' break start to a daypart key, honestly None when unparseable."""
    try:
        from kairos.data.dayparts import daypart_for_hour
    except Exception:  # pragma: no cover - taxonomy optional
        return None
    text = str(start_time or "").strip()
    if not text or ":" not in text:
        return None
    head = text.split(" ")[0].split("T")[-1]
    hour_part = head.split(":")[0]
    if not hour_part.isdigit():
        return None
    return daypart_for_hour(int(hour_part))


# One honest sentence shipped with the retention-cost band, naming its source.
_RETENTION_BAND_BASIS = (
    "The retention-cost band re-prices the committed plan's per-break audience loss at "
    "each segment's calibrated 95 percent coefficient interval bounds, the same interval "
    "seam risk_lambda uses for worst-plausible pricing: ci_low (the more damaging bound) "
    "yields retention_cost_high and ci_high yields retention_cost_low."
)


def _optimistic_impact(point: float, ci_low: float, ci_high: float) -> float:
    """The least-damaging plausible per-break coefficient in the credible band.

    Exact mirror of :func:`kairos.optimize.objective.conservative_impact` at
    ``risk_lambda=1``: where that returns the most damaging bound for
    worst-plausible pricing, this returns the least damaging one for the low
    edge of the retention-cost band. Same guards, mirrored: a non-finite value
    degrades to the point, bounds are clamped non-positive, and the result is
    never MORE damaging than the point, so the band always brackets the point
    estimate. No new statistics; only the opposite end of the same interval.
    """
    if ci_low != ci_low or ci_high != ci_high or point != point:  # NaN guard
        return min(0.0, point)
    best = max(min(0.0, ci_low), min(0.0, ci_high), point)
    return min(0.0, best)


@lru_cache(maxsize=8)
def _plan_cost_band_cached(signature: tuple, scope_channel: Optional[str] = None) -> dict[str, Any]:
    """Retention-cost band for the committed weekly plan, in ILS.

    Rebuilds the plan's own ProgramSegment objects
    (:func:`kairos_api.core._plan_segment_index`, the same seams that priced the
    plan), joins them to the saved CSV by ``segment_id``, and re-prices each
    committed row's per-break retention cost through
    :func:`kairos.optimize.revenue_net.segment_retention_cost_ils` three times:
    at the measured point coefficient and at the two calibrated interval bounds,
    each swapped onto the segment exactly the way ``risk_lambda`` pricing swaps
    in the conservative coefficient
    (:func:`kairos.optimize.objective.conservative_impact` at ``risk_lambda=1``
    for the damaging bound; its mirror for the optimistic one). A segment
    without a measured interval contributes its point cost to both bounds, the
    same certainty treatment the optimizer's risk path applies. Honest
    ``available: False`` with the reason when the saved plan no longer joins the
    EPG rebuild; nothing is proxied.
    """
    del signature  # cache key only
    from dataclasses import replace

    from kairos.optimize.objective import conservative_impact
    from kairos.optimize.revenue_net import segment_retention_cost_ils
    from kairos_api.core import _plan_segment_index

    server = _server()
    schedule = server._load_break_schedule()
    if schedule.empty or "segment_id" not in schedule.columns:
        return {"available": False, "reason": "No saved weekly schedule with segment ids on disk."}
    # The band must live on the same scope as the point estimate it decorates:
    # a whole-network band around an owned-channel point can never bracket it.
    if scope_channel and "channel" in schedule.columns:
        schedule = schedule[schedule["channel"].astype(str).str.strip() == scope_channel]
        if schedule.empty:
            return {"available": False, "reason": "The saved plan carries no rows for the configured operator channel."}
    try:
        pairs = tuple(
            (str(channel), str(day))
            for channel, day in schedule[["channel", "date"]]
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        index = _plan_segment_index(pairs, server._model_dump(server._load_settings()))
    except Exception:
        logger.exception("plan segment rebuild failed for the retention-cost band")
        return {"available": False, "reason": "Plan segment rebuild failed; see the server log."}
    if not index:
        return {"available": False, "reason": "The optimization engine is unavailable."}

    joined = 0
    cost_point = cost_low = cost_high = 0.0
    for row in schedule.itertuples(index=False):
        segment = index.get(str(getattr(row, "segment_id", "")).strip())
        if segment is None:
            continue
        joined += 1
        try:
            num_breaks = int(float(getattr(row, "num_breaks", 0) or 0))
        except (TypeError, ValueError):
            num_breaks = 0
        if num_breaks <= 0:
            continue
        cost_point += segment_retention_cost_ils(segment, num_breaks)
        if segment.impact_ci_low is None or segment.impact_ci_high is None:
            worst = best = segment.impact_coefficient
        else:
            worst = conservative_impact(
                segment.impact_coefficient,
                segment.impact_ci_low,
                segment.impact_ci_high,
                risk_lambda=1.0,
            )
            best = _optimistic_impact(
                segment.impact_coefficient, segment.impact_ci_low, segment.impact_ci_high
            )
        cost_high += segment_retention_cost_ils(replace(segment, impact_coefficient=worst), num_breaks)
        cost_low += segment_retention_cost_ils(replace(segment, impact_coefficient=best), num_breaks)

    if joined < len(schedule) * 0.99:
        # The EPG no longer reproduces the saved plan (a re-ingest without a
        # recompute); a partial band would be dishonest.
        return {
            "available": False,
            "reason": "Saved plan no longer joins the EPG rebuild; recompute the schedule.",
        }
    return {"available": True, "low": cost_low, "high": cost_high, "point": cost_point}


def _plan_cost_band(scope_channel: Optional[str] = None) -> dict[str, Any]:
    """The cached committed-plan retention-cost band, keyed on its real inputs
    and the channel scope it is computed for."""
    from kairos_api.core import (
        DATA_DIR,
        MODELS_DIR,
        OUTPUT_DIR,
        SETTINGS_PATH,
        _signature,
    )

    return dict(_plan_cost_band_cached(_signature([
        OUTPUT_DIR / "weekly_break_schedule.csv",
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        SETTINGS_PATH,
        MODELS_DIR / "tv_break_coefficients.json",
        MODELS_DIR / "tv_break_posterior.pkl",
    ]), scope_channel or None))


# ---------------------------------------------------------------------------
# 3. Yield per ad-second, from the real saved weekly schedule.
# ---------------------------------------------------------------------------
def _build_yield_per_second(schedule: pd.DataFrame, scope_channel: Optional[str] = None) -> dict[str, Any]:
    """Revenue per ad-second by daypart and by programme, from the saved schedule.

    ``predicted_revenue`` and ``total_break_time`` (ad-seconds) are the optimizer's
    own saved outputs. Yield is ``revenue / ad_seconds`` for groups that actually
    carry ad time. Revenue-net-of-retention (revenue minus the ad revenue foregone
    as breaks shed audience, in ILS) is computed by
    :func:`kairos.optimize.revenue_net.frame_revenue_net` and surfaced ONLY when it
    is honestly exact: that needs the per-segment ``baseline_tvr`` on the row, since
    the saved ``retention_used`` is the final share, not the per-break audience, and
    recovering the audience from it alone overstates the cost materially. When
    ``baseline_tvr`` is absent (the current CSV schema) the payload stays
    ``revenue_net_available: false`` with the exact missing input and a basis
    disclosure, never a fabricated or biased figure.
    """
    if schedule.empty:
        return {"available": False, "reason": "No saved weekly schedule on disk.", "by_daypart": [], "by_programme": []}

    # Imported here (not at module load) so this module keeps its engine imports lazy.
    from kairos.optimize.revenue_net import frame_revenue_net

    frame = schedule.copy()
    revenue = pd.to_numeric(
        frame.get("predicted_revenue", frame.get("revenue_ils", 0)), errors="coerce"
    ).fillna(0.0)
    ad_seconds = pd.to_numeric(
        frame.get("total_break_time", frame.get("break_length", 0)), errors="coerce"
    ).fillna(0.0)
    # retention_used and segment_id are persisted on the weekly CSV now, so the
    # per-group retention level and true segment count are faithful reads, not
    # engine re-derivations. Both are guarded so an older CSV that lacks them still
    # returns the revenue/yield figures rather than erroring.
    retention = (
        pd.to_numeric(frame["retention_used"], errors="coerce")
        if "retention_used" in frame.columns
        else pd.Series(float("nan"), index=frame.index, dtype="float64")
    )
    frame = frame.assign(_revenue=revenue, _ad_seconds=ad_seconds, _retention=retention)
    frame = frame[frame["_ad_seconds"] > 0]
    if frame.empty:
        return {"available": False, "reason": "Saved schedule has no ad-seconds to monetize.", "by_daypart": [], "by_programme": []}

    frame["_daypart"] = frame.get("start_time").map(_daypart_for_start)
    has_segment_id = "segment_id" in frame.columns

    def _weighted_retention(part: pd.DataFrame) -> Optional[float]:
        """Ad-seconds-weighted mean of the persisted retention_used, or None when
        no row in the group carries a measured value (never fabricated to 0)."""
        valid = part[part["_retention"].notna() & (part["_ad_seconds"] > 0)]
        weight = float(valid["_ad_seconds"].sum())
        if valid.empty or weight <= 0:
            return None
        return round(float((valid["_retention"] * valid["_ad_seconds"]).sum() / weight), 4)

    def _aggregate(group_key: str, label_unknown: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, part in frame.groupby(frame[group_key].fillna(label_unknown)):
            seconds = float(part["_ad_seconds"].sum())
            rev = float(part["_revenue"].sum())
            if seconds <= 0:
                continue
            rows.append(
                {
                    "group": str(key),
                    "revenue": round(rev, 2),
                    "ad_seconds": int(round(seconds)),
                    "yield_per_second": round(rev / seconds, 4),
                    "break_count": int(pd.to_numeric(part.get("num_breaks", 1), errors="coerce").fillna(1).sum()),
                    "segment_count": int(part["segment_id"].nunique()) if has_segment_id else None,
                    "avg_retention": _weighted_retention(part),
                }
            )
        return sorted(rows, key=lambda row: row["yield_per_second"], reverse=True)

    by_daypart = _aggregate("_daypart", "unclassified")
    by_programme = _aggregate("program_type", "Other")
    total_seconds = float(frame["_ad_seconds"].sum())
    total_revenue = float(frame["_revenue"].sum())

    # Revenue net of retention damage (ILS), from the saved schedule. Available
    # only when the row carries baseline_tvr, so the cost is exact (gross potential
    # minus delivered revenue); otherwise honest-false with the exact missing input
    # and a basis disclosure. Computed on the whole saved frame, independent of the
    # yield ad-seconds filter above.
    money = frame_revenue_net(schedule)
    net_block: dict[str, Any] = {
        "revenue_net_available": bool(money.get("available")),
        "basis": money.get("basis"),
    }
    if money.get("available"):
        net_block["revenue_net_ils"] = money.get("revenue_net_ils")
        net_block["retention_cost_ils"] = money.get("retention_cost_ils")
        net_block["revenue_ils"] = money.get("revenue_ils")
        # Additive uncertainty band on the retention cost: the committed plan
        # re-priced at the calibrated coefficient-interval bounds through the
        # engine's own per-break cost primitive (see _plan_cost_band_cached).
        # Attached only when the band's own point re-pricing brackets THIS
        # frame's point estimate, so a stale rebuild or a synthetic frame can
        # never ship a band that does not contain the number beside it.
        try:
            band = _plan_cost_band(scope_channel)
        except Exception:
            logger.exception("retention-cost band computation failed")
            band = {"available": False, "reason": "Band computation failed; see the server log."}
        cost_point = money.get("retention_cost_ils")
        if (
            band.get("available")
            and cost_point is not None
            and float(band["low"]) - 0.01 <= float(cost_point) <= float(band["high"]) + 0.01
        ):
            net_block["retention_cost_low"] = round(float(band["low"]), 2)
            net_block["retention_cost_high"] = round(float(band["high"]), 2)
            net_block["retention_cost_basis"] = _RETENTION_BAND_BASIS
        else:
            net_block["retention_cost_low"] = None
            net_block["retention_cost_high"] = None
            net_block["retention_cost_basis"] = (
                "Retention-cost band unavailable: "
                + str(
                    band.get("reason")
                    or "the interval re-pricing does not bracket this schedule's point estimate."
                )
            )
    else:
        net_block["revenue_net_reason"] = money.get("reason")

    return {
        "available": True,
        **net_block,
        "currency": _server()._load_settings().currency,
        "totals": {
            "revenue": round(total_revenue, 2),
            "ad_seconds": int(round(total_seconds)),
            "yield_per_second": round(total_revenue / total_seconds, 4) if total_seconds > 0 else 0.0,
            "segment_count": int(frame["segment_id"].nunique()) if has_segment_id else None,
            "avg_retention": _weighted_retention(frame),
        },
        "by_daypart": by_daypart,
        "by_programme": by_programme,
    }


def scoped_yield_payload() -> dict[str, Any]:
    """The operator-channel-scoped yield payload the dashboard route returns.

    Extracted from the route body unchanged so every consumer (the route and the
    assistant's yield_totals context section) quotes the SAME scope: yield and
    the money story must never quote modeled competitor inventory as ours. The
    whole frame is used only when no channel is configured, and the
    ``scope_channel``/``n_channels_total`` fields disclose the scope either way.
    """
    server = _server()
    schedule = server._load_break_schedule()
    settings = server._load_settings()
    owned = str(getattr(settings, "operator_channel", "") or "").strip()
    scope_channel: Optional[str] = None
    n_channels_total: Optional[int] = None
    scoped = schedule
    if "channel" in schedule.columns:
        n_channels_total = int(schedule["channel"].astype(str).str.strip().nunique())
        if owned:
            scoped = schedule[schedule["channel"].astype(str).str.strip() == owned]
            scope_channel = owned
    if scope_channel and scoped.empty and not schedule.empty:
        payload: dict[str, Any] = {
            "available": False,
            "reason": "the saved plan carries no rows for the configured operator channel",
            "by_daypart": [],
            "by_programme": [],
        }
    else:
        payload = _build_yield_per_second(scoped, scope_channel=scope_channel)
    payload["scope_channel"] = scope_channel
    payload["n_channels_total"] = n_channels_total
    return payload


@router.get("/api/yield-per-second")
def yield_per_second() -> dict[str, Any]:
    return scoped_yield_payload()


# ---------------------------------------------------------------------------
# 5. Gold breaks, read straight from the saved weekly schedule CSV.
# ---------------------------------------------------------------------------
def _is_gold_truthy(value: Any) -> bool:
    """Whether a CSV ``is_gold`` cell marks the segment gold.

    Robust to how the round-tripped column typed itself: a native pandas bool, or
    the literal 'True'/'False' text a re-read CSV can carry. Anything else (blank,
    NaN, 'False') is honestly not gold.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "1.0"}


def _cell_or_none(value: Any) -> Optional[str]:
    """A schedule cell as a clean string, or None for blank/NaN, never 'nan'."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _build_gold_breaks() -> dict[str, Any]:
    """Gold breaks in the operator's current plan, read from the saved schedule CSV.

    Gold status is now materialized on ``output/weekly_break_schedule.csv`` as the
    per-segment ``is_gold`` column. The export sets it from each plan's placements
    (``is_gold = any(placement.is_gold)`` for the segment), so reading the column
    here is the faithful, cheap source: this endpoint no longer re-runs the whole
    optimizer just to recover gold status. Each gold row is one gold segment,
    carrying its own ``predicted_revenue`` and first-break ``start_time`` from the
    plan. ``realized_premium``/``potential_premium`` are advertiser-attribution
    figures that live only on the daily spot-pricing path, so they stay null with a
    ``source_pending`` marker, never invented.
    """
    server = _server()
    settings = server._load_settings()
    if not settings.sponsorships_enabled:
        return {"available": True, "enabled": False, "reason": "Sponsorships are disabled in settings.", "count": 0, "breaks": [], "by_day": []}
    if not settings.gold_breaks_enabled:
        return {"available": True, "enabled": False, "reason": "Gold breaks are disabled in settings.", "count": 0, "breaks": [], "by_day": []}

    schedule = server._load_break_schedule()
    if schedule.empty:
        return {"available": False, "reason": "No saved weekly schedule on disk.", "count": 0, "breaks": [], "by_day": []}
    if "is_gold" not in schedule.columns:
        return {
            "available": True,
            "enabled": True,
            "count": 0,
            "reason": "Saved weekly schedule predates gold-break tracking; recompute the schedule to populate is_gold.",
            "max_per_day": settings.gold_breaks_max_per_day,
            "breaks": [],
            "by_day": [],
        }

    # Competitor boundary: gold is quoted only for the operator's own channel,
    # like every other money surface. A competitor row with is_gold set must
    # never surface its channel name or figures here.
    owned = str(settings.operator_channel or "").strip()
    if owned and "channel" in schedule.columns:
        schedule = schedule[schedule["channel"].astype(str).str.strip() == owned]
    gold = schedule[schedule["is_gold"].map(_is_gold_truthy)]
    if gold.empty:
        return {
            "available": True,
            "enabled": True,
            "count": 0,
            "reason": "No gold breaks in the current plan (none configured as gold in overrides).",
            "max_per_day": settings.gold_breaks_max_per_day,
            "breaks": [],
            "by_day": [],
        }

    revenue = pd.to_numeric(gold.get("predicted_revenue", 0), errors="coerce").fillna(0.0)
    duration = pd.to_numeric(gold.get("break_length", 0), errors="coerce").fillna(0.0)
    breaks: list[dict[str, Any]] = []
    by_day_counts: dict[str, int] = {}
    for (_, row), rev, dur in zip(gold.iterrows(), revenue, duration):
        day = _cell_or_none(row.get("date")) or ""
        by_day_counts[day] = by_day_counts.get(day, 0) + 1
        breaks.append(
            {
                "segment_id": _cell_or_none(row.get("segment_id")),
                "channel": _cell_or_none(row.get("channel")),
                "day": day,
                "start_time": _cell_or_none(row.get("start_time")),
                "program_type": _cell_or_none(row.get("program_type")),
                "duration_seconds": round(float(dur), 1),
                "revenue": round(float(rev), 2),
                "realized_premium": None,
                "potential_premium": None,
                "premium_source": "source_pending",
                "premium_note": "Gold-break premium is realized on the daily spot-pricing path, not the weekly optimizer.",
            }
        )
    return {
        "available": True,
        "enabled": True,
        "count": len(breaks),
        "max_per_day": settings.gold_breaks_max_per_day,
        "breaks": breaks,
        "by_day": [{"day": day, "count": count} for day, count in sorted(by_day_counts.items())],
    }


@router.get("/api/gold-breaks")
def gold_breaks() -> dict[str, Any]:
    return _build_gold_breaks()


# ---------------------------------------------------------------------------
# 4. Scenario compare: two real optimizer runs, side by side.
# ---------------------------------------------------------------------------
def _scenario_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull the comparable fields from a run_scenario payload.

    ``objective`` is the optimizer's convex-blend score (a weighted blend of
    revenue and retention, NOT a literal revenue-minus-cost subtraction), so it is
    reported under its own name and never relabeled as revenue_net.
    """
    summary = payload.get("summary", {})
    return {
        "revenue_weight": payload.get("controls", {}).get("revenue_weight"),
        "projected_revenue": summary.get("projected_revenue"),
        "average_retention": summary.get("average_retention"),
        "total_breaks": summary.get("total_breaks"),
        "total_ad_seconds": summary.get("total_ad_seconds"),
        "objective": summary.get("objective"),
        "compliant": summary.get("compliant"),
        "channel": payload.get("channel"),
        "day": payload.get("day"),
    }


def _delta(a: dict[str, Any], b: dict[str, Any], key: str) -> Optional[float]:
    """b - a for a numeric summary field, or None when either side is missing."""
    av, bv = a.get(key), b.get(key)
    if av is None or bv is None:
        return None
    return round(float(bv) - float(av), 4)


def _build_scenario_compare(request: ScenarioCompareRequest) -> dict[str, Any]:
    server = _server()
    if not server._ENGINE_AVAILABLE:
        return {"available": False, "reason": "Optimization engine unavailable."}

    saved = server._load_settings()
    floor = request.retention_floor if request.retention_floor is not None else saved.min_retention_floor
    max_bph = request.max_breaks_per_hour if request.max_breaks_per_hour is not None else saved.max_breaks_per_hour
    risk = request.risk_lambda if request.risk_lambda is not None else saved.risk_lambda

    from kairos.service import run_scenario

    # The full saved settings and pacing reference date, threaded into both legs
    # exactly as /api/optimizer-plan and the frontier do (server._pacing_call_kwargs),
    # so the A/B baseline honours every operator guardrail, the pricing overrides and
    # the operator channel scope instead of silently falling back to engine defaults.
    # The scenario overrides (floor/max_bph/risk) still apply on top of this base,
    # so the scenario-control semantics are unchanged.
    reference_today = server._reference_today(saved)
    settings_map = server._model_dump(saved)
    # Scope both A/B legs to the operator's owned channel-day (the shared selector
    # the scenario slider and frontier use), so the comparison is the owned
    # channel's, never the source's first channel-day (a competitor).
    from kairos_api.dashboard_api import _owned_scope

    channel, day = _owned_scope(saved)

    def _run(weight: int) -> dict[str, Any]:
        return run_scenario(
            revenue_weight=weight,
            retention_floor=floor,
            max_breaks_per_hour=max_bph,
            risk_lambda=risk,
            today=reference_today,
            settings=settings_map,
            channel=channel,
            day=day,
        )

    try:
        payload_a = _run(request.weight_a)
        payload_b = _run(request.weight_b)
    except Exception as exc:  # pragma: no cover - data/environment dependent
        return {"available": False, "reason": f"Optimizer run failed: {str(exc)[:200]}"}

    a = _scenario_summary(payload_a)
    b = _scenario_summary(payload_b)
    return {
        "available": True,
        "guardrails": {"retention_floor": floor, "max_breaks_per_hour": max_bph, "risk_lambda": risk},
        "a": a,
        "b": b,
        "delta": {
            "revenue": _delta(a, b, "projected_revenue"),
            "retention": _delta(a, b, "average_retention"),
            "breaks": _delta(a, b, "total_breaks"),
            "ad_seconds": _delta(a, b, "total_ad_seconds"),
            "objective": _delta(a, b, "objective"),
            "revenue_net": None,
        },
        "revenue_net_note": (
            "A literal revenue-net-of-retention figure is not a summary field of run_scenario; "
            "the optimizer exposes a convex-blend objective instead, reported under 'objective'."
        ),
    }


@router.post("/api/scenario-compare")
def scenario_compare(request: ScenarioCompareRequest) -> dict[str, Any]:
    return _build_scenario_compare(request)


# ---------------------------------------------------------------------------
# 6. Make-good alerts, from the pacing make-good projection helper.
# ---------------------------------------------------------------------------
@router.get("/api/make-good-alerts")
def make_good_alerts() -> dict[str, Any]:
    """At-risk campaigns from :func:`kairos.optimize.pacing.project_make_goods`.

    Data-pending: ``campaign_flights.csv`` is header-only until the owner uploads
    real flights, so ``load_campaigns`` returns ``[]`` and this returns an empty
    alert list with ``data_available: false``. It never fabricates an alert.
    """
    try:
        from datetime import date

        from kairos.optimize.pacing import load_campaigns, project_make_goods
    except Exception as exc:  # pragma: no cover - module optional
        return {"alerts": [], "data_available": False, "reason": f"Pacing module unavailable: {str(exc)[:200]}"}

    settings = _server()._load_settings()
    today = _reference_today(settings)
    campaigns = load_campaigns()
    if not campaigns:
        return {
            "alerts": [],
            "data_available": False,
            "reason": "campaign_flights.csv has no campaign rows yet (header-only seed).",
            "as_of": today.isoformat(),
        }

    at_risk = project_make_goods(campaigns, today)
    alerts = [
        {
            "campaign_id": c.campaign_id,
            "elapsed_frac": round(c.elapsed_frac, 4),
            "delivered_frac": round(c.delivered_frac, 4),
            "projected_frac": round(c.projected_frac, 4),
            "projected_shortfall": round(c.projected_shortfall, 4),
        }
        for c in at_risk
    ]
    return {"alerts": alerts, "data_available": True, "count": len(alerts), "as_of": today.isoformat()}


def _reference_today(settings: Any) -> Any:
    """The reference date the pacing projection runs against (settings.effective_date)."""
    from datetime import date

    text = str(getattr(settings, "effective_date", "") or "").strip()
    parts = text.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            pass
    return date.today()
