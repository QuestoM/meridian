"""Phase B read-only/additive endpoints for the Kairos UX wave.

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

from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["phase-b"])


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


# ---------------------------------------------------------------------------
# 3. Yield per ad-second, from the real saved weekly schedule.
# ---------------------------------------------------------------------------
def _build_yield_per_second(schedule: pd.DataFrame) -> dict[str, Any]:
    """Revenue per ad-second by daypart and by programme, from the saved schedule.

    ``predicted_revenue`` and ``total_break_time`` (ad-seconds) are the optimizer's
    own saved outputs. Yield is ``revenue / ad_seconds`` for groups that actually
    carry ad time. Revenue-net-of-retention is intentionally omitted at the row
    level because the saved weekly CSV does not persist a per-row retention-cost
    column; it is surfaced as ``revenue_net_available: false`` so the UI does not
    imply a number that is not on disk.
    """
    if schedule.empty:
        return {"available": False, "reason": "No saved weekly schedule on disk.", "by_daypart": [], "by_programme": []}

    frame = schedule.copy()
    revenue = pd.to_numeric(
        frame.get("predicted_revenue", frame.get("revenue_ils", 0)), errors="coerce"
    ).fillna(0.0)
    ad_seconds = pd.to_numeric(
        frame.get("total_break_time", frame.get("break_length", 0)), errors="coerce"
    ).fillna(0.0)
    frame = frame.assign(_revenue=revenue, _ad_seconds=ad_seconds)
    frame = frame[frame["_ad_seconds"] > 0]
    if frame.empty:
        return {"available": False, "reason": "Saved schedule has no ad-seconds to monetize.", "by_daypart": [], "by_programme": []}

    frame["_daypart"] = frame.get("start_time").map(_daypart_for_start)

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
                }
            )
        return sorted(rows, key=lambda row: row["yield_per_second"], reverse=True)

    by_daypart = _aggregate("_daypart", "unclassified")
    by_programme = _aggregate("program_type", "Other")
    total_seconds = float(frame["_ad_seconds"].sum())
    total_revenue = float(frame["_revenue"].sum())
    return {
        "available": True,
        "revenue_net_available": False,
        "revenue_net_reason": "Saved weekly schedule does not persist a per-row retention-cost column.",
        "currency": _server()._load_settings().currency,
        "totals": {
            "revenue": round(total_revenue, 2),
            "ad_seconds": int(round(total_seconds)),
            "yield_per_second": round(total_revenue / total_seconds, 4) if total_seconds > 0 else 0.0,
        },
        "by_daypart": by_daypart,
        "by_programme": by_programme,
    }


@router.get("/api/yield-per-second")
def yield_per_second() -> dict[str, Any]:
    server = _server()
    return _build_yield_per_second(server._load_break_schedule())


# ---------------------------------------------------------------------------
# 5. Gold breaks, from a real optimizer run on the saved settings.
# ---------------------------------------------------------------------------
def _build_gold_breaks() -> dict[str, Any]:
    """Gold breaks in the operator's current plan, from a live optimizer run.

    Gold status is an override/guardrail concept (manual_overrides.csv ``gold``
    column -> optimizer placement ``is_gold``); the saved weekly CSV does not carry
    it, so the honest source is one real :func:`run_scenario` run on the saved
    settings. ``realized_premium``/``potential_premium`` are advertiser-attribution
    figures that exist only on the daily spot-pricing path, so they are returned as
    null with a ``source_pending`` marker, never invented.
    """
    server = _server()
    if not server._ENGINE_AVAILABLE:
        return {"available": False, "reason": "Optimization engine unavailable.", "count": 0, "breaks": [], "by_day": []}

    settings = server._load_settings()
    if not settings.sponsorships_enabled:
        return {"available": True, "enabled": False, "reason": "Sponsorships are disabled in settings.", "count": 0, "breaks": [], "by_day": []}
    if not settings.gold_breaks_enabled:
        return {"available": True, "enabled": False, "reason": "Gold breaks are disabled in settings.", "count": 0, "breaks": [], "by_day": []}

    try:
        from kairos.service import run_scenario

        payload = run_scenario(
            revenue_weight=settings.revenue_weight,
            retention_floor=settings.min_retention_floor,
            max_breaks_per_hour=settings.max_breaks_per_hour,
            risk_lambda=settings.risk_lambda,
            # Thread the operator's full saved settings and pacing reference date,
            # exactly as /api/optimizer-plan and the frontier do (server._pacing_call_kwargs).
            # Without these the run silently uses default guardrails, default YAML
            # pricing and an unscoped channel, so the "current plan" claim would be
            # false whenever the operator's settings deviate from the defaults.
            today=server._reference_today(settings),
            settings=server._model_dump(settings),
        )
    except Exception as exc:  # pragma: no cover - data/environment dependent
        return {"available": False, "reason": f"Optimizer run failed: {str(exc)[:200]}", "count": 0, "breaks": [], "by_day": []}

    gold = [p for p in payload.get("placements", []) if p.get("is_gold")]
    if not gold:
        return {
            "available": True,
            "enabled": True,
            "count": 0,
            "reason": "No gold breaks in the current plan (none configured as gold in overrides).",
            "max_per_day": settings.gold_breaks_max_per_day,
            "breaks": [],
            "by_day": [],
        }

    breaks: list[dict[str, Any]] = []
    by_day_counts: dict[str, int] = {}
    for placement in gold:
        day = str(placement.get("day") or "")
        by_day_counts[day] = by_day_counts.get(day, 0) + 1
        breaks.append(
            {
                "segment_id": placement.get("segment_id"),
                "channel": placement.get("channel"),
                "day": day,
                "start_time": placement.get("start_time"),
                "program_type": placement.get("program_type"),
                "duration_seconds": placement.get("duration_seconds"),
                "revenue": placement.get("revenue"),
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

    def _run(weight: int) -> dict[str, Any]:
        return run_scenario(
            revenue_weight=weight,
            retention_floor=floor,
            max_breaks_per_hour=max_bph,
            risk_lambda=risk,
            today=reference_today,
            settings=settings_map,
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
