"""Living obligations: standing, pace, projection and alarm per committed term.

An approved agreement's obliging terms (family D and E of the taxonomy) become
Obligation records here, each computed against the product's own measured
ledgers — never against numbers a caller asserts. The honesty rules mirror
the delivery ledger they read from:

- **Counted figures are floors.** Days without a per-spot source are unknown,
  not zero, and every standing names its basis and its as-of instant.
- **A guarantee in a currency the product does not hold reports UNKNOWN.**
  Rating standing is planned break rating on the all-viewers base; a guarantee
  naming another audience is measured as unknown with the reason spelled out,
  exactly as the delivery board refuses the same comparison.
- **Thresholds come from the term.** A tolerance the agreement states is the
  tolerance; the alarm ladder's default bands are visible constants, labeled
  as defaults on every payload that used them.

This module is pure computation over injected frames so tests need no disk;
kairos_api.trade_obligations does the wiring to the real stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Mapping, Optional

import pandas as pd

# Alarm ladder. Bands apply to the pace ratio (counted vs expected-to-date);
# they are DEFAULTS, used only where the term states no tighter figure, and
# every payload that used them says so.
ON_TRACK = "on_track"
WATCH = "watch"
AT_RISK = "at_risk"
BREACHED = "breached"
UNKNOWN = "unknown"
DEFAULT_WATCH_RATIO = 0.95
DEFAULT_RISK_RATIO = 0.85

OBLIGING_KINDS = (
    "budget-commitment",
    "trp-delivery-guarantee",
    "effective-cpp-cap",
    "preferred-position-guarantee",
    "daypart-mix",
    "flighting-obligation",
    "length-mix",
    "share-commitment",
    "makegood-accrual-policy",
    "added-value-media",
    "gold-break-allocation",
)

@dataclass
class Inputs:
    """The measured substrate an evaluation reads. Frames follow the on-disk
    schemas: delivery = campaigns_delivery.COLUMNS; campaigns maps campaign_id
    to advertiser (+ agency via links); spots is the per-spot money ledger."""

    delivery: pd.DataFrame
    campaigns: pd.DataFrame
    agency_links: pd.DataFrame
    today: date
    preferred_rate: Optional[Callable[..., dict[str, Any]]] = None
    # Optional forecast provider, called as ``forecast_points(counted=,
    # scheduled=)`` with the obligation's measured points and its own
    # not-yet-aired delivery rows. It returns the block
    # kairos.trade.obligations_forecast.forward_line builds, so a TRP standing
    # can carry a third projection line beside booked-forward and pace-forward.
    # Absent (the default) nothing changes: no line appears and every existing
    # figure is byte-identical.
    forecast_points: Optional[Callable[..., dict[str, Any]]] = None


@dataclass
class Obligation:
    """One committed term under continuous measurement."""

    obligation_id: str
    agreement_id: str
    version_id: str
    instance_id: str
    term_id: str
    params: dict[str, Any]
    scope: dict[str, Any]
    window: dict[str, Any]
    counterparty: dict[str, Any]

    def window_dates(self, agreement_window: Mapping[str, Any]) -> tuple[Optional[date], Optional[date]]:
        raw_from = self.window.get("from") or agreement_window.get("starts_on") or agreement_window.get("from")
        raw_to = self.window.get("to") or agreement_window.get("ends_on") or agreement_window.get("to")
        return (_to_date(raw_from), _to_date(raw_to))


def _to_date(raw: Any) -> Optional[date]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def materialise(termset: Mapping[str, Any], head: Mapping[str, Any]) -> list[Obligation]:
    """Obliging instances of an approved termset become Obligation records."""
    out: list[Obligation] = []
    version_id = str(termset.get("version_id", ""))
    agreement_id = str(termset.get("agreement_id", head.get("agreement_id", "")))
    for inst in termset.get("instances", []):
        if inst.get("term_id") not in OBLIGING_KINDS:
            continue
        out.append(Obligation(
            obligation_id=f"ob-{version_id}-{inst['instance_id']}",
            agreement_id=agreement_id,
            version_id=version_id,
            instance_id=str(inst["instance_id"]),
            term_id=str(inst["term_id"]),
            params=dict(inst.get("params", {})),
            scope=dict(inst.get("scope", {})),
            window=dict(inst.get("window", {})),
            counterparty=dict(head.get("counterparty", {})),
        ))
    return out


# ------------------------------------------------------------- scope resolution

def campaigns_for(ob: Obligation, inputs: Inputs) -> tuple[list[str], str]:
    """The campaign ids an obligation measures over, and how they were found."""
    frame = inputs.campaigns
    if frame.empty:
        return [], "no campaigns on file"
    scoped = frame
    scope_campaigns = [str(c) for c in ob.scope.get("campaigns", [])]
    if scope_campaigns:
        scoped = scoped[scoped["campaign_id"].astype(str).isin(scope_campaigns)]
        return scoped["campaign_id"].astype(str).tolist(), "scoped to named campaigns"
    advertisers = {str(a) for a in ob.scope.get("advertisers", [])}
    counterparty = ob.counterparty or {}
    if not advertisers and counterparty.get("advertiser"):
        advertisers = {str(counterparty["advertiser"])}
    if not advertisers and counterparty.get("agency"):
        agency = str(counterparty["agency"])
        links = inputs.agency_links
        if not links.empty:
            mask = links.get("agency_id", pd.Series(dtype=str)).astype(str) == agency
            if "agency_name" in links.columns:
                mask = mask | (links["agency_name"].astype(str) == agency)
            advertisers = set(links[mask]["advertiser"].astype(str))
        if not advertisers:
            return [], f"no advertiser links on file for agency {agency}"
    if advertisers and "advertiser" in scoped.columns:
        scoped = scoped[scoped["advertiser"].astype(str).isin(advertisers)]
        return scoped["campaign_id"].astype(str).tolist(), (
            "campaigns of " + ", ".join(sorted(advertisers))
        )
    return scoped["campaign_id"].astype(str).tolist(), "all campaigns on file"


def _delivery_slice(ob: Obligation, inputs: Inputs,
                    agreement_window: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    campaign_ids, how = campaigns_for(ob, inputs)
    frame = inputs.delivery
    if frame.empty or not campaign_ids:
        return frame.iloc[0:0], {"campaigns": campaign_ids, "resolved": how}
    sliced = frame[frame["campaign_id"].astype(str).isin(campaign_ids)].copy()
    w_from, w_to = ob.window_dates(agreement_window)
    dates = pd.to_datetime(sliced["broadcast_date"], errors="coerce").dt.date
    if w_from:
        sliced = sliced[dates >= w_from]
        dates = dates[dates >= w_from]
    if w_to:
        sliced = sliced[dates <= w_to]
    return sliced, {"campaigns": campaign_ids, "resolved": how}


def _floor_sums(sliced: pd.DataFrame) -> dict[str, Any]:
    aired = sliced[sliced["air_state"] == "aired"]
    scheduled = sliced[sliced["air_state"] == "scheduled"]
    unknown = sliced[sliced["air_state"] == "unknown"]

    def _sum(frame: pd.DataFrame, column: str) -> float:
        return float(pd.to_numeric(frame.get(column), errors="coerce").fillna(0).sum())

    return {
        "aired_spend": round(_sum(aired, "spend_ils"), 2),
        "scheduled_spend": round(_sum(scheduled, "spend_ils"), 2),
        "aired_points": round(_sum(aired, "rating_points_planned"), 3),
        "scheduled_points": round(_sum(scheduled, "rating_points_planned"), 3),
        "unknown_days": int(len(unknown)),
        "aired_days": int(len(aired)),
        "scheduled_days": int(len(scheduled)),
    }


# --------------------------------------------------------------- pace & alarm

def _expected_fraction(w_from: Optional[date], w_to: Optional[date], today: date) -> Optional[float]:
    if not w_from or not w_to or w_to < w_from:
        return None
    total = (w_to - w_from).days + 1
    elapsed = (min(today, w_to) - w_from).days + 1
    if elapsed <= 0:
        return 0.0
    return min(1.0, elapsed / total)


def _alarm(ratio: Optional[float], window_closed: bool, tolerance_percent: Optional[float],
           counted: Optional[float], target: Optional[float]) -> tuple[str, str, bool]:
    """(level, reason, used_default_bands)."""
    if ratio is None or counted is None or target is None or target <= 0:
        return UNKNOWN, "אין בסיס מדוד להשוואה", False
    if window_closed:
        tolerance = (tolerance_percent or 0.0) / 100.0
        if counted >= target * (1 - tolerance):
            return ON_TRACK, "החלון נסגר בעמידה ביעד", False
        return BREACHED, "החלון נסגר מתחת ליעד בניכוי הסטייה המותרת", False
    if ratio >= DEFAULT_WATCH_RATIO:
        return ON_TRACK, "בקצב היעד", True
    if ratio >= DEFAULT_RISK_RATIO:
        return WATCH, "מעט מאחורי הקצב", True
    return AT_RISK, "מאחורי הקצב באופן מהותי", True


def _pace_block(counted: float, scheduled: float, target: Optional[float],
                w_from: Optional[date], w_to: Optional[date], today: date,
                tolerance_percent: Optional[float],
                forecast_forward: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    fraction = _expected_fraction(w_from, w_to, today)
    expected = (target * fraction) if (target and fraction is not None) else None
    ratio = (counted / expected) if expected else None
    window_closed = bool(w_to and today > w_to)
    level, reason, defaults = _alarm(ratio, window_closed, tolerance_percent, counted, target)
    projection = None
    method = None
    if target and fraction and fraction > 0:
        # Committed-forward projection: what is already counted plus what is
        # already booked ahead; the pace-forward estimate rides beside it.
        booked_forward = counted + scheduled
        pace_forward = counted / fraction
        projection = round(max(booked_forward, 0.0), 2)
        method = {
            "booked_forward": round(booked_forward, 2),
            "pace_forward": round(pace_forward, 2),
            "note": "התחזית המחייבת היא מדוד + מתוזמן; קצב-קדימה מוצג לצידה",
        }
        # A third line, only when a forecast was actually available for these
        # days (see kairos.trade.obligations_forecast). It rides BESIDE the two
        # above and replaces neither: the committed projection stays booked
        # points, because a model's expectation is not a booking.
        if forecast_forward and forecast_forward.get("available"):
            method["forecast_forward"] = forecast_forward
    return {
        "expected_to_date": round(expected, 2) if expected is not None else None,
        "window_fraction": round(fraction, 4) if fraction is not None else None,
        "ratio": round(ratio, 4) if ratio is not None else None,
        "window_closed": window_closed,
        "alarm": level,
        "alarm_reason": reason,
        "used_default_bands": defaults,
        "projection": projection,
        "projection_method": method,
    }


def evaluate(ob: Obligation, inputs: Inputs,
             agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    """One obligation's full snapshot: target, standing, pace, alarm."""
    # Late import: the evaluators read this module's own helpers, so the table
    # cannot be resolved at import time without a cycle.
    from kairos.trade import obligations_evaluators as ev

    evaluator = ev.EVALUATORS.get(ob.term_id)
    if evaluator is not None:
        body = evaluator(ob, inputs, agreement_window)
    else:
        body = ev.eval_untracked(ob, ev.UNTRACKED_NOTES.get(
            ob.term_id, "מונח מחויב שטרם חובר למדידה רציפה"))
    w_from, w_to = ob.window_dates(agreement_window)
    return {
        "obligation_id": ob.obligation_id,
        "agreement_id": ob.agreement_id,
        "version_id": ob.version_id,
        "instance_id": ob.instance_id,
        "term_id": ob.term_id,
        "window": {"from": w_from.isoformat() if w_from else None,
                   "to": w_to.isoformat() if w_to else None},
        "evaluated_at": inputs.today.isoformat(),
        **body,
    }


def evaluate_all(termset: Mapping[str, Any], head: Mapping[str, Any],
                 inputs: Inputs) -> list[dict[str, Any]]:
    agreement_window = head.get("window", {}) or {}
    return [
        evaluate(ob, inputs, agreement_window)
        for ob in materialise(termset, head)
    ]
