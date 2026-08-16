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

# The audience wording the planned-break-rating basis can honestly serve.
_ALL_VIEWERS_MARKERS = ("כלל", "all viewers", "בתי אב", "households")


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
                tolerance_percent: Optional[float]) -> dict[str, Any]:
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


# ----------------------------------------------------------------- evaluators

def _eval_budget(ob: Obligation, inputs: Inputs, agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    sliced, resolution = _delivery_slice(ob, inputs, agreement_window)
    sums = _floor_sums(sliced)
    amount = ob.params.get("amount", {})
    target = float(amount.get("amount") or 0) or None
    w_from, w_to = ob.window_dates(agreement_window)
    counted = sums["aired_spend"]
    pace = _pace_block(counted, sums["scheduled_spend"], target, w_from, w_to,
                       inputs.today, _tolerance(ob))
    return {
        "target": {"value": target, "unit": "ILS",
                   "basis": str(amount.get("basis", "unstated"))},
        "standing": {
            "counted": counted, "unit": "ILS",
            "scheduled_ahead": sums["scheduled_spend"],
            "floor_note": "רצפת מדידה: ימים ללא מקור אינם נספרים כאפס",
            "unknown_days": sums["unknown_days"],
            "basis": "מתומחר מנוע מתוך ספר התשדירים; אינו חשבונית",
        },
        "resolution": resolution,
        **pace,
    }


def _audience_is_measurable(audience: str) -> bool:
    text = str(audience or "").strip().lower()
    if not text:
        return True
    return any(marker in text for marker in _ALL_VIEWERS_MARKERS)


def _eval_trp(ob: Obligation, inputs: Inputs, agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    audience = str(ob.params.get("audience", ""))
    points = ob.params.get("points")
    target = float(points) if points is not None else None
    sliced, resolution = _delivery_slice(ob, inputs, agreement_window)
    sums = _floor_sums(sliced)
    if not _audience_is_measurable(audience):
        return {
            "target": {"value": target, "unit": "rating_points", "audience": audience},
            "standing": {
                "counted": None, "unit": "rating_points",
                "basis": "רייטינג מתוכנן על בסיס כלל הצופים בלבד",
            },
            "resolution": resolution,
            "alarm": UNKNOWN,
            "alarm_reason": (
                "ההתחייבות נקובה בקהל יעד שאין למוצר מדידה עבורו; השוואה בין "
                "שני מטבעות שונים לא תיעשה"
            ),
            "used_default_bands": False,
        }
    w_from, w_to = ob.window_dates(agreement_window)
    counted = sums["aired_points"]
    pace = _pace_block(counted, sums["scheduled_points"], target, w_from, w_to,
                       inputs.today, _tolerance(ob))
    return {
        "target": {"value": target, "unit": "rating_points", "audience": audience,
                   "incomplete": target is None},
        "standing": {
            "counted": counted, "unit": "rating_points",
            "scheduled_ahead": sums["scheduled_points"],
            "floor_note": "רצפת מדידה: ימים ללא מקור אינם נספרים כאפס",
            "unknown_days": sums["unknown_days"],
            "basis": "רייטינג ברייק מתוכנן מיומן השידור, בסיס כלל הצופים",
        },
        "resolution": resolution,
        **pace,
    }


def _eval_effective_cpp(ob: Obligation, inputs: Inputs, agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    sliced, resolution = _delivery_slice(ob, inputs, agreement_window)
    sums = _floor_sums(sliced)
    cap = ob.params.get("cap")
    cap_value = float(cap) if cap is not None else None
    audience = str(ob.params.get("audience", ""))
    if not _audience_is_measurable(audience):
        return {
            "target": {"value": cap_value, "unit": "ILS_per_point", "audience": audience},
            "standing": {"counted": None, "unit": "ILS_per_point"},
            "resolution": resolution,
            "alarm": UNKNOWN,
            "alarm_reason": "התקרה נקובה בקהל יעד שאין למוצר מדידה עבורו",
            "used_default_bands": False,
        }
    points = sums["aired_points"]
    spend = sums["aired_spend"]
    effective = round(spend / points, 2) if points > 0 else None
    if effective is None or cap_value is None:
        level, reason = UNKNOWN, "אין עדיין נקודות מדודות לחישוב CPP אפקטיבי"
    elif effective <= cap_value:
        level, reason = ON_TRACK, "ה-CPP האפקטיבי בתוך התקרה"
    elif effective <= cap_value * 1.05:
        level, reason = WATCH, "ה-CPP האפקטיבי צמוד לתקרה"
    else:
        level, reason = AT_RISK, "ה-CPP האפקטיבי חורג מהתקרה"
    return {
        "target": {"value": cap_value, "unit": "ILS_per_point", "audience": audience},
        "standing": {
            "counted": effective, "unit": "ILS_per_point",
            "spend_floor": spend, "points_floor": points,
            "unknown_days": sums["unknown_days"],
            "basis": "הוצאה מתומחרת-מנוע חלקי נקודות מתוכננות; רצפת מדידה",
        },
        "resolution": resolution,
        "alarm": level,
        "alarm_reason": reason,
        "used_default_bands": False,
    }


def _eval_preferred_positions(ob: Obligation, inputs: Inputs,
                              agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    target = ob.params.get("target_percent")
    method = str(ob.params.get("counting_method", "unstated"))
    positions = [str(p) for p in ob.params.get("preferred_positions", [])]
    if inputs.preferred_rate is None:
        return {
            "target": {"value": float(target) if target is not None else None,
                       "unit": "percent", "method": method, "positions": positions},
            "standing": {"counted": None, "unit": "percent"},
            "alarm": UNKNOWN,
            "alarm_reason": "מדד המיקומים המועדפים אינו זמין בהקשר זה",
            "used_default_bands": False,
        }
    campaign_ids, how = campaigns_for(ob, inputs)
    measured = inputs.preferred_rate(
        positions=positions, method=method, campaigns=campaign_ids,
    )
    counted = measured.get("rate_percent")
    if counted is None or target is None:
        level, reason = UNKNOWN, str(measured.get("reason") or "אין מדידה זמינה")
    elif float(counted) >= float(target):
        level, reason = ON_TRACK, "שיעור המיקומים המועדפים עומד ביעד"
    elif float(counted) >= float(target) * DEFAULT_RISK_RATIO:
        level, reason = WATCH, "שיעור המיקומים המועדפים מעט מתחת ליעד"
    else:
        level, reason = AT_RISK, "שיעור המיקומים המועדפים מתחת ליעד מהותית"
    return {
        "target": {"value": float(target) if target is not None else None,
                   "unit": "percent", "method": method, "positions": positions},
        "standing": {"counted": counted, "unit": "percent",
                     "method": method, "detail": measured,
                     "basis": "נמדד בשיטה הנקובה בהסכם; השיטה חלק מהמספר"},
        "resolution": {"campaigns": campaign_ids, "resolved": how},
        "alarm": level,
        "alarm_reason": reason,
        "used_default_bands": target is not None and counted is not None,
    }


def _eval_untracked(ob: Obligation, note_he: str) -> dict[str, Any]:
    return {
        "target": {"value": None, "unit": None},
        "standing": {"counted": None, "unit": None},
        "alarm": UNKNOWN,
        "alarm_reason": note_he,
        "used_default_bands": False,
    }


def _tolerance(ob: Obligation) -> Optional[float]:
    raw = ob.params.get("tolerance_percent")
    return float(raw) if raw is not None else None


_EVALUATORS: dict[str, Callable[[Obligation, Inputs, Mapping[str, Any]], dict[str, Any]]] = {
    "budget-commitment": _eval_budget,
    "trp-delivery-guarantee": _eval_trp,
    "effective-cpp-cap": _eval_effective_cpp,
    "preferred-position-guarantee": _eval_preferred_positions,
}

_UNTRACKED_NOTES: dict[str, str] = {
    "share-commitment": "נתח דורש מכנה חיצוני (הצהרת תקציב כוללת) שטרם נמסר; לא ננחש",
    "daypart-mix": "תמהיל רצועות נמדד עם חיבור ספר התשדירים המפולח; בהמתנה",
    "length-mix": "תמהיל אורכים נמדד עם חיבור ספר התשדירים המפולח; בהמתנה",
    "flighting-obligation": "רציפות נמדדת על לוח הפעילות; בהמתנה לחיבור",
    "makegood-accrual-policy": "מדיניות הצבירה מופעלת דרך ספר הצבירות; ראו יתרות שם",
    "added-value-media": "הענקת מדיה נוספת נרשמת בספר הצבירות; ראו יתרות שם",
    "gold-break-allocation": "מעקב הקצאת ברייקי זהב יחובר ללוח השיבוץ",
}


def evaluate(ob: Obligation, inputs: Inputs,
             agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    """One obligation's full snapshot: target, standing, pace, alarm."""
    evaluator = _EVALUATORS.get(ob.term_id)
    if evaluator is not None:
        body = evaluator(ob, inputs, agreement_window)
    else:
        body = _eval_untracked(ob, _UNTRACKED_NOTES.get(
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
