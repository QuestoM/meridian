"""Comparing several competing versions of one day, with the reasoning shown.

Three people propose three versions of one Tuesday. The product's job is not to
render three columns of numbers; it is to let one person decide in seconds and
be able to defend the decision afterwards. So every figure here arrives with the
thing that produced it.

**The delta is attributed, never narrated.** A money difference between two
versions of a day is decomposed from the row-level diff itself, in
:mod:`kairos_api.day_compare_attribution`: each programme lands in exactly one
bucket according to what actually changed on it, each bucket is cut by daypart,
and the whole thing sums to the scoped headline difference in integer agorot with
any residue reported as an ``unattributed`` bucket rather than absorbed.

**Everything is scoped once.** Totals, net-of-retention, inventory and the
attribution are all computed from the same operator-scoped frame through
:mod:`kairos_api.channel_scope`, so the headline money and the sum of the
explanation cannot be two different scopes. The scope note travels with every
money figure on every side.

**A side that cannot be scored says why.** An unknown is never a zero: a missing
frozen file, a missing baseline, rows on a different basis from the baseline's -
each produces an explicit unavailable with the reason named, and the other sides
still compare.

The two dimensions a money figure cannot carry - what the day leaves sellable,
and which commitments each version advances or endangers - live in
:mod:`kairos_api.day_compare_standing` and are folded in here.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from kairos_api import channel_scope, day_compare_attribution as attribution
from kairos_api import day_compare_standing as standing
from kairos_api import day_proposal_store as store
from kairos_api.plan_version_store import _totals


def _money(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """One side's headline money, plus the scoped frame every other figure uses.

    The frame comes back with the money on purpose: totals, net-of-retention,
    inventory and the attribution all have to be computed on the very rows the
    headline was summed over, or the explanation and the number it explains are
    on two different scopes.
    """
    from kairos.optimize.revenue_net import frame_revenue_net

    owned, note = channel_scope.scope_frame(frame)
    totals = _totals(owned)
    net = frame_revenue_net(owned)
    return {
        "revenue": totals["revenue"],
        "breaks": totals["breaks"],
        "ad_seconds": totals["ad_seconds"],
        "rows": totals["rows"],
        "revenue_net_of_retention": net.get("revenue_net_ils") if net.get("available") else None,
        "retention_cost": net.get("retention_cost_ils") if net.get("available") else None,
        "net_available": bool(net.get("available")),
        "net_reason": "" if net.get("available") else str(net.get("reason") or ""),
        "currency": "ILS",
        "scope": note,
    }, owned


def _money_delta(side: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    net = None
    if side["net_available"] and base["net_available"]:
        net = round(side["revenue_net_of_retention"] - base["revenue_net_of_retention"], 2)
    revenue_delta = round(side["revenue"] - base["revenue"], 2)
    percent = (round(100.0 * revenue_delta / base["revenue"], 2)
               if base["revenue"] else None)
    return {
        "revenue": revenue_delta,
        "revenue_percent": percent,
        "breaks": side["breaks"] - base["breaks"],
        "ad_seconds": round(side["ad_seconds"] - base["ad_seconds"], 1),
        "revenue_net_of_retention": net,
        "net_reason": "" if net is not None else "אחד הצדדים אינו נושא את הנתונים לחישוב הכנסה בניכוי נטישה",
    }


def _commitment_phrase(dimension: dict[str, Any]) -> str:
    if not dimension.get("available"):
        # The dimension says WHY it is unmeasured; "not checked" would claim
        # the check never ran, when it ran and found its basis missing.
        return str(dimension.get("phrase_he") or "עמידה בהתחייבויות: לא נבדק")
    counts = dimension.get("counts") or {}
    for key, one, many in (
        (standing.BREAKS, "התחייבות אחת בהפרה", "התחייבויות בהפרה"),
        (standing.ENDANGERS, "התחייבות אחת בסיכון", "התחייבויות בסיכון"),
        (standing.ADVANCES, "התחייבות אחת מתקדמת", "התחייבויות מתקדמות"),
    ):
        total = int(counts.get(key, 0))
        if total == 1:
            return one
        if total > 1:
            return f"{total} {many}"
    # An all-unknown standing is NOT "no change": nothing was measured, and
    # saying "no change" converts an honest unknown into a reassurance. The
    # adversarial suite planted a strict tripwire on exactly this line.
    unknown = int(counts.get(standing.UNKNOWN, 0))
    measured_any = any(
        int(counts.get(key, 0)) > 0
        for key in (standing.BREAKS, standing.ENDANGERS, standing.ADVANCES,
                    standing.UNCHANGED)
    )
    if unknown and not measured_any:
        return ("השפעה על התחייבויות אינה ידועה: "
                f"{unknown} התחייבויות ללא בסיס מדידה")
    return "אין שינוי בעמידה בהתחייבויות"


def _inventory_phrase(inventory: dict[str, Any]) -> str:
    remaining = inventory.get("daily_ad_seconds_remaining")
    if remaining is None:
        return "תקרת שניות יומית אינה מוגדרת בהגדרות"
    if remaining < 0:
        return f"חריגה של {abs(remaining):,.0f} שניות מהתקרה היומית"
    return f"נותרו {remaining:,.0f} שניות פרסום ביום"


def _headline(delta: dict[str, Any], attributed: dict[str, Any],
              inventory: dict[str, Any], commitments: dict[str, Any]) -> str:
    """The one line a decision-maker reads before opening anything beneath it."""
    money = f"{delta['revenue']:+,.0f} ₪"
    if delta["revenue_percent"] is not None:
        money += f" ({delta['revenue_percent']:+.1f}%)"
    breaks = (f"{delta['breaks']:+d} ברייקים" if delta["breaks"]
              else "ללא שינוי במספר הברייקים")
    parts = [money, breaks]
    cells = attributed.get("cells") or []
    if cells:
        parts.append(f"בעיקר: {cells[0]['sentence_he']}")
    parts.append(_commitment_phrase(commitments))
    parts.append(_inventory_phrase(inventory))
    return " · ".join(parts)


def _standing_for(channel: str, date: str, baseline_rows: pd.DataFrame,
                  side_rows: pd.DataFrame, context: Optional[dict[str, Any]]) -> dict[str, Any]:
    if context is None:
        return {"available": False,
                "reason": "no trade context was supplied, so contractual standing was not measured",
                "reason_he": "לא נמסר הקשר מסחרי, ולכן עמידה בהתחייבויות לא נמדדה",
                "counts": {}, "obligations": []}
    return standing.contractual_standing(
        channel=channel, date=date, baseline_rows=baseline_rows, side_rows=side_rows,
        approved=context["approved"], delivery=context["delivery"],
        campaigns=context["campaigns"], links=context["links"], today=context["today"],
    )


def _side(*, side_id: str, label: str, rows: pd.DataFrame, baseline_frame: pd.DataFrame,
          baseline_money: dict[str, Any], channel: str, date: str, caps: dict[str, Any],
          context: Optional[dict[str, Any]], manifest: Optional[dict[str, Any]],
          basis: str, baseline_basis: str) -> dict[str, Any]:
    money, owned = _money(rows)
    delta = _money_delta(money, baseline_money)
    inventory = standing.inventory_consequence(owned, caps)
    commitments = _standing_for(channel, date, baseline_frame, owned, context)
    if basis == baseline_basis:
        attributed = attribution.attribute(baseline_frame, owned)
    else:
        attributed = {
            "available": False,
            "reason": f"this side's rows are on the {basis!r} basis and the baseline is on "
                      f"{baseline_basis!r}; a placement-level attribution across two bases would "
                      "compare two different things",
            "reason_he": "שורות הצד הזה נשענות על בסיס אחר מזה של הבסיס להשוואה, ולכן שיוך ברמת "
                         "השיבוץ לא ייעשה",
            "cells": [], "buckets": [],
        }
    engine = (manifest or {}).get("engine") or {}
    verdict = engine.get("compliance") if isinstance(engine, dict) else None
    return {
        "side_id": side_id,
        "label": label,
        "available": True,
        "basis": basis,
        "basis_matches_baseline": basis == baseline_basis,
        "money": money,
        "delta": delta,
        "attribution": attributed,
        "inventory": inventory,
        "commitments": commitments,
        "compliance": verdict if isinstance(verdict, dict) else {
            "available": False,
            "reason": "no engine guardrail verdict was recorded when this side was authored, and "
                      "the plan schema does not carry the placements a guardrail run needs",
            "reason_he": "לא נרשם פסק מנוע על מגבלות הרישיון בעת יצירת הגרסה, ושורות הלוח אינן "
                         "נושאות את מיקומי הברייקים הדרושים לבדיקה מחדש",
        },
        "headline": _headline(delta, attributed, inventory, commitments),
        "manifest": manifest,
    }


def compare(
    channel: str,
    date: str,
    proposal_ids: list[str],
    *,
    include_live: bool = False,
    baseline_rows: Optional[pd.DataFrame] = None,
    baseline_ref: Optional[dict[str, Any]] = None,
    baseline_label: str = "התוכנית כפי שהיא",
    live_rows: Optional[pd.DataFrame] = None,
    live_basis: str = "committed-weekly-plan",
    caps: Optional[dict[str, Any]] = None,
    trade_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Two or more competing versions of one day, side by side, with the reasoning.

    ``baseline_rows`` is the day everybody proposed against, as it stands right
    now; every side is diffed against it and every proposal's staleness against
    it is reported in the same payload, so a version authored before the day
    moved cannot look current merely because its own numbers are internally
    consistent.
    """
    wanted = [str(item).strip() for item in (proposal_ids or []) if str(item).strip()]
    if baseline_rows is None or not len(baseline_rows):
        return {"available": False,
                "reason": "the day this comparison is against could not be read, so there is "
                          "nothing common to diff the versions from",
                "reason_he": "לא ניתן לקרוא את היום שאליו משווים, ולכן אין בסיס משותף להשוואה",
                "sides": []}
    if len(wanted) + (1 if include_live else 0) < 2:
        return {"available": False,
                "reason": "a comparison needs at least two sides",
                "reason_he": "להשוואה נדרשים לפחות שני צדדים",
                "sides": []}

    caps = caps or {}
    baseline_basis = str((baseline_ref or {}).get("basis") or "engine-day-plan")
    baseline_money, baseline_frame = _money(baseline_rows)
    baseline_inventory = standing.inventory_consequence(baseline_frame, caps)

    sides: list[dict[str, Any]] = []
    for proposal_id in wanted:
        manifest = store.get(channel, date, proposal_id)
        if manifest is None:
            sides.append({"side_id": proposal_id, "available": False,
                          "reason": f"no proposal {proposal_id} for {channel} on {date}",
                          "reason_he": "ההצעה המבוקשת אינה קיימת ליום הזה"})
            continue
        rows = store.rows_for(channel, date, proposal_id)
        if rows is None or not len(rows):
            sides.append({"side_id": proposal_id, "available": False,
                          "label": manifest.get("name"),
                          "reason": f"proposal {proposal_id} has no frozen day file",
                          "reason_he": "להצעה הזו אין קובץ יום שמור, ולכן אי אפשר לתמחר אותה"})
            continue
        side = _side(
            side_id=proposal_id, label=str(manifest.get("name") or proposal_id), rows=rows,
            baseline_frame=baseline_frame, baseline_money=baseline_money, channel=channel,
            date=date, caps=caps, context=trade_context, manifest=manifest,
            basis=str((manifest.get("baseline_ref") or {}).get("basis") or baseline_basis),
            baseline_basis=baseline_basis,
        )
        side["status"] = manifest.get("status")
        side["author"] = manifest.get("author")
        side["created_at"] = manifest.get("created_at")
        side["staleness"] = store.staleness(manifest, baseline_ref)
        sides.append(side)

    if include_live:
        if live_rows is None or not len(live_rows):
            sides.append({"side_id": "live", "available": False, "label": "התוכנית החיה",
                          "reason": "the committed weekly plan carries no row for this channel-day",
                          "reason_he": "התוכנית השבועית השמורה אינה מחזיקה שורה ליום הזה בערוץ הזה"})
        else:
            sides.append(_side(
                side_id="live", label="התוכנית החיה", rows=live_rows,
                baseline_frame=baseline_frame, baseline_money=baseline_money, channel=channel,
                date=date, caps=caps, context=trade_context, manifest=None,
                basis=live_basis, baseline_basis=baseline_basis,
            ))

    scored = [side for side in sides if side.get("available")]
    best = max(scored, key=lambda side: side["delta"]["revenue"], default=None)
    return {
        "available": True,
        "channel": channel,
        "date": date,
        "sides": sides,
        "side_count": len(sides),
        "scored_sides": len(scored),
        "baseline": {
            "label": baseline_label,
            "basis": baseline_basis,
            "ref": baseline_ref or {},
            "money": baseline_money,
            "inventory": baseline_inventory,
        },
        "highest_revenue_side": (best or {}).get("side_id"),
        "note_he": ("כל סכום מוצג בהיקף הערוץ של המפעיל, וההסבר לפער נגזר משורות הלוח עצמן "
                    "ולא מתיאור מילולי"),
    }
