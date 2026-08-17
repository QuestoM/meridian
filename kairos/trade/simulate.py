"""Simulation: what an agreement WOULD have done, before anyone signs it.

A commercial director evaluates a deal by asking what it does to the business,
not by reading its clauses. This module applies a PROPOSED (pre-approval)
termset to real historical or planned activity and reports what would have
changed — money, constraint violations, obligation standings, make-good
exposure — without touching one live store.

Three honesty rules, inherited from the engine's own:

- **Simulation never writes.** It compiles the proposal in memory and
  evaluates against loaded frames; the live rule stores are read-only here.
- **What cannot be simulated is named.** A term the compiler skips, a
  guarantee in an audience the product cannot measure, a discount whose basis
  the document never stated — each appears in ``not_simulated`` with its
  reason. A simulation that quietly drops half the agreement is worse than
  no simulation.
- **Every money figure carries its basis.** Ladders and commissions are
  period arithmetic over the spend the period actually produced; positional
  and constraint effects are counted as placements affected, NOT as revenue
  they might have produced, because the counterfactual schedule is not run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Optional

import pandas as pd

from . import obligations as ob
from . import taxonomy
from .compile import compile_termset


@dataclass
class SimulationInputs:
    """Real activity to simulate against: the same frames the obligations
    engine reads, plus the per-spot ledger when placement effects are wanted."""

    delivery: pd.DataFrame
    campaigns: pd.DataFrame
    agency_links: pd.DataFrame
    today: date
    spots: Optional[pd.DataFrame] = None
    window: Optional[dict[str, str]] = None


def _window_slice(frame: pd.DataFrame, column: str,
                  window: Optional[Mapping[str, str]]) -> pd.DataFrame:
    if frame.empty or not window:
        return frame
    dates = pd.to_datetime(frame[column], errors="coerce").dt.date
    out = frame
    if window.get("from"):
        start = date.fromisoformat(str(window["from"])[:10])
        out = out[dates >= start]
        dates = dates[dates >= start]
    if window.get("to"):
        end = date.fromisoformat(str(window["to"])[:10])
        out = out[dates <= end]
    return out


def _period_spend(inputs: SimulationInputs, campaign_ids: list[str]) -> dict[str, float]:
    frame = inputs.delivery
    if frame.empty or not campaign_ids:
        return {"aired": 0.0, "scheduled": 0.0, "unknown_days": 0}
    sliced = frame[frame["campaign_id"].astype(str).isin(campaign_ids)]
    sliced = _window_slice(sliced, "broadcast_date", inputs.window)
    def _sum(state: str) -> float:
        part = sliced[sliced["air_state"] == state]
        return round(float(pd.to_numeric(part.get("spend_ils"), errors="coerce")
                           .fillna(0).sum()), 2)
    return {
        "aired": _sum("aired"),
        "scheduled": _sum("scheduled"),
        "unknown_days": int((sliced["air_state"] == "unknown").sum()),
    }


def _ladder_effect(term: Mapping[str, Any], spend: float) -> dict[str, Any]:
    """What a ladder would take off the period's own spend."""
    tiers = sorted(
        ({"threshold": float(t.get("threshold") or 0),
          "discount_percent": float(t.get("discount_percent") or 0)}
         for t in term.get("tiers", [])),
        key=lambda t: t["threshold"],
    )
    if not tiers:
        return {"available": False, "reason_he": "אין מדרגות בטבלת ההנחות"}
    mechanics = str(term.get("mechanics") or "unstated")
    reached = [t for t in tiers if spend >= t["threshold"]]
    current = reached[-1] if reached else None
    ahead = next((t for t in tiers if t["threshold"] > spend), None)
    if mechanics == "retroactive":
        discount = (current or {"discount_percent": 0})["discount_percent"] / 100.0 * spend
        basis_he = "רטרואקטיבי: שיעור המדרגה שהושגה חל על מלוא ההיקף"
    elif mechanics == "marginal":
        # Each tier prices the band between its own threshold and the next
        # tier's (or the spend, whichever comes first). Bands never overlap
        # and always sum back to the spend — the property the test pins.
        discount = 0.0
        for index, tier in enumerate(tiers):
            lower = tier["threshold"]
            upper = tiers[index + 1]["threshold"] if index + 1 < len(tiers) else float("inf")
            band = max(0.0, min(spend, upper) - lower)
            discount += band * tier["discount_percent"] / 100.0
        basis_he = "שולי: כל מדרגה מתמחרת את הפלח שלה בלבד"
    else:
        return {"available": False,
                "reason_he": "מנגנון המדרגות לא נקבע במסמך (רטרואקטיבי או שולי)"}
    return {
        "available": True,
        "spend_basis": round(spend, 2),
        "tier_reached_percent": (current or {}).get("discount_percent"),
        "discount_value": round(discount, 2),
        "next_tier": ahead,
        "distance_to_next": round(ahead["threshold"] - spend, 2) if ahead else None,
        "mechanics": mechanics,
        "basis_he": basis_he,
    }


def _commission_effect(term: Mapping[str, Any], gross: float,
                       discount: float) -> dict[str, Any]:
    percent = term.get("percent")
    if percent is None:
        return {"available": False, "reason_he": "שיעור העמלה לא נקבע"}
    base = str(term.get("base") or "unstated")
    amount_base = gross - discount if base == "net_of_discount" else gross
    return {
        "available": True,
        "percent": float(percent),
        "base": base,
        "base_value": round(amount_base, 2),
        "commission_value": round(amount_base * float(percent) / 100.0, 2),
        "basis_he": ("על הנטו לאחר הנחות" if base == "net_of_discount"
                     else "על הברוטו"),
    }


def simulate(termset: Mapping[str, Any], head: Mapping[str, Any],
             inputs: SimulationInputs) -> dict[str, Any]:
    """Apply a proposed agreement to real activity. Writes nothing."""
    artifacts = compile_termset(termset, head)
    obligations = ob.materialise(termset, head)
    ob_inputs = ob.Inputs(
        delivery=inputs.delivery, campaigns=inputs.campaigns,
        agency_links=inputs.agency_links, today=inputs.today,
    )
    campaign_ids: list[str] = []
    resolution = "no obligation to resolve scope from"
    if obligations:
        campaign_ids, resolution = ob.campaigns_for(obligations[0], ob_inputs)
    else:
        probe = ob.Obligation(
            obligation_id="probe", agreement_id=str(head.get("agreement_id", "")),
            version_id="", instance_id="", term_id="budget-commitment",
            params={}, scope={}, window={},
            counterparty=dict(head.get("counterparty", {})),
        )
        campaign_ids, resolution = ob.campaigns_for(probe, ob_inputs)

    spend = _period_spend(inputs, campaign_ids)
    gross = spend["aired"]

    money: dict[str, Any] = {"gross_aired": gross,
                             "scheduled_ahead": spend["scheduled"],
                             "unknown_days": spend["unknown_days"],
                             "basis_he": "מתומחר-מנוע מספר התשדירים; רצפת מדידה"}
    not_simulated: list[dict[str, Any]] = [
        {"instance_id": s["instance_id"], "term_id": s["term_id"],
         "reason_he": s["reason_he"]}
        for s in artifacts.skipped
    ]

    settlement_terms = {t["kind"]: t for t in artifacts.settlement.get("terms", [])}
    discount_value = 0.0
    if "discount_ladder" in settlement_terms:
        ladder = _ladder_effect(settlement_terms["discount_ladder"], gross)
        money["discount_ladder"] = ladder
        if ladder.get("available"):
            discount_value = ladder["discount_value"]
        else:
            not_simulated.append({
                "term_id": "volume-discount-ladder",
                "reason_he": ladder.get("reason_he", ""),
            })
    if "agency_commission" in settlement_terms:
        commission = _commission_effect(
            settlement_terms["agency_commission"], gross, discount_value)
        money["agency_commission"] = commission
        if not commission.get("available"):
            not_simulated.append({
                "term_id": "agency-commission",
                "reason_he": commission.get("reason_he", ""),
            })
    money["net_after_simulated_terms"] = round(
        gross - discount_value - (money.get("agency_commission", {}).get("commission_value") or 0.0),
        2,
    )

    # Placement effects are counted, never priced: the counterfactual schedule
    # is not run, so claiming revenue for a constraint would be fabrication.
    placement = {
        "conditions": len(artifacts.conditions),
        "frequency_rules": len(artifacts.frequency_rules),
        "note_he": (
            "אילוצי שיבוץ נספרים ואינם מתומחרים: לוח חלופי לא הורץ, ולכן "
            "ייחוס הכנסה לאילוץ יהיה המצאה"
        ),
    }
    if inputs.spots is not None and not inputs.spots.empty:
        placement["spots_in_window"] = int(len(
            _window_slice(inputs.spots, "date", inputs.window)))

    standings = ob.evaluate_all(termset, head, ob_inputs)
    for snapshot in standings:
        if snapshot.get("alarm") == ob.UNKNOWN:
            not_simulated.append({
                "instance_id": snapshot.get("instance_id"),
                "term_id": snapshot.get("term_id"),
                "reason_he": snapshot.get("alarm_reason", ""),
            })

    exposure = [
        {"term_id": s["term_id"], "instance_id": s["instance_id"],
         "alarm": s["alarm"], "reason_he": s.get("alarm_reason", ""),
         "target": s.get("target"), "standing": s.get("standing")}
        for s in standings if s.get("alarm") in (ob.AT_RISK, ob.BREACHED)
    ]

    return {
        "agreement_id": head.get("agreement_id"),
        "window": inputs.window or head.get("window", {}),
        "scope": {"campaigns": campaign_ids, "resolved": resolution},
        "money": money,
        "placement": placement,
        "obligations": standings,
        "exposure": exposure,
        "not_simulated": not_simulated,
        "compiled": artifacts.summary(),
        "headline_he": _headline(money, exposure, not_simulated),
    }


def _headline(money: Mapping[str, Any], exposure: list[dict[str, Any]],
              not_simulated: list[dict[str, Any]]) -> str:
    gross = money.get("gross_aired") or 0
    net = money.get("net_after_simulated_terms") or 0
    parts = [f"על פעילות מדודה של {gross:,.0f} ₪ ברוטו, ההסכם המוצע מותיר "
             f"{net:,.0f} ₪ לאחר ההנחות והעמלות שניתן היה לחשב"]
    if exposure:
        parts.append(f"{len(exposure)} התחייבויות בסיכון או בהפרה")
    if not_simulated:
        parts.append(f"{len(not_simulated)} מונחים לא ניתנים לסימולציה ומפורטים בנפרד")
    return "; ".join(parts) + "."
