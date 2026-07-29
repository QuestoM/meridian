"""The event pipeline for the assistant: one honest snapshot plus the write gate.

Two responsibilities, both server-side:

- :func:`pipeline_snapshot` powers the ``get_event_pipeline`` READ tool and the
  keyword grounding section: every stage of the pipeline in operational order
  (events store, operator-asserted pricing layer, schedule freshness, measured
  training gate, and the acting account's write permission), each stage built
  from the real store or artifact and honestly absent-with-reason on failure,
  never invented. The training gate is tri-state: until a rebuild writes the
  ``event_layer_gate`` metadata key the verdict reads ``unknown``.

- :func:`company_refusal` enforces the company-only event-write policy in the
  propose path itself (kairos_api.assistant_tools.handle_tool_use), not just by
  tool-list filtering: propose_event_change, propose_agency_change and any
  pricing proposal touching ``pricing_activation.events`` are refused for
  channel-affiliated accounts with a clear Hebrew reason, and no proposal item
  is captured. Read tools stay open to every authenticated account.

Affiliation comes from ``auth_store.is_company_user`` (frozen contract: company
when the affiliation field is company or missing, or when auth is disabled).
Until that helper lands the module degrades conservatively: with auth disabled
every actor is company; with auth enabled and no helper, nobody is.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

MAX_OPEN_ENDED = 10
MAX_NONNEUTRAL = 12

COMPANY_ONLY_PROPOSE_TOOLS = frozenset({"propose_event_change", "propose_agency_change"})

COMPANY_REFUSAL_HE = "הפעולה שמורה לצוות החברה: חשבון משויך ערוץ רשאי לקרוא את נתוני האירועים אך אינו רשאי להציע שינויי אירועים, סוכנויות או תמחור אירועים"

PERMISSION_POLICY_HE = "כתיבת אירועים שמורה לצוות החברה בלבד; חשבונות משויכי ערוץ קוראים את הנתונים אך אינם מציעים בהם שינוי"

GATE_UNKNOWN_REASON_HE = "המודל טרם נבנה מחדש עם שכבת האירועים"

GATE_BASIS = ("measured five-fold held-out gate; the verdict self-activates on each "
              "coefficient rebuild and is never asserted by an operator or the assistant")

# The pipeline in operational order, the exact order the assistant must present
# when a new war or event starts. Each step is one display string per line.
OPERATIONAL_ORDER_HE = (
    "שלב 1, רישום: יצירת האירוע ביומן האירועים (תאריכים, עוצמה 1-5, תאריך סיום ריק למלחמה מתמשכת) דרך propose_event_change; הרישום לבדו אינו משנה אף מספר",
    "שלב 2, תמחור: הצהרת מפעיל, לא מדידה: מכפיל מחיר על האירוע והפעלת pricing_activation.events דרך propose_pricing_change; רק כשהשכבה פעילה משתנה ההכנסה הצפויה בימי האירוע והתוכנית השמורה מסומנת כלא מעודכנת",
    "שלב 3, חישוב מחדש: propose_recompute מחיל את התמחור המאושר על התוכנית השבועית",
    "שלב 4, אימון: מדידה בלבד, לעולם לא הצהרה: אנוטציות האירוע זורמות אוטומטית למסגרת המדידה בבנייה הבאה, ושער held-out (event_layer_gate) מחליט בכל בנייה אם מקדם שימור לאירועים אמיתי; עד שקיימים נתונים עם קונטרסט אמיתי הפסיקה נשארת off ואסור לזייף מקדם שימור",
)


# --- permission gate --------------------------------------------------------------
def actor_is_company(user: "str | None") -> bool:
    """Whether the acting account is company-affiliated (may write events).

    Uses auth_store.is_company_user when the parallel permissions build has
    landed it. Until then: company only when auth is disabled (single-operator
    deployments keep working), never when auth is enforced, so a channel
    account can never slip through the gap.
    """
    from kairos_api import auth, auth_store

    checker = getattr(auth_store, "is_company_user", None)
    if callable(checker):
        try:
            return bool(checker(str(user or "")))
        except Exception:  # noqa: BLE001 - a broken checker must fail closed
            logger.exception("is_company_user failed for %r", user)
            return not auth.auth_active()
    return not auth.auth_active()


def _touches_events_activation(args: dict[str, Any]) -> bool:
    changes = args.get("changes")
    activation = changes.get("pricing_activation") if isinstance(changes, dict) else None
    return isinstance(activation, dict) and "events" in activation


def company_refusal(name: str, args: dict[str, Any], user: "str | None") -> "str | None":
    """The Hebrew refusal when this propose call is company-only and the actor
    is not company; None when the call may proceed. Enforced in the propose
    path itself, so no proposal item is ever captured for a refused call."""
    gated = name in COMPANY_ONLY_PROPOSE_TOOLS or (
        name == "propose_pricing_change" and _touches_events_activation(args))
    if not gated or actor_is_company(user):
        return None
    return COMPANY_REFUSAL_HE


# --- snapshot stages --------------------------------------------------------------
def _stage_events_store() -> dict[str, Any]:
    from kairos_api import events_api

    frame = events_api._load_frame()
    records = [events_api._record(row) for _, row in frame.iterrows()]
    active = [record for record in records if record["active"]]
    open_ended = [
        {"event_id": record["event_id"], "name": record["name"], "type": record["type"],
         "start_date": record["start_date"], "intensity": record["intensity"]}
        for record in active if record["end_date"] is None
    ]
    stage: dict[str, Any] = {
        "events_total": len(records),
        "active_count": len(active),
        "active_by_type": dict(sorted(Counter(record["type"] for record in active).items())),
        "open_ended_active_count": len(open_ended),
        "open_ended_active": open_ended[:MAX_OPEN_ENDED],
        "nonneutral_multiplier_active_count": sum(
            1 for record in active if record["price_multiplier"] != 1.0),
        "store": "data/calendar_events.csv",
        "note": "recording an event changes nothing in any number by itself",
    }
    return stage


def _events_layer_enabled() -> bool:
    from kairos.optimize.pricing import PricingModel
    from kairos_api.core import _load_settings

    overrides = getattr(_load_settings(), "pricing_overrides", None) or {}
    return bool(PricingModel.from_config(overrides).enable_events)


def _stage_pricing_layer(enabled: bool) -> dict[str, Any]:
    from kairos_api import events_api

    frame = events_api._load_frame()
    plan_dates = events_api._plan_dates()
    rows = []
    for _, row in frame.iterrows():
        record = events_api._record(row)
        if not record["active"] or record["price_multiplier"] == 1.0:
            continue
        rows.append({
            "event_id": record["event_id"], "name": record["name"],
            "price_multiplier": record["price_multiplier"],
            "plan_overlap_count": len(events_api._plan_overlap_dates(record, plan_dates)),
        })
    stage: dict[str, Any] = {
        "enabled": enabled,
        "activation_flag": "pricing_activation.events",
        "nonneutral_active_events": rows[:MAX_NONNEUTRAL],
        "nonneutral_active_count": len(rows),
        "basis": "operator assertion per event, never measured; retention coefficients are untouched",
    }
    if len(rows) > MAX_NONNEUTRAL:
        stage["nonneutral_events_omitted"] = len(rows) - MAX_NONNEUTRAL
    if not enabled:
        stage["note"] = "the layer is OFF, so multipliers change no forecast until pricing_activation.events is activated"
    return stage


def _stage_freshness(events_layer_enabled: bool) -> dict[str, Any]:
    from kairos.export.schedule_freshness import GROUP_LABELS, schedule_freshness
    from kairos_api import events_api

    verdict = schedule_freshness(events_api.ROOT)
    changed = list(verdict.get("changed") or [])
    events_label = GROUP_LABELS.get("events", "special events")
    return {
        "schedule_status": verdict.get("status"),
        "computed_at": verdict.get("computed_at"),
        "changed_groups": changed,
        "stale_from_events": events_label in changed,
        "events_group_tracked": events_layer_enabled,
        "note": "the events store is fingerprinted for staleness only while pricing_activation.events is on; with the layer off the engine never reads it",
    }


def _stage_training_gate() -> dict[str, Any]:
    from kairos_api import events_api

    metadata = events_api._coefficients_metadata()
    if metadata is None:
        return {"verdict": "unknown", "reason": GATE_UNKNOWN_REASON_HE,
                "coefficients_available": False, "basis": GATE_BASIS}
    raw = metadata.get("event_layer_gate")
    if not isinstance(raw, dict):
        return {"verdict": "unknown", "reason": GATE_UNKNOWN_REASON_HE,
                "coefficients_available": True, "basis": GATE_BASIS}
    return {
        "verdict": raw.get("verdict"),
        "reason": raw.get("reason"),
        "held_out_delta_pct": raw.get("held_out_delta_pct"),
        "measured_at": raw.get("measured_at"),
        "coefficients_available": True,
        "basis": GATE_BASIS,
    }


def _stage_permissions(user: "str | None") -> dict[str, Any]:
    return {
        "actor": user or None,
        "can_propose_event_writes": actor_is_company(user),
        "policy": PERMISSION_POLICY_HE,
    }


def pipeline_snapshot(user: "str | None", include_permissions: bool = True) -> dict[str, Any]:
    """Every pipeline stage in operational order; a failed stage reads as an
    honest {"error": ...} instead of failing or faking the whole snapshot."""
    try:
        enabled = _events_layer_enabled()
    except Exception:  # noqa: BLE001 - activation unreadable stays honest below
        enabled = False
    builders: list[tuple[str, Any]] = [
        ("events_store", _stage_events_store),
        ("pricing_layer", lambda: _stage_pricing_layer(enabled)),
        ("freshness", lambda: _stage_freshness(enabled)),
        ("training_gate", _stage_training_gate),
    ]
    if include_permissions:
        builders.append(("permissions", lambda: _stage_permissions(user)))
    snapshot: dict[str, Any] = {"operational_order": list(OPERATIONAL_ORDER_HE)}
    for name, build in builders:
        try:
            snapshot[name] = build()
        except Exception as exc:  # noqa: BLE001 - honest absence beats fabrication
            logger.exception("event pipeline stage %s failed", name)
            snapshot[name] = {"error": f"{name} could not be read ({type(exc).__name__})"}
    return snapshot


def _read_get_event_pipeline(args: dict[str, Any], user: "str | None" = None) -> dict[str, Any]:
    return pipeline_snapshot(user, include_permissions=True)


PIPELINE_SOURCE = "event pipeline snapshot (events store, pricing layer, schedule freshness, training gate)"


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge the pipeline executor and its source label into the shared registry."""
    executors["get_event_pipeline"] = _read_get_event_pipeline
    sources["get_event_pipeline"] = PIPELINE_SOURCE
