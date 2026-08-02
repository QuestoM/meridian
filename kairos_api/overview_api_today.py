"""Today: the three answers a general manager reads without clicking anything.

Is this week on plan, is anything broken, what needs a decision. This module
composes those three from what the product already measures, and it is a
projection of the same cached overview body ``/api/overview`` serves, never a
second computation, so the two surfaces can never disagree about a figure.

Four rules hold it honest.

**One scope per figure, printed with it.** The money answer is the projected
revenue of one named window on one named channel, taken from the saved plan,
and the window is the same slice ``summary.week`` reports. The per-day rows sum
to the headline and the payload says whether they reconciled, so the drill from
a figure to the rows behind it is checkable rather than asserted.

**A figure that cannot be scoped is withheld.** The saved plan carries every
channel in the market because the retention model is measured against the
lineup. Until the operator has declared which one is theirs, the only honest
answers on this surface are none of them: serving the four-channel total as the
operator's expected revenue would be both a fabrication and a competitor
breach, and it is the exact failure ``channel_scope`` returns its reason for.
So the money, the days, the drill and the decisions all become one absence with
one cause and one control that ends it.

**No target is invented.** Without a stored target the verdict is
``unavailable`` with the reason ``no_target``, and the surface offers the
control that supplies one. A plan compared against itself would always be on
plan, which is why the number has to come from a person.

**No training word reaches this payload.** A model change is reported as a
newer model version with the date it was trained, and never as a coefficient, a
gate or a p-value. That is section 4.2's lexicon test applied at the boundary
this module owns.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import pandas as pd

from kairos_api import channel_scope

# Freshness group labels that mean the model changed, not the operator. The
# engine emits its own internal vocabulary here; Today translates it once, at
# this boundary, into the one thing an operator can act on.
MODEL_SIDE_GROUPS = frozenset(
    {
        "coefficients",
        "the impact model",
        "impact_model",
        "the audience model",
        "audience_model",
    }
)

# The operator-facing name for every other changed group, in both languages.
CHANGED_GROUP_LABELS: dict[str, tuple[str, str]] = {
    "settings": ("settings", "הגדרות"),
    "constraints": ("restrictions", "הגבלות"),
    "overrides": ("pins", "נעיצות"),
    "data": ("source data", "נתוני מקור"),
    "inventory data": ("inventory", "מלאי"),
    "inventory": ("inventory", "מלאי"),
    "advertiser rules": ("advertiser rules", "כללי מפרסמים"),
    "advertiser": ("advertiser rules", "כללי מפרסמים"),
    "campaign flights": ("campaigns", "קמפיינים"),
    "campaigns": ("campaigns", "קמפיינים"),
    "program classifications": ("programme genres", "ז'אנרים של תוכניות"),
    "classifications": ("programme genres", "ז'אנרים של תוכניות"),
    "special events": ("calendar events", "אירועי לוח שנה"),
    "events": ("calendar events", "אירועי לוח שנה"),
}

# Lookup tables indexed by ``date.weekday()``, which counts from Monday. They
# are not a display order and nothing iterates them: the day rows come out in
# date order, so a window that is a calendar week already reads Sunday first and
# a window that is not is never re-sorted into a week it is not.
WEEKDAY_HE = ("יום שני", "יום שלישי", "יום רביעי", "יום חמישי", "יום שישי", "שבת", "יום ראשון")
WEEKDAY_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Israeli week: the weekend is Friday and Saturday only, which is index 4 and 5
# on the same Monday-based scale.
WEEKEND_WEEKDAYS = frozenset({4, 5})

# The one cause that withholds every figure on this surface, and the control
# that ends it. Named as data so the surface prints it in either language and
# opens the place it is set rather than describing it.
NO_CHANNEL = {
    "reason": "no_operator_channel",
    "reason_en": "The operator's own channel has not been declared, and the saved plan carries every channel in the market. Nothing here can be reported as yours until one of them is.",
    "reason_he": "לא הוגדר הערוץ של המפעיל, והתוכנית השמורה כוללת את כל הערוצים בשוק. אי אפשר לדווח כאן על שום מספר כשלכם עד שייבחר אחד מהם.",
    "needs_en": "Choose the operator's channel in settings.",
    "needs_he": "בחרו את ערוץ המפעיל בהגדרות.",
    "opens": "settings",
}


def owned_channel(settings: Any = None) -> str:
    """The one channel this surface is about, empty when none is declared.

    Every part of Today asks this one function, so the refusal below has a
    single cause and cannot be true in one panel and false in another.
    """
    return channel_scope.operator_channel(settings)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _money(value: Any) -> Optional[float]:
    number = _number(value)
    return None if number is None else round(number, 2)


def window_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """The window the money answer is about, taken from the plan's own slice.

    ``summary.week`` is the planning-week slice the engine computes: the
    Sunday-to-Saturday week around the reference date when that date falls
    inside the saved plan, otherwise the plan's first seven dates. The basis
    field says which rule fired, and it is printed rather than smoothed over,
    because a seven-day span that is not a calendar week must not be read as
    one.
    """
    week = summary.get("week") if isinstance(summary.get("week"), dict) else None
    if week and week.get("date_from") and week.get("date_to"):
        return {
            "available": True,
            "date_from": str(week["date_from"])[:10],
            "date_to": str(week["date_to"])[:10],
            "n_dates": int(week.get("n_dates") or 0),
            "basis": str(week.get("basis") or ""),
            "is_calendar_week": str(week.get("basis") or "") == "reference_date",
        }
    return {
        "available": False,
        "date_from": None,
        "date_to": None,
        "n_dates": 0,
        "basis": "",
        "is_calendar_week": False,
    }


def _weekday(iso: str) -> tuple[str, str, bool]:
    try:
        index = date.fromisoformat(iso).weekday()
    except ValueError:
        return "", "", False
    return WEEKDAY_EN[index], WEEKDAY_HE[index], index in WEEKEND_WEEKDAYS


def day_rows(schedule: pd.DataFrame, window: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The per-day rows behind the window figure, and the boundary disclosure.

    The competitor boundary is applied through the shared helper rather than by
    hand, so the payload carries the same disclosure every scoped surface does
    and a rival channel's day can never appear among the operator's.
    """
    scoped, note = channel_scope.scope_frame(schedule)
    if not note.get("scoped"):
        # Nothing was filtered, so what is in hand is the whole market. It is
        # not this operator's week and it is not served as one.
        return [], note
    if scoped is None or len(scoped) == 0 or not window["available"] or "date" not in getattr(scoped, "columns", []):
        return [], note
    dates = scoped["date"].astype(str).str.strip()
    inside = (dates >= window["date_from"]) & (dates <= window["date_to"])
    frame = scoped[inside]
    if len(frame) == 0:
        return [], note
    revenue = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    breaks = pd.to_numeric(frame.get("num_breaks", 0), errors="coerce").fillna(0)
    seconds = pd.to_numeric(frame.get("total_break_time", frame.get("break_length", 0)), errors="coerce").fillna(0)
    grouped = pd.DataFrame(
        {
            "date": dates[inside],
            "revenue": revenue,
            "breaks": breaks,
            "seconds": seconds,
        }
    ).groupby("date", sort=True).sum()
    rows: list[dict[str, Any]] = []
    for iso, row in grouped.iterrows():
        weekday_en, weekday_he, weekend = _weekday(str(iso))
        rows.append(
            {
                "date": str(iso),
                "weekday_en": weekday_en,
                "weekday_he": weekday_he,
                "is_weekend": weekend,
                "projected_revenue": _money(row["revenue"]),
                "total_breaks": int(row["breaks"]),
                "total_ad_seconds": int(row["seconds"]),
            }
        )
    return rows, note


def money_block(
    summary: dict[str, Any],
    window: dict[str, Any],
    rows: list[dict[str, Any]],
    note: dict[str, Any],
    timezone_name: str = "",
) -> dict[str, Any]:
    """The one money figure this surface carries, with its scope and its rows.

    The scope carries the four facts a financial figure needs before it can be
    read: which entity it is about, over which range and on which side of that
    range's edges, in which zone those edges fall, and in which currency. They
    are fields on the figure rather than a tooltip, because a basis a reader has
    to hover for is a basis most readers never see.
    """
    # Two independent reads decide this figure: the engine's own summary, which
    # scopes the total it returns and discloses the channel it scoped to, and
    # the boundary helper, which scopes the rows behind it. They must name the
    # same channel or the figure and its label are about different things, so
    # the figure is withheld rather than printed under a scope it does not have.
    summary_channel = str(summary.get("scope_channel") or "").strip()
    note_channel = str(note.get("scope_channel") or "").strip()
    scoped = bool(note.get("scoped")) and bool(summary_channel) and summary_channel == note_channel
    week = summary.get("week") if isinstance(summary.get("week"), dict) else None
    amount = _money((week or summary).get("projected_revenue")) if scoped else None
    rows_total = _money(sum(row["projected_revenue"] or 0 for row in rows)) if rows else None
    residual = None if amount is None or rows_total is None else round(amount - rows_total, 2)
    return {
        "metric": "projected_revenue",
        "currency": "ILS",
        "amount_ils": amount,
        "available": amount is not None,
        "unavailable": None if scoped else NO_CHANNEL,
        "scope": {
            "channel": summary_channel if scoped else None,
            "date_from": window["date_from"],
            "date_to": window["date_to"],
            "n_dates": window["n_dates"],
            "inclusive": True,
            "timezone": str(timezone_name or "").strip() or None,
            "currency": "ILS",
            "source": "saved_plan",
            "basis": window["basis"],
        },
        "boundary": note,
        "days": rows,
        "days_total_ils": rows_total,
        "residual_ils": residual,
        "reconciled": residual is not None and abs(residual) < 0.5,
        "total_breaks": None if week is None or not scoped else week.get("total_breaks"),
        "total_ad_seconds": None if week is None or not scoped else week.get("total_ad_seconds"),
        "average_retention": None if week is None or not scoped else week.get("average_retention"),
    }


def _changed_split(changed: list[str]) -> tuple[list[dict[str, str]], bool]:
    """Split the changed input groups into what a person changed and whether the
    model moved. The model side never reaches the payload by its engine name."""
    yours: list[dict[str, str]] = []
    model_changed = False
    for raw in changed:
        key = str(raw or "").strip()
        if not key:
            continue
        if key in MODEL_SIDE_GROUPS:
            model_changed = True
            continue
        label_en, label_he = CHANGED_GROUP_LABELS.get(key, (key, key))
        yours.append({"key": key, "label_en": label_en, "label_he": label_he})
    return yours, model_changed


def health_block(
    body: dict[str, Any],
    freshness: dict[str, Any],
    model_trained_at: Optional[str],
    channel: str = "",
) -> dict[str, Any]:
    """Is anything broken: every check computed from a real payload field."""
    checks: list[dict[str, Any]] = []
    if not str(channel or "").strip():
        # The first thing broken, when it is broken, is that nothing on this
        # screen can be attributed to anybody. It leads the list for that reason.
        checks.append({"id": "operator_channel_unset", "status": "attention", "opens": "settings", **NO_CHANNEL})
    status = str(freshness.get("status") or "unknown").lower()
    changed = freshness.get("changed") if isinstance(freshness.get("changed"), list) else []
    yours, model_changed = _changed_split(list(changed))

    if status == "stale" and yours:
        checks.append(
            {
                "id": "plan_out_of_date",
                "status": "attention",
                "opens": "plan",
                "changed": yours,
                "plan_run_at": freshness.get("computed_at"),
            }
        )
    if status == "stale" and model_changed:
        checks.append(
            {
                "id": "newer_model_version",
                "status": "notice",
                "opens": "plan",
                "model_trained_at": model_trained_at,
                "plan_run_at": freshness.get("computed_at"),
            }
        )
    if status == "fresh":
        checks.append({"id": "plan_current", "status": "ok", "opens": "plan", "plan_run_at": freshness.get("computed_at")})
    if status == "unknown":
        checks.append({"id": "plan_currency_unknown", "status": "unknown", "opens": "plan", "plan_run_at": freshness.get("computed_at")})

    compliance = body.get("compliance") if isinstance(body.get("compliance"), dict) else {}
    all_checks = compliance.get("checks") if isinstance(compliance.get("checks"), list) else []
    breached = [check for check in all_checks if str(check.get("status", "")).lower() not in {"compliant", "ok"}]
    # No checks is not a pass. Zero of zero rules satisfied means the licence was
    # never evaluated, and reporting that as compliant is the one direction this
    # row must never be wrong in.
    checks.append(
        {
            "id": "licence",
            "status": "attention" if breached else "ok" if all_checks else "unknown",
            "opens": "licence",
            "checks_total": len(all_checks),
            "checks_breached": len(breached),
            "breached_labels_en": [str(check.get("label_en", "")) for check in breached],
            "breached_labels_he": [str(check.get("label_he", "")) for check in breached],
            "profile": compliance.get("profile"),
            "effective_date": compliance.get("effective_date"),
        }
    )

    counts = body.get("source_counts") if isinstance(body.get("source_counts"), dict) else {}
    missing = [key for key in ("programmes", "spots", "planned_break_rows") if not int(counts.get(key) or 0)]
    checks.append(
        {
            "id": "inputs",
            "status": "attention" if missing else "ok",
            "opens": "sources",
            "programmes": int(counts.get("programmes") or 0),
            "spots": int(counts.get("spots") or 0),
            "planned_break_rows": int(counts.get("planned_break_rows") or 0),
            "missing": missing,
            "newest_input_at": body.get("data_freshness"),
        }
    )

    attention = [check for check in checks if check["status"] == "attention"]
    return {
        "state": "attention" if attention else "clear",
        "attention_count": len(attention),
        "checks": checks,
    }


def decisions_scope(body: dict[str, Any], channel: str = "") -> dict[str, Any]:
    """The span the ranking was drawn from, carried with the list that came out of it.

    The five rows are the highest earners in the whole saved plan, not in the
    money window above them, and on the reference data those are different
    spans: the window is the plan's first seven dates and the ranking scans all
    thirty. Rows dated outside the window named a few centimetres above them
    read as a contradiction unless the list says which span it came from, so it
    says it, out of the same summary the ranking scanned.

    The channel has to agree with the one the surface is about before any date
    is stated, for the same reason the money figure is withheld when it does
    not: a span taken off somebody else's rows is not this operator's.
    """
    summary = body.get("summary") if isinstance(body.get("summary"), dict) else {}
    scoped = str(summary.get("scope_channel") or "").strip()
    agreed = bool(scoped) and scoped == str(channel or "").strip()
    date_from = str(summary.get("date_from") or "")[:10]
    date_to = str(summary.get("date_to") or "")[:10]
    return {
        "channel": scoped if agreed else None,
        "date_from": date_from if agreed and date_from else None,
        "date_to": date_to if agreed and date_to else None,
        "n_dates": int(summary.get("n_dates") or 0) if agreed else 0,
        "inclusive": True,
        "source": "saved_plan",
        "grain": "whole_saved_plan",
    }


def decisions_block(body: dict[str, Any], channel: str = "") -> dict[str, Any]:
    """What needs a decision: the same list, the same figures, with its ranking
    stated. The candidate list each row carries is dropped from this payload
    because it is a resolver input, not something a person reads here.

    With no declared channel the list is withheld rather than served, for the
    same reason the money figure is: the plan's highest-earning segments are
    then somebody else's, and calling them the operator's priorities would put a
    competitor's schedule on an operator's home screen.
    """
    items = body.get("recommendations") if isinstance(body.get("recommendations"), list) else []
    scope = decisions_scope(body, channel)
    if not str(channel or "").strip():
        return {"count": 0, "ranked_by": "projected_revenue", "items": [], "scope": scope, "unavailable": NO_CHANNEL}
    trimmed = [{key: value for key, value in item.items() if key != "candidates"} for item in items]
    return {
        "count": len(trimmed),
        "ranked_by": "projected_revenue",
        "items": trimmed,
        "scope": scope,
        "unavailable": None,
    }
