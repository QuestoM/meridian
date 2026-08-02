"""What a restriction costs, on two named bases that are never blended.

A restriction is a decision about somebody else's revenue, so the cost belongs
on screen before the save. There are two honest ways to produce it and they
answer different questions, so both are computed and both are labelled.

**Scored.** The revenue and the retention of exactly the breaks the restriction
removes, at the counts it sets, through the frozen seam
:mod:`kairos.optimize.evaluate`. It is the same arithmetic the optimizer scores
its own plan with, measured at microseconds, so it covers every affected day at
no cost. What it does not do is let the optimizer put a removed break somewhere
else, and the payload says so rather than implying the plan would simply lose
this money.

**Re-allocated.** The commit path's own ``_optimize_one_day`` run twice on the
affected days, once with the stored constraints and once with the draft added.
That is the plan the save would produce, guardrails, daily cap and all. It costs
about a second a day, so it runs inside a day budget and reports the budget when
it stops rather than pricing a part as the whole.

Both bases price the change list :func:`kairos_api.constraints_airings.resolved_changes`
produces, which is the engine's own resolver applied to the airings the rule
matches. The first round of this module derived the change list from the
compiled rows instead, so the four kinds that compile to one scope-level row
priced nothing at all: a rule removing 117 breaks across 19 broadcast days
reported no figure on either basis and each empty panel referred the reader to
the other. A basis that cannot be computed now states what it would have priced
and never cites a figure that is not on the screen beside it.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from kairos_api.constraints_language import CompiledRow

# One optimizer leg measured at about 0.9 s on a real channel-day, so two days
# is four legs and still lands inside the three second preview bar. Above it the
# scored basis is the answer and the payload names the days it did not run.
EXACT_DAY_BUDGET = 2

CURRENCY = "ILS"

# The two bases do not start from the same number, and that is the single thing
# a reader has to be told rather than left to notice. Measured on
# ``רשת 13 / 2024-11-01``: the saved plan carries 1,067,845.55 and an optimizer
# run today starts from 1,062,669.88, because the saved plan was run before the
# inputs it was built on last moved. Both are real and neither is the other, so
# every figure states which starting point it is counted from.
SCORED_START_EN = "counted from the plan as saved"
SCORED_START_HE = "נספר מהתוכנית כפי שנשמרה"
EXACT_START_EN = "counted from a run today without this rule"
EXACT_START_HE = "נספר מהרצה היום בלי הכלל הזה"

GAP_NOTE_EN = "The two start from different numbers because one is the plan as saved and the other is a run today."
GAP_NOTE_HE = "השניים מתחילים ממספרים שונים משום שהאחד הוא התוכנית כפי שנשמרה והשני הוא הרצה היום."


def _unavailable(reason_en: str, reason_he: str, **extra: Any) -> dict[str, Any]:
    """An empty state that names its own scope in both languages.

    A basis that cannot be computed says what it would have priced and why it did
    not. It never refers a reader to the other basis unless that basis is known to
    carry a figure, because two empty panels pointing at each other is a screen
    with no number and no reason on it.
    """
    return {"available": False, "reason_en": reason_en, "reason_he": reason_he, **extra}


def placement_constraints(rows: Sequence[CompiledRow], prefix: str) -> list[Any]:
    """The compiled rows as engine objects, held in memory and never written."""
    from kairos.optimize.constraints_store import PlacementConstraint

    return [
        PlacementConstraint(
            constraint_id=f"{prefix}-{index}",
            scope_type="always",
            effect=row.effect,
            count=row.count,
            offset_seconds=row.offset_seconds,
            where=row.where,
        )
        for index, row in enumerate(rows)
    ]


def _plan_counts() -> dict[str, Optional[int]]:
    from kairos_api.constraints_airings import all_airings

    airings, _segments = all_airings()
    return {airing.segment_id: airing.planned_breaks for airing in airings}


def _nothing_to_price(bound: int, matched: int) -> dict[str, Any]:
    """The honest empty state for a rule that moves no break count.

    Two different facts, and they are not interchangeable. A rule that binds
    airings but leaves every count where it was moves a break rather than
    removing one, and this basis prices counts. A rule the engine's resolver
    binds to nothing at all has no effect on the plan yet, which is a thing its
    author has to be told before they save it, not after.
    """
    if bound:
        return _unavailable(
            f"This rule binds {bound} airings and leaves the break count on every one of them where it was, so this basis has no count change to price. What it moves is where a break sits.",
            f"הכלל הזה מחייב {bound} שידורים ומשאיר את מספר הברייקים בכולם כפי שהיה, ולכן אין לבסיס הזה שינוי כמותי לתמחר. מה שהוא מזיז הוא מיקום הברייק.",
            bound_airings=bound,
        )
    return _unavailable(
        f"The plan engine resolves this rule to no break count on any of the {matched} airings it matches, so it moves nothing in the plan as it stands and there is nothing to price.",
        f"מנוע התוכנית אינו גוזר מהכלל הזה מספר ברייקים באף אחד מ-{matched} השידורים שהוא תואם, ולכן הוא אינו מזיז דבר בתוכנית הנוכחית ואין מה לתמחר.",
        bound_airings=0,
        matched_airings=matched,
    )


def scored(changes: Sequence[dict[str, Any]], bound: int = 0, matched: int = 0) -> dict[str, Any]:
    """The revenue and retention of the breaks this restriction takes out.

    Built per channel-day on the day's whole segment set, so the basis is a
    declared scope rather than a fragment, with the counts the plan of record
    holds against the counts this restriction sets. ``changes`` is the engine
    resolver's own answer for each airing, so this prices what the optimizer would
    enforce and not a second reading of the sentence.
    """
    from kairos.optimize.evaluate import evaluation_basis, score
    from kairos_api.preview_inputs import preview_inputs

    if not changes:
        return _nothing_to_price(bound, matched)
    plan_counts = _plan_counts()
    after_counts = {row["segment_id"]: int(row["after_breaks"]) for row in changes}
    days = sorted({(row["channel"], row["day"]) for row in changes})
    revenue_before = revenue_after = 0.0
    retention_before = retention_after = 0.0
    breaks_before = breaks_after = 0
    priced: list[tuple[str, str]] = []
    unknown: list[str] = []
    for channel, day in days:
        segments, kwargs = preview_inputs(channel, day, None)
        if not segments:
            continue
        counts_before = {s.segment_id: plan_counts.get(s.segment_id) for s in segments}
        if any(count is None for count in counts_before.values()):
            unknown.append(day)
            continue
        basis = evaluation_basis(segments, risk_lambda=kwargs["risk_lambda"])
        counts_after = dict(counts_before)
        for segment_id, count in after_counts.items():
            if segment_id in counts_after:
                counts_after[segment_id] = count
        before = score(basis, counts_before, revenue_weight=kwargs["revenue_weight"])
        after = score(basis, counts_after, revenue_weight=kwargs["revenue_weight"])
        revenue_before += before.revenue
        revenue_after += after.revenue
        retention_before += before.retention
        retention_after += after.retention
        breaks_before += sum(counts_before.values())
        breaks_after += sum(counts_after.values())
        priced.append((channel, day))
    if not priced:
        return _unavailable(
            f"The plan of record carries no break count for {len(unknown)} of the broadcast days this rule touches, so there is nothing here to count from. Run the weekly plan to fill them.",
            f"בתוכנית הרשומה אין מספר ברייקים ל-{len(unknown)} מימי השידור שהכלל נוגע בהם, ולכן אין ממה לספור. הריצו את התוכנית השבועית כדי למלא אותם.",
            days_without_a_plan=len(unknown),
        )
    day_count = max(len(priced), 1)
    return {
        "available": True,
        "basis": "scored",
        "starting_point": "saved_plan",
        "starting_point_en": SCORED_START_EN,
        "starting_point_he": SCORED_START_HE,
        "revenue_before": round(revenue_before, 2),
        "revenue_after": round(revenue_after, 2),
        "revenue_delta": round(revenue_after - revenue_before, 2),
        "retention_before": round(retention_before / day_count, 6),
        "retention_after": round(retention_after / day_count, 6),
        "breaks_before": breaks_before,
        "breaks_after": breaks_after,
        "days": len(priced),
        "days_without_a_plan": len(unknown),
        "airings_changed": len(changes),
        "currency": CURRENCY,
    }


def collateral(
    changes: Sequence[dict[str, Any]],
    derived_ids: Sequence[str],
    bound_ids: Sequence[str],
    *,
    applies: bool,
) -> dict[str, Any]:
    """The part of the change list the sentence never asked for, priced on its own.

    A window rule is derived from the airings that breach it, and any other airing
    a compiled row reaches is surplus: the compiler judged it already compliant
    and the rule moves its break count anyway. That surplus used to be invisible
    and it was most of the money. Measured on ``משחקי השף עונה 7 ש.ח`` before the
    hour was pinned into each row's predicate: of 470,562.01 ILS on screen,
    404,538.45 came off 10 airings nobody had asked about.

    Priced by scoring exactly those airings and nothing else, which is exact
    rather than apportioned, because :func:`kairos.optimize.evaluate.score` sums
    revenue per segment so the segments both arrangements share cancel. The pass
    runs only when there is surplus, so an honest rule pays nothing for it.

    ``bound`` and ``changed`` are two facts. An airing held without its count
    moving is surplus that costs nothing; an airing whose count moves is surplus
    that costs money. ``applies`` is false for the four scope-level kinds, where
    every airing bound is an airing the sentence names, so the distinction does
    not exist and a nought there would invent one.
    """
    if not applies:
        return {"applies": False, "bound": 0, "changed": 0, "breaks_removed": 0}
    known = set(derived_ids)
    extra = [row for row in changes if str(row.get("segment_id")) not in known]
    body: dict[str, Any] = {
        "applies": True,
        "bound": len({str(item) for item in bound_ids} - known),
        "changed": len(extra),
        "breaks_removed": sum(
            int(row["before_breaks"]) - int(row["after_breaks"]) for row in extra
        ),
        "days": sorted({str(row["day"]) for row in extra}),
    }
    if extra:
        body["revenue"] = scored(extra, bound=len(extra), matched=len(extra))
    return body


def already_in_force(rows: Sequence[CompiledRow]) -> dict[str, Any]:
    """Whether the store already holds every row this draft would write.

    A restriction that is already in force costs nothing more, and the
    re-allocated basis correctly reports zero for it. Zero with no reason reads
    as a broken preview, so the duplicate is detected exactly, by comparing the
    compiled rows against the stored ones, rather than inferred from the zero.
    """
    from kairos.optimize.constraints_store import load_constraints
    from kairos_api.constraints import CONSTRAINTS_PATH

    if not rows:
        return {"all": False, "matched": 0, "of": 0}
    stored = load_constraints(CONSTRAINTS_PATH)
    held = {
        (str(row.effect), row.count, row.offset_seconds, json_key(row.where))
        for row in stored
    }
    matched = sum(
        1 for row in rows
        if (str(row.effect), row.count, row.offset_seconds, json_key(row.where)) in held
    )
    return {"all": matched == len(rows), "matched": matched, "of": len(rows)}


def json_key(where: Any) -> str:
    """A predicate as a stable string, so two identical trees compare equal."""
    import json

    try:
        return json.dumps(where, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(where)


def starting_points(scored_side: dict[str, Any], exact_side: dict[str, Any]) -> dict[str, Any]:
    """Whether the two bases start from the same number, and by how much they do not.

    Two money figures on one screen whose starting points differ silently is the
    exact defect the money rule exists to stop. Where both bases are available
    the gap is reported as a number with the reason for it, so the difference is
    a stated fact rather than a contradiction a reader has to resolve alone.
    """
    if not (scored_side.get("available") and exact_side.get("available")):
        return {"comparable": False}
    before_scored = float(scored_side.get("revenue_before") or 0.0)
    before_exact = float(exact_side.get("revenue_before") or 0.0)
    gap = round(before_exact - before_scored, 2)
    return {
        "comparable": True,
        "same_start": gap == 0.0,
        "gap": gap,
        "note_en": "" if gap == 0.0 else GAP_NOTE_EN,
        "note_he": "" if gap == 0.0 else GAP_NOTE_HE,
        "currency": CURRENCY,
    }


def _said_plainly(raw: str) -> tuple[str, str]:
    """One engine refusal, in the words a programming person uses.

    The engine's reasons are two closed sets, the resolver's
    (``kairos/optimize/constraints_store.py``) and the optimizer's
    (``kairos/optimize/_override_logic.py``). They are translated rather than
    printed, because an internal token on an operator surface is an engine word
    and JS-4's target is none of those anywhere on the path. An unrecognised
    reason falls back to the class of refusal, never to the raw string.
    """
    capacity = re.search(r"exceeds max_breaks (\d+)", raw)
    if capacity:
        limit = capacity.group(1)
        return (
            f"the airing can hold {limit} breaks and the rule asks for more",
            f"השידור יכול לשאת {limit} ברייקים והכלל מבקש יותר",
        )
    if "guardrail" in raw:
        return (
            "the breaks it pins breach the spacing or load limit for that broadcast day",
            "הברייקים שהוא נועץ חורגים ממגבלת המרווח או העומס של יום השידור",
        )
    if "conflicting" in raw or "already" in raw:
        return (
            "another rule already sets this airing, and the two disagree",
            "כלל אחר כבר קובע את השידור הזה, והשניים אינם מסכימים",
        )
    if "needs an offset" in raw or "soft hint" in raw:
        return (
            "this rule has no minute to place the break at, so the plan cannot pin one",
            "לכלל הזה אין דקה למקם בה את הברייק, ולכן התוכנית אינה יכולה לנעוץ אותו",
        )
    if "gold" in raw:
        return (
            "a gold mark gilds a break that is already pinned, and this airing has none",
            "סימון זהב מסמן ברייק שכבר נעוץ, ולשידור הזה אין כזה",
        )
    return (
        "the plan engine cannot place the breaks this rule asks for",
        "מנוע התוכנית אינו יכול למקם את הברייקים שהכלל מבקש",
    )


def refusals(rejected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every engine refusal for one draft, in both languages.

    A pinned count the plan cannot carry is the one outcome a composer must not
    hide: the count basis prices the arrangement the rule asks for, the optimizer
    run shows the plan will not carry it, and without this the two figures simply
    disagree with nothing on screen to explain them.
    """
    out: list[dict[str, Any]] = []
    for item in rejected:
        reason_en, reason_he = _said_plainly(str(item.get("reason") or ""))
        out.append({
            "segment_id": str(item.get("segment_id") or ""),
            "reason_en": reason_en,
            "reason_he": reason_he,
        })
    return out


def affected_days(rows: Sequence[CompiledRow]) -> list[tuple[str, str]]:
    """The channel-days a restriction touches, from its own compiled rows."""
    from kairos_api import constraints_airings as airings_lib

    per_airing = sorted({
        (row.airing.channel, row.airing.day) for row in rows if row.airing is not None
    })
    if per_airing:
        return per_airing
    if not rows:
        return []
    matched = airings_lib.matching(rows[0].where)
    return sorted({(airing.channel, airing.day) for airing in matched})


def reallocated(
    rows: Sequence[CompiledRow],
    days: Optional[Sequence[tuple[str, str]]] = None,
    scored_days: int = 0,
) -> dict[str, Any]:
    """The commit path's own answer: the optimizer with and without the draft.

    ``scored_days`` is how many broadcast days the scored basis actually priced.
    It is passed in rather than assumed, because the sentence this basis prints
    when it stops must only send a reader to a figure that exists.
    """
    from kairos.optimize.constraints_store import load_constraints
    from kairos.optimize.day_core import _optimize_one_day
    from kairos_api.constraints import CONSTRAINTS_PATH
    from kairos_api.overrides import _resolved_store_overrides
    from kairos_api.preview_inputs import preview_inputs

    scope = list(days) if days is not None else affected_days(rows)
    if not scope:
        return _unavailable(
            "No broadcast day in the plan window matches this restriction, so there is no day to run the optimizer on.",
            "אף יום שידור בחלון התוכנית אינו תואם את ההגבלה הזאת, ולכן אין יום להריץ עליו את האופטימייזר.",
            days=0,
        )
    if len(scope) > EXACT_DAY_BUDGET:
        covered_en = f" The scored figure prices all {scored_days} of them." if scored_days >= len(scope) else ""
        covered_he = f" המספר המחושב מתמחר את כל {scored_days} הימים." if scored_days >= len(scope) else ""
        return _unavailable(
            f"An optimizer run costs about a second a broadcast day, so this basis prices at most {EXACT_DAY_BUDGET} at a time and this rule touches {len(scope)}. Narrow it to one night to price it this way.{covered_en}",
            f"הרצת אופטימייזר עולה כשנייה ליום שידור, ולכן הבסיס הזה מתמחר עד {EXACT_DAY_BUDGET} ימים בכל פעם וההגבלה הזאת נוגעת ב-{len(scope)}. צמצמו ללילה אחד כדי לתמחר כך.{covered_he}",
            days=len(scope),
            budget=EXACT_DAY_BUDGET,
        )
    stored = load_constraints(CONSTRAINTS_PATH)
    draft = placement_constraints(rows, "draft")
    revenue_before = revenue_after = 0.0
    retention_before = retention_after = 0.0
    breaks_before = breaks_after = 0
    rejected: list[dict[str, Any]] = []
    run_days = 0
    for channel, day in scope:
        segments, kwargs = preview_inputs(channel, day, None)
        if not segments:
            continue
        active, _stale = _resolved_store_overrides(segments)
        overrides = active if active.overrides else None
        baseline = _optimize_one_day(segments, constraints=stored, overrides=overrides, **kwargs)
        after = _optimize_one_day(
            segments, constraints=[*stored, *draft], overrides=overrides, **kwargs,
        )
        run_days += 1
        revenue_before += baseline.total_revenue
        revenue_after += after.total_revenue
        retention_before += baseline.aggregate_retention
        retention_after += after.aggregate_retention
        breaks_before += baseline.total_breaks
        breaks_after += after.total_breaks
        rejected.extend(
            {"segment_id": item.segment_id, "kind": item.kind, "reason": item.reason}
            for item in after.rejected_overrides
        )
    divisor = max(run_days, 1)
    return {
        "available": run_days > 0,
        "basis": "reallocated",
        "starting_point": "optimizer_today",
        "starting_point_en": EXACT_START_EN,
        "starting_point_he": EXACT_START_HE,
        "revenue_before": round(revenue_before, 2),
        "revenue_after": round(revenue_after, 2),
        "revenue_delta": round(revenue_after - revenue_before, 2),
        "retention_before": round(retention_before / divisor, 6),
        "retention_after": round(retention_after / divisor, 6),
        "breaks_before": breaks_before,
        "breaks_after": breaks_after,
        "days": run_days,
        "day_list": [{"channel": channel, "day": day} for channel, day in scope],
        "rejected_overrides": rejected,
        "refusals": refusals(rejected),
        "currency": CURRENCY,
    }
