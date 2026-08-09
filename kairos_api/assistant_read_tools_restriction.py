"""What a restriction would cost, before anybody writes it.

A restriction is a decision about somebody else's revenue, and
:mod:`kairos_api.constraints_restrictions` already puts its cost on screen
before the save rather than after. Kai could read the constraints that exist
(``list_constraints``) and propose a new one (``propose_constraint``) and had no
way at all to answer the question a person actually asks first: what does this
one cost. So it could offer a rule and could not price it.

This is a READ tool over the preview route, in the shape of
``simulate_settings_change``: it writes nothing, it is the product's own
arithmetic rather than a second one, and it exists so a what-if can be answered
without proposing anything.

**Two bases, named, never blended.** The preview reports the money twice and the
distinction is the whole point, so nothing here sums or averages them.
``scored`` is the exact revenue and retention of the breaks the restriction
removes, at the counts it sets, with the optimizer held still. ``exact`` is the
commit path's own optimizer run on the affected days, which is the plan the save
would actually produce, and it declines with a stated reason when the
restriction touches too many days to run. A caller that reads one must say which
one, which is why both keep their own names here.

**Collateral is kept separate too.** A compiled row can reach an airing the
sentence never named, and the preview counts and prices that half on its own.
Flattening it into one total is exactly what made the first round of the
restriction surface lie, so the tool passes it through.

The day-level change list is capped, with the true total beside the cap, because
a restriction over a season compiles to hundreds of rows and the model reasons
about the totals, not the rows.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_CHANGES = 25

# The kinds and their parameters are the frozen restriction language's, published
# here rather than restated, so a description cannot drift from what compiles.
KINDS_NOTE = "The kinds and the parameters each takes come from the restriction language; call this with a kind it does not hold and the refusal names the allowed ones."


def _read_estimate_restriction_cost(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.constraints_restrictions import RestrictionDraft, preview_restriction

    kind = str(args.get("kind", "") or "").strip()
    if not kind:
        return {"error": "kind is required: the restriction kind to price"}
    params = args.get("params")
    where = args.get("where")
    draft = RestrictionDraft(
        kind=kind,
        params=dict(params) if isinstance(params, dict) else {},
        where=dict(where) if isinstance(where, dict) else None,
        starts_on=str(args.get("starts_on", "") or ""),
        expires_on=str(args.get("expires_on", "") or ""),
    )
    body = preview_restriction(draft)
    return _summarise(body)


def _summarise(body: dict[str, Any]) -> dict[str, Any]:
    """The preview with its change list capped and its two bases left apart."""
    changes = list(body.get("changes") or [])
    out: dict[str, Any] = {
        "sentence_en": body.get("sentence") or body.get("sentence_en"),
        "sentence_he": body.get("sentence_he"),
        "kind": body.get("kind"),
        "params": body.get("params"),
        "channel": body.get("channel"),
        "starts_on": body.get("starts_on"),
        "expires_on": body.get("expires_on"),
        "matched_airings": body.get("matched_airings"),
        "bound_airings": body.get("bound_airings"),
        "bound_days": body.get("bound_days"),
        "asked_for_airings": body.get("asked_for_airings"),
        "unchanged_airings": body.get("unchanged_airings"),
        "airings_without_a_plan": body.get("airings_without_a_plan"),
        "compiled_rows": body.get("compiled_rows"),
        "changes": changes[:MAX_CHANGES],
        "changes_total": len(changes),
        # The two money bases, each under its own name and neither summed into
        # the other. The preview's own words for what each one is ride with them.
        "revenue_delta": -1.0,
        "scored": body.get("scored"),
        "exact": body.get("exact"),
        "collateral": body.get("collateral"),
        "starting_points": body.get("starting_points"),
        "already_in_force": body.get("already_in_force"),
        "engine_skipped": body.get("engine_skipped"),
        "wrote_nothing": True,
    }
    if len(changes) > MAX_CHANGES:
        out["changes_omitted"] = len(changes) - MAX_CHANGES
    return {key: value for key, value in out.items() if value is not None}


RESTRICTION_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "estimate_restriction_cost",
        "description": (
            "Price a restriction WITHOUT saving it: what sentence it reads as, how many "
            "airings it matches, binds and actually changes, and what it costs on two named "
            "bases that are never blended. 'scored' is the exact revenue and retention of the "
            "breaks it removes at the counts it sets, with the optimizer held still, and it "
            "covers every affected day. 'exact' is the commit path's own optimizer run on the "
            "affected days, which is the plan a save would really produce; it states its own "
            "reason when the restriction touches too many days to run rather than guessing. "
            "'collateral' is the part a compiled row reaches that the sentence never named, "
            "counted and priced on its own. Call this whenever the operator asks what a "
            "restriction, a ban or a placement rule would cost, or before proposing one with "
            "propose_constraint, and report which basis each figure is on. Nothing is written."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "The restriction kind. A kind the language does not hold is refused with the allowed ones named."},
                "params": {"type": "object", "description": "The kind's own parameters, for example a break count or a window in seconds."},
                "where": {"type": "object", "description": "The scope predicate in the frozen constraint contract shape (combinator plus conditions), or omitted for the whole channel."},
                "starts_on": {"type": "string", "description": "ISO date the restriction would start, or omitted for immediately."},
                "expires_on": {"type": "string", "description": "ISO date the restriction would stop, or omitted for open-ended."},
            },
            "required": ["kind"],
        },
    },
]

_RESTRICTION_READ_EXECUTORS = {"estimate_restriction_cost": _read_estimate_restriction_cost}

# A provenance line answers WHERE THIS CAME FROM and nothing else. It named the
# two internal seams and promised no side effect, and neither belongs on a chip
# an operator reads under every step of every run: the seam names are
# implementation detail nobody buying airtime has a use for, and "nothing
# written" is a fact about this tool's behaviour, which is why it rides the
# payload as ``wrote_nothing`` instead.
RESTRICTION_SOURCE_BY_TOOL = {
    "estimate_restriction_cost": "restriction preview on the owned channel: the saved weekly plan, and an optimizer run on the days it touches",
}


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge this executor and its source label into the shared registry."""
    executors.update(_RESTRICTION_READ_EXECUTORS)
    sources.update(RESTRICTION_SOURCE_BY_TOOL)
