"""The restriction language: what a programming representative says, compiled.

A programming representative does not think in offsets and effects. They think
"keep the last eight minutes of the finale clean". This module is the
translation layer between that sentence and the frozen placement-constraint
contract, and it is deliberately small: six kinds, each of which compiles to
effects the engine already honours, and none of which invents a primitive the
optimizer does not have.

The one that needed real work is the clean window. Offsets in the store run
forward from a programme start, so "the last eight minutes" has no direct
representation. It does have an exact one, because the engine places breaks
deterministically when nothing pins them: ``_segment_break_objects`` in
``kairos.optimize._segment_math`` spaces k breaks at ``duration / (k + 1)``, so
the last break of a k-break programme ends at ``duration * k / (k + 1) + length
/ 2``. Requiring that to land before the protected window gives a largest
honest break count, :func:`max_breaks_before_tail`, derived from the engine's
own placement rule rather than asserted. A programme already inside that count
is untouched; one above it is capped, so a restriction can only ever remove
breaks and never add one.

Two limits are stated here rather than discovered later.

- A clean window compiles per airing, because two airings of the same programme
  can differ in length and therefore in the count that keeps the window clean.
  The airings come from the plan of record, so a restriction covers the plan's
  own window and says so. The predicate a row carries names the programme, the
  date and the clock hour the airing starts in, which is the finest scope the
  frozen contract has. The first round pinned only the programme and the date,
  and that was five times wider than the sentence: measured on
  ``משחקי השף עונה 7 ש.ח`` with an eight minute tail, 43 airings match, exactly 7
  breach, and those 7 rows bound 17 airings and took 38 breaks, 31 of them off
  airings the compiler itself had judged compliant. Priced through
  :mod:`kairos_api.constraints_cost`, 404,538.45 of the 470,562.01 ILS on screen,
  86.0 percent of it, was revenue the rule destroyed without being asked to. With
  the hour pinned the same rule binds 7, moves 7 and takes 7 breaks, for
  66,023.56 ILS. Two airings of one programme can still share a clock hour, so
  the preview never infers the answer: it asks the engine's own resolver, lists
  each bound airing with its own before and after, and names in words any airing
  it binds that the sentence did not ask for.
- The engine has no soft window. A capped count is the exact enforcement of the
  sentence under the placement rule above, not an approximation of it, but if a
  later engine gains a placement window this compiler is the only thing that
  has to change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

# The authored columns. They are appended to the frozen store columns and the
# engine loader reads by name, so a column it does not know is a column it
# ignores: the restriction identity, the sentence and its attribution ride
# beside the compiled row without changing what the optimizer sees.
AUTHORING_COLUMNS = (
    "restriction_id",
    "rule_kind",
    "rule_params_json",
    "rule_where_json",
    "author",
    "reason",
    "starts_on",
    "expires_on",
    "created_at",
)

# The kinds, in the order the composer offers them. Each names the effects it
# compiles to so a reader can check the claim without reading the code.
CLEAN_TAIL = "clean_tail"
CLEAN_OPEN = "clean_open"
NO_BREAKS = "no_breaks"
EXACT_BREAKS = "exact_breaks"
FIXED_SLOT = "fixed_slot"
GOLD = "gold"

KINDS: tuple[str, ...] = (CLEAN_TAIL, CLEAN_OPEN, NO_BREAKS, EXACT_BREAKS, FIXED_SLOT, GOLD)

# Kinds that need the plan of record to compile, because the count that keeps a
# window clean depends on how long each airing actually is.
PER_AIRING_KINDS = frozenset({CLEAN_TAIL, CLEAN_OPEN})

MAX_COMPILED_ROWS = 60


class RestrictionError(ValueError):
    """A restriction that cannot be stated exactly, with the reason."""


@dataclass(frozen=True)
class Airing:
    """One occurrence of a programme, as the plan of record holds it.

    ``planned_breaks`` is ``None`` when the plan of record does not carry this
    airing at all, which is an unknown count and never a zero. A window
    restriction cannot judge an airing whose count is unknown, so it skips it and
    the preview reports how many it skipped.
    """

    segment_id: str
    channel: str
    day: str
    title: str
    start_seconds: float
    duration_seconds: float
    break_length_seconds: float
    planned_breaks: Optional[int]


@dataclass(frozen=True)
class CompiledRow:
    """One store row a restriction compiles to, with the airing it came from."""

    effect: str
    count: Optional[int]
    offset_seconds: Optional[float]
    where: dict[str, Any]
    airing: Optional[Airing]
    before_breaks: Optional[int]
    after_breaks: Optional[int]


def max_breaks_before_tail(duration_seconds: float, break_length_seconds: float,
                           protected_seconds: float) -> int:
    """The largest break count whose last break still ends before the window.

    The engine spaces k unpinned breaks at ``duration / (k + 1)``, so break j
    starts at ``duration * j / (k + 1) - length / 2`` and the last one ends at
    ``duration * k / (k + 1) + length / 2``. Requiring that to be at or before
    ``duration - protected`` and solving for k gives the answer below. It is the
    placement rule read backwards, so it is exact for as long as that rule is.
    """
    duration = float(duration_seconds or 0.0)
    length = float(break_length_seconds or 0.0)
    protected = float(protected_seconds or 0.0)
    if duration <= 0:
        return 0
    share = (duration - protected - length / 2.0) / duration
    if share <= 0:
        return 0
    if share >= 1:
        return MAX_COMPILED_ROWS * 100
    return int(math.floor(share / (1.0 - share) + 1e-9))


def max_breaks_after_open(duration_seconds: float, break_length_seconds: float,
                          protected_seconds: float) -> int:
    """The largest break count whose first break starts after the opening window.

    Mirror of :func:`max_breaks_before_tail`. The first break starts at
    ``duration / (k + 1) - length / 2``, which has to be at or after
    ``protected``, and that bound loosens as k falls, so the answer is the
    largest k satisfying ``duration / (k + 1) >= protected + length / 2``.
    """
    duration = float(duration_seconds or 0.0)
    length = float(break_length_seconds or 0.0)
    edge = float(protected_seconds or 0.0) + length / 2.0
    if duration <= 0 or edge <= 0:
        return MAX_COMPILED_ROWS * 100 if duration > 0 else 0
    if edge > duration:
        return 0
    return max(0, int(math.floor(duration / edge - 1.0 + 1e-9)))


def _condition(field: str, operator: str, value: Any) -> dict[str, Any]:
    return {"field": field, "operator": operator, "value": value}


def _group(conditions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"combinator": "and", "conditions": list(conditions)}


def dated_predicate(where: Optional[dict[str, Any]], starts_on: str, expires_on: str) -> Optional[dict[str, Any]]:
    """The author's predicate with its own dates folded in as engine conditions.

    An expiry that only lives in a column is a promise; an expiry that lives in
    the predicate is enforced by the same matcher the optimizer runs, so an
    expired restriction stops binding whether or not anything cleans it up.
    """
    conditions: list[dict[str, Any]] = []
    if where:
        conditions.append(where)
    if starts_on:
        conditions.append(_condition("date", "after", starts_on))
    if expires_on:
        conditions.append(_condition("date", "before", expires_on))
    if not conditions:
        return None
    if len(conditions) == 1 and "combinator" in conditions[0]:
        return conditions[0]
    return _group(conditions)


def airing_hour(airing: Airing) -> int:
    """The clock hour the engine reads for this airing, by its own arithmetic.

    ``kairos.optimize.predicate`` derives the ``hour`` field as
    ``int(start_seconds // 3600) % 24`` and the airing carries the very
    ``start_seconds`` of the segment the resolver evaluates, so this is the same
    number rather than a second opinion about it.
    """
    return int(airing.start_seconds // 3600) % 24


def airing_predicate(where: Optional[dict[str, Any]], airing: Airing) -> dict[str, Any]:
    """The predicate that names exactly one airing, under the author's own scope.

    Three pinned conditions, because two are not enough to name one night. A
    programme and a date select every airing of that programme that day, and a
    window rule derived from one of them then bound all of them, including the
    ones already keeping the window clean. The hour is the third and last
    coordinate the frozen contract offers, and it closes the gap on real data:
    across the operator's 2,540 airings, programme and date leave 1,880 airings
    sharing a key and programme, date and hour leave 1,400, and on every window
    rule measured it binds exactly the airings the sentence asks for.

    It is a narrowing, never a widening, so no airing outside the author's own
    scope can be reached by it. Where it is still not enough, because two airings
    of one programme really do start inside one clock hour, the preview reports
    the surplus in words rather than letting it pass as intended.

    The author's own conditions are kept, minus any the airing's own three state
    exactly. Repeating "programme is X" beside itself changes nothing the engine
    matches and makes the rule read as a stutter.
    """
    pinned = (
        _condition("programme", "is", airing.title),
        _condition("date", "is", airing.day),
        _condition("hour", "eq", airing_hour(airing)),
    )
    conditions = [node for node in _flatten(where) if node not in pinned]
    conditions.extend(pinned)
    return _group(conditions)


def _flatten(where: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """An AND tree's own conditions, or the whole node when it is not one."""
    if not where:
        return []
    if "combinator" in where and where.get("combinator") == "and":
        return list(where.get("conditions") or [])
    return [where]


def _in_window(airing: Airing, starts_on: str, expires_on: str) -> bool:
    if starts_on and airing.day < starts_on:
        return False
    if expires_on and airing.day >= expires_on:
        return False
    return True


def compile_restriction(
    kind: str,
    params: dict[str, Any],
    where: Optional[dict[str, Any]],
    airings: Sequence[Airing],
    *,
    starts_on: str = "",
    expires_on: str = "",
) -> list[CompiledRow]:
    """Turn one authored restriction into the store rows that enforce it.

    ``airings`` is the matched set from the plan of record and is only read by
    the two window kinds; the other four are scope-level and compile to a single
    row whatever the plan holds.
    """
    if kind not in KINDS:
        raise RestrictionError(f"'{kind}' is not a restriction kind.")
    scoped = dated_predicate(where, starts_on, expires_on)

    if kind == NO_BREAKS:
        return [CompiledRow("forbid", None, None, scoped or _match_all(), None, None, None)]
    if kind == EXACT_BREAKS:
        count = _int_param(params, "count", low=0, high=20)
        return [CompiledRow("pin_count", count, None, scoped or _match_all(), None, None, None)]
    if kind == FIXED_SLOT:
        offset = _int_param(params, "offset_seconds", low=0, high=24 * 3600)
        return [CompiledRow("fix_offset", None, float(offset), scoped or _match_all(), None, None, None)]
    if kind == GOLD:
        return [CompiledRow("gold", None, None, scoped or _match_all(), None, None, None)]

    minutes = _int_param(params, "protected_minutes", low=1, high=120)
    protected = float(minutes) * 60.0
    ceiling = max_breaks_before_tail if kind == CLEAN_TAIL else max_breaks_after_open
    rows: list[CompiledRow] = []
    for airing in airings:
        if not _in_window(airing, starts_on, expires_on):
            continue
        if airing.planned_breaks is None:
            continue
        allowed = ceiling(airing.duration_seconds, airing.break_length_seconds, protected)
        if airing.planned_breaks <= allowed:
            continue
        effect = "forbid" if allowed <= 0 else "pin_count"
        rows.append(CompiledRow(
            effect=effect,
            count=None if effect == "forbid" else allowed,
            offset_seconds=None,
            where=airing_predicate(where, airing),
            airing=airing,
            before_breaks=airing.planned_breaks,
            after_breaks=0 if effect == "forbid" else allowed,
        ))
    if len(rows) > MAX_COMPILED_ROWS:
        raise RestrictionError(
            f"This restriction touches {len(rows)} airings, more than the {MAX_COMPILED_ROWS} the store can name exactly. Narrow it to one programme or a shorter date range."
        )
    return rows


def _match_all() -> dict[str, Any]:
    """The predicate a scope-free restriction gets: every segment on the channel.

    The engine applies the operator channel itself, so this is the operator's
    own inventory and never a competitor's.
    """
    return _group([_condition("hour", "gte", 0)])


def _int_param(params: dict[str, Any], name: str, *, low: int, high: int) -> int:
    raw = (params or {}).get(name)
    try:
        value = int(float(raw))
    except (TypeError, ValueError) as exc:
        raise RestrictionError(f"{name} must be a whole number.") from exc
    if not low <= value <= high:
        raise RestrictionError(f"{name} must be between {low} and {high}.")
    return value


def params_cell(params: dict[str, Any]) -> str:
    return json.dumps(params or {}, ensure_ascii=False, separators=(",", ":"))


def parse_params(raw: Any) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
