"""Exact interval-sweep dynamic program as the optimizer's top refiner tier.

The greedy allocator and the F1 local search both leave value on the table on a
real channel-day: re-spacing at ``duration / (k + 1)`` makes feasibility
non-monotone, so a coordinated better-and-feasible move can be unreachable one
break at a time. This module solves a single channel-day's break-count allocation
to the EXACT optimum of the engine's own per-group objective (blend contribution
or revenue net) under the engine's own guardrails, then hands the answer back only
when a set of runtime preconditions hold. Outside those preconditions it silently
keeps the greedy+F1 counts it was given, so it never answers a case it does not
cover and never regresses a group.

Why a dynamic program and not a chain: real production segments overlap in time,
so the daily guardrails (spacing, per-hour count and seconds, daily ad seconds,
gold count) couple non-adjacent segments. The sweep keeps a joint state over every
OPEN earlier segment (one whose breaks can still interact with a later segment),
plus two small integer budgets (total breaks for the daily ad-seconds cap, gold
count) and prunes by dominance. The global budgets are carried as explicit state
dimensions, not relaxed into a Lagrangian, so there is no duality gap. Every
geometry, retention, revenue and compliance value comes from the engine's own
primitives (:mod:`kairos.optimize._segment_math`), so the DP truth equals engine
truth by construction and its plan compares to the cent.

Because those two budgets are scoped to the whole channel-day, the open-window
boundary in :func:`~kairos.optimize.dp_refine_prep._window_ends` closes only the
three LOCAL guardrails. It is not an independence cut: two segments hours apart
still compete for the same daily ad-seconds and gold allowances, so a channel-day
is never split into separately-solved sub-groups.

Interface contract with :func:`kairos.optimize.optimizer.optimize_breaks`: this
tier consumes the ALREADY risk-adjusted segments the greedy and F1 paths consume
(the risk pre-pass ran once upstream, so ``risk_lambda`` works here unchanged), and
the SAME global ``revenue_scale`` and ``total_tvr`` normalisers, so the counts it
proposes maximise exactly the scalar the shipped per-group scorer measures. The
strictly-beats adoption gate and the belt-and-braces compliance re-check are the
caller's, mirroring the F1 refiner precedent, so the tier is never-worse by
construction; this module only produces the exact counts or an honest fallback.

Operator overrides are honored, not surrendered to: a per-segment floor or cap
bounds the counts the sweep explores, and a gold mark is charged against the daily
gold budget off the same segment-or-override union the guardrail counts (see
:mod:`kairos.optimize.dp_refine_prep`). A single gold mark used to take the exact
tier off the WHOLE channel-day; it no longer does.

Preconditions, each a silent fallback to the greedy+F1 counts (never an exception):
  * placement pins on any segment in the group (a pin carries its own break
    duration, which the count-based daily ad-seconds budget cannot express);
  * an override floor above every count that clears the retention floor and the
    segment's own spacing (no allowed count is left to propose);
  * a non-finite input on any segment (a corrupt coefficient never reaches the DP);
  * heterogeneous break length across the group (the daily ad-seconds budget would
    embed a 0/1 knapsack and the small integer break-count state is no longer exact);
  * measured open depth above the guard (worst-case cost is exponential in this
    data-dependent depth; see ``DEFAULT_MAX_OPEN_DEPTH``);
  * an exhausted per-group compute budget mid-sweep (pruned state count above
    ``DEFAULT_MAX_STATES`` or wall time above ``DEFAULT_WALL_BUDGET_SECONDS``), so
    an adversarial in-guard day degrades honestly to the greedy+F1 counts with a
    named reason instead of stalling the recompute.

Every fallback is labeled (``fell_back`` / ``reason`` / ``reason_code``) and
:func:`apply_dp_tier` aggregates per-run coverage counters, so an auditor can see
exactly how much of a run the exact tier covered and why the rest fell back.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from kairos.optimize.guardrails import Guardrails, is_compliant
from kairos.optimize._segment_math import _group_breaks, _segment_break_objects
from kairos.optimize._types import ProgramSegment
from kairos.optimize.dp_refine_prep import (
    DEFAULT_MAX_OPEN_DEPTH,
    DEFAULT_MAX_STATES,
    DEFAULT_WALL_BUDGET_SECONDS,
    OBJECTIVE_BLEND,
    OBJECTIVE_REVENUE_NET,
    DPBudgetExceeded,
    _all_finite,
    _allowed_break_counts,
    _blocking_constraint,
    _contributions,
    _effective_gold_flags,
    _max_open_depth,
    _window_ends,
)

# Re-exported so the module's public surface (and every existing importer) is
# unchanged by the prep-layer split.
__all__ = [
    "DEFAULT_MAX_OPEN_DEPTH", "DEFAULT_MAX_STATES", "DEFAULT_WALL_BUDGET_SECONDS",
    "OBJECTIVE_BLEND", "OBJECTIVE_REVENUE_NET", "DPBudgetExceeded",
    "DPRefineOutcome", "apply_dp_tier", "dp_refine_group",
]

_EPSILON = 1e-9


@dataclass
class DPRefineOutcome:
    """The DP tier's answer for one channel-day, plus how it got there.

    ``counts`` maps ``segment_id`` to the proposed break count; on any fallback it
    is exactly the greedy+F1 ``current_counts`` the caller passed in, so proposing
    the fallback is a no-op the adoption gate rejects. ``fell_back`` is True when a
    precondition tripped, and ``reason`` names which one (empty on the exact path),
    so a fallback is always observable and never dressed up as a DP win.
    ``reason_code`` is the stable machine key for that reason (empty on the exact
    path), the unit :func:`apply_dp_tier` histograms coverage by.
    """

    counts: dict          # segment_id -> proposed break count
    fell_back: bool       # True when a precondition failed and the input was kept
    reason: str           # "" on the exact path, else the precondition that tripped
    peak_states: int      # largest per-stage state table seen (0 on fallback)
    max_open_depth: int   # measured simultaneous-open depth (-1 on fallback)
    elapsed: float        # wall seconds
    dp_objective: float   # the DP's accumulated group objective (0.0 on fallback)
    reason_code: str = ""  # stable key for the fallback reason ("" on the exact path)


def _dp_core(group, contributions, kmax, allowed, guardrails, *, protected,
             gold_flags, deadline, wall_budget_seconds, max_states):
    """Interval-sweep DP over one start-sorted channel-day.

    Returns ``(counts_list, objective, peak_states)`` where ``counts_list[i]`` is
    the exact break count for ``group[i]``, chosen only from ``allowed[i]`` (the
    override-bounded, retention-floor and own-spacing feasible counts).
    ``gold_flags[i]`` is the segment-or-override gold union the daily gold budget
    is charged against, the same union :func:`~kairos.optimize.guardrails
    .check_gold_breaks` counts. Raises :class:`DPBudgetExceeded` when the pruned
    state table outgrows ``max_states`` or the wall clock passes ``deadline``,
    which the caller turns into the labeled never-worse fallback. Raises
    :class:`RuntimeError` when the sweep empties, which an unconstrained day (k = 0
    always feasible) never does but an override-floored one can; the caller turns
    that into a fallback too.
    """
    n = len(group)
    bl = group[0].break_length_seconds
    max_total = int(guardrails.max_daily_ad_seconds // bl)
    breaks_of = [[_segment_break_objects(s, k, is_gold=gold_flags[i])
                  for k in range(kmax[i] + 1)]
                 for i, s in enumerate(group)]
    starts = [s.start_seconds for s in group]
    window_end = _window_ends(group, guardrails, bl)

    def _check_wall(j: int) -> None:
        if time.perf_counter() > deadline:
            raise DPBudgetExceeded(
                "wall_budget",
                f"per-group wall budget of {wall_budget_seconds:.1f}s exhausted at segment {j} of {n}",
            )

    def feasible_local(local):
        items = []
        for i, k in local:
            items.extend(breaks_of[i][k])
        items.sort(key=lambda b: b.start_seconds)
        hours = {}
        for b in items:
            sec, cnt, prot = hours.get(b.hour, (0.0, 0, False))
            hours[b.hour] = (sec + b.duration_seconds, cnt + 1,
                             prot or b.program_type.lower() in protected)
        for sec, cnt, prot in hours.values():
            if cnt > guardrails.max_breaks_per_hour:
                return False
            # Mirror the engine EXACTLY: a protected hour's cap is protected_max (an
            # if/else, not both caps). With the shipped defaults (480 < 720) the two
            # forms coincide, but the DP must equal is_compliant for any config.
            limit = (guardrails.protected_max_ad_seconds_per_hour if prot
                     else guardrails.max_ad_seconds_per_hour)
            if sec > limit:
                return False
        for prev, cur in zip(items, items[1:]):
            gap = cur.start_seconds - (prev.start_seconds + prev.duration_seconds)
            if gap < guardrails.min_break_spacing_seconds:
                return False
        return True

    states = {(0, 0, ()): (0.0, None, None)}
    trace = []
    peak = 1
    expansions = 0
    for j in range(n):
        _check_wall(j)
        next_start = starts[j + 1] if j + 1 < n else float("inf")
        new_states = {}
        for key, (value, _, _) in states.items():
            # The wall check must live INSIDE the sweep too: a single stage over a
            # bloated state table can burn the whole budget before the next
            # per-stage check, so probe the clock every 512 expansions.
            expansions += 1
            if expansions % 512 == 0:
                _check_wall(j)
            budget, gold, open_ks = key
            # ``allowed[j]`` ascends, so the budget/gold breaks below stay valid.
            for k in allowed[j]:
                budget2 = budget + k
                if budget2 > max_total:
                    break
                gold2 = gold + (k if gold_flags[j] else 0)
                if gold2 > guardrails.gold_breaks_max_per_day:
                    break
                local = list(open_ks) + [(j, k)]
                if not feasible_local(local):
                    continue
                open2 = tuple((i, ki) for i, ki in local if window_end[i] > next_start)
                nk = (budget2, gold2, open2)
                val = value + contributions[j][k]
                cur = new_states.get(nk)
                if cur is None or val > cur[0]:
                    new_states[nk] = (val, key, k)
        # Dominance prune on the break budget within identical (gold, open) rest.
        by_rest = {}
        for (budget, gold, o), payload in new_states.items():
            by_rest.setdefault((gold, o), []).append((budget, payload))
        pruned = {}
        for (gold, o), lst in by_rest.items():
            lst.sort()
            best = -float("inf")
            for budget, payload in lst:
                if payload[0] > best + 1e-12:
                    best = payload[0]
                    pruned[(budget, gold, o)] = payload
        peak = max(peak, len(pruned))
        if len(pruned) > max_states:
            raise DPBudgetExceeded(
                "state_budget",
                f"pruned state count {len(pruned)} exceeds the {max_states} budget at segment {j} of {n}",
            )
        trace.append(pruned)
        states = pruned
        if not states:
            raise RuntimeError(f"DP infeasible at segment {j}")

    best_key = max(states, key=lambda kk: states[kk][0])
    best_val = states[best_key][0]
    counts = [0] * n
    key = best_key
    for j in range(n - 1, -1, -1):
        _, parent, k = trace[j][key]
        counts[j] = k
        key = parent
    return counts, best_val, peak


def dp_refine_group(
    group: Sequence[ProgramSegment],
    current_counts: Mapping[str, int],
    guardrails: Guardrails,
    *,
    revenue_weight: float,
    revenue_scale: float,
    total_tvr: float,
    objective_mode: str = OBJECTIVE_BLEND,
    net_of: Optional[Callable[[ProgramSegment, int], float]] = None,
    floors: Optional[Mapping[str, int]] = None,
    caps: Optional[Mapping[str, int]] = None,
    gold_by_id: Optional[Mapping[str, bool]] = None,
    placements: Optional[Mapping[str, Sequence]] = None,
    max_open_depth: int = DEFAULT_MAX_OPEN_DEPTH,
    max_states: int = DEFAULT_MAX_STATES,
    wall_budget_seconds: float = DEFAULT_WALL_BUDGET_SECONDS,
) -> DPRefineOutcome:
    """Exactly optimize ONE channel-day's break counts, or keep the greedy+F1 input.

    ``group`` is a single channel-day's ALREADY risk-adjusted segments (a subset of
    the optimizer's post-pre-pass ``segs``); ``current_counts`` is the greedy+F1
    plan for it. ``revenue_scale`` and ``total_tvr`` are the optimizer's global
    normalisers, so the proposed counts maximise the same per-group scalar the
    shipped scorer measures. In ``revenue_net`` mode ``net_of`` must be the caller's
    per-segment net primitive. ``floors``, ``caps`` and ``gold_by_id`` are honored
    exactly (bounds on the explored counts, and the daily gold budget charged off
    the segment-or-override union); ``placements`` still forces the fallback.
    ``max_states`` and ``wall_budget_seconds`` bound the sweep's per-group compute
    (see :data:`DEFAULT_MAX_STATES` / :data:`DEFAULT_WALL_BUDGET_SECONDS` for the
    measured real-corpus headroom).

    Returns a :class:`DPRefineOutcome`. On the exact path ``counts`` is the DP
    optimum and ``fell_back`` is False; on any precondition failure or exhausted
    budget ``counts`` is exactly ``current_counts``, ``reason`` names what tripped,
    and ``reason_code`` is its stable histogram key. The caller owns the
    strictly-beats adoption gate against the shipped scorer, so this function never
    itself changes a plan.
    """
    t0 = time.perf_counter()
    kept = dict(current_counts)

    def _fallback(code: str, reason: str) -> DPRefineOutcome:
        return DPRefineOutcome(
            kept, True, reason, 0, -1, time.perf_counter() - t0, 0.0, reason_code=code)

    if not group:
        return _fallback("empty_group", "empty group")

    blocking_code, blocking = _blocking_constraint(group, placements)
    if blocking:
        return _fallback(blocking_code, blocking)
    if not _all_finite(group):
        return _fallback("non_finite_input", "non-finite segment input")
    if objective_mode == OBJECTIVE_REVENUE_NET and net_of is None:
        return _fallback("net_without_primitive", "revenue_net mode without a net primitive")

    # The interval sweep and its closure lemma both require start order; the group
    # arrives keyed by segment_id, so sort a local copy by start_seconds. The daily
    # normalisers are order-invariant sums, so only the sweep sequence changes.
    ordered = sorted(group, key=lambda s: s.start_seconds)

    lengths = set(round(s.break_length_seconds, 6) for s in ordered)
    if len(lengths) > 1:
        return _fallback(
            "heterogeneous_break_lengths", f"heterogeneous break lengths {sorted(lengths)}")

    bl = ordered[0].break_length_seconds
    depth = _max_open_depth(ordered, _window_ends(ordered, guardrails, bl))
    if depth > max_open_depth:
        return _fallback("open_depth", f"open depth {depth} exceeds guard {max_open_depth}")

    allowed = _allowed_break_counts(ordered, guardrails, floors, caps)
    # An override floor can sit above every count that clears the retention floor or
    # the segment's own spacing. That leaves nothing legal to propose, so say so
    # rather than quietly dropping the operator's floor to reach a feasible count.
    empty = [s.segment_id for s, ks in zip(ordered, allowed) if not ks]
    if empty:
        return _fallback(
            "no_allowed_count", f"no allowed break count for segment(s) {sorted(empty)}")

    kmax = [ks[-1] for ks in allowed]
    gold_flags = _effective_gold_flags(ordered, gold_by_id)
    protected = frozenset(p.lower() for p in guardrails.protected_program_types)
    contribs = _contributions(
        ordered, revenue_scale, total_tvr, kmax,
        objective_mode=objective_mode, revenue_weight=revenue_weight, net_of=net_of,
    )
    try:
        counts_list, dp_objective, peak = _dp_core(
            ordered, contribs, kmax, allowed, guardrails, protected=protected,
            gold_flags=gold_flags, deadline=t0 + wall_budget_seconds,
            wall_budget_seconds=wall_budget_seconds, max_states=max_states)
    except DPBudgetExceeded as exc:
        return _fallback(exc.code, f"dp budget exceeded ({exc})")
    except RuntimeError as exc:
        return _fallback("dp_infeasible", f"dp infeasible ({exc})")

    counts = {s.segment_id: k for s, k in zip(ordered, counts_list)}
    # Belt-and-braces: the DP guarantees compliance by construction, but reconstruct
    # and re-run the engine's own check so a proposed plan that somehow breaches a
    # guardrail is dropped rather than handed to the adoption gate. The override
    # side maps go in too, so the reconstruction carries the same gold marks and
    # geometry the shipped plan would emit.
    if not is_compliant(_group_breaks(ordered, counts, gold_by_id, placements), guardrails):
        return _fallback("dp_noncompliant", "dp plan failed engine is_compliant")

    return DPRefineOutcome(
        counts, False, "", peak, depth, time.perf_counter() - t0, dp_objective)


def apply_dp_tier(
    groups: Mapping,
    state: dict,
    decisions_by_group: dict,
    guardrails: Guardrails,
    *,
    revenue_weight: float,
    revenue_scale: float,
    total_tvr: float,
    objective_mode: str,
    net_of: Optional[Callable[[ProgramSegment, int], float]],
    floors: Optional[Mapping[str, int]],
    caps: Optional[Mapping[str, int]],
    gold_by_id: Optional[Mapping[str, bool]],
    placements: Optional[Mapping[str, Sequence]],
    group_score: Callable,
    replay_decisions: Callable,
    max_open_depth: int = DEFAULT_MAX_OPEN_DEPTH,
) -> dict:
    """Adopt the exact DP plan per channel-day where it strictly beats greedy+F1.

    Mutates ``state`` (segment_id -> count) and ``decisions_by_group`` in place for
    every group the DP improves, and leaves both untouched for a group it does not
    cover or cannot beat, so the plan is never-worse than the greedy+F1 input.
    ``group_score`` is the caller's shipped per-group scorer (the strictly-beats gate
    is measured on it, not on the DP's own math) and ``replay_decisions`` its
    decision-trace rebuilder, so an adopted group's reported trace matches the F1
    tier's exactly. The engine's own :func:`is_compliant` is re-run on each proposed
    plan as belt-and-braces before adoption.

    Returns the tier's per-run coverage counters so the run can be audited:
    ``groups_total`` channel-days examined, ``groups_exact`` solved to the exact
    optimum, ``groups_adopted`` where the exact counts strictly beat greedy+F1 and
    were taken, ``groups_not_better`` where greedy+F1 already matched the optimum,
    ``groups_noncompliant`` where the belt-and-braces check rejected the proposal,
    and ``fallback_reasons``, a histogram keyed by each fallback's stable
    ``reason_code``. Counters are measured, never estimated.
    """
    fallback_reasons: dict[str, int] = {}
    stats: dict = {
        "groups_total": 0,
        "groups_exact": 0,
        "groups_adopted": 0,
        "groups_not_better": 0,
        "groups_noncompliant": 0,
        "fallback_reasons": fallback_reasons,
    }
    for key, group in groups.items():
        stats["groups_total"] += 1
        greedy_f1_counts = {s.segment_id: state[s.segment_id] for s in group}
        base_value = group_score(group, greedy_f1_counts)
        outcome = dp_refine_group(
            group, greedy_f1_counts, guardrails,
            revenue_weight=revenue_weight, revenue_scale=revenue_scale,
            total_tvr=total_tvr, objective_mode=objective_mode, net_of=net_of,
            floors=floors, caps=caps, gold_by_id=gold_by_id, placements=placements,
            max_open_depth=max_open_depth,
        )
        if outcome.fell_back:
            code = outcome.reason_code or "unlabeled"
            fallback_reasons[code] = fallback_reasons.get(code, 0) + 1
            continue
        stats["groups_exact"] += 1
        dp_value = group_score(group, outcome.counts)
        if dp_value <= base_value + _EPSILON:
            stats["groups_not_better"] += 1
            continue  # greedy+F1 already reached this group's optimum
        # Explicit raise, not assert (stripped under -O): a score the comparison
        # above did not order can only be non-finite, and adopting it would corrupt
        # the plan silently.
        if not dp_value >= base_value:
            raise RuntimeError(
                f"dp adoption gate for group {key} saw a non-finite or inconsistent score "
                f"(dp {dp_value!r} vs greedy+F1 {base_value!r}); refusing to adopt"
            )
        if not is_compliant(
            _group_breaks(group, outcome.counts, gold_by_id, placements), guardrails
        ):
            stats["groups_noncompliant"] += 1
            continue  # never adopt a plan the engine's own check rejects
        for segment in group:
            state[segment.segment_id] = outcome.counts[segment.segment_id]
        decisions_by_group[key] = replay_decisions(group, outcome.counts)
        stats["groups_adopted"] += 1
    return stats
