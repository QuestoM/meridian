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

Interface contract with :func:`kairos.optimize.optimizer.optimize_breaks`: this
tier consumes the ALREADY risk-adjusted segments the greedy and F1 paths consume
(the risk pre-pass ran once upstream, so ``risk_lambda`` works here unchanged), and
the SAME global ``revenue_scale`` and ``total_tvr`` normalisers, so the counts it
proposes maximise exactly the scalar the shipped per-group scorer measures. The
strictly-beats adoption gate and the belt-and-braces compliance re-check are the
caller's, mirroring the F1 refiner precedent, so the tier is never-worse by
construction; this module only produces the exact counts or an honest fallback.

Preconditions, each a silent fallback to the greedy+F1 counts (never an exception):
  * placement pins, per-segment override floors or caps, or a gold-forcing
    constraint on any segment in the group (folding those into the DP state is
    unprototyped, so a constrained group is left to the greedy+F1 path);
  * a non-finite input on any segment (a corrupt coefficient never reaches the DP);
  * heterogeneous break length across the group (the daily ad-seconds budget would
    embed a 0/1 knapsack and the small integer break-count state is no longer exact);
  * measured open depth above the guard (worst-case cost is exponential in this
    data-dependent depth; see ``DEFAULT_MAX_OPEN_DEPTH``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from kairos.optimize.guardrails import Guardrails, is_compliant
from kairos.optimize._segment_math import (
    _group_breaks,
    _segment_break_objects,
    _segment_retention,
    _segment_revenue,
)
from kairos.optimize._types import ProgramSegment

OBJECTIVE_BLEND = "blend"
OBJECTIVE_REVENUE_NET = "revenue_net"
_EPSILON = 1e-9

# Default open-depth guard. The exact DP's worst-case state count is exponential in
# the simultaneously-open depth; measured across all 120 real channel-days the max
# open depth is 13 (Kan 11 2024-11-09, ``analysis/dp-exactness/verify``), so the
# guard defaults ABOVE 13 to keep the real corpus on the exact path. A day deeper
# than the guard falls back to greedy+F1 rather than risk a runtime blowup.
DEFAULT_MAX_OPEN_DEPTH = 14


@dataclass
class DPRefineOutcome:
    """The DP tier's answer for one channel-day, plus how it got there.

    ``counts`` maps ``segment_id`` to the proposed break count; on any fallback it
    is exactly the greedy+F1 ``current_counts`` the caller passed in, so proposing
    the fallback is a no-op the adoption gate rejects. ``fell_back`` is True when a
    precondition tripped, and ``reason`` names which one (empty on the exact path),
    so a fallback is always observable and never dressed up as a DP win.
    """

    counts: dict          # segment_id -> proposed break count
    fell_back: bool       # True when a precondition failed and the input was kept
    reason: str           # "" on the exact path, else the precondition that tripped
    peak_states: int      # largest per-stage state table seen (0 on fallback)
    max_open_depth: int   # measured simultaneous-open depth (-1 on fallback)
    elapsed: float        # wall seconds
    dp_objective: float   # the DP's accumulated group objective (0.0 on fallback)


def _all_finite(group: Sequence[ProgramSegment]) -> bool:
    """Every numeric input the DP reads is finite (no NaN or inf reaches the sweep)."""
    for s in group:
        for value in (
            s.impact_coefficient, s.baseline_tvr, s.cpp, s.premium, s.unit_seconds,
            s.retention_baseline, s.start_seconds, s.duration_seconds,
            s.break_length_seconds, float(s.first_break_multiplier),
        ):
            if not math.isfinite(value):
                return False
    return True


def _blocking_constraint(
    group: Sequence[ProgramSegment],
    floors: Optional[Mapping[str, int]],
    caps: Optional[Mapping[str, int]],
    gold_by_id: Optional[Mapping[str, bool]],
    placements: Optional[Mapping[str, Sequence]],
) -> str:
    """The first constraint that puts the group off the DP's free path, or ""."""
    floors = floors or {}
    caps = caps or {}
    gold_by_id = gold_by_id or {}
    placements = placements or {}
    for s in group:
        sid = s.segment_id
        if placements.get(sid):
            return "placement pins present"
        if floors.get(sid, 0) > 0:
            return "segment override floor present"
        if caps.get(sid, s.max_breaks) < s.max_breaks:
            return "segment override cap present"
        if gold_by_id.get(sid, False) and not s.is_gold:
            return "gold-forcing constraint present"
    return ""


def _window_ends(group: Sequence[ProgramSegment], guardrails: Guardrails, bl: float) -> list:
    """When each segment's breaks can no longer interact with a later start.

    A segment stays OPEN until the later of its last possible break plus the minimum
    spacing and the ceiling of the clock hour that last break can reach: past both,
    no future break can share a spacing pair or an hour with it.
    """
    ends = []
    for s in group:
        tail = s.start_seconds + s.duration_seconds + bl / 2.0
        spacing_end = tail + guardrails.min_break_spacing_seconds
        hour_ceiling = (int(tail // 3600.0) + 1) * 3600.0
        ends.append(max(spacing_end, hour_ceiling))
    return ends


def _max_open_depth(group: Sequence[ProgramSegment], window_end: Sequence[float]) -> int:
    """Largest set of segments simultaneously open at any later segment's start."""
    depth = 0
    for s in group:
        live = sum(
            1 for j, t in enumerate(group)
            if t.start_seconds <= s.start_seconds and window_end[j] > s.start_seconds
        )
        depth = max(depth, live)
    return depth


def _retention_capped_kmax(group: Sequence[ProgramSegment], guardrails: Guardrails) -> list:
    """Per-segment break cap from the retention floor and own consecutive spacing."""
    kmax = []
    for s in group:
        cap = 0
        for k in range(s.max_breaks + 1):
            if k == 0 or _segment_retention(s, k) >= guardrails.min_retention_floor:
                breaks = _segment_break_objects(s, k)
                ok = True
                for prev, cur in zip(breaks, breaks[1:]):
                    gap = cur.start_seconds - (prev.start_seconds + prev.duration_seconds)
                    if gap < guardrails.min_break_spacing_seconds:
                        ok = False
                        break
                if ok:
                    cap = k
                else:
                    break
            else:
                break
        kmax.append(cap)
    return kmax


def _contributions(
    group: Sequence[ProgramSegment],
    revenue_scale: float,
    total_tvr: float,
    kmax: Sequence[int],
    *,
    objective_mode: str,
    revenue_weight: float,
    net_of: Optional[Callable[[ProgramSegment, int], float]],
) -> list:
    """Per-segment additive objective term for k in 0..kmax[i].

    Blend mirrors :func:`_group_objective_contribution` exactly (the unclamped
    additive share ``revenue_weight * revenue / revenue_scale + (1 -
    revenue_weight) * tvr-weighted retention / total_tvr``), so the DP maximises the
    same scalar the shipped scorer measures. Revenue-net sums the caller's own
    per-segment net primitive. Both use the global normalisers passed in, not
    per-group ones, so the argmax matches the shipped objective.
    """
    if objective_mode == OBJECTIVE_REVENUE_NET:
        return [[net_of(s, k) for k in range(kmax[i] + 1)]
                for i, s in enumerate(group)]
    out = []
    for i, s in enumerate(group):
        row = []
        for k in range(kmax[i] + 1):
            rev = revenue_weight * _segment_revenue(s, k) / revenue_scale
            ret = ((1.0 - revenue_weight) * s.baseline_tvr * _segment_retention(s, k)
                   / total_tvr if total_tvr > _EPSILON else 0.0)
            row.append(rev + ret)
        out.append(row)
    return out


def _dp_core(group, contributions, kmax, guardrails, *, protected):
    """Interval-sweep DP over one start-sorted channel-day.

    Returns ``(counts_list, objective, peak_states)`` where ``counts_list[i]`` is
    the exact break count for ``group[i]``. Raises :class:`RuntimeError` only if the
    sweep empties, which the free path (k = 0 is always feasible) never does; the
    caller still treats it as a fallback for safety.
    """
    n = len(group)
    bl = group[0].break_length_seconds
    max_total = int(guardrails.max_daily_ad_seconds // bl)
    breaks_of = [[_segment_break_objects(s, k) for k in range(kmax[i] + 1)]
                 for i, s in enumerate(group)]
    starts = [s.start_seconds for s in group]
    window_end = _window_ends(group, guardrails, bl)

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
    for j in range(n):
        next_start = starts[j + 1] if j + 1 < n else float("inf")
        new_states = {}
        for key, (value, _, _) in states.items():
            budget, gold, open_ks = key
            for k in range(kmax[j] + 1):
                budget2 = budget + k
                if budget2 > max_total:
                    break
                gold2 = gold + (k if group[j].is_gold else 0)
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
) -> DPRefineOutcome:
    """Exactly optimize ONE channel-day's break counts, or keep the greedy+F1 input.

    ``group`` is a single channel-day's ALREADY risk-adjusted segments (a subset of
    the optimizer's post-pre-pass ``segs``); ``current_counts`` is the greedy+F1
    plan for it. ``revenue_scale`` and ``total_tvr`` are the optimizer's global
    normalisers, so the proposed counts maximise the same per-group scalar the
    shipped scorer measures. In ``revenue_net`` mode ``net_of`` must be the caller's
    per-segment net primitive.

    Returns a :class:`DPRefineOutcome`. On the exact path ``counts`` is the DP
    optimum and ``fell_back`` is False; on any precondition failure ``counts`` is
    exactly ``current_counts`` and ``reason`` names the tripped precondition. The
    caller owns the strictly-beats adoption gate against the shipped scorer, so this
    function never itself changes a plan.
    """
    t0 = time.perf_counter()
    kept = dict(current_counts)

    def _fallback(reason: str) -> DPRefineOutcome:
        return DPRefineOutcome(kept, True, reason, 0, -1, time.perf_counter() - t0, 0.0)

    if not group:
        return _fallback("empty group")

    blocking = _blocking_constraint(group, floors, caps, gold_by_id, placements)
    if blocking:
        return _fallback(blocking)
    if not _all_finite(group):
        return _fallback("non-finite segment input")
    if objective_mode == OBJECTIVE_REVENUE_NET and net_of is None:
        return _fallback("revenue_net mode without a net primitive")

    # The interval sweep and its closure lemma both require start order; the group
    # arrives keyed by segment_id, so sort a local copy by start_seconds. The daily
    # normalisers are order-invariant sums, so only the sweep sequence changes.
    ordered = sorted(group, key=lambda s: s.start_seconds)

    lengths = set(round(s.break_length_seconds, 6) for s in ordered)
    if len(lengths) > 1:
        return _fallback(f"heterogeneous break lengths {sorted(lengths)}")

    bl = ordered[0].break_length_seconds
    depth = _max_open_depth(ordered, _window_ends(ordered, guardrails, bl))
    if depth > max_open_depth:
        return _fallback(f"open depth {depth} exceeds guard {max_open_depth}")

    kmax = _retention_capped_kmax(ordered, guardrails)
    protected = frozenset(p.lower() for p in guardrails.protected_program_types)
    contribs = _contributions(
        ordered, revenue_scale, total_tvr, kmax,
        objective_mode=objective_mode, revenue_weight=revenue_weight, net_of=net_of,
    )
    try:
        counts_list, dp_objective, peak = _dp_core(
            ordered, contribs, kmax, guardrails, protected=protected)
    except RuntimeError as exc:
        return _fallback(f"dp infeasible ({exc})")

    counts = {s.segment_id: k for s, k in zip(ordered, counts_list)}
    # Belt-and-braces: the DP guarantees compliance by construction, but reconstruct
    # and re-run the engine's own check so a proposed plan that somehow breaches a
    # guardrail is dropped rather than handed to the adoption gate.
    if not is_compliant(_group_breaks(ordered, counts), guardrails):
        return _fallback("dp plan failed engine is_compliant")

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
) -> None:
    """Adopt the exact DP plan per channel-day where it strictly beats greedy+F1.

    Mutates ``state`` (segment_id -> count) and ``decisions_by_group`` in place for
    every group the DP improves, and leaves both untouched for a group it does not
    cover or cannot beat, so the plan is never-worse than the greedy+F1 input.
    ``group_score`` is the caller's shipped per-group scorer (the strictly-beats gate
    is measured on it, not on the DP's own math) and ``replay_decisions`` its
    decision-trace rebuilder, so an adopted group's reported trace matches the F1
    tier's exactly. The engine's own :func:`is_compliant` is re-run on each proposed
    plan as belt-and-braces before adoption.
    """
    for key, group in groups.items():
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
            continue
        dp_value = group_score(group, outcome.counts)
        if dp_value <= base_value + _EPSILON:
            continue  # greedy+F1 already reached this group's optimum
        if not is_compliant(
            _group_breaks(group, outcome.counts, gold_by_id, placements), guardrails
        ):
            continue  # never adopt a plan the engine's own check rejects
        assert dp_value >= base_value, "dp tier regressed a group below greedy+F1"
        for segment in group:
            state[segment.segment_id] = outcome.counts[segment.segment_id]
        decisions_by_group[key] = replay_decisions(group, outcome.counts)
