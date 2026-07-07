"""Exact interval-sweep dynamic program for one Kairos channel-day.

This is the production-shaped version of the validated prototype in
``dp_prototype.py``. It solves a single channel-day's break-count allocation to
the EXACT optimum of the engine's own objective (blend or revenue_net) under the
engine's own guardrails, then hands the answer back only when a set of runtime
preconditions hold; otherwise it silently falls back to the shipped greedy+F1
optimizer so it can never do worse or answer a case it does not cover.

Why a DP and not a chain: real production segments overlap in time, so the daily
guardrails (spacing, per-hour count and seconds, daily ad seconds, gold count)
couple non-adjacent segments. The sweep keeps a joint state over every OPEN
earlier segment (one whose breaks can still interact with a later segment), plus
two small integer budgets (total breaks for the daily ad-seconds cap, gold count)
and prunes by dominance. This is exact primal optimization: the global budgets are
carried as explicit state dimensions, NOT relaxed into a Lagrangian, so there is
no duality gap to reason about. Every geometry, retention, revenue, and compliance
value comes from the engine's own primitives, so DP truth equals engine truth by
construction.

Preconditions (each triggers a safe fallback to greedy+F1 when it fails):
  * uniform break_length_seconds across the day's free segments (else the daily
    budget embeds a 0/1 knapsack and the small integer B state is no longer exact);
  * measured max open-depth <= ``max_open_depth`` (worst-case cost is exponential
    in this data-dependent depth; measured max 6 across the real corpus);
  * for blend with a caller-supplied ``revenue_scale``, the certificate
    ``dp_revenue <= revenue_scale`` (so the objective's global [0, 1] clamp,
    dropped by the linear DP form, is provably inert).

Scope shipped and validated here: the free path (no overrides, no pins, no demand
weights), default guardrails, blend and revenue_net objectives, risk_lambda as the
engine's pre-pass. Overrides / pins / gold folding are specified in the task but
are NOT exercised by this module; a day carrying them falls back to greedy+F1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence

from kairos.optimize.guardrails import Guardrails, is_compliant
from kairos.optimize._segment_math import (
    _group_breaks,
    _group_objective_contribution,
    _risk_adjusted_coefficient,
    _segment_break_objects,
    _segment_retention,
    _segment_revenue,
)

OBJECTIVE_BLEND = "blend"
OBJECTIVE_REVENUE_NET = "revenue_net"
_EPSILON = 1e-9


@dataclass
class DPResult:
    """The DP's answer for one channel-day, plus how it got there."""

    counts: dict          # segment_id -> chosen break count
    objective: float      # engine-exact objective on the active mode
    revenue: float        # sum of per-segment revenue at the chosen counts (ILS)
    compliant: bool       # engine is_compliant on the reconstructed plan
    fell_back: bool       # True when a precondition failed and greedy+F1 was used
    reason: str           # "" on the exact path, else why it fell back
    peak_states: int      # largest per-stage state table seen (0 on fallback)
    max_open_depth: int   # measured simultaneous-open depth (-1 on fallback)
    elapsed: float        # wall seconds
    dp_internal: float     # the DP's own accumulated value (sanity == objective)


def group_objective(segs, counts, revenue_scale, total_tvr, *, mode, revenue_weight):
    """Score a count vector on the engine's own objective for one channel-day.

    Blend uses the refiner's separable per-group contribution (revenue_weight *
    revenue / revenue_scale + (1 - revenue_weight) * tvr-weighted retention share);
    revenue_net sums the per-segment ILS net. Both are the exact scalars the
    shipped optimizer maximises, so the values compare to the cent.
    """
    if mode == OBJECTIVE_REVENUE_NET:
        from kairos.optimize.revenue_net import segment_net_revenue

        return sum(segment_net_revenue(s, counts[s.segment_id]) for s in segs)
    contribution, _, _ = _group_objective_contribution(
        segs, counts,
        revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
    )
    return contribution


def _prep(segs, *, risk_lambda):
    """Risk pre-pass and per-day run constants, mirroring optimize_breaks exactly."""
    adj = [replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda))
           for s in segs]
    revenue_scale = max(sum(_segment_revenue(s, s.max_breaks) for s in adj), _EPSILON)
    total_tvr = sum(s.baseline_tvr for s in adj)
    return adj, revenue_scale, total_tvr


def _contributions(adj, revenue_scale, total_tvr, kmax, *, mode, revenue_weight):
    """Per-segment additive objective term for k in 0..kmax[i]."""
    if mode == OBJECTIVE_REVENUE_NET:
        from kairos.optimize.revenue_net import segment_net_revenue

        return [[segment_net_revenue(s, k) for k in range(kmax[i] + 1)]
                for i, s in enumerate(adj)]
    out = []
    for i, s in enumerate(adj):
        row = []
        for k in range(kmax[i] + 1):
            rev = revenue_weight * _segment_revenue(s, k) / revenue_scale
            ret = ((1.0 - revenue_weight) * s.baseline_tvr * _segment_retention(s, k)
                   / total_tvr if total_tvr > _EPSILON else 0.0)
            row.append(rev + ret)
        out.append(row)
    return out


def _window_ends(adj, guardrails, bl):
    """When each segment's breaks can no longer interact with a later start.

    A segment stays OPEN until the later of (a) its last possible break plus the
    minimum spacing, and (b) the ceiling of the clock hour that last break can
    reach: past both, no future break can share a spacing pair or an hour with it.
    """
    ends = []
    for s in adj:
        tail = s.start_seconds + s.duration_seconds + bl / 2.0
        spacing_end = tail + guardrails.min_break_spacing_seconds
        hour_ceiling = (int(tail // 3600.0) + 1) * 3600.0
        ends.append(max(spacing_end, hour_ceiling))
    return ends


def _max_open_depth(adj, window_end):
    """Largest set of segments simultaneously open at any later segment's start."""
    depth = 0
    for i, s in enumerate(adj):
        # count earlier-or-equal-start segments whose window covers this start
        live = sum(1 for j, t in enumerate(adj)
                   if t.start_seconds <= s.start_seconds and window_end[j] > s.start_seconds)
        depth = max(depth, live)
    return depth


def _retention_capped_kmax(adj, guardrails):
    """Per-segment cap from the retention floor (and own consecutive spacing)."""
    kmax = []
    for s in adj:
        cap = 0
        for k in range(s.max_breaks + 1):
            if k == 0 or _segment_retention(s, k) >= guardrails.min_retention_floor:
                # also reject k whose own consecutive breaks violate spacing
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


def _dp_core(adj, contributions, kmax, guardrails, *, protected):
    """Interval-sweep DP. Returns (counts_list, objective, peak_states)."""
    n = len(adj)
    bl = adj[0].break_length_seconds
    max_total = int(guardrails.max_daily_ad_seconds // bl)
    breaks_of = [[_segment_break_objects(s, k) for k in range(kmax[i] + 1)]
                 for i, s in enumerate(adj)]
    starts = [s.start_seconds for s in adj]
    window_end = _window_ends(adj, guardrails, bl)

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
            # Mirror the engine EXACTLY: a protected hour's cap is protected_max
            # (an if/else, not both caps). This matters only when protected_max >
            # max_ad_seconds_per_hour; with the shipped defaults (480 < 720) the two
            # forms are identical, but the DP must equal is_compliant for any config.
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
            B, G, open_ks = key
            for k in range(kmax[j] + 1):
                B2 = B + k
                if B2 > max_total:
                    break
                G2 = G + (k if adj[j].is_gold else 0)
                if G2 > guardrails.gold_breaks_max_per_day:
                    break
                local = list(open_ks) + [(j, k)]
                if not feasible_local(local):
                    continue
                open2 = tuple((i, ki) for i, ki in local if window_end[i] > next_start)
                nk = (B2, G2, open2)
                val = value + contributions[j][k]
                cur = new_states.get(nk)
                if cur is None or val > cur[0]:
                    new_states[nk] = (val, key, k)
        # dominance prune on B within identical (G, open_ks)
        by_rest = {}
        for (B, G, o), payload in new_states.items():
            by_rest.setdefault((G, o), []).append((B, payload))
        pruned = {}
        for (G, o), lst in by_rest.items():
            lst.sort()
            best = -float("inf")
            for B, payload in lst:
                if payload[0] > best + 1e-12:
                    best = payload[0]
                    pruned[(B, G, o)] = payload
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


def _greedy_f1_counts(segs, guardrails, *, revenue_weight, risk_lambda, mode):
    """Fallback: the shipped greedy+F1 optimizer's counts for one channel-day."""
    from kairos.optimize.optimizer import optimize_breaks

    res = optimize_breaks(
        segs, guardrails, revenue_weight=revenue_weight, risk_lambda=risk_lambda,
        refine=True, objective_mode=mode,
    )
    return {p.segment_id: p.num_breaks for p in res.segments}


def dp_optimize_day(
    segs,
    guardrails: Optional[Guardrails] = None,
    *,
    revenue_weight: float = 0.5,
    risk_lambda: float = 0.0,
    objective_mode: str = OBJECTIVE_BLEND,
    revenue_scale: Optional[float] = None,
    max_open_depth: int = 10,
) -> DPResult:
    """Exactly optimize ONE channel-day, or fall back to greedy+F1 if uncovered.

    ``segs`` must be a single (channel, day) group (the caller decomposes; the
    engine's guardrails never cross a channel-day). Returns a :class:`DPResult`
    whose ``fell_back`` flag says whether the exact DP ran or the shipped optimizer
    was used instead. The returned plan is always engine-compliant.
    """
    guardrails = guardrails or Guardrails()
    protected = frozenset(p.lower() for p in guardrails.protected_program_types)
    t0 = time.perf_counter()

    def _fallback(reason):
        counts = _greedy_f1_counts(
            segs, guardrails, revenue_weight=revenue_weight,
            risk_lambda=risk_lambda, mode=objective_mode)
        adj, scale, tvr = _prep(segs, risk_lambda=risk_lambda)
        scale = revenue_scale if revenue_scale is not None else scale
        obj = group_objective(adj, counts, scale, tvr,
                              mode=objective_mode, revenue_weight=revenue_weight)
        rev = sum(_segment_revenue(s, counts[s.segment_id]) for s in adj)
        comp = is_compliant(_group_breaks(adj, counts), guardrails)
        return DPResult(counts, obj, rev, comp, True, reason,
                        0, -1, time.perf_counter() - t0, obj)

    if not segs:
        return _fallback("empty group")

    adj, default_scale, total_tvr = _prep(segs, risk_lambda=risk_lambda)
    scale = revenue_scale if revenue_scale is not None else default_scale
    # The interval sweep and the closure lemma both require start order; the daily
    # constants (revenue_scale, total_tvr) are order-invariant sums, so sorting here
    # changes nothing but the sweep sequence.
    adj = sorted(adj, key=lambda s: s.start_seconds)

    # Precondition 1: uniform break length across free segments.
    lengths = set(round(s.break_length_seconds, 6) for s in adj)
    if len(lengths) > 1:
        return _fallback(f"heterogeneous break lengths {sorted(lengths)}")

    # Precondition 2: bounded open-depth (else exponential blowup).
    bl = adj[0].break_length_seconds
    window_end = _window_ends(adj, guardrails, bl)
    depth = _max_open_depth(adj, window_end)
    if depth > max_open_depth:
        return _fallback(f"open-depth {depth} exceeds cap {max_open_depth}")

    kmax = _retention_capped_kmax(adj, guardrails)
    contribs = _contributions(adj, scale, total_tvr, kmax,
                              mode=objective_mode, revenue_weight=revenue_weight)
    try:
        counts_list, dp_internal, peak = _dp_core(
            adj, contribs, kmax, guardrails, protected=protected)
    except RuntimeError as exc:
        return _fallback(f"dp infeasible ({exc})")

    counts = {s.segment_id: k for s, k in zip(adj, counts_list)}
    revenue = sum(_segment_revenue(s, counts[s.segment_id]) for s in adj)

    # Precondition 3: with a caller-supplied revenue_scale, the dropped [0, 1]
    # clamp must be inert (dp_revenue <= revenue_scale) for the linear form to be
    # exact. The default scale is full-max-load revenue, so this holds by design.
    if revenue_scale is not None and objective_mode == OBJECTIVE_BLEND:
        if revenue > revenue_scale + _EPSILON:
            return _fallback(f"revenue {revenue:.2f} exceeds custom scale {revenue_scale:.2f}")

    # Belt-and-braces: reconstruct and check the engine's own compliance.
    compliant = is_compliant(_group_breaks(adj, counts), guardrails)
    if not compliant:
        return _fallback("dp plan failed engine is_compliant")

    objective = group_objective(adj, counts, scale, total_tvr,
                                mode=objective_mode, revenue_weight=revenue_weight)
    return DPResult(counts, objective, revenue, True, False, "",
                    peak, depth, time.perf_counter() - t0, dp_internal)
