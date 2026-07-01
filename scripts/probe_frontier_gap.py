"""Probe: does the greedy frontier proxy preserve the revenue-vs-retention slope.

READ-ONLY diagnostic. Runs no writes, commits nothing, edits no engine code.

Context
-------
The dashboard revenue-vs-retention frontier (kairos_api/server.py
``_frontier_points_cached``) traces the trade-off by sweeping revenue weights on a
single representative owned-channel day with ``refine=False`` (pure greedy), so a
multi-day sweep stays interactive. The committed weekly plan, by contrast, is
optimized with ``refine=True`` (greedy plus the F1 local-search refiner).

That means the curve the operator reads and the plan the operator ships come from
two different optimizer settings. This probe measures the gap between them across
the same seven revenue weights the frontier uses, on the same representative day,
under the same saved settings, to answer one question:

  Is the greedy-vs-refined gap roughly FLAT across revenue weights, or does it
  VARY with the weight.

If the gap is a near-constant offset, the greedy curve is the refined curve
translated vertically: the SLOPE (the revenue-per-point-of-retention trade-off the
operator steers by) is preserved, and building a whole-week refined frontier is
polish. If the gap VARIES with weight, greedy deforms the curve, distorts the
slope, and can silently mis-steer the weight choice: the refined frontier is then a
correctness fix, not polish.

This script reuses the server's own conventions (operator channel, representative
day, saved guardrails, pacing kwargs) so the probe basis matches the real frontier.
It only differs by also running ``refine=True`` alongside ``refine=False``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}%"


def _fmt_ret(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def main() -> int:
    # Import the SAME helpers the live frontier uses, so the probe basis is
    # identical to the real curve (settings, owned channel, representative day,
    # pacing kwargs). Only the refine flag is varied by this probe.
    try:
        from kairos.service import run_scenario
        from kairos_api.server import (
            _frontier_data_signature,
            _load_settings,
            _owned_representative_day,
            _pacing_call_kwargs,
        )
    except Exception as exc:  # pragma: no cover - environment guard
        print("PROBE ABORTED: could not import engine/server helpers.")
        print(f"  reason: {exc!r}")
        print("VERDICT: UNKNOWN (engine unavailable, no data measured)")
        return 1

    settings = _load_settings()
    owned = str(settings.operator_channel or "").strip()
    if not owned:
        print("No owned channel is configured (settings.operator_channel is empty).")
        print("The frontier itself returns the honest 'no_channel' state here, so")
        print("there is nothing to measure.")
        print("VERDICT: UNKNOWN (no owned channel configured, no data measured)")
        return 0

    signature = _frontier_data_signature()
    effective_day = _owned_representative_day(signature, owned)
    if not effective_day:
        print(f"Owned channel {owned!r} has no dated programmes on disk, so the")
        print("frontier has no representative day to trace. Nothing to measure.")
        print("VERDICT: UNKNOWN (no representative day, no data measured)")
        return 0

    saved_weight = int(settings.revenue_weight)
    retention_floor = float(settings.min_retention_floor)
    max_breaks_per_hour = int(settings.max_breaks_per_hour)
    risk_lambda = float(settings.risk_lambda)
    pacing = _pacing_call_kwargs()

    # The exact weight set the frontier sweeps (the six fixed stops plus the saved
    # weight), sorted and de-duplicated.
    weights = sorted({0, 20, 40, 60, 80, 100, saved_weight})

    print("Frontier greedy-vs-refined gap probe")
    print("=" * 78)
    print(f"owned channel        : {owned}")
    print(f"representative day    : {effective_day}  (busiest broadcast day)")
    print(f"saved revenue_weight  : {saved_weight}")
    print(f"retention_floor       : {retention_floor}")
    print(f"max_breaks_per_hour   : {max_breaks_per_hour}")
    print(f"risk_lambda           : {risk_lambda}")
    print(f"weights swept         : {weights}")
    print("refine=True  -> committed-plan optimizer (greedy + F1 refiner)")
    print("refine=False -> current frontier proxy (pure greedy)")
    print("=" * 78)

    rows: list[dict[str, object]] = []
    for weight in weights:
        row: dict[str, object] = {"weight": weight}
        for label, refine in (("refined", True), ("greedy", False)):
            try:
                payload = run_scenario(
                    revenue_weight=weight,
                    retention_floor=retention_floor,
                    max_breaks_per_hour=max_breaks_per_hour,
                    risk_lambda=risk_lambda,
                    channel=owned,
                    day=effective_day,
                    refine=refine,
                    **pacing,
                )
                summary = payload.get("summary", {}) or {}
                row[f"{label}_rev"] = summary.get("projected_revenue")
                row[f"{label}_ret"] = summary.get("average_retention")
            except Exception as exc:  # pragma: no cover - per-weight guard
                print(f"  run failed at weight={weight} refine={refine}: {exc!r}")
                row[f"{label}_rev"] = None
                row[f"{label}_ret"] = None
        rows.append(row)

    # Derive deltas.
    for row in rows:
        rr = row.get("refined_rev")
        gr = row.get("greedy_rev")
        if isinstance(rr, (int, float)) and isinstance(gr, (int, float)):
            row["rev_delta"] = float(rr) - float(gr)
            row["rev_delta_pct"] = (
                (float(rr) - float(gr)) / float(gr) * 100.0 if float(gr) != 0.0 else None
            )
        else:
            row["rev_delta"] = None
            row["rev_delta_pct"] = None
        rt = row.get("refined_ret")
        gt = row.get("greedy_ret")
        if isinstance(rt, (int, float)) and isinstance(gt, (int, float)):
            row["ret_delta"] = float(rt) - float(gt)
        else:
            row["ret_delta"] = None

    # Print the table.
    print()
    header = (
        f"{'wt':>3} | {'refined_rev':>13} {'greedy_rev':>13} "
        f"{'rev_delta':>12} {'delta_pct':>10} | "
        f"{'ref_ret':>7} {'grd_ret':>7} {'ret_delta':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        mark = " *" if row["weight"] == saved_weight else "  "
        print(
            f"{str(row['weight']):>3}"
            f" | {_fmt_money(row.get('refined_rev')):>13}"
            f" {_fmt_money(row.get('greedy_rev')):>13}"
            f" {_fmt_money(row.get('rev_delta')):>12}"
            f" {_fmt_pct(row.get('rev_delta_pct')):>10}"
            f" | {_fmt_ret(row.get('refined_ret')):>7}"
            f" {_fmt_ret(row.get('greedy_ret')):>7}"
            f" {_fmt_ret(row.get('ret_delta')):>9}{mark}"
        )
    print("-" * len(header))
    print("* = saved revenue_weight (the point the dashboard marks 'selected')")

    # Verdict math.
    #
    # A vertical translation (constant ABSOLUTE revenue offset at every weight)
    # leaves the curve's slope, and therefore the operator's weight choice,
    # unchanged. So the honest test is whether the greedy-to-refined revenue gap is
    # roughly constant across the weights that actually place breaks (weight 0 is a
    # degenerate no-break point where both optimizers collapse to the same plan and
    # revenue ~ 0, so it carries no slope information and is excluded from spread).
    active = [
        row
        for row in rows
        if isinstance(row.get("rev_delta"), (int, float))
        and isinstance(row.get("greedy_rev"), (int, float))
        and float(row["greedy_rev"]) > 0.0
    ]

    print()
    print("Verdict basis")
    print("-" * 78)
    if len(active) < 2:
        print("Fewer than two revenue-bearing weights returned, so the slope cannot")
        print("be characterised from this run.")
        print("VERDICT: UNKNOWN (insufficient revenue-bearing points measured)")
        return 0

    abs_deltas = [float(row["rev_delta"]) for row in active]
    pct_deltas = [
        float(row["rev_delta_pct"])
        for row in active
        if isinstance(row.get("rev_delta_pct"), (int, float))
    ]
    refined_revs = [float(row["refined_rev"]) for row in active]

    abs_spread = max(abs_deltas) - min(abs_deltas)
    mean_refined = sum(refined_revs) / len(refined_revs)
    rel_spread = abs_spread / mean_refined * 100.0 if mean_refined else None
    pct_spread = (max(pct_deltas) - min(pct_deltas)) if pct_deltas else None

    print(f"revenue-bearing weights : {[row['weight'] for row in active]}")
    print(f"abs rev_delta range      : {_fmt_money(min(abs_deltas))} .. {_fmt_money(max(abs_deltas))}")
    print(f"abs rev_delta spread     : {_fmt_money(abs_spread)} ILS")
    print(f"mean refined revenue     : {_fmt_money(mean_refined)} ILS")
    print(
        "spread as pct of revenue : "
        + (f"{rel_spread:.3f}%" if rel_spread is not None else "n/a")
    )
    print(
        "delta_pct spread          : "
        + (f"{pct_spread:.3f} pp" if pct_spread is not None else "n/a")
    )

    # Threshold: the gap's spread across weights, measured as a fraction of the
    # revenue magnitude, is what a slope reads. If that relative spread stays under
    # half a percent of revenue, the greedy curve is a near-vertical translation of
    # the refined curve and the slope is preserved. Above that, the curve deforms
    # weight-by-weight and the slope the operator steers by is distorted.
    threshold_rel = 0.5  # percent of mean refined revenue
    print(f"flatness threshold        : rel spread < {threshold_rel:.3f}% of revenue")
    print("-" * 78)

    if rel_spread is not None and rel_spread < threshold_rel:
        print(
            "VERDICT: FLAT (greedy proxy preserves slope, Phase 4 is polish). "
            f"The greedy-to-refined revenue gap varies by only {_fmt_money(abs_spread)} "
            f"ILS ({rel_spread:.3f}% of the ~{_fmt_money(mean_refined)} ILS revenue) "
            f"across the revenue-bearing weights {[row['weight'] for row in active]}, "
            "a near-constant vertical offset that leaves the revenue-vs-retention "
            "slope, and therefore the operator's weight choice, unchanged."
        )
    else:
        rel_txt = f"{rel_spread:.3f}%" if rel_spread is not None else "n/a"
        print(
            "VERDICT: WEIGHT-VARYING (slope distorted, Phase 4 is load-bearing). "
            f"The greedy-to-refined revenue gap swings by {_fmt_money(abs_spread)} "
            f"ILS ({rel_txt} of the ~{_fmt_money(mean_refined)} ILS revenue) across "
            f"the revenue-bearing weights {[row['weight'] for row in active]}, above "
            f"the {threshold_rel:.3f}% flatness bar. That weight-dependent gap deforms "
            "the revenue-vs-retention curve rather than translating it, so the greedy "
            "proxy distorts the slope and can mis-steer the weight choice."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
