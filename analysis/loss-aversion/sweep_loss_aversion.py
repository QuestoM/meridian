"""Loss-aversion extraction: resolve the 1.947 vs 1.69 first-break paradox.

Question: the pooled first-break multiplier 1.947 is inflated (honest ~1.69),
yet lowering it was measured revenue-negative. Hypothesis: the inflation is a
hidden asymmetric loss weight. Design under test: honest multiplier 1.69 plus
an explicit asymmetry weight w > 1 on retention cost in the net objective:

    decision objective = gross_revenue(seg, k) - w * retention_cost_ils(seg, k)

Protocol (anti-circularity): every arm's plan (baseline 1.947 w=1, and honest
1.69 for each w in the grid) is re-optimized under its OWN belief, then every
plan is evaluated as a FIXED plan under BOTH evaluation worlds:

    world 1.947: segments rebuilt with first_break_multiplier = 1.947, w = 1
    world 1.690: segments rebuilt with first_break_multiplier = 1.690, w = 1

Evaluation basis is the engine-exact per-break net used by the net objective:
gross = _segment_revenue(seg, k), cost = segment_retention_cost_ils(seg, k)
(each break valued at its own retention). No pins are applied at evaluation
time, identically for every arm, so comparisons are like for like.

Twelve evaluation days: all 4 real channels x 2024-11-01..2024-11-03, the same
channel-day subset recorded in the F1 refiner commit 7cecd35.

Optimization runs through the same shared core the weekly export uses
(kairos.optimize.day_core._optimize_one_day) with objective_mode='revenue_net'
and refine=True, so plans are what a real recompute would produce. The weight
w is injected by wrapping kairos.optimize.revenue_net.segment_net_revenue,
which optimize_breaks imports lazily on every call and threads into the
refiner as net_of, so greedy and refiner both climb the same weighted scalar.

Outputs (this directory):
    eval_matrix_world_1947.csv   all plans scored in the 1.947 world
    eval_matrix_world_1690.csv   all plans scored in the 1.690 world
    per_day_detail.csv           per channel-day gross/cost/net per arm x world

Run:
    /Users/home/.venvs/meridian/bin/python analysis/loss-aversion/sweep_loss_aversion.py
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data.loaders import load_programmes  # noqa: E402
from kairos.data.transform import build_segments_from_programmes  # noqa: E402
from kairos.export.schedule import (  # noqa: E402
    DEFAULT_IMPACT_MODEL_PATH,
    _build_classifier,
    _load_constraints,
)
import kairos.optimize.revenue_net as revenue_net_mod  # noqa: E402
from kairos.optimize.revenue_net import segment_retention_cost_ils  # noqa: E402
from kairos.optimize._segment_math import (  # noqa: E402
    _segment_break_objects,
    _segment_revenue,
)
from kairos.optimize.day_core import _optimize_one_day  # noqa: E402
from kairos.optimize.guardrails import evaluate  # noqa: E402
from kairos.optimize.optimizer import optimize_breaks  # noqa: E402
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings  # noqa: E402
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine  # noqa: E402
from kairos.optimize.inventory import load_inventory  # noqa: E402
from kairos.optimize.pacing import load_campaigns  # noqa: E402
from kairos.model.impact import load_impact_model  # noqa: E402
from kairos.service import (  # noqa: E402
    _pacing_knobs_from_settings,
    guardrails_from_settings,
)
from kairos_api.core import _load_settings, _model_dump, _reference_today  # noqa: E402

BASELINE_MULT = 1.947
HONEST_MULT = 1.690
EVAL_DAYS = ("2024-11-01", "2024-11-02", "2024-11-03")
W_GRID = (1.0, 1.05, 1.10, 1.152, 1.20, 1.25, 1.30, 1.40,
          1.50, 1.75, 2.00, 2.25, 2.50, 3.00)
OUT_DIR = Path(__file__).resolve().parent

# The unweighted primitives, captured once so the weighted wrapper can never
# recurse into a patched module attribute.
_ORIG_NET = revenue_net_mod.segment_net_revenue


def _weighted_net_factory(w: float):
    """Net objective with an explicit asymmetry weight on retention cost."""
    if w == 1.0:
        return _ORIG_NET

    def weighted_net(segment, k: int) -> float:
        if k <= 0:
            return 0.0
        return _segment_revenue(segment, k) - w * segment_retention_cost_ils(segment, k)

    return weighted_net


class World:
    """Segment builder for one fixed first-break-multiplier belief."""

    def __init__(self, shared: "Shared", multiplier: float):
        self.multiplier = multiplier
        self.assumptions = replace(
            OptimizerAssumptions(), first_break_multiplier=multiplier
        )
        self.impact_model = load_impact_model(
            DEFAULT_IMPACT_MODEL_PATH, assumptions=self.assumptions
        )
        self.shared = shared
        self._cache: dict[tuple[str, str], list] = {}

    def segments_for(self, channel: str, day: str) -> list:
        key = (channel, day)
        if key not in self._cache:
            self._cache[key] = build_segments_from_programmes(
                self.shared.programmes, self.shared.classifier, self.shared.pricing,
                assumptions=self.assumptions, impact_model=self.impact_model,
                channel=channel, day=day,
            )
        return self._cache[key]


class Shared:
    """Resources loaded once, exactly as the weekly export loads them."""

    def __init__(self):
        settings = _load_settings()
        self.settings_map = _model_dump(settings)
        self.pricing = pricing_from_settings(self.settings_map, None)
        self.classifier = _build_classifier()
        self.guardrails = guardrails_from_settings(self.settings_map)
        self.weight = self.settings_map["revenue_weight"] / 100.0
        self.risk_lambda = self.settings_map["risk_lambda"]
        self.operator_channel = str(self.settings_map.get("operator_channel", "") or "")
        self.constraints = _load_constraints(None)
        self.demand_engine = AdvertiserRuleEngine.from_files()
        self.inventory_pool = load_inventory()
        self.campaigns = load_campaigns()
        self.pacing_today = _reference_today(settings)
        self.pacing_knobs = _pacing_knobs_from_settings(self.settings_map)
        self.programmes = load_programmes()
        self.channels = sorted(set(self.programmes["Channel"].dropna().astype(str)))


def optimize_arm(shared: Shared, world: World, w: float) -> dict[tuple[str, str], dict[str, int]]:
    """Re-optimize the 12 channel-days under (world.multiplier, w). Returns counts."""
    plans: dict[tuple[str, str], dict[str, int]] = {}
    revenue_net_mod.segment_net_revenue = _weighted_net_factory(w)
    try:
        for channel in shared.channels:
            for day in EVAL_DAYS:
                segments = world.segments_for(channel, day)
                if not segments:
                    continue
                result = _optimize_one_day(
                    segments,
                    guardrails=shared.guardrails,
                    revenue_weight=shared.weight,
                    risk_lambda=shared.risk_lambda,
                    demand_engine=shared.demand_engine,
                    inventory_pool=shared.inventory_pool,
                    campaigns=shared.campaigns,
                    pacing_today=shared.pacing_today,
                    pacing_knobs=shared.pacing_knobs,
                    constraints=shared.constraints,
                    operator_channel=shared.operator_channel,
                    refine=True,
                    objective_mode="revenue_net",
                    optimize_fn=optimize_breaks,
                )
                plans[(channel, day)] = {
                    sp.segment_id: sp.num_breaks for sp in result.segments
                }
    finally:
        revenue_net_mod.segment_net_revenue = _ORIG_NET
    return plans


def evaluate_plan(shared: Shared, eval_world: World,
                  plans: dict[tuple[str, str], dict[str, int]]) -> dict:
    """Score a FIXED plan (counts) under one evaluation world at w = 1."""
    total_gross = total_cost = 0.0
    total_breaks = 0
    violations_total = 0
    floor_violations = 0
    per_day = []
    for (channel, day), counts in sorted(plans.items()):
        segments = eval_world.segments_for(channel, day)
        seg_index = {s.segment_id: s for s in segments}
        day_gross = day_cost = 0.0
        day_breaks = 0
        breaks = []
        for sid, k in counts.items():
            seg = seg_index.get(sid)
            if seg is None:
                continue
            day_gross += _segment_revenue(seg, k)
            day_cost += segment_retention_cost_ils(seg, k)
            day_breaks += k
            breaks.extend(_segment_break_objects(seg, k))
        vs = evaluate(breaks, shared.guardrails)
        violations_total += len(vs)
        floor_violations += sum(1 for v in vs if "retention" in v.code.lower())
        total_gross += day_gross
        total_cost += day_cost
        total_breaks += day_breaks
        per_day.append({
            "channel": channel, "day": day, "gross_ils": day_gross,
            "cost_ils": day_cost, "net_ils": day_gross - day_cost,
            "breaks": day_breaks, "violations": len(vs),
        })
    return {
        "gross_ils": total_gross, "cost_ils": total_cost,
        "net_ils": total_gross - total_cost, "total_breaks": total_breaks,
        "violations_total": violations_total,
        "retention_floor_violations": floor_violations,
        "per_day": per_day,
    }


def plan_similarity(a: dict, b: dict) -> tuple[float, int, int]:
    """Share of segment break-count decisions identical between two plans."""
    same = total = 0
    keys = set(a) | set(b)
    for key in keys:
        ca, cb = a.get(key, {}), b.get(key, {})
        sids = set(ca) | set(cb)
        for sid in sids:
            total += 1
            if ca.get(sid, 0) == cb.get(sid, 0):
                same += 1
    return (same / total if total else 1.0), same, total


def main() -> int:
    t0 = time.time()
    shared = Shared()
    world_baseline = World(shared, BASELINE_MULT)
    world_honest = World(shared, HONEST_MULT)

    arms = [("baseline_1.947_w1.00", world_baseline, 1.0)]
    arms += [(f"honest_1.690_w{w:.3f}", world_honest, w) for w in W_GRID]

    plans_by_arm = {}
    for name, world, w in arms:
        t = time.time()
        plans_by_arm[name] = optimize_arm(shared, world, w)
        n_days = len(plans_by_arm[name])
        print(f"optimized {name}: {n_days} channel-days in {time.time() - t:.1f}s")

    eval_worlds = {"1.947": world_baseline, "1.690": world_honest}
    baseline_plans = plans_by_arm[arms[0][0]]

    rows_by_world = {k: [] for k in eval_worlds}
    detail_rows = []
    baseline_net = {}
    for wkey, eworld in eval_worlds.items():
        base_eval = evaluate_plan(shared, eworld, baseline_plans)
        baseline_net[wkey] = base_eval["net_ils"]
    for name, world, w in arms:
        plans = plans_by_arm[name]
        sim, same, total = plan_similarity(plans, baseline_plans)
        for wkey, eworld in eval_worlds.items():
            ev = evaluate_plan(shared, eworld, plans)
            rows_by_world[wkey].append({
                "arm": name, "plan_multiplier": world.multiplier, "w": w,
                "eval_world_multiplier": wkey,
                "gross_ils": round(ev["gross_ils"], 2),
                "cost_ils": round(ev["cost_ils"], 2),
                "net_ils": round(ev["net_ils"], 2),
                "delta_net_vs_baseline_ils": round(ev["net_ils"] - baseline_net[wkey], 2),
                "total_breaks": ev["total_breaks"],
                "violations_total": ev["violations_total"],
                "retention_floor_violations": ev["retention_floor_violations"],
                "identical_decision_share_vs_baseline": round(sim, 6),
                "identical_decisions": same,
                "total_decisions": total,
            })
            for d in ev["per_day"]:
                detail_rows.append({
                    "arm": name, "w": w, "eval_world_multiplier": wkey, **{
                        k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in d.items()
                    },
                })

    for wkey, rows in rows_by_world.items():
        path = OUT_DIR / f"eval_matrix_world_{wkey.replace('.', '')}.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")

    detail_path = OUT_DIR / "per_day_detail.csv"
    with open(detail_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"wrote {detail_path}")

    print(f"total wall time {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
