"""Adversarial re-verification of the loss-aversion sweep (12 eval days).

Independent re-implementation of the load-bearing computation:
  - own optimize loop (not sweep.optimize_arm), with a CALL COUNTER on the
    patched weighted net to prove w actually threads into the engine,
  - own strict evaluator: hard-fails if any plan segment_id is missing from
    the evaluation world (the original evaluate_plan silently skips those),
  - an EXTRA evaluation world at multiplier 1.0, the currently shipped
    honest belief (models/tv_break_coefficients.json: first_break_active
    false, multiplier 1.0), which the original analysis never scored under.

Arms re-run: baseline 1.947 w=1; honest 1.690 at w in {1.0, 1.05, 1.152, 2.25}.
Output: reverify_12day.csv in this directory.

Run:
  /Users/home/.venvs/meridian/bin/python analysis/loss-aversion/verify/reverify_12day.py
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
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

EVAL_DAYS = ("2024-11-01", "2024-11-02", "2024-11-03")
_ORIG_NET = revenue_net_mod.segment_net_revenue
CALLS = {"n": 0}


def weighted_net_factory(w: float):
    def weighted(segment, k: int) -> float:
        CALLS["n"] += 1
        if k <= 0:
            return 0.0
        return _segment_revenue(segment, k) - w * segment_retention_cost_ils(segment, k)
    return weighted


class World:
    def __init__(self, shared, multiplier: float):
        self.multiplier = multiplier
        self.assumptions = replace(OptimizerAssumptions(), first_break_multiplier=multiplier)
        self.impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=self.assumptions)
        self.shared = shared
        self._cache = {}

    def segments_for(self, channel: str, day: str):
        key = (channel, day)
        if key not in self._cache:
            self._cache[key] = build_segments_from_programmes(
                self.shared.programmes, self.shared.classifier, self.shared.pricing,
                assumptions=self.assumptions, impact_model=self.impact_model,
                channel=channel, day=day,
            )
        return self._cache[key]


class Shared:
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


def optimize_arm(shared, world, w: float, days=EVAL_DAYS):
    plans = {}
    CALLS["n"] = 0
    revenue_net_mod.segment_net_revenue = weighted_net_factory(w)
    try:
        for channel in shared.channels:
            for day in days:
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
                plans[(channel, day)] = {sp.segment_id: sp.num_breaks for sp in result.segments}
    finally:
        revenue_net_mod.segment_net_revenue = _ORIG_NET
    if CALLS["n"] <= 0:
        raise RuntimeError(f"patched net was never called for w={w}: threading broken")
    return plans, CALLS["n"]


def evaluate_strict(shared, eval_world, plans):
    """Score a fixed plan; hard-fail on any missing segment_id (no silent drops)."""
    gross = cost = 0.0
    breaks_total = violations = 0
    for (channel, day), counts in sorted(plans.items()):
        segments = eval_world.segments_for(channel, day)
        seg_index = {s.segment_id: s for s in segments}
        missing = [sid for sid in counts if sid not in seg_index]
        if missing:
            raise RuntimeError(f"{channel} {day}: {len(missing)} plan segments missing in eval world")
        day_breaks = []
        for sid, k in counts.items():
            seg = seg_index[sid]
            gross += _segment_revenue(seg, k)
            cost += segment_retention_cost_ils(seg, k)
            breaks_total += k
            day_breaks.extend(_segment_break_objects(seg, k))
        violations += len(evaluate(day_breaks, shared.guardrails))
    return {"gross": gross, "cost": cost, "net": gross - cost,
            "breaks": breaks_total, "violations": violations}


def similarity(a, b):
    same = total = 0
    for key in set(a) | set(b):
        ca, cb = a.get(key, {}), b.get(key, {})
        for sid in set(ca) | set(cb):
            total += 1
            same += int(ca.get(sid, 0) == cb.get(sid, 0))
    return same, total


def main() -> int:
    t0 = time.time()
    shared = Shared()
    w1947 = World(shared, 1.947)
    w1690 = World(shared, 1.690)
    w1000 = World(shared, 1.0)  # eval-only: the currently shipped honest belief

    arms = [
        ("baseline_1.947_w1.00", w1947, 1.0),
        ("honest_1.690_w1.00", w1690, 1.0),
        ("honest_1.690_w1.05", w1690, 1.05),
        ("honest_1.690_w1.152", w1690, 1.152),
        ("honest_1.690_w2.25", w1690, 2.25),
    ]
    plans_by_arm = {}
    for name, world, w in arms:
        t = time.time()
        plans_by_arm[name], ncalls = optimize_arm(shared, world, w)
        print(f"optimized {name}: {len(plans_by_arm[name])} channel-days, "
              f"{ncalls} weighted-net calls, {time.time() - t:.1f}s")

    baseline = plans_by_arm[arms[0][0]]
    eval_worlds = {"1.947": w1947, "1.690": w1690, "1.000": w1000}
    base_net = {k: evaluate_strict(shared, ew, baseline)["net"] for k, ew in eval_worlds.items()}

    rows = []
    for name, world, w in arms:
        plans = plans_by_arm[name]
        same, total = similarity(plans, baseline)
        for wkey, ew in eval_worlds.items():
            ev = evaluate_strict(shared, ew, plans)
            rows.append({
                "arm": name, "w": w, "eval_world": wkey,
                "gross_ils": round(ev["gross"], 2), "cost_ils": round(ev["cost"], 2),
                "net_ils": round(ev["net"], 2),
                "delta_vs_baseline": round(ev["net"] - base_net[wkey], 2),
                "breaks": ev["breaks"], "violations": ev["violations"],
                "identical": same, "decisions": total,
            })
            print(rows[-1])

    out = HERE / "reverify_12day.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}; wall {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
