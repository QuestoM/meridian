"""Whole-week NET-ILS measurement harness for the net-mode refiner build.

Builds the full weekly allocation in memory under a chosen ``objective_mode`` /
``refine`` combination and evaluates every channel-day on the ENGINE-EXACT truth
basis:

    revenue(seg, k) = kairos.optimize._segment_math._segment_revenue(seg, k)
    cost(seg, k)    = seg.cpp * seg.premium * seg.baseline_tvr
                      * (1 - _segment_retention(seg, k))
                      * (k * seg.break_length_seconds)
    net             = revenue - cost

Segments are rebuilt per channel-day with the CURRENT shipped impact model
(``load_impact_model`` on the posterior pkl, which prefers the measured
coefficients JSON), so the valuation matches what the optimizer decided with.

The allocation is produced by the SAME shared core the weekly export uses
(:func:`kairos.optimize.day_core._optimize_one_day`, the seam
``build_weekly_schedule`` loops over), so the counts are byte-identical to a real
recompute for the same ``objective_mode`` / ``refine``. Driving that core here
lets the harness set ``refine`` explicitly per configuration without touching the
export module. Demand / pacing / constraint resources are loaded once, exactly as
the export does, so every steer is the same identity no-op it is on disk today.

This is a measurement tool, not an assertion test; it prints the net-ILS total,
gross and break count for each configuration so the net-mode refiner's
whole-week impact can be read against the three truth-basis baselines.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data.loaders import load_programmes  # noqa: E402
from kairos.data.transform import build_segments_from_programmes  # noqa: E402
from kairos.export.schedule import (  # noqa: E402
    DEFAULT_IMPACT_MODEL_PATH,
    _build_classifier,
    _load_constraints,
)
from kairos.optimize._segment_math import _segment_retention, _segment_revenue  # noqa: E402
from kairos.optimize.day_core import _optimize_one_day  # noqa: E402
from kairos.optimize.optimizer import optimize_breaks  # noqa: E402
from kairos.optimize.pricing import (  # noqa: E402
    OptimizerAssumptions,
    pricing_from_settings,
)
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine  # noqa: E402
from kairos.optimize.inventory import load_inventory  # noqa: E402
from kairos.optimize.pacing import load_campaigns  # noqa: E402
from kairos.model.impact import load_impact_model  # noqa: E402
from kairos.service import (  # noqa: E402
    _apply_first_break_multiplier,
    _pacing_knobs_from_settings,
    guardrails_from_settings,
)

from kairos_api.core import _load_settings, _model_dump, _reference_today  # noqa: E402


def _truth_net(seg, k: int) -> tuple[float, float]:
    """(revenue, cost) in ILS on the engine-exact truth basis at k breaks."""
    revenue = _segment_revenue(seg, k)
    cost = (
        seg.cpp
        * seg.premium
        * seg.baseline_tvr
        * (1.0 - _segment_retention(seg, k))
        * (k * seg.break_length_seconds)
    )
    return revenue, cost


class _WeekBuilder:
    """Loads the shipped resources once and optimizes any channel-day on demand."""

    def __init__(self, settings_map: dict):
        self.settings_map = settings_map
        self.assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
        self.pricing = pricing_from_settings(settings_map, None)
        self.classifier = _build_classifier()
        self.impact_model = load_impact_model(
            DEFAULT_IMPACT_MODEL_PATH, assumptions=self.assumptions
        )
        self.guardrails = guardrails_from_settings(settings_map)
        self.weight = settings_map["revenue_weight"] / 100.0
        self.risk_lambda = settings_map["risk_lambda"]
        self.operator_channel = str(settings_map.get("operator_channel", "") or "")
        self.constraints = _load_constraints(None)
        self.demand_engine = AdvertiserRuleEngine.from_files()
        self.inventory_pool = load_inventory()
        self.campaigns = load_campaigns()
        self.pacing_today = _reference_today(_load_settings())
        self.pacing_knobs = _pacing_knobs_from_settings(settings_map)
        self.programmes = load_programmes()
        self.channels = sorted(set(self.programmes["Channel"].dropna().astype(str)))
        self.days = sorted(
            set(self.programmes["start_dt"].dropna().dt.strftime("%Y-%m-%d"))
        )

    def segments_for(self, channel: str, day: str) -> list:
        return build_segments_from_programmes(
            self.programmes, self.classifier, self.pricing,
            assumptions=self.assumptions, impact_model=self.impact_model,
            channel=channel, day=day,
        )

    def optimize_day(self, segments: list, *, objective_mode: str, refine: bool):
        return _optimize_one_day(
            segments,
            guardrails=self.guardrails,
            revenue_weight=self.weight,
            risk_lambda=self.risk_lambda,
            demand_engine=self.demand_engine,
            inventory_pool=self.inventory_pool,
            campaigns=self.campaigns,
            pacing_today=self.pacing_today,
            pacing_knobs=self.pacing_knobs,
            constraints=self.constraints,
            operator_channel=self.operator_channel,
            refine=refine,
            objective_mode=objective_mode,
            optimize_fn=optimize_breaks,
        )


def build_and_measure(objective_mode: str, refine: bool, builder: _WeekBuilder) -> dict:
    total_rev = 0.0
    total_cost = 0.0
    total_breaks = 0
    priced = 0
    for channel in builder.channels:
        for day in builder.days:
            segments = builder.segments_for(channel, day)
            if not segments:
                continue
            seg_index = {s.segment_id: s for s in segments}
            result = builder.optimize_day(
                segments, objective_mode=objective_mode, refine=refine
            )
            for plan in result.segments:
                seg = seg_index.get(plan.segment_id)
                if seg is None:
                    continue
                rev, cost = _truth_net(seg, plan.num_breaks)
                total_rev += rev
                total_cost += cost
                total_breaks += plan.num_breaks
                priced += 1
    return {
        "objective_mode": objective_mode,
        "refine": refine,
        "gross_ils": total_rev,
        "cost_ils": total_cost,
        "net_ils": total_rev - total_cost,
        "breaks": total_breaks,
        "priced_segments": priced,
    }


def _fmt(m: float) -> str:
    return f"{m / 1e6:.4f}M"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="revenue_net:false,revenue_net:true",
        help="comma list of mode:refine pairs to build and measure",
    )
    args = parser.parse_args()
    settings = _load_settings()
    builder = _WeekBuilder(_model_dump(settings))
    for spec in args.configs.split(","):
        mode, _, refine_txt = spec.partition(":")
        refine = refine_txt.lower() in ("1", "true", "yes", "on")
        r = build_and_measure(mode.strip(), refine, builder)
        print(
            f"mode={r['objective_mode']} refine={r['refine']}: "
            f"net={_fmt(r['net_ils'])} gross={_fmt(r['gross_ils'])} "
            f"cost={_fmt(r['cost_ils'])} breaks={r['breaks']} "
            f"priced_segments={r['priced_segments']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
