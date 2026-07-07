"""Measure wall time to optimize one full real channel-day: greedy vs greedy+F1.

Builds the real segments for the operator's owned channel on its busiest real
broadcast day (most segments), then times optimize_breaks with refine=False
(pure greedy) and refine=True (greedy plus the F1 tiered refiner). Prints
segment count, break counts, objective, and per-mode wall time. No fabrication:
every number comes from this run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.export.schedule import DEFAULT_IMPACT_MODEL_PATH, _build_classifier
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
from kairos.model.impact import load_impact_model
from kairos.service import _apply_first_break_multiplier, guardrails_from_settings
from kairos_api.core import _load_settings


def main() -> int:
    settings = _load_settings()
    smap = settings if isinstance(settings, dict) else settings.model_dump()
    owned = str(smap.get("operator_channel", "") or "")
    weight = smap.get("revenue_weight", 50) / 100.0
    risk = smap.get("risk_lambda", 0.0)

    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    pricing = pricing_from_settings(smap, None)
    classifier = _build_classifier()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    guardrails = guardrails_from_settings(smap)

    programmes = load_programmes()
    days = sorted(set(programmes["start_dt"].dropna().dt.strftime("%Y-%m-%d")))

    # Pick the owned channel's busiest real day (most segments) as the full-day case.
    best_day = None
    best_segments: list = []
    for day in days:
        segs = build_segments_from_programmes(
            programmes, classifier, pricing,
            assumptions=assumptions, impact_model=impact_model,
            channel=owned, day=day,
        )
        if len(segs) > len(best_segments):
            best_segments, best_day = segs, day

    n = len(best_segments)
    print(f"owned_channel={owned!r} busiest_day={best_day} segments={n} "
          f"revenue_weight={weight} risk_lambda={risk}")

    def run(refine: bool):
        t0 = time.perf_counter()
        res = optimize_breaks(
            best_segments, guardrails,
            revenue_weight=weight, risk_lambda=risk, refine=refine,
        )
        dt = time.perf_counter() - t0
        return dt, res

    # One warm run to pay import/JIT-free but function-cache costs, then measure.
    for label, refine in (("greedy", False), ("greedy+F1", True)):
        dt, res = run(refine)
        print(f"{label:10s} wall={dt:8.3f}s breaks={res.total_breaks:4d} "
              f"objective={res.objective:.6f} revenue={res.total_revenue:,.0f} "
              f"retention={res.aggregate_retention:.4f} compliant={res.is_compliant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
