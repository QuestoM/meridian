"""Bare optimize_breaks A/B over all 120 channel-days (no export config), to
corroborate the orchestrator's 209.14M->221.87M and explain the 3 baselines."""
from __future__ import annotations

from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.model.impact import load_impact_model
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
from kairos.service import (
    DEFAULT_IMPACT_MODEL_PATH, _apply_first_break_multiplier, _build_classifier,
)

GR = Guardrails()
assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
pricing = pricing_from_settings(None, None)
classifier = _build_classifier()
impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
programmes = load_programmes()

chans = sorted(programmes["Channel"].dropna().unique().tolist())
programmes = programmes.copy()
programmes["_day"] = programmes["start_dt"].dt.strftime("%Y-%m-%d")
days = sorted(programmes["_day"].dropna().unique().tolist())

off_total = on_total = 0.0
obj_off = obj_on = 0.0
regress = 0
rose = 0
equal = 0
noncompliant = 0
ndays = 0
for ch in chans:
    for d in days:
        segs = build_segments_from_programmes(
            programmes, classifier, pricing, assumptions=assumptions,
            impact_model=impact_model, channel=ch, day=d)
        if not segs:
            continue
        ndays += 1
        off = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                              refine=True, dp_refine=False)
        on = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                             refine=True, dp_refine=True)
        off_total += off.total_revenue
        on_total += on.total_revenue
        obj_off += off.objective
        obj_on += on.objective
        if on.objective < off.objective - 1e-9:
            regress += 1
        elif on.objective > off.objective + 1e-9:
            rose += 1
        else:
            equal += 1
        if on.violations:
            noncompliant += 1

print(f"channel-days optimized: {ndays}")
print(f"BARE off total_revenue = {off_total:.2f}")
print(f"BARE on  total_revenue = {on_total:.2f}")
print(f"BARE revenue delta = {on_total-off_total:.2f} ({100*(on_total-off_total)/off_total:.4f}%)")
print(f"objective: rose={rose} equal={equal} regressed={regress}")
print(f"sum objective off={obj_off:.4f} on={obj_on:.4f}")
print(f"noncompliant on plans: {noncompliant}")
