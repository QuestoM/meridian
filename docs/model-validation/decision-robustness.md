# Decision robustness of the retention model: does its uncertainty matter in the plan and the money?

**Review type:** decision-theory / operations-research referee review, everything computed on the
real repo engine and real data (no opinions without numbers).
**Date:** 2026-07-05. **Engine state:** git `79c2c768` with a concurrent, behavior-preserving
`kairos/model` edit in the working tree during the runs (replay fidelity re-verified before and
after it: identical plans, revenue to the cent).
**Scope:** the owned channel (`operator_channel` = עכשיו 14) on its representative day
**2024-11-04** — the channel-day whose shipped gross revenue is closest to the channel's median
across the saved schedule (deterministic pick, tie → earliest date): 53 segments, 74 breaks,
1,558,103 ILS gross, plus two annex regimes (peak day 2024-11-06, thin day 2024-11-29).
**Determinism:** all coefficient draws use `numpy.random.default_rng(42)` over the artifact's 36
cells in sorted-name order; every script re-runs identically.

Scripts: `scripts/validation/decision_uncertainty_lib.py` (harness),
`decision_sensitivity.py` (items 1–2), `model_value.py` (item 3),
`risk_lambda_efficacy.py` (item 4). Raw results: `docs/model-validation/results/*.json`.
Standing guard: `tests/validation/test_decision_robustness.py` (seeded, K=3, ~15 s, `realdata`).

## Method (exactly what was computed)

* **Uncertainty draws.** K=200 vectors over the 36 measured cells of
  `models/tv_break_coefficients.json`: Normal at each cell's coefficient with the CI-implied sd
  (95% interval width / 3.92; per-cell sd 0.0084–0.0105 on coefficients −0.048…−0.031), truncated
  (clipped) to the sign-plausible range [−1, 0] (clip probability ~Φ(−4), negligible). Cells are
  drawn independently; the EB posterior's shared grand mean would correlate them slightly, which
  could only *narrow* between-cell spread further.
* **Decision process replayed.** The shipped one, bit for bit:
  `kairos.service.run_scenario` (blend objective, saved settings: revenue_weight 60,
  retention floor 0.72, 4 breaks/hour, risk_lambda 0.0, refine=True), with the drawn coefficients
  injected in memory at the loader seam (`kairos.service.load_impact_model`) as a real
  `PosteriorImpactModel`. The artifact on disk and `output/weekly_break_schedule.csv` were never
  written. Replay at the shipped coefficients reproduces the saved CSV **exactly** (53/53 break
  counts; gross 1,558,103.12 vs CSV row-rounded 1,558,103.15), so "shipped plan" and "the process
  replayed today" coincide.
* **Money.** Plans are priced with the product's own machinery
  (`kairos.optimize._segment_math._segment_revenue` + `kairos.optimize.revenue_net.
  segment_retention_cost_ils`) — the same model that prices the shipped week at 215.34M ILS gross /
  16.82M retention cost / 198.52M net. A harness/service consistency assertion (repricing the
  service's own plan to the cent) ran on every draw.

**A wiring fact that frames everything:** segment construction reads the model at the segment's
pricing class averaged over the three position cells at the *standard* length bucket (every
programmes-path segment carries the default 120 s break). So only **12 of the 36 cells reach the
plan at all**, collapsed to **4 effective numbers**: News −0.0425, PrimeShow1 −0.0371,
PrimeShow2 −0.0386, Other −0.0377 (n-weighted pooled mean −0.0391). The between-class spread
(~0.005) is about the same size as the sd of each class mean (~0.005–0.006).

## 1. Decision sensitivity: the plan barely moves

K=200 re-optimizations under drawn coefficients, main scope (2024-11-04):

| quantity | result |
|---|---|
| total breaks | **74 in all 200 draws** (min = max = shipped) |
| plan identical to shipped | **197 / 200 draws (98.5%)** |
| Hamming distance (share of segments whose count changes) | mean 0.06%, P90 0.00%, **max 3.77%** (2 of 53 segments) |
| gross revenue at the draw (re-optimized) | mean 1,557,790; P10 1,542,475; P90 1,573,259 ILS |
| revenue-net at the draw (re-optimized) | mean 1,416,076; **P10 1,385,461; P90 1,446,460** ILS |
| revenue-net at the draw (shipped plan, fixed) | mean 1,416,117; P10 1,385,461; P90 1,447,062 ILS |

Annex regimes (K=50, same seed): peak day 2024-11-06 — breaks 73 in every draw, identical plan
68% of draws, Hamming max 6.7%; thin day 2024-11-29 — breaks 51 in every draw, identical plan
50% of draws, Hamming max 13.2%. **Break volume never moved by a single break on any day in any
draw**; only within-day allocation wiggles at the margin. The model card's honest headline (per-cell
discrimination ≈ 0 out of sample, R² ±0.008) is therefore **decision-irrelevant where it counts**:
the thin cells cannot steer the plan because the blend objective and the guardrails pin it.

The P10–P90 band of revenue-net (±31k ILS/day, ±2.2%) is **valuation** uncertainty, not decision
movement: it is the same 74-break plan repriced. Equivalently, the shipped plan's retention cost
(point 141,385 ILS/day) reprices to P10 126,213 / P90 157,013 across draws (**−11% / +11%**). The
weekly 16.8M ILS retention-cost figure should be read with that error bar; the plan behind it
should not.

## 2. Regret under misspecification: THE number is zero

Regret = net(per-draw re-optimized plan) − net(shipped plan), both priced at the drawn truth.

| scope | mean | median | P90 | max | zero-regret draws |
|---|---|---|---|---|---|
| **2024-11-04 (representative), K=200** | **−41 ILS/day** | 0 | **0** | **0.00** | 98.5% |
| 2024-11-06 (peak), K=50 | +314 ILS/day | 0 | 4,282 | 4,981 | 68% |
| 2024-11-29 (thin), K=50 | −36 ILS/day | 0 | 1,544 | 2,576 | 50% |

Extrapolated to a week (×7, owned channel): representative day **−288 ILS mean / 0 P90**; worst
observed regime (peak day) 2,198 ILS mean / 29,974 ILS P90 — i.e. **≤ 0.22% of that day's net even
at P90**. On the representative day the maximum regret over 200 draws is exactly 0.00 ILS: *no
plausible coefficient vector exists at which knowing the truth would have made the process more
than zero shekels.* Note the negative means: when the plan does move, the blend-objective process
re-optimized under the drawn truth sometimes lands slightly *net-worse* than the shipped plan
(1.5% / 12% / 32% of draws per scope) — the binding decision-maker is the blend objective and the
guardrails, not the coefficients. Better coefficient precision cannot fix what the objective does
not optimize; the already-built opt-in `objective_mode='revenue_net'` is the lever for that.

**Model uncertainty costs the current decision process ≈ 0 ILS/day (upper bound across regimes:
~5k ILS/day at P90 on the peak day, ~0.2% of net).**

## 3. Value of the model: the constant does all the work

Three retention models drive the same process; all three plans priced at the SHIPPED coefficients
as reference truth (2024-11-04):

| model | breaks | gross ILS | retention cost | net ILS | plan vs shipped |
|---|---|---|---|---|---|
| (a) shipped 36-cell EB | 74 | 1,558,103 | 141,385 | 1,416,718 | identical |
| (b) one global constant (−0.0391) | 74 | 1,558,103 | 141,385 | **1,416,718** | **identical (0 of 53 segments)** |
| (c) zero retention cost | 74 | 1,553,346 | 148,554 | **1,404,792** | 6 segments differ (11.3%) |

* **36-cell structure over a global constant: +0.00 ILS/day.** The plans are *identical*. This
  computes, at the decision level, exactly what the model card concedes at the statistical level:
  today the 36-cell model *is* the pooled constant with error bars.
* **Any retention model over none: +11,926 ILS/day (+0.84% of net), ≈ +83,485 ILS/week** on the
  owned channel. Volume is guardrail-pinned either way (74 breaks in both plans); the model's
  entire value is *where* the breaks go (6 reallocations). Real, but modest.

## 4. risk_lambda: decision-inert where it counts

Plans computed under the shipped artifact (its real credible intervals) at λ = 0 / 0.5 / 1.0
(λ replaces each coefficient with `conservative_impact(point, ci_low, ci_high, λ)` before
allocation — at λ=1 the whole plan decides at the worst credible bound, ≈ −0.058 per class):

| λ | breaks | plan vs λ=0 | net @ point | mean net over draws | P10 | P90 |
|---|---|---|---|---|---|---|
| 0.0 | 74 | — | 1,416,718 | 1,416,117 | 1,385,461 | 1,447,062 |
| 0.5 | 74 | **identical** | 1,416,718 | 1,416,117 | 1,385,461 | 1,447,062 |
| 1.0 | 74 | **identical** | 1,416,718 | 1,416,117 | 1,385,461 | 1,447,062 |

**The three plans are the same plan.** Even full worst-case pessimism does not move one break, so
risk_lambda currently provides zero tail protection and costs zero mean — it only changes the
*reported* objective (0.6317 → 0.6219) and the `retention_cost_used` disclosure columns. At current
CI widths and the blend weights, the knob is honest bookkeeping, not a decision lever. (P10
improvement from raising λ: **+0 ILS**.)

## Verdict

1. **Is current model uncertainty decision-material?** **No — quantified.** Break volume is
   invariant across 300 seeded re-optimizations spanning three day regimes; plan identity holds in
   98.5% of draws on the representative day; regret is bounded by 0 there and by ~0.2% of net at
   P90 in the least stable regime. The thin per-cell R² is decision-irrelevant.
2. **Where uncertainty IS material:** the *valuation*. The same plan's retention cost carries a
   ±11% (P10–P90) band, so the 16.8M ILS/week "retention cost" headline is a modeled figure with a
   roughly ±1.8M ILS/week error bar from coefficient uncertainty alone. Report it that way.
3. **What the model is worth:** having *a* measured pooled level beats ignoring retention by
   +11.9k ILS/day (+0.84% net) on the representative day; the 36-cell granularity adds +0.00 on top
   today.

## Ranked recommendations

1. **Keep the 36-cell EB model, and say what it is.** It decides identically to the pooled
   constant, so it costs nothing, the EB shrinkage + held-out gates protect it as data grows (the
   scale-readiness audit projects real per-cell discrimination at 12–24 months), and ripping it out
   buys nothing. Add one line to the model card: *"as of 2026-07, per-cell structure changes zero
   decisions vs the pooled constant (see decision-robustness review)."*
2. **Leave `risk_lambda` at 0.0 and stop presenting it as tail protection.** It is measured
   decision-inert at λ up to 1.0 on the owned channel; its honest role today is the
   `retention_cost_used` disclosure. If robustness-to-worse-costs is ever wanted for real, it must
   bind through guardrails (retention floor / breaks-per-hour), which do move plans.
3. **Do not fund coefficient-precision work for decision reasons.** The value of perfect
   coefficient information to the current process is ≈ 0 ILS/day (≤ ~5k/day P90, peak regime).
   The after-window de-bias and any future re-measurement are justified as *valuation honesty*
   (the ±11% cost band and the +52% level shift already documented), not as plan improvements.
4. **If net shekels are the goal, the lever is the objective, not the model.** In 1.5–32% of draws
   (by regime) the blend process picks a slightly net-worse plan than the incumbent even under
   perfect knowledge. The built, opt-in `objective_mode='revenue_net'` addresses exactly this;
   evaluating it owner-side is worth more than any coefficient refinement (bounded by the +0.84%
   model-vs-none gap on this scope).
5. **Keep the standing test green** (`tests/validation/test_decision_robustness.py`): it pins the
   replay-equals-CSV premise, the seeded draw determinism, and the ±10% repricing sanity band; a
   legitimate settings/coefficients change that trips it means this review's numbers are stale —
   re-run the three scripts (~2 minutes total at K=200).

## Reproduce

```
/Users/home/.venvs/meridian/bin/python scripts/validation/decision_sensitivity.py --k 200 --seed 42   # ~70 s
/Users/home/.venvs/meridian/bin/python scripts/validation/model_value.py                              # ~10 s
/Users/home/.venvs/meridian/bin/python scripts/validation/risk_lambda_efficacy.py --k 200 --seed 42   # ~10 s
/Users/home/.venvs/meridian/bin/python -m pytest tests/validation/test_decision_robustness.py -q      # ~15 s
```

Caveats owned honestly: one owned channel, three day regimes (300 re-optimizations total), draws
independent across cells per the review spec, the default coefficient for unmeasured cells held
fixed (no lookup on this scope ever fell back to it — asserted), and regret is measured through the
shipped blend process (the decision-relevant question), not against a hypothetical net-optimal
oracle.
