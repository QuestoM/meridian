# Causal identification review: Kairos retention-cost model

Referee-style review of the per-break retention effect shipped in
`models/tv_break_coefficients.json` (pooled delta -0.0391, 2,532 breaks,
November 2024). Every number in the managed sections below is computed from
the real reference data by the seeded, re-runnable scripts named in each
section (`scripts/validation/`). Regenerating a script rewrites only its own
section. Reviewed 2026-07-05; the concurrent `kairos/model` edits (seasonal
baseline gate) were verified behavior-preserving before and after (pooled mu
identical to 1e-12). A fast standing version of the placebo check lives at
`tests/validation/test_placebo_fast.py` (realdata marker, ~12 s).

<!-- BEGIN:placebo -->
## 1. Placebo / negative control (`scripts/validation/run_placebo.py`, seed 42)

Real pooled effect reproduced from the current data: mu = -0.03984 (log), delta = -0.03906 (shipped: -0.0391). Pseudo-breaks were sampled at minutes with no detected break, windows fully inside the programme and clear of every detected break span, then measured with the exact shipped arithmetic (same window means, same detrend curve, same drop rules).

| design | n | clusters | placebo mean (delta) | 95% cluster CI (delta) | z vs 0 | share of real effect |
|---|---|---|---|---|---|---|
| matched | 6141 | 121 | +0.01506 | [+0.00810, +0.02165] | +4.12 | -37.5% |
| matched-strict | 6106 | 121 | +0.01506 | [+0.00742, +0.02249] | +3.86 | -37.5% |
| uniform-EPG | 2882 | 121 | +0.01351 | [+0.00590, +0.02220] | +3.25 | -33.7% |
| uniform, breakless programmes only | 885 | 120 | +0.01584 | [+0.00162, +0.03053] | +2.18 | -39.4% |

Matched placebo (primary): mean pseudo effect +0.01506 explains -37.5% of the shipped -0.0391.

**Placebo-corrected real effect** (real mean minus matched placebo mean, joint channel-day bootstrap): delta = -0.05331, 95% CI [-0.06459, -0.04260]. Implied multiplicative correction to every shipped coefficient: x1.365.

Placebo mean by genre cell dimension (matched design, log scale): News +0.01311 (n=1399); Other +0.01466 (n=4132); PrimeShow1 +0.00959 (n=342); PrimeShow2 +0.03572 (n=268). By pseudo position: first +0.01154; last +0.01193; middle +0.02238.

Runtime 9s; fully deterministic (default_rng(42)).
<!-- END:placebo -->

<!-- BEGIN:selection -->
## 2. Selection-on-placement bias (`scripts/validation/run_selection_bias.py`, seed 42)

Pre-anchor audience trajectories, real break starts (n=2259 unclipped-window breaks) vs eligible non-break minutes in the same programmes (n=6141 matched pseudo minutes):

| metric | real mean | pseudo mean | standardized diff |
|---|---|---|---|
| excess level, 3-min before window (log obs/base) | -0.06705 | -0.08635 | +0.037 |
| pre-anchor slope of excess log TVR (per minute, 10 min) | +0.00178 | +0.00097 | +0.022 |
| pre-anchor volatility (std of 1-min excess changes) | +0.08853 | +0.09354 | -0.052 |
| raw TVR level, 3-min before window | +3.84087 | +3.80885 | +0.007 |
| in-break density on other days, after minus before window | +0.08513 | +0.00981 | +0.460 |

Mean-reversion exposure: on pseudo breaks (machinery only, no ad aired) the fitted slope of log_effect on excess_before is -0.1018 (95% cluster CI [-0.1288, -0.0760]); on real breaks -0.0872 [-0.1136, -0.0624]. The real-vs-pseudo gap in excess_before is +0.01930, so the implied placement-selection bias on the pooled log effect is -0.00196 (cost overstated by that amount; compare pooled -0.0398 log).

Per-genre placement gap (excess_before, real minus pseudo): News +0.02628 (d=+0.08); Other +0.01523 (d=+0.03); PrimeShow1 +0.08439 (d=+0.15); PrimeShow2 +0.05781 (d=+0.13). The gap has the same sign in every genre cell, so within-cell pooling does NOT absorb it; it is a placement-timing effect inside cells, orthogonal to the genre x position x length cell structure.

Runtime 10s; deterministic (default_rng(42)).
<!-- END:selection -->

<!-- BEGIN:inference -->
## 3. Randomization inference and clustering (`scripts/validation/run_inference.py`, seed 42)

**Permutation test** (labels permuted within programme strata, 1349 strata, 2,000 permutations): observed real-vs-pseudo gap in mean log effect -0.05032, permutation sd 0.00571, two-sided p = 0.0004998. The break effect is not an artifact of which minutes got measured: no label reassignment within shows comes close to the observed gap.

**Cluster (channel-day) bootstrap** of the full DL/EB pipeline, 1,000 draws over 120 channel-day blocks:

| quantity | naive (shipped assumption) | cluster-robust | inflation |
|---|---|---|---|
| pooled-mean SE (log) | 0.00479 | 0.00488 | x1.02 |
| pooled 95% CI (delta) | [-0.04805, -0.02999] | [-0.04812, -0.02984] | x1.02 width |

Per-cell, comparing each shipped credible-interval halfwidth with the cluster-bootstrap sampling halfwidth of the same EB estimator (median over 36 cells): inflation x1.05 (min x0.53 PrimeShow1_last_long, max x2.29 Other_last_standard). These are different objects (posterior credible vs frequentist sampling), but the gap measures how much uncertainty the independence assumption hides from the operator-facing high/medium/low confidence labels.

Runtime 10s; deterministic (default_rng(42)).
<!-- END:inference -->

<!-- BEGIN:loo -->
## 4. Leave-one-out stability (`scripts/validation/run_leave_one_out.py`, deterministic)

Full-sample pooled delta -0.03906 (n=2532). Channel and week drops re-run the ENTIRE measurement pipeline (break detection, clipping, detrend baseline) on the reduced data; genre drops filter the measured effects.

| unit dropped | n breaks | pooled delta | shift |
|---|---|---|---|
| channel: כאן 11 | 1675 | -0.03147 | +0.00759 |
| channel: עכשיו 14 | 2271 | -0.04029 | -0.00123 |
| channel: קשת 12 | 1795 | -0.04493 | -0.00587 |
| channel: רשת 13 | 1855 | -0.03868 | +0.00038 |
| week: ISO W44 | 2257 | -0.03975 | -0.00070 |
| week: ISO W45 | 1873 | -0.03292 | +0.00614 |
| week: ISO W46 | 1927 | -0.03981 | -0.00075 |
| week: ISO W47 | 1949 | -0.04075 | -0.00169 |
| week: ISO W48 | 2084 | -0.04137 | -0.00231 |
| genre: News | 2043 | -0.03530 | +0.00376 |
| genre: Other | 777 | -0.04436 | -0.00530 |
| genre: PrimeShow1 | 2382 | -0.04098 | -0.00192 |
| genre: PrimeShow2 | 2394 | -0.03863 | +0.00043 |

Channel-day jackknife (120 units): pooled delta range [-0.04023, -0.03724]; the single most influential channel-day is כאן 11|2024-11-10 (shift +0.00182 = 4.7% of the pooled effect).

Maximum single-unit influence: dropping channel **כאן 11** moves the pooled effect by +0.00759 = **19.4%** of its value. Every leave-one-out estimate stays within [-0.04493, -0.03147]; the pooled cost is not driven by any single channel, week, genre or channel-day.

Runtime 21s; no randomness.
<!-- END:loo -->

<!-- BEGIN:cleanbase -->
## 5. Fix preview: content-only detrend baseline (`scripts/validation/run_clean_baseline.py`, seed 42)

The shipped baseline averages break minutes into the 'typical' curve: 12.1% of channel-minutes lie inside a detected break, and audience during those minutes runs -9.2% vs the shipped baseline at the same broadcast minute. Rebuilding the baseline from content-only minutes and re-running the ENTIRE shipped pipeline (runtime rebind of `_baseline_levels`; no source edit):

| baseline | pooled delta | matched placebo mean | placebo-corrected delta | 95% CI (joint cluster bootstrap) |
|---|---|---|---|---|
| shipped (ad minutes included) | -0.03906 | +0.01506 | -0.05331 | [-0.06459, -0.04260] |
| clean, breaks-only excluded | -0.03627 | +0.01393 | -0.04951 | [-0.06168, -0.03755] |
| clean, all ad airtime excluded | -0.03595 | +0.01433 | -0.04957 | [-0.06166, -0.03757] |

Moving to the content-only baseline shifts the raw pooled effect from -0.03906 to -0.03595 and moves each of the 36 shipped cell coefficients by +0.00311 on average (largest single-cell move +0.00671). The residual placebo mean under the clean baseline (+0.01433) is within-show audience drift that the baseline cannot and should not absorb; the honest per-break cost is the placebo-corrected -0.04957.

Runtime 13s; deterministic (default_rng(42)).
<!-- END:cleanbase -->

<!-- BEGIN:verdict -->
## 6. Referee verdict and ranked fixes

### What survives review

* **The cost is real.** Permuting real/pseudo labels within 1,349 programme
  strata never approaches the observed contrast (gap -0.05032 log, permutation
  sd 0.00571, two-sided p = 0.0005, the resolution floor of 2,000
  permutations). Every leave-one-out estimate (channel, week, genre,
  channel-day) stays inside [-0.0449, -0.0315]: always negative, always
  material. The result is not one show, one week, or one broadcaster.
* **The machinery manufactures no spurious cost.** The negative control comes
  out POSITIVE (+0.0151 delta, z = +4.12 vs zero), not negative: the
  detrend/window pipeline does not fabricate shedding at no-break minutes.
  The prior concern that the -0.0391 might be partly a machinery artifact is
  refuted by direct computation, in all four placebo designs.
* **Selection-on-placement is small and bounded.** Pre-break trajectories are
  balanced between real break starts and eligible in-show minutes
  (standardized differences: level +0.007, slope +0.022, volatility -0.052,
  excess level +0.037; all far under the 0.10 imbalance convention).
  Schedulers do place breaks at slightly hot moments (excess_before gap
  +0.0193), which via the measured mean-reversion slope (-0.102 on pseudo
  breaks) overstates the cost by only -0.0020 log, about 5% of the pooled
  effect, direction: shipped cost slightly too big through this channel.
* **The pooled interval is honest.** Channel-day cluster bootstrap inflates
  the pooled SE by x1.02 over the shipped independence assumption; the naive
  CI [-0.0481, -0.0300] is effectively correct.

### What is biased, and by how much

* **The headline bias runs the OTHER way from the referee's suspicion: the
  shipped -0.0391 UNDERSTATES the causal per-break cost by roughly a
  quarter to a third.** The shipped estimator's implicit null is "hold your
  level relative to the daily curve", but the measured no-break counterfactual
  at eligible in-show minutes is +0.0151 (audience builds within shows at
  about +0.0013 log per minute of gap; monotone in gap length; robust to
  equal-weighting per source break, +0.0155, and present in breakless
  programmes, +0.0158). Against that counterfactual the causal cost is
  -0.0533 [-0.0646, -0.0426] under the shipped baseline, -0.0496 [-0.0617,
  -0.0376] under a content-only baseline, and -0.0491 as the stratified
  within-programme contrast. Call it **-0.049 to -0.053**: the optimizer is
  under-charging every break by x1.27-1.37.
* **The detrend baseline is contaminated by ad airtime.** 12.1% of
  channel-minutes sit inside a detected break at -9.2% audience, and the
  contamination is spatially structured (in-break density at real
  after-windows exceeds before-windows by +0.085, std diff +0.46, because
  breaks recur at similar clock minutes across days). Direct recompute shows
  the net effect on the raw pooled coefficient is -0.0031 (shipped raw cost
  slightly too big), while the placebo-corrected causal estimate barely moves
  (-0.0533 to -0.0496): the two machinery biases largely cancel inside the
  corrected estimator, and neither is visible without it.
* **A few cell intervals hide clustering.** Median shipped-vs-cluster
  halfwidth inflation is x1.05, but the tail reaches x2.29
  (Other_last_standard); the operator-facing confidence labels are too
  confident there (and over-cautious at x0.53, PrimeShow1_last_long).
* **Concentration risk, quantified, not fatal:** channel "כאן 11" carries 34%
  of measured breaks and dropping it moves the pooled effect by 19.4%
  (to -0.0315); dropping ISO week W45 moves it 15.7%. One month, four
  channels remains the binding limitation the model card already declares.

### Ranked fixes (each with the computed expected effect on the shipped coefficients)

1. **Charge the placebo-corrected cost.** Estimate each cell's coefficient as
   (real cell effect) minus (matched no-break drift), i.e. build the matched
   pseudo-break sample into the rebuild and subtract per cell, exactly as
   computed here. Expected effect: pooled -0.0391 -> -0.053 (x1.365); per-cell
   drift to subtract is +0.0131 (News), +0.0147 (Other), +0.0096 (PrimeShow1),
   +0.0357 (PrimeShow2) in log units. This is the load-bearing fix; gate it
   like every optional layer (held-out skill plus measured revenue movement)
   before flipping the optimizer's charge.
2. **Content-only detrend baseline** (exclude all ad airtime from
   `_baseline_levels`). Expected effect standalone: every cell moves +0.0031
   toward zero on average (max +0.0067), pooled -0.0391 -> -0.0360. WARNING:
   standalone adoption moves coefficients AWAY from the causal -0.049 to
   -0.053; ship it only together with fix 1 (combined: -0.0496). Its value is
   interpretability (the "typical curve" stops encoding the break schedule),
   not bias reduction.
3. **Mean-reversion adjustment for placement selection.** Regression-adjust
   the per-break effect for excess_before (or subtract the measured
   b x gap = -0.0020 log at the pooled level). Expected effect: pooled cost
   shrinks by about 5% of itself; second-order next to fix 1, cheap, and
   removes the one true selection channel this review found.
4. **Cluster-robust cell intervals at rebuild time.** Channel-day bootstrap
   (1,000 draws, seconds of compute) feeding the high/medium/low confidence
   labels. Expected effect: no coefficient moves; the ~x2.3 understated cells
   (e.g. Other_last_standard) drop a confidence grade, honest labels
   elsewhere.
5. **Data collection, not code:** the two-year window already planned is the
   real cure for the 19.4% single-channel influence and the W45 sensitivity;
   re-run this whole review (scripts are seeded and re-runnable) when it
   lands, and keep `tests/validation/test_placebo_fast.py` green in between.

### One-line verdict

The shipped per-break retention cost is genuine, stable, honestly pooled and
conservatively small: against a computed no-break counterfactual the true
cost is -0.049 to -0.053, not -0.0391, so the model under-prices audience
shedding by roughly 25-35% and the model card's "measured causally" claim
should say "measured against the daily-curve null; causal counterfactual
correction pending (fix 1)".
<!-- END:verdict -->
