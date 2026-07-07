# Convexity of retention cost in break length: verdict memo

Date: 2026-07-07. Lane: analysis/convexity. All numbers below come from
commands run in this task on the real Nov-2024 measurement (the shipped
break_effects pipeline, default config, after-window clip ON).

Reproduce:

    PYTHONPATH=/Users/home/Code/questo/meridian /Users/home/.venvs/meridian/bin/python analysis/convexity/prepare_data.py
    PYTHONPATH=/Users/home/Code/questo/meridian /Users/home/.venvs/meridian/bin/python analysis/convexity/fit_convexity.py

Outputs: prep_summary.json, breaks_measured.csv (2,532 measured breaks with
continuous length), instances.csv (1,476 fully measured programme instances),
results.json, fitted_shape.csv, raw_bin_means.csv, prep_run.log, fit_run.log.

## Verdict: regime (a), fixed cost per interruption plus a weak marginal
## per-minute cost. The data favor CONSOLIDATING breaks, not splitting.

Shedding is defined as shed = -log_effect (positive = audience lost relative
to the time-of-day baseline). All models control for first-break flag,
break position in programme, program class, daypart, and channel, with
cluster-robust SEs on 1,782 programme-instance clusters.

Three independent lines of evidence agree on the sign:

1. The fixed interruption cost a is positive and the length slope is small.
   Linear fit: a = 0.0503 (SE 0.0243, p = 0.038), slope = 0.0033 per minute
   (SE 0.0019, p = 0.080). At representative controls (Other class,
   afternoon, middle position, not first break, modal channel) a 2-minute
   break sheds 0.0569 log points and a 4-minute break sheds 0.0636. Doubling
   the length adds ~0.007; adding a second interruption adds ~0.05. The
   interruption itself is roughly 7x more expensive than the second pair of
   minutes inside it.

2. The shape is NOT superlinear. The quadratic length term is negative:
   len_sq = -0.00123 (SE 0.00062, p = 0.046), i.e. mildly CONCAVE, the
   opposite of the split-favoring regime (b). The flexible fits (natural
   cubic spline, 10 length bins) show the same picture: shed rises from
   ~0.03-0.06 at 1-2.5 minutes to ~0.06-0.09 at 4-8 minutes and then
   flattens (raw_bin_means.csv), nothing accelerating.

3. Decision delta, computed in-support (no extrapolation to length 0):
   delta = 2*s(2min) - s(4min) = extra shedding from splitting one 4-minute
   break into two 2-minute breaks, cluster bootstrap (B = 800, prog_key
   clusters), from results.json "decision":

   | model     | delta   | 95% CI            | frac positive |
   |-----------|---------|-------------------|---------------|
   | linear    | +0.0503 | [-0.002, +0.101]  | 0.970         |
   | quadratic | +0.0401 | [-0.012, +0.091]  | 0.948         |
   | spline    | +0.0191 | [-0.038, +0.082]  | 0.738         |
   | bins      | +0.0303 | [-0.053, +0.117]  | 0.763         |

   Every shape says splitting costs MORE audience; none says it saves. In
   audience terms, splitting a representative 4-minute break into two
   2-minute breaks costs roughly an additional 2 to 5 percent of the
   before-break audience (log points ~ percent at this scale).

4. Quasi-experimental cross-check (the recon's equal-total-minutes cells):
   among fully measured programme instances with identical (channel, class,
   daypart, rounded total ad minutes) but different split counts (50 cells,
   517 instances), one extra break at FIXED total minutes adds +0.0191 total
   shed, bootstrap CI [-0.089, +0.121], 63 percent positive. Underpowered on
   its own, but the sign agrees with the observational curve.

## Confidence

- Sign of the decision (consolidate over split): MODERATE. Parametric fits
  put ~95 percent of bootstrap mass on "splitting costs audience"; flexible
  shapes and the within-cell estimator put 63-76 percent. Nothing points the
  other way.
- Rejection of superlinearity (regime b): MODERATE-HIGH. The quadratic term
  is significantly negative (p = 0.046) and both nonparametric shapes are
  concave-to-flat above 2 minutes. There is no evidence for split-favoring
  convexity anywhere in the observed 0.2-12.8 minute range.
- Magnitude of the fixed cost: LOW-MODERATE. a = 0.05 is an extrapolation to
  length 0; the shortest observed bin (<= 45s, n = 112) sheds only 0.009
  (SEM 0.018), so the "fixed" cost is better read as "the cost of a standard
  interruption of a minute or more" rather than a literal step at zero
  length. The 2-vs-4-minute delta does not depend on this extrapolation.
- Overall explanatory power is thin, as expected for this signal: R2 is
  0.010-0.016 across models (the known ~0.008-0.01 retention-skill reality).
  The decision rides on a small but consistently signed effect.

## Confounding limits, stated honestly

- Schedulers did NOT randomize break lengths or split counts. Measured
  associations (results.json "selection"): length bucket vs channel Cramer's
  V = 0.496 (p ~ 1e-266), vs daypart V = 0.298 (p ~ 1e-92), vs first-break
  V = 0.111, vs position V = 0.077, vs program class V = 0.076. Instance
  split count vs channel V = 0.180, vs daypart V = 0.101, vs class V = 0.094,
  and split count tracks total ad minutes (Spearman rho = 0.413). Channel and
  daypart are the big sorters; both are IN the regression controls, and the
  within-cell estimator conditions on them exactly.
- Residual confounding remains at the specific-programme-title level: cells
  key on pricing class, not title. If channels split breaks precisely in the
  shows whose audiences tolerate interruptions best, the true splitting
  penalty is LARGER than estimated here (selection would bias the measured
  split cost downward), which strengthens, not weakens, the consolidate
  verdict; the reverse selection story (splitting only fragile shows) would
  bias it upward and cannot be excluded with one month of data.
- The split scenario sums two independent per-break log effects; any
  interaction between closely spaced breaks is not measured (the after-window
  clip drops contaminated windows rather than modeling recovery), so the
  compounding of two 2-minute breaks is an approximation.
- Single month (Nov 2024), 4 channels, aggregate TVR only; no viewer panel.
  Sub-45-second and beyond-13-minute lengths are out of support.
- first_break here means keyed ordinal == 1 (64.6 percent of measured breaks;
  the after-window clip preferentially drops mid-run neighbours, so isolated
  first breaks over-survive). This is a control, not the shipped gate's
  first-vs-later contrast.

## What this means for the optimizer

The current linear/bucketed retention cost is the WRONG SHAPE in one specific
way that matters: it prices two 2-minute breaks the same as (or cheaper than)
one 4-minute break, when the measured data say the two-break configuration
sheds ~0.02-0.05 more log-audience. A per-interruption fixed cost term
(order 0.03-0.05 log points at standard lengths, on top of a small ~0.003
per-minute slope) would steer the optimizer toward consolidation and is
supported by every specification tried here. Given the moderate confidence
and one-month window, ship it as a measured, owner-gated term (same
held-out-skill discipline as the other retention knobs), not as a hardcoded
constant.
