# Kairos retention model card

The document to read before trusting the model. Every number below is measured
on the real reference data (November 2024, one month, four channels) with the
scripts named next to it. Nothing here is a target or an opinion. Last
measured: 2026-07-17.

## What the model measures

For every real commercial break in the aired-spots log, the model measures how
much audience the break itself shed: the mean minute-level TVR in the 3
minutes before the break versus the 3 minutes after it, DIVIDED by the ratio
the channel's typical daily curve would produce at those same broadcast
minutes (detrending, so prime-time growth is not credited to breaks), with the
windows CLIPPED so they never overlap an adjacent break's air time
(`kairos/model/measure.py::break_effects`). Per-break effects are pooled into
36 genre cells (pricing class x break position x length bucket) by an
empirical-Bayes hierarchical model whose shrinkage strength is learned from
the data (DerSimonian-Laird tau squared), not hand-set. The result is the
per-break retention delta the optimizer charges when it places a break.

## The numbers that drive the plan today

From the shipped `models/tv_break_coefficients.json` (placebo-corrected,
content-only baseline, fingerprint-fresh against the data on disk, verified
2026-07-17):

* 2,532 breaks measured (5,696 raw; 3,164 dropped by the adjacent-break
  window clip rather than measured on contaminated audience).
* 36 cells, ALL negative. Mean per-break retention delta -0.0499; range
  -0.0617 (`Other_first_long`, n=238) to -0.0413 (`Other_last_short`, n=146).
* Cell sample sizes: min 8, median 24, max 292. Confidence labels: 1 high,
  25 medium, 10 low (the placebo correction widens the intervals, so most
  cells sit at medium rather than high on one month).
* Pooling: tau squared 1.729e-04, pooled within-variance 0.0651, learned
  pseudo-count 376.4. At the mean cell size (n about 70) a cell keeps only
  about 16 percent of its own mean and takes 84 percent from the global mean;
  at the median cell (n 24) it keeps about 6 percent. The cells genuinely
  differ by little on one month, and the model says so rather than inventing
  spread.

## Honest skill statement (read this before selling the model)

* The pooled LEVEL is solid: the average break costs about 5.0 percent of the
  audience multiplier (pooled -0.0499), measured causally (detrended, clipped,
  then placebo-corrected). Two corrections built that level from the earlier
  contaminated measurement: the adjacent-break window clip moved the pooled
  cost from -0.0257 to -0.0391 (+52 percent, see de-bias below), and the
  content-only placebo correction then subtracts each genre's measured no-break
  drift (on the content-only raw of -0.0356, a x1.40 correction) to reach the
  shipped pooled -0.0499 (see the placebo section). That level is what the
  optimizer's revenue-vs-retention tradeoff runs on.
* The per-cell DISCRIMINATION is thin out of sample: predicting held-out
  breaks with per-cell means beats a single global constant by approximately
  nothing on one month (own-split out-of-sample R squared -0.008 on the
  clipped pipeline; prior audit measured +0.008 on the pre-clip pipeline;
  both are zero for practical purposes;
  `scripts/analyze_afterwindow_bias.py`). The EB shrinkage is what keeps
  those thin cells honest. Expect real discrimination only as data grows
  (trajectory below).

## The after-window de-bias (what changed and why you can trust it)

32.6 percent of pre-fix after-windows overlapped the next break (60.3 percent
of measurements were touched when counting both windows and the drop rule).
Contamination biased measured effects TOWARD ZERO (both the observed and the
expected window absorbed the neighbour's dip), understating the per-break cost:
pooled delta -0.0257 contaminated vs -0.0391 clean. It also fabricated a
first-break multiplier of 1.947 (p=8.8e-08) that vanished once windows were
clipped (p=0.203). The clip lives at the single window-derivation chokepoint
in `break_effects`; the clean post-clip recompute is
`models/candidates/tv_break_coefficients_afterwindow.json` (pooled -0.0391).
The shipped artifact is that clean base with the placebo correction layered on
top (each genre's no-break drift subtracted), so it equals the after-window
recompute plus a per-genre placebo shift of mean -0.0108 (shipped pooled
-0.0499); the placebo section below is the authority on that step.

Residual: 9.44 percent of surviving breaks still have a window minute
overlapping a SINGLE-SPOT ad run (2.39 percent of window minutes; single
spots are not detected as breaks). An opt-in spot-level clip
(`clip_to_all_ad_airtime=True`) removes this to a measured 0.00 percent, but
it shifts coefficients by only -0.00095 pooled and does NOT improve held-out
skill (-0.38 percent, within noise), so it ships OFF
(`scripts/measure_spotlevel_clip.py`). Also measured and left alone: 4.27
percent of after-windows cross a programme START (a content junction, not an
ad; a candidate covariate for the two-year build).

## Optional layers: what is ON and OFF, and the gate values

Every optional layer re-evaluates its own held-out gate at each coefficient
rebuild (`scripts/compute_measured_coefficients.py`) and self-activates only
by measurement. Today:

| layer | state | gate value (real month) |
|---|---|---|
| 36 genre-cell coefficients | ON | the shipped measurement itself |
| EB learned pooling | ON | tau2 1.729e-04, learned pseudo-count 376.4 |
| Content-only placebo correction | ON | each genre's no-break drift subtracted (pooled +0.01422 over 6141 matched pseudo-breaks, se 0.00388); corrected pooled -0.0499 |
| Series/title layer | OFF | held-out RMSE 0.2624 vs genre 0.2420: 8.5 percent WORSE; gate needs 2 percent better |
| First-break multiplier | OFF | p=0.203 (needs <0.01), n_first=476, n_later=816; the earlier 1.947 was a contamination artifact |
| Spot-level window clip | OFF (new, opt-in) | held-out -0.38 percent, within noise; coefficient shift -0.00095 pooled |
| Counter-programming covariate | OFF (new, gated) | held-out RMSE 0.24214 vs 0.24200: 0.1 percent worse; strength beta real (-0.00186, CI [-0.00323, -0.00050]) but does not transfer yet |
| Advertiser demand signal | ON as a provable no-op (1.0 weight) | see kairos-model docs |

The competitor-information boundary is law: competitor data (programming,
ratings) informs ONLY the retention model; competitor revenue is never used
(none exists in the data), and nothing is ever projected or placed for
competitor channels. Training-only rival ad-placement signals cannot reach a
live decision: the boundary raises `ForwardBoundaryError` in code
(`kairos/model/competitor_features.py`, tested in
`tests/test_counterprogramming.py`).

## What will change when two years of data land

Measured extrapolations (`scripts/audit_scale_readiness.py`):

* EB shrinkage self-transitions from pooled to per-cell with NO code change:
  holding the learned variance components fixed, a mean cell goes from 90
  percent pooled (today) to 42 percent at 12 months and 26 percent at 24
  months. The per-cell discrimination the model honestly lacks today is
  exactly what more data buys.
* All held-out gates (series, first-break, counter-programming) re-run at
  rebuild time on the richer data; each can flip ON only by beating its bar.
* Runtime: the full measurement pipeline extrapolates linearly to about 3
  minutes at 24 months (fine for an offline rebuild). One quadratic term is
  flagged: the per-break programme-title scan grows 576x (7.8e9 operations);
  switch it to bisect when the data lands.
* Memory: the minute-lookup structures grow linearly to about 840 MB; fine on
  a build machine, worth indexing if rebuilds move to small instances.
* The detrend baseline (per-minute mean over the WHOLE window) becomes a
  2-year average: seasonality (summer vs winter curves, DST shifts) will smear
  it. Before trusting 24-month coefficients, the baseline should become
  season-aware (e.g. month-of-year x broadcast-minute), gated on held-out
  skill like everything else. Not changed now: unmeasurable on one month.
* Freshness stays honest across the data swap: fingerprints follow the
  RESOLVED source files (xlsx or uploaded CSV), so replacing the workbooks
  flips the model to `stale` instead of `unknown`.

## Candidate artifacts and what adopting them would move

Candidates live under `models/candidates/` and never drive the plan; the lead
decides adoption. Predicted-revenue movement was measured by running the exact
recompute path in memory with each candidate's coefficients
(`scripts/estimate_candidate_revenue_movement.py`); baseline weekly predicted
revenue 215.34M:

| candidate | content | held-out gate | predicted revenue if adopted |
|---|---|---|---|
| `tv_break_coefficients_afterwindow.json` | recompute verification; byte-equal to shipped | n/a (identity) | +0 (identical) |
| `tv_break_coefficients_spotclip.json` | spot-level window clip variant | FAILED (-0.38 percent) | -460,768 (-0.214 percent) |
| `tv_break_coefficients_competitor.json` | competition-de-confounded coefficients | FAILED (-0.1 percent) | +565,489 (+0.263 percent) |

Recommendation: adopt NEITHER on one month. Both movements are model-internal
shifts that their held-out gates could not justify; a revenue delta without a
skill improvement is noise wearing a suit. Re-run both gates when the two-year
data lands.

## Known limitations

* One month, one country, four channels; November only. No holiday, summer,
  or special-event coverage.
* Retention effects are measured on minute-level channel TVR, not individual
  viewer journeys; "retention" is audience level held across the break.
* Cells with n under 15 (10 of 36) are nearly fully pooled; their labels say
  `low` confidence and the dashboard shows that honestly.
* The optimizer applies the coefficient linearly per break
  (`baseline + coefficient x k`); nonlinearity in k was probed earlier and
  found flat, but only within the observed 0-6 breaks-per-programme range.

## Validation panel verdict (2026-07-05)

Three independent computed referee reviews (causal identification, uncertainty
calibration, decision robustness) were run against the real month; the full
synthesis lives in docs/model-validation/README.md. The headlines the owner
should know:

- BIAS CORRECTED, LIVE: a matched placebo experiment showed the earlier
  per-break retention cost was understated (within-show audience-build drift
  absorbed into the estimate; corrected pooled cost -0.0533 vs the earlier
  -0.0391 under the shipped-ad-minutes baseline, permutation p = 0.0005). That
  correction now SHIPS in `models/tv_break_coefficients.json`
  (`placebo_correction_active` is true): the content-only-baseline variant
  subtracts each genre's measured no-break drift and the shipped pooled cost is
  -0.0499 (about x1.40 on the content-only raw). The retention-cost headline
  therefore already reflects the debiased level; the earlier "reads about 6.1M
  ILS per week low until the correction ships" gap is closed. The decision
  layer is largely insensitive to this level shift (see next bullet), so plans
  did not move materially, but the reported cost now carries the correction.
- DECISIONS ARE ROBUST: across 200 coefficient draws the plan is identical to
  shipped in 98.5 percent of draws and break counts never move; the 36-cell
  structure and one pooled constant currently produce the same plan (only 12
  cells are reachable and they collapse to 4 class coefficients). Retention
  modeling as a whole earns +11,926 ILS per day against ignoring retention.
  risk_lambda does not alter the plan at any setting today; it is honest
  bookkeeping of the worst plausible cost, not tail protection.
- INTERVALS: the per-cell ci is a latent-cell-mean band, holdout-consistent as
  such, but it covers only 4.9 percent of individual break outcomes at nominal
  95 percent, and honest cell coverage today needs about 1.77x wider intervals
  (self-heals to about 1.07x at 24 months of data). tau2 is on a knife edge at
  this sample size; the pooling machinery itself recovers correctly at 12x and
  24x data with no code change.
