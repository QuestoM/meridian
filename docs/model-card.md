# Kairos retention model card

The document to read before trusting the model. Every number below is measured
on the real reference data (November 2024, one month, four channels) with the
scripts named next to it. Nothing here is a target or an opinion. Last
measured: 2026-07-05.

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

From the shipped `models/tv_break_coefficients.json` (fingerprint-fresh
against the data on disk, verified 2026-07-05):

* 2,532 breaks measured (5,696 raw; 3,164 dropped by the adjacent-break
  window clip rather than measured on contaminated audience).
* 36 cells, ALL negative. Mean per-break retention delta -0.0390; range
  -0.0477 (`Other_first_long`, n=238) to -0.0313 (`Other_last_short`, n=146).
* Cell sample sizes: min 8, median 24, max 292. Confidence labels: 14 high,
  12 medium, 10 low.
* Pooling: tau squared 9.687e-05, pooled within-variance 0.0582, learned
  pseudo-count 600.8. At the mean cell size (n about 70) a cell keeps only
  about 10 percent of its own mean and takes 90 percent from the global mean.
  The cells genuinely differ by little on one month, and the model says so
  rather than inventing spread.

## Honest skill statement (read this before selling the model)

* The pooled LEVEL is solid: the average break costs about 3.9 percent of the
  audience multiplier, measured causally (detrended, clipped), and moving to
  that measured level from the earlier contaminated measurement changed the
  pooled cost by +52 percent (see de-bias below). That level is what the
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
in `break_effects`; the shipped artifact was verified to be EXACTLY the
post-clip recompute (max coefficient difference 0.0,
`models/candidates/tv_break_coefficients_afterwindow.json`).

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
| EB learned pooling | ON | tau2 9.687e-05, learned pseudo-count 600.8 |
| Series/title layer | OFF | held-out RMSE 0.2646 vs genre 0.2442: 8.3 percent WORSE; gate needs 2 percent better |
| First-break multiplier | OFF | p=0.203 (needs <0.01), n_first=476, n_later=816; the earlier 1.947 was a contamination artifact |
| Spot-level window clip | OFF (new, opt-in) | held-out -0.38 percent, within noise; coefficient shift -0.00095 pooled |
| Counter-programming covariate | OFF (new, gated) | held-out RMSE 0.24452 vs 0.24424: 0.1 percent worse; strength beta real (-0.00201, CI [-0.00325, -0.00076]) but does not transfer yet |
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
