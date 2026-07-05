# Kairos retention model: uncertainty calibration and pooling-machinery review

A statistician's audit of the uncertainty pipeline behind
`models/tv_break_coefficients.json`: the per-cell credible intervals, the
DerSimonian-Laird empirical-Bayes pooling, and the held-out gates. Every
number below is computed, on the real November 2024 reference month or on
synthetic data with known truth pushed through the actual pipeline code
(`kairos.model.measure.channel_coefficients` and the estimators it calls;
nothing reimplemented). No opinions without numbers.

Reviewed 2026-07-05 at git `79c2c768` with in-tree, behavior-preserving edits
to `kairos/model/measure.py` (sha256 `28de7d64...`) and
`kairos/model/competitor_gate.py` (sha256 `ff1ca6c3...`) landing concurrently.
Consistency under those edits was verified twice: (a) the coverage study
produced numerically identical results before and after the edits landed, and
(b) a from-scratch recompute under the edited code reproduces all 36 shipped
coefficients to a max abs difference of exactly 0.0 (section 5).

| study | script | results JSON |
|---|---|---|
| Interval coverage + calibration (temporal holdout) | `scripts/validation/coverage_holdout.py` | `scripts/validation/out/coverage_holdout.json` |
| DL parameter-recovery simulation (1x/12x/24x) | `scripts/validation/parameter_recovery.py` | `scripts/validation/out/parameter_recovery.json` |
| Determinism + gate seed sensitivity | `scripts/validation/determinism_seeds.py` | `scripts/validation/out/determinism_seeds.json` |
| Standing smoke test (seeded, ~4 s) | `tests/validation/test_parameter_recovery_smoke.py` | run via pytest |

All scripts are seeded (`numpy default_rng`), re-runnable to identical output
(verified: a full re-run of the coverage study byte-matches modulo the
environment stamp), and each finishes in well under 10 minutes (17 s / 56 s /
34 s).

## 0. What the shipped interval actually is

`channel_coefficients` publishes, per cell, `theta = mu + (1-B)(ybar - mu)`
with posterior variance `(1-B) sigma_i^2 + B^2/sum(w)` and a 95% band
`exp(theta +/- 1.96 sd) - 1`. This is a credible interval on the LATENT cell
mean (the long-run per-break retention delta of the cell), not a predictive
interval for a realized week's cell average and not a band for a single
break's outcome. `conservative_impact` (risk_lambda) prices `ci_low`, i.e.
the lower tail of the latent-mean posterior. The three target quantities
have very different dispersions on this data (log-effect space):

* latent-mean posterior sd: ~0.005-0.010 per cell;
* one held-out week's cell-mean sampling sd: `sqrt(s2/n_test)` ~ 0.06-0.17;
* single-break outcome sd: `sqrt(s2)` = 0.242.

Most of the "coverage failures" below are the arithmetic of that gap, and the
review separates interval DEFECTS from interval MISREADINGS accordingly.

## 1. Credible-interval coverage on a temporal holdout

Split: train days 1-23 (2,002 breaks), evaluate days 24-30 (530 breaks, 35 of
36 cells present). Train-side coefficients and intervals come from the shipped
pipeline run on the training split; 50/80/95 bands rescale the same posterior
sd (exact inversion of the shipped 1.96 band). Primary analysis shares the
full-month detrend baseline exactly as a shipped rebuild would; a fully
leak-free variant (train-only fit, own-week test baseline) is reported below.

Empirical coverage, primary split ("impl" = what the model itself expects the
naive number to be given the target's own sampling noise; brackets are Wilson
95% for cells, cell-cluster bootstrap 95% for individuals):

| nominal | cell mean, naive | impl | cell mean, noise-adjusted | single break vs shipped band | impl | single break vs predictive band |
|---|---|---|---|---|---|---|
| 50% | 0.029 [0.005, 0.145] | 0.039 | 0.571 [0.409, 0.720] | 0.023 [0.013, 0.036] | 0.012 | 0.687 [0.645, 0.741] |
| 80% | 0.029 [0.005, 0.145] | 0.075 | 0.771 [0.610, 0.879] | 0.036 [0.021, 0.052] | 0.023 | 0.853 [0.819, 0.890] |
| 95% | 0.114 [0.045, 0.260] | 0.113 | 0.914 [0.776, 0.970] | 0.049 [0.033, 0.066] | 0.035 | 0.938 [0.912, 0.963] |

Leak-free sensitivity (train 2,002 / test 520): cell naive 0.057/0.143/0.171,
cell adjusted 0.657/0.800/0.943, individual shipped 0.017/0.033/0.063,
individual predictive 0.696/0.892/0.940 - same picture, so the shared
baseline is not driving any conclusion.

Findings, in order of importance:

1. **The shipped 95% band covers 4.9% of individual break outcomes** (and
   11.4% of realized weekly cell means). This is not an estimation bug - the
   observed numbers match the model's own implied values (0.035, 0.113), and
   a mean-interval covering single outcomes was never promised - but it is a
   CONTRACT hazard: the plan CSV's `retention_ci_low/high` and the
   risk_lambda "worst plausible cost" quantify estimation uncertainty of the
   cell mean only. A single break's realized log effect has sd 0.242 vs a
   posterior sd of ~0.005: the outcome band is ~40x wider than the shipped
   band. Anyone reading `retention_ci_low` as "how bad can this one break
   plausibly be" is off by that factor.
2. **Read as what it is (a latent-mean interval), the band is roughly honest
   on the holdout**: adding the held-out target's own sampling noise gives
   0.571/0.771/0.914 vs nominal 0.50/0.80/0.95 - all Wilson intervals cover
   nominal (95% level marginally: 32/35 cells). The holdout has limited power
   (35 cells); the simulation in section 3 resolves the remaining question
   and finds true undercoverage that this test cannot see.
3. **Normal predictive intervals would fix the single-break contract at 95%
   but not deeper**: `theta +/- z sqrt(post_sd^2 + s2)` achieves 0.938
   [0.912, 0.963] at 95% but over-covers at 50% (0.687). The within-cell
   noise is strongly non-normal: skew -1.10, excess kurtosis +7.87
   (train residuals). A peaked, heavy-tailed distribution passes at 95% by
   luck of the crossing point; tail pricing beyond 95% needs empirical
   quantiles, not z-scores.
4. **Week-to-week level drift exceeds the interval.** Weekly grand means of
   the measured log effect: -0.0438 (wk1), -0.0561 (wk2), -0.0342 (wk3),
   -0.0233 (wk4); train-to-test shift +0.0202 (se 0.0104). The shipped 95%
   half-width in log space is ~0.0106: the LEVEL the whole plan runs on moves
   between weeks by about twice the band that is supposed to bound it. Under-
   coverage of naive weekly readings is therefore not only sampling noise;
   the process is not week-stationary at the +/-1% precision the interval
   implies.

## 2. Calibration curve and the shrinkage it implies

On this training split the pipeline's own tau2-hat is exactly 0, so every EB
prediction equals the grand mean: the shipped predictor is a CONSTANT, its
calibration slope is undefined, and its out-of-sample R^2 vs the constant
baseline is 0 by construction. That is itself the central calibration fact:
on 3 of 4 weeks of data the model chooses zero discrimination.

To measure what the data would have supported, calibrate the RAW (unpooled)
train cell means on the held-out breaks (log-effect space, cluster-bootstrap
CIs over the 35 cells; predictions constant within cell so clustering is
mandatory):

* slope 0.362, 95% CI (-0.327, 0.898); intercept -0.0079. Leak-free variant:
  slope 0.180, CI (-0.587, 0.750).
* Decile curve (raw predictor): realized means do not track predictions
  monotonically except at the extremes; consistent with a slope far below 1
  and mostly noise in between.
* RMSE-optimal single shrinkage multiplier on raw deviations: m* = 0.35
  (sweep over [0, 1.5]; m = 0 is the global constant, m = 1 is no pooling).
* Out-of-sample R^2 vs the constant: EB 0.000, raw means -0.0046.

Implied shrinkage arithmetic: a slope of 0.36 means only ~36% of a raw cell
deviation survives out of sample, i.e. raw means must be shrunk by ~2/3. The
shipped EB machinery applies keep-factors (1-B) of 0 on this split (tau2-hat
= 0) and mean 0.168 under the full-month artifact tau2 - i.e. **EB already
shrinks at least as hard as the holdout demands; there is no evidence that
ADDITIONAL shrinkage beyond DL is needed** (if anything the point estimate
says the artifact could keep slightly more, 0.35 vs 0.17, but the slope CI
spans 0 to 0.9, so no change is defensible). Translating the slope to a
between-cell variance: tau2 = slope/(1-slope) x E[s2/n_i] = 5.98e-4, vs the
DL 9.7e-5 (full month) and 0 (train split) - one month cannot pin tau2 to
better than an order of magnitude, which section 3 confirms from the other
direction.

Fragility of the learned tau2 on real data (actual DL estimator on
sub-windows of the shipped full-month effects):

| window | n breaks | tau2-hat | learned pseudo-count |
|---|---|---|---|
| full month (days 1-30) | 2,532 | 9.687e-05 | 601 |
| days 1-23 | 2,002 | 0 | full pooling |
| days 8-30 | 1,878 | 0 | full pooling |
| days 1-15 | 1,353 | 0 | full pooling |
| days 16-30 | 1,179 | 0 | full pooling |
| days 4-26 | 1,992 | 0 | full pooling |

The shipped tau2 = 9.687e-05 exists only on the exact full-month window;
every sub-window collapses to 0. The "learned pseudo-count 601" in the
artifact metadata is a knife-edge quantity, not a stable measurement.

## 3. Parameter recovery of the DL pipeline (known truth, actual code)

500 replications per scenario; 36 cells with the real artifact break counts
(min 8, median 24, max 292, total 2,532); mu = -0.0399, s2 = 0.0582 as
shipped; every replication scored through `channel_coefficients` +
`_dersimonian_laird`. "infl95" = the factor the shipped 95% interval must be
widened by to actually cover 95% of true cell effects. Oracle = same formula
fed the true tau2 and mu (isolates tau2-estimation error).

| scenario | tau2-hat mean (truth) | P(tau2-hat=0) | cover 50/80/95 | oracle 95 | mean abs B error | RMSE EB vs raw | infl95 |
|---|---|---|---|---|---|---|---|
| normal, tau2=0, 1x | 7.4e-05 (0) | 0.52 | .641/.896/.977 | 1.000 | 0.057 | .0047/.0500 | 0.79 |
| normal, tau2=shipped, 1x | 1.54e-04 (+59%) | 0.36 | .444/.685/.826 | 0.949 | 0.093 | .0111/.0498 | **1.77** |
| normal, tau2=shipped, 12x | 9.65e-05 (-0%) | 0.00 | .464/.755/.914 | 0.949 | 0.078 | .0077/.0146 | 1.16 |
| normal, tau2=shipped, 24x | 9.69e-05 (-0%) | 0.00 | .487/.780/.935 | 0.954 | 0.067 | .0065/.0102 | 1.07 |
| normal, tau2=4x, 1x | 3.91e-04 (+1%) | 0.13 | .426/.686/.836 | 0.949 | 0.103 | .0184/.0501 | 1.94 |
| normal, tau2=4x, 24x | 3.84e-04 (-1%) | 0.00 | .489/.784/.942 | 0.951 | 0.042 | .0087/.0103 | 1.03 |
| empirical tails, shipped, 1x | 1.49e-04 (+54%) | 0.37 | .437/.682/.828 | 0.953 | 0.091 | .0110/.0495 | 1.76 |
| empirical tails, shipped, 24x | 9.82e-05 (+1%) | 0.00 | .481/.782/.935 | 0.952 | 0.066 | .0066/.0102 | 1.07 |
| heteroskedastic, shipped, 1x | 3.3e-05 (-66%) | 0.81 | .289/.500/**.670** | 0.948 | 0.133 | .0108/.0392 | **2.19** |
| heteroskedastic, shipped, 12x | 7.1e-05 (-27%) | 0.02 | .437/.706/.865 | 0.951 | 0.238 | .0076/.0113 | 1.49 |
| heteroskedastic, shipped, 24x | 8.9e-05 (-8%) | 0.00 | .498/.781/.928 | 0.949 | 0.192 | .0062/.0082 | 1.11 |

(12x rows for tau2=0/4x and empirical omitted for space; full table in the
JSON.)

What this proves:

1. **The interval formula is correct; the coverage gap is tau2 estimation.**
   With the true tau2 plugged in, coverage is 0.948-0.954 in every scenario.
   With DL's estimated tau2 at TODAY's sample size, the 95% interval covers
   82.6% (normal) - the classic naive-EB failure (tau2-hat error is not
   propagated). The needed widening is **1.77x today, decaying to 1.16x at
   12x and 1.07x at 24x data**.
2. **tau2-hat at today's n is noise around a knife edge.** Truth = shipped:
   the estimate is 0 in 36% of replications and averages +59% above truth
   (the max(0, .) floor inflates the mean). Truth = 0: DL still reports a
   phantom mean of 7.4e-05 - the same order as the shipped 9.7e-05 - in 48%
   of months. A one-month tau2 of 9.7e-05 is statistically indistinguishable
   from 0, exactly matching the real-data sub-window collapse in section 2.
   By 12x the estimator is unbiased (rel bias 0.00, P(=0) = 0) - the
   machinery sharpens correctly as data grows, with no code change.
3. **Heavy tails do not hurt the pooling** (CLT protects cell means): the
   empirical-residual variant (excess kurtosis +7.9) is within Monte Carlo
   error of the normal one everywhere.
4. **Heteroskedasticity is the real structural flaw.** The pipeline assumes
   one pooled within-cell variance (`sigma_i^2 = s_p^2/n_i`); the real cells'
   variances span 0.0052-0.1186 (23x). Under that reality, today's 95% CI
   covers 67%, tau2-hat is biased -66%, and the shrinkage weights stay wrong
   even with 12x data (mean |B error| 0.238, vs 0.078 well-specified). This
   does not wash out with scale as fast as the tau2 problem does.
5. **The pooled point estimates are sound in every scenario** - EB RMSE beats
   raw cell means always (0.011 vs 0.050 today; 0.011 vs 0.039 even under
   heteroskedasticity), so the coefficients driving revenue decisions are the
   best available point numbers. The defect is confined to the UNCERTAINTY
   statements.

## 4. Determinism and gate seed sensitivity

**Determinism.** Two from-scratch coefficient computations in separate
processes (fresh interpreter, fresh data load) are byte-identical
(sha256-equal canonical JSON). Against the shipped artifact (built
2026-06-17): all 36/36 coefficients equal with max abs diff exactly 0.0;
tau2, series-gate RMSEs equal to the last bit; 3 of 180 detail CI bounds and
the first-break p-value differ by 1 ULP (~1.1e-16) - cross-build float
reproducibility to the last bit, run-to-run determinism absolute. All three
source-file fingerprints verify fresh. This also confirms the concurrent
`kairos/model` edits preserved behavior on the money path.

**Seed study** (the module seed constant patched at runtime across seeds
0-19 vs shipped 42; product source untouched):

| gate | shipped verdict | flip rate over 20 seeds | improvement: mean +/- sd (range) | margin vs seed noise |
|---|---|---|---|---|
| series/title layer | OFF (-8.35%) | **0/20** | -9.81% +/- 4.08pp (-21.7 to -5.0) | 2% bar sits +2.9 sd above the mean |
| counter-programming | OFF (-0.11%) | **0/20** | +0.12% +/- 0.19pp (-0.27 to +0.46) | 2% bar sits ~9.9 sd above the mean |
| first-break | OFF (p=0.203) | deterministic (no split); bit-identical across runs | - | - |

The gates are reproducible: no seed in 20 flips either verdict. Note the
counter-programming improvement's SIGN is pure seed noise (-0.11% at seed 42,
+0.12% mean) - it is the 2% activation margin, several sd wide, that makes
the gate a gate. A margin below ~0.5% would seed-flip; the current design is
safe and measured to be so.

**Fix candidates, evaluated:** 5-fold CV of the series comparison (every
break predicted exactly once) gives -10.55% pooled with per-fold sd 2.27pp -
same verdict, and roughly half the dispersion of single 80/20 splits (4.08pp)
at the same cost. The seed-averaged gate gives -9.81% +/- 0.91 (SE over 20
splits). Either upgrade makes the gate statistic quotable with an error bar;
neither changes any current verdict.

## 5. Verdict

1. **Is the uncertainty honest?** Partly. As a latent-cell-mean interval it
   is consistent on the holdout (0.571/0.771/0.914 noise-adjusted vs
   50/80/95), but the recovery simulation proves genuine undercoverage at
   today's sample size - true 95% coverage is 82.6% under the model's own
   assumptions and 67% under the real cells' heteroskedasticity - because
   tau2-estimation error is not propagated. And as consumed (per-segment
   `retention_ci_low`, risk_lambda worst-case), the band covers only ~5% of
   single-break outcomes and ~11% of weekly cell means; risk pricing
   understates any tail other than the latent-mean tail by an order of
   magnitude or more.
2. **Is the point prediction calibrated?** The level is real but drifts:
   weekly grand means moved over a 0.033 range in one month (+0.0202 train
   to test, se 0.0104), about twice the 95% band. Discrimination is ~zero
   and the model says so itself (tau2-hat = 0 on every sub-window; EB
   predictor constant on the train split; R^2_oos = 0). The raw-mean
   calibration slope 0.36 (CI -0.33 to 0.90) confirms hard shrinkage is
   right; no ADDITIONAL shrinkage beyond DL is warranted, and none should be
   removed either.
3. **Is the pooling machinery sound?** For points, yes, everywhere tested -
   EB beats unpooled means in all 15 scenarios and never regresses. For
   intervals: formula correct (oracle 0.95), tau2 plug-in undercovers today
   (needs 1.77x width), self-heals by 12x/24x (1.16x/1.07x) - provided the
   homoskedasticity assumption is addressed, since under real per-cell
   variances the weights stay materially wrong (|B error| 0.19-0.24) even at
   scale.
4. **Are the gates reproducible?** Yes. 0/20 seed flips on both split-based
   gates with the margin 2.9-9.9 sd above the seed noise; the first-break
   gate and the whole coefficient build are bit-deterministic across
   processes, and the shipped artifact reproduces to 0.0 on every
   coefficient.

## 6. Ranked fixes (each justified by a measured number)

1. **Propagate tau2 uncertainty into the interval - or widen it by a
   scale-aware factor.** Today multiply the posterior half-width by ~1.8
   (measured requirement 1.77 well-specified, 2.19 under real
   heteroskedasticity; a parametric bootstrap over tau2-hat achieves the same
   adaptively and needs no schedule). Decay with data: ~1.15 at 12x, ~1.05 at
   24x. Without this, risk_lambda's "worst plausible cost" is optimistic even
   for the quantity it prices.
2. **Moderate the within-cell variances instead of pooling them.** Real cell
   variances span 0.0052-0.1186 (23x); the pooled-s2 assumption costs 28pp of
   95% coverage today (0.67 vs 0.95) and leaves shrinkage weights wrong at
   any scale tested (mean |B error| up to 0.24). A limma-style
   `sigma_i^2 = (d0 s_p^2 + (n_i-1) s_i^2) / (d0 + n_i - 1)` slots into
   `_cell_stats`/`_dersimonian_laird` without changing the architecture.
3. **Fix the uncertainty CONTRACT at the consumers.** Document (model card +
   CSV header docs) that `retention_ci_low/high` bounds the CELL MEAN, and
   add a per-break predictive band where per-break tail risk is actually
   wanted: +/-z sqrt(post_sd^2 + s2) measured 0.938 at nominal 95 on held-out
   breaks; beyond 95% use empirical residual quantiles (excess kurtosis +7.9
   makes z-tails wrong).
4. **Monitor level drift; it is the binding risk, not cell ordering.** Weekly
   grand mean moved +0.0202 in one week (2x the 95% band). A weekly
   control-chart on the pooled mean log effect (limits from s2/n_week) with a
   recompute/stale trigger costs nothing and catches the failure mode the
   intervals cannot.
5. **Harden the gates' statistic, keep their verdicts.** Replace the single
   80/20 split with the 5-fold pooled comparison (same verdicts today;
   dispersion 2.27pp vs 4.08pp) or report the 20-seed mean +/- SE
   (-9.81 +/- 0.91 series; +0.12 +/- 0.04 counter-programming), and record
   margin-vs-noise in the JSON metadata. Priority low - measured flip rate is
   0/20 - but it makes the 2% bar auditable.
6. **State the tau2 knife edge in the model card.** The learned
   pseudo-count "601" rests on a tau2 (9.7e-05) that is zero on every
   sub-window of the same month, is zero in 36% of simulated months when
   true, and is matched in size by the phantom tau2 DL invents 48% of the
   time when the truth is exactly 0. The honest one-month statement is
   "between-cell variance not yet detectable; shrinkage near-total"; the 12x
   simulations show the artifact will say so on its own once data lands.

## Reproduction

```
/Users/home/.venvs/meridian/bin/python scripts/validation/coverage_holdout.py
/Users/home/.venvs/meridian/bin/python scripts/validation/parameter_recovery.py
/Users/home/.venvs/meridian/bin/python scripts/validation/determinism_seeds.py
/Users/home/.venvs/meridian/bin/python -m pytest tests/validation/ -q
```
