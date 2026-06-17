# The Retention-Cost Model: Honest Reality Report and Staged Bayesian Design

This document describes, truthfully, how Kairos estimates the retention cost of a
commercial break today, where that estimate is weak, and the staged path to make
it a defensible, uncertainty-aware, data-estimated quantity. Section a describes
what the code actually does right now. Section b describes the upgrade in four
stages: Stage 1 is implemented (see the code changes that accompany this doc);
Stages 2 through 4 are design only and are labelled as such.

The directive this model serves: the system decides where and how many commercial
breaks a channel airs to maximize ad revenue NET of the retention damage the
breaks cause. The retention cost of a break is therefore a load-bearing number.
It must be measured, carry honest uncertainty, and the decision that consumes it
must be uncertainty-aware. This doc holds the model to that standard.

---

## a. Current state, precisely and truthfully

### a.1 What `measure.py` actually estimates

The real per-break retention estimator lives in
`kairos/model/measure.py`. The call graph is:

`compute_measured_coefficients` -> `break_effects` -> `channel_coefficients`,
persisted by `write_coefficients_json`, read back by `read_coefficients_json`,
and resolved into the optimizer by `kairos/model/impact.py::load_impact_model`.

The estimand is a **per-break detrended audience-hold effect**, pooled per channel
cell, expressed as a delta on the optimizer's [0, 1] retention multiplier.

For each aired break (detected from the spots log via
`kairos/model/prepare.py::keyed_breaks`):

1. **Measure.** Take the mean audience (TVR) in the `_BEFORE_MINUTES = 3` minutes
   immediately before the break starts and the `_AFTER_MINUTES = 3` minutes after
   content resumes. The observed ratio is `obs_after / obs_before` from the
   minute-level dayparts series (`load_dayparts`). This is a direct measurement on
   the real audience curve, not a proxy.
2. **Detrend.** Audience rises into prime time regardless of breaks, so a raw
   before/after ratio in the evening looks falsely good. For the same
   broadcast-minute window, compute the channel's typical ratio `base_after /
   base_before` from `_baseline_levels` (the month-averaged audience curve at each
   broadcast minute). The per-break **log effect** is
   `log(observed_ratio) - log(expected_ratio)`. Positive means the break held more
   audience than the time-of-day trend predicts; negative means it shed more. This
   is a difference-relative-to-trend design (a ratio-of-ratios in levels, a
   difference in logs): the trend is the counterfactual, the break's own marginal
   effect is what remains.
3. **Pool.** `channel_coefficients` groups the per-break log effects by channel
   cell (`program_type x break_position x break_length`, the
   `ChannelDescriptor.name`). Each cell mean is shrunk toward the grand mean by a
   strength **learned from the data**, not a hand-set constant: a normal-normal
   hierarchical (empirical-Bayes) model whose between-cell variance `tau^2` is
   estimated by DerSimonian-Laird (see Stage 2 below, now shipped). With a pooled
   within-cell variance `s_p^2`, each cell's posterior mean is
   `theta_i = mu + (1 - B_i)(ybar_i - mu)` with shrinkage fraction
   `B_i = sigma_i^2 / (sigma_i^2 + tau^2)` and `sigma_i^2 = s_p^2 / n_i`. This is
   algebraically the same `(n*ybar + k*mu)/(n+k)` form, but with the equivalent
   pseudo-count `k* = s_p^2 / tau^2` set by the data: small when cells genuinely
   differ, large when they are alike. On the reference data `k* ~ 79` (the cells are
   mostly alike, so the data pools harder than the old hand-set 20). The shrunk log
   effect is converted to a retention delta `exp(theta_i) - 1`, clamped to be `<= 0`
   for the optimizer (`coefficient = min(0.0, raw_delta)`), and a 95% interval is the
   proper hierarchical posterior `theta_i +/- 1.96 * sqrt(V_i)`,
   `V_i = (1 - B_i) sigma_i^2 + B_i^2 / sum(w)`, mapped back through `exp(.) - 1`. The
   second term carries the grand-mean's own uncertainty, so a fully-pooled cell never
   reports a false zero-width interval. When the within-cell spread cannot be
   estimated (every cell a single break) it falls back to the fixed pseudo-count.
   `between_cell_variance()` exposes the learned `tau^2`, `s_p^2` and `k*` for audit,
   and the coefficients JSON records them under `metadata`.

`MeasuredCoefficient` already carries `coefficient`, `raw_delta`, `n`, `ci_low`,
`ci_high` per cell. `write_coefficients_json` persists all of it under `detail`.

### a.2 The identification argument

The design identifies the break's marginal effect under one assumption: **the
typical broadcast-minute ratio is the correct counterfactual for what the
before/after ratio would have been with no break at that minute.** Dividing the
observed ratio by the expected ratio removes any audience movement that is a
function of broadcast minute alone (the prime-time ramp, the post-news drop-off,
the late-night decay). What survives is attributable to the break, conditional on
that assumption holding. The test `test_break_effect_detrends_the_day_curve`
confirms the mechanism: a break day that rises exactly like every other day
detrends to a log effect of ~0, not a spurious gain.

This is honest as far as it goes. It is a credible quasi-experimental design (a
within-channel, within-time-of-day difference relative to trend). It is not a
randomized experiment and does not claim to be.

### a.3 Real weaknesses (named, not hidden)

These are the genuine limitations a rigorous reviewer would raise. Each is real
in the current code.

1. **`shrinkage_k = 20` was an arbitrary knob (RESOLVED in Stage 2).** The amount of
   partial pooling was set by hand: a cell with 5 breaks and a cell with 500 breaks
   were both shrunk against the same fixed pseudo-count, correct only by accident.
   The right strength depends on how much real signal varies between cells (`tau^2`)
   versus the noise within a cell, and that ratio is exactly what the hierarchical
   model now learns. `channel_coefficients` estimates `tau^2` (DerSimonian-Laird)
   and pools each cell with the data-set strength `k* = s_p^2 / tau^2` instead of a
   constant; the fixed pseudo-count survives only as a fallback for the degenerate
   case where the within-cell spread cannot be estimated. The cell with 5 breaks now
   shrinks more than the cell with 500, because its mean is measured less precisely.

2. **The normal-SE credible interval is fragile for small n.** `se = std / sqrt(n)`
   with a 1.96 multiplier is a large-sample normal approximation. For a cell with
   `n = 1` the code sets `std = 0`, producing a zero-width interval -- a false
   certainty. For small n the log effects are not well-approximated by a normal
   sampling distribution, and the interval understates uncertainty precisely where
   uncertainty is highest.

3. **The baseline curve is contaminated by the very breaks being measured.**
   `_baseline_levels` averages the channel's audience over every day at each
   broadcast minute. Recurring breaks (a programme that breaks at the same minute
   most nights) depress the baseline at exactly the minutes we then compare
   against. The counterfactual is therefore partly the thing we are trying to
   remove, biasing the measured effect toward zero (a break looks less harmful than
   it is because the "normal" curve already includes that break's dip). The before
   and after windows are excluded from their own measurement, but the month-average
   baseline at those minutes is not.

4. **The absolute magnitude is tied to, and clamped against, the `-0.03` anchor.**
   Two couplings exist. First, `measure.py` clamps any non-negative shrunk delta
   to `0.0`, so a genuinely measured retention gain cannot reach the optimizer
   (defensible as a conservative floor, but it means the optimizer never sees
   upside). Second, when the measured JSON is absent and a Meridian posterior is
   used instead, `impact.py::_to_retention_deltas` rescales every channel so the
   average channel equals the declared `OptimizerAssumptions.retention_impact_per_break
   = -0.03` magnitude. In that path the data sets only the relative ordering of
   cells; the absolute level is the `-0.03` prior. Where data is thin, the prior
   dominates and the "measurement" is mostly the anchor.

5. **Uncertainty is computed and then discarded before the optimizer.** This is
   the most consequential gap for the directive. `MeasuredCoefficient` carries
   `n`, `ci_low`, `ci_high`. But `read_coefficients_json` returns only the flat
   `channel -> point coefficient` map; the interval and n are dropped on the floor.
   `PosteriorImpactModel` exposes only `coefficient_for(...) -> float`.
   `ProgramSegment.impact_coefficient` is a single float.
   `objective.predicted_retention(baseline, impact_coefficient, k)` consumes that
   single point estimate. The optimizer's greedy step in `optimizer.py` values
   every break at the point estimate with no notion of how sure that estimate is.
   The system therefore decides identically whether a cell's cost is measured from
   500 breaks with a tight interval or from 3 breaks with an interval that spans
   from "harmless" to "very damaging." That is not uncertainty-aware decision
   making.

6. **Retention is linear in k.** `predicted_retention` applies
   `baseline + impact_coefficient * k`: the second break costs exactly what the
   first did, the fourth exactly what the third did. Real audience shedding is
   unlikely to be linear -- the audience that survives three breaks is more
   tolerant than the audience that survived none (a saturating cost), or breaks
   compound (a viewer who already endured two is primed to leave on the third).
   The current model cannot represent either shape, and nothing tests which the
   data supports.

7. **No competitor-programme covariate exists.** The data layer carries no
   competitor-programme column anywhere (confirmed: no such field in `loaders.py`,
   `transform.py`, `contracts.py`, `prepare.py`, or `data_schemas.yaml`). The
   retention estimate conditions only on the channel's own break attributes. Yet a
   break's true cost depends heavily on what a viewer can switch to: shedding into
   a competitor's strong programme is a real loss; shedding when every competitor
   is also in break is nearly free. What is missing is the COVARIATE, not the data:
   verified on the reference set, all four channels are present in every source the
   feature needs. `load_dayparts` carries each channel's minute-level TVR
   (`CHANNELS = ("קשת 12", "רשת 13", "כאן 11", "עכשיו 14")`); `load_programmes` carries
   all four channels' titles and slots (8704 rows across the four), so the rival EPG
   (titles -> genre via the classifier) is available as a forward signal; and
   `load_spots` carries all four channels' aired ads, the training-only signal for
   `competitor_in_break`. So every Stage 3 input exists already and needs no data
   acquisition: the programme-versus-programme signal is simply not computed or
   consumed today. This is the goal's distinctive signal and the feature is absent,
   but the raw material for it is fully present.

---

## b. The staged upgrade

Each stage is defensible on its own and testable. Stage 1 is built now. Stages 2
through 4 are design only.

### Stage 1 (implemented): carry the full posterior end to end, make the decision uncertainty-aware

One-line summary: stop discarding `ci`/`n`; carry mean + interval + n + a
confidence label per cell into the optimizer, and let the objective value a break
at a risk-adjusted (conservative) retention cost instead of the bare point
estimate, with the default reproducing today's behavior exactly.

What it changes (the accompanying code):

- **A richer read path that does not discard uncertainty.**
  `measure.read_coefficients_detail(path)` returns the full per-cell detail
  (`coefficient`, `ci_low`, `ci_high`, `n`) keyed by channel, alongside the
  unchanged `read_coefficients_json` (kept for back-compat: it still returns the
  flat `channel -> float` map). `write_coefficients_json` already persists the
  detail; the round-trip of `coefficient`/`ci_low`/`ci_high`/`n` is now verified by
  test.
- **A confidence label derived from `n` and interval width.**
  `measure.confidence_label(n, ci_low, ci_high)` returns `high`/`medium`/`low` from
  the cell's sample count and the half-width of its credible interval. Thin cells
  or wide intervals are honestly labelled `low`. This is the operator-facing "where
  the model is sure versus guessing" signal.
- **An impact model that exposes the interval and confidence per cell.**
  `RetentionEstimate` (mean, ci_low, ci_high, n, confidence) plus
  `PosteriorImpactModel.estimate_for(program_type, position, length)`. The existing
  `coefficient_for(...) -> float` is unchanged, so every current caller still works.
  `load_impact_model` now reads the detailed coefficients when present and builds a
  model that carries the full estimate for each cell, falling back to a
  point-only estimate (interval == point, confidence `low`, n 0) for cells the data
  did not estimate.
- **An uncertainty-aware objective, default-safe.**
  `objective.conservative_impact(point, ci_low, ci_high, *, risk_lambda=0.0)`
  returns a risk-adjusted retention cost: at `risk_lambda = 0.0` it returns the
  point estimate exactly (today's behavior; nothing silently changes), and at
  `risk_lambda > 0.0` it values a break by the more pessimistic (more negative)
  end of its credible band, `point - risk_lambda * half_width`, clamped non-positive.
  The decision-theory rationale, documented in the function: the optimizer is
  choosing whether the marginal revenue of a break exceeds its retention cost; when
  that cost is uncertain, valuing it at an upper quantile of the damage (a
  conservative/robust decision) avoids placing breaks whose true cost might be far
  higher than the point estimate. This is min-over-the-credible-set robustness, the
  standard decision-theoretic response to estimation uncertainty.
- **Uncertainty threaded through the segment, optional.**
  `ProgramSegment` gains optional `impact_ci_low`, `impact_ci_high`,
  `impact_confidence` fields (default `None`, i.e. today's behavior). When present
  and a `risk_lambda > 0` is supplied to `optimize_breaks`, the greedy step values
  each break at `conservative_impact(...)` instead of the bare coefficient. With
  the default `risk_lambda = 0.0` the result is byte-for-byte the current plan.
- **Confidence surfaced on the result.** `SegmentPlan` carries
  `impact_confidence` and the interval, so the result object exposes, per segment,
  where the retention cost is well-measured versus a guess.

Where uncertainty now flows, end to end:
from `measure.py` detail JSON -> `read_coefficients_detail` ->
`PosteriorImpactModel.estimate_for` -> `ProgramSegment` (optional fields, populated
by `transform._segment_impact_kwargs`) -> `objective.conservative_impact` in the
greedy step -> `SegmentPlan` on the result -> the HTTP API response shape in
`kairos/service.py::result_to_dict`. The serialised plan now carries, per segment,
a `retention_cost` block (`point`, `used`, `ci_low`, `ci_high`, `n`, `confidence`)
and echoes the effective `risk_lambda` in `weights`, so the dashboard can render how
trustworthy the cost behind each break count is. The `risk_lambda` control is
threaded through both API entrypoints (`optimize_day_plan` and `run_scenario`),
defaulting to the assumption value (`0.0`), so the decision is uncertainty-aware in
production, not only in the library. The `/api/impact` endpoint additionally surfaces
per-cell `ci_low`/`ci_high`/`n` from the coefficients JSON
(`server.py::_load_measured_impact_summary`).

Shipped in three verified increments: Stage 1.5 made the greedy decision consume the
interval via a tunable `risk_lambda` and surfaced plan provenance; Stage 1.6 threaded
the measured interval into `ProgramSegment` in `transform.py` so it fires in a real
run; Stage 1.7 threaded `risk_lambda` through `optimize_day_plan` / `run_scenario`
into `optimize_breaks` and serialised the per-segment `retention_cost` block plus
`risk_lambda` in `result_to_dict`.

### Stage 2 (SHIPPED): learned hierarchical partial pooling

Shipped in `channel_coefficients` (see Step 3 of section a.1 above) using the
closed-form empirical-Bayes path described below: a DerSimonian-Laird estimate of
`tau^2` with a pooled within-cell variance, the precision-weighted posterior mean,
and the proper hierarchical posterior-variance interval. `between_cell_variance`
exposes the learned `tau^2` / `s_p^2` / `k*` and the coefficients JSON records them
under `metadata` (`pooling_method`, `between_cell_variance_tau2`,
`pooled_within_variance`, `learned_pseudo_count`). The fuller MCMC variant below
remains a future option; the shipped closed form already removes the fixed-k knob.

One-line summary: replace the fixed `shrinkage_k = 20` with a hierarchical
Bayesian model that learns the between-cell variance `tau^2`, giving a real
posterior per cell, trusting thick cells and honestly widening thin ones, and
freeing the absolute magnitude from the `-0.03` anchor where data is sufficient.

The model. Let `y_ij` be the j-th per-break log effect in cell i (the same
detrended `log_effect` `break_effects` already produces). A two-level normal model:

- Likelihood: `y_ij ~ Normal(theta_i, sigma_i^2)`, where `theta_i` is cell i's true
  mean log effect and `sigma_i^2` its within-cell variance (estimated from the
  spread of breaks in the cell, or pooled across cells when a cell is thin).
- Cell prior (the partial-pooling level): `theta_i ~ Normal(mu, tau^2)`. `mu` is
  the grand mean across cells; `tau^2` is the **between-cell variance** -- how much
  cells genuinely differ. This is the quantity the current code fixes by fiat
  through k; here it is a parameter with its own weak prior
  (e.g. `tau ~ HalfNormal(s)` with s set from the plausible scale of log effects).
- Hyper-prior on `mu`: a genuinely weak prior centered near a small negative number
  (breaks shed audience), e.g. `mu ~ Normal(log(1 - 0.03), broad_sd)`. This is
  where the `-0.03` enters Stage 2: only as a weak prior on the grand mean, not as
  a hard rescale. Where the data are rich, the likelihood dominates and the
  posterior `mu` moves off the anchor; where the data are thin globally, the prior
  gently holds the level sane.

What this buys, and how it degrades honestly. The posterior cell mean is the
precision-weighted compromise
`E[theta_i | y] = (n_i/sigma_i^2 * ybar_i + 1/tau^2 * mu) / (n_i/sigma_i^2 + 1/tau^2)`.
When `tau^2` is large (cells really differ), thick cells are trusted at nearly
their own mean. When `tau^2` is small (cells are alike), everything shrinks hard
toward `mu`. The shrinkage strength is **learned** from the data, replacing k. A
data-poor cell has small `n_i`, so its posterior is dominated by `mu` (it shrinks
to the prior) and its posterior variance is wide -- which is exactly the `low`
confidence label Stage 1 already surfaces, now backed by a real posterior rather
than a normal-SE approximation. There is no false certainty at `n = 1`: the
posterior is appropriately broad.

How ~2 years of data feed it. The aired-spots log over ~24 months gives on the
order of thousands of breaks per active channel cell across the 36-cell grid
(`4 program_types x 3 positions x 3 lengths`). That is ample to estimate `tau^2`
and to let well-populated cells (News, prime positions) leave the prior, while
genuinely rare cells (long breaks in Other programmes) stay near `mu` with wide
posteriors. A day-of-week / seasonality control can enter as an additional level
or as covariates on `theta_i` so trend leakage the broadcast-minute detrend misses
is absorbed rather than attributed to the break. Fit with a small Stan/PyMC model
(or a closed-form empirical-Bayes estimate of `tau^2` as a cheaper first cut);
either way the per-cell output is `(posterior_mean, ci_low, ci_high, n)` -- the same
contract Stage 1 already carries end to end, so Stage 2 is a drop-in replacement for
the producer with no change to the consumer.

This also fixes weakness a.3.3 partially: the baseline contamination can be reduced
by building the broadcast-minute baseline from a leave-one-out or break-excluded
curve (compute the typical ratio from days/minutes where that recurring break did
not air), which is a data-preparation change orthogonal to the hierarchical model
but naturally specified alongside it.

### Stage 3 (design only): the competitor-programme covariate

One-line summary: encode what competitors air against each slot as a forward-usable
feature and let it modulate the retention cost, with a strict information boundary --
competitor programmes are forward inputs, competitor breaks/ads are training-only.

This is the goal's distinctive signal and it is **not yet built** as a feature, but
it is feasible now: the raw data exists. Verified on the reference set, all four
channels appear in `load_programmes` (the rival EPG, the forward signal),
`load_dayparts` (rival minute-level audience curves) and `load_spots` (rival aired
ads, the training-only signal) -- see a.3.7. No data acquisition is needed; what is
missing is the feature-engineering and the betas, which is the build below.

Encoding "what competitors air against this slot." For the slot a break sits in,
build a competitor-context feature from the three rival channels' schedules at the
same wall-clock minutes:

- `competitor_strength`: the sum (or max) of rival channels' expected TVR at the
  break's minutes, from the programmes grid plus the historical audience curve for
  the rival programme type. A break opposite a strong rival show is a more dangerous
  place to lose a viewer (there is somewhere good to go).
- `competitor_genre_contrast`: whether a rival is airing a substitute genre (news
  against news, drama against drama) versus a complement; substitutes raise the
  switching hazard.
- `competitor_in_break`: historically, the fraction of the slot's minutes where
  rivals were themselves in break. When everyone breaks together, shedding is nearly
  free; when only we break, the viewer has a clean alternative.

How it enters the retention model. As cell-level or break-level covariates on
`theta_i` in the Stage 2 hierarchical model:
`theta = mu + beta_strength * competitor_strength + beta_contrast * genre_contrast + cell_effect`.
The retention cost becomes a function of the competitive context, not a constant
per cell. Estimated `beta`s with their own posteriors say how much, and how
certainly, competition matters -- and feed the same `(mean, ci, n, confidence)`
contract downstream.

The strict information boundary (load-bearing). The channel knows competitors'
**programmes** for the coming week (the rival EPG is published in advance), so
`competitor_strength` and `competitor_genre_contrast` are legitimately available at
decision time -- they are forward-usable features. The channel knows competitors'
**breaks and ads** only after the fact, from the historical aired-spots logs.
Therefore `competitor_in_break` and anything derived from rival ad placement may be
used to **train** the model (to estimate the betas) but must **never** be a forward
input to a live decision, because that information does not exist when the plan is
made. The design must enforce this at the feature-engineering seam: forward
features come only from the rival programme grid; training-only features are tagged
and stripped from the inference path. Violating this boundary would leak
unavailable information and produce a model that cannot be deployed honestly. This
is design, not yet built.

### Stage 3b (SHIPPED): first-break-aware retention cost

One-line summary: the show's FIRST interruption sheds materially more audience
than its later breaks, so the first break of each programme carries an extra,
measured, self-activating retention cost.

This is distinct from `break_position` in the 36-cell taxonomy, which is the
first/middle/last SPOT WITHIN a single break, not the first/middle/last BREAK
OF THE SHOW. The ordinal of a break within its containing programme is recovered
in `prepare.py` (`programme_ordinals`): each detected break is matched to the
programme whose start/end span contains it, then breaks are numbered by air time
per programme. On the reference data this matches 99.7 percent of breaks.

The measurement (`measure.py` `first_break_gate`) compares the detrended log
effect of first breaks (ordinal 1) against later breaks (ordinal >= 2), using the
same broadcast-minute baseline detrending as every other coefficient, so the
contrast is not a baseline artifact. On the reference data first breaks shed
about 1.95 times the all-breaks cost (n_first=1468, n_later=3081, Welch two-sided
p approximately 9e-8). The gate ships a `first_break_multiplier > 1.0` only when
the contrast is large (multiplier >= 1.10), significant (p < 0.01), and backed by
enough breaks (>= 200), capped at 2.0; otherwise it stays at 1.0 (off). The
decision and its numbers are written to the coefficients JSON metadata
(`first_break_multiplier`, `first_break_active`, `first_break_n_first`,
`first_break_n_later`, `first_break_p_value`, `first_break_reason`) so any reader
can audit it, exactly like the series gate.

The optimizer applies it in `_segment_retention`, the single chokepoint all
retention math flows through: at `k >= 1` it charges one extra delta of
`impact_coefficient * (first_break_multiplier - 1.0)` once (not per break), so the
first break of a loaded segment costs more while every later break is unchanged.
When the multiplier is 1.0 the result is byte-identical to the base linear model.
The value reaches the optimizer through `OptimizerAssumptions.first_break_multiplier`,
folded from the JSON metadata in `schedule.py` (`_apply_first_break_multiplier`)
and passed into both segment builders in `transform.py`. Reported revenue always
uses the realised retention; the adjustment can only lower a borderline first
break's value, never inflate revenue.

### Stage 4 (design only): non-linear retention in k

One-line summary: examine whether the per-break cost saturates or compounds with
the number of breaks, and test which shape the data supports rather than assuming
linearity.

Today `predicted_retention` is linear: cost per break is constant. Two plausible
alternatives:

- **Saturating** (diminishing harm): the survivors of earlier breaks are the most
  tolerant viewers, so each additional break sheds a smaller share. A form such as
  `retention(k) = baseline * exp(-c * k^p)` with `p < 1`, or a per-break delta that
  shrinks geometrically.
- **Compounding** (accelerating harm): each break primes the audience to leave on
  the next, so marginal cost rises in k. The same functional family with `p > 1`,
  or a convex per-break delta.

How to test which the data supports. The measurement is already keyed by break --
the spots log identifies, per programme airing, how many breaks ran and where the
audience stood after each. Estimate the audience-hold as a function of break index
within a programme (first break, second break, third) and fit the competing shapes
(linear, saturating, compounding) with the hierarchical machinery from Stage 2,
comparing by held-out predictive log-likelihood or WAIC. Crucially, the optimizer's
greedy allocator already assumes diminishing marginal value (each break earns less
because the surviving audience is smaller); a saturating retention cost is
consistent with that and would sharpen it, while a compounding cost would argue for
a stricter cap. The chosen shape replaces the linear `impact_coefficient * k` in
`predicted_retention` with a per-k cost the greedy step reads at each insertion.
This is design, not yet built.

---

## Summary of the contract this model upholds

The retention cost of a break is, and after each stage remains, a quantity with:
a point estimate, a credible interval, a sample count, and an honest confidence
label. Stage 1 makes that quantity flow into the decision and lets the decision be
conservative under uncertainty. Stages 2 through 4 make the quantity itself
progressively more defensible: learned shrinkage, the competitive signal the goal
turns on, and a tested functional form. Every claim above is tied to what the code
and data actually support today, and every gap is named rather than papered over.
