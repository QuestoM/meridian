# Counter-programming covariate: design, contract, measured verdict

Status: machinery SHIPPED and inert, covariate OFF by the held-out gate.
Measured on the real one-month reference data, 2026-07-05. Re-run
`scripts/measure_counterprogramming.py` after any data drop; the gate
re-evaluates and the recommendation updates itself.

## What this is

A break does not shed audience in a vacuum: whether a viewer who leaves comes
back depends on what rival channels air opposite the break. This covariate
encodes that context as numeric features of the per-break retention effect,
trains their sensitivities on PAST competitor data, and at prediction time can
consume the competitors' published schedule for the coming week.

## The information boundary (law, enforced in code)

Competitor data informs ONLY the churn/retention model, never revenue
projection or placement for competitor channels. Within retention:

* FORWARD features (usable in a live plan) come only from the rival EPG and
  the rivals' HISTORICAL audience curves, both known before the week airs.
* TRAINING-ONLY features (rival ad placement) exist only in historical logs.
  They may de-confound the fit but can never adjust a live decision.
  `kairos.model.competitor_features.assert_forward_only` raises
  `ForwardBoundaryError` if one reaches a forward path, and
  `kairos.model.future_epg.forward_adjustment` filters by role FIRST and then
  asserts, so a mislabeled beta fails loudly (tested).

## Features

| feature | role | source | month mean / nonzero |
|---|---|---|---|
| `competitor_strength` | forward | rivals' historical minute-level TVR curve summed opposite the break | 10.64 / 100.0% |
| `competitor_genre_contrast` | forward | fraction of rivals airing the same classifier genre at the break's middle minute | 0.19 / 39.1% |
| `competitor_prog_start` | forward | fraction of rivals with a programme START in `[break_start - 1 min, break_end + 3 min]` (the capture junction; tail matches the retention after-window) | 0.32 / 64.8% |
| `competitor_in_break` | training-only | fraction of break minutes where a rival also aired a break (from rival spots logs) | 0.37 / 52.3% |

`competitor_prog_start` is new in this build. The original pinned pair lives in
`FORWARD_FEATURES`; the full forward set is `EXTENDED_FORWARD_FEATURES`
(`kairos/model/competitor_features.py`). Estimation is a within-cell
(fixed-effects) OLS on the cell-demeaned log retention effect, so the betas
measure competition variation WITHIN a genre cell, not the confound that some
cells always face stronger rivals.

## Measured betas (real month, 2532 breaks, within-cell OLS)

| feature | beta | 95% CI | verdict |
|---|---|---|---|
| `competitor_strength` | -0.00201 per rating point | [-0.00325, -0.00076] | significant, causal direction as expected: a break opposite a stronger rival sheds more |
| `competitor_genre_contrast` | +0.01775 | [-0.02121, +0.05671] | not significant on one month |
| `competitor_prog_start` | +0.01759 | [-0.01777, +0.05296] | not significant on one month |
| `competitor_in_break` (control) | +0.02767 | [+0.00286, +0.05248] | significant: when rivals break too, shedding is cheaper (nowhere good to go) |

## The adoption gate and its verdict

`kairos.model.competitor_gate.counterprogramming_holdout_gate`: deterministic
80/20 split (seed 42), WITHOUT = training-cell-mean prediction, WITH = adjusted
cell mean plus the forward betas applied to the test break's own context.
Activation requires a 2 percent relative RMSE improvement, the same bar as the
series gate.

Measured verdict (n_test 506): RMSE WITH 0.24452 vs WITHOUT 0.24424, an
improvement of -0.1 percent. **The covariate does not improve held-out skill
on one month of data and therefore ships OFF.** The strength beta is real
within-sample, but most of its variation is absorbed by the cell means, and
what remains does not transfer out of sample yet. Expect this gate to be worth
re-running first when the two-year data lands: the beta's standard error
shrinks roughly with sqrt(n), and 24 months supply seasonal EPG variety one
November cannot.

## The future-week EPG file contract (prediction time)

Path: `data/reference/CompetitorProgrammes.xlsx` (preferred) or
`data/reference/CompetitorProgrammes.csv`. Schema: EXACTLY the reference
`Programmes` schema, parsed by the same loader (`Channel`, `Title`, `Date`
DD/MM/YYYY, `Start time`, `End time`, optional `Duration`; a `TVR` column is
IGNORED because future ratings do not exist; audience strength always comes
from the historical curve).

* `kairos.model.future_epg.load_future_competitor_epg()` returns the parsed
  frame plus an honest status payload (`present`, `path`, `rows`, `channels`,
  `window_start`, `window_end`, `reason`).
* `counterprogramming_features_for_window()` computes the three forward
  features for any future break window. Round-trip verified on 200 real
  breaks: max |contract feature minus training feature| = 0.0 (exact parity;
  anchor `own_category` at the break's middle minute).
* `forward_adjustment()` returns the log-effect contribution. ABSENT STATE:
  when no file is present the adjustment is exactly 0.0, `applied` is false,
  and the reason says "the counter-programming covariate contributes nothing".
  A channel in the EPG with no daypart history contributes 0.0 strength. An
  EPG carrying only the own channel yields no covariate at all (None).

## Artifacts and tests

* Candidate: `models/candidates/tv_break_coefficients_competitor.json`
  (competition-adjusted coefficients; de-confounding moves cells by mean
  0.000725, max 0.003088 in delta units; gate verdict and betas in metadata).
  If adopted it would move predicted weekly revenue by +565,489 (+0.263
  percent, measured in-memory on the exact recompute path), but the failed
  held-out gate means that movement is not evidence-backed; do not adopt on
  one month.
* Tests: `tests/test_counterprogramming.py` (hand-computed feature math,
  planted-effect gate activation, noise-stays-off, absent-state honesty,
  loud boundary failure), plus the pre-existing
  `tests/test_competitor_features.py` / `tests/test_competitor_model.py`.
* Measurement: `scripts/measure_counterprogramming.py`.
