# Model validation panel: integrated verdict

Date: 2026-07-05. Three independent referee reviews were computed against the
real November 2024 data and the live engine, each with seeded, re-runnable
scripts under `scripts/validation/` and standing tests under `tests/validation/`:

- `causal-identification.md` (placebo, selection, inference, stability)
- `uncertainty-calibration.md` (coverage, calibration, pooling recovery, seeds)
- `decision-robustness.md` (plan sensitivity, regret, value of model, risk_lambda)

This file is the panel synthesis: what the three verdicts mean TOGETHER, and
the single ranked program that follows. Every number below is computed in one
of the three reports.

## The three verdicts interlock

1. The LEVEL is understated. The placebo experiment (6,141 matched pseudo
   breaks through the exact shipped machinery) shows the measurement does not
   fabricate cost; instead it reveals a within-show audience-build drift of
   +0.015 that the shipped estimate absorbs. The placebo-corrected causal cost
   is -0.0533 against the shipped -0.0391: every break is under-charged by a
   factor of about 1.365 (permutation p = 0.0005; leave-one-out always
   negative; selection bias only about 5 percent and opposite-signed).
2. The INTERVALS answer a narrower question than the product implies. The
   shipped per-cell ci is a latent-cell-mean band: read that way it is
   holdout-consistent, but it covers only 4.9 percent of individual break
   outcomes at nominal 95 (single-break sd about 40x the posterior sd), and the
   tau2 estimate sits on a knife edge at this sample size (P(tau2-hat = 0) =
   0.36, +59 percent bias when nonzero). Honest 95 percent cell intervals need
   about 1.77x widening today; the recovery study shows this self-heals with
   data (1.16x at 12 months, 1.07x at 24), so it is a disclosure-and-patch
   problem, not an architecture problem. The one structural flaw is
   heteroskedasticity (real cell variances span 23x; 67 percent coverage in
   that scenario even at scale).
3. The DECISIONS do not care, today. Across 200 coefficient draws the plan is
   identical to shipped in 98.5 percent of draws and total breaks never move;
   worst-regime regret is 0.22 percent of net at P90. The 36-cell structure
   and a single pooled constant produce the same plan (+0.00 ILS), because
   only 12 of 36 cells are reachable by the plan wiring and they collapse to 4
   class coefficients. But retention modeling as a whole earns +11,926 ILS per
   day (+0.84 percent) against ignoring retention. risk_lambda does not change
   the plan at any setting: it is honest bookkeeping, not tail protection.

Together: the decision layer is robust and needs nothing urgent; the valuation
layer is materially off. Applying the placebo correction to the current plan
implies the true retention cost headline is about 22.9M ILS per week rather
than 16.8M (revenue-net about 192.4M rather than 198.5M), with a draw-implied
error bar of about plus-minus 1.8M per week. Until the correction ships, the
model card carries this as a disclosed known bias.

## The ranked program

P1. Placebo-drift correction layer in the coefficient rebuild (from
    causal-identification fixes 1 and 2, which must ship together: the
    content-only baseline alone moves AWAY from truth). Implementation: a
    gated correction in the measurement pipeline (per-genre drift subtraction
    measured from pseudo-breaks on each rebuild), candidate artifact first,
    measured plan and revenue movement, adoption as an owner-visible decision.
    Expected effect: coefficients scale by about 1.365; plan movement expected
    small (decision-robustness shows counts are guardrail-pinned) but must be
    measured, and the revenue-net headline moves by about -6.1M per week.
P2. Interval honesty (uncertainty-calibration fixes 1 and 3): document the ci
    columns as cell-mean bands, add a predictive band for single-break
    questions (measured 0.938 coverage at 95 when built), and widen the 95
    percent cell intervals by about 1.8x now (or parametric-bootstrap tau2),
    with the width factor re-measured at each data drop.
P3. Moderated per-cell variances, limma style (uncertainty-calibration fix 2),
    the one structural fix that data growth does not solve.
P4. Keep and disclose (decision-robustness): keep the 36-cell EB model (costs
    nothing, gates protect it, data growth activates it), keep risk_lambda
    default 0 and stop describing it as tail protection, add the
    decision-irrelevance and 12-reachable-cells disclosures to the model card.
P5. Operations: a weekly level-drift control chart (drift +0.0202 per week is
    about twice the 95 percent half-width, nonstationarity binds before cell
    ordering); move the gate statistic to 5-fold or seed-averaged (verdicts
    unchanged today, tighter at the margin); at the two-year drop re-run the
    counter-programming and seasonality gates (already wired) and revisit the
    Kan-11 concentration (19.4 percent single-channel influence).

## Standing protection

- `tests/validation/test_placebo_fast.py` re-runs a seeded placebo subsample.
- `tests/validation/test_parameter_recovery_smoke.py` re-runs seeded pooling
  recovery within tolerances.
- `tests/validation/test_decision_robustness.py` asserts draw determinism,
  replay-equals-CSV, and the repricing band.
- Each report's scripts rewrite their own marker-managed sections, so the
  documents cannot silently drift from the code that produced them.
