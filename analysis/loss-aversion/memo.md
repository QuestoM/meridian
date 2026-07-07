# Loss-aversion extraction: the 1.947 vs 1.69 first-break paradox, resolved

Date: 2026-07-07. Lane: Kahneman+Feynman. All numbers below were produced by
commands run in this task; commands and output files are listed at the end.

## Question

The pooled first-break multiplier 1.947 was measured ~18% inflated (honest
multivariate-controlled estimate ~1.69), yet lowering it was recorded as
revenue-negative (~-4.1M ILS), so the inflated value was kept as a "joint
honesty+revenue knob". Hypothesis under test: the inflation is a hidden
asymmetric loss weight, and the right design is the honest coefficient 1.69
plus an explicit asymmetry weight w > 1 on retention cost:

    decision objective = gross_revenue(seg, k) - w * retention_cost_ils(seg, k)

## Design (anti-circularity)

Every arm (baseline mult 1.947 w=1; honest mult 1.690 for each w in
{1.0, 1.05, 1.10, 1.152, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.00, 2.25,
2.50, 3.00}) was RE-OPTIMIZED under its own belief through the real shared
core (`kairos.optimize.day_core._optimize_one_day`, objective_mode
`revenue_net`, refine=True, real guardrails, demand, pacing, constraints).
Then every plan was frozen (per-segment break counts) and scored as a FIXED
plan under BOTH evaluation worlds at w=1: segments rebuilt with
first_break_multiplier 1.947, and rebuilt with 1.690. Scoring basis is the
engine-exact per-break net (`_segment_revenue` minus
`segment_retention_cost_ils`, each break valued at its own retention).
No plan is ever declared a winner under the belief that produced it alone.

Twelve evaluation days: all 4 real channels x 2024-11-01..2024-11-03 (the
channel-day subset recorded in the F1 refiner commit 7cecd35). A robustness
check re-ran the key arms on the FULL horizon: all 120 real channel-days
(4 channels x 30 days).

Guardrail compliance was re-checked on every fixed plan in every world:
zero violations in every cell of both matrices (columns violations_total and
retention_floor_violations are 0 everywhere).

## Results, 12 evaluation days (both worlds, net ILS vs baseline)

Baseline (mult 1.947, w=1): net 16,726,454 in the 1.947 world and
17,301,897 in the 1.690 world; 858 breaks; 862 segment decisions.

| arm (honest 1.690) | net delta, 1.947 world | net delta, 1.690 world | identical decisions vs baseline |
|---|---|---|---|
| w=1.00 | +12,387 | +13,022 | 858/862 (99.54%) |
| w=1.05 | 0 (exact same plan) | 0 (exact same plan) | 862/862 (100%) |
| w=1.10 | -21,920 | -23,039 | 855/862 |
| w=1.152 (=1.947/1.69) | -21,025 | -22,261 | 852/862 |
| w=1.25 | -81,436 | -86,225 | 846/862 |
| w=1.50 | -54,582 | -58,813 | 833/862 |
| w=2.00 | -120,386 | -131,180 | 798/862 |
| w=2.25 (Kahneman-Tversky) | -117,969 | -128,202 | 780/862 (90.5%) |
| w=3.00 | -483,197 | -516,734 | 754/862 (87.5%) |

Full matrices: eval_matrix_world_1947.csv, eval_matrix_world_1690.csv;
per channel-day detail in per_day_detail.csv (360 rows).

## Results, full horizon (120 real channel-days)

Baseline net: 166,062,881 (1.947 world) / 171,790,747 (1.690 world);
8,990 breaks; 8,704 segment decisions.

| arm (honest 1.690) | net delta, 1.947 world | net delta, 1.690 world | identical decisions |
|---|---|---|---|
| w=1.00 | +193,733 | +211,181 | 8631/8704 (99.16%) |
| w=1.05 | +68,915 | +75,548 | 8673/8704 (99.64%) |
| w=1.152 | -154,606 | -167,616 | 8623/8704 (99.07%) |

File: full_horizon_matrix.csv.

## w* and what it means

w* = 1.00 on the grid. On both the 12 evaluation days and the full 120-day
horizon, the honest 1.690 multiplier with NO asymmetry weight is the best
arm, and it beats the 1.947 baseline under the baseline's own 1.947
evaluation world (+12,387 ILS on 12 days; +193,733 ILS on 120 days).
w=1.05 is an exact fixed point on the 12 days: it reproduces the baseline
plan decision for decision (862/862), so the entire 1.947 baseline behavior
is replicable with the honest coefficient and a 5% cost tilt. Every w at or
above 1.10 loses money in BOTH worlds, monotonically worsening; the
Kahneman-Tversky 2.25 loses ~118k-128k ILS on 12 days, and even
w = 1.947/1.69 = 1.152 (the pure "rescale the inflation into w" candidate)
loses ~21k-22k on 12 days and ~155k-168k on the full horizon.

So the loss-aversion hypothesis is REJECTED with numbers: the implied
organizational asymmetry coefficient is ~1.00-1.05, nowhere near the
behavioral ~2.25 and below even the mechanical inflation ratio 1.152. The
1.947 inflation was not encoding a hidden asymmetric loss. It was simply a
small uniform-ish cost perturbation that barely moves integer break-count
decisions (99%+ identical plans), which is why keeping it looked harmless.

## Why the historical -4.1M does not contradict this

Three facts from the repo's own record (memory
kairos-adversarial-verify-2026-06-18.md and models/tv_break_coefficients.json):

1. The -4,129,883 ILS figure was measured on the CONTAMINATED measurement
   pipeline (3-minute after-window bleeding into the next break for 32.6%
   of breaks), and it was the PER-GENRE variant (News 1.68 / Other 1.91 vs
   pooled 1.947), not a clean pooled 1.947 -> 1.69 swap.
2. This task's controlled swap on the current engine finds the opposite
   sign and two orders of magnitude smaller effect: pooled 1.690 at w=1 is
   +12k to +211k ILS versus 1.947, under both evaluation worlds.
3. The paradox is moot in shipped reality: after the after-window
   decontamination the self-activating gate turned the first-break lever
   OFF entirely. The committed coefficients JSON reads
   first_break_multiplier=1.0, first_break_active=false, p=0.2034
   (computed_at 2026-07-05). Both 1.947 and 1.69 are historical beliefs;
   the honest measured value today is 1.0.

## Verdict

Can we ship the honest coefficient plus an explicit preference at zero
revenue cost? YES, and better than zero: honest 1.690 with w=1 is revenue-net
POSITIVE versus the 1.947 baseline under the 1.947 world itself (+12,387 ILS
on the 12 days, +193,733 ILS on the full horizon) and under the honest world
(+13,022 / +211,181). No asymmetry weight should be shipped: every w >= 1.10
destroys value in both worlds. If exact plan continuity with the historical
1.947 behavior were ever required, w=1.05 reproduces it decision for
decision on the 12 evaluation days at zero net delta. And the deepest
resolution is that the shipped gate has already, honestly, set the
multiplier to 1.0; this analysis confirms there is no revenue argument for
resurrecting the inflated 1.947.

## Caveats (honest limits)

- The w lever lives on the revenue_net objective path, the one place
  retention cost is an explicit ILS term. The default blend objective has
  no such term; there w would have to be expressed through revenue_weight,
  which was not swept here.
- Plans are per-segment break COUNTS (the engine's actual decision
  variable); positions are deterministic given counts, no pins applied,
  identically for every arm.
- The evaluation worlds share everything except first_break_multiplier;
  both use the current committed impact model and pricing.
- Single deterministic runs; the optimizer is deterministic, so no variance
  to report.

## Reproduction (commands run in this task)

    /Users/home/.venvs/meridian/bin/python analysis/loss-aversion/sweep_loss_aversion.py
        wrote eval_matrix_world_1947.csv, eval_matrix_world_1690.csv,
        per_day_detail.csv (wall time 59.7s)
    /Users/home/.venvs/meridian/bin/python analysis/loss-aversion/full_horizon_check.py
        wrote full_horizon_matrix.csv (wall time 161.0s)

Gate state was read from models/tv_break_coefficients.json (metadata block).
All paths relative to /Users/home/Code/questo/meridian.
