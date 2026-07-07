# Exact DP for the Kairos per-day ad-break optimization

This lane implements and evaluates the exact dynamic program specified by the
mathematician for one Kairos channel-day, and measures it against the shipped
greedy and greedy+F1 optimizers on the engine's own objective and money basis.

Everything here reuses the engine's own primitives (break geometry, retention,
revenue, and the guardrail predicates), so "DP truth" equals "engine truth" by
construction. Nothing under `data/` was written; all outputs live in this folder.

## Files

- `dp_exact.py` - the production-shaped DP module. `dp_optimize_day(segs, ...)`
  exactly optimizes one channel-day (blend or revenue_net objective, default
  guardrails, risk_lambda pre-pass) or silently falls back to greedy+F1 when a
  runtime precondition fails.
- `brute_validator.py` - randomized exactness gate: DP vs exhaustive enumeration
  on tiny synthetic days, using the engine's own `is_compliant` and objective.
- `eval_12day.py` - the 12 real evaluation days: DP vs greedy vs greedy+F1, per
  day and total, both objective modes, with wall times and a compliance check.
- `debug_mismatch.py` - the harness used to localize the two exactness bugs below.
- `dp_prototype.py` - the pre-existing validated prototype (kept for reference).
- `out_validate.txt` - validator output across 4 seeds (1000 instances).
- `out_eval_12day.txt`, `out_eval_12day_risk1.txt` - the 12-day results tables.

## How to run

```
cd analysis/dp-exactness
PYTHONPATH=<repo>:. python brute_validator.py 250 20260707
PYTHONPATH=<repo>:. python eval_12day.py 0.6 0.0
```

(`<repo>` = /Users/home/Code/questo/meridian; interpreter =
/Users/home/.venvs/meridian/bin/python.)

## What was proven (correctness first)

`brute_validator.py` builds synthetic channel-days of 3 to 8 overlapping segments
under guardrail profiles that make every constraint type bind, then compares the
DP's objective to an exhaustive enumeration scored by the engine's own
`is_compliant` + objective, for both blend and revenue_net and across
risk_lambda in {0, 0.5, 1.0}.

Result (command: `brute_validator.py 250 <seed>` for seeds 20260707, 1, 42, 2024;
saved in `out_validate.txt`):

- 4 x 250 = 1000 instances, all on the exact DP path (0 fallbacks).
- MISMATCHES: 0. Worst |DP - brute| among all checked instances: 2.22e-16
  (machine epsilon), tolerance 1e-6.
- Guardrail codes exercised during the brute search (per 250-instance run):
  break_spacing ~235, breaks_per_hour ~202, hourly_ad_load ~193,
  retention_floor ~147, gold_breaks ~55, daily_ad_load ~38. All six constraint
  types were live in the test suite.

Two real exactness bugs were caught by the validator and fixed before any real-day
run (this is why correctness came first):

1. Sweep order. The DP core requires segments in start order (its closure lemma
   depends on it). The first wrapper passed them unsorted; the validator's random
   start order surfaced it immediately. Fix: sort by `start_seconds` inside
   `dp_optimize_day`.
2. Protected-hour cap. The engine's `check_hourly_ad_load` uses
   `limit = protected_max if protected else max_ad_seconds` (an if/else). The DP's
   local feasibility check applied `sec > max_ad_seconds` unconditionally AND the
   protected cap, which wrongly rejects a protected hour when
   `protected_max > max_ad_seconds`. With the shipped defaults (protected 480 <
   max_ad 720) the two forms are provably identical, so the prototype was correct
   on real data; the general fix makes the DP equal `is_compliant` for ANY
   guardrail config. Fix: mirror the engine's if/else exactly.

## The 12 real evaluation days (measured, not reproduced from a commit)

The four channels (kan 11 / keshet 12 / reshet 13 / akhshav 14) crossed with
2024-11-01..03. Each channel-day optimized three ways on identical settings
(`revenue_weight=0.6`, `risk_lambda=0.0`), scored on the engine's objective, per
day and total, with wall times. Full per-day table in `out_eval_12day.txt`.

Command: `eval_12day.py 0.6 0.0`

Blend objective (default shipped objective, `revenue_weight=0.6`):
- DP strictly better than greedy+F1 on 10/12 days, equal on 2, worse on 0.
- Total blend contribution: greedy 6.813903, greedy+F1 6.863900, DP 6.929271.
- On the net-ILS money basis (all three scored on net revenue): greedy+F1
  18,980,320.81, DP 19,901,902.35, so DP is +921,581.54 ILS (+4.86%) over
  greedy+F1 and +1,339,234.09 ILS (+7.21%) over pure greedy.

revenue_net objective (opt-in, maximises ILS net directly):
- DP strictly better than greedy+F1 on 10/12 days, equal on 2, worse on 0.
- Total net-ILS: greedy 18,274,569.11, greedy+F1 18,893,160.28, DP
  19,969,904.31, so DP is +1,076,744.03 ILS (+5.70%) over greedy+F1 and
  +1,695,335.20 ILS (+9.28%) over pure greedy.

Wall times (12 days, single timed run): greedy ~2.0s total, greedy+F1 ~3.5s
total, DP ~3.5s total (mean ~0.29s per channel-day; largest day n=96 at ~0.49s).
The DP is roughly the wall time of greedy+F1 and stays well under the spec's
~0.5s-per-day budget.

The +5.70% net-mode gain on these 12 days is consistent with the spec's full
120-day figure (+5.26%). This is a genuine independent 12-day measurement, not a
reproduction of the commit-message number (which was a one-off, per the task's own
note that no runnable 12-day harness exists).

The risk_lambda pre-pass path was also checked on the same 12 days at
`risk_lambda=1.0` (`out_eval_12day_risk1.txt`): DP never worse (worse=0 both
modes), all plans compliant, +3.47% net over greedy+F1.

## Primal feasibility and the (absent) duality gap

The spec does NOT use a Lagrangian relaxation. The two global channel-day budgets
(daily ad-seconds via the integer break-count state B, and gold-break count via
the state G) are carried as EXACT integer state dimensions, so the DP solves the
constrained primal directly. There is therefore no duality gap to report: the DP
optimum is a primal-feasible integer solution, not a relaxed bound.

This was verified operationally: on all 24 day-and-mode runs (12 days x 2
objectives) the reconstructed DP plan passed the engine's own `is_compliant`
(0 failures), and the DP was never worse than greedy+F1 on the true objective
(the shipped never-regress guarantee holds).

## Honest limits

- Scope validated end to end: the free path (no overrides, no pins, no demand
  weights), default guardrails, blend and revenue_net, risk_lambda. Days carrying
  overrides / placement pins / gold overrides are NOT handled by the exact core
  here; the module detects any uncovered case and falls back to greedy+F1, so it
  can never answer a case it does not cover and can never regress. The pins/gold
  folding is specified in the task but was not prototyped, and the real corpus
  carries none, so it is left behind the fallback rather than shipped unverified.
- Preconditions that trigger the safe greedy+F1 fallback: heterogeneous
  break_length across the day's segments (the daily budget would embed a 0/1
  knapsack and the small integer B state stops being exact); measured open-depth
  above the cap (default 10; the real corpus max is 6, so this never fired);
  a caller-supplied `revenue_scale` smaller than the DP's own revenue (which would
  make the objective's global [0,1] clamp bind, breaking the linear form). On the
  1000 validator instances and the 24 real day-runs the exact path always ran.
- Worst-case cost is exponential in the per-day open-depth (a data property), so
  the DP is polynomial only for bounded depth; the runtime guard makes this a
  detectable-and-fall-back condition rather than a blowup.
- `demand_weights` is a greedy-ranking-only placement bias with no objective
  semantics; the objective-exact DP ignores it (the task's own semantic decision).
- This lane is an analysis/evaluation deliverable. It does NOT wire the DP into
  `kairos/optimize/` as a third refiner tier; that production integration (behind
  the existing strictly-beats adoption gate) is the specified next step and would
  reuse `dp_exact.py` plus the standing `brute_validator.py` exactness gate.
