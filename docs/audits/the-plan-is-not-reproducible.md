# The plan is a function of machine load

> ## FIXED 2026-08-10
>
> The wall clock is no longer a planning input. The exact DP now falls back only
> on deterministic state and transition budgets. Elapsed seconds remain
> telemetry and cannot change counts. The hardest measured production day used
> 328,877 of 5,000,000 transition units, and the full 120 channel-day export
> reproduced the committed `1b9d4298...` plan byte for byte. pytest also forces
> the shipped plan read-only, and the golden now compares directly with the
> shipped CSV and fingerprint. The historical experiment below remains as the
> evidence for the defect and the reason for the repair.

> ## CORRECTED THE SAME DAY: THIS DOES NOT EXPLAIN THE ARTIFACT
>
> The four rewrites of `output/weekly_break_schedule.csv` were **NOT pollution
> and NOT this mechanism.** A fresh export run to a scratch path produced
> `d4573c0037f557dd2cfdb4badba57320`, **byte-identical to the file that kept
> appearing**, and different from the committed `6a5944b4a3e0504ca761c0bab937c598`.
>
> So the committed plan is **STALE**. Every recompute produced the correct
> current answer and I restored an out-of-date file over it four times.
>
> **CONFIRMED BY TWO BACK-TO-BACK EXPORTS.** Both full exports, run consecutively
> to separate scratch paths on the same machine, produced byte-identical output:
> `TWO RUNS IDENTICAL: True`. So the recompute IS deterministic under this
> machine's normal load, and the difference from git is drift, not a race.
>
> The evidence is six independent occurrences of the same bytes: four rewrites
> plus two deliberate consecutive exports. Byte-identical recurrence is evidence AGAINST
> load-randomness for this artifact, and the agent that proposed the mechanism
> downgraded its own explanation on exactly that ground before I measured it.
>
> **What survives below is a direct experiment and it stands:** varying only the
> DP wall budget changed the plan on five of six real channel-days, always for
> the worse. The plan is not reproducible ACROSS MACHINES OR LOADS. That is a
> real defect and decision 11 still needs an answer.
>
> **What does not survive:** the suggestion that it explains this artifact.
>
> **AND THE DRIFT IS OLDER THAN TODAY.** I extracted the exact tree at
> `fec20f1d`, the commit that COMMITTED the plan, and ran a full export in it.
> It produced `d4573c00` as well. **So the artifact committed in that commit was
> never what that commit's own code produced.** It had already been carried
> forward by restore rather than by export, across several commits, and the
> content `6a5944b4` appears twice in the file's history with different content
> in between.
>
> That is only possible because THE GOLDEN AND THE ARTIFACT ARE DIFFERENT THINGS.
> `tests/golden_weekly_schedule.py` asserts against its own embedded baseline, not
> against `output/weekly_break_schedule.csv`, so the golden can be green while the
> shipped artifact matches nothing. It was, and it has been.
>
> So the 17,966.31 is NOT attributable to this session's engine work.
>
> The measured drift between the committed plan and what the current tree
> produces: 68 of 8,704 segments carry a different break count, the break total
> is IDENTICAL at 9,026, and revenue is **17,966.31 lower**, of which 9,350.68 is
> on the operator's own channel. That is the effect of this session's engine
> changes on the plan, not a race.


Measured 2026-08-09. This is the answer to the question left open in commit
`9dbe5d31`, which asked why the same inputs were reaching a different and worse
plan.

**The exact DP tier aborts on a WALL CLOCK deadline.** So the plan this product
exports depends on how busy the machine was when it ran.

---

## The mechanism, quoted

`kairos/optimize/dp_refine.py:148-153`:

    def _check_wall(j: int) -> None:
        if time.perf_counter() > deadline:
            raise DPBudgetExceeded(
                "wall_budget",
                f"per-group wall budget of {wall_budget_seconds:.1f}s exhausted at segment {j} of {n}",
            )

with `kairos/optimize/dp_refine_prep.py:60`:

    DEFAULT_WALL_BUDGET_SECONDS = 5.0

applied **per channel-day group**. A group that finishes on an idle machine and
times out on a loaded one falls back to the labelled greedy result and therefore
**adopts a different plan**.

There is a second gate beside it, `max_states = 200_000` at `dp_refine.py:227`,
and that one is deterministic. The wall clock is the only non-deterministic gate
in the tier.

## The measurement

Real operator channel-days, varying NOTHING but the budget:

| day | segments | generous | starved | counts identical | revenue delta |
|---|---|---|---|---|---|
| 2024-11-01 | 82 | 1.17s | 0.66s | NO | -30,575.55 |
| 2024-11-02 | 75 | 0.99s | 0.56s | yes | 0.00 |
| 2024-11-03 | 91 | 1.20s | 0.48s | NO | -122,886.51 |
| 2024-11-04 | 88 | 1.89s | 0.88s | NO | -115,944.75 |
| 2024-11-05 | 84 | 1.51s | 0.49s | NO | -91,406.35 |
| 2024-11-06 | 64 | 1.06s | 0.59s | NO | -2,318.72 |

**Five of six channel-days produce a different plan, and the starved plan is
worse every single time, never better.** Segments carrying different break
counts: 15 of 82, 9 of 91, 21 of 88, 16 of 84, 9 of 64.

**The exception predicts itself, which is why this is believable.** The one day
that did not change, 11-02, is precisely the day whose `dp_stats` read
`groups_adopted 0, groups_not_better 1`. The DP had nothing better to offer
there, so starving it could not change anything.

## How close the real plan runs to the edge

On an idle machine the DP takes 0.48 to 1.89 seconds per channel-day against its
5.0-second budget, so the worst real day has about **2.6x headroom**. Under the
load this repository carried on the day of measurement, five agents plus
concurrent test suites, a 2.6x slowdown is a normal afternoon rather than an
exotic condition.

A full export is 4 channels x 30 days = **120 independent groups, each with its
own 5-second budget**, so an arbitrary subset can flip on any given run.

## What is proven and what is not

**PROVEN:** the mechanism exists, is reachable on real data, and produces exactly
the signature observed in the four plan pollutions, including the DIRECTION of
the revenue move. The polluted artifact each time had the same row count, the
same break count and the same total ad seconds to the second, with breaks
redistributed inside a channel-day and revenue slightly lower.

**NOT PROVEN:** that the specific polluting runs were caused by it. That would
need the load conditions at those moments, which are gone. This is the leading
explanation with a demonstrated mechanism, not a closed case.

## A second finding alongside it

`apply_dp_tier` does not accept or forward `wall_budget_seconds`. It is a
definition-time keyword default on `dp_refine_group` (`dp_refine.py:264`,
visible in `__kwdefaults__`). **So the budget is not settable by any caller** —
not by settings, not by the recompute endpoint. It cannot be raised for a
production export or lowered for a fast preview without editing the module.

## The recommended fix, NOT made

Replace the wall-clock gate with a **deterministic work budget**. The state-count
gate beside it already bounds compute and already does the job; counting expanded
states or segments instead of seconds would make the same inputs yield the same
plan on any machine at any load.

That is what "the plan is reproducible" has to mean before anything else about it
can be trusted, including every golden test.

**It is not made here because it moves real money on the days where the DP
currently times out.** Owner decision.

## The trap this was nearly lost to, for the third time in one day

The first probe of this **reported no effect and was vacuous.** It patched the
module constant `DEFAULT_WALL_BUDGET_SECONDS` and re-ran, getting "0 of 6 days
changed". That was a no-op: the value is a keyword default bound at
function-definition time, so patching the module attribute after import changes
nothing.

What gave it away was the TIMING. The "starved" run took 1.01s against the
generous 0.94s, which is impossible if the budget were really 0.01s. Patching the
actual `__kwdefaults__` binding inverted the result from 0 of 6 to 5 of 6.

Same class as the empty-evening cap fixture and the unbound lever fixture: **a
probe returning a comfortable pass because the knob it thought it was turning was
not connected.** The rule in `tests/lever_probe.py` — refuse to rule on a fixture
you cannot show is binding — applies to determinism probes too. Assert the
manipulated knob actually changed the behaviour, by timing or by counter, before
believing a null result.

## Consequence for every measurement in this repository

If the golden ever exercises the DP tier near its budget, **the golden is
load-sensitive too**. A single green run is weaker evidence than it looks, and
that is worth knowing before any future finding rests on one.
