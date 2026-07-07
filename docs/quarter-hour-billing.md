# Quarter-hour billing: the settlement currency the engine does not yet model

Status: owner-provided market convention, recorded 2026-07-07. Not yet measured
in our data and not yet modeled anywhere in the engine. This document exists so
the fact is not lost; every claim below is labeled as owner-stated, code-fact,
or open question.

## The market mechanic (owner-stated, refined 2026-07-07)

The settlement rule is PER SPOT, and it has two independent sides:

1. Rating side: the billable viewing points of a spot are the average TVR of
   the pure ROUND quarter hour (:00, :15, :30, :45) in which that spot airs.
   That average includes the surrounding programme-content minutes. This is how
   "how many viewing points were there at that moment" is determined, for every
   second of the spot.
2. Price side: the cost per point is then modulated by everything else, spot
   position within the BREAK (not within the quarter hour), programme, break
   type, and so on. These are exactly the premium layers the engine already
   models in kairos/optimize/pricing.py.

A break does NOT administratively split into two breaks when it straddles a
boundary. But its spots bill by their own quarter hour: spots airing before the
boundary get the first window's average, spots after it get the second's. So a
straddling break spreads its audience dip across two settlement windows, each
diluted by high-rated content minutes, and the spots in each window bill at a
higher average than if the entire break sat inside one quarter hour. Schedulers
know this and place breaks accordingly. This is measurement optics on the
settlement currency, not viewer behavior.

Engine gap, stated precisely: the premium side is already modeled; the missing
piece is ONLY the rating basis. The engine bills a spot on the break's or
programme's own planned rating (baseline_tvr), where the market bills it on the
round-quarter-hour average of the spot's own window.

## What the engine actually does today (code facts)

- Retention measurement (kairos/model/measure.py): shed is computed from
  minute-level TVR in windows just before and just after each break. It is NOT
  quarter-hour based, so our retention estimates measure true minute-level
  audience behavior and are clean of the quarter-hour averaging artifact.
- Revenue (kairos/optimize/objective.py, break_revenue): revenue equals
  cpp * rating_points * duration_units * premium, where rating_points comes
  from baseline_tvr, the mean planned break rating of the programme
  (kairos/data/transform.py). The engine therefore bills in the currency of
  the break's own rating, not the round-quarter-hour average the market
  settles on.
- Nothing in the repo is aware of quarter-hour boundaries. As of 2026-07-07
  the word "quarter" appears once, in an unrelated context in
  docs/model-validation/causal-identification.md.

## Why this matters, in priority order

1. Split-vs-consolidate advice can optimize the wrong currency. The convexity
   finding (analysis/convexity/VERDICT.md, verified partially-confirmed in
   analysis/convexity/verify/VERIFY_MEMO.md) says consolidating breaks retains
   more true audience. But consolidation concentrates the dip inside one
   settlement window, while the split configuration hides it across two. A
   consolidation term added to the optimizer without quarter-hour-aware
   revenue could raise true retention while lowering BILLED points. The two
   effects must be priced in the same objective before any consolidation knob
   ships.
2. It is an unmodeled selection mechanism behind observed scheduling. Channels
   split and position breaks partly for boundary optics. Any causal reading of
   observed break patterns (including the convexity lane and the first-break
   work) has this as an omitted motive on the treatment-assignment side.
3. Boundary placement is a potential real revenue lever. If settlement truly
   follows round quarter hours, shifting a break a few minutes to straddle a
   boundary changes billed points at zero audience cost. The optimizer cannot
   see this lever today.

## Open questions before modeling it

- Is planned_tvr in the weekly plan already a quarter-hour-based figure, or a
  per-break forecast? (Determines how wrong the current revenue basis is.)
- Settlement rule: ANSWERED by the owner 2026-07-07, see the mechanic section
  above (per-spot round-quarter-hour average on the rating side, premiums on
  the price side). Still open: which ratings source is contractual, and
  whether overnight/consolidated figures replace live ones at settlement.
- Measurable in our data now: from the Nov-2024 minute-level TVR, quantify how
  much boundary-straddling actually moves quarter-hour averages for real
  breaks, and how often schedulers straddle vs contain. That turns this from
  convention into a measured effect size. (In progress: the
  quarter-hour-expression measurement wave writes to analysis/quarter-hour/.)

## Note for the Express design (reads this doc before working)

Any billed-points computation must be per spot, not per break: assign each
spot's seconds to their round quarter hour, bill each spot at its own window's
average TVR, and apply the existing premium layers on the CPP side unchanged.
Do not model a straddling break as two breaks; the break stays one scheduling
entity with per-spot window assignment underneath.

## Where the code points here

Header notes referencing this file live in kairos/optimize/objective.py
(revenue basis) and kairos/model/measure.py (measurement basis). Keep those
pointers when refactoring.
