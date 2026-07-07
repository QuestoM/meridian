# Quarter-hour billing: the settlement currency, measured and expressed

Status: owner-provided market convention recorded 2026-07-07, MEASURED the same
day on the real Nov-2024 month (two lanes: analysis/quarter-hour/settlement and
analysis/quarter-hour/dynamics), and now EXPRESSED in the engine as an
owner-gated revenue-basis option (kairos/optimize/qh_billing.py, activation
flag `pricing_activation.qh_settlement`, OFF by default). Every claim below is
labeled owner-stated, code-fact, or measured with its artifact path.

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
higher average than if the entire break sat inside one quarter hour.

Measured confirmation of the rule in the plan data (settlement lane Q4,
analysis/quarter-hour/settlement/planned_tvr_results.json): in the one real
daily plan file (Reshet 13, 2025-04-27, 175 spot rows, 10 breaks), every
occupied quarter-hour window carries exactly one distinct planned_tvr, all 8
value changes bracket a :00/:15/:30/:45 boundary, and two changes happen
MID-BREAK exactly at :45 (6.4 to 5.9 between spots at 20:44:30 and 20:45:05;
3.9 to 4.3 between 22:44:36 and 22:45:06). planned_tvr is a round-quarter-hour
figure, not a per-break or per-programme one. Caveat: one plan file, one
channel, one evening; generalization unverified (no other files exist in
data/daily_input).

## What the month of real data measured (2026-07-07)

All artifacts under analysis/quarter-hour/ (settlement and dynamics lanes,
Nov 2024, 4 channels, 30 days, 172,800 channel-minutes, 5,775 keyed breaks).

Scheduler behavior (settlement lane Q1, straddle_results.json,
conditional_null_results.json): 17.58 percent of breaks straddle a
quarter-hour boundary, 7.62 percent a half-hour one. Against the honest
programme-conditional null (break start uniform within its real containing
programme, the content-constraint complication) there is a small but
significant EXCESS of straddling (observed 17.54 vs expected 16.70 percent,
z = +2.01), concentrated in short breaks (1-4 minutes, roughly +2 to +5
points) and on Now 14 and Reshet 13; 6-9 minute breaks are contained MORE
than chance. The naive uniform null shows the opposite (avoidance) and is a
programme-junction artifact: 20.75 percent of programme starts sit within one
minute of a quarter-hour mark vs 13.3 uniform. Verdict: at most weak boundary
optics; programme structure dominates observed placement.

Shared windows are the norm (settlement lane Q2, windows_results.json): 60.3
percent of ALL breaks share at least one quarter-hour window with another
break (Keshet 12: 75.0, Now 14: 72.5, Reshet 13: 61.7, Kan 11: 11.6 percent);
33.7 percent of break-carrying quarter-hour windows hold 2 or more breaks.
Per occupied half hour, Keshet 12 averages 2.05 breaks (68 percent hold 2+).
Boundary placement is a JOINT problem over co-window breaks for about two
thirds of inventory.

Size of the boundary lever (settlement lane Q3, settlement_results.json):
matched (channel x daypart x length-bin) observational effect of straddling on
the billed window average is +0.26 percent of content level (53 cells, 34
positive; the naive pooled comparison REVERSES purely because straddlers are
longer). Mechanical ceiling by length bin (even straddle vs containment, from
each bin's measured median in-break dip): under 1 minute 0.07 percent, 1-2m
0.21, 2-3m 0.43, 3-4m 0.79, 4-6m 1.05, 6-9m 2.15, 9m+ 6.85 percent of content
level; containment window deficits are double these. The lever is negligible
for the modal 1-3 minute break and material (1-7 percent) only at 4+ minutes.

Leave/return dynamics and the optimal position (dynamics lane,
analysis/quarter-hour/dynamics/VERDICT.md, results.json, placement.csv,
3,748 measured breaks): there is NO cliff at break start (the first full break
minute holds 95.2 percent of the pre-break level; loss builds at 1-2 points
per minute and long breaks bottom around minute 4, with a real U-shape on
5-7 minute breaks, bottom -9.7 percent), and the return is a fast snap-back
(pooled +1 minute 0.969, +2 0.985, +3 0.998; long breaks reach within about 1
percent in 5-7 minutes). The owner's asymmetry conjecture is answered: the
asymmetry is real but moves the optimal break position AT MOST one minute
from symmetric straddling, only for touched-length 6+ (worth +0.0004 relative
billed points, 77 percent bootstrap support at L=6, 0 percent at L=3). For
practical purposes SYMMETRIC boundary-straddling is optimal. The economically
real lever is straddle vs contain: +0.24 (L=2) to +2.0 (L=7) percent of billed
rating per break minute, growing with length, robust across all four detrend
variants. The uncapped as-measured profile flips short-L optima by exploiting
a non-causal post-break ramp and must NOT be used for pricing.

Measured in-break dips (the module's audience model,
settlement_results.json key dip_frac_by_len_bin, medians over 5,747 breaks):
under 1m 0.0377, 1-2m 0.0434, 2-3m 0.0548, 3-4m 0.0677, 4-6m 0.0620, 6-9m
0.0908, 9m+ 0.2040 (fraction of surrounding content level; the 3-4m value
sitting above 4-6m is the measured reality, kept as is).

## What the engine does today (code facts, updated 2026-07-07)

- Retention measurement (kairos/model/measure.py): shed is computed from
  minute-level TVR in windows just before and just after each break. It is NOT
  quarter-hour based, so retention estimates measure true minute-level
  audience behavior and are clean of the quarter-hour averaging artifact.
  Adding QH-aware revenue double-counts nothing on the cost side.
- Revenue basis (kairos/optimize/objective.py, break_revenue): revenue equals
  cpp * rating_points * duration_units * premium, where rating_points is
  baseline_tvr times realised retention. On the daily-plan path baseline_tvr
  is the programme MEAN of the plan's quarter-hour planned_tvr values
  (kairos/data/transform.py), so the revenue LEVEL is already
  quarter-hour-derived, but the within-programme window steps (6.4 vs 4.4 in
  the same show) and the position-vs-boundary lever are averaged away.
- Settlement currency (kairos/optimize/qh_billing.py, NEW): billed_points
  assigns every placed break's seconds to their round quarter-hour windows,
  computes each window's average TVR from the segment lineup (content at the
  segment baseline, break seconds diluted by the measured median dip for that
  break length), and bills each break at its windows' averages with the same
  price stack. restate_on_billed_points restates a finished schedule's
  revenue onto that basis; maybe_restate gates it behind
  pricing_activation.qh_settlement (PricingModel.enable_qh_settlement),
  wired at the single shared seam every optimize path uses
  (kairos/optimize/day_core.py), and is an exact identity (the same result
  object) while the flag is off. Tests: tests/test_qh_billing.py.

## Design: how QH settlement enters the optimizer

Decided 2026-07-07 from the measurements above. Three pieces, in order of
what the evidence supports.

1. Billed points as a revenue-basis option: SHIPPED, owner-gated OFF.
   The mechanic is confirmed in the plan data and the dips are measured, so
   the currency is computable today. With the flag on, the schedule the
   optimizer already chose is REVALUED in the settlement currency: every
   break bills at its round-window average (own dip diluted by content,
   co-window breaks compounding in one average, straddles split across two
   windows). Break counts, positions, retention and guardrails are untouched;
   the objective is recomputed from the restated revenue with the same weight
   and scale. It ships OFF because switching the basis moves real reported
   revenue (the window average sits ABOVE the engine's retention-discounted
   rating: on the real first channel-day, Kan 11 2024-11-01, activating the
   flag restates total revenue from 826,743.62 to 888,368.91 ILS, +7.45
   percent, with retention, break counts and placements identical),
   exactly the position/ad_type activation discipline. Activation is a
   deliberate owner decision in config or dashboard pricing overrides.

2. A boundary-placement lever: RULED OUT as a near-term optimizer knob,
   RULED IN as a documented placement guideline. What the measurements
   settled: symmetric straddling is optimal (no asymmetric-timing term is
   needed; the leave/return asymmetry buys at most one minute and +0.0004
   relative billed points at L>=6), and the straddle-vs-contain gain is worth
   pricing only for 4+ minute breaks (1-7 percent of billed rating). What
   blocks a knob: programme content constraints (which cut points are
   permissible) are not in our data, so an optimizer that slides breaks onto
   boundaries would optimize an unconstrained fiction; and 60.3 percent of
   breaks share windows, so placement is a joint problem the single-break
   answer does not solve. If it is ever built, it belongs inside the
   optimizer's placement geometry (today: even spacing or operator pins), fed
   by real cut-point data, evaluated with qh_billing.billed_points as the
   scorer, and owner-gated like everything that moves revenue. Until then the
   honest guideline for schedulers: straddle long (4+ minute) breaks
   symmetrically when content allows; do not bother re-timing 1-3 minute
   breaks for boundary optics.

3. Interaction with the consolidation finding: the two currencies must enter
   ONE objective before any consolidation knob ships. The convexity lane
   (analysis/convexity/VERDICT.md, adversarially verified PARTIALLY CONFIRMED
   in analysis/convexity/verify/VERIFY_MEMO.md) says consolidating breaks
   retains more TRUE audience (about +0.02 to +0.05 log points per avoided
   interruption, with 10-30 percent sign-error risk after the within-channel
   attack). Settlement optics pull the other way at exactly the same lengths:
   consolidation concentrates a deeper dip inside fewer windows, and the
   dilution gain from spreading is 1-7 percent of billed rating at 4-9
   minutes, the SAME order of magnitude as the consolidation saving there.
   Resolution shipped here: kairos/optimize/qh_billing.py makes the
   settlement side of that trade computable per schedule, so a future
   consolidation term can be evaluated in both currencies (true audience via
   the retention model, billed points via billed_points) before it earns a
   knob. Per the owner directive, the consolidation knob itself is NOT
   touched by this work.

What the measurements ruled in and out, in one list:

- Ruled in: the round-quarter-hour rule itself (plan file evidence); the
  computable billed-points currency with measured median dips; shared-window
  coupling as a first-class accounting fact; straddle-beats-contain as a
  directional fact growing with break length; symmetric straddling as the
  practical optimum.
- Ruled out: an asymmetric-timing term (worth 0.0004 relative billed points
  at best); a boundary-placement optimizer knob without content cut-point
  data; using the uncapped post-break ramp for pricing (selection, not
  causation); treating boundary optics as a large scheduler motive (the
  programme-conditional excess is +0.8 points overall); any change to the
  consolidation verdict (nothing here overturns consolidate-over-split on
  true audience).

## Open questions that remain

- Which ratings source is contractual, and whether overnight consolidated
  figures replace live ones at settlement. Unreachable in our data.
- Whether planned quarter-hour values already embed the planner's expected
  break dips (would make the billed restatement partially double-count the
  dip when driven from planned_tvr). Unknown from one plan file.
- Generalization of the planned_tvr basis beyond the single Reshet 13 plan
  file (within-file evidence is unambiguous; other channels/days unverified).
- Within-programme window steps: carrying per-window planned TVR through
  build_segments_from_daily_input instead of the programme mean would let
  billed_points use the plan's own window values. A future increment; today
  the module models windows from the programme-mean baseline, which keeps the
  LEVEL right and the geometry honest but flattens window-to-window steps.

## Note for the Express design (reads this doc before working)

Any billed-points computation must be per spot, not per break: assign each
spot's seconds to their round quarter hour, bill each spot at its own window's
average TVR, and apply the existing premium layers on the CPP side unchanged.
Do not model a straddling break as two breaks; the break stays one scheduling
entity with per-spot window assignment underneath.
kairos/optimize/qh_billing.py implements this at break granularity (the
engine's planning entity); the per-spot refinement belongs in the daily spot
path where individual spots exist.

## Where the code points here

Header notes referencing this file live in kairos/optimize/objective.py
(revenue basis), kairos/model/measure.py (measurement basis) and
kairos/optimize/qh_billing.py (settlement currency). Keep those pointers when
refactoring.
