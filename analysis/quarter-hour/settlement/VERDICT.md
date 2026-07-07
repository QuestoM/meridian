# Quarter-hour settlement mechanics in the real Nov-2024 data: verdict memo

Date: 2026-07-07. Lane: analysis/quarter-hour/settlement. Every number below
comes from commands run in this task on the real reference data (Spots.xlsx,
Programmes.xlsx, Dayparts.xlsx, Nov 2024, 4 channels, 30 days) and the single
daily-input plan file (Wally_Prime_Reshet_Example_2025-04-27.csv). Read
docs/quarter-hour-billing.md first (owner-stated per-spot round-quarter-hour
settlement rule).

Reproduce (in order; the last three read breaks_qh.csv written by the first):

    PYTHONPATH=/Users/home/Code/questo/meridian /Users/home/.venvs/meridian/bin/python analysis/quarter-hour/settlement/prep_breaks.py
    PYTHONPATH=/Users/home/Code/questo/meridian /Users/home/.venvs/meridian/bin/python analysis/quarter-hour/settlement/conditional_null.py
    PYTHONPATH=/Users/home/Code/questo/meridian /Users/home/.venvs/meridian/bin/python analysis/quarter-hour/settlement/settlement_effect.py
    PYTHONPATH=/Users/home/Code/questo/meridian /Users/home/.venvs/meridian/bin/python analysis/quarter-hour/settlement/planned_tvr_basis.py

Outputs: breaks_qh.csv (5,775 keyed breaks with straddle flags),
breaks_settlement.csv (5,747 breaks with billing proxies),
straddle_results.json, conditional_null_results.json, windows_results.json,
settlement_results.json, planned_tvr_results.json, run logs (*_run.log).

## Q1 verdict: schedulers are NOT visibly chasing boundary optics at scale.
## Against the honest programme-respecting null there is a small but
## significant EXCESS of quarter-hour straddling in short breaks; the naive
## null shows the opposite (avoidance) and is a programme-junction artifact.

5,775 detected breaks (engine definition, min 2 spots). 1,015 (17.58 percent)
cross a :00/:15/:30/:45 boundary; 440 (7.62 percent) cross a :00/:30 boundary.

Two nulls, same length distribution per break:

- Uniform-within-hour null (placement anywhere on the clock): expected QH
  straddle 18.32 percent vs observed 17.58, z = -1.61 (mild avoidance);
  half-hour expected 9.16 vs observed 7.62, z pooled about -4 across channels
  (Keshet 12 z = -2.41, Reshet 13 z = -3.07, Now 14 z = -2.53). Looks like
  deliberate containment.
- Programme-conditional null (start uniform within the break's real containing
  programme, the placement-must-respect-content complication): expected QH
  straddle 16.70 vs observed 17.54 percent, z = +2.01. The avoidance
  DISAPPEARS and flips to mild excess. Half-hour: expected 8.06 vs observed
  7.62, z = -1.37 (neutral).

Why the flip: programme junctions cluster on round marks (20.75 percent of
programme starts sit within 1 minute of a QH mark vs 13.3 uniform; 16.08
percent within 1 minute of a half-hour mark vs 6.7 uniform), and breaks cannot
cross junctions. The naive null misreads that content constraint as
boundary avoidance. This is exactly the owner's complication 2 in action.

Where the conditional excess lives:

| slice        | n     | observed | conditional-expected | z     |
|--------------|-------|----------|----------------------|-------|
| all          | 5,759 | 0.1754   | 0.1670               | +2.01 |
| 1-2m         | 1,554 | 0.1075   | 0.0890               | +2.68 |
| 2-3m         | 756   | 0.1587   | 0.1381               | +1.70 |
| 3-4m         | 315   | 0.2444   | 0.1979               | +2.22 |
| 4-6m         | 730   | 0.3014   | 0.3042               | -0.18 |
| 6-9m         | 615   | 0.4228   | 0.4621               | -2.12 |
| Now 14       | 960   | 0.0979   | 0.0751               | +2.83 |
| Reshet 13    | 1,701 | 0.2475   | 0.2320               | +1.82 |
| Kan 11       | 940   | 0.1372   | 0.1222               | +1.47 |
| Keshet 12    | 2,158 | 0.1696   | 0.1762               | -0.96 |

Reading: short breaks (1-4 minutes) straddle boundaries about 2-5 percentage
points more than programme-respecting chance; 6-9 minute breaks are contained
MORE than chance (z = -2.12). Channel differences are real but modest: the
excess is clearest on Now 14 and Reshet 13, absent on Keshet 12. The start
offset histogram (minute mod 15) is nearly flat (0.054-0.074 per minute vs
0.067 uniform), so there is no visible pile-up just before boundaries. Bottom
line: at most WEAK boundary optics, not the systematic straddling the
owner-stated convention implies schedulers practice; the dominant placement
force in this data is programme structure.

## Q2 verdict: shared settlement windows are the NORM on the commercial
## channels. Half of occupied quarter hours on Keshet/Now hold 2+ breaks.

Breaks per occupied window (window counted when >= 1 break overlaps it; a
straddling break counts in each window it touches), from windows_results.json:

Quarter hours: Kan 11 mean 1.06 breaks per occupied window (5.3 percent hold
2+), Reshet 13 1.38 (32.4 percent 2+), Keshet 12 1.57 (47.2 percent 2+),
Now 14 1.60 (47.5 percent 2+, 10.6 percent 3+).

Half hours: Kan 11 1.18 (17.3 percent 2+), Reshet 13 1.68 (48.4 percent 2+),
Now 14 1.77 (56.3 percent 2+), Keshet 12 2.05 (68.0 percent 2+, 27.0 percent
3+). By daypart (pooled channels), 2+ breaks per occupied half hour ranges
from 25.0 percent overnight to 65.0 percent morning; prime slices sit at
42-48 percent.

Break-level view: 60.3 percent of ALL breaks share at least one quarter-hour
window with another break (Keshet 12 75.0, Now 14 72.5, Reshet 13 61.7,
Kan 11 11.6); 72.1 percent share a half-hour window. The owner's "2+ breaks
in a half hour is common" is confirmed and understated for Keshet 12. Any
boundary-placement optimization is therefore a JOINT problem over co-window
breaks for roughly two thirds of inventory, not a per-break knob.

## Q3 verdict: matched observational effect of straddling on the billed
## quarter-hour average is positive but tiny (+0.26 percent of content
## level); the mechanical ceiling is 0.2-2.2 percent for standard lengths.

Design: for each of 5,747 measurable breaks, B = mean TVR of non-ad minutes
within 15 minutes either side of the break (min 5 content minutes); billed =
mean over the break's minutes of the FULL 15-minute round-window average
containing that minute (the per-spot settlement rule approximated at minute
grain); dilution = billed / B. Straddled vs contained compared inside matched
(channel, daypart, length-bin) cells with >= 5 breaks of each kind.

- Naive pooled: contained dilution 0.9765, straddled 0.9648. The straddled
  breaks look WORSE naively because straddlers are much longer (a length
  composition artifact, not the mechanic).
- Matched: 53 cells, straddled minus contained weighted mean +0.0026 (34 of
  53 cells positive). So at equal channel, daypart and length, a straddled
  break's spots bill about 0.26 percent of content level higher than a
  contained break's. Real but near the noise floor of this design.
- Limits, stated honestly: placement is scheduler-chosen (no randomization);
  B is estimated from minutes adjacent to the break and is itself depressed
  when neighbouring breaks sit in the window; matching is coarse (length BIN,
  not exact length, so residual within-bin length confounding remains and
  straddlers skew longer within bins, biasing the matched estimate DOWN);
  minute grain approximates per-spot billing.

Mechanical bound (uniform-dip even straddle vs containment, per length bin,
using each bin's measured median in-break dip d and median length L; gain =
d * L / 30 as a fraction of B):

| len bin | n     | median dip | max straddle gain (frac of B) | containment window deficit |
|---------|-------|------------|-------------------------------|-----------------------------|
| <1m     | 1,643 | 0.0377     | 0.0007                        | 0.0014                      |
| 1-2m    | 1,544 | 0.0434     | 0.0021                        | 0.0042                      |
| 2-3m    | 746   | 0.0548     | 0.0043                        | 0.0087                      |
| 3-4m    | 314   | 0.0677     | 0.0079                        | 0.0158                      |
| 4-6m    | 728   | 0.0620     | 0.0105                        | 0.0209                      |
| 6-9m    | 619   | 0.0908     | 0.0215                        | 0.0431                      |
| 9m+     | 153   | 0.2040     | 0.0685                        | 0.1369                      |

Reading: for the modal 1-3 minute break the boundary lever is worth AT MOST
0.2-0.4 percent of billed rating, which is why Q1 finds only weak optics.
The lever becomes material (1-7 percent) only for 4+ minute breaks, exactly
the lengths where consolidation concentrates the dip; this is the currency
tension the convexity lane (analysis/convexity/VERDICT.md: consolidation
saves 2-5 percent of true audience per avoided interruption, verified
partially-confirmed) must be priced against. The two effects are the same
order of magnitude at 4-9 minute lengths.

Owner complication 1 (asymmetric optimal position) is NOT measured here: the
even-straddle bound assumes a uniform within-break dip. The within-break
minute profile (when viewers leave after break start and return after content
resumes) is measurable from this same minute data but was out of scope; the
matched +0.0026 reflects actual (asymmetric) scheduler placements.

## Q4 verdict: planned_tvr IS a round-quarter-hour figure. The engine then
## averages it per programme, erasing the window structure it already had.

Evidence from the one real daily plan file (175 spot rows, 10 breaks, 4
programmes, Reshet 13, 2025-04-27, planned_tvr_results.json):

- All 10 occupied quarter-hour windows carry EXACTLY ONE distinct planned_tvr.
- All 8 value changes bracket a :00/:15/:30/:45 boundary. Two of them happen
  MID-BREAK exactly at the boundary: 6.4 -> 5.9 between spots at 20:44:30 and
  20:45:05, and 3.9 -> 4.3 between 22:44:36 and 22:45:06 (the same break's
  spots bill differently on each side of :45, precisely the owner's per-spot
  rule).
- Spots in the same quarter hour but DIFFERENT breaks share the value (the
  21:15 window holds 3 breaks, all at 5.4; the 22:45 window holds 2 breaks,
  both at 4.3).
- planned_tvr is NOT constant per break (8 of 10 breaks single-valued) and
  NOT constant per programme (1 of 4 programmes single-valued).

So the plan's granularity is the settlement granularity: constant within
round :15 blocks, stepping only at boundaries. Code facts on what the engine
does with it: build_segments_from_daily_input (kairos/data/transform.py)
collapses planned_tvr to the MEAN over the programme's spot rows, and
break_revenue (kairos/optimize/objective.py) bills every break of the
programme at that one scalar. The Programmes.xlsx path is coarser still: TVR
is one value per programme row (8,704 rows; a 72.5-row channel-day averages
39.7 distinct values, confirming per-programme granularity). Consequence: the
engine's revenue LEVEL is already quarter-hour-derived (less wrong than a
pure per-break forecast), but the within-programme window-to-window variation
and the position-vs-boundary lever are averaged away. Caveat: this rests on
ONE plan file, one channel, one evening; whether the planned QH values
already price in the dip of planned break placement is UNKNOWN from this data.

## Q5 verdict: what the settlement mechanic already inside the model, and
## what is structurally invisible

Already implicitly inside:

1. Revenue level, partially: baseline_tvr on the daily path is the programme
   mean of QUARTER-HOUR planned ratings (Q4), so the settlement currency's
   level, including whatever dilution the planner baked into each window,
   leaks into the engine's rating basis at programme resolution.
2. Placement selection in the retention coefficients: the shipped
   coefficients (kairos/model/measure.py break_effects) were fit on breaks
   placed by schedulers under real content constraints and whatever weak
   boundary optics Q1 found (17.6 percent straddled; short breaks straddle
   2-5 points above programme-conditional chance). The coefficients are
   averages over that placement mix, so the observed policy, including its
   settlement motive, is embedded in the treatment assignment. Given the
   measured optics are weak, this selection channel is correspondingly weak.
3. The measurement itself is CLEAN of the artifact: shed is minute-level
   before/after, never window-averaged (code fact, confirmed reading
   measure.py), so retention numbers are true audience behavior, and adding
   QH-aware revenue later will not double-count anything on the cost side.

Structurally invisible today:

1. The boundary-placement revenue lever itself: nothing maps a break's
   minutes to settlement windows; moving a 4-6 minute break onto a boundary
   is worth up to about 1 percent of billed rating (2.2 percent at 6-9m, 6.8
   percent at 9m+) and the optimizer cannot see it (code fact: no
   quarter-hour logic anywhere in kairos/).
2. Within-programme window variation: one scalar baseline_tvr per programme
   bills a 20:40 break and a 21:10 break of the same show identically, while
   the plan itself priced them 6.4 vs 4.4 (Q4 change points).
3. Shared-window coupling: with 60.3 percent of breaks sharing their quarter
   hour with another break (Q2), per-break billing is coupled across breaks;
   the engine has no joint-window concept.
4. The consolidate-vs-split currency tension: the convexity term (true
   audience favors consolidation) and the settlement term (billed points
   favor spreading the dip across windows) are comparable in size at 4-9
   minute lengths (Q3) and must enter one objective together, as
   docs/quarter-hour-billing.md already demands.

## Confidence and honesty

- Q1/Q2 are population facts of the Nov-2024 month (no sampling): HIGH
  confidence as descriptions, one month and 4 channels as scope. The
  conditional-null z-values assume independent placements; breaks within a
  programme are not independent, so treat z of about 2 as suggestive.
- Q3 matched effect (+0.0026) is observational and coarse-matched: LOW
  confidence in magnitude, sign agrees with the mechanic. The mechanical
  bounds are arithmetic on measured dips: HIGH confidence as ceilings.
- Q4 rests on a single plan file: the within-file evidence is unambiguous
  (10 of 10 windows single-valued, mid-break steps exactly on :45), but
  generalization to other channels/days is UNVERIFIED (no other plan files
  exist in data/daily_input).
- Unreachable in this data: the contractual ratings source, whether overnight
  consolidated figures replace live ones at settlement, and whether planned
  QH values embed planned break dips.
