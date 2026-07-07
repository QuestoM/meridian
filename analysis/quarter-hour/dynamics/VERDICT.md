# Within-break leave/return dynamics and quarter-hour placement: verdict memo

Date: 2026-07-07. Lane: analysis/quarter-hour/dynamics. Every number below
comes from commands run in this task on the real Nov-2024 minute-level TVR
(4 channels, 30 days, 172,800 channel-minutes) and the aired-spots log, via
the shipped loaders and break detection (kairos.data.loaders,
kairos.model.prepare.keyed_breaks).

Reproduce:

    PYTHONPATH=/Users/home/Code/questo/meridian /Users/home/.venvs/meridian/bin/python analysis/quarter-hour/dynamics/prepare_trajectories.py
    PYTHONPATH=/Users/home/Code/questo/meridian /Users/home/.venvs/meridian/bin/python analysis/quarter-hour/dynamics/curves_and_placement.py
    PYTHONPATH=/Users/home/Code/questo/meridian:analysis/quarter-hour/dynamics /Users/home/.venvs/meridian/bin/python analysis/quarter-hour/dynamics/sensitivity_check.py

Outputs: trajectories.csv (64,181 break-minute rows), breaks_meta.csv,
leave_curve.csv, return_curve.csv, placement.csv, results.json,
sensitivity.json, qh_sharing.json, prep_summary.json, prep_run.log,
fit_run.log, sensitivity_run.log.

## Sample

5,775 keyed breaks in the month; 3,748 measured (2,027 dropped because fewer
than 2 clean pre-break minutes survived the strict all-ad-airtime clip, so
the sample leans toward breaks that are not closely preceded by other ad
runs; 0 dropped for missing channel or bad normalization). 2,506 clusters
(programme instances, chan-day fallback). Minute offsets are floor-minute
offsets: offset 0 and the last break minute are PARTIAL (on average 47.5 and
46.3 percent break airtime respectively), so "minute 1" is the first full
break minute. Length-bin counts (minutes): <1.5: 1,140; 1.5-2.5: 941;
2.5-3.5: 340; 3.5-5: 370; 5-7: 522; 7-13: 435.

## Method, and a measured trap in the detrend baseline

Each break minute's TVR is divided by the channel's typical audience at that
broadcast minute (measure.py's detrend idea) and then by the break's own
pre-window level (minutes -3..-1), so 1.0 = this break's own pre-break level
net of time-of-day trend. Return minutes are included per minute only while
they lie strictly before the next ad-air run (min_spots=1 boundary, stricter
than the shipped clip).

Trap found and measured before trusting any curve: at habitual break clock
minutes the two obvious baselines disagree sharply, in opposite directions.
The shipped global typical curve encodes the recurring break dips themselves
(it averages over in-break minutes), while the content-only curve at those
same clock minutes is built only from the atypical, lower-audience days when
content happened to air there: the measured ratio global/content-only
baseline averages 1.032 over leave minutes and peaks at 1.069 at leave
offset 1, versus 0.995 on pre-break minutes. Both carry sharp artifacts
exactly where the leave curve lives. The primary detrend (rel_s) is
therefore a circular 15-minute rolling MEDIAN of the typical curve: it keeps
the time-of-day trend and rides over the localized break-schedule artifacts.
rel_g (shipped global), rel_c (content-only) and rel_raw (no detrend) are
carried as sensitivity columns in every output. The pre-break check confirms
the normalization is on trend: minutes -6..-4 average 1.001 to 1.003.

## 1. Leave curve: no cliff in minute 1; most loss builds over 2-4 minutes

Pooled (rel_s, cluster-bootstrap 95 percent CIs, B=800):

| offset | mean  | n     | clusters | CI              |
|--------|-------|-------|----------|-----------------|
| 0      | 0.989 | 3,741 | 2,501    | [0.984, 0.994]  |
| 1      | 0.952 | 3,478 | 2,396    | [0.945, 0.959]  |
| 2      | 0.931 | 2,619 | 1,886    | [0.923, 0.940]  |
| 3      | 0.919 | 1,713 | 1,267    | [0.909, 0.928]  |
| 4      | 0.903 | 1,339 | 951      | [0.893, 0.914]  |
| 5      | 0.889 | 1,095 | 760      | [0.877, 0.900]  |
| 6      | 0.881 | 814   | 572      | [0.868, 0.894]  |
| 7      | 0.866 | 543   | 406      | [0.850, 0.884]  |
| 8      | 0.841 | 339   | 273      | [0.819, 0.862]  |
| 9      | 0.809 | 192   | 165      | [0.782, 0.836]  |
| 10     | 0.775 | 90    | 79       | [0.736, 0.817]  |

TRUNCATED at minute 10: offsets 11+ have <= 35 breaks (32 clusters) and are
not trusted. The pooled curve past minute ~4 is COMPOSITION, not dynamics:
only long breaks reach those offsets (the per-bin curves are the honest
read).

Shape answer for the owner: NOT an immediate cliff. The partial first minute
holds 98.9 percent; the first full break minute holds 95.2 percent; loss
then accumulates at roughly 1 to 2 points per minute and flattens. Per-bin
(rel_s, trusted points):

- <1.5 min (n=1,140): 0.991, 0.972, 0.964. Total in-break loss ~3 percent.
- 1.5-2.5 (n=941): 0.998, 0.954, 0.948. Plateaus near -5 percent.
- 2.5-3.5 (n=340): 0.992, 0.976, 0.976, 0.985. Oddly shallow, CIs cross 1.
- 3.5-5 (n=370): 0.995, 0.955, 0.947, 0.944, 0.953, 0.949. Bottoms ~-5.5
  percent by minute 3 then stabilizes.
- 5-7 (n=522): 0.989, 0.944, 0.917, 0.907, 0.903, 0.906, 0.919, 0.921.
  Bottoms at -9.7 percent around minute 4, then viewers begin returning
  BEFORE content resumes (a real U-shape: minute 7 back to 0.921).
- 7-13 (n=435): 0.958, 0.895, 0.859, 0.853, 0.853, 0.849, 0.848, 0.852,
  0.841, 0.809, 0.775. Fast 2-minute drop to ~-14 percent, long plateau,
  further sag past minute 8 (thinning composition).

## 2. Return curve: fast snap-back, ~3 minutes pooled, 5-7 for long breaks

Pooled (rel_s): +1: 0.969 [0.959, 0.978], +2: 0.985 [0.975, 0.996],
+3: 0.998 [0.987, 1.009], +4: 1.004, then drifts UP to 1.050 by +10. The
above-1.0 drift from ~+4 onward is not a causal break effect: it is
within-show audience growth plus selection of when breaks are placed, and it
appears identically in rel_raw (1.064 at +10), so it is not a detrend
artifact. Honest reading: the recoverable deficit is ~3.1 percent in the
first content minute, ~1.5 percent in the second, gone by the third.

By length: short breaks snap back almost instantly (<1.5: +1 = 0.988;
1.5-2.5: +1 = 0.966, +2 = 0.994); long breaks return with a deeper deficit
and take longer (5-7: 0.940, 0.958, 0.971, 0.975, 0.978, 0.991, 0.995 at
+1..+7; 7-13: 0.938, 0.959, 0.975, 0.976, 0.982, 0.981, 0.990). Recovery to
within ~1 percent takes 5 to 7 minutes after a 5+ minute break, ~2-3 minutes
after a short one. Return offsets are trusted through +10 in every exact-L
group used for placement (>= 30 clusters and >= 50 rows each).

## 3. Placement: straddle, and symmetric straddling is measured near-optimal

Simulation: the measured exact-L leave curve plus trusted return curve is
slid across start offsets 0..14 of a quarter-hour window; each minute not in
break/recovery is 1.0; window averages are computed on round 15-minute
windows; the billed metric is the average of the containing window's mean
over the break's minutes (the market's CPP settlement currency per
docs/quarter-hour-billing.md). L is in TOUCHED minutes (mean true lengths:
L=2: 1.04 min, 3: 1.93, 4: 2.81, 5: 3.96, 6: 4.99, 7: 5.90; n = 861, 911,
375, 244, 281, 271). Primary profile is DEFICIT-ONLY (capped at 1.0):
above-pre-level audience (carry-in spikes, post-break ramp) is selection or
content drift the scheduler cannot move by moving the break.

Results (capped, recovered fill, billed metric; offset = break-start minute
within the window, boundary after minute 14):

| L | opt offset | symmetric | best-sym gap | best-center gap | bootstrap opt distribution (B=800) |
|---|-----------|-----------|--------------|-----------------|------------------------------------|
| 2 | 14        | 14        | 0.00000      | +0.0024         | 14: 68%, 13: 32%                   |
| 3 | 13        | 13/14     | 0.00000      | +0.0048         | 13: 100%                           |
| 4 | 13        | 13        | 0.00000      | +0.0034         | 13: 94%, 12: 6%                    |
| 5 | 12        | 12/13     | 0.00000      | +0.0086         | 12: 78%, 11: 18%                   |
| 6 | 11        | 12        | +0.00038     | +0.0154         | 11: 76%, 12: 23%                   |
| 7 | 11        | 11/12     | 0.00000      | +0.0204         | 11: 87%, 10: 13%                   |

Verdict on the owner's conjecture: the asymmetry of leave vs return is REAL
(loss builds slowly and peaks late; recovery is fast) but it moves the
optimum by AT MOST one minute from symmetric straddling, and that only for
L >= 6 (start one minute earlier than symmetric, i.e. slightly more of the
break BEFORE the boundary, worth 0.0004 relative billed points, with 77
percent bootstrap support at L=6 versus 0 percent at L=3 and 5.6 percent at
L=4). For practical purposes
symmetric boundary-straddling IS the optimum; the fast snap-back means the
recovery tail barely spills, and the late-break depth is split by centering.

The economically real lever is straddle vs contain: straddling beats the
center-contained placement by 0.24 (L=2) to 2.0 (L=7) percent of billed
rating per break minute, growing with break length. All four detrend
variants agree on the optimal offsets and on this ordering (sensitivity.json;
rel_c degenerates to a flat all-1.0 profile for L=2/4, which is itself
evidence of that baseline's day-selection contamination, not of a different
optimum). Under the as-measured (uncapped) profile the short-L optimum flips
to offset 0 because it exploits the post-break above-trend ramp; that answer
is fragile and selection-driven and should NOT be used for pricing.

## Owner complications 2 and 3, measured where measurable

- Programme content constraints (complication 2): not measurable from this
  data; the offsets above are unconstrained optima. Real placement must snap
  to permissible cut points, so the value of the lever is the gap between
  the feasible offset nearest the boundary and the contained alternative.
- Shared windows (complication 3): in the real month, 33.7 percent of the
  4,832 quarter-hour windows containing any break minute contain minutes of
  2+ breaks (27.2 percent two, 6.1 percent three, 0.4 percent four,
  distribution in qh_sharing.json), and 17.7 percent of breaks already
  straddle a boundary. The single-break optimization above is the base case
  only; in shared windows the dips compound inside one average and the
  placement problem couples across breaks. Schedulers already straddle
  sometimes, so observed placement is partially optimized (a selection force
  on all observational break measurements, as docs/quarter-hour-billing.md
  section 2 warns).

## Honest caveats

1. Composition across curve positions: deep pooled-curve offsets exist only
   for long breaks; per-minute clipping means far return offsets
   over-represent isolated breaks (which shed MORE per the convexity verify
   memo A3), so far return points may be slightly pessimistic. Per-bin and
   per-exact-L curves are the mitigation and are what placement uses.
2. Sample selection: 35 percent of keyed breaks were dropped for a
   contaminated pre-window (closely preceded breaks). The curves describe
   relatively isolated breaks; a split schedule's closely spaced breaks are
   exactly what gets dropped.
3. Partial boundary minutes: offset 0 and the last leave minute average ~47
   percent break airtime, so minute-0/last-minute values blend content and
   break audience; the true instant-of-cut behavior is sub-minute and not
   observable at this granularity.
4. The above-1.0 post-break drift (to +5 percent by minute +10, in ALL
   variants including raw) marks the limit of the pre-break-normalized
   counterfactual: past ~minute +3 the curve measures show dynamics plus
   selection, not break aftermath. Placement caps it away; nothing causal
   should be read from it.
5. Units: the simulation works in detrended relative audience and assumes
   the baseline is flat within a 15-minute window; billing runs on raw TVR.
   The billed-point gaps are relative (fractions of the pre-break level).
6. The settlement rule itself (pure round quarter hours) is owner-stated,
   not confirmed contractually (open question in docs/quarter-hour-billing.md);
   if settlement weights by spot position the placement metric changes.
7. One month, 4 channels, aggregate TVR, no viewer panel. Behavioral
   dynamics may differ by season and genre; daypart/genre splits were not
   fit here (thin cells).
8. No causal claim on WHY minute-0/1 audience holds near pre-level
   (schedulers may cut at engagement peaks); the placement conclusion only
   needs the shape of the dip relative to the break span, which is robust
   across all four detrend variants.

## Interaction with the convexity lane

Consolidation concentrates a deeper, longer dip (7-13 bin bottoms at ~-15
percent) inside settlement windows, while this memo shows straddling can
hide 0.2 to 2 percent of that dip per break minute from the billed average.
These are the two sides docs/quarter-hour-billing.md warned must be priced
in ONE objective: true-audience retention (favors consolidation, VERDICT in
analysis/convexity, verified partially-confirmed) and billed-points optics
(favors boundary-straddling placement of whatever breaks exist). Nothing
here overturns the consolidate-over-split sign; it prices the placement of
the consolidated break.
