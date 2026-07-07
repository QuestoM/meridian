# Habit-horizon test: does today's break load predict tomorrow's tune-in?

Question: for a recurring programme strip (same Title + Channel), does a heavier
in-programme ad-break load on day t predict lower audience (tune-in) on day t+1?

## Feasibility verdict: FEASIBLE at aggregate slot level (not true habit)

The data DOES support linking the same programme across consecutive days by
Title + Channel, and each programme airing carries its own average TVR (the
tune-in / audience measure). So the day-over-day slot-audience-vs-break-density
regression the task asks for is runnable and was run.

What is NOT available: any individual-viewer / panel / household identity. There
is no way to observe whether the SAME viewers tuned back in. This test therefore
measures slot-level audience continuity, not viewer-level habit. An erosion
signal here would be suggestive of habit; its absence does not disprove
viewer-level habit that is masked by audience turnover.

## Design (analysis/habit-horizon/habit_horizon_test.py)

- Unit: one linked day-pair per programme strip where the strip airs on day t and
  day t+1 (consecutive calendar days).
- Outcome: tvr_t1 = duration-weighted mean programme TVR of the strip on day t+1.
- Predictors (three separate specs): break density on day t inside that strip:
  n_breaks_t (count), break_min_t (total ad minutes), end_break_min_t (ad
  minutes in the last third of the programme span, the "near the end" load).
- Breaks: engine definition (identify_breaks: runs of >=2 spots, gap <= 15s),
  each assigned to the content programme instance whose [start,end) contains it.
  Ad-block programme rows ("קובץ פרומו/פרסומות") are excluded as content.
- Controls: programme-strip fixed effect (absorbs each slot's baseline level;
  identification is within-strip day-to-day variation) + weekday-of-t+1 fixed
  effect. Robustness spec adds lagged tune-in tvr_t (mean-reversion guard).
- Inference: cluster-robust SE by strip. Verified two ways (dummy-variable OLS
  and iterative within-transform demeaning) which reproduce identical betas.

## Result: NULL. No detectable habit-horizon effect.

Window 2024-11-01..2024-11-30, 4 channels. N = 1,444 linked day-pairs across
141 strips (strips with >=2 within-strip pairs; 171 strips / 1,474 pairs before
the >=2 filter). Outcome mean TVR = 2.88.

| predictor (day t)          | beta (TVR per unit) | 95% CI (cluster)      | t    |
|----------------------------|---------------------|-----------------------|------|
| n_breaks_t (count)         | +0.0236             | [-0.0249, +0.0722]    | 0.95 |
| break_min_t (total min)    | +0.0085             | [-0.0143, +0.0312]    | 0.73 |
| end_break_min_t (end min)  | +0.0147             | [-0.0054, +0.0349]    | 1.43 |

Adding lagged tvr_t barely moves the estimates (see habit_results.json). Every
band includes zero and the point signs are, if anything, slightly POSITIVE (the
opposite of the erosion hypothesis). The largest effect (end-load, +0.0147 TVR
per extra end-of-programme ad minute) is 0.5% of mean TVR, and even its upper
bound is ~1.2% of mean. There is no evidence that a heavier ad load today lowers
same-slot audience tomorrow at this granularity and this single-month window.

## Honest caveats

- Aggregate audience only; no viewer panel, so viewer-level habit is unobservable.
- 30 consecutive days, one month: lags are limited to 1 day here and the panel
  cannot see weekly/seasonal strip repetition across months.
- Strip fixed effects need day-to-day break-load variation within the same strip;
  strips with a single linked pair carry no within information and drop out.
- TVR is the outcome AND the ad load sits inside the same programme span, so a
  same-day mechanical link is possible; the t -> t+1 design and lagged-TVR
  robustness spec guard against that but cannot fully rule out turnover masking.

## Reproduce

    PYTHONPATH=. /Users/home/.venvs/meridian/bin/python \
      analysis/habit-horizon/habit_horizon_test.py

Outputs: analysis/habit-horizon/panel.csv (per-pair panel),
analysis/habit-horizon/habit_results.json (all six specs).
