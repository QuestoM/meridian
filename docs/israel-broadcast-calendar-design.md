# Israeli broadcast calendar: design

Owner ask (2026-07-29): chol hamoed is a real audience condition (like Hanukkah
but not in winter); mind the seasons adapted to Israel, election days, and the
connection between shabbat and the holidays on which religious viewers do not
turn on the TV. Organize the data correctly and design how it manifests in the
models, in training, and in the weekly maximization, so that a model trained on
two years of history genuinely predicts what is coming.

This layer is the DETERMINISTIC calendar: holidays, chol hamoed, Hanukkah,
seasons, elections, shabbat and yom tov. It is a separate feature family from
the operator-maintained events store (wars, specials, intensities) and its
`event_active` / `event_intensity` / `event_type` annotation seam with
`kairos/model/event_gate.py` and the `event_layer_gate` metadata key. The two
families compose side by side on the same break date and never duplicate each
other: what an operator (מפעיל) must judge lives in events; what a calendar can
state in advance lives here.

Shipped in this wave: `kairos/config/israel_calendar.csv` (the range table),
`kairos/data/israel_calendar.py` (pure feature functions plus the additive
`annotate_calendar` frame helper), and `tests/test_qa8_israel_calendar.py`.
No existing file was modified; the wiring seams are section (g).

## (a) Feature taxonomy and why each matters

The Israeli week runs Sunday to Saturday and the weekend is Friday and
Saturday only. Every feature below is date-level and deterministic.

* `religious_blackout` (is_shabbat or is_yom_tov). The owner's composite: the
  days on which religiously observant viewers do not turn on the TV at all.
  Shabbat is every Saturday; yom tov adds both Rosh Hashana days, Yom Kippur,
  the first day of Sukkot, Shmini Atzeret/Simchat Torah, the first and seventh
  days of Pesach, and Shavuot. The audience that remains is composition-shifted
  (secular-skewed), which plausibly moves both retention and the value of a
  slot. On Yom Kippur Israeli broadcast largely shuts down entirely. This is
  the single strongest calendar condition and it recurs weekly, so even short
  histories contain some contrast; two years contain about 100 shabbatot plus
  two full chagim cycles.
* `is_erev_shabbat` and `is_erev_yom_tov`. Friday and the day before a yom
  tov: early wind-down of the observant audience, prep-day daytime patterns,
  and a different prime-time shape. Date-level here; the actual candle-lighting
  clock hour is a v2 clock-level refinement.
* `is_chol_hamoed`. The intermediate days of Sukkot and Pesach: a national
  quasi-vacation (schools out, many workplaces closed or short) that is NOT in
  winter and NOT on the school-summer calendar. Daytime viewing swells the way
  it does on school holidays, families travel, and evening viewing shifts.
  Modeled jointly with school holidays as one "children and families are home"
  family, because on 30 days of history neither appears at all and on two
  years they carry similar mechanisms.
* `is_hanukkah`. Eight evenings from Kislev 25, in December (occasionally
  spilling into January). School is out for part of it, candle-lighting sits
  right at access prime time, and it is the one school-holiday-like window
  that happens inside winter viewing levels; that interaction is why it gets
  its own family instead of folding into chol hamoed plus school.
* `is_school_holiday` and school summer (החופש הגדול, July 1 to August 31).
  Children and teens at home all day restructure daytime audience and push
  family viewing earlier.
* `is_election_day`. A statutory day off with saturation news coverage,
  exit-poll prime time at 22:00, and depressed regular-programming audiences;
  campaign weeks also shift news viewing before the day itself (the day-level
  flag captures only the day; a window feature is a v2 option). The table
  carries the 2024-02-27 municipal election and the scheduled 26th Knesset
  election marked provisional.
* `season` (Israeli bands: summer June to September, autumn October and
  November, winter December to February, spring March to May). Israel's summer
  heat and daylight run well into September, the rains and the clock change
  land in late October, and winter evenings are the annual peak of indoor
  viewing. Season is a slow-moving audience-level regime, mostly a confounder
  to control so that holiday contrasts are not credited with what is really
  darkness at 17:00.
* `weekday_iso`. Already partially represented in pricing (the rate-card
  program-class by day premium); carried here so the measurement side can
  condition on it without touching pricing.

## (b) Data organization

* Deterministic bundled table versus operator events. Everything computable
  in advance ships as the checked-in `kairos/config/israel_calendar.csv`.
  Everything requiring operator judgment (wars, intensities, ad-hoc specials)
  stays in the events store. No overlap: a war never appears in the calendar
  table, a holiday never needs operator entry.
* Ranges, not single days. Chol hamoed, Hanukkah, and the school summer are
  inclusive `start_date`/`end_date` ranges; single-day entries repeat the
  date. This is the structural upgrade over the existing single-date
  `israel_holidays.csv`, which stays untouched as the Calendar tab's simple
  read-only list. Internal consistency is enforced by tests: every chol
  hamoed range sits exactly between its bounding yom tov days, and every
  Hanukkah range spans eight days.
* Verification. The table header states it in Hebrew: לאימות מול הלוח הרשמי
  לפני שימוש תפעולי. Hebrew-calendar Gregorian mappings were constructed from
  knowledge, cross-checked against `israel_holidays.csv` where the two tables
  share dates, and internally consistency-tested; rows where confidence is
  lower say "verify" in their notes (Tisha BeAv all years, Tu BiShvat 2027,
  Hanukkah 2027 start, and the provisional 2026 Knesset election date) rather
  than being dropped. Jewish days begin at sundown the prior evening; dates
  listed are the Gregorian daytime dates.

## (c) Model integration plan

The annotation seam: `annotate_calendar(frame, date_column)` adds `cal_`
prefixed columns (weekday, shabbat and erev flags, yom tov, chol hamoed,
Hanukkah, school holiday, election, season, holiday kind, religious_blackout)
beside the parallel wave's `event_` columns, on the same per-break effects
frame, before pooling. Purely additive, tolerant of a missing table, never
mutating.

One gated additive layer PER FEATURE FAMILY, not one monolith, so each family
earns activation on its own evidence:

1. `religious_blackout` (with erev flags as secondary contrasts),
2. `chol_hamoed_school` (chol hamoed and school holidays jointly),
3. `hanukkah`,
4. `season`,
5. `election`.

Each layer copies the shipped gate pattern of `kairos/model/series_gate.py`
and the parallel `event_gate.py`: five temporal folds, held-out RMSE, the
+2 percent relative-improvement bar, self-activating on every rebuild, honest
abstention when test data is too thin. Verdicts land in the coefficients
metadata under `israel_calendar_gates`, one entry per family with the same
shape as `series_gate_holdout` (statistic, fold sd, n_test, reason), so the
dashboard can disclose each decision verbatim. A family whose gate fails
contributes exactly nothing to scoring; there is no asserted fallback on the
retention side.

## (d) The two-year training story

The current 30-day window cannot decide any of this: it contains zero
holidays, zero elections, one season, and only four to five of each weekday.
Two years of history change the arithmetic qualitatively. About 100 shabbatot
and the same number of Fridays give the religious_blackout family dense,
weekly contrast. Two full chagim cycles supply two Pesach and two Sukkot chol
hamoed windows, two Hanukkahs, and sixteen yom tov days. Two school summers
give the school family roughly 120 treated days. Every season appears twice.
An election appears if the window covers one (the table knows which dates
qualify); otherwise that gate abstains honestly for lack of treated days,
which is the correct verdict, not a failure.

So the same gates that stay off on 30 days genuinely decide on two years:
activation becomes a measurement outcome, not a data artifact. Forward
scoring then applies whatever coefficients passed their gate to the plan
week's real dates in the weekly maximization: the planner already scores
concrete Gregorian dates (the pricing premium path proves the plumbing), so
`calendar_features` on each plan day supplies the same `cal_` features at
decision time that the model saw at training time. A plan week containing
chol hamoed, a Hanukkah evening, or an election day is then optimized under
its measured condition rather than under an average day.

## (e) Pricing boundary

Any calendar-based PRICE change is an operator assertion and flows only
through the existing pricing machinery: the rate-card weekday premium, the
`price_slot` layer stack, and the operator events pricing hook, all owner
gated as today. Nothing in this layer writes a price, and no measured
retention coefficient is ever rebadged as a premium. The inverse also holds:
asserted premiums never leak into the measured retention path. The honesty
line from `docs/calendar-events-design.md` is unchanged: retention effects
must be measured; pricing premiums may be operator-asserted.

## (f) Honest current-state verdict

On the shipped 30-day window (2024-11-01 to 2024-11-30) every one of the five
gates is expected OFF: no holiday, no election, and no season boundary occurs
in the window, and weekday contrast alone is thin. The UI must say exactly
that: the Calendar tab's disclosure panel lists each family with its gate
verdict and reason ("no treated days in the training window" where that is
the truth), never displaying a calendar coefficient that was not measured.
Shipping the layer OFF-by-verdict is the designed outcome today; the value is
that the day two-year history lands, a ~19 second rebuild flips only the
gates the data actually supports.

## (g) Wiring plan for the follow-up wave

One-line seams, in dependency order, all in files owned by the follow-up
wave (none touched here):

1. Measurement frame join: where the per-break effects frame gains the
   parallel wave's `event_` columns (the transform/measurement join on the
   break date), add `frame = annotate_calendar(frame, <date_column>)` so the
   `cal_` family rides the same frame into pooling.
2. Gate registry: beside the `event_gate.py` call in the rebuild path, add a
   new `kairos/model/israel_calendar_gate.py` (series_gate pattern, one
   evaluation per family from section (c)) and write its verdicts into the
   metadata under `israel_calendar_gates`.
3. Disclosure: one line per family in the Calendar tab's "what the model
   conditions on" panel, rendered from `israel_calendar_gates` verbatim, with
   the section (f) current-state wording while everything is off.
4. Forward scoring: at the `transform.py` premium and segment call sites
   where the plan day already reaches scoring as a date, call
   `calendar_features(day)` and hand any gate-passed family coefficients to
   the retention scorer; with every gate off this is exactly a no-op.
5. Optional import: the Calendar tab may render the rich table read-only
   beside `israel_holidays.csv`; no store changes needed.
