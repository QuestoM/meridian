# The empirical shape of a real break

Measured 2026-08-09 against `data/Spots.csv` and the single traffic file in
`data/daily_input/`, in answer to Step A of
`docs/break-shape-and-capacity-design.md`. Nothing here recalls a number; every
figure was computed on this tree on this date, and the row count behind it is
stated beside it.

The companion `docs/break-shape-measured-tables.md` carries the full per-hour,
per-daypart and per-day tables. This file carries the argument.

**This changes no engine behaviour, no constant, no default and no setting.** It
is a measurement and a document, which is the whole of what Step A was for.

## The three things that stop this file being misread

**1. Every figure is the operator's own channel.** The channel came from
`kairos_api.channel_scope.operator_channel()`, which read `רשת 13` from settings.
The file holds four channels because the retention model is measured against the
competitive lineup; the other three appear here only as a row count, never as a
name beside a number, and no figure below is a rival's.

There is a trap in the data and it is worth naming. `Spots.csv` carries an
`is_target_channel` column, and it is **True for a channel that is not the
operator's** (23,707 rows, against the operator's 18,669). Anything that trusts
that column reports a competitor's break shape as the channel's own. It was not
used here and it should not be used anywhere.

**2. This dataset is time and structure. It is not money.** Its `revenue_ils`
column is a synthetic price from a constant base rate, verified on 99.67 percent
of 50,386 rows by an earlier audit. No shekel figure is computed or reported
below, and none should be computed from it later.

**3. The word "break" has to be qualified before any number attaches to it.**
That is the substance of this file. The design document's pooled median of 39
seconds and the traffic file's 322 are both correct, and they are answers to
different questions. Section 3 is the reconciliation.

## 1. The reproduction: exact, to every digit

Asked to reproduce before extending. Reproduced, and every figure matches.

`data/Spots.csv`, 50,386 rows, filtered to the operator channel: **18,669 rows**,
30 days of November 2024, grouped by the file's own `break_id`: **3,055 breaks.**

| | design document | measured 2026-08-09 |
|---|---|---|
| breaks | 3,055 | **3,055** |
| minimum | 6 | **6** |
| 10th percentile | 6 | **6** |
| 25th percentile | 12 | **12** |
| median | 39 | **39** |
| 75th percentile | 186 | **186** |
| 90th percentile | 373 | **373** |
| maximum | 747 | **747** |
| mean | 117 | **117.2** |
| exactly 120 seconds | 10 | **10** |
| spots per break, median / p90 / max | 2 / 19 / 47 | **2 / 19 / 47** |

One structural check the design document did not state, run because a restarting
identifier would have invalidated the grouping: `break_id` is **globally unique
across the whole file**. 9,492 distinct values against 9,492 distinct
(channel, date, break_id) triples. Grouping by `break_id` alone is safe and
carries no confound.

**The break-length reproduction is exact. The hourly block did not reproduce**,
and section 7 says so with the four methods tried.

## 2. The correction that moves every pooled number

**1,145 of the 3,055 "breaks" are not breaks.** They are single-row artefacts
with no clock time at all.

1,145 operator rows, 6.1 percent, carry a `Start time` of `01/01/1900`, an Excel
epoch-zero serial. Their `Start_dt` and `End_dt` are null and the file's own
`hour_of_day` is `-1`. They cannot be placed in an hour, a daypart or a day.

What makes them structural rather than a rounding nuisance:

- **Each one is its own break.** 1,145 timeless rows produce 1,145 distinct
  `break_id` values, every one holding exactly one row. Not one real break is
  partly timeless; the split is clean.
- Every one carries `position_in_break = 1`, which is what a one-spot break has.
- They are **37.5 percent of the 3,055**, and they are short: median 15 seconds,
  maximum 74, none above 120.

So the pooled distribution is a mixture of 1,910 real breaks and 1,145 orphan
rows, and the orphans sit entirely below the median. **They are what drags the
median to 39.** Dropping them, and changing nothing else:

| | all 3,055 | the 1,145 timeless | the 1,910 timed |
|---|---|---|---|
| median | 39 | 15 | **108** |
| 75th percentile | 186 | 36 | **310** |
| 90th percentile | 373 | 49 | **426** |
| maximum | 747 | 74 | **747** |
| mean | 117.2 | 23.5 | **173.4** |
| longer than 120s | 29.4% | 0% | **47.1%** |
| spots per break, median | 2 | 1 | **6** |

They carry 26,856 of the month's 358,096 aired seconds, 7.5 percent, so they are
a real if small quantity of airtime. What they are not is 37.5 percent of the
channel's breaks, and no distribution should count them as such.

**Whether these rows are a broadcast fact or an export defect is not something
this data can settle, and I have not guessed.** They are evenly spread across all
30 days, 38 to 51 per day, and across all four spot types. That regularity looks
like an export path rather than an on-air one, but the honest state is unknown
and it is a question for the owner and for the traffic system.

## 3. The commercial versus promotional split, which was the first thing owed

### What the columns actually contain

`Spot type` carries four values. `Promotion` carries two and is **derived**: it
is `פרומו` for every `פרומו` row and `ספוט` for every other row, with no
exceptions in 18,669 rows. It holds no information `Spot type` does not.

On the operator channel:

| `Spot type` | rows | seconds | share of aired seconds | median length | modal length |
|---|---|---|---|---|---|
| `פרסומת` commercial | 7,368 | 158,128 | 44.2% | 17s | 15s (1,667 spots) |
| `פרומו` promo | 3,832 | 141,095 | 39.4% | 38s | 48s |
| `חסות` sponsorship | 7,259 | 49,216 | 13.7% | **6s** | **6s (5,261 spots)** |
| `תשדיר שרות` public service | 210 | 9,657 | 2.7% | 49s | 65s |

The sponsorship row is the answer to the design document's open question, and it
is unambiguous. `חסות` has a **median and mode of exactly 6 seconds**, which is
precisely the form the trade research attests: a 6-second on-screen mention with
voiceover, `תשדיר חסות באורך 6 שניות בתוספת קריינות`, sold at a pre-agreed fixed
rate rather than by position. The design document's guess that "a 6-second break
at the 10th percentile looks like a billboard" was right. It is a billboard, and
there are 1,071 breaks made of nothing else.

### Only a quarter of the breaks are commercial pods

Composition of the 3,055, by which spot types share a break:

| composition | breaks |
|---|---|
| sponsorship only | 1,071 |
| sponsorship + promo + commercial | 579 |
| promo only | 569 |
| commercial only | 394 |
| sponsorship + promo | 211 |
| everything else | 231 |

**1,973 of the 3,055 carry no commercial spot at all.** Pure promo, pure
sponsorship billboard, or the two together. Whatever a regulatory minute cap
counts, and that is a question for the owner, a break holding no commercial is
not the thing the optimizer is selling.

### The distribution under each definition of a break

Timeless rows dropped throughout, so this table and the one in section 2 are on
the same 1,910.

| definition | n | p10 | p25 | **median** | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|
| all timed breaks, all airtime | 1,910 | 6 | 18 | **108** | 310 | 426 | 747 | 173 |
| a commercial pod, commercial seconds | 760 | 81 | 120 | **190** | 256 | 337 | 582 | 199 |
| a commercial pod, all its airtime | 760 | 119 | 248 | **340** | 427 | 501 | 747 | 333 |
| paid seconds (commercial + sponsorship) | 1,658 | 6 | 12 | **46** | 213 | 313 | 676 | 119 |

**A commercial pod is 760 breaks in a month, 25.3 a day**, and it holds 190
seconds of commercial airtime at the median inside 340 seconds of total break.

### What this does to the 120-second constant, honestly

The design document's reading was that 120 is "the mean of a wildly skewed
distribution, the one summary statistic that describes none of it". Under the
commercial split that reading has to be qualified, and the qualification runs
**in the constant's favour**, so it is stated first.

On commercial pods, 120 seconds is close to the **25th percentile** of commercial
airtime (exactly 120.0) and the constant sits at the low end of a real
distribution rather than nowhere in it. It is not the absurd artefact the pooled
figure made it look.

What does not change, and it is still decisive:

- The median commercial pod is **190 seconds, 58 percent longer** than the
  constant, and **72.8 percent of pods carry more commercial airtime than 120
  seconds.**
- Only **18 of 760 pods, 2.4 percent**, hold exactly 120 seconds of commercial.
- Counting the whole break rather than its commercial part, the median is **340
  seconds and 88.9 percent exceed 120.**
- The spread is the point: p10 of 81 against p90 of 337, a factor of four, and a
  maximum of 582.

So the finding survives the split and its shape changes. One number cannot
represent this, and the number chosen is low rather than arbitrary.

## 4. Does a break's length follow its programme? A clean answer, and it is no

Joined to `data/Programmes.csv` on operator channel, date and title.
**1,663 of 1,910 timed breaks matched, 87.1 percent.** The 12.9 percent that did
not are titles the spot file concatenated across a programme boundary, of the
form `שיחת היום עם לוסי אהריש] * [חדשות שש`, which is two programmes in one cell.
They are excluded and counted, not repaired.

Over 356 programme airings that carry at least one commercial pod:

| relationship | Pearson r | Spearman |
|---|---|---|
| programme length vs **break count** | **+0.843** | +0.757 |
| programme length vs **commercial pod count** | **+0.821** | +0.763 |
| programme length vs total commercial seconds | +0.707 | +0.588 |
| programme length vs **this pod's commercial seconds** | **+0.002** | +0.035 |

**A longer programme carries more breaks. It does not carry longer ones.** The
last row is the one that matters and it is as close to zero as a correlation
gets, on 622 pods.

The banding says the same thing without a correlation coefficient. Median
commercial seconds in a pod, by the length of the programme carrying it:

| programme | airings | median breaks | median pods | median commercial seconds **per pod** |
|---|---|---|---|---|
| 15 to 30 min | 16 | 1 | 1 | 151 |
| 30 to 45 min | 60 | 1 | 1 | 192 |
| 45 to 60 min | 99 | 2 | 1 | 178 |
| 60 to 90 min | 67 | 2 | 1 | 229 |
| over 90 min | 114 | **6** | **3** | 195 |

The break count goes 1, 1, 2, 2, 6. The pod length goes 151, 192, 178, 229, 195,
which is noise around 190.

**This contradicts step C of the design document as written.** Step C proposes
that "the length a segment gets is the length that programme, daypart and channel
actually carries". Programme duration does not carry a length. It carries a
count, and the count is already a decision the optimizer makes.

By programme type, on commercial pods: `News` median 241 seconds of commercial
(n=78), `Other` median 194 (n=544). News pods are longer, by about 25 percent.
`Sports` has 8 programme rows in the month and no usable pod sample, so it is
unknown rather than zero.

## 5. Count and length together, and they do not trade off

The brief's hypothesis was that an hour with more breaks has shorter ones, and
that a design varying one without the other is wrong. **Measured, the trade-off
is not there.** Over 510 date-hour cells carrying at least one commercial pod:

| commercial pods in the hour | cells | median seconds **per pod** | median commercial seconds in the hour |
|---|---|---|---|
| 1 | 278 | 166 | 166 |
| 2 | 217 | 192 | 384 |
| 3 | 12 | 222 | 667 |
| 4 | 3 | 246 | 983 |

- pods in the hour vs mean seconds per pod: **r = +0.199**, Spearman +0.189
- pods in the hour vs total commercial seconds: **r = +0.725**, Spearman +0.746

The relationship is **weakly positive, not negative.** A busier hour carries more
pods *and* slightly longer ones, so the two compound rather than substitute, and
capacity scales close to multiplicatively.

**The honest limit on this finding: the range is thin.** 495 of 510 cells hold
one or two pods, the three-pod cell count is 12 and the four-pod count is 3. The
positive slope rests almost entirely on one-versus-two. It is enough to say the
strong negative trade-off is absent; it is **not** enough to characterise the
relationship at four or more pods an hour, and nothing should extrapolate there.

### What actually explains a pod's length

Descriptive variance share on the 760 pods, single-factor, no model fitted:

| factor | groups | variance explained |
|---|---|---|
| **spot count in the pod** | 26 | **85.0%** |
| programme title | 109 | 46.1% |
| clock hour | 21 | 26.4% |
| calendar date | 30 | 8.2% |
| day of week | 7 | 3.0% |

**Pod length is spot count times a near-constant spot length.** Mean seconds per
spot inside a pod, by pod size: 21.2 (4 to 6 spots), 20.9 (7 to 9), 21.0 (10 to
12), 20.2 (13 to 15), 21.3 (16 or more). Flat, to within a second, across the
whole range. The one exception is the 1-to-3-spot band at 38.0 seconds a spot,
which is a different product and is discussed in section 6.

This is the most consequential structural fact in the file and section 9 returns
to it: **break length is not an independent quantity. It is how many spots were
sold into the break.**

## 6. The envelope, as a description and never as a permission

What occurred, with its frequency. The bounds come from the owner and the trade
document, per section 5 of the design document. They do not come from here.

**Commercial airtime in a commercial pod, 760 pods:**

| p0 | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 25 | 48 | 81 | 120 | **190** | 256 | 337 | 396 | 520 | **582** |

The heaviest 60-second band is **181 to 240 seconds at 24.1 percent** of pods,
and **no pod anywhere in the month exceeds 600 seconds** of commercial airtime.
The full band-frequency table is in the companion file.

**Commercial spots in a commercial pod:** p10 = 2, p25 = 6, median = 9, p75 = 12,
p90 = 16, p95 = 18, maximum = 29.

Two comparisons to the trade, both of which the trade document and the vocabulary
research outrank and neither of which overturns anything:

- The published Israeli average of **twelve to fifteen advertisements a break**
  sits at this channel's **p75 to p90**. 141 pods, 18.6 percent, fall inside the
  band exactly. The channel's median pod of 9 is smaller than the published
  national average, which is a fact about this operator and not a correction to
  the trade.
- The trade defines a **gold break as one to three spots**. 113 pods, 14.9
  percent, are that size, and their mean spot length of 38.0 seconds against 21
  everywhere else marks them out as a genuinely different product rather than a
  short version of the same one. `Spots.csv` carries **no `break_type` column**,
  so gold breaks cannot be identified as such in this dataset. The traffic file
  does carry one, with values `Regular` and `EB`.

**Spot length is always a whole number of seconds** in all 7,368 commercial rows,
minimum 5, median 17, maximum 171, with 15 seconds the mode at 1,556 spots. That
corroborates the trade document's "length is in whole seconds, there are no
milliseconds".

Per-hour, per-daypart and per-day-of-week envelopes are in
`docs/break-shape-measured-tables.md`.

## 7. The hourly block did not reproduce, and here is exactly what I get

The design document reports 682 hours, of which 352 above 8.0 minutes, 241 above
10, 157 above 12, breaks per hour median 3 and **maximum 40**, 148 hours above 4.

I could not reproduce those figures by any grouping I tried. Four were tried and
all four are stated so nobody repeats them:

| method | cells | median | p90 | max | >8.0 | >10 | >12 | breaks max | >4 breaks |
|---|---|---|---|---|---|---|---|---|---|
| design document | 682 | 8.1 | 14.7 | 24.3 | 352 | 241 | 157 | **40** | 148 |
| per-spot `hour_of_day`, includes the `-1` bucket | 653 | 8.2 | 15.2 | 24.3 | 354 | 249 | 171 | 51 | 121 |
| per-spot `hour_of_day`, `-1` dropped | 623 | 8.1 | 14.9 | 24.3 | 324 | 222 | 151 | 7 | 91 |
| break assigned to its start hour, `-1` kept | 652 | 8.1 | 15.5 | 24.3 | 340 | 251 | 187 | 51 | 119 |
| break assigned to its start hour, `-1` dropped | 622 | 8.0 | 15.2 | 24.3 | 310 | 224 | 167 | 89 | 89 |

The maximum of 24.3 minutes reproduces under every method. Nothing else does
exactly, and I am not going to invent the method that would.

**What the discrepancy does establish is where the "40 breaks in an hour" came
from.** The real maximum, over hours that have a clock, is **7 breaks**. The
figure of 40 is only reachable through the `-1` bucket, which is one cell per day
holding that day's 38 to 51 timeless orphans. It is not an hour of broadcast and
there is no hour in this month carrying 40 breaks. Anyone sizing a breaks-per-hour
guardrail against 40 would be sizing it against an export defect.

## 8. The two sources compared, and they agree once compared like with like

The design document flagged the traffic file's median of 322 seconds against the
month's 39 as a gap to chase. Chased. **It is almost entirely a definition gap,
not a prime-versus-day gap.**

Median seconds of total break airtime, each step changing one thing:

| | median | n |
|---|---|---|
| all 3,055 grouped `break_id` values | 39 | 3,055 |
| drop the 1,145 timeless orphans | 108 | 1,910 |
| keep only breaks carrying a commercial | **340** | 760 |
| restrict those to prime, 20:00 to 22:59 | **389** | 157 |
| the traffic file, its 8 breaks carrying a commercial | **500** | 8 |
| the traffic file, all 10 breaks, as the design document counted | 322 | 10 |

The month's prime commercial pods run p25 = 171, median 389, p75 = 519, max 747.
The traffic file's eight run p25 = 168, median 500, p75 = 607, max 803. **The
quartiles line up and the file's median sits between the month's p50 and p75**,
which is what one evening of eight breaks should look like inside a month of 157.
Spots per break: 20 median in the month's prime, 18 in the traffic file.

The traffic file's own 322 is itself a mixed number, computed over ten breaks two
of which are 18-second breaks holding three `חסות` billboards and no commercial
at all, exactly the composition problem section 3 describes.

**Where the two sources genuinely disagree, and it is not resolved here.** The
traffic file is a single evening from **April 2025**; the spot file is **November
2024**. They are five months and one channel-schedule apart, so a difference
between them cannot be attributed to sampling alone, and I have not averaged
them. Both are shown above and neither was picked.

**The 22:00 hour, which the design document flagged as deserving its own look.**
It was right to flag it and the suspicion was correct. The traffic file's 35.6
minutes in that hour includes a break starting at **22:59:40**, twenty seconds
before the boundary, running 432 seconds into the 23:00 hour. Excluding that
straddler leaves **28.4 minutes**, still far above every ceiling. And the month
corroborates the hour independently: **22:00 is this channel's heaviest hour**,
median 13.6 commercial minutes, maximum 19.3, against a whole-day median of 3.6.

## 9. What stages B and C now have to work with

**The one thing stage B most needs to know.** Break length is not an independent
quantity that a model can price on its own. **Spot count explains 85 percent of a
pod's length, and the mean spot inside a pod is 21 seconds regardless of how big
the pod is.** A break is long because more spots were sold into it. So the
question stage B has to answer is not quite "does an extra thirty seconds cost
audience"; it is "does an extra *spot* cost audience", and the two are the same
question only if the cost is per second rather than per interruption. The design
document's trap analysis holds and sharpens: retention is currently priced per
break, revenue per second, and the ratio between them is a spot count nobody has
modelled.

**The identification range for a length term is real and it is wide.** Commercial
pods span 20 to 582 seconds on the operator's channel over 30 days, 760 of them,
with **27.2 percent at or below 120 seconds and 7.5 percent above 360**. That is enough
spread to identify a length term if one exists. It is a within-channel,
within-month range and it needs no cross-channel comparison to work.

**What stage C must not do.** It must not set a per-segment length from programme
duration. Measured r = +0.002 on 622 pods. Programme duration predicts break
**count**, r = +0.843, which the optimizer already decides. If stage C wants a
per-segment fact from this data, the honest ones are: programme **type**
(News 241s, Other 194s), clock **hour** (26.4 percent of variance,
`docs/break-shape-measured-tables.md` has the table), and above all the number of
spots actually sold into that break.

**What the capacity ledger can be built against today.** Commercial pods are
25.3 a day and 158,128 commercial seconds a month, 87.8 commercial minutes a day
against 198.9 minutes of total airtime. Commercial minutes per hour: median 3.6,
p90 8.3, maximum 19.3, and 11.6 percent of hours exceed 8.0. That last figure is
the honest one to compare a guardrail against, and it is **not** the design
document's 52 percent, which is all airtime including promos and billboards.
Both are true and they answer different questions.

## 10. What I could not establish

- **Whether the 1,145 timeless rows are real airtime or an export defect.** They
  are consistent enough to look systematic. The data cannot say which, and this
  is the first question for the owner.
- **The design document's hourly block.** Four methods, none reproduce it. My own
  figures are in section 7 and in the tables file, with the method named.
- **Gold breaks in the month.** `Spots.csv` has no `break_type` column. The
  1-to-3-spot pods are a suggestive 14.9 percent with a distinct 38-second mean
  spot length, but that is an inference from size and not a reading of a field.
- **Sports programming.** 8 programme rows in the month and no usable pod sample.
  Unknown, not zero.
- **Whether a regulatory cap counts sponsorship or promo airtime.** This decides
  whether the channel's busiest hour is 19.3 minutes or 24.3. It is a question
  for the owner and the regulator, not for this file.
- **Any figure about money.** Deliberately, per section 1.
- **Anything about a break's *position* in its programme.** Neither file carries
  a mid-programme versus end-adjoining marker, so the `אתנחתה` distinction the
  regulation draws cannot be measured here.
