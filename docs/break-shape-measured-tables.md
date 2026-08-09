# The empirical shape of a real break: the tables

Companion to `docs/break-shape-measured.md`, which carries the argument, the
caveats and the conclusions. This file carries the per-context tables that would
have taken it past the 450-line law. **Read the main file first**; several of
these tables are meaningless without the definitions in its sections 2 and 3.

Measured 2026-08-09. Same source, same scope, same rules: `data/Spots.csv`,
operator channel from `channel_scope.operator_channel()` which read `רשת 13`,
November 2024, 18,669 operator rows. **The 1,145 timeless rows are excluded from
every table here**, because none of them can be placed in an hour or a day; that
leaves **1,910 timed breaks**, of which **760 are commercial pods**.

Two columns appear throughout and they are different quantities:

- **all airtime** is every second in the break, commercial plus promo plus
  sponsorship billboard plus public service.
- **commercial** is `Spot type = פרסומת` only, and the row is restricted to
  breaks that carry at least one.

No figure here is money. No figure here is a rival channel's.

## 1. Commercial pod length, full band frequency

The 760 commercial pods, by 60-second band of commercial airtime.

| band | pods | share |
|---|---|---|
| 1 to 60s | 55 | 7.2% |
| 61 to 120s | 152 | 20.0% |
| 121 to 180s | 148 | 19.5% |
| **181 to 240s** | **183** | **24.1%** |
| 241 to 300s | 99 | 13.0% |
| 301 to 360s | 66 | 8.7% |
| 361 to 420s | 27 | 3.6% |
| 421 to 480s | 16 | 2.1% |
| 481 to 540s | 10 | 1.3% |
| 541 to 600s | 4 | 0.5% |
| over 600s | **0** | **0%** |

76.6 percent of pods fall between 61 and 300 seconds. The distribution is
single-peaked on commercial seconds, which the pooled all-airtime figure in the
design document was not.

## 2. By daypart

Dayparts are stated here as clock bands and are **this file's own cut, not the
operator's configured dayparts**. `data/Dayparts.csv` is a minute-level rating
table and carries no named band, so nothing was read from it. Anyone comparing
these to the channel's own `רצועה` definitions must re-cut them.

**All airtime, all 1,910 timed breaks:**

| daypart | breaks | p10 | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|
| overnight 02-06 | 142 | 46 | 97 | 182 | 219 | 297 | 350 | 169 |
| morning 06-12 | 647 | 6 | 12 | 93 | 302 | 408 | 560 | 158 |
| afternoon 12-17 | 512 | 6 | 13 | 84 | 270 | 368 | 614 | 148 |
| access 17-20 | 280 | 13 | 40 | 100 | 313 | 453 | 625 | 170 |
| **prime 20-23** | 234 | 6 | 29 | **187** | 458 | 593 | **747** | 252 |
| late 23-02 | 95 | 8 | 57 | 285 | 373 | 434 | 580 | 241 |

**Commercial seconds, the 760 pods:**

| daypart | pods | p10 | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|
| overnight 02-06 | **3** | 36 | 40 | 46 | 60 | 68 | 74 | 51 |
| morning 06-12 | 245 | 99 | 144 | 193 | 236 | 281 | 369 | 191 |
| afternoon 12-17 | 152 | 79 | 106 | 156 | 226 | 275 | 426 | 167 |
| access 17-20 | 142 | 95 | 100 | 127 | 240 | 316 | 521 | 174 |
| **prime 20-23** | 157 | 47 | 126 | **261** | 366 | 463 | **582** | 263 |
| late 23-02 | 61 | 118 | 142 | 206 | 272 | 325 | 409 | 212 |

Two things worth naming. **Overnight is not a commercial daypart on this
channel**: three commercial pods in thirty nights, against 142 breaks of promo
and billboard. And prime's median commercial pod of 261 seconds is **1.7 times**
the afternoon's 156, which is the largest daypart effect in the data.

## 3. By day of week

Israeli week, ISO-keyed, ordered Sunday to Saturday. The weekend is **Friday and
Saturday**, ISO 5 and 6, per `kairos/data/israel_calendar.py`, where
`is_shabbat` is ISO 6 and `is_erev_shabbat` is ISO 5.

**All airtime, all 1,910 timed breaks:**

| day | breaks | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|
| Sunday | 247 | 18 | 108 | 341 | 456 | 673 | 187 |
| Monday | 259 | 18 | 100 | 337 | 474 | 707 | 182 |
| Tuesday | 262 | 18 | 108 | 348 | 466 | **747** | 192 |
| Wednesday | 280 | 28 | 105 | 294 | 412 | 629 | 166 |
| Thursday | 273 | 24 | 102 | 319 | 403 | 675 | 170 |
| **Friday (weekend)** | 300 | 12 | 94 | 280 | 366 | 538 | 150 |
| **Saturday (weekend)** | 289 | 28 | 135 | 274 | 390 | 728 | 173 |

**Commercial seconds, the 760 pods:**

| day | pods | p10 | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|
| Sunday | 110 | 61 | 119 | 200 | 263 | 358 | 524 | 204 |
| Monday | 110 | 71 | 120 | 212 | 294 | 357 | 582 | 216 |
| Tuesday | 110 | 82 | 134 | 213 | 306 | 354 | 549 | 223 |
| Wednesday | 110 | 88 | 127 | 205 | 262 | 326 | 505 | 203 |
| Thursday | 102 | 100 | 145 | 197 | 243 | 296 | 524 | 201 |
| **Friday (weekend)** | 111 | 84 | 118 | **160** | 203 | 255 | 451 | 170 |
| **Saturday (weekend)** | 107 | 60 | 99 | **144** | 223 | 346 | 575 | 176 |

**The weekend carries shorter commercial pods.** Friday 160 and Saturday 144
against a weekday median of 197 to 213: against a weekday median of 205, Friday
is **22 percent shorter** and Saturday **30 percent** shorter. The pod count
barely moves, 102 to 111 on every day of the
week, so this is a length effect and not a count effect, and it is the one place
in the data where length varies materially with something other than spot count.

Day of week explains only 3.0 percent of pod-length variance overall, so the
effect is real at the median and small against the spread.

## 4. By clock hour: break length

Hours 00 and 01 carry no operator break in this month and are absent. Hour 04
holds three breaks in thirty days and its percentiles are not meaningful.

| hour | breaks | median all | p90 all | max all | pods | median comm | p90 comm | max comm |
|---|---|---|---|---|---|---|---|---|
| 02 | 43 | 200 | 318 | 350 | 1 | 74 | 74 | 74 |
| 03 | 46 | 186 | 279 | 331 | **0** | - | - | - |
| 04 | 3 | 190 | 199 | 201 | 1 | 33 | 33 | 33 |
| 05 | 50 | 139 | 222 | 344 | 1 | 46 | 46 | 46 |
| 06 | 62 | 86 | 178 | 278 | 12 | 38 | 54 | 58 |
| 07 | 116 | 115 | 412 | 520 | 48 | 203 | 284 | 359 |
| 08 | 139 | 91 | 428 | 514 | 56 | 196 | 310 | 349 |
| 09 | 119 | 120 | 423 | 560 | 53 | 201 | 298 | 361 |
| 10 | 115 | 49 | 376 | 494 | 44 | 176 | 239 | 305 |
| 11 | 96 | 22 | 439 | 517 | 32 | 211 | 276 | 369 |
| 12 | 120 | 44 | 384 | 614 | 31 | 223 | 292 | 329 |
| 13 | 120 | 106 | 353 | 575 | 30 | 138 | 190 | 295 |
| 14 | 116 | 68 | 377 | 569 | 31 | 205 | 307 | 426 |
| 15 | 74 | 106 | 340 | 393 | 31 | 101 | 161 | 197 |
| 16 | 82 | 96 | 423 | 490 | 29 | 198 | 276 | 320 |
| 17 | 111 | 100 | 451 | 625 | 50 | 120 | 316 | 401 |
| 18 | 89 | 151 | 484 | 623 | 44 | 219 | 331 | 519 |
| 19 | 80 | 108 | 363 | 569 | 48 | 105 | 223 | 521 |
| 20 | 42 | 339 | 473 | 509 | 38 | 234 | 329 | 420 |
| 21 | 117 | 37 | 434 | 704 | 53 | 227 | 354 | 516 |
| **22** | 75 | **535** | **639** | **747** | 66 | **371** | **522** | **582** |
| 23 | 95 | 285 | 434 | 580 | 61 | 206 | 325 | 409 |

**Hour 22 is this channel's heaviest hour on every measure**, and it corroborates
the traffic file's 22:00 spike independently of it. See section 8 of the main
file for the boundary-straddler caveat that inflates the traffic file's own
figure for that hour.

## 5. By clock hour: the capacity envelope

Per date-hour cell that carries at least one timed break. **622 cells** of a
possible 720 (30 days by 24 hours); the 98 missing are hours with no break at
all, chiefly 00, 01 and 04.

| hour | cells | median comm min | p90 comm min | max comm min | median all min | max all min | median breaks | max breaks | median pods | max pods |
|---|---|---|---|---|---|---|---|---|---|---|
| 02 | 30 | 0.0 | 0.0 | 1.2 | 3.9 | 13.1 | 1 | 3 | 0 | 1 |
| 03 | 25 | 0.0 | 0.0 | 0.0 | 5.3 | 10.5 | 2 | 3 | 0 | 0 |
| 04 | 3 | 0.0 | 0.4 | 0.6 | 3.2 | 3.4 | 1 | 1 | 0 | 1 |
| 05 | 29 | 0.0 | 0.0 | 0.8 | 3.4 | 11.3 | 2 | 3 | 0 | 1 |
| 06 | 30 | 0.0 | 0.8 | 1.0 | 3.1 | 7.1 | 2 | 3 | 0 | 1 |
| 07 | 30 | 5.9 | 8.2 | 11.4 | 13.2 | 16.9 | 4 | 6 | 2 | 2 |
| 08 | 30 | 6.6 | 8.8 | 11.6 | 13.4 | 16.5 | 5 | **7** | 2 | 2 |
| 09 | 29 | 6.3 | 9.0 | 11.8 | 12.8 | 19.1 | 4 | 6 | 2 | 2 |
| 10 | 30 | 4.0 | 6.2 | 11.1 | 8.7 | 14.9 | 4 | 6 | 1 | 3 |
| 11 | 30 | 3.9 | 5.7 | 6.2 | 8.0 | 13.0 | 3 | 5 | 1 | 2 |
| 12 | 30 | 3.7 | 4.9 | 13.2 | 8.0 | 16.5 | 4 | **7** | 1 | 3 |
| 13 | 29 | 2.3 | 3.2 | 4.9 | 10.3 | 17.8 | 4 | 5 | 1 | 2 |
| 14 | 30 | 3.5 | 5.1 | 7.1 | 7.8 | 15.2 | 4 | 6 | 1 | 2 |
| 15 | 30 | 1.7 | 2.7 | 4.7 | 6.1 | 11.5 | 2 | 4 | 1 | 2 |
| 16 | 30 | 3.2 | 4.6 | 5.3 | 7.9 | 12.1 | 2 | 5 | 1 | 1 |
| 17 | 30 | 4.7 | 7.4 | 8.3 | 10.3 | 15.0 | 3 | 6 | 2 | 2 |
| 18 | 30 | 4.8 | 9.4 | 14.3 | 9.5 | 19.8 | 3 | 5 | 1 | 2 |
| 19 | 30 | 2.2 | 6.8 | 11.1 | 5.3 | 12.8 | 2 | 5 | 2 | 3 |
| 20 | 27 | 5.2 | 6.1 | 10.1 | 7.2 | 14.9 | 1 | 3 | 1 | 3 |
| 21 | 30 | 5.6 | 9.2 | 14.6 | 7.9 | 19.8 | 4 | **7** | 2 | 3 |
| **22** | 30 | **13.6** | **16.4** | **19.3** | **19.6** | **24.3** | 2 | 5 | 2 | **4** |
| 23 | 30 | 7.3 | 9.5 | 12.8 | 12.8 | 16.7 | 3 | 5 | 2 | 3 |

### Against the model's ceilings

Over all 622 date-hour cells. **Both rows are true and they answer different
questions**; which one a guardrail should be compared against depends on whether
that guardrail counts promo and sponsorship airtime, which is an owner question.

| measure | median | p90 | max | above 8.0 | above 10.0 | above 12.0 |
|---|---|---|---|---|---|---|
| **all airtime** minutes/hour | 8.0 | 15.2 | 24.3 | 310 (49.8%) | 224 (36.0%) | 167 (26.8%) |
| **commercial** minutes/hour | 3.6 | 8.3 | 19.3 | **72 (11.6%)** | 38 (6.1%) | 23 (3.7%) |

| measure | median | max | above 4 |
|---|---|---|---|
| breaks per hour | 3 | **7** | 89 (14.3%) |
| commercial pods per hour | 1 | **4** | **0 (0%)** |

Three readings, each of which the main file's section 9 uses:

- The design document's "52 percent of hours exceed the 8-minute ceiling" is a
  statement about **all** airtime. On commercial airtime alone it is **11.6
  percent**. Neither figure is wrong; they count different things.
- **No hour in this month carries more than four commercial pods**, and the
  model's cap of four breaks an hour is therefore never exceeded by commercial
  pods at all. It is exceeded by total breaks in 14.3 percent of hours, because
  promo and billboard breaks are breaks too.
- The real maximum of breaks in a clocked hour is **7**, not the 40 the design
  document reports. See section 7 of the main file for why.

## 6. The programme join

`data/Programmes.csv`, operator channel, 989 programme rows, joined to breaks on
date and cleaned title. **1,663 of 1,910 timed breaks matched, 87.1 percent.**

Programme types present on the operator channel: `Other` 839 rows, `News` 142,
`Sports` 8.

**Commercial pod length by programme type:**

| type | pods | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|
| News | 78 | 133 | **241** | 314 | 347 | 420 | 226 |
| Other | 544 | 131 | **194** | 250 | 327 | 582 | 201 |
| Sports | - | - | - | - | - | - | - |

`Sports` has 8 programme rows in the month and no usable pod sample. It is
**unknown, not zero**, and no figure should be shown for it.

**Breaks and pods per programme airing**, over 541 airings carrying at least one
break:

| measure | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|
| breaks per airing | 1 | 2 | 3 | 7 | 17 | 3.1 |
| commercial pods per airing | 0 | 1 | 1 | 3 | 6 | 1.1 |
| programme minutes | 36 | 56 | 74 | 120 | 185 | 65 |

The correlation table, and the banding that shows the same thing without a
coefficient, are in section 4 of the main file. The short version: programme
length predicts break **count** at r = +0.843 and this pod's **length** at
r = +0.002.

## 7. Reproducing any of this

Every table above came from `data/Spots.csv` and `data/Programmes.csv` read
read-only through `~/.venvs/meridian/bin/python`, with the operator channel taken
from `kairos_api.channel_scope.operator_channel()` and never from the file's own
`is_target_channel` column, which marks a different channel.

The rules that reproduce the row counts:

1. Filter `Channel` to the operator's. 18,669 rows.
2. A break is a distinct `break_id`. It is globally unique in this file, so no
   date or channel key is needed. 3,055 breaks.
3. Parse `Start_dt` with `dayfirst=True`. The 1,145 rows that fail to parse are
   the timeless orphans of the main file's section 2; drop them and count them.
   1,910 breaks remain.
4. A commercial pod is a break holding at least one row with
   `Spot type = פרסומת`. 760 pods.
5. Commercial seconds are the sum of `Duration` over those rows only. All
   airtime is the sum over every row in the break.
6. The Israeli week is ISO-keyed with the weekend at ISO 5 and 6, Friday and
   Saturday.
