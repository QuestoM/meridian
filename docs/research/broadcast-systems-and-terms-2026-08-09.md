# Broadcast systems and the words the trade uses

Primary research, 2026-08-09, working tree at `a7fb5e09`. Written under
`docs/audits/research-scope-ruling.md`: **Israel is the only market that
matters.** Foreign documentation is read to find the NAME of something Israel
already does, never to import a feature.

The document has two parts and they are not equal.

- **Part 1, relevant to Israel.** Every finding carries Israeli evidence: the
  trade document, the shipped traffic file, the regulator, or an Israeli source.
  Foreign evidence alone never qualifies a finding.
- **Part 2, foreign, not applied.** Documentation only. Nothing in the product
  changes because of it. It is written down so that the next person does not
  research it again.

Ranking rule for Part 1: what it blocks, not how interesting it is.

**Companion file.** `docs/research/hebrew-trade-vocabulary.md` covers the Hebrew
vocabulary itself — term by term, with the Israeli source and the queries that
found nothing — and is the place to look for a word rather than a mechanism.
This file is about systems, fields and money, and it cites that one rather than
repeating it. The two were written from different sources and agree where they
overlap: both could attest only first, second, third and last as preferred
positions from the open web, and both record that the trade document's first
through fifth plus L outranks that.

## How to read the labels

- **MEASURED** — read in a named file in this repository, or at a URL that is
  given. Repository measurements were computed directly against the shipped
  files and the counts are reproducible. Web measurements were gathered by a
  parallel research fan-out and each carries its URL and, where it matters, the
  quoted phrase.
- **INFERRED** — reasoned from how the trade works. It is not evidence, and it
  is never the basis for a finding on its own.
- **NOT CONFIRMED** — looked for and not established. What was searched is
  stated. This is a first-class result, because the owner's standing point is
  that this trade is small and largely undocumented online.

Two things are worth saying about how this was produced. Every number drawn from
`data/`, `kairos/`, `kairos_api/` and `tv-break-dashboard/` in Part 1 was
computed here, against the working tree, and can be re-run. The Hebrew statutory
quotations and the vendor and trade-press citations came back from the research
fan-out with their URLs attached; they were read, not summarised from a search
snippet, and where a source turned out to be a search summary rather than a
document that is said in the line itself.

Two research streams had not returned when this was written: the airtime-trading
mechanics of other markets, and a systematic glossary sweep. Nothing in this
document depends on them.

`docs/media-domain-from-the-trade.md` outranks every foreign manual, every
vendor page and every regulator document on every question where they disagree.
Where the web contradicts it, the contradiction is recorded and left standing.

---

# The best source we have is already in the repository

Before any of the web research: the single richest piece of "real documentation
of a broadcaster's system" in this project is **not on the web**. It is
`data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv`, one prime-time
evening of Reshet 13, 175 rows, in the incumbent system's own Hebrew column
names. `kairos/data/loaders.py:81-99` maps them.

It is the closest thing to a schema for the incumbent traffic system that exists
anywhere I could reach, and most of Part 1 is read straight off it.

The eighteen fields, as the incumbent names them (MEASURED,
`kairos/data/loaders.py:81-99` and the file's own header):

| Hebrew | Our name | What the shipped example holds |
| --- | --- | --- |
| `תאריך` | date | `4/27/2025` |
| `שעה` | spot_time | the spot's own air second, e.g. `20:44:30` |
| `שעת התחלת ברייק` | break_start | the break's start second |
| `משרד / MB` | agency | `OMD`, `יוניברסל`, `לפמ`, `פיתוח עסקי`, … |
| `סוג תשדיר` | spot_type | `פרסומת` 124, `חסות` 51 |
| `מפרסם` | advertiser | 41 distinct |
| `קמפיין` | campaign | 51 distinct |
| `שם גרסה` | creative | 87 distinct version names |
| `House Number` | house_number | 76 distinct, e.g. `CGB007546` |
| `אורך תשדיר` | duration_sec | 27 distinct lengths |
| `תוכנית מוזמנת` | program | **the ORDERED programme**, 4 distinct |
| `שעת התחלת תוכנית` | program_start | `19:58`, `21:29`, `22:33`, `22:59` |
| `סוג ברייק` | break_type | `Regular` 111, `EB` 64 |
| `סוג תמחור` | pricing_type | `CPP` 124, `FIX` 51 |
| `מחיר` | price | empty — an output |
| `רייטינג ברייקים מתוכנן` | planned_tvr | **per spot, not per break** |
| `מיקום בברייק` | position_in_break | `0`, `1`..`26`, and `99` |
| `סטטוס` | status | empty — an output |

Four of those columns answer questions this project has open, and two of them
contradict a shape the product enforces everywhere.

---

# PART 1 — RELEVANT TO ISRAEL

---

## 1. Two rating granularities. The incumbent states quarter-hour; our reference month is minute-by-minute; every shekel we hold is computed off the minute

This is the largest measured thing in this document and it was found in our own
data, not on the web.

### 1a. The reference month's TVR is a MINUTE rating, exactly

**Israeli evidence, measured.** `data/Dayparts.csv` is a minute-by-minute rating
table: one row per clock minute (`2:00`, `2:01`, `2:02`, …), four columns named
in Hebrew for the four channels (`עכשיו 14`, `קשת 12`, `רשת 13`, `כאן 11`), with
a `quarter_id` column alongside.

I joined every row of `data/Spots.csv` to that table on
(channel, date, start MINUTE):

```
rows 50,386   time unparsable 3,262   no matching minute 0
TVR equals the minute rating:  47,124 of 47,124   ->  100.0%
```

**Not 99.9%. Every single parseable spot.** The `TVR` on a spot in our reference
month is the channel's rating for the clock minute the spot started in.

And the money is linear in it. `revenue_ils` on the same file equals
`base_rate x TVR x Duration x total_premium` on **50,221 of 50,386 rows**. (The
165 exceptions all carry `TVR = 0` and are priced at an implied rating of
exactly `0.1` — a floor, applied in the data. Our engine's own floor is
`TVR_FLOOR = 0.01` at `kairos/model/audience_model.py:66`, ten times smaller.
0.3% of rows, recorded because a silent floor is the kind of thing that later
reads as a fabricated number.)

### 1b. The incumbent's own file states a QUARTER-HOUR rating, per spot

**Israeli evidence, measured.** In
`data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv`, break `20:40:09`
is 27 rows, one ordered programme, one break — and **two planned ratings**,
switching at an exact quarter line:

```
20:44:30  q=20:30  tvr=6.4  pos=8
20:45:05  q=20:45  tvr=5.9  pos=9      <- the boundary
```

Break `22:33:15`: `3.9` through `22:44:36` at position 23, then `4.3` from
`22:45:06` at position 24. Break `22:59:40`: one quarter, one rating. **Every
multi-rating break in the file splits on `:00/:15/:30/:45` and nowhere else.**

### 1c. How far apart the two are

For every spot in the reference month I compared its minute rating against the
mean of the minute ratings in its own quarter — which is what a quarter-hour
average rating is:

```
spots compared                                    47,075
median |minute - quarter| / quarter                7.15%
mean                                              10.48%
p90                                               23.73%
share of spots differing by more than  5%          62.3%
share of spots differing by more than 10%          36.6%
```

Revenue is linear in the rating, so those are revenue differences of the same
size, per spot.

### 1d. An Israeli trade source states the rule, and states the arbitrage

**MEASURED, and this is the best single quote in the whole research.** The
Israeli Marketing Association's media guide, `ishivuk.co.il/מדריך-מדיה-2/`:

> `כיום נהוג לתמחר ספוט על פי הרייטינג הממוצע ברבע השעה בה הוא שודר. כלומר, אם שודר ספוט בשעה 8:03, עלותו בפועל תהיה מכפלת ה-CPP שנקבע לרצועה ברייטינג הממוצע שהיה בין 8:00-8:14.`

> *Today a spot is priced on the AVERAGE RATING OF THE QUARTER-HOUR in which it
> aired. That is, if a spot aired at 8:03, its actual cost is the slot's CPP
> multiplied by the average rating between 8:00 and 8:14.*

And then, unprompted, the same guide states the arbitrage that follows from it:

> `כיוון שהתשלום עבור הספוט מתבצע לפי רייטינג רבעי שעה, וכיוון שבמהלך מקבצי פרסומות יורד הרייטינג ביחס לתכנית עצמה, השאיפה היא למקסם את רייטינג הדקות ביחס לרייטינג רבעי השעות. נתון זה מושפע רבות ממיקום הספוט בתוך מקבץ הפרסומות.`

> *Because payment for the spot is made by the quarter-hour rating, and because
> during advertising bundles the rating drops relative to the programme itself,
> **the aim is to maximise the minute rating against the quarter-hour rating**.
> This is heavily influenced by the spot's position within the advertising
> bundle.*

That is an independent Israeli industry source stating, in one sentence, both
halves of what this product exists to do: the money settles on the quarter, the
audience is lost by the minute, and the gap between them is won or lost by
placement inside the break. It is the clearest external statement of this
product's own thesis that I found anywhere, and it was on an Israeli trade page,
not in a vendor manual.

**Note precisely what it says the two units are FOR.** The quarter-hour rating
is the BILLING basis. The minute rating is the DELIVERY reality. They are not a
mistake for each other — they are the two sides of the trade. Our product holds
both (finding 1a, 1b) and labels neither.

**What it means commercially.** The trade document says the currency is
"**Jewish households, quarter-hour rating, overnight plus one**"
(`docs/media-domain-from-the-trade.md:105-108`). The incumbent's file is the
proof of the middle third of that sentence at the level money is computed, and
the media guide is independent Israeli corroboration of the same rule. Our
reference month's revenue is computed on the other unit, and it is not a
rounding difference: 7% at the median and over 10% on more than a third of
spots.

**Whether Israel does it differently.** This IS the Israeli way. The finding is
not that Israel is unusual; it is that we hold two units, we know what each is
for, and nothing in the data says which is which.

**Whether our product has it.** Half, and the halves are not connected.

- The per-second, per-quarter-hour settlement math exists:
  `kairos/optimize/qh_billing.py:1-19`, owner-gated OFF at
  `kairos/optimize/pricing.py:206`. `docs/quarter-hour-billing.md` measured its
  activation at +7.45%.
- The daily path does read the incumbent's own per-row quarter figure:
  `kairos/data/loaders.py:96` maps `רייטינג ברייקים מתוכנן` to `planned_tvr`,
  and `kairos/export/spots.py:243` names it a per-row field.
- Nothing anywhere records WHICH unit a held TVR is in. There is no field on
  `data/Spots.csv`, `data/Programmes.csv` or `data/Dayparts.csv` that says
  "minute", and none on the daily file that says "quarter". The re-audit found
  the same absence for overnight-versus-consolidated
  (`docs/audits/trade-reaudit-2026-08-09.md:121-126`); this is the same hole one
  level down, and it is the level the money is computed at.

**What it blocks if it stays.** The quarter-hour restatement is currently framed
as an owner-gated pricing option worth +7.45%. On this evidence it is not an
option: it is the difference between the unit the reference month is in and the
unit the market settles in, and the +7.45% is the net of a per-spot spread whose
median magnitude is 7.15% in both directions. Until a unit is recorded on the
data, nobody can tell which of our figures are in which currency.

---

## 2. The rating is a property of the SPOT, not of the break, and the pod surface takes the first row

**Israeli evidence.** The same two breaks as 1b: one break, one air time, two
ratings, because a quarter line runs through it.

**What it means commercially.** A spot's rating is not inherited from its break.
It is looked up from the quarter it lands in. Two spots thirty-five seconds
apart in the same pod settle differently.

**Whether our product has it.** The break-level surfaces carry one rating per
break. `kairos_api/break_api_pod.py:231` takes break-level fields from
`ordered.iloc[0]` — the first row of the pod. On break `20:40:09` the first row
is a `6.4`, and fourteen of that break's twenty-seven spots are `5.9`. I did not
drive the endpoint, so this is a read of the code and not a measured payload:
**the shape is first-row-wins, and on this file the first row is not
representative of its break.**

**What it blocks if it stays.** Any figure that prices a whole break at one
rating is wrong on every break that crosses a quarter line — on this file, two
of the four large breaks. It also blocks the honest version of the transcript's
"the time is a range": moving a break by forty seconds moves spots onto a
different rating, which is a revenue consequence of a placement the optimiser
makes freely today.

---

## 3. The incumbent encodes position as `0`, `1..N`, and `99` — and `99` is Last

**Israeli evidence.** The shipped traffic file, measured. `מיקום בברייק` takes
28 distinct values. Grouped by break, commercials only:

```
20:40:09  [1..18, 99]
21:01:37  [1..23, 99]
22:03:06  [1..26, 99]
22:33:15  [1..26, 99]
22:53:49  [1..6,  99]
22:59:40  [1..17, 99]
```

Every break that carries commercials carries **exactly one `99`**, the ordinals
below it are contiguous from 1, and the `99` row's air second is the latest in
its break (`20:49:47` after position 18's `20:49:17`). And **every one of the 51
`חסות` rows is position `0`** — no sponsorship anywhere in the file holds an
ordinal.

So the incumbent's vocabulary is: `0` = a sponsorship notice, outside the
numbered sequence; `1..N` = the ordinals; `99` = Last.

**What it means commercially.** This is the transcript's `L`
(`docs/media-domain-from-the-trade.md:73-78`) as it exists on the wire. The
trade says Last is a distinct position and not the last ordinal; the incumbent
agrees so completely that it gives Last a sentinel that can never collide with
an ordinal, and never numbers the tail.

It also settles a question our own position work left open. The transcript's
preferred set is "1 to 5 plus L", and it is easy to read that as a short break.
It is not: the observed pods run to 26 commercials plus a Last. **1 to 5 plus L
is six worthwhile slots out of twenty-seven**, which is why the preference is
worth paying for.

**Whether Israel does it differently.** The `0`/`99` encoding is the incumbent's
and I found no international equivalent that uses these numbers. What I can say
is that the CONCEPT of a sentinel position outside the ordinals is not
Israel-specific (see Part 2, "position codes"), but the codes are.

**Whether our product has it.** Partly, and the two halves disagree on `99`.

- `kairos_api/break_api_pod_spots.py:188-210` already decodes both sentinels:
  `LAST_POSITION_CODE` maps to `"L"`, `UNPOSITIONED_CODE` maps to a spot that
  holds no position, with the honest reason "No position was requested for this
  spot." Correct, and the surface that reads the file gets it right.
- `kairos/optimize/positions.py:181-196` — `occupied_tokens()` decides Last by
  `int(ordinal) == int(size)`, i.e. **it derives Last from the break size and
  has no knowledge of the sentinel.** On the shipped file a Last spot arrives
  as the integer 99 in a break of 19, so `99 != 19` and the spot is not
  recognised as Last. `parse_preferred` accepts the word `"last"` and the token
  `"L"` (`positions.py`, `_WORD_TO_TOKEN`), and neither is `99`.
- So the pod board can say "L" for a spot that
  `preferred_position_rate(...)` — the number the channel and the agency audit
  each other with, `positions.py:322` — would count as ordinal 99 and never as
  Last.

I did not run `preferred_position_rate` against the shipped file, because the
re-audit already established it has zero callers outside its own test
(`docs/audits/trade-reaudit-2026-08-09.md:51`). The disagreement is stated from
reading both, and it is worth a test before it is worth a fix.

**What it blocks if it stays.** The preferred-position percentage — already the
re-audit's finding 1 — would be wrong as well as unreachable, and wrong in the
direction that matters: it would score every Last as a miss.

---

## 4. `סוג ברייק` = `EB`, and on the shipped file every EB break spans two ordered programmes

**Israeli evidence.** The shipped traffic file. `סוג ברייק` takes `Regular`
(111 rows) and `EB` (64 rows). The owner has an open question about what EB
means (`docs/ux-gauntlet/decisions-for-owner.md:240`, and
`docs/ux-gauntlet/RESUME-HERE.md:101-104` records that nothing in the trade
document says). **I did not find out what EB stands for and I am not going to
guess a name for it.** What I did was measure the one thing the file can settle,
which the open question did not have:

```
break      type      ordered programmes in it
20:24:23   Regular   1   חדשות 13
20:40:09   Regular   1   חדשות 13
21:01:37   Regular   1   חדשות 13
21:22:12   Regular   1   חדשות 13
21:23:10   Regular   1   חדשות 13
21:24:38   Regular   1   חדשות 13
22:03:06   Regular   1   המקור - עונה 24
22:33:15   EB        2   המקור - עונה 24  |  המקור - עונה 24 - דיון באולפן
22:53:49   Regular   1   המקור - עונה 24 - דיון באולפן
22:59:40   EB        2   המקור - עונה 24 - דיון באולפן  |  היום שהיה
```

**Eight of eight Regular breaks hold spots ordered against one programme. Two of
two EB breaks hold spots ordered against two, and in both cases the two are
adjacent in the evening's running order.** That is a clean separation on a small
sample, and it is evidence rather than a definition.

**INFERRED, and labelled as such:** an EB is a break at a programme boundary —
the break between one programme and the next, which both programmes' orders can
be filled from. I am not confident enough to name it in the product, and under
the standing ruling a foreign term for the same idea is not a licence to rename
anything (see Part 2, "the break between programmes").

**Whether our product has it.** No, and worse, the name is already taken by
something else.

- `kairos/data/loaders.py:94` maps `סוג ברייק` to `break_type`, so the file's
  `Regular`/`EB` does travel into the frame.
- `kairos/export/incremental.py:60` defines a DIFFERENT `_break_type`, computed
  from the break's LENGTH: `short` under 90s, `medium` under 180s, else `long`.
  `kairos/export/schedule.py:103` puts that one on the exported schedule.
- `tv-break-dashboard/src/plan/break/BreakLibraryPage.jsx:45,100` renders
  `row.break_type` through `breakLengthLabel`
  (`tv-break-dashboard/src/shell/labels.js:120-128`, which knows only
  `short/standard/medium/long`) under the Hebrew column heading **`סוג ברייק`**
  — the incumbent's own words for the Regular/EB field.
- `tv-break-dashboard/src/plan/break/PodBoard.jsx:147-149` is the honest one: it
  prints the file's value raw and labels it "Break type, from the file /
  סוג ברייק, מהקובץ".

So an operator meets the Hebrew phrase `סוג ברייק` on two screens, and on one of
them it means the length class and on the other it means the incumbent's field.
`data/breaks.csv:1` — the product's own break record — has no break-type column
at all, only `is_gold`.

**What it blocks if it stays.** The owner's own open question cannot be answered
by the product even after he answers it, because there is nowhere to put the
answer; and the length-class column standing under the incumbent's Hebrew label
will read to an operator as a wrong value rather than a different field.

---

## 5. `סוג תמחור`: the incumbent states the pricing basis per spot, and our engine only half-reads it

**Israeli evidence.** The shipped traffic file: `סוג תמחור` takes `CPP` (124
rows) and `FIX` (51). On this file it aligns exactly with `סוג תשדיר` — every
`פרסומת` is CPP, every `חסות` is FIX. The trade document's own words:
"Sponsorships are the exception" (`media-domain-from-the-trade.md:17`), and
`kairos/optimize/objective.py:10` already records "Sponsorships are usually
priced at a fixed amount (FIX) rather than CPP."

**What it means commercially.** A CPP spot's price is discovered from the rating
after the fact; a FIX spot's price was agreed in advance and does not move with
the rating. They are two different revenue objects sharing a break.

**Whether our product has it.** In the export path, yes; in the campaign store,
it is dead.

- `kairos/export/spots.py:364-367` reads `pricing_type` and branches:
  `if pricing_type == "FIX" and stated_price > 0`. Live.
- `kairos_api/campaigns_commitment.py:98-137` validates a campaign-level
  `price_model` of `cpp | flat` and `kairos_api/campaigns_api_store.py:200`
  stores it — and the re-audit measured that `grep price_model kairos/` has no
  hits (`docs/audits/trade-reaudit-2026-08-09.md:344-350`). So the product holds
  the same distinction twice: once as the incumbent's own per-spot `FIX`, which
  works, and once as a campaign-level `flat`, which prices identically to `cpp`.
- The vocabulary also differs: the incumbent writes `FIX`, the campaign store
  writes `flat`. `kairos/optimize/agreements.py:7` uses `FIX`.

**What it blocks if it stays.** Nothing today, because the export path is the
one that computes money. It matters because the campaign-level field is the one
an operator sets, and setting it does nothing.

---

## 6. `תוכנית מוזמנת` — the ORDERED programme is a per-spot field, and two spots in one break can name different ones

**Israeli evidence.** The shipped traffic file. The column's own name is
"ordered programme". Inside break `22:33:15`, positions 1 to 12 are ordered
against `המקור - עונה 24` and positions 13 to 26 against
`המקור - עונה 24 - דיון באולפן` — one break, one air time, two ordered
programmes. And the trade document's order fields are
`שם ערוץ, תאריך, שעה, שם תוכנית, אורך תשדיר`
(`docs/media-domain-from-the-trade.md:58-59`), with the warning that "the time
is approximate and everyone knows it... the spot may land in a different break
entirely" (lines 61-64).

**What it means commercially.** This is the order surviving into the traffic
file. The agency ordered a programme; the channel placed the spot where it
placed it; the file keeps BOTH the request and the outcome, side by side, and
that is exactly how a channel proves it honoured an order it did not honour
literally.

**Whether our product has it.** No. The re-audit's finding 6
(`docs/audits/trade-reaudit-2026-08-09.md:209-236`) is that there is no order
entity anywhere, and of the trade's five order fields the product holds channel
and a flight window and "no time, no programme and no spot length". The finding
this research adds is narrower and more useful: **the order's programme and the
order's time are already in the file we load every day, under
`תוכנית מוזמנת` and `שעה`, and we drop them.** `kairos/data/loaders.py:91-92`
renames them to `program` and `spot_time`; nothing downstream compares the
ordered programme to the programme the break actually sat in.

**What it blocks if it stays.** The transcript's "requested 20:40, aired 20:50"
question is unanswerable — not for want of a data model, but because the two
sides of the comparison are already loaded and never put next to each other. It
also blocks the whole preferred-position audit conversation, which is a
comparison between what was asked for and what was delivered.

---

## 7. The rating currency has no universe recorded anywhere, and our own two documents disagree about deferred viewing

**Israeli evidence, and it is a contradiction inside this repository.**

The trade document, which outranks everything:

> The trading currency is **Jewish households, quarter-hour rating, overnight
> plus one**, where plus one is deferred viewing.
> (`docs/media-domain-from-the-trade.md:105-108`)

> **The final rating is only known the day after broadcast**, and it moves: one,
> two, even three points can be added. (lines 110-113)

Our earlier web research says the opposite about the deferred half:

> **Delayed viewing:** The IARB counts live viewing plus time-shifted viewing up
> to 2:00 AM the same night. Catch-up viewing on the following days is reported
> separately and is generally not included in the traded currency.
> (`docs/campaign-rate-card-research.md:40-42`)

**These cannot both be true, and the transcript wins.** The record here is that
`campaign-rate-card-research.md` is web-sourced and the transcript is a person
who trades in this market. I did not resolve it in the web's favour and I am not
recommending that anyone edit either file; the correct move is to put the
question to the same media professional, because he is the source that outranks
both.

The transcript is also the only one of the two that is consistent with the
second sentence: a rating that is final at 02:00 the same night cannot "only be
known the day after broadcast" and cannot gain three points.

**On the universe.** `docs/campaign-rate-card-research.md:22-25` records "One
TVR point equals one percent of the total Israeli Jewish TV-household universe",
which AGREES with the transcript's "Jewish households". Both of our documents
say Jewish households; the product says all viewers:

- `kairos_api/campaigns_commitment.py:37-88` — five audiences, `all_viewers` the
  only one marked measurable. **Jewish households is not among them.**
- `data/Programmes.csv:1` and `data/Spots.csv:1` carry a bare `TVR` column, and
  the shipped daily file carries `רייטינג ברייקים מתוכנן`, with no universe
  qualifier on either.
- Nothing records whether a held TVR is an overnight or a consolidated figure
  (re-audit finding 3, `docs/audits/trade-reaudit-2026-08-09.md:121-126`).

**What this research adds to the re-audit.** Two things. First, the two
documents in this repo already agree on "Jewish households" and the product
disagrees with both, so this is not a matter of a thin source — it is a gap
between our own documents and our own code. Second, the deferred-viewing
contradiction is new, and it is the half that decides whether the day-after
revision the transcript describes is a thing the product must model at all.

**What it blocks.** Every ILS figure is denominated in a unit the market does
not settle in, and a client who bought on the market's currency cannot state
that goal. Unchanged from the re-audit; the evidence is now doubled.

---

## 8. Israeli commercial lengths are arbitrary whole seconds, and the 30-second spot is not the trading unit

**Israeli evidence, measured on `data/Spots.csv`, 50,386 rows, four channels,
November 2024.** Restricted to `סוג תשדיר = פרסומת` (21,365 commercials):

| Length | Count | Share |
| --- | --- | --- |
| 15s | 4,261 | 19.9% |
| 20s | 1,745 | 8.2% |
| 6s | 1,185 | 5.5% |
| 30s | **835** | **3.9%** |
| 10s | 676 | 3.2% |

**61 distinct commercial lengths, from 5s to 171s. Only 36.7% fall in the
standard set {10, 15, 20, 30, 45, 60}.** The rest are 7, 8, 11, 12, 16, 17, 18,
21, 22, 25, 32, 34, 35, 36 and so on — the whole seconds the trade document
insists on: "Length is in whole seconds. There are no milliseconds... the trade
unit is the second" (`docs/media-domain-from-the-trade.md:64-66`).

Sponsorships are the opposite and confirm the other half of the picture: 16,385
of 18,550 `חסות` rows (88%) are exactly 6 seconds.

**Half of our own research document is wrong, and half of it is right — and I
had this backwards until an Israeli trade source settled it.**

`docs/campaign-rate-card-research.md:143-149` says two things. "5-10s commercial
spots are non-standard and unusual in the spot market" is **wrong** on our data:
6-second commercials outnumber 30-second ones. But "The 30-second spot is the
universal trading unit" is **right**, and it is right about PRICING while saying
nothing true about LENGTH DISTRIBUTION. Those are two different claims and I
first read them as one.

**The Israeli source that settles it (MEASURED).** The Israeli Marketing
Association's media guide, `ishivuk.co.il/מדריך-מדיה-2/`:

> `האורך הבסיסי של תשדיר בטלוויזיה הינו 30″. הוא מהווה את ה-100%… פקטור אורכים קצרים מ-30″ מחושבים לפי טבלה שנקבעה בשוק ואינה מבטאת את החלק היחסי של האורך לעומת 30″. תשדירים ארוכים מ-30״ יחושבו על פי פרורטה (לדוגמא: פקטור של תשדיר 45″ יהיה 150%).`

> *The base length of a television spot is 30″, which constitutes 100%. The
> factor for lengths SHORTER than 30″ is computed from **a table agreed in the
> market which does not express the proportional share** of the length against
> 30″. Spots LONGER than 30″ are computed pro rata (for example, the factor for
> a 45″ spot is 150%).*

**So the price curve is a kinked function, not a line.** Above 30 seconds it is
linear in seconds. Below 30 seconds it is a market-agreed table that is
explicitly non-proportional — and a 15-second spot does not cost half a
30-second spot.

**Whether our product has it. No — and this corrects what I wrote first.**
`config/optimization_weights.yaml:6` prices from
`base_price_per_second_per_tvr_point`, strictly linear in duration
(`kairos/optimize/objective.py:42`,
`revenue = cpp * rating_points * (duration_seconds / unit_seconds) * premium`).
Linear is correct above 30 seconds and **wrong below it**, which is where most
of the market lives: on our own data **93.9% of commercials are shorter than 30
seconds**. Our earlier research note already recorded the direction ("15s is
priced at approximately 60-75% of 30s rate",
`campaign-rate-card-research.md:145-146`) — 60-75% where linear says 50%.

I am not proposing the table. Nobody has the actual Israeli factor table; the
guide says it exists and does not print it, and inventing numbers is forbidden.
**The finding is that the engine assumes a linearity the market does not have,
in the length range that carries almost all of the volume, and the sign of the
error is known: linear pricing UNDER-charges short spots.**

**What stays true, and should be defended.** The per-SECOND basis is the right
primitive — the trade document is explicit that "the trade unit is the second"
and our data shows 61 distinct lengths. The fix, when the table exists, is a
length-factor layer on top of a per-second base, not a retreat to a 30-second
unit.

---

## 9. The agency column already carries the direct-client marker, and a government buyer

**Israeli evidence.** `משרד / MB` on the shipped traffic file, all nine values:

```
יוניברסל 65   OMD 56   ישירים 15   יוניון 12   פובליסיס 12
מדיהקום 6     פיתוח עסקי 5   רואים קונים 2   לפמ 2
```

Two of those nine are not media agencies.

- **`פיתוח עסקי`** — literally "business development". A channel's own sales
  desk, sitting in the agency column. MEASURED that the token is there;
  INFERRED that it is the direct-sales route.
- **`לפמ`** — the acronym of the Israeli Government Advertising Bureau. MEASURED
  that the token is there and that it buys; the expansion is INFERRED from the
  acronym and from finding 9 below.

**What it means commercially.** The re-audit's finding 7
(`docs/audits/trade-reaudit-2026-08-09.md:239-266`) is that "a direct client is
byte-identical to a failed agency lookup" and that
`kairos_api/campaigns_read_clients.py:38-41` treats an unlinked client as a data
defect. This research adds the thing that makes it fixable: **the incumbent does
not leave the direct client blank. It writes a name in the agency column.** The
distinction the product cannot express already exists as a value in the file the
product loads.

I am deliberately not proposing a schema. The owner has to confirm what
`פיתוח עסקי` is before anything is built on it, and one evening of one channel
is five rows.

**What it blocks.** Pricing a direct deal, and one of the six run priorities the
transcript names (`docs/media-domain-from-the-trade.md:37-47`).

---

## 10. On the public broadcaster, the commercial rows are almost all government and public bodies

**Israeli evidence.** `data/Spots.csv`, `כאן 11` (Kan 11, the public
broadcaster), rows typed `פרסומת`: 281 of them. By campaign:

```
45  משרד החינוך                     Ministry of Education
38  נגה תקשורת - נגה ניהול מערכות החשמל   Noga, the electricity system operator
33  שירות בתי הסוהר                 Israel Prison Service
32  המועצה להסדר ההימורים בספורט     Sports Betting Council
30  משרד הרווחה                     Ministry of Welfare
24  מפעל הפיס                       Mifal HaPayis, the national lottery
18  משרד הביטחון - אגף שיקום         Ministry of Defence, rehabilitation
18  קופ"ח מאוחדת                    Meuhedet health fund
```

Kan 11 also carries 1,142 `חסות` rows and 2,616 `פרומו` rows. For comparison,
Keshet 12 carries 11,304 commercials against 281 on Kan.

**MEASURED:** on our data, Kan 11's commercial inventory is overwhelmingly
government ministries, statutory bodies and public corporations.

**And the law says Kan television may not carry commercials at all.** MEASURED,
`חוק השידור הציבורי`: §70(א) permits paid spots **ברדיו** — on radio — only;
§69(א) forbids embedding promotional messages in content; §72 permits
sponsorship on radio and television, **except §72(ב), which bars sponsorship
entirely on the children's and youth television channel**. The Broadcasting
Council's 2021 rules set 9 minutes an hour and 10% of the day for Kan
television, and 15% for radio.

**So this is now a contradiction, and it is a finding about our data rather than
about Kan.** The statute says Kan television carries no commercials; our file
carries 281 rows on `כאן 11` typed `פרסומת`. Three readings fit and I did not
resolve between them:

1. the rows really are sponsorship or public-service announcements, and
   `סוג תשדיר` in this export is the SELLER's label rather than the regulatory
   class;
2. government and public-body messages sit in a category the statute treats
   differently from commercial advertising;
3. the classification in the source export is simply loose on this channel.

Whichever it is, the consequence is the same and it is worth stating plainly:
**`סוג תשדיר` cannot be assumed to be a regulatory classification.** Anything
that enforces a rule by reading it — an ad-minutes cap that counts `פרסומת` and
not `חסות`, say — is enforcing the seller's label, not the regulator's category.

Note that `לפמ` — the government bureau — appears as a buyer in the Reshet 13
daily file too, so government money is not confined to the public channel.

**Whether our product has it.** No. There is no advertiser-class field and no
per-channel eligibility rule anywhere. `data/kairos_settings.json` holds a
single `operator_channel` (`רשת 13`) and one regulatory profile.

**What it blocks.** Nothing today, because the operator channel is Reshet 13. It
would block the moment the product is pointed at Kan, and it is the kind of rule
that is invisible until it is violated.

---

## 11. A Hebrew typographic mark for "seconds" that the copy-length check does not read

**Israeli evidence.** The version names on the shipped traffic file declare
their own length, and they use three different marks to do it:

```
'15'''            straight apostrophes
'Mix&wash 25"'    straight double quote
'19״'             Hebrew GERSHAYIM, U+05F4
'26״ מחליפה'
'6״ חדכ'
'טורנדו אמיגו ראשי 52״'
'נגזרת 25״'
'אריאל ענקי 30 שניות'   the word שניות
```

**Whether our product has it.** Not the Hebrew mark.
`kairos_api/break_api_pod_spots.py:95`:

```python
_COPY_LENGTH_MARK = re.compile(r"(\d+)(?:[\"']|\s?שניות)")
```

The character class holds the ASCII double quote and the ASCII apostrophe. It
does not hold `״` (U+05F4). Measured on the shipped file: 35 version names
declare a length, the regex reads 30 of them, and **5 (14%) use the Hebrew
gershayim and are read as "The copy version names no length."**

The check itself is a good one and its own docstring explains the care taken to
avoid a false alarm ("the 15 in `סרט 15 ימי מכירות` is a count of sale days",
lines 92-94). The gap is one character in a character class, and it is exactly
the sort of thing this research was asked to look for: **a piece of Israeli
notation the product cannot read.** Adding `״` (and, for symmetry, the geresh
`׳`) is a one-character-class change; I have written no code.

**What it blocks.** One in seven copy-length mismatches goes unreported on the
one surface built to make them impossible to miss.

---

## 12. House numbers are prefixed, and the prefix separates commercial from sponsorship

**Israeli evidence.** 76 distinct house numbers on the shipped file. Grouped by
their three-letter prefix against `סוג תשדיר`:

```
C**  ->  פרסומת  only:   CID 70, CMK 17, CGB 10, CRP 10, CBN 9, CBB 8
H**  ->  חסות    only:   HID 35, HGB 6, HMK 6, HBN 2, HRP 2
```

**Perfect separation on 175 rows: `C` for commercial, `H` for sponsorship
(חסות), and the same second-and-third letters recur across both families
(`ID`, `GB`, `MK`, `BN`, `RP`).** MEASURED. What the second and third letters
encode is NOT CONFIRMED — the recurrence across both families suggests an
advertiser or agency code, but six pairs on one evening is not evidence and I
did not test it against `data/campaign_assets.csv`.

The trade document says the house number is "the channel-side identifier for one
creative version, issued by Owner" and that "the same creative has a different
house number per channel" (`docs/media-domain-from-the-trade.md:164-167`).

**Whether our product has it.** The field, yes —
`data/campaign_assets.csv` carries `house_number`, and
`kairos/data/loaders.py:90` maps it from the daily file. The structure, no:
nothing parses the prefix, and nothing would notice a `C`-prefixed house number
on a row typed `חסות`. Cross-channel creative identity is still absent
(`docs/trade-gap-analysis.md:117-126`).

**What it blocks.** Nothing today. Recorded because the trade document names the
house number as the binding between Owner and Jumbo, and because a checkable
internal structure in an identifier is worth knowing about before anyone builds
the binding.

---

## 13. The regulator's own placement rules are free to read in Hebrew, and the product enforces the wrong SHAPE as well as the wrong number

**Israeli evidence, primary and MEASURED.** The rulebook is
`כללי הרשות השניה לטלויזיה ורדיו (שיבוץ תשדירי פרסומת בשידורי טלויזיה), תשנ"ב-1992`
— the Second Authority's rules on the PLACEMENT of advertising spots in
television broadcasts. Full text, free, at
`he.wikisource.org/wiki/כללי_הרשות_השניה_לטלויזיה_ורדיו_(שיבוץ_תשדירי_פרסומת_בשידורי_טלויזיה)`
and `nevo.co.il/law_html/law00/4941.htm`. Quoted from the text:

- **§3(a)** — `זמן השידור המרבי לתשדירי פרסומת שבעל זיכיון רשאי להקצות בכל שעה, לא יעלה על 10 דקות`
  — the maximum broadcast time for advertising spots a licensee may allocate in
  any hour shall not exceed **10 minutes**.
- **§3(b)(1)** — between 20:00 and 24:00 the allocation is flexible **provided
  the total across those hours does not exceed 40 minutes**.
- **§10(b)** — `לא ישובץ תשדיר פרסומת וקדימונים במהלכה של תכנית אלא באתנחתה`
  — a spot or promo may be placed mid-programme **only at an `אתנחתה`**, a
  natural pause.
- **§20(b)** — transition periods of a **minimum of 3 seconds**, with visual
  identification.
- **§13a(a)** — live sport: placement permitted at halftime, between quarters,
  and at timeouts.

Companion rulebooks, also public: advertising ethics
(`תשנ"ד-1994`, `nevo.co.il/law_html/law00/4944.htm`) and Kan's own advertising
rules (`תשפ"א-2021`, `nevo.co.il/law_html/law00/201776.htm`).

**And the primary law says something different again.** `חוק הרשות השניה` §85(א)
sets the ceiling at `שש דקות` — **six minutes** an hour for television — while
the rules made under it allow ten, with §3(א1) making six the floor of any
punitive reduction. So the statute and the rules are deliberately two numbers,
and neither is twelve.

**Four numbers in this repository, and none of them is either legal number.**

| Source | Ad minutes per hour |
| --- | --- |
| `data/regulatory_guardrails.json` | `max_ad_minutes_per_hour: 12.0` |
| `data/kairos_settings.json` | `max_ad_minutes_per_hour: 12.0` |
| `docs/campaign-rate-card-research.md:224` | 10, correctly, and correctly cited |
| **rules §3(א)** | **10** |
| **law §85(א)** | **6** |

**One thing the product gets exactly right, and it should be defended.** §1 of
the rules defines the hour: `שעה` is
`60 הדקות שבין שעה תמימה לשעה התמימה שלאחריה` — the sixty minutes from one
whole hour to the next. A CLOCK hour, not a rolling window.
`kairos/optimize/guardrails.py` buckets by `item.hour`, the clock hour, which is
the definition the rules use. Anyone who "improves" that into a rolling
sixty-minute window would be moving away from the law, not toward it.

**Three more numeric limits, and our data's verdict on each.**

- **§6, maximum spot length 90 seconds.** MEASURED on `data/Spots.csv`: exactly
  three commercial rows exceed it, all 171 seconds, all the same creative
  (`2024-11 - ראובני פרידן — סרט חטופים`, a hostages film) on three different
  channels on 15 and 17 November 2024. Either a permitted exception or three
  rows worth asking about; three rows in 21,365 is not a pattern.
- **Sponsorship, ten seconds each.** MEASURED: **0 of 18,550** `חסות` rows
  exceed 10 seconds. The cap is respected perfectly.
- **§3(ג), no more than 10% of total daily broadcast time**, and §3(ו) partial
  hours pro rata. `data/kairos_settings.json` carries
  `max_daily_ad_minutes: 160`, which is 11.1% of a 24-hour day — again more
  permissive than the rule, on the same side as the 12.

**The largest single finding here: six clauses are REPEALED, and we still
enforce them.**

§§11, 12, 13, 14, 15 and 17 of the placement rules all now read `(בוטל)` —
repealed. Between them they were the structural clauses. **On the current text
there is no minimum interval between breaks and no maximum number of breaks per
programme for Second Authority licensees.** The only structural gate left is
§10's `אתנחתה` test: a break in continuity
`שהיתה מתרחשת מאליה, גם ללא שיבוץ תשדיר הפרסומת` — one that would have occurred
anyway, without the advertisement. That is editorial judgement about the
programme, which a system must consume as an input and cannot compute.

Our product enforces both repealed shapes as regulatory guardrails:
`data/kairos_settings.json` carries `max_breaks_per_hour: 4` and
`min_break_spacing_minutes: 7`, and `kairos/optimize/guardrails.py`
(`check_breaks_per_hour`) enforces the first unconditionally.

**The direction of this error is the opposite of the minutes error, and that
matters.** On minutes we are more permissive than the law. On break structure we
are more restrictive than the law — we forbid schedules that are currently
legal, which costs revenue silently and forever. Neither of these is recorded
anywhere as a policy choice; both read as regulation.

Benchmarks for the other two Israeli regulators, which DO cap structurally:
cable dedicated channels at most 10 bundles an hour (8 if drama or film exceeds
half the hour); Kan at most 4 bundles per television hour. So a four-per-hour
number is real somewhere in Israel — it is Kan's, not the Second Authority's.

**Sponsorship is ADDITIVE inventory with its own budget, and it is anchored.**
Sponsorship rules §8: `זמן השידור של הודעות חסות… לא ייכלל בזמן המרבי המותר לתשדירי פרסומת`
— sponsorship time **is not counted** in the maximum permitted for advertising
spots. It has its own ceilings (§5): 10 seconds each; **54 seconds an hour**;
20:00-24:00 a pooled maximum of **3:36**; 24:00-20:00 a pooled maximum of
**18:00**. And §6 anchors it: only immediately before or after its own
programme, or immediately adjacent to an advertising bundle inside that
programme.

That last clause is a real coupling the product does not model: **mid-programme
sponsorship inventory exists only if a break is scheduled in that programme.**
Deciding not to place a break does not merely forgo the break's spot revenue, it
destroys the sponsorship slots that could only have sat beside it. On our own
data sponsorship is 18,550 of 50,386 rows — not a rounding error.

Our engine has one ad-minutes budget covering both, and
`kairos/optimize/guardrails.py` sums break duration without separating
`פרסומת` from `חסות`.

**The regulator also sets a make-good exchange rate.** §26(א): each lost peak
minute (20:00-24:00) is compensated by one peak minute **or three off-peak
minutes**; each lost off-peak minute by one off-peak minute or one third of a
peak minute. **A 3:1 peak-to-off-peak rate**, requiring the Director's PRIOR
WRITTEN approval. `kairos_api/makegood_store.py` and `data/make_goods.csv` model
a decision ledger with states and actors; they carry no daypart exchange rate
and no external-approval state.

**Content-timing rules that are mechanisable, and are not modelled at all.**

- **Alcohol** (ethics §27): barred in or around programmes aimed at or appealing
  to minors, and a programme is DEEMED minor-directed if it **started** before
  22:00 on weekdays and Shabbat, or before 22:30 on Fridays. Note the trigger is
  the programme's START, so a 22:30 break inside a 21:45 programme still cannot
  carry alcohol. Threshold 1.2% ABV here and 2% in ethics §23 — two thresholds
  in two instruments.
- **Sex, violence, cruelty** (§36): not between **14:30 and 21:00 on weekdays**,
  and **06:00 to 21:00 on Shabbat and מועד**. The most mechanisable rule in the
  corpus, and calendar-driven — which is exactly what
  `kairos/data/israel_calendar.py` already computes.
- **Children** (§§28-35): merchandising advertisements barred **±2 hours**
  around the related programme; character advertisements **±30 minutes** around
  children's programming; §34 bars an advertisement containing programme clips
  from being **first in the bundle** adjoining that programme — a
  position-scoped content rule, the same axis as finding 3. §28(ג) obliges the
  licensee to plan around **school holidays and unplanned school closures**.
- **Blackout days** (§9(א)), and they are **ceremony-anchored, not
  midnight-anchored**: Memorial Day runs from the start of its opening ceremony
  to the end of the Independence Day opening ceremony; Holocaust Remembrance Day
  from its opening ceremony to **19:55** the next day; Tisha B'Av from **20:00**
  on the eve to the end of the fast. Kan and the cable rules end the Holocaust
  Remembrance ban at 20:00, not 19:55 — a genuine per-regulator branch. §9(ב)
  gives the Director an unbounded override on days of national mourning or
  unforeseen disaster, so the schedule must be re-plannable at short notice.

**Two measured absences, which are as valuable as the rules.**

- **There is no ban on advertising on Shabbat or festivals** in any instrument
  read. Shabbat appears only as a MODIFIER of other rules (the widened §36
  window, the alcohol deeming time). Do not assume a Shabbat advertising ban.
- **Gambling could not be confirmed.** No clause naming `הימורים`,
  `מפעל הפיס` or `הטוטו` appears in any rulebook read, and finding 10 shows both
  the lottery and the sports-betting council advertising on our own data. The
  restrictions are likely in a Finance Ministry permit that was not read. **Do
  not encode a gambling watershed.**

**Loudness: Israel has a hard number and it is neither CALM nor R128.**
`כללי עוצמת הקול בשידורים, התשס״ט–2009` §2 sets an ATSC A/52 `dialnorm` target
of **−28 dB, within −26 to −30**, binding on programmes as well as
advertisements. `data/campaign_assets.csv` carries `loudness_lufs` and
`loudness_standard` — LUFS is the R128/A/85 unit, and the Israeli rule is stated
in dialnorm. Worth one question before anything validates against it.

**Live legal change, and a review date.** Both consolidated texts now carry
`ס״ח תשפ״ו, 1024 | חוק התקשורת (שידורים)` followed by
`בג״ץ 49765-07-26 | צו ארעי`. Per Hebrew Wikipedia (secondary, not primary), a
new Communications (Broadcasting) Law passed 16 July 2026 and was partly frozen
by a High Court interim order on 19 July 2026. **The new law's own text could
not be read and its effect on the minute caps is unknown.** Everything above is
current as of the consolidated texts read on 2026-08-09 and should be
re-checked.

**What the traffic log shows.** `data/Spots.csv`, 1,652 channel-clock-hours
carrying at least one commercial: commercials only, median 3.9 min, p90 9.2,
max 22.8, with **7.4% of channel-hours over 10 minutes and 4.3% over 12**;
commercials plus sponsorship, median 4.4, p90 10.9, 13.8% over 10 and 6.8% over
12. So a flat per-clock-hour cap at either number is not what the observed
schedule obeys — which §3(b)(1) explains, because prime is a **pooled
forty-minute budget across four hours**, not four ten-minute ceilings.

**The shape is the finding, not the number.** `kairos/optimize/guardrails.py`
buckets by `item.hour`, the clock hour, and compares each bucket to a single
ceiling. Three modelling choices are embedded there and none is recorded as a
choice:

1. a per-hour ceiling, where §3(b)(1) gives a four-hour pooled budget for
   20:00-24:00;
2. it sums BREAK duration, where §3(a) limits the time allocated **to
   advertising spots** — and on our own data commercials, sponsorships and
   promos are different `סוג תשדיר` values with different lengths;
3. it applies one limit to `פרסומת` and `חסות` alike, where the Second
   Authority regulates sponsorship notices under separate rules.

And `אתנחתה` — the natural-pause requirement in §10(b) — has no representation
anywhere in the product. Break placement is optimised against retention and
guardrails; nothing marks where a programme actually pauses.

The re-audit already found that the enforced values come from
`data/kairos_settings.json` and not from the attested
`data/regulatory_guardrails.json`, and that the declared cutover has no
production caller (`docs/audits/trade-reaudit-2026-08-09.md:359-379`).

**What it blocks.** The product can refuse a prime-time schedule §3(b)(1) allows
and pass hours §3(a) forbids, and the operator cannot see which reading is in
force. Note the direction: 12 is more permissive than 10, so the current setting
errs toward over-selling, not under-selling.

**Honest limit.** I read the quoted clauses at the URLs above. I did not read
the whole rulebook, I did not check for later amendments, and I am not counsel.
This is a discrepancy worth putting to the owner with the citation attached, not
a legal conclusion.

---

## 14. Top and Tail is called **Bookend** internationally, and the international schema is LESS expressive than ours

This is the ruling's target case, and it lands on the "confirmation" side rather
than the "we were missing it" side.

**Israeli evidence.** `docs/media-domain-from-the-trade.md:80-98` — "A campaign
can hold both the **Top and the Tail** of the same break: the first spot and the
last spot", and separately the paired-creative sense (a 10-second spot plus a
6-second closer, one or two other advertisements between them).

**Foreign confirmation (MEASURED).** The TVB **TIP** OpenAPI schemas
(`github.com/tip-initiative/tip-initiative-apis`, branch `develop`) define a
`LinkType` object whose enum is
`Billboard, Piggyback, Bookend, Sandwich, Donut, Sponsorship, Package`,
described as "Indicates the link constraint between two or more units (spots)",
carrying `linkNum` ("Unique number to communicate the association of two or more
units within a link type") and `linkSeq` ("Airing sequential order for the units
linked together such as A or B"). Independently, the Next TV traffic glossary:
"**BOOKENDS (also known as TOPS 'N' TAILS): The very first and last avail in a
pod, requested by an advertiser.**"

**What this is worth, and what it is not.** The trade's term is a real named
international mechanism, not local slang — and `linkNum`/`linkSeq` is a proven
minimal encoding (one association id plus a sequence letter) if a link ever has
to cross a system boundary. **The product must keep saying Top and Tail.** The
ruling forbids renaming to the international term, and the transcript outranks
the schema.

Our own implementation is already ahead: `kairos/optimize/positions.py:364-367`
handles the double counting, `kairos/optimize/_pair_placement.py` enforces the
paired-creative separation, and `docs/top-and-tail-design.md` names the two
senses apart. TIP has one enum value where we have two distinct constraints.

---

## 15. Make good has a SECOND cause, and it is the one Israel's own destination creates

**Israeli evidence, and both halves are Israeli.** The trade document defines
make good as covering "a spot that **did not air or aired wrong**"
(`docs/media-domain-from-the-trade.md:126-128`) — a delivery-failure cause, and
the only one it names. The same document then says the destination is
goal-based orders, "a GRP or target-audience goal instead of a spot list, and
**the channel is accountable for delivering it**", and calls that "**the
product's real thesis**" (lines 169-179).

Those two sentences, both Israeli, create a second cause between them: **the
moment the channel accepts a TRP goal instead of a spot list, it owns audience
shortfall** — and a shortfall is a make-good cause that has nothing to do with a
spot airing wrong.

**Foreign confirmation that this is a normal distinction to draw (MEASURED).**
TIP's `makegoodsSchemas.yaml` has `makegoodType` as an enum with exactly two
values: `Resolve preemption` and **`Audience Underdelivery`**. It also carries
`makegoodRatio` ("Maximum number of makegood spots allowed to resolve
preempt/missed unit") and `makegoodWindow`
(`Original Broadcast Week` / `Original Broadcast Month` / `Within Flight Dates`).

**Whether our product has it.** `data/make_goods.csv:1` carries `kind`,
`deficit_kind`, `unit`, `goal_value`, `counted_value` and a state machine — so
the store has room for a cause. What it does not have, per the re-audit, is the
agency level or an accrual balance
(`docs/audits/trade-reaudit-2026-08-09.md:431-437`).

**Raise against the trade document; do not build from the schema.** The right
next step is one question to the media professional: when the channel takes a
TRP goal and misses it, is the remedy a make good in the same ledger, or a
different thing with a different Hebrew name? I have no Israeli evidence either
way and did not guess.

---

## 16. The annual gantt has a name — the **format**, or the **clock** — and mature systems version it

**Israeli evidence.** `docs/media-domain-from-the-trade.md:31-34`: "The break
schedule itself is laid out roughly at the **end of the previous year** as a
schematic gantt of programmes and break counts. Launches move, programmes get
swapped, and rival channels' moves change it during the year."

**Foreign confirmation (MEASURED).** WideOrbit's public help index
(`help.wideorbit.com/7.1.0/833100.htm`) carries a whole family of screens:
`Format Templates`, `Format Grid`, `Format Schedules`, `Format Instances`,
`Format Codes`, `Format Filler Changes`, `Format Spot Changes` — "The Format
Grid displays the formats scheduled during a given week". SDS's traffic page
says it plainly: "Break formats or **'clocks'**."

**Why it clears the bar.** Not the vocabulary — under the ruling a foreign name
is not a licence to rename anything, and I did not find the Hebrew word for this
object. What clears the bar is the SHAPE: the trade document already says the
annual layout is a living thing that moves during the year, and the mature
foreign systems model that as a **template with dated instances**, plus a
separate record of what changed on an instance. Our product has no object for
the annual layout at all — `data/breaks.csv` is a flat break list with `actor`
and `saved_at`, which is a change log for individual breaks and not a versioned
template.

**Open, and I am not resolving it:** what the Israeli trade calls this object.
The transcript describes it and does not name it.

---

## 17. An Israeli broadcaster does outsource ad sales to a commission-based sales house — but the evidence is Kan only

**Israeli evidence (MEASURED, three Israeli outlets).** Kan
(תאגיד השידור הישראלי) tenders its advertising sales to an external agent.
Globes reports **Target Spirit (טרגט ספיריט)** won it, bidding a commission of
about **16% on sales of 60-120M ILS**, against rivals bidding over 20%; the
mandate covers `פרסום ברדיו, חסויות בטלוויזיה ותוכן שיווקי` — radio
advertising, **television sponsorships**, and marketing content. Calcalist and
ICE cover the later re-tender (around 100M ILS a year, multi-year with
extensions).
`globes.co.il/news/article.aspx?did=1001162464` ·
`calcalist.co.il/local_news/article/b167kn2uk` ·
`ice.co.il/advertising-marketing/news/article/843675`

**The honest limits, and they are the important part.** This is **Kan**, the
public broadcaster, and its television mandate is **sponsorships**, not ordinary
spots — consistent with finding 10, where Kan's commercial rows in our own data
are government and public bodies. A Hebrew search for a commercial-channel
equivalent (a `בית מכירות` or exclusive sales agency for Keshet or Reshet)
**found nothing either way**.

**So this is a question for the owner, not a finding to act on:** does the
channel this product serves sell its own airtime, or through an agent on
commission? It changes who the operator is. `data/agencies.csv:2` carries
`commission_percent=15.0` on an agency record, which is the buy-side commission;
a sell-side sales-house commission would be a different number on a different
object, and there is nowhere to put it.

---

## 18. "Owner" could not be confirmed to exist publicly, and a different named Israeli traffic vendor was found instead

**The negative first, because it is the honest headline.** **NOT CONFIRMED:**
no evidence, anywhere on the open web, in Hebrew or English, that an Israeli
broadcast traffic system called "Owner" / `עונר` exists. Two independent sweeps
reached the same nil result.

Searched: `עונר` and `אונר` against `מערכת שידורים`, `פרסומות`, `לוח שידורים`,
`שיבוץ`, `זכיינית`; `"מערכת עונר"`; `"תוכנת עונר"`; `Owner` with broadcast
traffic and Israel; the Hebrew trade vocabulary `מחלקת תנועה`, `רכז/ת תנועה`,
`מערכת שיבוץ תשדירים`, `מערכת ניהול שידורים`; the Israeli job boards drushim,
alljobs, taasiya and Indeed IL, plus the channels' own careers pages; and the
Second Authority and Kan tender pages. **Israeli job advertisements named zero
systems**, which was the opposite of what I expected and is worth recording so
nobody repeats that line of search. Kan's tender archive returned HTTP 403 and
is unread; a public broadcaster's procurement record is the most likely place a
system name would surface.

**This is a legitimate finding and it is exactly what the owner said would
happen.** The trade document stands unchallenged:
`docs/media-domain-from-the-trade.md:155-158` — "Owner (`עונר`) is the incumbent
Israeli traffic system each channel runs. Closed, no public API." A system with
no public documentation is precisely a system you would fail to find. **The
absence of a web trace is not evidence against the transcript.**

**What did surface (MEASURED).** **KLH Solutions**, an Israeli vendor, Tel Aviv,
5 Ha'Haskala Blvd, founded 2014, 11-50 staff (`klh-tv.com`,
`il.linkedin.com/company/klh`). A Pebble Beach press release on Kan's playout
build, 11 December 2017, names it in the chain: "A **KLH traffic system**
provides the schedule and Marina controls and integrates with Harmonic Video
Servers, VizRT MAM, Oracle (DIVA) Archive, Evertz EQX"
(`tvtechnology.com/the-wire-blog/9357-509357`). KLH describes itself as covering
"traffic and management of scheduling, playlists, Ads, promos, EPG, finance,
regulation reports, content, assets, and analytics", its module list includes
**Sales, Billing & Finance**, and its logo wall shows **Kan**, Altice and
Sport1 (`klh-tv.com/products`).

**Why it matters, stated carefully.** One Israeli vendor spans traffic, ad
sales, billing, EPG and **regulation reports** in a single system, and holds the
public broadcaster. That is simultaneously an integration target and a direct
competitor. **INFERRED and unproven:** "Owner" may be an internal or colloquial
name for a system whose vendor brand is different. **Do not assume KLH is
Owner.** Only the owner or the media professional can settle that, and it is one
question.

**Explicitly NOT confirmed:** what Keshet 12, Reshet 13 or Channel 14 run for
traffic, ad sales, playout or ERP. Nothing, from any source. Also recorded
because it was chased and discarded: a search summary claimed SintecMedia's
OnAir is used at Israeli channels; the underlying pages do not say it. SintecMedia
(Jerusalem, later Operative) and Pilat Media (Bnei Brak, IBMS) are
Israeli-FOUNDED, which is a different claim from Israeli-DEPLOYED, and only the
first is evidenced.

**One more Israeli name worth having (MEASURED).** **Ifat**
(`יפעת בקרת פרסום`) is the Israeli advertising-monitoring incumbent, monitoring
television since 1993, and the Second Authority published a single-supplier
(`ספק יחיד`) contracting notice for Ifat dated 09.01.2025
(`rashut2.org.il/חדשות-ועדכונים/מכרזים-והתקשרויות`). Competitive spend and spot
verification live there. The media guide also names **TV InfoSys** as the system
media buyers query, and `midrug-tv.org.il` states the rating committee
"operates a viewing-measurement system by the People Meter method", listing
Reshet, Keshet, Channel 24, Channel 14, Channel 9, i24NEWS and the Marketing
Union among its members. Kantar as the measurement contractor is **INFERRED**
from search summaries only and is not confirmed here.

---

## 19. Jumbo Media is confirmed from its own shipped code, and it carries a SEPARATE sponsorship house number

**Israeli evidence, MEASURED from the vendor's own application bundle.**
JumboMedia, `media.jumbomail.me`, a product of the Israeli file-transfer company
JumboMail. The public site is a JavaScript application that a fetcher cannot
read, so the shipped bundle was downloaded and mined directly
(`media.jumbomail.me/static/js/main.8b681ec8.js`, 2.96 MB). Everything below is
from the vendor's own code and copy.

Market position, their words:

> "JumboMedia dominates the Israeli TV advertising market with hundreds of
> companies using our system. It is used by all TV channels and broadcasters, as
> well as most post-production companies and advertisers."

The workflow, their words — the transcript's claim in the vendor's language:

> **Advertising Agency:** "Begin the process by creating ad delivery projects and
> selecting relevant post companies and broadcasters."
> **Post Company:** "Upload the required media files, which go through quality
> check and can then be viewed online and confirmed by the ad agency."
> **Broadcaster:** "Receive the files, view them online, **set a house number**,
> and automatically transfer the files to their servers with a click of a
> button."

**One refinement to the trade document, not a contradiction.** The transcript
says "the agency uploads and selects the channel"
(`docs/media-domain-from-the-trade.md:159-162`). The vendor's own copy has the
agency CREATE THE PROJECT and select the channel, and the **post house** upload
the media. A sharpening, and the transcript still outranks it on anything they
genuinely disagree about.

**The house number, richer than we had recorded (all MEASURED in the bundle):**

- the "Set house number" action is gated to the **Broadcaster** role
  (`accessFor:[Kb.Broadcaster]`) — issued channel-side, exactly as the
  transcript says;
- validation is `CAPITAL_LATIN_AND_NUMBERS`, forced uppercase — consistent with
  every value in our own shipped traffic file (finding 12);
- **there is a second, separate `sponsershipHouseNumber` field, labelled
  "Sponsorship house number", with its own content type**;
- delivery is hard-gated on it: "You can't send to FTP with empty house number",
  with fields `FTPSendAllowed`, `HouseNumber`, `SecondHouseNumberType`;
- a `useFor2Spots` flag exists on a version — one file serving two spots.

**The separate sponsorship house number is the finding.** Our product has one
`house_number` per asset row (`data/campaign_assets.csv`), and finding 12
measured that the prefix letter already separates `C` for commercial from `H`
for sponsorship on the traffic file. Jumbo makes the same split explicit as a
second field. Two independent Israeli systems draw the same line, and our data
model does not.

**Quality control, partly corroborated with one honest gap.** MEASURED in the
bundle: a resolution check ("Required resolution: width - 1920, height - 1080"),
`NotSuitableForBroadcast` with a reason of Technical or Other set by the
broadcaster, `LowResOnly`, `Encoding Error`, and duration fields
`RequiredLength`, `ActualLength`, `ActualFrameCount`. Also a hard-coded
`isChannel5or9` flag — per-channel special handling for Israeli channels 5 and
9. **NOT FOUND: any frame-rate, loudness, LUFS, codec or bitrate check string in
the client bundle.** The trade document says Jumbo does QC "including frame
rate"; `ActualFrameCount` is consistent with that and is not proof of it, and
the check is presumably server-side. **Do not state as measured that Jumbo
checks frame rate.**

**The API, partly confirmed.** MEASURED: a REST host exists,
`api-media.jumbomail.me`, with endpoints of the shape `api/Projects/UserGet`,
`api/Clients/GetUserInfo`, `api/Suppliers/AdminPost`, and admin surfaces for
"House Numbers" and "Delivery Orders". **No public or partner documentation
exists** — `/swagger` returns 404, the root returns 403, nothing is indexed. The
transcript's "it has an API and there is an existing relationship" is credible
and the API is private: **the relationship is the access path, not a published
spec.** Anyone planning that integration should budget for a conversation, not
for reading a reference.

---

## 20. The Israeli media guide: four more trade rules, and one contradiction with the transcript

**Israeli evidence.** The Israeli Marketing Association's media guide,
`ishivuk.co.il/מדריך-מדיה-2/`, published 2014-09-08 and modified 2023-11-01. Its
channel list is stale — it still names Channel 10 and Channel 1 — so it is dated
on FACTS. On MECHANISMS it is the trade's own, and it is the best Israeli trade
document found. Every Hebrew string below was verified character by character
against the raw page.

**Top and Tail has a MONEY rule, and we only had the placement rule.**

> `תשדירי T&T אלו שני תשדירים של אותו קמפיין אשר משובצים באותו ברייק פרסומות. התמחור בעבורם מתבצע לפי אורכו המצטבר של שני התשדירים (כלומר, תשדיר 5″+15″ יתומחר לפי תשדיר אחד של 20″).`

> *T&T spots are two spots of the same campaign placed in the same advertising
> break. **Pricing is by the COMBINED length of the two spots** — a 5″+15″ pair
> is priced as a single 20″ spot.*

This composes with finding 8 and the composition is the point: the pair is
priced as ONE spot of the summed length, which then goes through the
non-proportional sub-30″ factor table. A 5″ and a 15″ priced separately, each
through a table that over-charges short lengths, is not the same number as one
20″. `docs/top-and-tail-design.md` models the adjacency and says nothing about
the price. **We had the constraint and not the money.**

**Gold break, with the owner's spelling confirmed by an independent source.**

> `ברייק זהב: ברייק המורכב מפרסומת אחד ועד 3 פרסומות. עבורו לרוב נגבית תוספת של עד 25% לפרסומת בודדת.`

> *Gold break: a break made up of one to three advertisements. A premium of up
> to 25% is usually charged for a single advertisement in it.*

Two things. The guide spells it `זהב`, independently confirming the owner's
correction recorded at `docs/media-domain-from-the-trade.md:204-210`. And a gold
break is **defined by its size** — one to three spots — where our product treats
gold as a flag on a break (`data/breaks.csv` `is_gold`,
`config/optimization_weights.yaml:58` `max_gold_breaks_per_hour`) with no size
rule at all. Corroborated commercially: Globes reported Reshet marketing ten
gold breaks at ₪160,000 each for the Big Brother premiere with no ordinary
breaks (`globes.co.il/news/article.aspx?did=1001351815`).

**The Israeli daypart names, verbatim:** Morning 6:00-9:00 · Day 9:00-16:00 ·
**OFF** 16:00-19:00 · **Semi Prime** 19:00-20:00 · Prime Time 20:00-23:00 ·
Late 23:00-24:00 · Late Night 24:00+.

Compare our own bands (`kairos/model/audience_frame.py:54`):
`overnight, morning, afternoon, access, prime, late`. The shapes are close and
the words are not the trade's. `OFF` and `Semi Prime` are the trade's names for
what we call `afternoon` and `access`. Under the ruling this is not a rename to
perform on a foreign authority — but this authority is Israeli, and it is worth
one question to the owner whether the operator's screens should say `סמי פריים`.

**`יחס המרה`, the conversion ratio, is a named Israeli metric we do not have.**
The guide defines it as **TRP ÷ GRP**, varying by audience, programme, day,
period and channel. It is precisely the number a goal-based order needs: it
converts a general-audience rating into a target-audience one. The transcript
names the destination as goal-based orders against a named audience
(`docs/media-domain-from-the-trade.md:169-179`), and the re-audit found the
product has exactly one measurable audience, all viewers
(`docs/audits/trade-reaudit-2026-08-09.md:105-115`). **The conversion ratio is
the bridge between the one base we can measure and the bases clients actually
buy**, and it has a Hebrew name and no representation in the product.

**The contradiction, recorded and not resolved.** The guide says:

> `המיקומים המועדפים בברייק הינם: ראשון, שני, שלישי ואחרון, ועבורם נהוג לגבות תוספת תשלום הנעה בין 5% לעד 20%.`

> *The preferred positions in the break are: first, second, third and last, and
> a premium of between 5% and up to 20% is usually charged for them.*

The trade document says **first through fifth plus L**
(`docs/media-domain-from-the-trade.md:73-78`). **The trade document outranks
this** — it is a 2014 guide and the media professional works in the market
today, and the ruling is explicit. Recorded because two Israeli sources
disagreeing is worth knowing, and because the guide's premium BAND, 5% to 20%,
is new information either way and sits below our shipped first-position premium
of 1.30 (`config/optimization_weights.yaml:22-29`).

---

## 21. The `משרד / MB` column names BUYING HOUSES, not advertising agencies — and Israeli published spend figures are gross rate-card

**Israeli evidence, and it is a cross-check between an Israeli source and our
own file.**

The Israeli trade distinguishes two layers that our product collapses into one
word. `משרדי פרסום` are advertising agencies — creative. `חברות רכש מדיה` are
media buying houses — the people who actually buy airtime. The ranked lists
Israeli trade press publishes are of the FIRST layer; the counterparty on the
other side of a channel's airtime negotiation is the SECOND.

**The cross-check.** The nine values in `משרד / MB` on our shipped traffic file
(finding 9) are `יוניברסל` 65, `OMD` 56, `ישירים` 15, `יוניון` 12, `פובליסיס`
12, `מדיהקום` 6, `פיתוח עסקי` 5, `רואים קונים` 2, `לפמ` 2. Four of those —
Universal, Union, Publicis and Mediacom — are among the handful of houses
Israeli reporting names as the buying layer. **The column header even says it:
`משרד / MB`, where `משרד` is the office and `MB` reads as media buying.** So the
incumbent's own field is the buying house, and our loader renames it `agency`
(`kairos/data/loaders.py:85`).

**INFERRED and explicitly not asserted:** that Israeli media buying is
concentrated in roughly five houses, most of them owned by the advertising
agencies, with one reported at around 30% of television advertising. The
underlying Globes articles are paywalled and were not read. The shape of the
claim matters commercially — a channel negotiates with a handful of
counterparties, not dozens — but **no name and no percentage from that
paragraph should be relied on** until a readable primary source exists.

**The second half, and it is the one to remember.** The ranked agency table
Israeli trade press publishes (ice.co.il, 31/12/2023, on Ifat data for 2023) is
priced at `מחירי יפעת הלא מפוקטרים` — **UNFACTORED Ifat prices**, gross
rate-card, before the discounting that actually happens. The same article says
plainly that digital `לא נמדד ביפעת` — is not measured by Ifat at all — and that
the rankings cover commercial agencies only, which is why the government bureau
`לפ"מ` is absent from them while appearing as a buyer in our own file.

**Why this is a finding and not trivia.** It is the same distinction the trade
document draws about orders: "An order quantity is a negotiating position, not a
demand forecast" (`docs/media-domain-from-the-trade.md:24-27`). **Israeli
published advertising numbers are systematically gross rate-card, not net.** Any
benchmark, market-share figure or competitive-spend number that ever enters this
product from Israeli trade press is in a different unit from the product's own
revenue, in exactly the way finding 1 describes for ratings. It is the same
mistake waiting in a different place.

**Whether our product has it.** No. `data/agencies.csv` carries
`rebate_percent` and `commission_percent`, and `kairos/export/agency_layer.py`
applies them — so the gross-versus-net distinction exists in the money model.
Nothing labels an external figure as gross or net, and nothing distinguishes a
creative agency from a buying house.

---

## Open questions where only half the evidence exists

Deliberately not findings. Each needs an Israeli answer before it means
anything, and none of them is a build item.

1. **How are twenty creatives assigned across a campaign's spots?** The trade
   document says a campaign carries "many creatives, up to twenty versions"
   (`media-domain-from-the-trade.md:93-98`) and never names the assignment
   mechanism. TIP names three, mutually exclusive: `Rotation Share` (a
   percentage per creative), `Pattern` ("A,B,B,C represents how the creative A,
   B and C should be sequentially assigned to units"), and `Unit Specific` (an
   explicit list). If Israeli operators do one of these, we cannot currently
   name it. **Ask; do not pick one.**
2. **Is competitive separation a channel policy or a per-deal number?** The
   trade document lists it as a per-run priority (line 46). TIP's
   `TimeSeparation` models it as a per-ORDER negotiated term with a value in
   **seconds** (example `900`) that may vary by `unitLength`. Those are
   different objects and I have no Israeli evidence for which one Israel means.
3. **Is there a name for a spot pulled by telephone on the night?** The trade
   document describes it happening (lines 118-123) and gives it no name.
   Foreign systems code it (`Reason Codes` in WideOrbit, `PreemptDetail` in
   TIP). Our `data/campaign_delivery.csv` `air_state` is
   `aired|scheduled|unknown` with no cancelled or pulled value
   (`docs/audits/trade-reaudit-2026-08-09.md:278-280`).
4. **What does `EB` stand for?** Finding 4 measures its behaviour and does not
   name it. Already the owner's open question 7.
5. **"Overnight plus one".** Israeli sources publish an `OVERNIGHT` definition;
   the "plus one" wording could not be confirmed as published Israeli usage.
   The transcript stands and the ambiguity goes to the owner as one sentence.

---

# PART 2 — FOREIGN, NOT APPLIED

**Nothing in this part is a recommendation. Nothing in the product changes
because of it.** It is recorded so the next person does not research it again,
and so that when the owner or the media professional says a Hebrew word we do
not know, there is a chance the mechanism is already described here under an
English one.

Everything below was tested against `docs/media-domain-from-the-trade.md`, the
shipped traffic file, and this document's own Part 1, item by item. Three
matches were found, all three already correctly named and already built or
already logged as a known gap elsewhere. They are recorded once, at the end of
this part, and not repeated as new Part 1 findings. Nothing else below has
Israeli evidence.

## SMPTE ST 2021, "BXF" (Broadcast Exchange Format)

NOT APPLIED. No Israeli source names BXF, an XML interchange schema, or
anything resembling one. `docs/media-domain-from-the-trade.md:155-158` is
explicit that the incumbent, Owner (`עונר`), is closed with no public API; a
database link for synchronisation "is possible" but nothing in the transcript
or the shipped file suggests XML message exchange of any kind. This is
recorded because BXF's own object model is the clearest published vocabulary
for what a traffic/automation interchange looks like, and the closest
candidate to something Israel does under a different name (Part 1, item 5) was
found by reading the shipped CSV directly, not by matching it to BXF.

**What BXF is** (MEASURED, `pub.smpte.org/doc/st2021-1/20151009-pub/st2021-1-2015.pdf`,
Clause 1 Scope, read directly): "The Broadcast eXchange Format (BXF) defines
the format and content of XML Messages for the interchange of data and
metadata among professional systems," covering (1) broadcast schedules
including playout and record schedules, (2) as-run information, (3) content
metadata, (4) content management requests (dub, purge), (5) content-transfer
requests, (6) TCP/IP ports. The suite: **OV 2021-0** (roadmap), **ST 2021-1**
(master document, requirements), **ST 2021-2:2019** (protocol), **EG 2021-3**
(use cases, title only confirmed via search, not read), **ST 2021-4:2023**
(schema documentation, 33 XSD files, root file `bxfschema.xsd`), **RP 2021-9:2017**
(Implementing BXF, read in full). Root/message elements, read directly from
the published XSDs (`github.com/SMPTE/st2021-4`, `main/schema/bxfschema.xsd`):
`BxfMessage` (root of every message), `BxfData` (payload, choice of `Schedule`,
`ContentTransfer`, `Format`, `Content`, `Configuration`, `TrafficInstructions`,
`QCProfiles`), `BxfQuery`. RP 2021-9's own table of contents names the
sub-messages: Dub Order, Purge Order, Record Order, Transfer Order (all under
Content Transfer), and EPG/Playlist/As-run notifications (under Schedule).

**Why this is not a build recommendation even where it overlaps.** BXF's
As-run object and the trade's own "As Run" (below) name the same idea, but the
Israeli evidence for As Run comes entirely from the transcript and the gap is
already logged (`docs/trade-gap-analysis.md`, item 5, "As Run ingestion, with
delivery and billing computed from it"). Nothing about BXF's specific XML
shape, message-lifecycle protocol (RP 2021-9 clauses 5-10), or transport
layer (ST 2021-2) has any Israeli source. Owner is closed and there is no
stated intention to speak BXF to it.

## AMWA AS-11 (incl. X6, commercial/sponsorship delivery) and NMOS

NOT APPLIED. No Israeli source names AS-11, MXF file-delivery constraints, or
NMOS. `docs/media-domain-from-the-trade.md:159-162` names **Jumbo Media** as
Israel's creative-upload hub, with an API and an existing relationship, and
says Jumbo (not the channel) performs quality control including frame rate.
Nothing in the transcript, the shipped traffic file, or the code names a file
wrapper format, a delivery profile, or a body resembling the UK's Digital
Production Partnership.

**What AS-11 is** (MEASURED, `amwa.tv`, `aafassociation.org`, DPP site):
constrained MXF file-delivery formats for finished media, developed with UK
broadcasters (BBC, Channel 4, ITV, Sky, BT Sport) for programme delivery
(AS-11 UK DPP HD/SD, live since 1 October 2014), with variants for UHD, HD
Intra/Long-GOP, and SD. **AS-11 X6, specifically, covers Commercials,
Sponsorship, and Infomercials** (MEASURED, `thedpp.com/news/dpp-publishes-version-2.0-of-the-as-11-delivery-documents`:
"AS-11 X6 commercial, sponsorship, and informercial delivery documents," with
guidance document DPP012 and technical spec DPP013). This is the one part of
the whole international haul that maps onto the same JOB Jumbo Media does
(frame-rate QC, format constraint, at the ad-delivery step) — but the job
being the same is not evidence the mechanism is the same, and I found no
Israeli source describing what Jumbo actually checks or what format it
requires beyond "including frame rate."

**NMOS** (Networked Media Open Specifications): control-plane discovery and
registration for SMPTE ST 2110 IP media essence transport. Playout-layer
infrastructure, not a traffic/sales artifact. No Israeli evidence sought
beyond confirming it is out of scope by its own definition.

## EBUCore and egtaMETA

NOT APPLIED. No Israeli source names EBU, egta, or a metadata schema for
advertising-file exchange. Israel is not an EBU or egta member market in
anything I read.

**What they are:** EBUCore (Tech 3293, MEASURED `tech.ebu.ch`) is the EBU's
core metadata vocabulary for audiovisual content. **egtaMETA (Tech 3340)** is
the closer match to this project's domain: a metadata schema specifically for
exchanging advertising/commercial files across Europe, co-developed by egta
(European Association of TV & Radio Sales Houses) and the EBU, based on
EBUCore 1.2, with FTV/RAI/WDR/ABMA from 2010 (this is a WebSearch-summary
read of a scanned conference PDF, not a primary quote — flagged lower
confidence even for the foreign claim itself). If Israel ever needs to
exchange creative metadata with a European sales house or broadcaster, this
is the document to open. It is not that today.

## Ad-ID (US)

NOT APPLIED, and the Israeli evidence points the other way: Israel does not
have this mechanism, it has a different one, and the different one is already
correctly modelled.

**What Ad-ID is** (MEASURED, IETF RFC 8107, `rfc-editor.org/rfc/rfc8107.html`):
a US **national, universal** spot identifier, 11 characters (12 with an
HD/3D suffix) — a 4-character advertiser prefix plus a 7-character code —
administered by Advertising Digital Identification, LLC (jointly tied to the
4A's/ANA). The same code follows a spot across every station and every
platform in the US.

**What Israel has instead** (Part 1, item 11, and
`docs/media-domain-from-the-trade.md:163-167`, both already in this
document): a **House Number**, issued per channel by Owner, not universal —
"the same creative has a different house number per channel." The shipped
traffic file's own header carries the English string `House Number` verbatim
inside an otherwise all-Hebrew row (`.playwright-mcp/Wally_2026-08-05.csv`,
column 9). This is not Israel missing a name for Ad-ID. It is Israel having a
structurally different, already-named, already-built mechanism
(`kairos_api/campaigns_assets.py`, field `house_number`) that happens to use
an English label. Do not read this as "Israel needs Ad-ID." Part 1, item 11
already flags the one real open question here — what the second and third
letters of the house-number prefix encode — and that question has nothing to
do with Ad-ID.

## Clearcast, the BACC, and the UK clock number

NOT APPLIED. No Israeli source names a pre-clearance body, a clock number, or
anything with that shape. The Second Authority for Television and Radio
(רשות השנייה) governs advertising *content/ethics rules* under a 1990 Knesset
law (MEASURED, Wikipedia, cross-checked against no Israeli primary source
read directly) but nothing found describes it issuing per-spot identifiers or
running a submission-and-clearance workflow comparable to Clearcast's. The
product already carries `clearance_verdict` / `clearance_authority` /
`clearance_checked_at` fields (`kairos_api/campaigns_assets.py:68-70`), and on
every shipped record the verdict is `unknown` and the authority is blank —
consistent with there being no connected clearance system to name, not with a
name being missing.

**What Clearcast is** (MEASURED, Wikipedia + `help.clearcast.co.uk`): the UK's
non-governmental TV-ad pre-clearance body since 1 January 2008, successor to
the Broadcast Advertising Clearance Centre (BACC, an ITV department funded by
all UK commercial channels), now owned by ITV/Channel 4/Sky/Warner Bros.
Discovery. Every cleared ad gets a **clock number**, format `AAA/BBBB123/456`:
3-letter Clearcast-issued agency code, 4 letters + 3 digits chosen by the
agency, then the spot's duration in seconds. Ireland was checked as a
tangent (not Israel, dropped from scope by the ruling, kept here only because
it surfaced): RTÉ runs its own clearance process (I downloaded and read
`about.rte.ie/.../RTE-Commercial-Clearance-Process.pdf` directly — script
pre-clearance, committee approval, post-production sign-off) but that
document itself never uses the term "clock number"; the term appears only in
a third-party (Cape Advanced TV / Peach) guide. Not carried further because
Ireland is not the market this ruling is about.

## 4A's/ANA eBusiness, ANSI X12, "SBMS", "TIP"

NOT APPLIED. All US-specific trade-EDI initiatives; no Israeli source names
any of them, an EDI transaction set, or a purchase-order/invoice standard for
spot buying.

- **"eBiz for Media"** (MEASURED via MediaPost-article summaries): a 4A's-era
  XML/EDI hub, buy-side/sell-side, "avail to invoice." Likely defunct — no
  2024-2026 aaaa.org content mentions it.
- **ANSI X12 810 (Invoice) / 850 (Purchase Order)**: generic US EDI
  transaction sets used across many industries. I could **not confirm** they
  are specifically the sets used for TV spot trading (the one page that would
  have confirmed it, Mediaocean's Prisma support article on cable EDI,
  returned HTTP 403).
  Treat any claim that "810/850 are what TV spot buying runs on" as
  unconfirmed.
- **"SBMS"**: a real, live product name inside FreeWheel's (Comcast/Strata)
  agency platform ("SBMS for Spot"). What the letters stand for is **not
  confirmed** — FreeWheel's own page returned HTTP 403. Not "Spot Buying Made
  Simple"; that was an unverified guess and should not be repeated as fact.
- **"TIP"** is real and confirmed, but is **not** "Television Invoice
  Processing." MEASURED (`businesswire.com`, `tvb.org`): TIP = **"TV Interface
  Practices,"** a 740+-station US consortium building open interfaces for
  local-TV ad transactions (Logtimes, Inventory Avails, RFP, Proposal,
  Commercial Instructions, Invoices), with an API since 2019.

## Where the foreign vocabulary and the Israeli evidence actually meet

Three points of contact, checked against Part 1 and against
`docs/media-domain-from-the-trade.md`, none of them new work:

1. **As Run.** The trade transcript uses this exact English term, unprompted,
   in an otherwise Hebrew-first conversation
   (`docs/media-domain-from-the-trade.md:115-123`): "`As Run` is a JSON file
   from the broadcast system, second by second, produced after the fact...
   Billing and delivery must be computed from As Run, never from the plan."
   This is the same object BXF's `asrun.xsd` and RP 2021-9 §13.5 name. The gap
   was already logged before this research (`docs/trade-gap-analysis.md`,
   item 5) and this adds no new build target, only a note that the
   vocabulary the trade already uses and the vocabulary the international
   industry standardised on are, for once, the same word.
2. **Make good.** The transcript uses this English term too
   (`docs/media-domain-from-the-trade.md:125-134`) for the same three-level
   accrual-and-utilisation concept the US trade calls a makegood, and the
   product already has a store for it (`kairos_api/makegood_store.py`).
   Already logged (`docs/trade-gap-analysis.md`, item 4). No new work.
3. **House Number and CPP.** Both covered above and in Part 1 items 5, 12 and
   19. Both already correctly named and already built. BXF's own
   `bxfcontentid.xsd` models `BxfContentId` as a choice of `Isan` /
   **`HouseNumber`** / `AlternateId`, so a channel-side house number is a PEER
   of a global registry identifier in the international standard, not a
   degraded substitute for one. The transcript's "the same creative has a
   different house number per channel" is orthodox. Use the term with
   confidence; that is the whole of the finding and there is no work in it.

No other term or mechanism from BXF, AMWA, EBU/egta, DPP, Ad-ID or the US EDI
initiatives found any Israeli evidence.

**Correction to an earlier draft of this line.** An earlier version of this
section ended "this research produced zero new Part-1 findings", written before
the TVB TIP schemas and the Israeli regulatory and trade sources were read. That
is no longer true: the foreign reading produced Part 1 items 14, 15 and 16, and
one foreign document (the DPP/ITV sponsorship-delivery spec) independently
corroborates the sponsorship-identifier split measured in JumboMedia's own code
at item 19. The line is left visible rather than deleted, because a research
document that quietly upgrades its own conclusions is not one anybody should
trust.

## The vendor map, by layer

NOT APPLIED. Recorded so nobody researches it twice. **No Israeli deployment
evidence was found for any vendor in this table** except where Part 1 item 18
says otherwise.

| Vendor | Products | Layer |
| --- | --- | --- |
| **WideOrbit** (Lumine Group, acq. Feb 2023) | WO Traffic, WO Media Sales, WO Network, WO Omni, WO Fusion, WO Digital Hub, WO Payments | Sales → traffic → billing → reconciliation, US local stations |
| **Imagine Communications** (Harris heritage) | Landmark Sales, OSI Traffic & Billing, xG Linear, BCM, CrossFlight, GamePlan (yield), SureFire | Sales/traffic/billing; ADC/Versio/Nexio are **playout, not traffic** |
| **Operative** (ex-SintecMedia; absorbed Pilat Media 2014, Broadway 2015, rebranded 2018) | AOS, Operative.One, OnAir, IBMS, Broadway, SIMS, MediaPro, STAQ | Full stack, cable networks and international |
| **Marketron** | Marketron Traffic, Visual Traffic, NXT, RadioTraffic, PayNow | Radio-led sales, traffic, billing |
| **BroadView Software** | Sales, Traffic, Programming, OnDemand | Rights + programming + traffic |
| **Myers ProTrack** | ProTrack TV / Radio | US public television |
| **Provys** (DCIT) | PROVYS Sphere, incl. Ads Planning | European broadcast management |
| **Mediagenix WHATS'ON** | planning, rights, linear + FAST | **Rights/programming, explicitly not ad sales** |
| **Etere** | Airsales, Airsales Scheduling, Promo Placement | Traffic + accounting; RAI a named customer |
| **FreeWheel** (Comcast) | MRM, Streaming Hub; **Strata is buy-side**; Beeswax is a DSP | Streaming monetisation |
| **Mediaocean** | Prisma, Spectra, Lumina, Ignitia, Radia | **All buy-side agency tools** — talks TO seller systems |
| **Salesforce Media Cloud ASM** | Order, Order Ad Placement, Media Plan, Inventory Slot, Rate Card, Spot Calendar | Sales/CRM + order management |
| **KLH Solutions** | traffic, scheduling, ads, promos, EPG, finance, regulation reports | **Israeli — see Part 1 item 18** |

**Named plainly as NOT traffic vendors**, because each was checked and each is
easy to mistake for one: **Vizrt** (graphics; its own documentation says it
RECEIVES playlists from third-party traffic systems), **Broadpeak** (CDN),
**Veset** (cloud playout), **Amagi** (playout plus dynamic ad insertion; no
avails, rate-card or order-entry product found), **Xytech** (MAM and
transmission logistics), **S4M** (drive-to-store measurement, wrong category
entirely), **Google Ad Manager "for TV"** (dynamic ad insertion for streaming;
no linear traffic module). **"DoubleClick for Broadcast" does not exist** — no
primary source uses that name. Do not cite it.

## TVB TIP, the one openly readable trading schema — and the two places it is WORSE than Israel

NOT APPLIED. **TIP = "TV Interface Practices"**, not "Television Invoice
Processing". A 740-plus-station US consortium building open interfaces for
local-television ad transactions, with an API since 2019
(`businesswire.com`, `tvb.org`). The specs are OpenAPI 3.0, free, on GitHub:
`github.com/tip-initiative/tip-initiative-apis`, default branch `develop`, over
a shared `commonSchemas.yaml` data dictionary.

Its endpoints are the trading lifecycle in one list: `rfps` · `proposal` ·
`orders` · `inventoryAvails` · `commercialInstructions` · `creativeAssets` ·
`makegoods` · `invoice` · `logTimes` · `buyerAudiences` · `sellerAudiences` ·
`sellerPoliticalCompetitives` · `impressionssub`. It is the clearest published
picture of what the whole buy-sell conversation contains, and it is the source
for Part 1 items 14 and 15.

**Two places it is less expressive than the Israeli trade, and this is why the
ruling is right.**

1. `InventoryPosition` is an enum of
   `First, Middle, Last, Pre-Roll, Mid-Roll, Post-Roll`. **It cannot express
   Israel's 1-5 plus L.** Our own `kairos/optimize/positions.py` is strictly
   more expressive than the American standard. Adopting TIP's vocabulary would
   be a downgrade.
2. TIP's makegood is a NEGOTIATION PROTOCOL — seller offers, buyer accepts or
   rejects, keyed to specific pre-empts. **It has no concept of the three-level
   accrual-and-utilisation ledger** the trade document describes at campaign,
   advertiser and agency level (`docs/media-domain-from-the-trade.md:125-134`),
   and **no international equivalent to that ledger was found in any vendor or
   standard**. Israel's make-good design cannot be validated against foreign
   documentation because it appears to be genuinely Israeli. It also cannot be
   dismissed as over-engineering on the grounds that no foreign system has it.

## The unglamorous module list, as one vendor actually names it

NOT APPLIED. From WideOrbit's public help index
(`help.wideorbit.com/7.1.0/833100.htm`), which is readable without a login and
lists roughly 200 screens. Recorded because the question "what does a real
traffic system have a screen for" has a concrete answer, and because most of
these are things a modern optimiser would not think of:

- **Materials and instructions:** `Dub List` · `Purge List` ·
  `Material Instructions` · `Material Placer` · `Material Groups` ·
  `Material Locations` · `Instruction Confirmations` ·
  `Missing Instructions Report` · `Missing Materials Report`
- **Logs and formats:** `Log Grid` · `Log Manager` · `Log Planner` ·
  `Log Timings Report` · `Format Templates` · `Format Grid` ·
  `Format Instances` · `Format Filler Changes` · `Format Spot Changes`
- **Reconciliation:** `Spot Recon Manager` · `Variance Activity Report` ·
  `Variance Log Report` · `Variance Rate Report`
- **Credit and collections:** `Advertiser Credit Report` · `Aging Manager` ·
  `Historical Receivables` · `Finance Charges` · `Lockbox Payments`
- **Inventory:** `Pending Avails Report` · `Avail Requests` ·
  `Inventory Quick View` · `Inventory Adjustments` · `Inventory Trees` ·
  `Pricing Grid` · `Break Summary Report`
- **Fill and compliance:** `Promo/PSA Filler` · `Promo/PSA Log Copy` ·
  `Preferred Program Lists` · `Exclusion Codes` · `Reason Codes` ·
  `Expiration Manager` · `Threshold Verification`
- **Deals and approvals:** `Deal Mgmt` · `Deal Discounts` · `Approval Groups` ·
  `Workflow Configuration` · `AE Entrustments`

For comparison, our own rail is fifteen entries
(`tv-break-dashboard/src/shell/nav.js:19-34`): Overview, Optimizer, Schedule,
Inventory, Break Library, Campaigns, Forecasts, Reports, Data, Advertisers,
Agencies, Overrides, Assistant, Versions, Settings. **None of that is a gap
list.** Most of the WideOrbit screens exist because a US local station bills,
invoices, chases receivables and files co-op claims, and this product explicitly
does not invoice (`kairos_api/campaigns_delivery.py:35` — "Nothing in this
product invoices"). The list is here for recognition, not for adoption.

## Foreign delivery and identifier specs, recorded only

NOT APPLIED, and none of it has Israeli evidence.

- **UK:** AMWA AS-11 / DPP HD Shim MXF, 1080i/25, 4:2:2, EBU R128 **−23.0 LUFS
  ±0.5 LU**, max true peak −1 dBTP, mandatory XML sidecar. Israel's own rule is
  a **dialnorm** target of −28 dB (Part 1 item 13) — a different unit and a
  different number.
- **US:** NBCUniversal, January 2024, **−24 LKFS ±2 dB**, Ad-ID mandatory, FTP
  delivery not accepted, assets purged 90 days after last air.
- **Ownership corrections worth having:** Peach IS the former IMD (Group IMD and
  Honeycomb, rebranded 25 March 2019) — not two vendors. Adstream was acquired
  by Extreme Reach and closed 9 June 2021. Clearcast is a clearance body, **not**
  a delivery pipe. **No Israeli presence was found for Peach, Extreme
  Reach/Adstream, Yangaroo or Comcast AdDelivery.** Israel's hub is JumboMedia
  (Part 1 item 19).

## The RFP hunt, reported as the negative result it was

NOT APPLIED, and recorded because it consumed the largest single search budget
and the hypothesis failed. Public-broadcaster procurement documents were
expected to contain exhaustive traffic-system requirement lists. They do not, or
they are not published.

- **CBC/Radio-Canada** ran the richest known procurement — two lots, "License
  Rights Management & Program Scheduling" and "Advertising Sales Campaign
  Management & Traffic", with a stated "response to over 800 functional and
  technical requirements", won by Arvato Systems in August 2023. **The
  requirements document was never published**; only coverage of the outcome
  exists.
- **TVNZ's "Total TV"** registration of interest is genuinely public
  (`gets.govt.nz`, id 30166358) but is an ROI, not a requirements list.
- **CPB's** entire archived RFP index, 135 titles pulled through the Wayback
  Machine, contains **no** traffic, billing or sales-system procurement. Member
  stations buy those outside CPB.
- **No US public-television station RFP for a traffic system was found at all.**

The one procurement source that would matter is Israeli and is closed: Kan's
tender archive returns HTTP 403.

---

# What we should build that we had not thought of

Every item here has Israeli evidence, and each is written as a finding to put to
the owner or the media professional, not as a licence to start. Ranked by what
it blocks. **Nothing on this list is justified by a foreign document.**

1. **Record the unit on every rating we hold.** Minute or quarter-hour;
   overnight or consolidated; initial or restated. Item 1 measures a 7.15%
   median gap between two units we hold at once and treat as one, and item 7
   shows the universe is unrecorded too. Everything else about money is
   downstream of this.
2. **Make the rating a property of the spot, not the break** (item 2). The
   incumbent already does. Our pod surface takes the first row of the pod and
   calls it the break's rating.
3. **Teach the engine the `99` and `0` sentinels** (item 3). The pod API knows
   them; `kairos/optimize/positions.py` does not, so the preferred-position
   percentage — the number the channel and the agency audit each other with —
   would score every Last as a miss.
4. **Stop enforcing six repealed clauses** (item 13). `max_breaks_per_hour` and
   `min_break_spacing_minutes` are enforced as regulation and the clauses behind
   them read `(בוטל)`. This one costs revenue every run, silently.
5. **Split the ad-minutes budget from the sponsorship budget** (item 13).
   Sponsorship is explicitly not counted in the advertising maximum, has its own
   54-seconds-an-hour ceiling, and can only exist beside a break that was
   scheduled — a coupling between the break plan and sponsorship revenue that
   the product does not represent.
6. **Model the ordered programme and the ordered time** (item 6). Both are
   already in the file we load daily, under `תוכנית מוזמנת` and `שעה`, and both
   are dropped. Everything about honouring an order is a comparison between what
   was asked and what aired.
7. **A length-factor layer below 30 seconds** (item 8), once the owner supplies
   the market table. The engine is linear where the market is not, across 93.9%
   of commercial volume, and linear under-charges.
8. **Price a Top-and-Tail pair on its combined length** (item 20). We have the
   adjacency constraint and not the money rule.
9. **The conversion ratio `יחס המרה`** (item 20), TRP over GRP. It is the bridge
   from the one audience we can measure to the ones clients buy, it has a Hebrew
   name, and it does not exist in the product.
10. **A second make-good cause, audience under-delivery** (item 15), and the
    regulator's 3:1 peak-to-off-peak exchange rate with its prior-written-
    approval gate (item 13).
11. **Add `״` to the copy-length regex** (item 11). One character; 14% of
    length-declaring version names on the shipped file.
12. **Say which break type the file meant** (item 4). `סוג ברייק` names two
    different things on two of our screens.
13. **Label any external Israeli spend figure gross or net** (item 21). The
    published ones are `לא מפוקטר`, unfactored rate-card. This is finding 1's
    mistake — two units treated as one — waiting in a second place, and it costs
    nothing to prevent before the first benchmark arrives.

Two of these are one-character or one-field changes (11, 12). Four are
questions to the owner before anything is built (7, 9, 10, and the `EB` label).
The first six are where the money is.

# What the international literature has that Israel does not, so we should NOT build it

This list is as valuable as the one above, and under the ruling it is binding.

1. **Ad-ID, ISCI, clock numbers, and any national creative registry.** Israel
   issues a house number **per channel**, from the channel. Item 12 measures the
   prefix structure; item 19 finds Jumbo gating delivery on it. No national
   registry exists here and none was found. Building toward a universal creative
   id would be building the wrong shape.
2. **A clearance body and a submission workflow.** No Israeli equivalent to
   Clearcast was found. `campaigns_assets.py` already carries
   `clearance_verdict` and `clearance_authority`, unknown and blank on every
   record — which is the honest state, not a gap.
3. **BXF, AS-11, EBUCore, egtaMETA, ANSI X12 810/850 and the whole interchange
   layer.** The incumbent is closed with no public API and the trade document
   says a database link is the realistic seam. There is no counterparty to speak
   XML to.
4. **TIP's `InventoryPosition` vocabulary.** `First/Middle/Last/Pre-Roll/
   Mid-Roll/Post-Roll` is strictly less expressive than 1-5 plus L. Adopting it
   would be a downgrade, and renaming to it is forbidden anyway.
5. **The upfront/scatter market, ADUs as a named unit, and C3/C7 commercial
   ratings.** None of these appeared in any Israeli source. Israel's forward
   commitment is the block booking the transcript describes, which is a
   different object.
6. **CALM Act / R128 loudness targets.** Israel's rule is a dialnorm target of
   −28 dB, not −24 LKFS and not −23 LUFS. Validating against the foreign numbers
   would fail correct material.
7. **Rolling-window ad-load enforcement.** §1 of the Israeli rules defines the
   hour as a clock hour. Our clock-hour bucketing is right and should be
   defended against anyone who "improves" it.
8. **Invoicing, receivables, aging, co-op and lockbox.** Half of a US traffic
   system's screens exist for billing, and this product explicitly does not
   invoice. Recognising those screens is not a reason to grow them.
9. **A gambling watershed.** No Israeli clause was found, and the lottery and
   the sports-betting council both advertise on our own data (item 10). Encoding
   a rule we could not find would be inventing regulation.
10. **A Shabbat advertising ban.** Measured absence across every instrument
    read. Shabbat modifies other rules; it does not stop advertising.

---

# What I could not confirm

Stated plainly, because the owner's standing point is that this trade is small
and largely undocumented online, and a clean negative is a result.

1. **"Owner" / `עונר` as a publicly documented system.** Zero hits, two
   independent sweeps, search terms listed at item 18. The transcript stands.
2. **What Keshet 12, Reshet 13 or Channel 14 run** for traffic, sales, playout
   or ERP. Nothing, from any source.
3. **What `EB` stands for.** Item 4 measures its behaviour on ten breaks and
   does not name it.
4. **Whether Jumbo checks frame rate.** `ActualFrameCount` exists in the bundle;
   no frame-rate check string was found. Presumably server-side.
5. **Any public Jumbo API documentation.** The API exists and is private.
6. **What the second and third letters of a house-number prefix encode**
   (item 12). Six pairs on one evening is not evidence.
7. **The Israeli sub-30-second length-factor table** (item 8). The trade guide
   says it exists and does not print it. **Do not invent it.**
8. **Israeli gambling advertising rules** (item 13). Likely in an unread
   Finance Ministry permit.
9. **The pre-2010 text of the repealed placement clauses §§11-15, 17.** The
   1992, 1997 and 2002 official PDFs are scans with no text layer, so I cannot
   say whether a minimum-interval rule ever existed.
10. **`חוק התקשורת (שידורים), התשפ״ו` and the July 2026 High Court interim
    order.** Not read; their effect on the minute caps is unknown.
11. **Whether any COMMERCIAL Israeli channel sells through a sales house**
    (item 17). The Kan evidence is real and covers the public broadcaster only.
12. **Israeli media-agency rankings and television ad-spend figures.** The
    research sweep that owned this did not return. Not covered; do not treat it
    as covered.
13. **Whether `TVR` in `data/Programmes.csv` is an all-viewers figure.** Carried
    forward unresolved from `docs/audits/trade-reaudit-2026-08-09.md:461-465`.
    Item 1 sharpens it: whatever universe it is in, it is a MINUTE figure.
14. **Kantar as the current Israeli measurement contractor.** Named in
    `docs/campaign-rate-card-research.md:11-13` from a 2018 trade article;
    `midrug-tv.org.il` confirms a People Meter panel and its own membership, and
    I did not confirm the contractor from a primary source.

