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

## How to read the labels

- **MEASURED** — I read it, in a file in this repository or at a URL I opened.
  The file and line, or the URL, is given.
- **INFERRED** — I reasoned it from how the trade works. It is not evidence.
- **NOT CONFIRMED** — I looked and could not establish it. What I searched is
  stated. This is a first-class result, because the owner's standing point is
  that this trade is small and largely undocumented online.

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

Two of those columns are the answer to questions this project has open, and one
of them contradicts a shape the product enforces everywhere. They are findings 1
and 2.

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

**What it means commercially.** The trade document says the currency is
"**Jewish households, quarter-hour rating, overnight plus one**"
(`docs/media-domain-from-the-trade.md:105-108`). The incumbent's file is the
proof of the middle third of that sentence at the level money is computed. Our
reference month is in a different unit, and it is not a rounding difference: it
is 7% at the median and over 10% on more than a third of spots.

**Whether Israel does it differently.** This IS the Israeli way. The finding is
not that Israel is unusual; it is that we hold two units and treat them as one.

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

## 2. The incumbent encodes position as `0`, `1..N`, and `99` — and `99` is Last

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

## 3. `סוג ברייק` = `EB`, and on the shipped file every EB break spans two ordered programmes

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

## 4. `סוג תמחור`: the incumbent states the pricing basis per spot, and our engine only half-reads it

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

## 5. `תוכנית מוזמנת` — the ORDERED programme is a per-spot field, and two spots in one break can name different ones

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

## 6. The rating currency has no universe recorded anywhere, and our own two documents disagree about deferred viewing

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

## 7. Israeli commercial lengths are arbitrary whole seconds, and the 30-second spot is not the trading unit

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

**This contradicts our own research document.**
`docs/campaign-rate-card-research.md:143-149` says "The 30-second spot is the
universal trading unit. All CPP figures above refer to 30-second spots" and
"5-10s commercial spots are non-standard and unusual in the spot market". On our
own data the 30-second spot is 3.9% of commercials and 6-second commercials
outnumber it. The transcript wins, the data agrees with the transcript, and the
research document is the one that is wrong.

**Whether our product has it.** Yes, and this is a place where the product is
already right and should be defended.
`config/optimization_weights.yaml:6` prices from
`base_price_per_second_per_tvr_point` — per SECOND, not per 30-second unit.
Anyone who "corrects" that toward a 30-second base, on the authority of a
foreign source or of our own rate-card research note, would be introducing an
error. That is worth writing down precisely because the correction looks like an
improvement.

---

## 8. The agency column already carries the direct-client marker, and a government buyer

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

## 9. On the public broadcaster, the commercial rows are almost all government and public bodies

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
**INFERRED:** this reflects a rule about who may buy on the public broadcaster
rather than a coincidence of one month. I could not confirm the rule itself and
I am not stating it as one.

Note that `לפמ` — the government bureau — appears as a buyer in the Reshet 13
daily file too, so government money is not confined to the public channel.

**Whether our product has it.** No. There is no advertiser-class field and no
per-channel eligibility rule anywhere. `data/kairos_settings.json` holds a
single `operator_channel` (`רשת 13`) and one regulatory profile.

**What it blocks.** Nothing today, because the operator channel is Reshet 13. It
would block the moment the product is pointed at Kan, and it is the kind of rule
that is invisible until it is violated.

---

## 10. A Hebrew typographic mark for "seconds" that the copy-length check does not read

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

## 11. House numbers are prefixed, and the prefix separates commercial from sponsorship

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

## 12. Our regulatory ceiling, our own research, and the observed data are three different numbers

**Israeli evidence, three sources in this repository.**

| Source | Ad minutes per hour |
| --- | --- |
| `data/regulatory_guardrails.json` | `max_ad_minutes_per_hour: 12.0` |
| `data/kairos_settings.json` | `max_ad_minutes_per_hour: 12.0` |
| `docs/campaign-rate-card-research.md:224` | "Maximum ad time: 10 minutes per hour (general rule)", cited to the Second Authority's 1992 regulations at nevo.co.il |

And what the traffic log actually shows. `data/Spots.csv`, 1,652 channel-clock-
hours that carry at least one commercial:

- commercials only: median 3.9 min, p90 9.2, max 22.8; **7.4% of channel-hours
  exceed 10 minutes and 4.3% exceed 12**
- commercials plus sponsorship: median 4.4, p90 10.9, max 25.6; 13.8% over 10
  minutes, 6.8% over 12

**INFERRED:** a per-clock-hour hard cap is not what the market obeys. Our own
research note records the shape that would explain it — "Prime time window
(20:00-24:00): maximum 40 minutes total across the block"
(`campaign-rate-card-research.md:226-227`) — a BLOCK AVERAGE, which permits an
hour over the average as long as the block is not.

**Whether our product has it.** It enforces the per-clock-hour reading.
`kairos/optimize/guardrails.py:105-` buckets by `item.hour`, the clock hour,
sums BREAK duration rather than commercial seconds, and does not separate
`פרסומת` from `חסות`. Three modelling choices, none of them recorded as choices.

The re-audit already found that the values enforced come from
`data/kairos_settings.json` and not from the attested
`data/regulatory_guardrails.json`, and that the declared cutover has no
production caller (`docs/audits/trade-reaudit-2026-08-09.md:359-379`).

**What it blocks.** The product can refuse a schedule the regulator would allow,
and allow one it would refuse, and the operator has no way to see which reading
is in force. This is the finding I am least certain about — I did not read the
1992 regulations myself (see the "could not confirm" list) — and it is stated as
a discrepancy between three numbers, not as a claim about the law.

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
3. **House Number and CPP.** Both covered above and in Part 1, items 4 and
   11. Both already correctly named and already built.

No other term or mechanism from BXF, AMWA, EBU/egta, DPP, Ad-ID, Clearcast, or
the US EDI initiatives found any Israeli evidence. This research produced zero
new Part-1 findings.

