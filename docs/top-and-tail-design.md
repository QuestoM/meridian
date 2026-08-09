# Top and Tail: paired creatives and the validity window

What this is: the design for a trade constraint the product did not model at all,
what was built at each level, and the measurement on the shipped daily file that
says how much of it is currently being got wrong.

## 1. The trade fact this answers

From `docs/media-domain-from-the-trade.md`, which outranks the code:

> A campaign carries **many creatives**, up to twenty versions. A common structure
> is a 10 second spot plus a 6 second closer, with the constraint that they air in
> the same break separated by **exactly one or two other advertisements**. These
> are hard placement constraints and the optimiser has to honour them.
>
> Each creative also carries a **validity window**: until when it may be
> scheduled. That is a constraint too.

Two constraints, and they are different shapes. One is a relation between two
tapes. The other is a property of one tape. They are modelled in two different
places for that reason, and section 3 says why.

## 2. The naming collision, named before anything else

The trade uses **Top and Tail** for two different things, and this product already
implements one of them.

| The name | What it means | Where it lives |
| --- | --- | --- |
| Top and Tail, positional | One campaign holds both position **1** and position **Last** of one break | `kairos_api/break_api_pod_spots.py::top_and_tail`, already shipped, on the pod payload at `positions.top_and_tail` |
| Top and Tail, paired creative | A **lead** creative and its short **closer**, both of one campaign, in one break, with one or two other advertisements between them | new, `pair_separation`, on the pod payload at `creative_pairs` |

A prior recon note recorded that a repo-wide search for `top.*tail` returns no
matches. **That is wrong.** `top_and_tail` has existed in
`kairos_api/break_api_pod_spots.py` since the pod surface shipped, and it means
the positional sense. The two are not the same constraint and one does not imply
the other: a campaign can hold 1 and Last with twenty spots between them, which
satisfies the positional reading and breaks the paired one.

Nothing was renamed. The new thing is called `pair_separation` in the constraint
vocabulary and `creative_pairs` on the payload, so a reader is never asked to work
out which Top and Tail a field means.

## 3. The data model, and why each half sits where it does

### The pair is a constraint row, not a column on the asset

`data/frequency_rules.csv`, limit type `pair_separation`.

Three reasons, in order of weight.

**A pair is an agreement, and the asset ledger is an observation ledger.**
`data/campaign_assets.csv` is rebuilt from the traffic log by
`scripts/seed_campaigns.py`, and every row records `identity_source` to say that
its identity came from a broadcast that happened. A pairing did not happen; it was
agreed. Written as a column, it would be erased the next time that ledger is
rebuilt from the log, and nothing would say it had been.

**A pair is a relation with its own parameters.** It has two subjects, a
direction (the lead leads), and a range (how many others may stand between). A
column would have to be written on both rows, and could then disagree with
itself. There is no honest answer to "the lead says one to two, the closer says
one to three".

**Placement constraints already have a home.** `_frequency_rules.py` is the
authored vocabulary the enforcement reads. Putting a placement constraint anywhere
else means placement constraints live in two places, and the day they disagree the
operator has two answers to one question.

### The validity window is a column on the asset

`data/campaign_assets.csv`, columns `valid_from`, `valid_until`, `validity_source`.

One subject, one value, no second party, no parameters. That is a field. It is
authored rather than observed, which is why it carries its own `validity_source`
and why it reads as **unknown** on every row shipped today: the traffic log records
what aired, not until when a tape may air, so no window can be read from it. An
unknown window is not an open one. `schedulable_on` therefore answers `unknown`
rather than `within` for every creative on disk, and says so with the path that
would resolve it.

### What identifies a creative, and what it costs

The pair rule names **house numbers**. That is the broadcast house's own filing
identity and the traffic log carries it on every row. A version name is **not** an
identity: on the shipped file the single house number `HGB007510` appears under two
different version names (`חסות כללי א'` and `חסות כללי ב'`).

`_occurrences` falls back to the version name only when **no** spot in scope
carries a house number at all, and every verdict records `matched_on` so a reader
never has to assume the stronger identity was available.

**The gap this leaves.** A house number is per channel. The same creative filed on
two channels carries two house numbers and nothing binds them, so a pair authored
for one channel does not bind the same creative elsewhere. Today the operator owns
exactly one channel from settings, so this costs nothing. It would cost something
the day the product held two. A cross-channel `creative_id` is named in section 8.

## 4. The constraint vocabulary

`kairos/optimize/_frequency_rules.py` gains one limit type beside the five that
were there, plus three columns.

```
PAIR_SEPARATION = "pair_separation"
```

| Column | Meaning |
| --- | --- |
| `pair_lead` | house number of the creative that leads |
| `pair_closer` | house number of the creative that closes |
| `value` | fewest other advertisements allowed between them |
| `value_max` | most other advertisements allowed between them (blank reads as `value`, so "exactly two" is one figure) |
| `unit` | forced to `between` |

The trade's own structure is authored as `value=1, value_max=2`.

**Why it sits beside the others rather than beneath them.** `MAX_CONSECUTIVE` and
`MIN_SEPARATION` are the right primitive shape and the wrong subject: both are
caps on ONE target, resolved by specificity through `resolve_effective`, and both
operate on an advertiser or a campaign, never on a named creative pair. A pair is
a **relation**, and the file already had one of those:
`COMPETITIVE_SEPARATION`, which is resolved per authored group by
`competitive_groups` and never by specificity, because asking which of two
relations is "more specific about one spot" is not a question. `pair_rules` is the
exact sibling of `competitive_groups`, and reads the same way.

**Its own unit.** The trade states the gap as "one or two other advertisements",
which is a count of intervening spots, not a distance between positions and not a
number of minutes. `BETWEEN` is separate from `_UNITS` precisely so a
`min_separation` row can never be authored in it by accident.

**Every refusal is a case where honouring the row would mean guessing.** A pair
with one creative named is not a pair. A pair naming one creative twice would be
satisfied by any single spot. A maximum below the minimum describes an empty range
nothing can satisfy, and a rule nothing can satisfy would mark every real break as
broken. A pair with no campaign named would let two house numbers that collide
across advertisers bind. Each is skipped with a stated reason, in the file's
existing `skipped` channel.

## 5. The enforcement

`kairos/optimize/_pair_placement.py`, reached through `kairos/optimize/frequency.py`.

**A pair judges; it does not drop.** Every other rule in `frequency.py` answers by
removing a spot. A pair cannot: dropping the closer of a broken pair leaves the
campaign with a lead and no closer, which is worse than the fault, and dropping
both throws away inventory the advertiser has already bought. So `pair_verdicts`
returns a verdict on an ordering, `enforce_spots` carries it on
`EnforcementResult.pairs`, and `test_a_pair_never_removes_a_spot` holds that line.

**The ordering is the input.** `others_between` is counted over the order the
caller passes, within one `break_id`. This is deliberate. The `position_in_break`
column of a traffic file is the position a campaign **contracted** for (1 to 5,
plus 99 for Last and 0 for unrequested); it is not the order the spots air in.
Counting other advertisements from it would count a contract rather than a
broadcast. Callers pass air order: `spot_views` sorts by break then spot time, and
`pod_spot_views` passes the pod's currently shown order, so an operator reordering
that breaks a pair is caught by the same check that caught the file breaking it.

**Three states, never two.**

| State | When | Why not the other answer |
| --- | --- | --- |
| `satisfied` | both present, same break, `value <= others <= value_max` | |
| `violated` | both present and placed wrong: wrong count, or never in the same break | |
| `unknown` | a creative is not in this traffic file at all, or no spot in scope carries a house number to identify it by | not a violation, because nothing was placed wrongly; not a pass, because nothing was checked |

A creative that airs twice in one break, which the shipped file really does, is
judged on its **best** occurrence. The trade states the constraint as something the
campaign must get, not something every spot must independently satisfy.

Both reasons are written in English and in Hebrew. The Hebrew nouns are the ones
already on shipped surfaces (`תשדיר` from `campaigns_read_money_reasons.py`,
`ברייק` from `break_api_pod_math.py` and the product vocabulary note that forbids
`הפסקות` for this object), with the same bidi isolates that module uses so a Latin
house number cannot reorder the sentence around it. Singular and plural are chosen
rather than left, so no reader sees `1 other advertisements` or `⁦1⁩ תשדירים`.

## 6. The verification surface

The pod already runs a verification list and already names a copy-length
disagreement, a missing length and a position the order does not honour. A broken
pair is exactly that kind of thing, so it joins that list rather than opening a
second one.

- `kairos_api/campaigns_assets_constraints.pod_pair_block` returns the verdicts,
  the three state counts, how many pairs are authored anywhere, and the
  verification entries.
- `break_api_pod.build_pod` appends those entries to `verification.errors` and
  carries the block whole at `creative_pairs`.
- **Only a violation becomes an error.** An unknown pair is not a red mark against
  a pod: a creative that is simply not in this file is not a fault this pod
  carries. The unknown verdicts still travel, counted, so they are visible rather
  than quietly dropped.
- `authored` is on the block because a pod showing no verdict has two possible
  causes, no pair authored anywhere and no pair touching this break, and a surface
  that could not tell them apart would read the first as the second.

## 7. The measurement, on `data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv`

This is the number the whole piece exists to produce. It is not a script: it is
`campaigns_assets_constraints.measure_file`, shipped and asserted in
`tests/test_top_and_tail.py`, so a change that moves it fails rather than quietly
restating it.

**How a candidate is found.** Two distinct creatives of one campaign, where the
lead runs at least 10 s and the closer runs at most 10 s and is shorter. Both
figures are the trade note's own. Each closer is offered exactly one lead, the one
sharing the most words of its version name, because a campaign running four
creatives would otherwise produce a candidate for every combination and none of
them would be the pair anybody bought. A word is three or more characters and not
a bare number, because a bare number is a length or a year.

**By duration and version name** (the reading the brief asks for):

| | |
| --- | --- |
| campaigns in the file | 51 |
| campaigns holding a candidate pair | **7** |
| candidate pairs | **9** |
| of those, airing with exactly one or two others between them | **1** |
| violated | 8 |
| unknown | 0 |

**By duration alone**, with no version-name evidence, as the looser upper bound:
17 candidates across 14 campaigns, 2 satisfied, 15 violated.

**The eight failures decompose into two kinds.** Five air in the same break at the
wrong distance, and three never share a break at all despite both airing that day.

| campaign | lead | closer | what happened |
| --- | --- | --- | --- |
| כלמוביל - מרצדס | CID179035 (28 s) | HID179039 (6 s) | same break at 20:40:09, **11** others between |
| ישרוטל עצמאות | CID178988 (34 s) | HID178989 (6 s) | same break at 22:03:06, 8 others between |
| לבנת פורן | CGB007453 (25 s) | HGB007510 (6 s) | same break at 22:59:40, 16 others between |
| פריסבי דאצ׳יה | CID178966 (45 s) | HID178934 (6 s) | same break at 22:33:15, 28 others between |
| אובלי | CID170632 (11 s) | CID170634 (10 s) | same break at 22:59:40, 3 others between, one over |
| כלמוביל - יונדאי | CID178967 (10 s) | HID178969 (6 s) | both air, never in the same break |
| כלמוביל - יונדאי | CID178968 (10 s) | HID178970 (6 s) | both air, never in the same break |
| קופ״ח מאוחדת | CID178977 (35 s) | HID178966 (6 s) | both air, never in the same break |

The one that airs correctly is אובלי, `CID170632` (11 s) with `HID165807` (6 s), in
the break at 22:59:40 with two others between them.

**The structural reason, which is the finding under the number.** Every closer in
this file is a `חסות` sponsorship billboard, and every one of them carries
position code `0`, unrequested. They air at the head and the tail of the pod while
their lead sits in the middle of it. The traffic file has no way to say "this
6 second tape belongs to that 28 second tape", so nothing in the ordering keeps
them together, and on this day nothing did. **The nearest miss is eleven
advertisements wide.**

That is not a claim that eight agreements were breached. No pair is authored in
`data/frequency_rules.csv`, because a pair is a commercial agreement this product
has not been told about, and inventing one would be inventing a constraint an
advertiser never bought. What the measurement says is narrower and sharper: **if
these nine campaigns bought the structure their creatives look like, eight of them
did not get it, and the product had no way to notice.**

`candidate_pairs` exists so an operator authors from evidence rather than from
memory. The Mercedes row would read:

```
MERCEDES_TOP_AND_TAIL,pair_separation,campaign,כלמוביל,2025-04 - כלמוביל - מרצדס — מרצדס STAR LEASE,,CID179035,HID179039,,,1,2,between,True,
```

## 8. What the trade fact implies that the product still cannot express

Found while building this, and outside the paths this piece owns.

1. **No cross-channel `creative_id`.** The same tape carries a different house
   number per channel and nothing binds them, so a pair is per channel. Harmless
   while the operator owns one channel; a real gap the day it owns two. The fix is
   an identity column on the asset, resolved by the seeder.
2. **The export path does not carry the house number.**
   `kairos/export/spots.py` builds `PricedSpot.ad` from the `creative` column (the
   version name) and never reads `house_number`. So on the daily export path a
   pair falls back to version-name matching, which the shipped file already proves
   unsafe (`HGB007510` under two names). `SpotView.house_number` exists and defaults
   to blank; threading `house_number` through `PricedSpot` and into the `SpotView`
   built in `_apply_frequency` is a two-line change in a file this piece does not own.
3. **The optimiser cannot honour the pair, only report it.** The trade says the
   optimiser has to honour it. It cannot: the weekly optimiser decides break
   COUNTS and has no advertiser attribution, and the daily path prices an order it
   is given rather than choosing one. Honouring the constraint needs an ordering
   step that does not exist. `pair_verdicts` is the measurement that step would be
   scored against, and it is deliberately shaped as a verdict on a proposed
   ordering so it can be that scoring function unchanged.
4. **A campaign carries up to twenty versions; nothing caps or counts that.**
   `campaign_assets.csv` has no version count and no relation between versions
   other than sharing a `campaign_id`.
5. **A rival channel's name reaches an operator surface through a creative
   version name.** On the shipped file, `HRP005600` is filed as
   `חסות מחליפה קשת רשת`. The pod renders `creative` verbatim, so that string is on
   screen today. It arrives in free text from the traffic file rather than from a
   channel column, so the structural boundary holds, but the string is there. This
   piece emits no channel at all, and `test_the_candidates_name_only_the_operators_own_channel`
   asserts the shape of what it emits. The pre-existing rendering is not in these
   paths and is reported rather than changed.
6. **`explain_drop` has no sentence for `pair_separation`.**
   `campaigns_read_money_reasons.LIMIT_SENTENCES` covers the five limit types that
   drop a spot. A pair never drops one, so it never reaches that function, and
   adding a sentence that cannot fire would be dead code. If a later change ever
   makes a pair drop, that table needs an entry, and the function's honest
   "limit is unknown" branch is what would run until it got one.
7. **The validity window is unenforced.** `schedulable_on` answers, and nothing
   calls it on a scheduling path, because no window has ever been recorded. Wiring
   it needs the window to exist first, which needs an editor, which is a surface
   this piece does not own.
8. **No `.jsx` consumes any of this.** `GET /campaigns/{id}/assets` had no
   consumer before this piece and still has none; `creative_pairs` on the pod
   payload has none either. Everything built here is reachable by API and by test
   and is not yet on a screen.

## 9. What was checked

- `tests/test_top_and_tail.py`: 24 pass.
- The rule proved load bearing by removal: with `PAIR_SEPARATION` taken out of the
  limit-type vocabulary and `pair_verdicts` returning early, **16 of the 24 fail
  and 8 pass**. Restored, 24 pass. The 8 that survive are the validity-window and
  measurement-shape tests, which is the right split: they do not depend on the
  rule existing.
- `tests/ -k "frequency or asset or campaign or pod or p10"`: 199 pass, 5 skipped.
- `tests/ -k "spots or export or optimize or money or pacing"`: 347 pass, 3 skipped.
- `tests/test_plan_artifact_fingerprint.py`: 5 pass before and 5 pass after. The
  weekly break-count plan does not attribute breaks to advertisers, so it holds no
  creative and cannot hold a pair. Verified rather than assumed.
