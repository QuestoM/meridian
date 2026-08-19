# Live, recorded, repeat — and the defect the question uncovered

A broadcast is one of three things: **live**, **recorded first run**, or
**repeat**. The engine treats it as at most two, and only by accident. This is
what a nine-way research pass found, what survived independent re-measurement,
and what was done about it.

The headline is not the one the question asked for. The live-versus-recorded gap
cannot be measured from the data on disk. The repeat gap is real and much smaller
than it looks. And the thing worth fixing turned out to be how the audience model
decides which factors to use at all.

## The three states, and why only two are visible

The schedule feed carries both flags, and they are clean:

| | broadcasts | share |
|---|---|---|
| Live (never also a repeat) | 240 | 34.1% |
| Recorded, first run | 183 | **26.0%** |
| Repeat | 281 | 39.9% |
| Live *and* repeat | 0 | — |

Per channel the mix is nothing alike: עכשיו 14 is 57.7% live, כאן 11 33.8%,
קשת 12 20.0%.

**No file that carries a rating carries a liveness label.** `data/Spots.csv`
(50,386 rated spots) and `data/Programmes.csv` (3,562 broadcasts) have no
broadcast-state column, and no live marker in any title — zero occurrences of
שידור חי, בשידור חי, שידור ישיר, ישיר, לייב or LIVE across both files. The one
"live" token in the corpus is `[חי בגימל ש.ח]`, a programme *name* that is itself
a repeat, which is a trap for any regex labeller.

The only Live column in the repository is in the feed, which covers 2026-08-18
to 2026-08-30 — **626 days after the training window**, with no TVR, and without
רשת 13, the operator's own channel and 37.1% of the spots.

Liveness is a property of a series rather than a date, so the label was
transferred by series key across the calendar gap. Three independent attempts:

| Transfer | rows labelled | live | recorded | result |
|---|---|---|---|---|
| canonical series, pure-labelled only | 6,378 (12.7%) | 5.391 | 6.613 | ratio 0.82, **p=0.21** |
| bracket-unwrapped join | 7,315 (18.0%) | 5.637 | 6.707 | **p=0.88** |
| 17 recurring series | 3,206 | 5.750 | 6.737 | pooled p=9e-16 but **the sign reverses inside קשת 12** |

Two of three find nothing; the third reverses within the largest channel, which
is what a mix artefact looks like. Stratified by channel and hour, exactly **one**
cell survives, holding 21 live and 14 recorded spots.

**The data cannot answer this question.** That is the answer, and no amount of
modelling changes it.

## The repeat gap is real, and it is about a fifth of what it looks like

The crude figure reproduces exactly: repeat spots mean TVR **1.934** against
**5.771**, a ratio of **2.98×**. Almost all of it is *when* the repeat airs.

| controls held | surviving ratio |
|---|---|
| none | 2.98× |
| hour of day | 1.93× |
| channel × hour | 1.58× |
| **same series, same hour** | **1.17× median, and only 15 of 25 cells in the expected direction** |
| most saturated identified set | 1.08×, not significant |

The lead's own first measurement — same series, hour *not* matched — gave 2.26×
and was wrong for exactly this reason. Holding the series alone still compares a
21:00 premiere with an 03:00 repeat.

Two facts about *where* it can be seen matter more than the point estimate:

- **In prime there are 165 repeat spots against 8,377 first runs.** Zero repeat
  breaks air at 19:00 or 20:00 in the whole month. Over half the statistical
  weight sits between midnight and 06:00. The prime repeat discount is not merely
  unmeasured, it is **unmeasurable from this data**.
- The effect **changes sign by daypart**: between 06:00 and 09:00 repeats rate
  *higher*.

And the money is not a second witness: `revenue_ils` is the rating multiplied by
a rate card, exact to the agora on 50,221 of 50,386 rows. Quoting the revenue
share of repeats restates the rating measurement; it does not corroborate it.

## The model already prices repeat, under another name

The audience model has no liveness family and no repeat family. It does not need
one, because its **series factor is a repeat detector**. Verified directly on the
shipped artifact:

- 603 series cells: **302** whose key carries the repeat marker, **301** that do
  not, and **zero mixed**.
- Mean TVR 1.934 in the marked cells against 5.771 in the clean ones — the
  repeat gap exactly.

That is not a coincidence. `canonicalize_series` takes a fallback path on 99.4%
of aired-spot titles that leaves `ש.ח` inside the key
(`docs/programme-identity.md` has the full account). So the factor named
"programme identity" is substantially fitting repeat-ness, which is why fixing
the identity measured worse and was reverted.

Elsewhere it reaches nothing. Pricing is blind to it — erasing every repeat
marker from every title leaves all 8,704 segment premiums bit-identical. The
optimizer has no liveness or repeat field. The weekly export carries neither.

## The defect this uncovered, which is the real finding

**Every factor family was scored alone against the pooled base, and prediction
adds the activated families together.**

A family that overlaps one already active can therefore clear the bar on its own
and then make the composed model worse. The two families that ship are
correlated at **r = +0.690** — a series key carrying the repeat marker and a
weekday_slot that is 95% repeat overnight are largely the same fact — and their
summed spread is 1.27× what independence would give.

Measured on five temporal folds:

| family set | held-out log RMSE |
|---|---|
| base only | 0.93535 |
| + weekday_slot | 0.69714 |
| + series | 0.63075 |
| **+ a repeat family** (which scores **+11%** alone) | **0.72039 — 14.21% worse** |

**The fix is in the gate, not the model.** Fitting families sequentially instead
was tried first and measured slightly worse on today's configuration (backtest
log RMSE 0.6834 → 0.6953), so it was dropped — paying something now to protect
against something later is the reasoning this work already refused once. Scoring
each candidate against *base plus what is already on* costs nothing now:

| family | alone | composed | verdict |
|---|---|---|---|
| weekday_slot | +25.6% | +25.6% | on, unchanged |
| series | +16.1% | **+9.6%** | on — the composed figure is the honest one |
| competitor_lineup | +2.2% | **−5.5%** | **off** |
| a repeat family | +11% | −14.2% | would be refused |

So the composed gate also caught a family that **ships as active today and
degrades the model it is added to**. `audience_model_activation` is off by
default and stays off, so no shipped number moves; what changes is what the
artifact would contribute if it were ever switched on.

## One correction to a docstring, which was mine

`keshet_epg.py` stated as fact that the programme classifier understands the
feed's `LiveBroadcast` / `RerunBroadcast` flags, "so a rival's repeat is not
counted as a fresh premiere". That was written from intent and is false.
`ProgramClassifier.classify` takes a title and derives its own marker from it; it
never sees either column, and it misses **111 of the 281** broadcasts the feed
flags as repeats — 39.5% — with **zero** false positives the other way. The feed's
flag is strictly better than the title marker. It is carried, correct, and unread.

## The tagging idea, tested and answered

The obvious next move is to generate the label rather than wait for it: number
every airing of a programme, and everything after the first is a repeat. It was
built and measured. **It does not work, for a reason the data states plainly.**

Airings were reconstructed properly first — spots clustered into broadcasts by a
two-hour gap, 2,419 airings over 47,124 spots, 19.5 spots each — because a spot
is not an airing and grouping by spot time answers a different question. Then:

| | TVR |
|---|---|
| first airing in window | 3.889 |
| later airings | 3.974 |
| **ratio** | **0.98** |

No separation at all. The cross-tab says why:

| | TVR |
|---|---|
| first + unmarked | 5.678 |
| later + unmarked | **5.300** |
| first + marked | 1.528 |
| later + marked | 1.152 |

The **marker** separates the ratings. The **ordinal** does not. A title in the
spots log names a SERIES, not an episode — `פרק N` appears in exactly zero of
50,386 titles — so the twentieth airing of a daily strip is its twentieth new
episode, not a repeat. There is nothing to number.

That is a negative result with a positive corollary, and it closes an open
question: **the marker is not systematically missing repeats in training.**
Unmarked later airings rate 5.300 against 5.678 for unmarked first airings. If
hidden repeats were sitting in that bucket they would drag it toward 1.2.

## And the factor is not what any of us thought it was

The series factor separates repeat from first-run perfectly — 302 marked cells,
301 clean, zero mixed — so it looked like a repeat detector wearing another
name. Every attempt to replace it with an explicit, semantically clean key lost
accuracy, and the pattern in the losses gives the real answer:

| key | cells | held-out log RMSE |
|---|---|---|
| clean series | 218 | 0.66665 |
| clean × repeat | 295 | 0.64164 |
| clean × repeat × season | 343 | 0.64494 |
| **shipped (the defective key)** | **603** | **0.63075** |
| **raw title, no normalisation at all** | **617** | **0.63299** |

Accuracy tracks CELL COUNT, and the raw title matches the shipped key. The
factor is not capturing programme identity, or repeat-ness, or season. **It is
memorising the title**, and on one month with shrinkage that is the best
predictor available. The clean repeat separation is a consequence of the marker
living inside the title, not the mechanism of the factor's value.

**Which means it does not generalise, and this is measurable.** On the 704
broadcasts actually pulled for the coming fortnight, the shipped series factor
finds a cell for **57 — 8.1%**. Per channel: כאן 11 5.0%, קשת 12 8.3%,
עכשיו 14 11.5%. The second-strongest family in the model, worth +9.5% on
in-window temporal folds where the same titles recur across every fold, is
**silent on 91.9% of the week it exists to plan**.

Nothing here is a reason to change the key. It is a reason to stop reading the
gate percentage as if it described the plan.

## So what DOES survive into the week being planned

The obvious rescue is the one the feed hands over for free: it flags Live and
Rerun on 100% of the coming fortnight, correctly, already inside the contract.
If the series factor is silent on 91.9% of those rows, surely a repeat factor
should speak there, where there is nothing left for it to be collinear with.

It was measured, by scoring the folds' test rows split into those whose series
cell WAS seen in training and those whose was not. The unseen rows are the
forward condition reproduced inside the data we have.

| held-out log RMSE | base | + weekday_slot | + series | + repeat |
|---|---|---|---|---|
| test rows with a **seen** series cell (n≈604) | 0.92096 | 0.67865 | **0.59980** | 0.71535 |
| test rows with an **unseen** series cell (n≈87) | 1.05027 | **0.81887** | 0.81887 | 0.84414 |

Three things, and the middle one is the good news:

- **The series factor contributes exactly 0.0% on unseen rows.** Not
  approximately: by construction, there is no cell.
- **weekday_slot keeps nearly all of its value: +22.0% on unseen rows against
  +26.3% on seen ones.** Its cells are a weekday crossed with a slot band, and
  every future broadcast has both, so it reaches 100% of any week. The forward
  model is a weekday_slot model, and that family is genuinely carrying it.
- **The repeat factor makes unseen rows 3.09% WORSE**, so the rescue does not
  exist. The composed gate refused a repeat family in-sample for being collinear
  with the series key; it turns out to be right forward as well, for a different
  reason. The overnight slot cells already carry when repeats air.

So there is no missing lever here. There is a model whose two halves cover a
future week very differently, and that fact is now published rather than
inferred: `kairos/model/audience_reach.py` measures, per family, the share of a
pulled schedule that family can reach, and `audience_model_status` carries it
beside the gate. Against the live artifact and the 704 broadcasts on disk:
weekday_slot 704 of 704, series 57 of 704.

## The repeat flag reaches nothing, which is why it was not "fixed"

The plan above this said to read the feed's `Rerun` flag instead of re-deriving
it from the title, on the strength of a real 39.5% miss rate. Before building it,
the field was traced. **`Classification.is_rerun` is read by no decision in this
repository.** The only non-test references are a pass-through in
`ai_classifier.py` and the docstring correcting itself in `keshet_epg.py`; the
whole of `kairos/`, `kairos_api/` and the dashboard contain no other reader.

It is computed on every classify call, carried in the dataclass, emitted as a
column by `classify_series`, counted by `coverage_report`, asserted by two tests
about itself, and consumed by nothing. Making it 39.5% more accurate would move
no number on any screen. That is worth knowing before building, not after, and
it is the same class as the inert levers already recorded elsewhere in this
engine. The flag stays carried and correct in the contract, the miss rate stays
documented, and the fix waits for a reader that would notice.

## What is worth doing, in order

1. **Archive the feed daily.** It is the only source that has ever carried
   liveness, and every day not archived is a day that can never be labelled.
   Cheap, certain, and it is the precondition for every answer below.
2. **Give the repeat flag a reader before making it accurate.** The feed's own
   `Rerun` is 39.5% better than the title-derived one and nothing consumes
   either, so the accuracy fix is worth nothing until something reads it. Not a
   forecast factor: measured above, a repeat family is worse on exactly the rows
   where it had room to help.
3. **Leave the series key alone** until the effect it smuggles has a name of its
   own, and disclose that it is doing so. Both are done.
4. **Do not build a liveness factor yet.** There is nothing to fit it on, and the
   only three attempts to create something returned nulls and a sign reversal.
   It becomes answerable when a rated file carries a structured live flag — which
   is what item 1 accumulates toward.
