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

## What is worth doing, in order

1. **Archive the feed daily.** It is the only source that has ever carried
   liveness, and every day not archived is a day that can never be labelled.
   Cheap, certain, and it is the precondition for every answer below.
2. **Read the feed's own Rerun flag** instead of re-deriving it from the title.
   It is already in the contract and it is 39.5% better.
3. **Leave the series key alone** until the effect it smuggles has a name of its
   own, and disclose that it is doing so. Both are done.
4. **Do not build a liveness factor yet.** There is nothing to fit it on, and the
   only three attempts to create something returned nulls and a sign reversal.
   It becomes answerable when a rated file carries a structured live flag — which
   is what item 1 accumulates toward.
