# One programme, three places, no identity

The model trains on what aired. The feed pulls what is about to air. The
optimizer plans against the coming week. For a forecast to use what a programme
has actually done, that programme has to be the SAME THING in all three — and
today it is not the same thing in any two of them.

Every number below was measured against the files on disk on 2026-08-19: the
live `data/reference/CompetitorProgrammes.csv` (704 broadcasts, 3 rivals, 287
distinct titles) and `data/Programmes.csv` (3,562 broadcasts, 4 channels, 417
distinct titles, November 2024).

## What is measured

### There is no programme identity anywhere

`data/Programmes.csv` carries a column called `programme_id`. It is a **row
number**: 3,562 distinct ids across 3,562 rows and 417 distinct titles, and 317
of those 417 titles carry more than one id. It does not identify a programme; it
identifies a broadcast. The name promises otherwise, which is worse than having
no column at all — anything that joined on it would join nothing and report a
clean empty result.

The competitor contract carries `ProgramCode` and `HouseNumber` from Keshet's
publication, and neither appears in history, so neither can join to it.

### The title does not carry across

| Match | Result |
|---|---|
| Future title == historical title, exactly | **12 of 287 (4.2%)** |
| After stripping `" – suffix"` and rerun markers | 25 of 160 series (15.6%) |

**The 21-month gap is a real confound and it is not the whole story.** History is
November 2024; the feed is August 2026. Some of that miss is genuine schedule
turnover — programmes that ended, programmes that launched — and no identity
scheme recovers those. But the second row shows the part that is identity: 13
series that exist in both and were only found once the episode suffix was
removed. Among them `אולפן שישי`, `הבוקר הזה`, `חדשות הערב`, `כאן בשש` — daily
strips that certainly ran across both windows and did not match on their
printed title.

The season case the owner named is visible in the feed and not yet in history:
`אוכל מהצומח 2` is in the future file, and history's only digit-suffixed titles
are `חדשות 12` / `חדשות 13`, which are channel names rather than seasons. So the
season problem is real and this corpus simply cannot show it: a one-month
history contains no second season of anything.

### The genre classifier was tuned on the history and is half blind on the feed

| Corpus | Broadcasts | Fall to `Other` |
|---|---|---|
| `Programmes.csv` — what the model trained on | 3,562 | **118 (3.3%)** |
| `CompetitorProgrammes.csv` — what it is applied to | 704 | **269 (38.2%)** |

**An eleven-fold gap.** The cause is in the classifier's own rule mix: 156 of the
417 historical titles resolve through a `specific` rule, that is, a hand-written
list of titles taken from that corpus. On titles it has never seen, coverage
collapses — worst on כאן 11 (47% of broadcasts unknown), then עכשיו 14 (42%),
then קשת 12 (29%).

### And `Other` is treated as a genre rather than as "unknown"

`kairos/model/competitor_features.py` never mentions `Other` anywhere.
`_category_at` returns it like any real category, and `_genre_contrast` counts a
rival as airing the same genre when `rival_category == own_category`. So two
unknowns count as a match, and a rival whose genre we failed to read counts as a
contrast we did not measure.

This matters more now than it did a week ago. The feature learned what "same
genre" means where the genre was unknown 3.3% of the time. It is being applied
where it is unknown 38.2% of the time. That is not drift at the margin: the
feature does not mean at inference what it meant at training.

### What the future schedule actually contributes today

Three forward features, and only one of them looks at the programme at all:

| Feature | Joins on | Uses the title? |
|---|---|---|
| `competitor_strength` | `(channel, broadcast minute)` from the historical audience curve | **No** |
| `competitor_genre_contrast` | classifier category, from the title | Yes, as a category only |
| `competitor_prog_start` | rival programme start times | No |

So the engine now knows exactly WHAT each rival is airing, and the audience part
of the model still only knows WHEN. Pulling the lineup was the prerequisite;
nothing yet reads the programme.

### The identity that was built is not connected

`kairos/model/keshet_enrich.py` has `series_of()` — the very
strip-the-episode-suffix rule the table above shows is worth 13 series — and a
`SeriesMemory` that remembers a resolved category per series with a
description-fingerprint so a rewritten synopsis re-asks and a cosmetic edit does
not. Its **only callers are tests**.

`data/reference/keshet-series-memory.json` holds 26 series already resolved by
the model, with a category, a reason and a confidence for each. **Nothing reads
it** — not a Python module, not a screen.

### One more trap, found while tracing and now closed

`future_epg._resolve_future_epg_path` prefers `.xlsx` over `.csv`, and this
schedule arrived as a hand-saved workbook for years. A leftover workbook beside
the file the feed writes would have made every pull a silent no-op: written,
logged, never read. The feed now refuses to pull in that state and names the
file. There is no workbook there today.

## The identity function does not run on the data it identifies

Found by an independent audit and then reproduced directly. `canonicalize_series`
strips bracketed content FIRST:

```python
_BRACKETED = re.compile(r"[\(\[\{].*?[\)\]\}]")   # "anything in brackets is a tag"
stripped = _BRACKETED.sub(" ", text)
```

That is right for `מאסטר שף (עונה 7)`, where the bracket holds a tag. It is
catastrophic for `[מאסטר שף עונה 7 ש.ח]`, where the bracket holds the **whole
title** — which is **99.4% of `data/Spots.csv`**. The rule eats everything, every
stripper below it runs on empty text, and the fallback returns the raw title with
the season number and the repeat marker intact:

```
'[זהו זה עונה 7 ש.ח]'  ->  'זהו זה ש ח'      # markers survive
'זהו זה עונה 7 ש.ח'    ->  'זהו זה'          # same title, unbracketed
```

Measured consequences on the shipped artifact: **603 series cells where 256
series exist**, and **348 of those 603 keys (58%) carry a season number or a
repeat marker**. One series, `אנחנו על המפית`, holds four separate cells —
season 2, season 3, unnumbered, and one where two concatenated programmes became
a single phantom series. 97 real series are split across 205 keys, touching
33.6% of the spot rows. A further 19.2% of rows are junction titles
(`[A] * [B]`), which today become a series that is neither A nor B and fragment
both.

`ProgramClassifier` was built for this format and handles it —
`text.replace("[", " ").replace("]", " ")`. Two identity functions in one
repository, opposite behaviour on the same string.

## Fixing it makes the forecast measurably WORSE

The fix is small: unwrap the brackets instead of deleting their contents, and
take a junction title as the programme it followed. Verified — the same title
with and without brackets now agrees, no key retains a marker, and **zero keys
absorb titles with different lead programmes**, so nothing over-collapses.

Then it was measured on the walk-forward backtest, out of sample, and the result
is the opposite of what a cleaner key was supposed to buy:

| | log RMSE | MAE (rating points) | bias | series baseline MAE |
|---|---|---|---|---|
| **Shipped, with the bug** | **0.6834** | **1.1883** | −0.2486 | **0.8976** |
| Identity fixed | 0.7008 | 1.5028 | +0.4017 | 1.1489 |

**Worse on every measure**, and the naive series-history baseline got worse too —
because it joins on the same key. The bias even flips sign, from the
under-prediction that fitting in log space produces to over-prediction.

The explanation is not subtle once measured. The markers the buggy key kept are
not noise. A repeat draws **a third of the audience of a first broadcast** — mean
TVR 5.77 against 1.93 across 50,386 rows, Mann-Whitney p below floating-point
resolution, one of the largest effects in this data. By leaving `ש.ח` inside the
key, the bug was smuggling that in as if it were programme identity, and the fit
was using it. Cleaning the key deleted a real signal along with the mess.

Season looks different: only 11 series have more than one season in this window
and the within-series differences are small on cells of n=1 and n=3. On this
corpus, repeat-ness is the signal and season is not — yet.

**So the change was reverted.** Nothing about the bug became less true; a
theoretically cleaner key that measures worse out of sample does not ship, and
the artifact is back to 603 cells and its 16.08 gate.

A rerun factor was also built and gated (`rerun`, 12.2% held-out improvement,
third strongest of nine families) and reverted with the rest: on the unfixed
identity it is collinear with the series key that is already carrying it, so it
would have been measuring the same thing twice.

## Two keys, and the coarse one is built

The two jobs need two keys, and conflating them is what the bug was accidentally
getting right.

**The coarse key for JOINING is shipped.** `series_join_key` in
`kairos/data/title_features.py`, beside the fit key rather than replacing it,
sharing its primitives and composed for the other job: brackets unwrapped rather
than deleted, a junction title taken as the programme it followed, the episode
name after a dash removed, then the same season and repeat strippers.

Measured on the real corpora:

| | future rows finding their history |
|---|---|
| `canonicalize_series` (the fit key) | 120 of 704 — 17.0% |
| `series_join_key` | **202 of 704 — 28.7%** |

The owner's own example resolves: **eleven MasterChef titles in one fortnight's
schedule become one key**, because a broadcaster names every episode and the
episode name is not the series. `הכוורת`, `שום פלפל ושמן זית` and
`ההמצאות של המחר` collapse the same way, and no key absorbs two different
programmes.

It is written into the competitor contract as a `SeriesKey` column rather than
recomputed by each reader, so the three places that need this identity cannot
drift apart about it. Nothing the model fits on moved: the forecast and
backtest suites are unchanged, because `canonicalize_series` was not touched.

**The fine key for PREDICTING is still the accidental one.** Series + season +
repeat, smuggled through a bug rather than named. Making it explicit is the next
piece, and it has to be measured family by family the way everything else here
is — `test_series_identity.py` pins the current behaviour so it is not "fixed"
by reasoning alone.
3. **Read the description — DONE.** 40.8% to 21.6% on the feed, history
   unchanged at 3.3%, zero already-correct titles disturbed. The title wins
   outright (scoring both together corrupts 39 of 378) and programme names are
   removed from a synopsis before it is scored, because 34 of them are also
   category keywords and a guest introduced as a star of "האח הגדול" was
   turning an unrelated exposé into Reality. That last part was claimed fixed
   once before it was: blocking the `specific` rule alone left the keyword path
   wide open, and the hijack was live on the feed.
4. **Connect the enrichment that exists — DONE**, and measured rather than
   assumed. It answers 21.6% to 18.7%, last in the ladder, never overruling the
   taxonomy on the 48 broadcasts where they disagree, and only from FRESH
   entries: 19 of the 38 it could match carry a synopsis the broadcaster has
   since rewritten. Keying it on the coarse `series_join_key` was measured and
   NOT done — both keys answer exactly the same 38 broadcasts here, so it would
   have been a rename that sounded better and did nothing.
5. **Stop counting unknown as a genre — DONE.** 56 of 638 forward slots were
   scoring a positive contrast with an unknown own genre, mean 0.5714. No
   shipped number moved: the same defect touches 2 of 2,532 history breaks
   (0.08%), because the corpus that judged the feature is 3.3% unknown and the
   corpus it is applied to is 21.6%.
6. **Join a future programme to its own history.** The largest forecast win and
   the largest change — and the one this corpus cannot yet honestly support: one
   month of history, twenty-one months before the week being planned. Behind a
   gate, off until a held-out measurement earns it, the way the audience model
   already is.
