# Resume here

## FINAL CODE-WAVE RE-AUDIT, 2026-08-10

Wave 0 and every piece from P1 through P13 were re-run on the current tree.
The disjoint gauntlet groups report **2,182 passed and 6 skipped**: 183 for W0,
739 for P1-P4, 593 for P5-P8, and 667 for P9-P13. P1-P12 are passed. P13's
implementation is passed and its live state is owner-blocked only on a real QC
report keyed by House Number and owner-approved playout standards. The complete
matrix and the four findings closed during the re-audit are in
`audits/completion-reaudit-2026-08-10.md`.

Two audit findings changed the tree after the P13 commit: the P4/P11 empty
make-good reason now follows the reader's locale, and a stale W0 guard now
protects the single shared genre table rather than requiring the duplicate P1
removed. Two oversized test files were split at real seams. Per the owner's
latest ruling, an existing file below 500 lines is no longer work by itself.
The two pre-existing weekly schedule output modifications remain unstaged.

## AUTHORITY UPDATE, 2026-08-10, after the completion wave

This section supersedes every older status sentence below it. The old material
is retained as evidence of what was found, not as a current queue.

**PASSED NOW:** P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11 and P12.
**OWNER-BLOCKED:** P13 only. Round P13-R4 supersedes the earlier code-complete
claim. A cold audit found and fixed two real defects: the asset was joined on
version name instead of House Number, and `blocks_lock` never reached the lock
route. The internal path now passes, including the full JS-8 metadata contract,
an atomic importer and independent server and surface lock gates. Live delivery
still has zero real media rows and zero owner-supplied playout standards. Do not
fabricate either to turn the unavailable state green. D4 remains the broader
current-week and as-run decision; the media QC source is a separate owner input.

The completion wave closed the following items:

- P2 refuses to freeze a plan collapsed to zero without a separate deliberate
  confirmation, and a zero-break run is an amber warning rather than an ordinary
  completion toast. The server independently enforces the confirmation.
- P3 refuses Gold while unsaved edits exist, refuses a stale partial save, turns
  stale score calls into a named conflict, and normalises rounded negative zero.
- P5 measures the exact draft constraint before save and refuses an empty or
  zero-match rule. The preview is invalidated by any subsequent edit.
- P6 lists all four previously invisible engine reads, and the inventory source
  now distinguishes a file being read from a pool that actually yields items.
- P1 localises every raw programme genre inside recommendation prose. P11 shows
  its trigger and threshold before the expandable explanation.
- P8 renders `airtime_caps` as a translated record with real, partial and not-set
  states, never `[object Object]`.
- P9 emits an honest local first line after grounding and before any provider
  call. The applied-change and addressable-restore paths remain green.
- P10 has no fabricated order row. Preferred-position status and both rate
  methods reach the surface, and the lead/closer pair check is recomputed on the
  order currently shown. Unknown configuration is printed as unavailable.
- P12 browser verdicts freeze the complete published common-basis evidence while
  keeping the adoption act unreachable from the application. The guard is
  proven both ways on a temporary tree: incorrect or absent approval refuses;
  an exact approval passes; revert restores the original bytes.
- P13 consumes a QC report keyed by House Number, not copy version, carries all
  eight measurement families from JS-8, and refuses finalisation on a measured
  failure in both the API and the dashboard. `config/media_standards.json` and
  `data/media_assets.csv` remain honestly empty until their real owners supply
  the rules and measurements.
- The fabricated row in `data/break_pod_order.csv` was removed. The two dirty
  weekly schedule output files predated this wave and must remain unstaged.

Validation recorded in this wave: focused groups of 217, 46, 19 and 74 tests;
198 P9/P12 browser and core tests after the evidence fix; 215 P8/P10 and trade
tests; 55 P10/P13 tests; the full dashboard build and every dashboard guard.
The P13 cold re-audit then added 18 contract tests; the complete P10/P13 group
now passes 66 tests, followed by the full dashboard build and all guards.
The broad P9/P12/P13 collection first reported 46 failures and two errors: 45
plus two were Chrome or localhost sandbox denials, all 67 passed in the approved
browser run; the one real training-line failure was fixed and its 131-test core
batch passed. No output schedule was regenerated.

Rewritten 2026-08-09 at the end of a very long session. If you are a new session
picking this up, read this file first and trust it over any summary you were
handed. Everything below was measured, not remembered.

---

## THE JUDGING ROUND, where it stands 2026-08-10

**EVERY PIECE NOW CARRIES A ROUND, INCLUDING P13.** PASSED: P1, P2, P4, P7, P11.
FAILED: P3, P5, P6, P8, P10. PARTIAL, no verdict claimed: P9, P12. BUILDING,
backend only: P13. All through `update_state.py`, none hand-edited. Before today
`state.json` carried eight rounds, all SPEC and wave zero; it carries 21.

**P13 existed nowhere and now has a tested backend.** `media_store.py`,
`media_verdict.py`, `media_api.py`, `data/media_assets.csv` and 11 tests. It
ships with an EMPTY store deliberately: nothing in this repository observes a
media file, so all four technical facts are honestly unavailable, and a
fabricated "verified" would clear a corrupt file to air. The rule the design
turns on: **unavailable is not a pass and not a failure, and only a MEASURED
failure blocks a lock** — if absence blocked, no pod could be locked at all today.
**THE SURFACE IS NOW BUILT TOO.** `PodBoard.jsx` was split rather than
compressed, along a real seam (the two readouts describing what is wrong or
missing moved to `PodBoardNotes.jsx`; the pod's own shape and order stayed), and
it now sits at 433. The verdict rides the pod payload rather than one request per
spot, and prints as a quiet mark in a new File column. Measured on the live
payload: all ten pods carry the block, zero assets on file, no lock blocked.
**What remains for P13 is NOT code: it is a data source.** Connect a real ingest
or transcode report and the verdict computes for real with no code change. That
is an owner question. The piece is built and unjudged.

**P9 and P12 were stopped mid-round by a controlled stop and neither invented a
verdict.** P9 reached about a third of bar 1; its one hard number is a first
token at 3,310 ms against a 2,000 ms clause, and it disowned its own 150 s wall
clock after working out its completion detector never fired. P12 passed what it
reached; its escalation guard is PROVEN TO FIRE (a real adoption refused with
four named stop reasons, coefficients byte-identical) but NOT proven to pass,
because the counter-probe never finished. **A guard seen refusing and never seen
approving is not yet proven to discriminate.** Both need a finishing round, not
a fresh one.

**P11 passed and REFUTED a lead I had written into this file as fact.** See
item 3 below. That is the second time this round a first-hand measurement
overturned a second-hand claim of mine, and the pattern is worth more than
either finding.

**P2's own largest gap is the most dangerous single finding of the round, and it
did not cost it the bar: THE PRODUCT WILL FREEZE, NAME AND PUBLISH A PLAN WORTH
₪0 ON THE OPERATOR'S OWN CHANNEL WITHOUT ONE WORD OF WARNING.** Measured: with
the plan collapsed to 0 breaks, the run reported done, the freeze button was
enabled, and the version rendered as "0 ברייקים · ₪0" in the SAME grey type as
four neighbours reading "2,391 ברייקים · ₪40.9M". The only place the collapse is
stated in figures is the diff, behind a click. The story's done-condition is that
everyone downstream reads the published plan, so this ships them a zero and calls
it a version. Fix: `PublishPanel.jsx` (compare owned breaks and revenue against
the predecessor before enabling the freeze; a collapse to zero raises the amber
note the panel already has and demands a second deliberate act),
`RunPanel.jsx`/`use-plan-surface.js:240` (a zero-break run is a refusal to be
quiet, not a completion toast), `week_api_publish.py` (publish should carry the
owned-vs-predecessor delta it already computes for the diff route).

**Two harness facts from P2 worth keeping.** It RETRACTED both of its first two
findings after reproducing them, and said so. And another critic's state reached
its ISOLATED /tmp mirror: `data/` there was rewritten twice by the server's own
restore machinery with NO restore request in its access log, cycling in a
historical snapshot that carried a rival `operator_channel` and a constraint row
literally named `CRITIC-P8-PROBE-ROW`. The live tree was never touched. Worth a
look on its own terms: a restore that rewrites `data/` with nothing in the log
is a product behaviour, not only a test artifact.

**Still unjudged: P9, P11, P12.** They were deliberately held back, because eight critics on one
machine produced a 44,866 ms first paint and bar 1 is a TIMING bar, so
concurrency poisons the measurement it exists to take.

**Every one of the five failures failed on bar 4 or on a single clause, never on
speed.** P3, P5, P6 and P8 all MET their stopwatch targets by wide margins (P8 at
90 ms against a 27,000 to 53,000 ms baseline; P5's preview at 124 ms against a
3 s target; P3's money verdict at 3.2 ms against 500 ms). The product is fast.
What it is not yet is honest at the edges, and that is a different repair.

**HALF A FIX IS ON MAIN AND NOTHING READS IT YET — START HERE.** `live_state()`
in `kairos_api/plan_version_store.py` now returns a `summary` block carrying the
operator-scoped totals a freeze would capture (measured live: 2,391 breaks,
₪40,935,408.65). It was added for ONE named consumer that has not landed: the
publish panel's collapse guard. Until that consumer exists this is a value
computed and carried that reaches nobody, which is the INERT LEVER class by this
campaign's own definition, and it is recorded here as such rather than counted as
a fix. The remaining half, all in P2's row: a helper deciding whether the plan has
collapsed against the newest version's `summary.owned`, an amber note plus a
required second act in `PublishPanel.jsx` before the freeze enables, a run toast
in `RunPanel.jsx` that refuses to be quiet at zero breaks, and a test. 394 tests
pass with the field added; it changes no behaviour.

**Four more located fixes, from the three verdicts that closed the round:**
- **P3, the worst of the round.** JS-3's own sequence dead-ends silently: with a
  pending move, pinning a gold break re-plans the segment, deletes the edited
  chip, and leaves an ENABLED Save that issues zero HTTP requests, after which
  every score call 404s. `break_api.py:218-232` plus the three day-board files.
- **P5.** The condition builder saves a live rule from an entirely empty form:
  two clicks, 201 Created, store 7→9 rows, no preview, no cost, no validation.
  `/api/constraints/effect` — which P5's own contract publishes as "what a
  constraint would do before it is saved" — is referenced ZERO times in the
  frontend while its sibling is wired twice. `ConstraintBuilder.jsx:177` and 325.
- **P8.** `airtime_caps` renders as `[object Object]` beside its raw engine key,
  one line above nine guardrails rendered correctly in Hebrew. An absent value
  shown as neither real, unavailable nor unknown. `HistoryDetail.jsx:61` and
  `history-labels.js:298`. NOTE: that field is recent work of mine.
- **P3 again.** The board prints a gap of "-0 ₪ (0%)" at rest with nothing
  edited, deterministic over three reads, because `day-board-model.js:236`
  compares a whole-day sum against a per-break tolerance.

**Eight of the first nine critics went idle without ever reporting**, after three
wake attempts each. One had to send three times before a message landed. They
were replaced rather than chased further. If it happens again, replace sooner and
tell the replacement to report in five lines, not forty.

### Three measured findings not yet fixed, each with its fix already located

1. **A Hebrew sentence prints the English genre, on 22.1% of the plan.**
   `overview_api.py:167` interpolates the raw English `program_type` into the
   Hebrew title. The frontend repairs it with a hand-kept regex list covering 7
   genres (`shell/labels.js:143-162`), and 9 of the 15 types in the plan are
   uncovered: **1,924 of 8,704 rows**. The complete 15-type map already exists
   **47 lines above in the same file** (`programTypeLabel`, `labels.js:83-106`),
   and its own comment states the invariant the title line breaks. Route the
   substitution through it. Four parallel genre vocabularies exist and two give
   DIFFERENT Hebrew words for the same genre, so reconcile them in the same pass.
   Nothing tests this; the fix must ship an assertion over all 15 types.

2. **`GET /api/parameters` ships every rival channel name to the browser on every
   boot**, unguarded: `.channels` and `.available_channels` both carry קשת 12,
   כאן 11 and עכשיו 14. It is NOT obviously a breach, because the list drives the
   operator-channel picker and you cannot choose your own channel from a list of
   one. It is a defect because **the fully worked-out right answer is already in
   this repository and this site does not use it**:
   `compliance_api_licence.py:301-307` gates the same list on
   `CHANNEL_WALL.allows(request)`, narrows to the current channel otherwise, and
   publishes `lists_every_channel` so the client knows which it got.
   `scenario_api_parameters.py:97` has no wall check at all. **Before fixing,
   settle whether a channel account can change the operator channel**; if it
   cannot, the narrowed shape is strictly correct for it. Two sites is a lead,
   three is a class, so count first.

3. **English-only sentences on a Hebrew surface — HALF OF THIS WAS REFUTED, read
   the correction before acting.** I recorded that `counted_as_of_basis` renders
   raw at three sites and that the seam was "half-applied even in the piece that
   invented it". **P11's critic went and looked, and that half is WRONG.** On the
   pacing board the sentence renders inside a disclosure, prefixed with a Hebrew
   lead-in announcing that the source is English-only, and marked `<q lang="en">`.
   That is honest disclosure of a source string that genuinely has no Hebrew
   twin, not a leak. The whole Hebrew board carries 4 Latin tokens and two of
   them are advertiser names. **The seam is FULLY applied there.**
   What survives, and is NOT yet re-measured by anyone who owns it: the campaigns
   board site (`DeliveryState.jsx:107-126`) and the single-language `reason` at
   `MakeGoodAlerts.jsx:86`. Treat those as leads, not findings, and check whether
   they already do what the pacing board does before writing any code.
   **The lesson is the one this round keeps paying for: I wrote a second-hand
   finding into this file as fact, and the critic who owned the file refuted it in
   one measurement.**

4. **The inputs page never names four of the files the engine reads**, and one of
   them is the INERT LEVER on that page's own purpose. `data/Spots - inventory.csv`
   holds 994 data rows, is read on every run (`kairos/service.py:344`) and is
   checksummed into the plan freshness fingerprint
   (`kairos/export/schedule_freshness.py:226`), and `load_inventory()` returns a
   pool of **0**. The discard itself is OLD NEWS and owner-gated, because fixing
   the parse activates a steer and moves money. **The new half is that P6's
   surface is silent about the file entirely**: 0 occurrences across `/api/files`,
   `/api/uploads/status`, `/api/reports`, `/api/parameters`, `/api/overview` and
   the whole frontend, on the destination whose stated purpose is "the inputs a
   run reads". One pass fixes it: `_also_read_paths()` at
   `kairos_api/downloads_api.py:326` is a hand-written list whose own docstring
   promises "no name on any card is a dead end". Add the file and give it a
   verdict FROM THE POOL rather than the row count, so a file whose 994 rows all
   die reads as read-and-yielding-nothing instead of absent. Three more
   engine-read files are undeclared the same way (`data/manual_overrides.csv`,
   `models/audience_model.json`, `data/kairos_settings.json`), so fixing the list
   once closes the class.

**CLOSED, and worth knowing before anyone re-reports it:** P6 contract §14's
declared competitor leak in `GET /api/export/schedule.csv` is GONE. Measured
today: 2,540 rows, all רשת 13, zero rival rows, against 6,164 rival rows and
₪180,938,215 when it was filed. That was the row's one standing red test.

**A pattern worth naming across 2, 3 and 4: THE HALF-APPLIED SEAM.** The correct
answer exists in the repository, is written down, sometimes with a comment
stating the invariant, and one site does not use it. That is cheaper to fix and
easier to miss than an absent solution, and it is now the shape to look for.

---

## The one-line state, 2026-08-10, CORRECTED

I wrote "all waves are closed" here and it was WRONG. It is true of the
follow-on workstreams and false of the thing this campaign is named after.

**WAVE ONE, THE GAUNTLET ITSELF, IS OPEN. One piece of twelve has passed.**
`state/P7.json` is the only `passed: true`. Every other piece reads `passed:
false`, and the note on most of them says exactly why:

> "Still false. The class fix did not re-run the four bars; no blind sweep has
> passed this piece."

That is the distinction I collapsed. **The defect CLASSES are closed** and each
has a guard proven to bite: direction, dates, cards, accents, colours, native
controls, six of them green with empty quarantines. **The PIECES are not judged.**
Closing a class is a measurement anyone can repeat; passing a piece requires a
BLIND CRITIC re-running the four bars against that piece, and none has been run
since the class fixes landed. A guard going green is not a verdict.

So: pieces P3, P4, P5, P6, P8, P9, P10, P11, P12 are unjudged, P7 passed, and P1
and P2 passed in wave one's earlier rounds.

### TWO THINGS ARE BOTH CALLED "WAVE TWO", which is how I got this wrong

Measured 2026-08-10 off `state.json`'s own `wave` field, which is the authority:

| the gauntlet's BUILD waves | pieces |
|---|---|
| wave 0 | W0-1 … W0-5, all passed |
| wave 1 | **P1 … P9**, nine pieces |
| wave 2 | **P10, P11, P12, P13**, four pieces |

The marathon's numbered SWEEP waves (the backlog rounds, two through six) are a
different sequence entirely, and those are the ones closed below. Saying "wave
two is closed" is true of the sweep and false of the build, and collapsing them
is exactly what produced the "all waves are closed" error this file opens with.
**Name which sequence you mean, every time.**

That also corrects the count above. "One of twelve" spans both build waves. Per
wave, measured:

- **Build wave 1 (P1-P9).** P7 passed and is the only piece with an artifact
  proving it: `state/P7.json` carries `passed: true` at round 5, beside a critic
  file. P3, P4, P5, P6, P8 and P9 are in critique now.
- **P1 AND P2 ARE CLAIMED PASSED AND I CANNOT EVIDENCE EITHER.** There is no
  `state/P1.json`, no `state/P2.json`, and no P1 contract file; P2 has a contract
  carrying no verdict. This file and `critic-briefing.md` both assert the pass and
  neither cites an artifact. `update_state.py` would refuse to publish it, and it
  is right to: a claim is not evidence, which is the standard the whole campaign
  judges the product by. Either the round record is found, or the two pieces are
  re-judged, or the claim is retracted. Do not carry it forward as fact.
- **Build wave 2 (P10-P13).** P10, P11 and P12 have state files dated 08-09 and
  are in critique now. **P13, Media verification, has not been started at all:**
  no contract, no state file, and none of its three owned files
  (`kairos_api/media_api.py`, `kairos_api/media_store.py`, `data/media_assets.csv`)
  exists. It is absent from the critic briefing, which is correct for a wave-one
  round and not a gap.

**`state.json` IS NOT THE CURRENT STATE.** Its `meta.generated_at` is
2026-07-31, its `campaign.phase` still reads `wave0_building`, and it carries
eight rounds, all of them SPEC and W0. Every P-piece in it reads `waiting` with
zero rounds while P4 alone has twelve rounds of recorded history. The workbench
page embeds a copy older still. **Trust this file and the per-piece
`state/<piece>.json` files; treat `state.json` as a wave-0 artifact until the
wave-one verdicts are published into it.** Publishing them is what closing wave
one MEANS, and it goes through `update_state.py` (never by hand).

WAVES TWO, THREE, FOUR, FIVE AND SIX OF THE SWEEP ARE CLOSED. **Sweep wave two
closed by measurement rather than by work:** its 80-row backlog is 12 CLASSES,
and 76 of the 80 rows were already covered by guards that exist and are green,
including all 62 rows of its single largest class. Of the four left, three are
judgement for the owner and one was a gap in a guard, now closed.

SIX frontend guards, all green, all quarantines empty, and `npm run test:all`
runs the BUILD FIRST because five design guards passed twice on a tree that would
not compile.

    npm run test:all      build, then the six
    npm run test:guards   the six alone

## The two things a new session must not repeat

**THE PLAN ARTIFACT IS NOT POLLUTED, IT IS STALE.** I restored it four times,
each time calling the correct file damage. A fresh export reproduces the on-disk
file byte for byte, and the tree at the commit that COMMITTED the plan produces
the same thing, so the artifact was never what its own commit's code produced.
The golden asserts against its OWN embedded baseline and not against this file,
which is why it stayed green throughout and why I misread it as confirmation.
Decision 12: replace it, nothing to attribute.

**A MEASUREMENT WHOSE FAILURE MODE IS A COMFORTABLE ANSWER IS NOT A MEASUREMENT.**
Caught four times in one day, three of them by agents and once by me: a cap
fixture measured on hours holding zero breaks; a determinism probe patching a
value bound at definition time; a bite harness whose cleanup did not run, leaving
an injected `revenue_delta: -1.0` that I then COMMITTED with `git add -A`; and my
own grep with a malformed flag reporting zero literal colours where there were 67.
`tests/lever_probe.py` enforces the rule for levers: refuse to rule on a fixture
you cannot show is binding.

## What is running RIGHT NOW, and do not duplicate it

Check `find . -newermt "-20 minutes" -not -path "./.git/*"` before touching
anything. Agents were live at the end of this session on:

- **Top and Tail** — `kairos/optimize/frequency.py`, `_frequency_rules.py`,
  `data/campaign_assets.csv`, `data/frequency_rules.csv` (adding `pair_lead`,
  `pair_closer`, `value_max`). It briefly made `load_frequency_rules`
  unimportable mid-edit, which broke the daily pricing pipeline and produced
  transient failures ANOTHER agent mis-attributed to itself. Expect that shape.
- **The goal-based order (wave five)** — `campaigns_api_store.py`,
  `campaigns_commitment.py`, a new `kairos/optimize/goal_seam.py`.
- **The mention picker (wave four, R1)** — `assistant_mentions.py`,
  `MentionPicker.jsx`, its own stylesheet.

## The rules this session paid for, in the order they cost the most

1. **A named gap is a defect CLASS, not a site.** Measured repeatedly: the accent
   bar was reported at one site and found at 23 across 11 files; the Latin `s`
   for seconds at one site and found at 8; the direction override at one and
   found at 68.
2. **A check whose failure mode is silence is not a check.** Met seven times in
   one day. The worst: `npm run test:smoke` died on an ENOENT from wave zero
   until yesterday, and it was the only thing banning native controls, so the
   tree drifted to 384 unseen.
3. **A number nobody re-derives is a number nobody checks.** Contract line counts
   were stale in 55 of 79 rows. My own dossier was refused by my own gate on five
   counts within a day of writing the gate.
4. **Measure before fixing.** Two queue items marked "serious, open" were already
   closed, one under the queue's own words "left open deliberately".
5. **A wave's size is unknown until its class is COUNTED**, and reading a backlog
   is not counting it. An 80-item backlog collapsed to a handful of real classes.
6. **Restore, then verify. Never trust a report over git.** An agent said a plan
   artifact's content was unchanged; git said 69 rows.
7. **A guard that counts literal substrings cannot tell code from a comment about
   code.** Two explanatory comments moved the counts they explained.
8. **A test that says what it measured and when is worth more than one that only
   says what it expects.** Twice this session a test failed because the product
   got BETTER, and both times the docstring made that legible.

## The four shared writable stores an agent has now polluted

Settings (twice), the override store, the agency layer. Each cost real money or
real time, and each guard was written after the fact and too narrow:

- `data/kairos_settings.json` — revenue_weight and min_retention_floor, then
  locale and direction. **15,844,833 ILS** and a declared licence breach.
- `data/manual_overrides.csv` — one gold mark. **131,878.70 ILS**, and it
  survived the settings restore because nothing guarded this file.
- `data/agencies.csv` + `agency_advertisers.csv` + `campaigns.csv` — twelve rows
  named "סוכנות ביקורת" at `critic.example`, with campaigns marked `is_demo`
  FALSE so seeded rows presented as real bookings. One pollution, five failing
  tests across three files, none of which named the cause.

Guards now: `tests/test_plan_artifact_fingerprint.py` (settings + an active-override
digest) and `tests/test_shared_stores_are_not_agent_leftovers.py` (the class).

**The gap that remains:** the fingerprint cannot catch an in-memory pollution,
because the exporter re-stamps the fingerprint in the same call, so the pair stays
consistent while the plan is wrong. What catches that is the golden's own
committed baseline hash.

## The six frontend guards, and every quarantine is empty

    npm run test:card       the card, its inset, off-scale padding
    npm run test:direction  isolation lives in shell/bidi.jsx and nowhere else
    npm run test:dates      a calendar day is read in shell/dates.js and nowhere else
    npm run test:accent     one-sided accent bars, AT ZERO, down only
    npm run test:smoke      native controls, at 350, down only
    node scripts/verify-card-rules.mjs

A directory is quarantined ONLY while an agent is actively holding it. Anything
longer is a budget with no number.

## What the owner ruled, and what still waits on him

**Ruled by the owner 2026-08-09:** the assistant runs the newest Opus
(`claude-opus-5`). That was his call because it spends his money.

**Ruled by me, and I should have ruled sooner:** ruling 009. The plan FILE keeps
every channel; the export ROUTE serves the operator's own. The two tests that
looked contradictory agreed about the file and differed about the route.

**Still waiting on the owner:**
1. `data/campaign_flights.csv` is header-only, owner decision 4.
2. What `EB` means in his traffic file. `סוג ברייק` takes Regular (111) and EB
   (64) on the shipped example, and nothing in the trade document says what EB
   is. Question 7 in `decisions-for-owner.md`, and the part that matters more
   than the label: whether EB prices or places differently.
3. The Cursor admin credential for cortex-lens.

## The adversarial re-audit, and its structural finding

`docs/audits/trade-reaudit-2026-08-09.md`. The finding that matters is a
mechanism, not a list:

> The transcript ends with its own nine-item summary. `trade-gap-analysis.md` has
> exactly nine sections in exactly that order. **It audited the summary, not the
> body.** Everything the summary did not lift became invisible.

Its top three, all confirmed:

1. **The preferred-position percentage was built, tested, bilingual and
   unreachable.** CLOSED 2026-08-09, `6dd9c1fd`. It has a seam and a route now,
   at `kairos_api/preferred_rate.py`, and it refuses to guess the preferred set
   or to pick a counting method. On the shipped file: 41 campaigns placeable, 51
   rows counted as dropped rather than vanishing, and with a set configured the
   two methods DISAGREE on 2 of 11 campaigns.
2. **Two contradictory answers to which positions are preferred.** CLOSED TODAY:
   the pod hardcoded the trade default and marked every ordinal preferred, while
   the pricing screen said a guessed percentage is worse than none. The pod reads
   the configured set now and answers UNKNOWN when it is unset.
3. **The rating currency is not the trade's currency.** RESEARCHED TO PRIMARY
   SOURCES 2026-08-09, `docs/research/israeli-rating-currency.md`. The trade
   settles on Jewish households and the round quarter-hour, both attested. The
   five modelled audiences do not include Jewish households and nothing records
   which rating vintage a held TVR is.

   **THIS FILE WAS CORRECTED AND THEN THE CORRECTION WAS CORRECTED, both on
   2026-08-09, and the second one is the interesting half.** I wrote "overnight
   plus one" here. A search of the measurement body's own publications found only
   OVERNIGHT, live plus deferred to 02:00, and no plus-one form in either
   language, so it was reported NOT CONFIRMED. Hours later a different agent
   found it attested in Hebrew in the buyers' own trade guide: „כולל צפייה נדחית
   של 24 שעות מרגע השידור (overnight +1)", in one sentence with Jewish households
   and the quarter-hour and the length factor.

   So the trade document was right all along, and the second reading is the true
   one: "+1" is a further 24 hours of deferred viewing, not the next morning's
   publication. **The lesson is the one worth carrying: a negative result from
   one set of sources is not a negative result.** The measurement body publishes
   what it MEASURES; the trade guide publishes what people PAY ON. Ours is the
   second question. Deferred viewing moves published figures by +0.1 to +8.1
   rating points, so the vintage is not a rounding question.

   The qualifier that stops this being misread: the PANEL is not Jewish-only. It
   measures television households excluding East Jerusalem and Gaza, and the
   Jewish cut is applied commercially on top.

## The engine, and the one number worth remembering

The last regression was **not a code bug**: one row in `data/manual_overrides.csv`
written by a browser walk. With it inert the engine reproduces the committed
golden byte-exact across all 120 channel-days.

Then the exact DP tier learned to plan WITH an operator's own overrides instead
of declining the whole channel-day. **Recovered 124,806.66 of 131,878.71.** The
residual 7,072.05 is real: the plan emits 3 gold breaks against a cap of 3.

**I proposed the wrong fix and the agent refused it with a runnable
counterexample.** I wanted to split the day at `_window_ends`; that closes the
three LOCAL guardrails and not the two DAILY ones, and the split doubles the
reported objective by shipping a plan that breaches the daily cap.

## Current handoff after closing waves 3 through 6

The broader campaign is no longer waiting at the end of wave 2. Wave 6's engine
regression was already closed by restoring the polluted override and reproducing
all 120 channel-days. Waves 3, 4 and 5 are now complete in code and verified; the
measured record is `audits/remaining-waves-completion-2026-08-10.md`.

The only stated dependency left from P1-P13 is external to the code waves: P13
needs a real House Number QC report and approved broadcast standards. Its schema,
importer, API, surface and server-side lock gate are complete, and the empty
owner-supplied files remain empty rather than carrying invented evidence.

The operational source-size threshold is now 500 lines. Files from 451 through
499 are not work merely because of their size; functional changes still take
precedence, and a file at or above 500 requires an explicit split decision.

## Closed today, in case a stale note sends you back to one

- The plan artifact was polluted a SECOND time, same 69-row shape, fingerprint
  re-stamped to match. Restored and the suite verified: 23 passed. Watch
  `git status output/` before trusting any engine measurement.
- My own "the plan breaches its hourly cap" was a BUCKETING ERROR, refuted with
  286,920.0 seconds identical both ways. Under the enforced unit: max 8.00
  minutes an hour, zero breaches of 713 hours.
- Underneath it: **the hourly minutes cap CANNOT BIND.** Break length is a
  hardcoded 120 seconds with no settings key, so 4 breaks an hour is an 8-minute
  hard ceiling under a 12-minute cap. Lowering it to the regulation's ten would
  change nothing. Decision 9.
- **Hour 24 exists** and no rule was written for it; 143 breaks land there.
  Named, measured, NOT changed, decision 9.
- `test_p5_pricing_draft` was the last caller of node's deprecated
  `--experimental-loader`, and it asserted against a STUB of a shell primitive.
  Both gone; it runs the real modules through the shared hook.

## What the owner ruled today

- **Israel is the only market that matters.** Foreign research is documentation;
  it becomes work only with Israeli evidence attached.
  `docs/audits/research-scope-ruling.md`.
- **`L` IS what the market uses for the last position.** The research had it NOT
  CONFIRMED. He works in this market and he outranks the search engine.
- **Regulatory caps are OFF BY DEFAULT**, because the commercial channels do not
  always work to the regulation, with the technical ability to turn them on. The
  value is not the product's to hold; being able to apply a rule and to say
  whether it did, is.
- **English-only names stay English.**

## Before writing any new wave script

`docs/ux-gauntlet/campaign-plan.md` now opens with an honest table of which of
its prescriptions are ENFORCED and which are only written down, plus a section on
what measurement did to its own ordering. `scripts/gauntlet/wave_preflight.py`
refuses a launch without a complete dossier and re-counts every line number in
it. `workflow-fixes.md` holds seven orchestration defects with the incident that
proved each.

Do not apply them by editing a running script: that invalidates the resume cache
for every agent in the run, which is the most expensive lesson this campaign has
paid for.
