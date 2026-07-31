# C2, the three-way on wave zero

Bar 3 run per piece against its regression row in section 8.5 and against the
frozen figures in `discovery/06-baseline.md`, whose provenance was settled on
2026-07-31 and whose numbers are therefore the correct floor.

The question here is not whether a piece met its own bar. Its critic judged
that. The question is whether anything a person can see, or any number anyone
reads, has changed. Verdicts are per piece, never in aggregate, because an
aggregate pass hides the one thing that broke.

Companion to `rulings.md`, which carries the test-change rulings.

## Method, and which half is which

The mechanical half is `scripts/gauntlet/verify_wave.py --reference 5a80a709`.
It materialises the reference with `git archive`, never a checkout, copies the
working tree because the suite writes into `data/`, and points every writable
store at throwaway space so neither side can contaminate the other. Run for this
report:

| Check | Verdict | What it measured |
|---|---|---|
| `api` | **pass**, 14.2 s | Every reference route intact plus two declared additions: 92 paths, 115 operations, 57 writes. Reference measures 90, 113, 56, matching the declared baseline |
| `engine` | **pass**, 219.7 s | Golden weekly schedule computed on both trees: CSV sha256 and aggregate sha256 identical on both sides |
| `moved` | **fail** | One data file changed that no piece declares. Discussed under "where the harness and I disagree" |

The half a script cannot do is reading each regression row and asking what a
person would notice. Every figure below that is not from the harness I measured
myself, and the command is named beside it.

## W0-1, router seams

Row: all 25 split routes return byte-identical bodies, and the response diff is
the bar rather than a smoke test.

| Capability the row promises | How I checked it | Verdict |
|---|---|---|
| The 25 routes still exist and still answer | Counted `@router` decorators per destination module | **Holds.** Exactly 25 across the fourteen split modules, and zero left in the four source files |
| No route changed shape | Harness `api`, which compares operation id, response model and parameters by identity rather than by count | **Holds.** 0 removed, 0 redefined, 0 changed in place |
| Bodies are byte-identical | 19 argument-free GETs from the contract's own route list, called on both trees with auth, versions and audit stores isolated, bodies hashed | **Holds, with two explained.** 17 of 19 byte-identical |

The 17 include the two largest payloads in the product, `/api/schedule` at
516,470 bytes and `/api/schedule/segments` at 610,149 bytes, plus
`/api/break-library` at 41,874 and `/api/break-operations` at 33,949. Payloads
that size matching to the byte is the strongest evidence available that the
split moved code and not behaviour.

The two that differ do so only in fields whose job is to report state:

- `/api/files`, 829 to 885 bytes. Identical file list, identical key set. Only
  `size` and `modified` move, and they move because `data/advertiser_rules.csv`
  legitimately grew by W0-3's three declared columns. A route that reports file
  sizes reporting a changed file size is that route working.
- `/api/overview`, 14,225 to 14,304 bytes. Identical key set. Only
  `data_freshness` and `schedule_freshness` differ, and the working tree reports
  `changed: ["settings", "coefficients", "advertiser rules"]`, which is the
  freshness engine correctly naming wave zero's own declared data changes.

**Verdict: passes.** No route lost, renamed, reshaped or altered in content.

## W0-2, shell seams

Row: every one of the 17 current routes renders the same DOM text after the
split, the drag in the schedule editor still moves a chip in 2.43 s
(`06-baseline.md:172`), and the `#Assistant` hash still opens the dock over the
current page.

| Capability the row promises | How I checked it | Verdict |
|---|---|---|
| The 17 routes still exist | Extracted the `navItems` array from both trees and diffed the sets | **Holds.** 17 on each side, identical set, no addition and no loss |
| The moved tree is coherent | Resolved every relative import against the filesystem across the moved tree | **Holds.** 321 of 321 imports resolve across 114 files, 0 unresolved |
| It still builds | `npm run build` in `tv-break-dashboard` | **Holds.** Built in 2.37 s with no error |
| The handover is complete | Listed the top level of `src/` | **Holds.** Only `index.jsx` and W0-4's two frozen files, `session.js` and `vocabulary.js`, remain, which is what the ownership table puts there |
| Drag still snaps at 30 s and 60 s | Read the surviving snap logic in the moved editor | **Holds.** Both sizes present in `plan/day/ScheduleEditorToolbar.jsx` and the snap calls in `plan/day/ScheduleEditor.jsx` |
| `#Assistant` opens the dock over the page | Read the hash handling in the moved shell | **Holds.** `shell/TVBreakDashboard.jsx:41-53` keeps the dock-not-a-page behaviour with its explanatory note |
| Every route renders the same DOM text | **Not verified** | See below |
| The drag still completes in 2.43 s | **Not verified** | See below |

**Verdict: passes on everything I could measure, with two items unverified and
named as such.** A DOM text diff and a stopwatch on the drag both require
driving two built apps in a browser, which the harness's `frontend` check exists
to do and which I did not get to run. The route inventory being identical, every
import resolving and the bundle building are necessary conditions, not
sufficient ones, and I am not going to report them as if they were the thing the
row asks for. This is the one piece whose row is not fully closed.

## W0-3, identity

Row: agencies still resolve 9 of 9 and still total gross 699,450, net 669,978
over 119 spots, and every engine figure stays byte-identical.

| Capability the row promises | How I checked it | Verdict |
|---|---|---|
| Agencies resolve 9 of 9 | `test_agencies_still_resolve_nine_of_nine` | **Holds** |
| Gross 699,450, net 669,978, 119 spots | `test_daily_ledger_totals_are_unchanged`, pinning `06-baseline.md:362` to the shekel | **Holds** |
| Every engine figure byte-identical | Harness `engine`, golden computed on both trees | **Holds.** Both hashes identical, 8,704 rows, 120 channel-days |
| An unknown advertiser is still allowed at premium 1.0 | `test_an_unknown_advertiser_is_still_allowed_at_premium_one` | **Holds.** The module's own honesty rule survives the re-keying |

Its whole suite is 22 passed. The data change is additive: three appended
columns, empty on every existing row, `advertiser_id` still first, and `key_for`
returns a stored key unchanged so no existing lookup moves.

**Verdict: passes.** This is the cleanest row in the wave.

## W0-4, the wall and the words

Row: the three event writes still refuse a channel account with the existing
Hebrew denial, `GET /api/events` still returns `can_edit`, and the five
`require_company_editor` call sites still fire.

| Capability the row promises | How I checked it | Verdict |
|---|---|---|
| The five call sites still fire | Diffed the call sites between trees | **Holds.** `events_api.py:378, 399, 426` and `pricing_api.py:234, 241`, same files, same line numbers, unchanged |
| The Hebrew denial is unchanged | `git diff` on `kairos_api/events_access.py` | **Holds.** The file is byte-unchanged, so the denial text and the gate logic cannot have moved |
| `can_edit` still returned | `test_stamp_writes_can_edit_and_the_reason_the_refusal_would_use`, plus `test_require_raises_403_with_the_hebrew_detail` | **Holds** |
| The surface grew by exactly one declared path | Harness `api` | **Holds.** `PUT /api/auth/job`, the one path the contract published |

**Verdict: passes on its row.** Nothing it was asked to preserve was lost.

But its row is narrower than what the piece is for, and the gap is the first
finding below. The wall it built guards nothing yet.

## W0-5, evaluation seam and cache

Row: no endpoint gets slower, the saved plan is byte-identical after the cache
lands, the extraction moves `overrides.py:244-302` and the five referring lines
and nothing else, and both effect endpoints return byte-identical bodies.

| Capability the row promises | How I checked it | Verdict |
|---|---|---|
| The extraction is exactly bounded | `git diff` on the two authorised files | **Holds.** `constraints.py` changes one docstring reference, two imports and two calls, the five referring lines the specification enumerates; `overrides.py` removes the function and rewrites its one import and one call |
| Both effect endpoints byte-identical | Called `/api/overrides/effect` and `/api/constraints/effect` on both trees for `רשת 13` / 2024-11-01, compared bodies | **Holds.** Byte-identical on both, and both report `before_revenue` and `after_revenue` of 1,067,845.55 |
| The saved plan is byte-identical | Harness `engine` | **Holds** |
| No endpoint gets slower | **Not separately timed by me** | The cache ships off, so the warm path is the pre-existing path |

One thing the row's wording does not prepare a reader for, and it deserves
stating because a later piece will build on this module. **The relocated
function is not a pure move.** It was refactored into `_seams`,
`_folded_assumptions` and `_build_segments`, and the read cache was added inside
it. That is legitimate: `preview_inputs.py` is W0-5's own created file and the
cache is its declared subject, and the bounded authorisation covers
`overrides.py` and `constraints.py`, which are exactly bounded. It is also
correct: `USE_READ_CACHE` is `False`, so the shipped path calls `build()`
directly, and `_build_segments` preserves the load-bearing ordering, loading the
impact model on raw assumptions and using the already-folded assumptions for the
segments, exactly as the original did and with a comment saying why reversing it
would change every retention number.

**Verdict: passes.** With the note that the cache, not the move, is what a
future reader should re-examine, and only if it is ever switched on.

## The product-wide row: five honest empty states

`06-baseline.md:445` recorded five places the product refuses to fabricate, and
section 8.5 makes their survival a Bar 3 row of its own. Each must remain a
control with a path forward and must never become a figure.

| Empty state | Verdict and evidence |
|---|---|
| Make-good panel names the missing file | **Survives.** `/api/make-good-alerts` returns `data_available: false` with `campaign_flights.csv has no campaign rows yet (header-only seed).` |
| Gold-breaks panel says none are configured | **Survives.** `/api/gold-breaks` returns `count: 0` with `No gold breaks in the current plan (none configured as gold in overrides).` |
| Net after retention cost reads "Not exposed" | **Survives.** `plan/week/ScenarioCompare.jsx:200`, bilingual, moved with its component |
| Campaign revenue dashes with the reason | **Survives.** `/api/campaigns` returns `revenue: null` per campaign with `revenue_available: false`, and `clients/CampaignsPage.jsx:9` reads that flag |
| Advertiser revenue null with the reason | **Survives.** `/api/advertisers/stats` still carries `revenue_note` |

**Verdict: passes.** All five survived the move, and two of them survived it as
byte-identical responses.

## Finding 1: the operator channel picker lists three rival channels

**Measured.** `GET /api/parameters` returns `operator_channel: "רשת 13"` and
`available_channels: ["כאן 11", "עכשיו 14", "קשת 12", "רשת 13"]`.
`rules/OperatorChannelPanel.jsx:44` renders every one of them as a selectable
option, so three rival channels are named on an operator surface.

**My reading: this is not the competitor-boundary breach it looks like, and
there is a real defect next to it.**

The picker is the surface where the operator declares which channel is theirs.
Before that declaration the product cannot know which of the four to hide, so
the list has to contain all four or the choice cannot be made. What the boundary
protects is rival figures and rival schedules, and this exposes neither: four
public broadcast names and no number attached to any of them. `_channel_options`
derives the list from the loaded EPG precisely so the picker cannot drift from
the ids the optimizer schedules on.

The real defect is that **nothing validates the choice**. `PUT /api/settings`
(`settings_api.py:46`) carries no guard, no affiliation check and no validation
of `operator_channel`. Anyone who can reach it can set the operator's channel to
a rival's. That does not leak rival data by itself; it does something stranger
and worse, which is invert the boundary, so the product would then hide the
operator's own channel and treat a competitor's as owned. The picker is not the
bug. The unvalidated, unguarded write behind it is.

**Owner: P5.** The wave-one table gives P5 the Rules workspace, which is where
`OperatorChannelPanel` lives (`src/rules/**`), and P5 already owns the
guardrail and activation stores W0-4 handed over. The fix is validation and a
permission on the write path, not a change to what the picker lists.

**Not fixed here**, per instruction, and it is not a wave-zero regression: the
same route was equally unguarded at 5a80a709.

## Finding 2: the affiliation wall guards no route

**Measured.** `@guard()` and `company_only()` appear on **zero routes**. The
only importers of `kairos_api/affiliation_wall.py` are `guardrail_store.py` and
`model_activation.py`, and both of those declare **zero routes** of their own.
The two model routes that exist today, `GET /api/model/audience`
(`model_audience_api.py:20`) and `GET /api/impact`
(`model_impact_api.py:175`), take no `request` parameter at all, so no
affiliation check is even reachable on them.

So the observation that prompted this is right, and the reason is more specific
than "a test is missing". **No test asserts that an unaffiliated account cannot
reach a route under models because there is no such assertion to make yet.** The
wall is a well-built and well-tested primitive, with 9 of its own passing tests
covering `can_edit`, the Hebrew 403 and the signature handling, and it is wired
to nothing. A test written today would have to assert that an unguarded route is
unguarded.

**This is not a W0-4 regression.** Its row asks it to preserve the event writes
and the five existing call sites, and it did. Section 8.2 has W0-4 create the
primitive as frozen and hand the stores to P5, so the wiring was always
somebody else's wave.

**Owners, both named by the wave-one table.** **P7** owns
`model_audience_api.py`, `model_impact_api.py`, `audience_api.py` and the new
`model_console_api.py`, so applying the wall to the model surface and asserting
the refusal is P7's. **P5** owns `model_activation.py`, `guardrail_store.py` and
`data/regulatory_guardrails.json`, so the company-only permission the owner
approved for the guardrail store is P5's. Neither can cite W0-4's unit tests as
coverage: those prove the primitive works, not that a door is locked.

## Where the harness and I disagree

The harness fails `moved` on one file and I do not, and the disagreement is
worth more than either verdict alone.

It reports `data/kairos_settings.json` as **UNEXPLAINED, keys added:
audience_model_activation=False**, because no wave-zero piece declares writing
that path, and it is right that none does.

My reading is that the content is inert. `audience_model_activation` is not new:
it is defined at `kairos_api/core.py:143` and section 4.5 of the specification
already cites that line as existing. The value written is `False`, which is the
model's own default, and the golden schedule is byte-identical on both trees,
which is the strongest available proof that nothing downstream moved. What
happened is that the settings store rewrote the full model and materialised a
default that had previously been implicit.

Both are correct at different levels, and the useful conclusion is neither
verdict. **A settings store that the running product rewrites is checked into
git**, so it will keep producing diff noise that a gate must classify by hand,
and one day that noise will hide something real. The harness is behaving
correctly by refusing to wave it through. Whether that file should be tracked at
all is a decision for the lead, not a defect in any piece.

## Wave-zero verdict

Four of the five pieces pass their regression row on evidence I gathered. W0-2
passes everything I could measure and carries two unverified items, the DOM text
diff across the 17 routes and the 2.43 s drag, both of which need the browser
half of the harness. Neither finding above is a wave-zero regression: the
channel write was equally unguarded at the reference, and the wall was never
wave zero's to wire.
