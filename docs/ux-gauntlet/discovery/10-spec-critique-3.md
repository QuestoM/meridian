# Third blind critique of the specification

I wrote neither `docs/ux-gauntlet/spec.md` nor either earlier critique. I read
`discovery/09-spec-critique-2.md` first to learn which gaps had to close, then the
specification, the frozen job stories, the owner decisions and the brief. Then I
went to the repository and to the running instance at `http://127.0.0.1:8010` and
measured everything myself. Where I inferred, it says INFERRED and from what.
Where my number differs from the specification's I give both methods and say which
is right.

## Method, and the tree I measured

`git rev-parse HEAD` returns `eef9ff91151de697296dac0f5860033b7d972f0a`, which is
the HEAD the specification's provenance line names. `git diff --name-only
5a80a709..eef9ff91` filtered to anything outside `docs/ux-gauntlet/` returns
nothing, so the provenance claim holds and every code measurement in the document
is measurable at this HEAD. I confirmed that myself rather than trusting it.

Three files are dirty in the working tree: `docs/ux-gauntlet/spec.md`,
`docs/ux-gauntlet/job-stories.md` and `data/kairos_settings.json`. The settings
diff is a single added key, `"audience_model_activation": false`. INFERRED, from
the timing and from section 4.5's subject: somebody threw that switch through the
live app during discovery. It is harmless, because `kairos_api/core.py:143`
declares `audience_model_activation: bool = False` as the default and the comment
at `core.py:139-142` says an absent key reads False and keeps the transform
byte-identical. So no measurement moves. It should still be recorded, because the
running instance every builder and critic will measure against is one uncommitted
write ahead of HEAD on a file the engine reads.

Python is `~/.venvs/meridian/bin/python`. Every count below is from a command I
ran in this session.

---

## 1. Are the six gaps closed

All nine rows of the specification's own closure table (section 11, lines
1796-1806) are closed in substance, and I verified each against the code rather
than against the table. The header above that table says "Six things remained" and
then lists nine rows; the arithmetic disagrees with itself, which is the smallest
finding in this document and the only one about that table.

| Gap the second critique left open | Verdict | What I checked |
|---|---|---|
| `tests/` named nowhere while Bar 4 requires the suite green | **Closed** | Section 8.2's "Tests" subsection. Measured: 125 Python files under `tests/`, 122 at the top level and exactly 3 under `tests/validation/`; `pytest --collect-only -q` returns **3102 tests collected**. Both figures reproduce. The reserved-prefix rule removes the collision, the derived-owner rule removes the table, and the four-part change protocol routes to C2. The one number inside it does not reproduce; see section 2 |
| `server.py` owned by W0-1, never handed over, four later routers | **Closed** | Measured: **20** `app.include_router(...)` calls, the first at `server.py:112` and the last at `:271`, exactly as stated. Frozen above a marker plus a two-line append below it is the right shape, and the bar is an OpenAPI diff I can run: live `openapi.json` gives **90 paths, 113 operations, 56 writes**, which is the baseline the subsection states |
| Six of the 51 `kairos_api` modules absent, five on wave-1 critical paths | **Closed, completely** | I diffed every backticked path in section 8 against the real file list by basename. **51 of 51 `kairos_api` modules are now named.** Line counts reproduce: `_constraint_options.py` 249, `audience_api.py` 114, `jobs.py` 114, `condition_validation.py` 74, `events_holidays.py` 35, `__init__.py` 2. Every cited import site is correct: `constraints.py:44`, `scenario_api.py:425`, `assistant_propose_tools.py:44`, `insights_api.py:638`, `core.py:452`, `assistant_audience_model.py:25`, `recompute_api.py:91,126,159`, `advertiser_conditions.py:44`, `agency_conditions.py:45`, `events_api.py:51` |
| The 450-line law will force helpers nobody owns | **Closed for the backend, open for the frontend** | The naming convention plus the 8.8 publication as the collision check is a sound rule. The six over-cap owned files reproduce exactly: `assistant.py` 775, `uploads.py` 713, `core.py` 698, `assistant_actions.py` 496, `scenario_api.py` 477, `overrides.py` 467. But "the frontend needs no rule" is false; see defect N1 |
| `preview_inputs.py` had two wave-0 claimants and neither could build it | **Closed, and this is the best-executed fix in the pass** | Every coordinate is right. `_preview_inputs` is at `kairos_api/overrides.py:244`, its `return segments, engine_kwargs` is at `:302` (59 lines), it is called at `:392`, named in a docstring at `constraints.py:324`, imported at `:328` and `:352` and called at `:330` and `:355`. A repository-wide grep for `_preview_inputs` under `tests/` returns nothing, so the rename to a public name is safe as claimed. The authorisation names the exact lines and the bar is a diff |
| `audience_model_activation` filed on the training side | **Closed, and consistently** | `kairos_api/core.py:143` is `audience_model_activation: bool = False`, and the comment the specification quotes is verbatim at `core.py:139-142`. The switch is now configuration on Rules in 4.1, 4.5 and 8.3, both stores hand to P5 and not to P7, and P7 owns a read-only mirror. The precedent it copies is real: `require_company_editor` has exactly **five** call sites, `events_api.py:378,399,426` and `pricing_api.py:234,241` |
| The upload-staleness closure claim pointed at absent text | **Closed, and the measurement is honest** | `models/tv_break_coefficients.json` carries `metadata.source_fingerprints` for exactly `data/reference/Spots.xlsx`, `Programmes.xlsx` and `Dayparts.xlsx`. `GET /api/uploads/status` returns 7 kinds; `programmes`, `spots` and `dayparts` all read `in_use: false` with the reason "the engine reads data/reference/...", at 3562, 50386 and 43200 rows. So "today an operator upload cannot make the model stale" is true and provable, and the one condition under which it changes is stated |
| Two measured claims that did not reproduce, plus a data defect | **Closed, and every corrected number reproduces on my own implementation** | See section 2. This is the strongest part of the pass |
| The revenue and yield owner's door did not hold their answer | **Closed** | `yield_api.py` is W0-1's create tagged `[P5]` and appears in P5's write column; P5's Bar 3 row carries the endpoint; job stories amendment 4 records that JS-13's "Money" reads as Rules, the rate card, with the 45 s target unmoved |

### The ownership diff the task asked for

I extracted every backticked path from section 8 and diffed it by exact basename
against the real file list.

| Tree | Files | Named in section 8 | Unowned by absence |
|---|---|---|---|
| `kairos_api/*.py` | 51 | 51 | **0** |
| `kairos/**/*.py` | 72 | 42 | 30, all engine internals the document intends to freeze |
| `tv-break-dashboard/src/*` | 80 | 65 by the `src/*.jsx` and `src/*.js` globs | **15, every CSS file** |
| `scripts/*.py` | 17 | 0 plus 2 created | 17, intended |
| `tests/**` | 125 | governed by the new rule | 0 |

**Double ownership: none.** I parsed the wave tables and looked for any path in
two different pieces' write or create columns. Twenty-six paths appear more than
once and every one is a declared pattern: W0-1 or W0-2 creates and a wave-1 piece
owns, W0-5's bounded authorisation over `overrides.py` and `constraints.py`, or
P3 handing `break_api.py` and `src/plan/break/**` to P10. **No two pieces in the
same wave claim the same path.** The wave-0 disjointness the second critique broke
is genuinely restored, and rule 8.0.1 is now stated correctly as one owner rather
than one mention.

**Frozen while a piece needs it: three cases, all small, all new.** They are N1,
N2 and N5 in section 3.

---

## 2. The corrected numbers, re-measured

Every number the pass corrected, I reimplemented from the specification's stated
method and ran. I did not copy a script from anywhere.

| Corrected claim | Spec | My measurement | Method |
|---|---|---|---|
| Version-store pollution | 200 manifests, **186** carry a pytest path, **188** point outside the repository, 2 more carry another temp path | 200 manifests, 201 directory entries, **186** pytest, **188** outside, **12** inside, **2** under `/var/folders/.../T/tmp*` | Parsed every `data/versions/*/manifest.json`, resolved each `files[].path` with `os.path.abspath` and tested `startswith(repo_root)` |
| Daily-file declared grouping | 10 groups, sizes 1, 1, 3, 3, 7, 28, 29, 30, 35, 38 | identical | `groupby('שעת התחלת ברייק')` on all 175 rows, all of which parse |
| The 60 s gap partition | 10 groups, sizes 1, 3, 4, 7, 7, 22, 28, 30, 35, 38 | identical | end equals `שעה` plus `אורך תשדיר`, sorted by start, split where the gap exceeds 60 s |
| Counterexample one | `21:22:12` holds one 32 s spot starting 21:22:16, and the next declared break `21:23:10` begins 22 s later | identical, gap is 21:23:10 minus 21:22:48 | as above |
| Counterexample two | `22:59:40` holds 29 spots, a 93 s internal gap before the spot at 23:12:58, span 642 s against 432 s of ad time | identical on all four figures | as above |
| The 38-spot group | 22:04:16 to 22:18:12, 836 s, **13.93 minutes** | identical | as above. The owner document's 22:18:06 and 13.4 minutes are both wrong |
| `Spots.csv` within-break contiguity | 15,614 pairs, **0** above 60 s, structural because 18,669 minus 3,055 | identical, and I confirmed the structural identity: the 1,145 unparsed rows resolve to 1,145 distinct `break_id` values and every one is a singleton, so dropping them leaves 17,524 rows over 1,910 breaks, which is the same 15,614 | `Channel == רשת 13`, `Date` plus `Start time` as `%d/%m/%Y %H:%M:%S`, end equals start plus `Duration`, sorted by date then start then `position_in_break` |
| `Spots.csv` boundaries, second resolution | 1,880 boundaries, 625 at 60 s or less, **33.24 percent** | identical to two decimals | as above |
| The other three channels | `קשת 12` 2,185 at 23.34 percent, `כאן 11` 892 at 4.15 percent, `עכשיו 14` 1,153 at 50.48 percent | identical on all six figures | as above |
| `break_id` never interleaves | 3,055 runs for 3,055 ids | identical, **after I used the stated method**. My first attempt sorted on the file's own `Start_dt` column, which parses only 6,252 of 18,669 rows, and returned 13,155 runs. That was my error, not the document's, and it is worth recording because a builder who reaches for `Start_dt` will get the same wrong answer | sorted each date by the parsed start then `position_in_break`, counted runs of equal `break_id` |
| `Spots.csv` channel flags | `competitor_flag` True on 26,679 and False on 23,707; the False side is `קשת 12` alone; `is_target_channel` the exact complement; `include_as_media` True only for `קשת 12` | identical, by crosstab. Channel counts `קשת 12` 23,707, `רשת 13` 18,669, `כאן 11` 4,039, `עכשיו 14` 3,971, and `data/kairos_settings.json` sets `operator_channel: רשת 13`, so the file does mark the operator as the competitor | `pd.crosstab(Channel, flag)` |
| `revenue_ils` is not money | constant `base_rate` 50, constant `adv_premium` 1, sums to ₪306,936,788 | identical, sum is 306,936,788.47 | pandas `unique()` and `sum()` |
| Grounding sections | **11** at `assistant_keywords.py:354` | `len(_SECTIONS)` is 11, imported in process | import and count |
| `assistant*.py` | **22** modules | 22 | glob |
| `kairos_api` package | **51** modules | 51 | glob |
| Test corpus | 125 files, 122 plus 3, 3,102 collected | identical | `find` and `pytest --collect-only -q` |
| Live API surface | 90 paths, 113 operations, 56 writes, zero campaign writes | identical | parsed live `openapi.json` |
| Agency money | gross ₪699,450, net ₪669,978, 119 priced, 56 dropped | `GET /api/export/spots.csv`: 175 rows, `priced` 119, `dropped_frequency` 56, revenue 699450.0, net_revenue 669978.0 | fetched and summed with pandas |

**One corrected number does not reproduce, and it is inside the subsection written
to close gap 1.** Section 8.2 says "16 of the 125 files resolve to exactly one
owner (W0-1 six, P9 five, W0-4 three, W0-3 two), 106 span owners or reach a frozen
path, and 3 import no production module at all". My method: build the owner map
from the section 8 tables, regex every `from|import (kairos_api|kairos|scripts)…`
in each test file, resolve each module to a path, and classify. I get **17 single
owner (W0-1 six, P9 six, W0-4 three, W0-3 two), 105 shared or frozen, 3 with no
production import**. The three harness files match the specification exactly
(`tests/conftest.py`, `tests/test_rebuild_equivalence.py`,
`tests/validation/test_placebo_fast.py`), which tells me our methods are close.
The difference is one file in the P9 bucket. I cannot say which of us is right,
because **no method is stated for this count**, and that is the point: it is the
one number in the new subsection that a critic cannot reproduce. The conclusion
does not move, since 105 or 106 shared files both make the change protocol the
load-bearing part.

**One stated critic check carries a baseline that reproduces under no variant I
tried.** Section 8.3 says the vocabulary grep "today returns 159 hits" for
`recompute`, `rebuild`, `חישוב מחדש` and `בנייה מחדש` under
`tv-break-dashboard/src/`. Measured four ways: case-sensitive occurrences 127,
case-insensitive occurrences **198**, matching lines 141, matching files 21.
`recompute` alone case-insensitive is 162. 159 is `04-training-vs-runs.md`'s count
for one word in the UI, and section 10 says so, but section 8.3 attaches it to a
four-term grep. A critic who runs the stated command and compares to the stated
baseline will think something changed when nothing has.

---

## 3. New defects this pass introduced

**N1. Every CSS file in the product is frozen by absence, and the piece that
must ship the design tokens may not touch any of them.** Measured:
`tv-break-dashboard/src/` holds 80 files, 52 `.jsx`, 13 `.js` and **15 `.css`,
9,306 lines, of which `styles.css` alone is 6,170**. W0-2's write column is
"every existing `src/*.jsx` and `src/*.js` during the move only", which covers 65
of the 80. Section 8.8 then names "the design tokens and shell from W0-2" as one
of the five shared surfaces, and section 3.5 justifies four rows by "the design
tokens". No CSS path appears anywhere in the specification; the words `css` and
`design token` appear on exactly two lines, 343 and 1635, neither naming a file.
Section 8.2 also asserts "the frontend needs no rule … this is a backend problem
only", which is false for the 15 files that match neither the two globs nor any
destination tree. Two consequences: `styles.css` at 6,170 lines is the second
largest file in the repository and breaches the 450-line law that the same
subsection applies to Python, and a builder that cannot edit it will either
escalate or, worse, duplicate tokens into per-tree CSS inside its own glob, which
fragments the design system silently. This is the same defect class the pass just
closed for `kairos_api`, left open one directory over.

**N2. Section 5.6 charters P5 to create a store whose path is frozen.** "The new
store is created explicitly rather than implicitly … so P5 creates the store and
changes no reader" (spec.md:1020-1027). `data/kairos_constraints.csv` appears
twice in the document, at `:1020` and `:1732`, and never in section 8. P5's write
column is `compliance_api.py`, `yield_api.py`, `constraints.py`,
`_constraint_options.py`, `pricing_api.py`, `events_api.py`, `events_holidays.py`,
`model_activation.py`, `guardrail_store.py`, `data/regulatory_guardrails.json`,
`data/frequency_rules.csv`. Every other piece that must create a data file has it
listed, `data/plan_targets.csv`, `data/breaks.csv`, `data/campaigns.csv`,
`data/make_goods.csv`, `data/media_assets.csv`, `data/advertiser_names.csv`, so
the omission reads as an oversight rather than a convention.

**N3. The arithmetic in that same paragraph is off by one.** It says "five of
those seven modules are frozen". Of the seven referencing modules, only
`kairos_api/constraints.py` has an owner (P5); `kairos/service.py` is on the
explicit frozen list and `kairos/optimize/constraints_store.py`,
`kairos/optimize/_constraints_io.py`, `kairos/export/schedule.py`,
`kairos/export/schedule_freshness.py` and `scripts/export_schedule.py` are frozen
by absence, which I confirmed by the basename diff. Six of seven, not five.

**N4. The 187 that the pass declares non-reproducible is still asserted in two
other places.** Section 5.6 corrects it to 186 and 188 with a method, and section
10 adds a row saying "187 reproduces under neither method". Section 9 item 10
still reads "187 of 200 entries point at pytest temporary paths" (spec.md:1675)
and the older section 10 row still reads "The load-bearing number is that 187 of
them point at pytest paths" (spec.md:1726). So the document asserts 187 twice,
denies it twice, and calls it load-bearing in one of the assertions. My
measurement says 186 and 188 as the correction states.

**N5. The rule-grammar consolidation has no owner for its engine-side readers.**
Section 5.6 says "the five rule grammars collapse to two", a placement rule and a
commercial rule folding advertiser conditions, agency conditions and frequency
rules into one scope grammar. The API halves are owned (W0-3, P4, P5). The engine
halves are frozen by absence: `kairos/optimize/_rule_models.py`,
`_rule_helpers.py`, `_frequency_rules.py`, `constraints_store.py` and
`_constraints_io.py` appear nowhere in section 8. If the consolidation is
schema-only above the engine, that is fine and should be said in one clause; if a
store's shape changes, five frozen modules read it. Section 5.6 also promises
`frequency_rules.csv` becomes "a visible, editable commercial rule", and its
enforcement lives in `kairos/optimize/frequency.py`, which is frozen by absence.

**N6. Two findings the second critique named and measured are neither closed nor
recorded as deferred, while the provenance line says the pass "closed the six
items the second blind critique left open".** The closure table itself is honest
about what it did, which is a real improvement over the previous pass's false
blur-3 claim, but the opening sentence is broader than the table. The two are:

- `/api/parameters`. Section 8.3 still says the four open reads are closed by
  "P7, P7, **W0-4**, P5". Measured: the route is `@router.get("/api/parameters")`
  at `kairos_api/scenario_api.py:388`, and `scenario_api.py` is **P2**'s
  (spec.md:1331). W0-4 cannot write it. I fetched the endpoint: it returns
  `channels: ['קשת 12', 'רשת 13', 'כאן 11', 'עכשיו 14']` and
  `available_channels` with the same four, beside `operator_channel: רשת 13` and
  a `coefficient_freshness` object. `TVBreakDashboard.jsx:1296` fetches it on
  every page load and `:5214` renders `available_channels` in the operator
  channel panel. So three rival names reach an operator surface today, the
  competitor-boundary row of section 8.3 names only `/api/schedule` and
  `/api/break-operations`, and the closure is assigned to a piece that cannot
  perform it. There is a real design question underneath, which is whether the
  channel picker may list the channels you are not, and the document does not
  answer it.
- The per-piece acceptance bar. Section 8 spans lines 1217 to 1644, 428 lines,
  and mentions a job story on exactly **3** of them, two inside the latency
  subsection and one in C1's remit. The wave-2 table has a "Depends on" column
  and the wave-1 table still does not. I weigh this lighter than the second
  critique did, because a wave-1 builder can join section 8.5's regression row to
  section 3.5's "Traced to" column to the frozen story's own target, and because
  the wave-1 pieces are genuinely file-disjoint so the missing dependency column
  may be empty by construction. It is a join the document could have done once
  and made every builder do instead.

**N7. `docs/ux-gauntlet/decisions-for-owner.md` is unchanged and still carries
three measurements the specification itself refutes.** The specification says so
at `:940-944` and again at `:1729`, and says they "must be corrected there before
that document goes to the owner". Nobody has corrected them. The file still reads,
at `decisions-for-owner.md:62-69`: "13.4 minutes of unbroken commercial time"
(measured 13.93, and the end time 22:18:06 is measured 22:18:12); "A gap rule at
60 seconds reproduces those same 10 groups exactly, so the file is internally
consistent and the groups really are contiguous" (measured: 10 groups, a different
partition, sizes 1, 3, 4, 7, 7, 22, 28, 30, 35, 38 against declared 1, 1, 3, 3, 7,
28, 29, 30, 35, 38, and the declared break at `22:59:40` has a 93 s hole in it);
"only 2 of 15,214 consecutive gaps exceed 60 seconds" and "702 of 2,412 break
boundaries, 29.1 percent" (measured 15,614 pairs with 0 above 60 s, and 1,880
boundaries with 625 at 60 s or less). This is the one document in the set that
goes to a human and asks him to choose. His decision does not change, because the
recommendation is the explicit break identifier and the refutation strengthens it,
but the evidence sentences under the recommendation are false and the document
that knows it cannot edit them.

**N8. The test-prefix rule contradicts its own example.** Section 8.2 says the
prefix is "the piece id lowercased and **hyphens dropped**" and then gives
`tests/test_w0_1_route_identity.py`. `W0-1` with hyphens dropped is `w01`, not
`w0_1`. Harmless for collisions either way, since both forms are unique per piece,
but it is a guess inside the rule written to remove guessing.

---

## 4. Can the build loop start

Yes for four of the five wave-0 pieces, and the fifth needs one sentence. I took
the first piece concretely, then checked the two that carry the most unstated
work.

**W0-1, router seams.**

- *Files it may write*: yes. Five named files, fourteen created modules with the
  final owner in brackets, its own test prefix, and `server.py` with the marker
  it adds as its last act.
- *Its bar*: yes, and it is checkable. All 25 split routes return byte-identical
  bodies, plus an OpenAPI diff against 90 paths and 113 operations. I enumerated
  the routes to confirm the 25 exists: `dashboard_api.py` 8 at `:1633, :1638,
  :1695, :1711, :1728, :1803, :1817, :1823`; `insights_api.py` 5 at `:400, :518,
  :622, :630, :646`; `catalog_api.py` 7 at `:571, :578, :591, :596, :605, :620,
  :641`; `version_store.py` 5 at `:420, :429, :445, :468, :479`. Eight plus five
  plus seven plus five is 25.
- *Its reference*: not applicable, and correctly so. This is a backend split with
  no surface.
- *Its regression floor*: yes, section 8.5's row, and it is the same as the bar.
- *What it must not touch*: yes, the frozen list plus the closure rule.
- *What it would have to guess*: **two things.** First, **which of the 25 routes
  goes into which of the 13 route-carrying modules.** The create column names the
  modules and their eventual owners, and three routes are pinned elsewhere
  (`/api/schedule` to P2 and `/api/break-operations` to P3 by the
  competitor-boundary row, `/api/yield-per-second` to P5 by section 2), but at
  least five have no named destination: `/api/inventory`, `/api/break-library`,
  `/api/forecasts` and both `/api/break-decisions` operations, one of which is a
  POST and therefore decides who owns a write for the whole run. A wrong guess
  hands a route to a piece that must not own it, which is the exact collision
  class section 8 exists to prevent. It is recoverable, because wave 0 closes
  before wave 1 opens, but it is a decision the document should make rather than
  the first builder. Second, **what `plan_read.py` contains.** It is created by
  W0-1, marked `[frozen]`, and named in section 8.8 as one of three frozen shared
  read layers, and its contents are specified nowhere. The same is true of
  `spot_ledger.py`, created by W0-3 and frozen at wave-0 close. Only
  `preview_inputs.py` has its contents fixed, by the extraction. A frozen
  interface whose surface nobody specified is a wave-1 escalation waiting to
  happen, and the remedy is already in the document's own machinery: publish the
  interface under section 8.8 before freezing it.

**W0-2, shell seams.** Files, bar and floor are all present, and the bar is
measurable in a browser (17 routes, same DOM text, the 2.43 s drag, the
`#Assistant` hash). It would have to guess whether it may touch any of the 15 CSS
files, where the design tokens it owes section 8.8 are to live, and what to do
with `styles.css` at 6,170 lines. That is N1, and it is one sentence to fix.

**W0-4, the wall and the words.** Files, bar and floor present, and its Bar 3 row
is precise and true: the three event writes at `events_api.py:378,399,426` and the
two at `pricing_api.py:234,241` are exactly the five call sites. It hits N6 on day
one, when it opens the file `/api/parameters` lives in and finds it belongs to P2.
Rule 8.0.1 tells it what to do, so this costs an escalation rather than wrong
code. Separately, and unaddressed since the second critique named it,
`kairos_api/auth_store.py:175` is `normalize_affiliation`, whose own docstring
says "Missing, empty or unrecognized values read as company (the permissive legacy
default)". Section 2.1 still cites that function as "the pattern to copy: a field
that defaults safely for every existing record". It is safe for `job`, where
`unset` costs a good first screen. It is the open side of the wall for
`affiliation`, where section 4.7 renders the Model console switcher and section
4.5 gates every training route on the same value. W0-4 is not told to change it
and no piece is told to test it. INFERRED consequence, from the docstring: an
account with a missing or misspelled affiliation is shown the training side.

**W0-3 and W0-5 are clean.** W0-3's floor reproduces to the shekel (I fetched the
ledger: 175 rows, 119 priced, 56 dropped, 699450.0 and 669978.0) and the document
says explicitly it may start before owner decision 1 lands because both options
bind the same 41 names. W0-5 is the best-specified piece in the document: exact
lines to move, exact call sites to rewrite, a diff as the bar, and a first
deliverable that is an attribution rather than a fix, which is the right answer to
13 unattributed seconds.

**One prerequisite that belongs to the run rather than the specification.** The
brief requires a live progress page at `docs/ux-gauntlet/workbench.html`, and
says the frozen stories and the proposed architecture go on it "before the first
builder starts" (`docs/ux-gauntlet-prompt.md:284-290, :337-339`). It does not
exist; `docs/ux-gauntlet/` holds `decisions-for-owner.md`, `job-stories.md`,
`spec.md` and `discovery/`. The specification never mentions it and assigns it no
owner. Related: the brief says "Every piece gets its own builder and its own
critic" with rounds until the blind comparison stops picking the reference, while
section 8.6 names three critics, C1, C2 and C3, all cross-cutting. Neither is a
specification defect, both are things the orchestrator must supply before wave 0
opens.

---

## 5. Honesty scan

**Nothing is fabricated, and I looked hard.** Every persona grade I spot-checked
carries its source line, every live figure I fetched matched, and the document
repeatedly corrects itself against its own earlier claims in public. The pass
under review found and printed its own errors on the daily file, on `Spots.csv`,
on the version store, on the module counts and on the agreements module, which is
the behaviour you want and the opposite of what a document does when it is
protecting itself.

Numbers without a reproducible method, after this pass:

- The test-owner split, "16 of the 125 … 106 span owners". I get 17 and 105 and no
  method is stated. Section 2.
- "Today that grep returns 159 hits" in section 8.3, measured 198 occurrences over
  141 lines in 21 files, or 127 occurrences case-sensitive. Section 2.
- "187 of 200" surviving at `:1675` and `:1726` against the corrected 186 and 188
  at `:1031-1036`. Defect N4.
- "five of those seven modules are frozen", measured six. Defect N3.
- Section 3.2 says Sources' eight source files are each "a record with a state, a
  row count and an `in_use` verdict". I fetched `GET /api/files`: eight records
  whose keys are exactly `path, exists, size, modified`. The state, the row count
  and the verdict are on `GET /api/uploads/status`, a different endpoint over
  seven upload kinds. The second critique named this and it is unchanged. It is
  cosmetic for a reader and misleading for a builder sizing the work.

Capability asserted beyond what the data supports: **none that I found.** The
three places it would be easy are all handled honestly, and one of them is
handled better than honestly. Section 3.4's limit on projected versus delivered is
stated in the section that would benefit from hiding it. Section 9 item 13 refuses
₪306.9M as money and I verified the formula that justifies it. Section 5.5's bar
is 41 of 41 because 41 is what exists.

Things asserted that I could not verify: one, and it is the specification's own
subject. Section 3.4's rule, "A break carries exactly two money quantities,
projected and delivered … A figure that cannot resolve to breaks does not render
as money", is contradicted by the product as it runs. I fetched
`GET /api/yield-per-second`: `revenue_ils` 40944759.33, `retention_cost_ils`
4145199.27, `revenue_net_ils` 36799560.06, `retention_cost_low` 2204684.36 and
`retention_cost_high` 6088227.95, all ILS, with a `basis.formula` naming every
input. The specification's own entity model at `:786-787` says the break carries
"projected money, delivered money, retention cost", which is three, and its own
P4 regression row at `:1584` requires agency records to keep "credit limit", which
is money that cannot resolve to breaks and which `agencies.csv` carries as
`credit_limit_ils`. The second critique raised this and the pass did not address
it. The layer claim is right and is the sharpest idea in the document; the word
"exactly two" is wrong and puts a builder's Bar 3 row in conflict with a Bar 4
style rule. Likewise section 4.2's lexicon test forbids the string `coefficient`
in any run surface's read payload, while section 5.6 requires Sources to name
`models/tv_break_coefficients.json` as the artifact that is read. Both are still
open and both were named before.

---

## Verdict

**READY.**

I did not expect to write that. I came in with two prior critics' findings and I
re-ran every corrected measurement myself, including the ones nobody has been able
to reproduce twice, and they all landed: the daily file's two partitions to the
spot, the 15,614 within-break pairs and the four channels' boundary rates to two
decimals, 186 and 188 manifests, 51 modules with zero now unowned, 3,102 tests,
125 test files, 20 router mounts, 90 paths, and `_preview_inputs` at exactly the
lines the authorisation names. The one number I could not reproduce, the test-owner
split, differs by a single file in a single bucket and changes no conclusion.

More importantly, the structural failure that made the second critique fail is
gone and I proved it rather than accepting it. Wave 0 is mutually disjoint by
measurement, not by assertion: no two pieces in one wave claim a path, every
cross-wave repeat is a declared hand-over, the one file that had two claimants and
no builder now has one owner with a line-exact authorisation and a diff for a bar,
and the closure rule no longer freezes the test suite, the router registrations or
five modules on the critical path. Every wave-0 builder can open an editor
tomorrow knowing its files, its bar, its floor and its frozen set.

What is left is a short list of one-line corrections and two escalations that the
document's own rules already route correctly. None of it produces a wrong number,
a fabricated screen, a lost capability or a broken engine, which are the four
things a specification can do that a later round cannot undo. Failing this pass
would buy corrections that a builder will surface on day one anyway, at the cost
of a third full cycle on a document that is now the most heavily measured artifact
in this repository.

**Two things I would fix before the first builder opens an editor**, because both
are outside a builder's reach:

1. `decisions-for-owner.md:62-69`. Three refuted measurements in the only document
   that asks a human to decide. The specification supplies the corrected sentences
   at `:929-945`; somebody with write access to that file has to paste them in.
2. One sentence giving W0-2 the 15 CSS files and naming where the design tokens
   live. Without it the piece that owns the shell cannot touch 9,306 lines of the
   product's appearance, and the failure mode is silent duplication rather than a
   visible escalation.

**What I would watch during the build**, in the order I expect it to bite:

- **The two unspecified frozen read layers.** `plan_read.py` and `spot_ledger.py`
  are frozen on delivery with no published surface. Make section 8.8's publication
  cover their function signatures before they freeze, or the first wave-1 piece
  that needs one more accessor stalls on an escalation to a closed wave.
- **W0-1's route placement.** Ask for the 25-to-13 mapping in W0-1's published
  contract before it writes a line, especially `/api/break-decisions` POST,
  `/api/inventory`, `/api/break-library` and `/api/forecasts`. Cheap now, an
  ownership dispute later.
- **`/api/parameters`.** Reassign it to P2, add it to the competitor-boundary row,
  and decide the real question underneath: whether the operator channel picker at
  `TVBreakDashboard.jsx:5214` may keep listing the three channels you are not.
  Today it does, on every page load, and C1 is currently the only thing that would
  catch it.
- **The money count.** The first builder who has to render `retention_cost_ils` or
  an agency credit limit will hit "exactly two" against their own Bar 3 row. Rule
  in advance that the layer is the invariant and the count is not.
- **The affiliation default.** `auth_store.py:175` reads a missing value as
  company. Put one test in W0-4's reserved prefix that asserts an account with no
  affiliation cannot see a `models/` route or the console switcher, and the wall's
  open side closes with it.
- **Bar 3's floor is not usable until C2 runs.** Section 10 says `06-baseline.md`
  was measured 31 commits before the specification's HEAD and makes C2's first act
  a re-measurement. Hold every piece's regression grading until that lands, or the
  floors are 31 commits old.
