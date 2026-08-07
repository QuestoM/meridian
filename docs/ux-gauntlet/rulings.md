# C2 rulings on test-change requests

The durable record of every test-change request ruled on under section 8.2 rule
4. A piece cites a ruling here instead of editing a test it does not own.

Each entry names the test path, the assertion said to be at stake, the
production change said to force it, the ruling, and the reasoning. C2 has write
authority over a test file only in the way its own ruling states, and deleting a
test is never a resolution.

The Bar 3 three-way on each wave-zero piece is the companion document,
`three-way.md`. Rulings live here, regression verdicts live there.

Method note. A ruling is only worth as much as the measurement under it, so
every request below was re-measured from the tree rather than inherited from the
requesting piece. Where a request rests on a premise about what a test does, the
premise is checked against the file before the behaviour question is asked.

---

## Ruling 001. W0-5, the relocation of `_preview_inputs`

**Request as received.** About 74 failures because `preview_inputs` moved into
its own module as W0-5's declared bounded extraction, while
`tests/test_overrides_preview.py` still patches
`kairos_api.overrides._preview_inputs`.

**Test path.** `tests/test_overrides_preview.py`.

**Assertion said to be at stake.** That `_preview_inputs` is reachable as an
attribute of `kairos_api.overrides`, patchable there by name.

**Production change said to force it.** W0-5's authorised extraction of
`overrides.py:244-302` into `kairos_api/preview_inputs.py`, with the leading
underscore dropped, per section 8.2 and `contracts/W0-5.md`.

**Ruling. No request is granted, because the premise does not hold. No test file
is edited, and W0-5 is not blocked.**

**Reasoning, measured.**

1. `tests/test_overrides_preview.py` does not exist. It is absent from the
   working tree, and `git log --all -- '*test_overrides_preview*'` returns
   nothing, so it has never existed on any branch in this repository.
2. No test patches the old name. `grep -rn "_preview_inputs" tests/` returns
   exactly one source file, `tests/test_w0_5_preview_extraction.py`, which is
   W0-5's own file under its own reserved prefix from section 8.2 rule 1. Its
   line 126 asserts `not hasattr(overrides_api, "_preview_inputs")`, which
   requires the relocation rather than obstructing it.
3. The reference tree agrees. `grep -rn "_preview_inputs"` over `tests/` in a
   read-only extraction of 5a80a709 returns zero source hits, which confirms the
   measurement the specification already recorded at `spec.md:1325`: "measured,
   zero files under `tests/` reference `_preview_inputs`". No test could fail for
   this reason at either tree.
4. The extraction itself meets its own bar. `git diff` on the two authorised
   files touches the moved function and the referring lines and nothing else:
   `constraints.py` changes one docstring reference, two imports and two calls,
   which is the five referring lines section 8.2 enumerates, and `overrides.py`
   removes the 59-line function and rewrites its one import and one call site.

**What W0-5 must preserve regardless.** The relocation is a rename of a private
symbol, so the behaviour that is protected is the behaviour of the two routes,
not the location of the function. `/api/overrides/effect` and
`/api/constraints/effect` must return byte-identical bodies, which is W0-5's own
Bar 3 row and is verified separately, not by this ruling.

---

## Ruling 002. W0-3, the advertiser identity field

**Request as received.** About 48 failures because `AdvertiserRule` gained a
required `name` field as W0-3's declared identity work, with tests not yet
updated.

**Test path.** Not named in the request. Resolved by measurement to the ten test
files that reference `AdvertiserRule`: `test_advertiser_rules.py`,
`test_advertiser_demand.py`, `test_demand_always_on.py`,
`test_demand_assembly_equivalence.py`, `test_qa7_custom_pricing.py`,
`test_layer_overrides.py`, `test_inventory_pacing_signals.py`,
`test_qa6_agencies.py`, `test_qa2_spot_ledger_units.py`, `test_overrides.py`.

**Assertion said to be at stake.** That an advertiser rule can be constructed
without a name.

**Production change said to force it.** W0-3's identity work in
`kairos/optimize/advertiser_rules.py` and `data/advertiser_rules.csv`.

**Ruling. No request is granted, because the premise does not hold. No test file
is edited, and W0-3 is not blocked.**

**Reasoning, measured.**

1. No required field was added anywhere. The dataclass that models a rule is
   `Baseline` in `kairos/optimize/_rule_models.py`, and that file has no diff at
   all against HEAD. Its fields are unchanged, and `advertiser_id` remains the
   only one without a default.
2. What W0-3 actually added to `AdvertiserRuleEngine` is
   `names: NameIndex = field(default_factory=NameIndex)`. It carries a default
   factory, so it is optional by construction and no existing constructor call
   can break on it. The comment on the field states the fallback the default
   encodes: empty means keys only.
3. Every existing construction uses keyword arguments. In the reference tree,
   all ten `AdvertiserRuleEngine(...)` call sites under `tests/` pass
   `baselines=` or `conditions=` by keyword or pass nothing, so a new trailing
   defaulted field is invisible to them.
4. The data change is additive. `data/advertiser_rules.csv` gained three
   appended columns, `name`, `display_name` and `aliases`, all empty for every
   existing row, with `advertiser_id` still first. The lookup path preserves the
   old behaviour explicitly: `key_for` returns its argument unchanged when it is
   already a stored key, and an unbound name resolves to nothing, which keeps the
   unknown-advertiser outcome the module's honesty rules require.
5. The tests pass. All ten files above were run together on the current working
   tree: **163 passed, 0 failed, in 10.06 s**. Nothing in this cluster is failing.

**What W0-3 must preserve regardless.** The identity work must not change who a
rule is about for any advertiser already keyed in the store. That is the
property `key_for`'s stored-key-first branch encodes, and it is the thing to
re-check if the resolver is ever reordered. W0-3's Bar 3 row also fixes the
agency figures and byte-identical engine output, which are verified separately.

---

## Ruling 003. W0-2, the components that moved out of flat `src/`

**Request as received.** None. This one was not requested by any piece. It is
the only real test-change request in wave zero and it surfaced by running the
suite rather than by being reported.

**Test path.** `tests/test_qa2_dashboard_components.py`. The file spans owners,
importing `kairos.export.schedule_freshness` and `kairos.data.dayparts` while
reading W0-2's frontend sources, so under section 8.2 rule 2 it has no single
owner and is frozen. It is C2's to rule on.

**Assertions said to be at stake.** Four tests failed with `FileNotFoundError`,
all from `_read` resolving `tv-break-dashboard/src/<name>` as a flat path:
`ScheduleStalenessBanner.jsx` for the group-label coverage, the unknown-label
passthrough and the double-verb check, and `surface-helpers.js` for the daypart
key coverage.

**Production change that forced it.** W0-2's move. Section 8.2 authorises it by
name as the one piece that may touch a frontend file it does not finally own,
and only to move it. Both files now sit in `tv-break-dashboard/src/shell/`.

**Ruling. The first kind: the test encodes a defect the specification declares
fixed, so C2 makes the edit, and W0-2 cites this ruling. Every assertion is
preserved exactly. None was relaxed, reordered or removed.**

**Reasoning.**

1. What failed is a path, not a contract. These tests exist to pin cross-layer
   vocabulary agreements, that the banner knows every backend group label, that
   an unknown label renders verbatim rather than being dropped, that the Hebrew
   frame stays agreement-free with no double verb, and that `daypartLabel`
   covers every key `daypart_for_hour` can emit. Not one of those is a claim
   about which directory the component lives in, and the specification
   deliberately relocated it.
2. So the edit is to the resolution, not to the expectations. `_read` now finds
   a component by basename anywhere under `src/`, and asserts the basename is
   unique so an ambiguous match cannot pass silently. The bodies of all four
   tests are untouched.

**The second defect this uncovered, which is why the ruling is wider than the
four failures.** `test_removed_dead_exports_stay_gone_and_unreferenced` was not
failing. It was passing vacuously. It swept `SRC.glob("*.js")` and
`SRC.glob("*.jsx")`, a flat glob, and after the move that reaches **3 files
instead of 114**. It had gone from checking that seven dead exports are
unreferenced across the whole dashboard to checking almost nothing, while still
reporting green. A test that stops testing without going red is worse than one
that fails, because nobody is told.

C2 restored its reach in the same edit: the sweep is recursive, and it now
asserts it reached the moved sources before it asserts anything about their
content, so it cannot go vacuous again without failing. This makes the file
strictly stronger. Measured after the edit: **6 passed**, and the sweep covers
114 files where it covered 3.

**What W0-2 must preserve.** Every string these tests pin. The five extended
group labels the banner must map, the `changedLabels[key] || String(key`
passthrough, the absence of `הקלט השתנה` and of the `${changedPhrase} השתנו`
double verb, the presence of `חל שינוי ב${changedPhrase}` and
`חל שינוי בקלט הלוח`, and a `daypartLabel` entry for every engine daypart key
plus `unclassified`. Moving a file may not edit its text.

---

## The count that was actually measured

The request that opened these two rulings reported the suite falling from 1,438
passing at the reference commit to roughly 1,323 passing with 79 failures and 43
errors. That was checked rather than inherited, and it did not reproduce.

| Measurement | Reference 5a80a709 | Working tree |
|---|---|---|
| Test files at top level | 122 | 136 |
| Tests collected | 1,442 | 1,595 |

The 1,438 figure for the reference is consistent with what is there: 1,442
collected, of which 4 are skipped. The current tree collects 153 more tests
across 14 more files, every one of them a new `tests/test_w0_*` file created
under the reserved prefix of section 8.2 rule 1, which is that rule working as
intended.

The reported failing run does not correspond to the current tree. Its own
numbers sum to 1,323 plus 79 plus 43, that is 1,445 tests, which is the
reference tree's collection count and not the current one. A run over the
current tree has 1,595 tests to place. The two wave-zero clusters named as the
causes were run directly and are green: 163 passed across the ten
`AdvertiserRule` files, and 150 passed across all fourteen `test_w0_*` files,
313 tests in total covering exactly the ground the diagnosis said was broken.

The full suite was then run to completion on the working tree, which is the
number that settles it:

**4 failed, 1,584 passed, 4 skipped, in 879.57 s.**

Not 79 failures and 43 errors. All four failures were in one file,
`tests/test_qa2_dashboard_components.py`, all four were `FileNotFoundError` on a
hardcoded flat path, and all four are ruling 003 above. After that ruling the
file is 6 passed, so the suite has no known failure attributable to wave zero.

One caution on how the number was taken. The tree is shared and at least two
other full suites were running against it throughout, at a load average above
60, so a concurrent run could in principle produce a failure that is about the
run rather than the code. That cuts toward this number being pessimistic rather
than optimistic, and every failure it did report was reproduced in isolation
before being ruled on.

## Why rulings 001 and 002 are none of the three named outcomes

Section 8.2 rule 4 gives C2 three ways to rule: the test encodes a defect the
specification declares fixed, so C2 edits it; the test encodes behaviour Bar 3
protects, so the piece changes its own code; or the only other owner has not
started, so ownership transfers. All three presuppose a test that exists and
fails. Ruling 003 is the first kind. Rulings 001 and 002 are none of them,
because neither had a test that exists and fails.

Recording a false premise as though it were a granted edit would have put a
fabricated licence into the durable record, and recording it as a Bar 3
protection would have told two pieces to change code that is already correct. So
both are recorded as unfounded, with the measurements that show it, and no test
file was written under either.

The shape of the mistake is worth keeping, because it will recur. Both requests
described a real and correctly declared production change, W0-5's extraction and
W0-3's identity work, and then attached to it a failure mode that change would
plausibly cause but did not: a patch of a moved private name, and a required
field that was actually given a default. Plausible is not measured. The failures
that did exist were in neither piece, were in a file nobody had named, and were
found only by running the suite.

The protocol still did its job here. It routed a claimed behaviour change to the
critic whose subject is behaviour changes, and the first thing that critic owes
is the measurement, which is what caught it.

---

# Wave one, ruled retroactively on 2026-08-07

The four rulings below are late, and the lateness is itself the finding. Rule 4
says a piece does not edit a test it does not own: it files a request naming
four things, and C2 rules one of three ways. Wave one produced four such
situations. Three were executed as edits with no ruling recorded here, which
means three self-granted licences entered main. The fourth, P6, followed the
protocol exactly, filed the request, restored the file it had started to edit
and proved the restoration three ways, then carried a red test rather than pay
for it with an edit it had no right to make. The piece that obeyed the rule is
the one that ended up red. That asymmetry is worth more attention than any
single assertion below.

Every ruling was re-measured from the tree by a read-only audit rather than
inherited from the piece that made the change.

## Ruling 004. `test_scenario_compare_runs_two_real_optimizations`

**Test path.** `tests/test_insights_endpoints.py`.

**Change.** `assert body["delta"]["revenue_net"] is None` replaced by an
objective-note assertion, an unavailable branch, and per-leg arithmetic
reconciliation.

**Ruling. Legitimate, and the rule is now protected more strongly than before.**

**Reasoning, measured.** The rule at stake is that a convex-blend objective may
never be relabelled as a net, and that money which is not computed must read
null with a reason rather than zero. Measured live: gross 9,337,729.85 minus
retention cost 748,741.15 equals revenue_net 8,588,988.70, matching the delta,
across seven by_day rows. The arithmetic branch genuinely runs, so this is not a
vacuous pass, and `assert leg["objective"] != leg["revenue_net"]` pins the exact
confusion the old null was standing in for.

**Hardening still owed.** The `if not body["money_available"]: return` early-out
means the test could silently degrade to its old shape if the pricer starts
refusing, with nothing going red. It should assert `money_available is True` for
the shipped plan, or record which branch it took.

## Ruling 005. `test_schedule_payload_discloses_total_rows_beside_the_display_slice`

**Test path.** `tests/test_qa2_api_seams.py`.

**Change.** The expected total is now recomputed from the file and from the
operator channel in settings, rather than from the file alone, plus four
assertions on the payload's scope block.

**Ruling. Legitimate, and strengthened.**

**Reasoning, measured.** The rule is that a display slice must disclose the size
of the set it was cut from. The file holds 8,704 rows across four channels; the
operator owns 2,540 of them. Two surfaces print this figure to the operator, so
the old market-wide 8,704 would have overstated the operator's own plan by 6,164
rival rows. The payload now discloses rows in, rows out and both excluded
counts, and the test derives its numbers from the file and settings rather than
from the payload's own note.

**Hardening still owed, and it is the same hazard twice.** The expectation reads
the operator channel from shared writable state. If that channel is ever unset
the boundary check is skipped and the test passes on a completely unscoped
payload. It must assert the channel is set before relying on it, the way
`tests/test_p6_boundary.py` pins the channel rather than reading it.

## Ruling 006. `test_no_file_this_piece_created_is_over_the_line_cap`

**Test path.** `tests/test_w0_2_shell_seams.py`.

**Change.** Equality against a frozen over-cap set became a subset check.

**Ruling. Legitimate in direction. The assertion now expresses the rule rather
than a snapshot, but one dimension was lost and must be restored.**

**Reasoning, measured.** The rule is that no file a wave writes may exceed 450
lines, with inherited breaches named rather than hidden. The known set names
three files; only one still breaches. Two were split, which is the law being
obeyed, and under equality that correct action failed the test.

**What was lost.** The known set is now a permanent allowlist that can never
shrink, already naming two files that no longer breach with nothing to say so.
It needs a companion assertion that fails when a named file drops under the cap,
and a ceiling on each inherited breach, because the one real breach stands at
6,140 lines, 13.6 times the cap, with nothing pinning its growth.

**Asymmetry worth recording.** In this same file another assertion is red at
HEAD, because navigation entries went from seventeen to fifteen when two rails
were removed. So in one file the assertion the wave's work needed relaxed was
relaxed, and the assertion the wave's work broke was left red and handed on.

## Ruling 007. `test_replay_reproduces_the_shipped_plan`

**Test path.** `tests/validation/test_decision_robustness.py`.

**Change.** A `pytest.skip` when the counts differ AND the saved plan is stale.

**Ruling. Accommodation. Reverted on 2026-08-07 to a hard assertion.**

**Reasoning, measured, three grounds.** The commit that added the skip is the
same commit that moved `revenue_weight` from 60 to 35 and
`min_retention_floor` from 0.72 to 0.78, which are exactly the inputs whose
movement makes the plan report stale: it shipped the escape hatch for the
staleness it had just caused. The counts differ even when freshness reports
fresh with nothing changed, measured again after the revert, so staleness never
explained the divergence and only hid it whenever some input happened to have
moved. And any one of twelve input groups going stale, several of which do not
drive weekly break counts at all, excused any difference at all, including a
genuine optimizer regression that merely coincided with an unrelated edit. A
skip is invisible without `-rs`.

**Stated fairly, two things cut the other way.** The plan metadata is gitignored,
so a fresh checkout reports unknown rather than stale and the assertion fires,
meaning it failed closed in CI. And the original docstring already acknowledged
that a failure here can mean stale review numbers; the change was an attempt to
make that machine-decidable. That is why the resolution is a narrowing and not a
plain revert: staleness is now its own loud failure carrying the same recompute
instruction, so the repairer keeps the entire benefit they wanted, but it can
never excuse a count difference.

**What the revert exposed, and it is the point.** The test now fails on the
counts assertion with the plan reporting fresh. That is the discrepancy this
project already calls the least defensible number in the repository. It was
being hidden by a skip. A red test naming a real defect is worth more than a
green skip concealing it, so it lands red and is recorded here.

## Ruling 008. The competitor boundary in the dayparts refusal

**Test path.** `tests/test_qa2_uploads.py`.

**The premise in the original request was wrong, and the correction matters.**
That file was never changed on main. It has exactly one commit, predating the
gauntlet. P6 started to edit it, caught itself, restored it from HEAD, proved
the restoration by empty diff, empty status and matching hash, and filed a rule
4 request instead. So what entered main unread is not a loosened assertion. It
is a production rule change with no ruling, and a red test standing on main that
demands the opposite of the shipped behaviour.

**The contradiction is live.** One test requires a rival channel name inside an
operator-facing refusal. Another asserts no rival name appears in that same
message. Both are on main. One must go.

**Ruling. The shipped behaviour is right and the old assertion is wrong, but
wrong in its means and not its subject. The test is not deleted.**

**Reasoning, measured.** The boundary is not enforced opportunistically here. It
is enforced at write, at read, at storage and at render, always by withholding
the name and disclosing the count: the scope note ships rows in, rows out and
both excluded counts; a name is withheld when a rival appears anywhere inside
it, so a header renamed with a stray space cannot carry it through; the store
freezes no rendered sentence and no channel name at all, only a code and a
withheld list, re-rendered per reading account; withholding may never move a
number; and the frontend hardcodes no channel name of its own, not even the
operator's. A single refusal body naming three rivals is not an acceptable
exception to that.

**And the actionable content survives.** The refusal states the matching rule,
the size of the recognised set, the operator's own channel, and the unrecognised
headers found. It only fires when no recognised column is present, which means
the operator's own column was renamed too, so the fix is always to re-export
with the headers the audience export writes. That never requires a rival's
spelling.

**What must land.** The assertion keeps its subject, that a refusal discloses
the contract rather than merely denying, and changes its means: assert the
operator's own channel is present and that the recognised-set size is stated,
and assert no rival name appears. It must also assert the operator channel is
set before relying on it, for the same vacuity hazard as ruling 005.

## The class this audit found under all of them

At least four wave-one tests turn a boundary assertion into a `pytest.skip` when
no operator channel is configured. A boundary assertion that another process can
turn into a skip by clearing a settings field is not an assertion. This is the
same shape as ruling 007's accommodation and the same vacuity hazard as rulings
005 and 008, and it should be swept as one class rather than three defects.

## Carried forward, not a ruling

`tests/test_advertiser_conditions_api.py` legitimately replaced an inert
position vocabulary with the trade's real one. Riding under that correct change
is a latent semantic flip: an advertiser rule row carrying `first,last` used to
resolve to nothing and now canonicalises to positions 1 and L, so its meaning
moves from allowed nowhere to allowed at two positions. It is inert today only
because no advertiser is bound to a rules row. No test asserts that. A
money-affecting flip whose only protection is an accident of the current data
needs a test before that accident changes.
