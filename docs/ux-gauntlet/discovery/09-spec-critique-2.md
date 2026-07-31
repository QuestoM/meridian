# Second blind critique of the specification

I did not write `docs/ux-gauntlet/spec.md` and I did not write
`discovery/08-spec-critique.md`. I read the spec, the frozen job stories, the
owner decisions and the first critique, then went to the repository at
`/Users/home/Code/questo/meridian` and the running instance at
`http://127.0.0.1:8010` and measured. Where I inferred something it says
INFERRED and from what. Where I could not reproduce a number I say so and give
my method, rather than asserting the spec is wrong.

## Method, and the state of the tree

`git rev-parse HEAD` returns `78b9f440d29a55a6632f15ec4adb8d257cedda42`. The
spec's first line says it was written against `5a80a709` and verified with the
same command. That is now one commit stale, and I checked what the commit
contains: `git diff --stat 5a80a709..HEAD` is 81 files, 7,046 insertions, all of
them under `docs/ux-gauntlet/`, plus the two job-story and spec files themselves.
No code, no data. **Every code and data measurement in the spec remains valid at
this HEAD.** I record it only so a later critic does not read the provenance
line as false.

Everything in the table below I ran during this session.

| Spec claim | Where | My measurement | Verdict |
|---|---|---|---|
| 113 operations, 90 paths, 56 writes | spec.md:1312 | live `openapi.json`: 90 paths, 113 ops, 56 writes | confirmed |
| Zero campaign write endpoints | spec.md:304 | 0 write ops whose path contains `campaign` | confirmed |
| `TVBreakDashboard.jsx` 6,236, `dashboard_api.py` 1,827, `insights_api.py` 697, `catalog_api.py` 656, `version_store.py` 489 | spec.md:1023-1027 | 6,236 / 1,827 / 697 / 656 / 489 | confirmed, all five |
| Route counts 8 / 5 / 7 / 5 and their line numbers | spec.md:1030-1035 | `dashboard_api.py` 8 at 1633, 1638, 1695, 1711, 1728, 1803, 1817, 1823; `insights_api.py` 5 at 400, 518, 622, 630, 646; `catalog_api.py` 7; `version_store.py` 4 decorated plus 1 at :420 | confirmed for the first three, `version_store.py` is 5 by the spec's own listing and 4 by a decorator grep plus one at `:420` |
| Nine logical files in the version store, none under `models/` | spec.md:383 | `version_store.py:46`: settings, constraints, overrides, advertisers, conditions, events, agencies, agency_links, agency_conditions | confirmed, 9, all `data/` |
| 45 advertiser rules, no name column, `notes` empty 45 of 45 | spec.md:813-817 | 45 x 8, columns as listed, `notes` null 45/45, ids `ADV_01..ADV_45` | confirmed |
| 41 observed names, matching 41 of 41, intersection with the 45 ids is zero | spec.md:818-822 | `agency_advertisers.csv` 41 rows all `source=observed`; daily `מפרסם` 41 distinct; intersection of the two name spaces 41; both intersections with `ADV_xx` = 0 | confirmed, all four |
| `agencies.csv` has `name` and `aliases`, 9 of 9 daily agencies resolve | spec.md:827-830 | both columns present, 9 of 9 present in `agencies.csv.name` | confirmed |
| Agency money: gross ₪699,450, net ₪669,978, 119 priced, 56 dropped | spec.md:737, :1182 | `GET /api/export/spots.csv`: 175 rows, `priced` 119, `dropped_frequency` 56, revenue 699,450.00, net_revenue 669,978.00, rebate 29,472.00 | confirmed to the shekel |
| `/api/schedule` 200-row projection: 96 / 73 / 28 / 3 | spec.md:1102, :1322 | `break_schedule` 200 rows: קשת 12 96, כאן 11 73, עכשיו 14 28, רשת 13 3 | confirmed exactly |
| `/api/break-operations` returns 12 programmes on each of four channels | spec.md:1102, :1322 | `programs` 48 rows, 12 per channel, all four | confirmed exactly |
| `/api/events` returns `can_edit` and a five-key `model_context` | spec.md:390, :415-418 | `can_edit: true`; `model_context` keys `training_window, weekday_premiums, measurement, wartime_disclosure, training_gate` | confirmed exactly |
| `/api/pricing` top keys carry no `can_edit` | spec.md:531-534 | `currency, units, base, layers, activation, events, has_overrides, note` | confirmed exactly |
| `run_log.jsonl` 489 records; `models/candidates/` five artifacts; the estimate script 2,885 B | spec.md:453, :485-486 | 489 lines; 5 files named afterwindow, calibrated, competitor, placebo_corrected, spotclip; 2,885 B | confirmed |
| `frequency_rules.csv` is one row, `DEFAULT_ONE_PER_BREAK`, drops 56 of 175 | spec.md:853-857 | one data row, `max_per_break`, scope `default`, value 1; ledger `dropped_frequency` 56 | confirmed |
| Kai: 31 read, 8 propose, 0 write | spec.md:933, :1193 | `READ_TOOL_NAMES` 31, `PROPOSE_TOOL_NAMES` 8, no write list | confirmed |
| `dayKeys` Monday-first at `TVBreakDashboard.jsx:585` | spec.md:1314 | `const dayKeys = ['Mon', ...]` at :585 | confirmed |
| `06-baseline.md` names `342a2896`, 31 commits back | spec.md:1313 | `06-baseline.md:5` says `342a2896`; `git rev-list --count 342a2896..5a80a709` = 31 | confirmed |
| 3,102 tests collected, pass count not claimed | spec.md:1318 | `pytest --collect-only -q`: 3102 collected in 6.66 s | confirmed |
| `_preview_inputs` 6.38 s cold, 0.01 s warm; `_optimize_one_day` 0.98 s then 0.80 s | spec.md:1118-1120 | in process on רשת 13 / 2024-11-01: cold 7.39 s, warm 0.0122 s, 82 segments; optimize 0.793 s then 0.805 s | confirmed within run variance |
| `_group_objective_contribution` 60 microseconds over 82 segments | spec.md:1121, :1151 | 200 iterations, mean 64.0 microseconds | confirmed |
| `data/Spots.csv` structure: 50,386 x 36, break_id 9,492, revenue ₪306,936,788, base_rate 50, adv_premium 1 | spec.md:760-765 | identical on every figure, including the revenue sum to the shekel | confirmed |
| 3,055 breaks on רשת 13, median 2, p95 24, max 47, 875 with seven or more | spec.md:790-792 | 3,055 / 2.0 / 24.0 / 47 / 875 | confirmed, all five |
| The 60 s gap rule on `Spots.csv`: 2 of 15,214 within, 702 of 2,412 boundaries | spec.md:794-798 | not reproducible by any method I tried; see section 4, defect 5 | **not reproduced** |
| A 60 s gap rule reproduces the daily file's declared grouping exactly | spec.md:798-800, decisions:63 | 10 groups for 10 groups, but a different partition, with two counterexample rows | **refuted** |
| `kairos_api/assistant*.py` is 11 modules | spec.md:1068 | 22 modules | **refuted** |
| Twelve keyword-triggered grounding sections | spec.md:958 | `assistant_keywords.py:354` `_SECTIONS` has 11 entries | **refuted** |
| 187 of 200 version manifests point at pytest paths | spec.md:862, :1321 | 186 of 200, by two methods | **refuted, by one** |
| Four modules reference `data/kairos_constraints.csv` | spec.md:860 | 7 non-test modules reference it, and the file is absent | **refuted** |
| `kairos/optimize/agreements.py` has zero callers | spec.md:867 | imported and re-exported at `kairos/optimize/__init__.py:26,44`, and has `tests/test_agreements.py` | **refuted** |

The revision is the most heavily and most accurately measured document in this
set. Twenty-two of my twenty-eight spot checks reproduced exactly, several to the
shekel and to the microsecond. My findings below are about three build-order
residuals, one false closure claim, one false measurement inside an owner
decision, and a handful of internal contradictions the re-cut introduced.

---

## 1. The three blockers

### Blocker 1, the build order: closed in substance, three measured holes remain

The first critique's six collisions are genuinely resolved. I checked each
against the code, not against the table.

- `insights_api.py`'s five routes are split into five modules, each handed to one
  wave-1 piece (spec.md:1046). Verified: the five routes are exactly where the
  spec says.
- `dashboard_api.py` is owned by W0-1 and split into six (spec.md:1046). Verified.
- `advertisers.py` has one claimant, W0-3, which owns both the file and its
  latency (spec.md:1048, :1354).
- All five cross-cutting rules have a named owner with a stated adoption duty per
  piece (spec.md:1098-1104).
- `ScheduleStalenessBanner` no longer has a surface owner claiming a global
  control, because ownership is now by file.

I then ran the check the task asks for. I extracted every backticked path from
section 8.2 and looked for duplicates. Twenty-five paths appear twice and one,
`src/plan/break/**`, appears three times. Twenty-four of those are the declared
create-and-hand-over pattern of rule 8.0.2, so they are not collisions, but they
do make rule 8.0.1 ("Every path in the table below appears exactly once",
spec.md:1010) literally false as written. One is a real collision:

**Hole 1. Wave 0 is not mutually disjoint, and the piece that breaks it is the
one both latency bars depend on.** `kairos_api/preview_inputs.py` is listed in
W0-1's create column tagged `[W0-5]` and again in W0-5's write column
(spec.md:1046, :1050). Both are wave 0, under a heading that reads "Wave 0, five
pieces, mutually disjoint" (spec.md:1042). Worse, the function it must contain
does not live in any file W0-1 owns:

```
kairos_api/overrides.py:244    def _preview_inputs(          <- P3 owns overrides.py
kairos_api/constraints.py:328  from kairos_api.overrides import _preview_inputs   <- P5 owns constraints.py
kairos_api/constraints.py:352  from kairos_api.overrides import _preview_inputs, _resolved_store_overrides
```

W0-1's writable paths are `dashboard_api.py`, `insights_api.py`,
`catalog_api.py`, `version_store.py`, `server.py`. It cannot extract
`_preview_inputs` without writing `overrides.py` (P3, wave 1) and rewriting two
import sites in `constraints.py` (P5, wave 1). W0-5's cache therefore cannot
reach `/api/constraints/effect` or `/api/overrides/effect`, which are the two
endpoints section 8.4 exists to make fast. This is the same species of defect as
revision 1's W0-A versus W0-B, one layer deeper.

**Hole 2. `kairos_api/server.py` is owned by W0-1 and never handed over, while
five later pieces must register a router in it.** I read the file: every router
is mounted by an explicit `app.include_router(...)` line, twenty of them
(`server.py:112` through `:271`). New routers named in section 8.2 are
`break_api.py` (P3), `campaigns_api.py` (P4), `model_console_api.py` (P7),
`media_api.py` (P13), plus `pacing_alerts_api.py` (P11, created by W0-1 so it can
be registered early). Under rule 8.0.1 each of the other four must escalate to a
wave-0 piece that has finished.

**Hole 3. Six of the fifty-one `kairos_api` modules are absent from the table, so
rule 8.2 freezes them, and five are on a wave-1 critical path.** Measured by
exact basename match against every backticked path in 8.2:

| Frozen by absence | Lines | Imported by, and the piece that owns the importer |
|---|---|---|
| `_constraint_options.py` | 249 | `constraints.py` (P5), `scenario_api.py` (P2), `assistant_propose_tools.py` (P9) |
| `audience_api.py` | 114 | `insights_api.py` (W0-1 to P7), `core.py` (W0-4), `assistant_audience_model.py` (P9) |
| `jobs.py` | 114 | `recompute_api.py` (P2) |
| `condition_validation.py` | 74 | `advertiser_conditions.py` (W0-3), `agency_conditions.py` (P4) |
| `events_holidays.py` | 35 | `events_api.py` (P5) |
| `__init__.py` | 2 | n/a |

Each of the five is a split created by the 450-line law, and each holds work its
owner will need. `audience_api.py` is the single reader of
`models/audience_model.json` on the API side, by its own docstring, and P7 has to
wall that read behind affiliation. `_constraint_options.py` builds the option
payload P5's restriction language translates.

**Hole 4. `tests/` appears nowhere in section 8.2.** I grepped the whole spec:
the only mentions of tests or pytest are the version-store pollution and the
collection count. Under "A path absent from this table is frozen"
(spec.md:1040-1041), **no builder may write a test**, while Bar 4 requires the
suite green and W0-1's own regression row is "All 25 split routes return
byte-identical bodies. The response diff is the bar, not a smoke test"
(spec.md:1180), which is a test.

**Verdict on blocker 1: closed for the six named collisions, and I verified all
six against the code. Four new holes, of which hole 4 touches every one of the
eighteen pieces and hole 1 breaks the one wave the spec calls mutually
disjoint.**

### Blocker 2, the impossible bar: fully closed

I re-measured every figure in section 5.5 myself with pandas and every one is
right (see the method table). Forty-five ids, zero intersection with either name
space; forty-one observed names matching each other forty-one of forty-one; nine
of nine agencies resolving today by string equality. The new bar, "every
advertiser that appears in the daily file resolves to a named record: 41 of 41.
Zero invented advertisers" (spec.md:834-836), is exactly what the data supports.
The choice that cannot be derived is escalated as owner decision 1 with both
options, the evidence, a recommendation and what is blocked
(`decisions-for-owner.md:12-52`), and the recommendation is supported by a
measurement I could check: `advertiser_rules ∩ daily = 0`, so the 45 premiums
have never priced anything.

**Verdict: closed, and closed better than the critique asked for.**

### Blocker 3, the two self-created leaks: one closed, one closed with a new instance of the same fault, and one claimed closure that is absent

**Events authoring: genuinely closed.** The training test at spec.md:365 is a
mechanical rule, and applied to event writes it returns `data/calendar_events.csv`,
so configuration. I checked the counterexample that would break the rule: version
restore writes the nine logical files, and `version_store.py:46` lists all nine,
none under `models/`. The rule holds where it is most likely to fail.

**History filtering by artifact root: closed as a design.** spec.md:392-397 is a
filter on the read, not a hidden section, which is the right shape for JS-18. No
code exists yet, so I can only judge the specification, and the specification is
correct.

**A new instance of the exact fault the rule was written to prevent.** The rule
says "An act is TRAINING if and only if its output is a file under `models/`.
Nothing else is training", and then, in bold, "A company-only permission is not
evidence that an act is training" (spec.md:365-368). Now apply it to
`audience_model_activation`. It is a field on `KairosSettings`
(`kairos_api/core.py:143`, verified) and flipping it writes
`data/kairos_settings.json`, so by the rule it is configuration. Section 4.5
makes it "a company-only model-activation control" (spec.md:518-522), and section
8.2 hands `kairos_api/model_activation.py` to **P7 Model console**
(spec.md:1049, :1066), whose surfaces are "Gates, Coverage, Drift, Candidates,
Releases" on the training side (spec.md:431). A configuration act is filed on the
training side because it shares an affiliation gate. That is blur 1, restated.

**Blur 3 is claimed closed and is not in the document.** spec.md:1360 says the
upload consequence is closed at "5.6 and P6". I grepped the whole spec for
`stale`, `fresh` and `upload`. Section 5.6 says nothing about it. P6's rows at
:1065 and :1190 say nothing about it. Nothing anywhere states that an operator
upload invalidates the model. The finding is real and I re-verified it in the
artifact: `models/tv_break_coefficients.json` carries
`source_fingerprints` for exactly `data/reference/Spots.xlsx`,
`Programmes.xlsx` and `Dayparts.xlsx`, and `POST /api/uploads/{kind}` accepts
`programmes`, `spots` and `dayparts` (`GET /api/uploads/status` returns those
three among its seven kinds, with `rows` 3562, 50386 and 43200). Section 4.4's
two-state banner split covers "your change is not in the plan" and "a newer model
version exists". Neither is "your upload invalidated the model and the remedy is
an act you cannot perform".

**Verdict on blocker 3: one of two closed cleanly, one closed as a design, one
new instance of the same taxonomy fault, and one closure claim that is false
about the spec's own contents.**

---

## 2. The nine secondary items

| Item | Status | Evidence I checked |
|---|---|---|
| `data/Spots.csv` dropped | **Addressed, with two bad numbers** | Section 5.4. Every structural figure reproduces exactly, including ₪306,936,788, `base_rate` constant 50 and `adv_premium` constant 1, so the "not money" ruling is correct and well earned. The four gap figures do not reproduce, and one channel-flag statement is wrong. See section 4 |
| Bar 3 absent | **Closed** | Section 8.5 gives every piece a named regression row, plus critic C2. I verified six rows against the running app: 8,704 plan rows (`break_schedule_total_rows`), 9 of 9 agencies, gross ₪699,450 and net ₪669,978 and 119 spots, 31 read and 8 propose tools, `effective_date` 2026-06-14 in settings, five report ids with owners Traffic, Legal / Ops, Revenue, Revenue, Data at `catalog_api.py:506-510`. The job stories gained a Bar 3 floor per story by amendment 1 |
| Model mandate reduced to a console | **Closed** | P12 at spec.md:1076 and :1292-1302, with JS-19 frozen by amendment 3. The five candidate artifacts and the 2,885 B estimate script exist as described |
| Reports dissolved on a story about someone else | **Closed** | spec.md:286 keeps it, cites `01-surfaces.md:301` ("Verdict: LOAD-BEARING", verified verbatim) and the four owner departments, and traces the row to Bar 3 rather than to a story |
| Three untraced rows presented as fully traced | **Closed** | spec.md:273-275 states the exception before the table and marks the three rows |
| INFERRED grades laundered | **Closed, and accurate** | I opened `03-people.md` at every cited line. :185 analyst, :438 data steward, :464 revenue owner, :492 compliance owner, :590 deployment owner all read INFERRED exactly as the spec reports. :151, :213, :245, :277, :308, :352, :380, :517, :539 all read EVIDENCED with the qualifier the spec quotes |
| Misattributed counts (159 and 124, six flags) | **Closed** | spec.md:352-354 and :495-500 both corrected, and section 10 records the correction. I enumerated the six `add_argument` calls and `--output` is indeed an output path |
| Evidence conflicts unrecorded (HEAD, `dayKeys`) | **Closed** | spec.md:1313 records the HEAD conflict, names the consequence for Bar 3, and makes C2's first act a re-measurement. `git rev-list --count 342a2896..5a80a709` is 31, as claimed |
| Owner-blocked marking inconsistent | **Partly closed** | Section 6's capability table names decisions 2, 3, 4 and 5 in the rows they block, and `decisions-for-owner.md` exists with all five. But section 8, which is what a builder reads, marks only P11 ("P4, decision 4"). P1 depends on decision 3 and P2 on decision 5, and neither says so in the build order |
| Hebrew labels for the destinations | **Closed** | Section 4.9 names all five workspaces and all four non-workspace destinations in both languages |

---

## 3. Would each role find their job in five seconds

The landing mechanism now exists and has a builder: `job` on the account, W0-4
owning the field, the list, the door map and `session.js`, with P1 rendering the
picker (spec.md:1100). The picker at 2.2 removes JS-11's dependency on an admin.
That is the biggest single improvement in the revision.

Walking all thirteen door-bearing roles plus the four non-doors:

| Role | Door | Verdict | What I checked |
|---|---|---|---|
| General manager | Today | clean | Unchanged and still the best-served |
| Planner | Plan, week | clean | |
| Scheduler | Plan, the day I am fixing | clean | The Days versus Plan duplication is gone. This role was ambiguous in revision 1 and is not now |
| Traffic operator | Plan, tonight's breaks, הברייקים של הערב | clean on naming, **blocked on delivery** | Their own word from `03-people.md:388` is now the door name. But the door is P10, and P10's "Depends on" is "P3", while `decisions-for-owner.md:88` says P10 is blocked on owner decision 2. The door table does not say the door is owner-blocked |
| Programming representative | Rules, restrictions, הגבלות | clean | Their word, not the engine's |
| Compliance owner | Rules, the licence | clean on naming, half blocked | Owner decision 5b blocks JS-14's second half, unmarked in the door table |
| **Revenue and yield owner** | Rules, the rate card | **the door contradicts the question** | Their stated question in the same table row is "What is a second of airtime worth". I fetched the product's answer: `GET /api/yield-per-second` returns `yield_per_second: 142.7044` with a `basis` formula. Section 8.2 puts `yield_api.py` and `src/plan/week/**` under **P2**. So the question lives on Plan and the door is Rules |
| Account manager | Clients, all clients | clean | |
| Campaign manager | Clients, campaigns on air | clean | |
| Analyst | Clients, delivered money | clean | Revision 1 sent them to Money while their spine was Clients. Fixed |
| Data steward | Sources, today's inputs | clean | |
| Administrator | Account menu | clean, no regression | `TVBreakDashboard.jsx:2288` is the admin-gated Manage accounts item, verified |
| Model steward | Model console via the switcher | clean | Section 4.7 specifies the control, its label in both languages, its render condition and its absence for a channel account |
| New starter | Today plus the picker | clean | |
| Deployment owner, channel account, Kai | not doors | n/a | Correctly excluded and named as such |

**Twelve of thirteen land clean. One, the revenue and yield owner, lands on a
destination that does not hold the answer to the question the spec itself
attributes to them.** Two more land on doors that owner decisions block, without
that being visible in the table a lead reads.

One internal contradiction a builder will hit on day one: spec.md:99 says the
`job` field takes "**thirteen values plus `unset`**" and spec.md:110 says the
picker has "thirteen rows", while spec.md:1100, the row that tells W0-4 what to
ship, says "the eleven-value list". The eleven is the count of roles landing
inside the five workspaces (spec.md:49). The builder's instruction carries the
wrong number.

---

## 4. New defects the revision introduced

**Defect 1. The daily-file grouping claim is false, and it is in the owner
decision.** spec.md:798-800 and `decisions-for-owner.md:63` both say a 60 second
gap rule "reproduces those same 10 groups exactly, so the file is internally
consistent and the groups really are contiguous". I parsed
`data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv` (175 rows, all 175
times parse), computed end times as start plus `אורך תשדיר`, and grouped at 60 s.
The rule yields 10 groups and the declared `שעת התחלת ברייק` yields 10 groups,
and they are not the same partition:

- Declared break `21:22:12` (1 spot, 21:22:16) has a **22 second** gap to the
  next declared break `21:23:10`, so the 60 s rule merges them.
- Declared break `22:59:40` (29 spots) contains a **93 second** internal gap at
  23:11:19, so the 60 s rule splits it. Its span is 642 s against 432 s of ad
  time, so it is not contiguous.

Group sizes under the rule are 1, 3, 4, 7, 7, 22, 28, 30, 35, 38. Declared sizes
are 1, 1, 3, 3, 7, 28, 29, 30, 35, 38. I also tried thresholds 30, 45, 90, 120,
180 and 300 seconds and none reproduces the declared partition. The owner is being
told a file is internally consistent when it is not, in a document written to
help him make a business decision. His decision does not change, it becomes more
obviously correct, but the evidence sentence must.

**Defect 2. The `Spots.csv` competitor flag row inverts the file's own
semantics, and a builder trusting it would breach Bar 4.** spec.md:766 reads
"`competitor_flag` | 100% | 2 | 23,707 rows are `קשת 12`, 18,669 are the
operator's `רשת 13`". Measured:

```
competitor_flag  True 26,679   False 23,707
is_target_channel  קשת 12 23,707 of 23,707;  רשת 13 0 of 18,669
Channel counts     קשת 12 23,707, רשת 13 18,669, כאן 11 4,039, עכשיו 14 3,971
```

`data/kairos_settings.json` sets `operator_channel: רשת 13`. So in this file the
target channel is a competitor and the operator's channel is flagged as the
competitor. The spec's row presents two channel counts under a two-valued flag,
which reads as if the flag partitioned them. Section 5.4 then adopts the file as
a test corpus and says "the channel scope of section 4.5 applies to it like
everything else". A builder who applies the channel scope using the file's own
`is_target_channel` or `competitor_flag` will select `קשת 12`.

**Defect 3. Section 3.4's "exactly two money quantities" is contradicted by a
live endpoint and by the spec's own Bar 3 rows.** The rule reads "A break carries
exactly two money quantities, **projected** and **delivered** ... A figure that
cannot resolve to breaks does not render as money" (spec.md:234-237). I fetched
`GET /api/yield-per-second`:

```
revenue_ils         40,944,759.33
retention_cost_ils   4,145,199.27
revenue_net_ils     36,799,560.06
basis.formula  "retention_cost_ils = base_rate * baseline_tvr * (1 - retention_share)
                * (ad_seconds / unit_seconds); revenue_net_ils = revenue_ils - retention_cost_ils"
```

Retention cost and revenue net are money, in ILS, today, on the endpoint section
8.2 assigns to P2. JS-2's whole comparison is "revenue net of retention cost".
The spec's own entity model at spec.md:688 says the break carries "projected
money, delivered money, retention cost", which is three. Separately, spec.md:1188
requires P4 to preserve agency "credit limit", and `agencies.csv` carries
`credit_limit_ils`, a money figure that cannot resolve to breaks and therefore
may not render under the rule as written.

**Defect 4. The re-cut of section 8 deleted the per-piece acceptance bar and the
wave-1 dependency column.** Revision 1 attached a bar to each piece; the first
critique quotes them (P1's three answers, P2's 180 s, P3's 500 ms, P6's 3 s).
Revision 2's section 8 spans lines 999 to 1236 and contains **three** lines
mentioning any JS story, two of which are inside the latency subsection, and no
per-piece bar other than W0-1's byte-identity row. The wave-2 table has a
"Depends on" column; the wave-1 table does not, although P3 depends on W0-3's
identity by section 5.3's own ordering and P5's 3 s bar is explicitly conditional
on W0-5's attribution (spec.md:1161). A builder reading section 8 gets files and
a regression floor, and no done condition.

**Defect 5. Four figures in section 5.4 are not reproducible and no method is
stated.** spec.md:794-798: "only **2 of 15,214** consecutive gaps exceed 60 s"
and "**702 of 2,412 break boundaries (29.1 percent)** have a gap of 60 s or
less". My method: filter `Channel == רשת 13` (18,669 rows), parse
`Date + Start time` as `%d/%m/%Y %H:%M:%S` (17,524 parse, 1,145 carry an Excel
`01/01/1900 HH:MM` artifact), end equals start plus `Duration`, sort within each
date, count consecutive pairs. Result: **15,614 within-break gaps, 0 above 60 s;
1,880 boundaries, 625 at 60 s or less, 33.24 percent.** Using the file's own
minute-resolution `Start_dt`/`End_dt` instead: same counts, 688 boundaries at 60 s
or less, 36.60 percent. No other channel produces 2,412 either (קשת 12 gives
2,185, עכשיו 14 gives 1,153, כאן 11 gives 892). The spec's conclusion, that a gap
rule does not reproduce `break_id`, is supported by my numbers as strongly as by
its own. The counts are not checkable.

**Defect 6. Section 4.2's lexicon test fails by construction on a surface the
spec designs.** Test 2 requires that a run surface's read endpoints return zero
hits for a lexicon that includes `coefficient` (spec.md:412-413). Section 5.6
requires Sources to list `models/tv_break_coefficients.json`, "the artifact that
is read", and to mark the posterior as an unused fallback (spec.md:882-887). I
fetched `GET /api/files` and it returns eight records, including
`models/tv_break_posterior.pkl` and omitting the coefficients JSON, exactly as
the spec says. Correcting it puts the string `tv_break_coefficients` into a run
surface's payload, and the test fails. Section 3.2 also puts "the model version"
into Sources' object family. The test needs an exemption for artifact names or
the rule needs restating.

**Defect 7. The affiliation wall's default is open, and the spec calls it safe.**
spec.md:91 cites `auth_store.normalize_affiliation` as "the pattern to copy: a
field that defaults safely for every existing record". I read it:

```
kairos_api/auth_store.py:175  def normalize_affiliation(value: Any) -> str:
    """Missing, empty or unrecognized values read as company (the permissive
    legacy default), so only an explicitly stored channel value restricts."""
```

The default is the permissive side. For `job` that is harmless, because `unset`
is genuinely safe and job never decides permission (spec.md:87-89). For the wall
it is not: section 4.7 renders the Model console switcher only when
`affiliation = company`, and section 4.5 gates every training route on the same
value, so any account whose affiliation value is missing or unrecognized reads as
company and is shown the other side. The spec never states this and W0-4 is not
told to change it. INFERRED consequence, from the docstring and from the fact
that `auth_store.py:211` defaults new users to company as well.

**Defect 8. `/api/parameters` is assigned to a piece that cannot write it, and it
leaks competitors as well as training content.** spec.md:1101 says the four open
reads "are closed by **P7, P7, W0-4, P5** respectively", so `/api/parameters` is
W0-4's. It is defined at `kairos_api/scenario_api.py:388`, and `scenario_api.py`
is **P2**'s (spec.md:1061). I fetched it:

```
channels:            ['קשת 12', 'רשת 13', 'כאן 11', 'עכשיו 14']
available_channels:  ['כאן 11', 'עכשיו 14', 'קשת 12', 'רשת 13']
coefficient_freshness: {"status": "fresh", ... "reason": "All fingerprinted source
                        files match the coefficients; ..."}
first_break_multiplier: 1.0
```

Section 8.3's competitor-boundary row names only `/api/schedule` and
`/api/break-operations` as needing the scope. This endpoint names three
competitors and is not on the list.

**Minor, but worth one line each.** `kairos_api/assistant*.py` is 22 modules, not
11 (spec.md:1068), which doubles P9's surface. The `InventoryHeatmap` proof is
14 lines, 5159 to 5172, not twelve (spec.md:295). Section 8.3's vocabulary check
says "Today that grep returns 159 hits" for a four-term grep over
`tv-break-dashboard/src/`; measured, `recompute` alone is 162 occurrences over
124 lines in 19 files, and the four terms together are 198 occurrences.
`decisions-for-owner.md:62` says the 38-spot group is 13.4 minutes; measured, it
runs 22:04:16 to 22:18:12, which is 836 s, 13.93 minutes. The conclusion, that it
exceeds the 12 minute per hour limit, holds either way.

---

## 5. Is the training line unmistakable and checkable

**Verdict: it is now genuinely checkable, which revision 1's was not, and running
the check against the spec's own architecture produces three failures.**

What is new and right: one rule stated once as an iff on the file root
(spec.md:365), a separate permission rule stated in the same box so the two can
never be conflated again, the rule applied to a nine-row table of every act with
its write target (spec.md:373-383), and three commands a critic can run on any
surface (spec.md:399-424). Test 1 is mechanical and I verified the hardest case
against the code: restore writes nine logical files, all under `data/`, so it is
configuration, which is what the table says. Test 3 is a grep. The release note
of section 4.6 closes the silent-money leak with the one piece of training-authored
text that may cross the line, and it is specified tightly enough to check (one
sentence, an optional direction, no p-value, no coefficient).

What still fails:

1. `audience_model_activation` writes `data/`, so the rule says configuration,
   and section 8.2 hands its control to P7 on the training side. Section 1 above.
2. Test 2 fails on Sources by construction, because Sources must name the
   coefficients artifact. Defect 6.
3. Test 2 fails today on `/api/parameters`, which the spec identifies, and the
   endpoint is assigned to a piece that cannot write it. Defect 8.

And one hole that is not a test failure but a missing design: the upload that
makes the model stale has no surface, no message and no owner, while section 11
says it does. That is the most likely way an operator will create a training
problem, and it is the one path the spec does not cover.

The two output nouns, גרסת מודל and גרסת תוכנית, are handled honestly: they
differ only by qualifier, and the answer is that they never co-occur for a
channel account because History filters by artifact root (spec.md:599-605). That
is a real answer rather than a rename.

---

## 6. Blind A/B against Linear and Google Ads

I re-read `07-references.md` at every line the spec cites and the citations hold,
with two off-by-a-few drifts: `Cmd K` is described at :105, not :101 (:101 is the
section heading), and `Cmd J` at :410, not :413. Same paragraphs, no invented
mechanic.

**Criticism 1, four classification axes versus one: substantially closed, with
two misfits.** The axis is stated ("every destination is a family of objects you
can open, except one home", spec.md:157-158), the exception is declared as an
exception, and the count is derived from the axis rather than asserted. That is
Google Ads' move, and `07-references.md:214` ("There is no fourth layer invented
for convenience") is quoted correctly. Two of the spec's own contents do not obey
it. **Target** is an authored record with a scope and a value, so by the axis it
belongs to a family, and it is placed on Today, the one destination declared to
hold no objects (spec.md:164, :306, :1060). **Reports** is filed under Sources,
defined as "the inputs a run reads" (spec.md:168), while a report is an output; a
newcomer applying the stated rule would look for exports under the thing they
export. The rule now exists and predicts most things, which revision 1's did not.

**Criticism 2, money on five surfaces: answered on the right substance, and
overstated.** The spec reinterprets the criticism from "how many surfaces" to
"which layer owns the quantity", and that is the better reading: Google Ads shows
spend everywhere and owns budget at exactly one layer. Naming the break as the
money layer, with one drill (figure to breaks to spots) and the scope printed
with the figure, is a genuine and specific answer, and the ₪686,475 diagnosis at
spec.md:251-255 is the sharpest paragraph in the document. It is overstated in
one word: "exactly two". Defect 3 shows four money quantities live today, one of
them the quantity JS-2 is about. Fix the count, keep the layer.

**Criticism 3, two destinations for one object at two zooms: fully closed.** Plan
is one destination with a segmented zoom control stepped by `Cmd B`, which is
Linear's mechanism as recorded at `07-references.md:119`. `01-surfaces.md:810`
names the Optimizer and Schedule duplication as the first of the three heaviest
structural facts, and the revision now deletes it rather than moving it. One
loose end: Linear's `Cmd B` toggles two states and Meridian's steps three, and
the spec does not say what happens at the end of the cycle or how you go up a
level.

**The fourth criticism, no way back from a record: closed.** spec.md:341-344
adopts the `1 / 31` counter and the two arrows on every drill.

**Where the spec still beats both, and it should be defended.** The basis rule
plus tri-state honesty is stronger than anything in either reference, and the
five measured honest empty states are now a product-wide Bar 3 row of their own
(spec.md:1196-1201). I confirmed the discipline is real in the shipped API:
`/api/yield-per-second` ships a `basis.formula` string with its inputs, and
`/api/uploads/status` ships `in_use`, `in_use_reason` and `engine_reads` per
kind. That habit is now a rule, and it survives the rebuild.

---

## 7. Honesty scan

**Nothing is fabricated. No persona is invented: all seventeen rows in section 2
map to `03-people.md` and I checked every evidence grade at its cited line. No
advertiser is invented, and the bar that would have required inventing four is
gone.** Section 9's fifteen declared omissions each carry a reason, and item 13
("No money from `data/Spots.csv`") is the honest conclusion of a measurement I
reproduced exactly.

Numbers that do not carry the basis they claim:

- The four `Spots.csv` gap counts, with no stated method and no reproduction
  under six variants. Defect 5.
- "Reproduces those same 10 groups exactly", refuted with two counterexample
  rows, and repeated in the owner decision. Defect 1.
- The `competitor_flag` row, which reads as a partition of two channels and is
  not. Defect 2.
- "187 of 200" version manifests, measured 186 by two methods, and called "the
  load-bearing number" at spec.md:1321.
- "Four modules reference" `data/kairos_constraints.csv`, measured 7 non-test
  modules: `kairos_api/constraints.py`, `kairos/service.py`,
  `kairos/optimize/constraints_store.py`, `kairos/optimize/_constraints_io.py`,
  `kairos/export/schedule.py`, `kairos/export/schedule_freshness.py`,
  `scripts/export_schedule.py`.
- "`kairos/optimize/agreements.py` (zero callers)" listed among things "proven
  dead before removal". It is imported and re-exported at
  `kairos/optimize/__init__.py:26` and `:44`, and `tests/test_agreements.py`
  exercises `load_agreements` seven times. Removing it fails the suite, and no
  piece owns `tests/`, so the removal the spec authorizes cannot be completed
  under its own rules.
- "Twelve keyword-triggered grounding sections", measured 11 at
  `assistant_keywords.py:354`.
- "11 modules" for `assistant*.py`, measured 22.
- Section 3.2 says Sources' eight source files are each "a record with a state, a
  row count and an `in_use` verdict". `GET /api/files` returns
  `path, exists, size, modified` and nothing else. The `in_use` verdict and the
  row count are on `GET /api/uploads/status`, which covers the seven upload
  kinds, and the cited `02-api-and-data.md:513` is that table. Two different
  record shapes fused into one sentence.

Capability implied beyond what the data supports: none that I found. The three
places where it would have been easy are all handled honestly. The projected
versus delivered limit is stated in the section that would benefit from hiding it
(spec.md:260-267). The terminal break state is `exported`, taken verbatim from
`05-gaps.md:241`, which I read and which lists exactly "draft, assembled,
verified, locked, exported". Section 9 item 2 refuses to derive a plan target
from the plan.

One residual honesty risk in the build order rather than in the prose: section
8.5 makes `06-baseline.md`'s figures the regression floor and section 10 records
that they were taken 31 commits before the spec's HEAD, then makes C2's first act
a re-measurement. That is the right handling, and it means the floor is not
usable until C2 runs.

---

## The single largest remaining gap

**Section 8.2 declares itself total ("Every path a builder may write. Nothing
else. A path absent from this table is frozen") and is silent about three classes
of work every piece must do, so as written the closure rule converts them into
escalations.** Concretely:

1. **Tests.** Zero paths under `tests/` appear anywhere in the spec, while Bar 4
   requires 3,102 collected tests to stay green, thirteen pieces create new
   stores, routers and data files, and W0-1's own regression bar is a response
   diff.
2. **`kairos_api/server.py` after wave 0.** Twenty explicit
   `app.include_router(...)` lines, W0-1 the only owner, and four wave-1 or
   wave-2 pieces creating routers that must be registered there.
3. **The five helper modules the 450-line law split off owned modules**, frozen
   by absence: `_constraint_options.py`, `audience_api.py`, `jobs.py`,
   `condition_validation.py`, `events_holidays.py`, 586 lines in total, each on
   the critical path of at least one wave-1 piece.

Immediately behind it, and sharper for wave 0 specifically: **`_preview_inputs`
lives at `kairos_api/overrides.py:244` and is imported by
`kairos_api/constraints.py:328` and `:352`, both wave-1 files, so neither W0-1
nor W0-5 can build `preview_inputs.py`, and both latency bars in section 8.4
depend on it.**

The five things I would put in the same pass, each one or two sentences:

- Add a `tests/` allocation per piece, `server.py` as a registration-only shared
  file with a stated append rule, and the five helper modules to their importer's
  owner.
- Give W0-1 an explicit, bounded authorisation to touch `overrides.py` and
  `constraints.py` for the extraction only, exactly as W0-2 has for the frontend
  move, or move the extraction into W0-5 with the same authorisation.
- Reassign `/api/parameters` to P2, whose file it is, and add it to the
  competitor-boundary row with its measured `available_channels`.
- Fix the daily-file grouping sentence in both the spec and the owner decision,
  and either state the method behind the four `Spots.csv` gap counts or replace
  them with counts a critic can reproduce.
- Fix the eleven versus thirteen job-value contradiction, restore a per-piece
  acceptance bar and a wave-1 dependency column, and correct
  `assistant*.py` to 22 modules.

---

## Verdict

**NOT READY, narrowly, and for a much shorter list than revision 1.**

Three reasons, in order.

1. **Wave 0 is not launchable as written**, and it is the wave whose heading says
   its five pieces are mutually disjoint. W0-5 writes a file W0-1 creates, W0-1
   cannot deliver that file without writing two wave-1 modules, W0-2's shell must
   import a `vocabulary.js` that W0-4 is creating in parallel, and no wave-0
   piece may write a test while W0-1's bar is a response diff.
2. **Two measurements a builder or the owner would act on are wrong.** The daily
   file's declared breaks are not reproduced by a 60 s gap rule and one of them
   is not contiguous, and that sentence sits in the document going to a human for
   a business decision. The `Spots.csv` competitor flag marks the operator's own
   channel as the competitor, and section 5.4 tells a builder to apply the
   channel scope to that file.
3. **The training rule, which is otherwise the best-improved part of the
   document, still files one configuration act on the training side, and one of
   its three tests fails by construction on a surface the spec designs.**

None of that touches the architecture. The people model with its restored
evidence grades, the single classification axis, the money layer, the zoom
control, the training test, the release note, the context switcher, the Bar 3
regression rows, the owner decisions and the latency diagnosis are all sound, and
I reproduced the load-bearing measurements behind them, including the 64
microsecond evaluation primitive that answers JS-3's bar and the 7.39 s versus
0.012 s segment-construction split that overturns the first critique's own
diagnosis. That correction, made against a critic rather than in agreement with
one, is the strongest single act in the revision.

What is missing is a paragraph of additions to section 8.2, one authorisation, one
reassignment and three corrected sentences. That is one short pass, and I would
expect to return READY on it.
