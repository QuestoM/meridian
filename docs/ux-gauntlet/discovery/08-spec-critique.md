# Blind critique of the specification

I did not write `docs/ux-gauntlet/spec.md` and I am not defending it. Everything
below is checked against the seven discovery reports, the frozen job stories, the
repository at `/Users/home/Code/questo/meridian`, and the running instance at
`http://127.0.0.1:8010`. Where I inferred something it says INFERRED and from
what.

## Method, and what I re-measured myself

The spec's own numbers were spot-checked rather than trusted. Everything in this
table I ran during this session.

| Claim | Where the spec says it | My measurement | Verdict |
|---|---|---|---|
| 113 operations, 90 paths, 56 writes | spec.md:711 | `openapi.json`: paths 90, ops 113, writes 56 | confirmed |
| Zero campaign write endpoints | spec.md:167 | 0 write ops whose path contains `campaign` | confirmed |
| HEAD is `5a80a709` | spec.md:3 | `git rev-parse HEAD` = `5a80a7098a64` | confirmed |
| Monday-first `dayKeys` at line 585 | spec.md:605 | `TVBreakDashboard.jsx:585` | confirmed (05-gaps.md:681 says 586, spec is right) |
| Four editor files hold zero "revenue"/"retention" | spec.md:24 | 0, 0, 0, 0 across the four files | confirmed |
| Five call sites of `require_company_editor` | spec.md:306 | `events_api.py:378,399,426`, `pricing_api.py:234,241` | confirmed |
| `run_log.jsonl` 489 records | spec.md:257 | 489 lines | confirmed |
| `models/candidates/` holds five artifacts | spec.md:286 | 5 files | confirmed |
| The 22:53:49 break is 7 spots, 212 s | spec.md:630 | 7 rows, 212 s | confirmed |
| No plan target in `/api/overview` | spec.md:168 | 111 distinct keys, 0 matching `goal\|target\|budget\|on_plan\|variance` | confirmed |
| `/api/pricing` carries no `can_edit`, `/api/events` does | spec.md:320-324 | pricing top keys have no `can_edit`; events returns `can_edit: true` | confirmed |
| Model internals open to any caller | spec.md:307 | `GET /api/model/audience` 200 in 1.45 s, `GET /api/impact` 200 in 1.10 s, no session | confirmed |
| 45 of 45 advertisers can be named | spec.md:603 | see section 7, finding 1 | **refuted** |

The spec is, on the whole, unusually well measured. My findings are about
structure, ownership and three overreaching claims, not about sloppiness.

---

## 1. Does every decision trace to a job story

**Verdict: no. Fifteen of eighteen rows in section 3.3 do trace. The section's
own claim about itself is false, the count of seven is not derived from anything,
and four decisions outrun the story cited for them.**

### 1.1 The section makes a claim about itself that its own table refutes

`spec.md:138` states: "Every row is justified against a story. Nothing is moved
for tidiness." Three of the eighteen rows cite no story:

- Optimizer, `spec.md:143`: justified as a "Measured duplicate".
- Break Library, `spec.md:146`: justified by a byte-identity and a broken sort.
- Kai, `spec.md:156`: justified by "It already is a dock".

Those are good reasons. Measured duplication is a better reason than a story.
The defect is that a lead reading `spec.md:138` will believe the story trace is
total and will not look for the three exceptions. Fix the sentence, not the rows.

### 1.2 The number seven is asserted, not derived

`spec.md:100-103` builds the structural argument: "there are only four depths and
one join. That is why the navigation collapses." Four depths are week, day,
break, spot. The architecture then ships **seven** workspaces, of which three are
the depths (Plan, Days, and the break inside Days) and four are not depths at all
(Clients is an entity family, Money is a topic, Rules is a rule type, Sources is
an input class). Today is a fifth axis, a time scope.

So the spine argument produces three of the seven and the other four arrive
without a derivation. The list is also not seven destinations: Kai's dock,
History, account settings and the Model console shell bring the real count to
eleven (`spec.md:119-131`, `spec.md:231`). "Seven workspaces. Not seventeen"
(`spec.md:107`) is the most quotable line in the document and it is the least
evidenced.

This is the clearest case of aesthetic preference wearing a justification. It may
still be the right seven. It is not derived from the spine, and the spec says it
is.

### 1.3 Four decisions outrun the story cited for them

**Reports dissolved, citing JS-9** (`spec.md:150`). JS-9 asks the analyst to
answer a money question without exporting. It does not ask for the export surface
to be removed. `01-surfaces.md:301` grades Reports LOAD-BEARING: "This is where
the real CSVs come from", five cards plus a Download all. Its five reports carry
four different owner departments (`catalog_api.py:506-510`: Traffic, Legal / Ops,
Revenue, Revenue, Data), so at least three personas other than the analyst use
it. The spec dissolves a working surface on the strength of a story about a
different person.

**Events authoring moved to the training side, citing JS-18** (`spec.md:149`,
`spec.md:229`). JS-18 is the boundary critic. It says nothing about calendar
authoring. Worse, `04-training-vs-runs.md:131` classifies `POST/PUT/DELETE
/api/events` as CONFIGURATION, and the spec's own section 4.1 defines training as
the activity that "fits coefficients, runs the held-out gates ... and writes a
model artifact" (`spec.md:211-213`). Creating a holiday writes no model artifact.
The spec files a configuration act under training because it happens to share an
affiliation gate. See section 3, blur 1.

**Plan and Days as two destinations, citing JS-2 versus JS-3** (`spec.md:144`).
Two jobs is not two destinations. Linear, which the spec names as the reference
for exactly this surface (`spec.md:605`), puts the equivalent flip inside the
content: `Cmd B` swaps list and board, and Active / Backlog / All is a segmented
control at the top of the content, not three rail entries
(`07-references.md:90-92`). See section 4.

**The break state machine's terminal state** (`spec.md:412`):
`planned -> assembled -> verified -> locked -> aired`. Nothing in the product can
observe that a break aired. `spec.md:501` says a delivery or as-run feed is
owner-supplied and does not exist, and `05-gaps.md:245-250` proposes the
achievable vocabulary instead: `draft, assembled, verified, locked, exported`.
The spec silently swapped an observable terminal state for an unobservable one.

### 1.4 Decisions that trace to a law or a reference rather than to a story

These are legitimate under Bar 2 and Bar 4 and should simply be labelled as such,
so a critic does not hunt for a story that does not exist: the 450-line law and
the design tokens in W0-C (`spec.md:605`), `Cmd K` (`spec.md:132`), `Cmd J` for
the dock (`spec.md:121`, from `07-references.md:413`), Sunday-first
(`spec.md:605`, Bar 4), and RTL.

### 1.5 What is genuinely well traced

Section 5.2's entity model, the identity-first ordering in 5.3, the vocabulary in
4.5 and 4.6, and the six-question lists in 4.3 are all traced to measured
artifacts, not to taste. Section 5.3's ordering argument (advertiser identity
first because JS-5, JS-6, JS-9 and half of Money all die without it) is the best
reasoning in the document, and section 7 does the same for Kai.

---

## 2. Five seconds of first sight, per role

**Verdict: four roles land clean, five land ambiguous, three cannot land at all
because the mechanism that lands them has no builder.**

The landing mechanism is the new `job` field (`spec.md:61-70`). It appears twice
in the whole document, both times in section 2, and **never in section 8**. No
piece owns it, no piece owns the per-job sidebar order, and W0-C's files are
`tv-break-dashboard/src/**` while the field lives in `auth_store.py`. So on the
build order as written, every role lands on the same default screen and section 2
does not ship. That is the single most consequential omission in the walk below.

Assuming the field exists, walking each role from a cold first sight:

| Role | Lands on | 5 s verdict | Where they get lost |
|---|---|---|---|
| General manager | Today | **clean** | Nothing. Best-served role in the architecture. |
| Planner | Plan | **clean** | Nothing. |
| Account manager | Clients | **clean** | Nothing. Their three entities are one word. |
| Revenue owner | Money | **clean** | Nothing. |
| Scheduler | Days | **ambiguous** | "Days" is a plural noun for a singular act. Their job is "open the day I am fixing". Worse, section 3.3 leaves grid, daypart and **timeline** on Plan and puts only the editor on Days, so a scheduler who clicks Plan finds a timeline of the week and stops there. Two destinations show a timeline of breaks. |
| Traffic operator | Days, break board | **lost** | "Days" is not their word. `03-people.md:388` lists their real vocabulary from the file they work from: ברייק, תשדיר, House Number, מיקום בברייק. Nothing in the seven names contains any of it. They are also the fifth different person told to go to Days. |
| Analyst | Money | **ambiguous** | JS-9 begins "which advertiser", so they click Clients. The spec itself says the analyst "walks the commercial spine down to the same spots from the other end" (`spec.md:101-102`), which is the Clients spine, and then houses them in Money. |
| Campaign manager | Clients, pacing view | **ambiguous** | Their object is a campaign and there is no Campaigns entry. Clients is a reasonable second guess, not a first-sight answer. |
| Programming representative | Rules | **ambiguous** | "Rules" also holds guardrails, compliance and frequency. Their word is restriction / הגבלה (`spec.md:384`), and the spec chose the operator-facing word for the object but not for the place. A representative thinking about a break might try Days first. |
| Compliance owner | Rules, compliance view | **ambiguous** | Same room as the programming representative, and their old route (Reports, "Compliance and guardrails", 7 rows) is dissolved. |
| Data steward | Sources | **near clean** | "Sources" is a noun about state; their act is upload. One second of hesitation, not a loss. |
| Administrator | account menu | **clean, no regression** | Verified this is where it already is (`TVBreakDashboard.jsx:2288`), so JS-15's passing baseline survives. |
| Model steward | Model console | **cannot land** | `spec.md:233` says a company account can switch contexts. Nothing says where the switcher is, what it is called, or which piece builds it. A different shell with no named door. |
| New starter, JS-11 | inherits the account manager | **cannot land** | JS-11's target is 300 s with 0 wrong pages opened. It depends entirely on an admin having set `job` correctly from a list the spec never enumerates, for a person who has been in the building for one morning. There is no "choose your job" fallback and no row for JS-11 in the section 2 table. |
| Channel-affiliated account | the wall | not a first sight | See section 3. |

The concentration problem is worth naming on its own: **five of thirteen human
roles are sent to Days or Plan**, and both surfaces render a timeline of breaks
over the same plan. That is the Optimizer-versus-Schedule duplication
(`01-surfaces.md:112`) reappearing under two new names.

---

## 3. Is the training line unmistakable, or only stated

**Verdict: stated, and materially better than today, but eight surfaces can still
blur it. Two of the eight are created by the spec itself.**

The definitions (4.1), the vocabulary (4.5), the permission sentence (4.4) and
the "one model fact" rule (`spec.md:264-266`) are all sharp and all correct. The
blur is at the seams.

**Blur 1, created by the spec. Events authoring sits on the training side.**
`spec.md:229` puts "plus Events authoring" in the Training row. Section 4.1
defines training as artifact-producing; `04-training-vs-runs.md:131` classifies
event writes as CONFIGURATION. A company user creating a holiday is now doing
configuration inside a shell whose "permanent marker ... says this is the company
side" (`spec.md:232`). The spec's four-class taxonomy has four classes and its
architecture has two homes, and the mismatch put one act in the wrong one.

**Blur 2, created by the spec. History merges model releases into an
affiliation-blind timeline.** `spec.md:623` gives P9 "Every change by anyone
including Kai and including runs in one attributed timeline, previewable before
restore, **separately permissioned**". Separate permissioning there means read
versus restore. Nothing scopes it by affiliation. But `spec.md:506` says the same
timeline adds "the model releases", and `spec.md:229` says training surfaces are
company-only. As written, a channel account opening History sees model versions
landing. This is the sharpest new leak in the document because History is
reachable "from the account menu and from every object's own header"
(`spec.md:125-127`).

**Blur 3. The upload that makes the model stale.** `04-training-vs-runs.md:391`
(F13) measured it: `POST /api/uploads/{kind}` accepts programmes, spots and
dayparts, which are exactly the coefficients artifact's `source_fingerprints`, so
an operator upload flips coefficient freshness to stale and "the only remedy is a
training run the uploader cannot start". Sources is P7 and neither section 4.3
nor section 6 mentions this. The single most likely way an operator will create a
training problem has no design.

**Blur 4. `model_context` rides onto two run surfaces.** I fetched
`/api/events` just now: `model_context` carries `training_window`,
`weekday_premiums`, `measurement`, `wartime_disclosure`, `training_gate`.
`spec.md:149` moves the events read side to "read overlay on Plan and Days" and
says nothing about stripping `model_context`. The wartime disclosure and the
training window would arrive on two operator surfaces.

**Blur 5. A model version can move money with no legible cause, by design.**
`04-training-vs-runs.md:346` (F9) measured that `kairos/service.py:102-125` reads
`first_break_multiplier` from the coefficients metadata on every optimization and
takes `max(assumption, measured)`, currently 1.0 and off at p=0.2034. A future
training that clears p < 0.01 raises the retention cost of every first break in
every plan. Under this spec the operator gets "a newer model version exists"
(`spec.md:358`) and, by rule, no gate verdicts. So revenue moves, the basis rule
at `spec.md:186` demands a basis, and the training-side rule forbids the only
explanation. The spec needs a fourth thing on the run side beyond version, date
and owner: a plain-language "what changed for you" note authored on the training
side. It has neither the note nor a stated decision to omit it.

**Blur 6. No piece owns the wall.** Section 4.4 is the strongest paragraph in the
spec and section 8 gives it to nobody. Concretely unowned: gating
`GET /api/impact`, `GET /api/model/audience`, `GET /api/parameters` and
`/api/events` `model_context` on read; moving `audience_model_activation` out of
the free-form settings document; adding `can_edit` to every walled endpoint;
making every surface read its own session affiliation. P8 builds new company
routes; nothing closes the existing open ones. P14 is a critic, not a builder.

**Blur 7. The two output nouns repeat the trap the discovery warned about.**
`04-training-vs-runs.md:749` is explicit: the split "has to differ in kind, not
in degree". The verbs obey it (אימון versus הרצה, different verbs, different
objects). The outputs do not: גרסת מודל and גרסת תוכנית differ only by the
qualifier, and they will appear side by side in exactly one place, the History
timeline of blur 2.

**Blur 8. The context switcher.** A control that moves a company user between the
operator side and the training side is by definition the one control that can be
mis-clicked across the line. `spec.md:233` asserts it exists and never specifies
it.

---

## 4. Blind A/B against two reference products

I compared the architecture to **Linear** and **Google Ads**, both captured in
`07-references.md` with measured mechanics rather than adjectives.

**Verdict: both read clearer to a newcomer, for three concrete reasons. The spec
beats both on one.**

### Reason 1: one classification axis versus four

Google Ads declares three layers and repeats the containment on every management
surface: account, campaign, ad group, and `07-references.md:214` records the rule
that matters, "There is no fourth layer invented for convenience". Budget belongs
to the campaign. Ads and keywords belong to the ad group. A newcomer learns one
sentence and can predict where anything lives.

Linear's rail is two to four ungrouped destinations plus three grouped sections
(`07-references.md:86-88`), and the destinations are all the same object at
different scopes of ownership.

Meridian's seven are sorted on four different axes at once: time scope (Today,
Days), artifact (Plan), entity family (Clients), topic (Money), rule type
(Rules), input class (Sources). A newcomer cannot form a rule that predicts
where the eighth thing goes, because there is no rule.

### Reason 2: money has one home in Google Ads and five here

This is the sharpest loss, and it is measurable against the spec's own text. A
break's money appears on Today (`spec.md:110`), on Plan as revenue net of
retention (`spec.md:616`), on Days within 500 ms of a drop (`spec.md:617`), on
Money as yield and gross-to-net (`spec.md:114`), and on Clients as what an
advertiser delivered (`spec.md:114`). Five surfaces.

`01-surfaces.md:816` measured the consequence of exactly that pattern today: one
word, revenue, on five surfaces at five bases, with a contradiction of ₪686,475
between two figures on one page. Section 3.4's basis rule is written to survive
five bases rather than to reduce them to one. Google Ads reduces them to one by
putting budget on exactly one layer. The spec never names the single home of a
money figure.

### Reason 3: two destinations for one object at two zooms

Linear's evidence, measured live in the demo: filtering between Active, Backlog
and All is a segmented control at the top of the content, `Cmd B` flips list and
board, and `Shift V` opens display options, all without leaving the destination
(`07-references.md:90`, `:119`). Zoom is a display property.

Meridian's Plan and Days are the week and the day of one plan, and the spec keeps
the same three views on Plan (grid, daypart, timeline) while moving the editor to
Days (`spec.md:144`). So a timeline of breaks renders in two destinations over
the same `schedule.rows`. `01-surfaces.md:103-110` measured that exact
duplication between Optimizer and Schedule and called it the first of the three
heaviest structural facts for the rebuild. The spec deletes the duplication and
reinstates it across the new boundary.

Linear also gives the newcomer a way back that the spec has no equivalent for:
opening a record keeps its place, `1 / 31` with up and down arrows
(`07-references.md:127`). Section 3.4's invariant sends people down into objects
and never says how they walk the set they came from.

### Where the spec beats both

Section 3.4's "a figure without a basis does not render" plus tri-state honesty
is stronger than anything in either reference. Google Ads' status vocabulary is
excellent and carries no basis discipline at all; Linear has no money. Stripe is
the only reference at this level and the spec matches it
(`07-references.md:200-205`). The honest empty states measured in five places
(`06-baseline.md:445-450`) survive into the architecture as a rule rather than a
habit. That is a genuine advantage and it should be defended when the seven get
renegotiated.

---

## 5. Evidence dropped without a reason

**Verdict: all eight ABSENT gaps are addressed. Ten DEAD, DUPLICATE or
load-bearing findings are dropped with no mention and no entry in section 9.**

I grepped `spec.md` for each. The count is occurrences in the whole spec.

| Dropped finding | Source | Mentions in spec |
|---|---|---|
| `data/Spots.csv`: 50,386 rows x 36 cols, `break_id` **9,492 distinct**, `position_in_break`, `revenue_ils`, all three 100 percent populated, revenue summing ₪306.9M, read by nothing | `02-api-and-data.md:626`, `:684`; I re-measured all four figures | **0** |
| `frequency_rules.csv`, one row, `DEFAULT_ONE_PER_BREAK`, the only rule that fires, drops **56 of 175** daily spots | `02-api-and-data.md:426` | **0** |
| `GET /api/settings/controls`: 4,253 B of bilingual lever schema with help text and bounds, built so the panel would not hardcode, orphaned | `02-api-and-data.md:267-270` | **0** |
| `AssistantUpload.jsx` DUPLICATE-OF `UploadCenter`, and a whole second upload system with its own store | `01-surfaces.md:575`, `02-api-and-data.md:437-443` | **0** |
| `InventoryHeatmap` (`TVBreakDashboard.jsx:5159-5172`), the one component graded DEAD by construction | `01-surfaces.md:121`, `:805` | **0** |
| `models/tv_break_posterior.pkl`: 1.2 MB listed in the operator's source files, changes nothing, while the 21 KB artifact that drives every retention number is hidden from that list | `04-training-vs-runs.md:368` (F11) | **0** |
| `data/Programmes - today.csv`, 125 rows, zero readers repo-wide | `02-api-and-data.md:738` | **0** |
| The make-good as an object you create: shortfall, compensating inventory, approval, link back to the flight | `05-gaps.md:290`, `:322` | **0** ("compensating" 0, "make-good entity" 0) |
| `GET /api/optimizer-plan`, 53 KB, the richest payload in the API, orphaned | `02-api-and-data.md:271-274` | **0** |
| Bar 3, never regress what already works | the brief, `docs/ux-gauntlet-prompt.md:178` | **0** ("regress", "Bar 3", "never get worse" all 0) |

Three of these matter enough to name individually.

**`data/Spots.csv` is the largest.** The spec's central structural claim is that
"the product has no object below the programme segment, and no object joining the
plan to the money" (`spec.md:79-81`), and it schedules the break entity as "the
largest new capability in the spec" (`spec.md:164`). On disk, unread, sits a file
with a populated historical break identity, a populated position within the
break, and populated per-spot revenue. I am not claiming it is fit for purpose. I
measured that the columns are complete and that `02-api-and-data.md:571` found
this file's `advertiser_id` column is derived garbage, so its quality is mixed
and unmeasured. The defect is that the spec neither uses it, nor tests it, nor
rejects it with a reason, while building both of the things it appears to carry.

**Bar 3 is absent from a document that dissolves or merges twelve of seventeen
surfaces.** The job stories carry Bar 3 explicitly (`job-stories.md:551`: "Bar 3
applies with full force: it may not get slower or harder"). The spec has no
regression register, no "what works today that this must not lose" list, and no
critic assigned to the three-way comparison. P15 is an integration critic over
the new product only (`spec.md:640-642`). The Reports dissolution in 1.3 is the
first thing that register would have caught.

**The brief's model-improvement mandate is gone.** `docs/ux-gauntlet-prompt.md:94`
makes it a titled section: "The model is in scope, and it must be the best one
this data can support ... Keep improving the retention model, the audience model
and the optimizer wherever measurement supports it." The spec turns this into a
console that displays gates and a view that shows what a candidate would move
(`spec.md:286-288`), which is showing, not improving. `models/candidates/` holds
five artifacts and `scripts/estimate_candidate_revenue_movement.py` already
computes the money each would move. No build piece adopts, rejects or improves
anything. This is not listed in section 9 with a reason. INFERRED cause: the
frozen job stories contain no model-improvement story, so the spec inherited the
gap from JS-16 rather than creating it. Either way it needs a line in section 9
or a piece.

---

## 6. Would two builders collide

**Verdict: yes, on every wave. The pieces are allocated by surface and the work
is shaped by layer. Six named collisions, three of them fatal to parallelism.**

`spec.md:589-591` claims each piece "owns a disjoint set of files and a disjoint
set of endpoints". I mapped the routers to the pieces.

**Collision 1, fatal. `kairos_api/insights_api.py` serves six pieces.** Measured
just now:

```
insights_api.py:400  GET  /api/yield-per-second    -> P5 Money, and W0-B (5.01 s)
insights_api.py:518  GET  /api/gold-breaks         -> P3 Days
insights_api.py:622  POST /api/scenario-compare    -> P2 Plan, and W0-B (20 to 25 s)
insights_api.py:630  GET  /api/model/audience      -> P8 Model console
insights_api.py:646  GET  /api/make-good-alerts    -> P13 Pacing board
```

P2's bar requires net of retention exposed for comparison, which is a change to
`scenario-compare`. P5 owns the yield money. P8 must wall `model/audience` behind
affiliation. W0-B must make two of them fast. Five pieces across two waves plus
wave zero, in one file, with no split assigned.

**Collision 2, fatal. `kairos_api/dashboard_api.py` is 1,827 lines and serves
four wave-one pieces, and no piece splits it.** `spec.md:594-597` names the
problem in its own words: "`TVBreakDashboard.jsx` is 6,236 lines ... and
`dashboard_api.py` is 1,827 lines. If ten builders edit those two files at once,
nothing is independent. So wave zero splits them." Wave zero then splits one of
them. W0-C's files are `tv-break-dashboard/src/**` (`spec.md:605`). Nobody owns
the backend half. Measured routes:

```
dashboard_api.py:1633 /api/compliance             -> P6 Rules
dashboard_api.py:1638 /api/overview               -> P1 Today
dashboard_api.py:1695 /api/schedule               -> P2 Plan
dashboard_api.py:1711 /api/schedule/segments      -> P3 Days
dashboard_api.py:1728 /api/schedule/segment/{id}  -> P3 Days
dashboard_api.py:1803 /api/break-operations       -> P3 Days
dashboard_api.py:1817 /api/break-decisions        -> P3 Days
```

**Collision 3, fatal. W0-A and W0-B collide inside wave zero, which is the one
wave the spec insists is mutually independent.** W0-A owns
`kairos_api/advertisers.py` (`spec.md:603`). W0-B owns "the eight worst
endpoints" (`spec.md:604`). Two of the eight worst live in that file:
`advertisers.py:211` (`GET /api/advertisers`, 12 to 24 s) and `advertisers.py:229`
(`GET /api/advertisers/stats`, 12 s). W0-A additionally owns "name-to-id
resolution on the daily path", which is the same daily spot path P5 needs for
per-spot attribution and P3 and P11 need for the break entity's engine seam
(`spec.md:498`).

**Collision 4. P8 contradicts the contract that closes the same section.**
`spec.md:622` gives P8 "Depends on: none, entirely separate file tree".
`spec.md:658-660` says "The shared surface is exactly two things: the design
tokens from W0-C, and the vocabulary". A console with no dependency on W0-C
cannot consume W0-C's tokens, and if its file tree is under
`tv-break-dashboard/src/**` it is inside W0-C's declared ownership. One of the
two statements has to give.

**Collision 5. P1 owns a component that renders on every workspace.**
`spec.md:615` gives P1 "the two-state freshness split from section 4.3". I
checked the mount: `ScheduleStalenessBanner` is imported at
`TVBreakDashboard.jsx:102` and mounted exactly once at `:2498`, in the shell. It
is on every page, which `06-baseline.md:443` also measured. P1 owns a global
control that P2, P3, P5, P6 and P7 all render.

**Collision 6. Nobody owns the cross-cutting obligations.** Five of them, each
touching most pieces, each stated as a rule and assigned to no builder:

1. The `job` field and per-job landing (section 2). Zero appearances in section 8.
2. The affiliation wall, `can_edit` on every walled endpoint, every surface
   reading its own affiliation (section 4.4).
3. The competitor boundary removal (`spec.md:717`). It requires backend changes:
   `GET /api/schedule`'s 200-row projection is 96 `קשת 12`, 73 `כאן 11`, 28
   `עכשיו 14` and 3 of the operator's own channel, and
   `GET /api/break-operations` returns all four channels. Both live in
   `dashboard_api.py`, which nobody owns.
4. The vocabulary renames (4.5), which touch every surface's copy in two
   languages.
5. Splitting the regulatory guardrails and `audience_model_activation` out of
   `KairosSettings`, which changes `core.py`, a module every piece reads.

There is also a bar that no piece can meet with the tools its owner is given.
**P6's preview in 3 s** (`spec.md:620`) is measured today at 16.55 s scoped and
55.60 s unscoped (`06-baseline.md:430`), because `/api/constraints/effect` runs
`_optimize_one_day` twice. **P3's money within 500 ms of the drop**
(`spec.md:617`) is the same class of computation. W0-B is scoped to "the read
paths in `kairos_api/`, a cache module", and a cache cannot answer a placement
nobody has made yet. The work that would meet these two bars is engine work and
it is in no piece's file list.

---

## 7. Is anything dishonest

**Verdict: nothing fabricated. One bar is impossible from the data, two counts
are misattributed, one evidence grade is laundered, and the owner-blocked marking
is applied inconsistently.**

### Finding 1: "45 of 45 advertisers named" cannot be met, and the blocker is not in the open-questions list

This is the most serious finding in the document, because W0-A is a wave-zero
prerequisite that P3, P4 and P5 all wait on.

`spec.md:603` sets the bar: "45 of 45 advertisers named." `spec.md:449-452`
prescribes the method: "Add `name` and `aliases` to `advertiser_rules.csv`,
exactly the columns `agencies.csv` already has, and resolve the daily file's
Hebrew names to ids the way agencies already resolve."

I measured why agencies resolve and advertisers cannot:

- `agencies.csv` carries `name`, `display_name` and `aliases`, and its `name`
  column literally equals the daily file's `משרד / MB` values. I checked: **9 of
  9** daily agency names are present in `agencies.csv.name`. Resolution is a
  string equality that already holds on disk.
- `advertiser_rules.csv` has 8 columns, none of them a name or an alias, and its
  `notes` column is **empty in 45 of 45 rows**. Its ids are `ADV_01..ADV_45`.
- The real advertiser vocabulary is 41 Hebrew names, present in
  `agency_advertisers.csv` (41 rows, `source` = `observed` on all of them) and in
  the daily file (41 distinct). Intersection with the 45 ids: **0**.

So adding the columns is trivial and filling them is impossible. Nothing on disk
says which `ADV_xx` is `בנק הפועלים`, and 45 ids cannot map onto 41 names anyway.
Meeting the bar as written would require inventing at least four advertisers,
which the honest-math law forbids. The honest options are an owner-supplied
mapping, or re-keying the store on the observed names and declaring the 45
synthetic rows demo data, which orphans their premiums (`ADV_02` carries
`default_premium` 1.27).

`spec.md:719-737` lists five things that still need a human decision. This is not
one of them, and `spec.md:504` says the money drill needs "Nothing beyond
advertiser identity" from the owner, which is circular given the above.

### Finding 2: two counts are misattributed

`spec.md:203-206`: "the same two words name both activities 159 and 124 times."
`04-training-vs-runs.md:670-671` measured `recompute` at 159 in the UI and 124 in
the backend, and `rebuild` at 9 and 85. The figures belong to one word. The
correct sentence is that two words name both activities, one of them 159 and 124
times and the other 9 and 85. `job-stories.md:658` states it correctly for the
single word, so the error is the spec's compression.

`spec.md:292-294`: "Six such flags exist with environment-variable twins."
`scripts/compute_measured_coefficients.py` has exactly 6 `add_argument` calls and
I read their names: `--series`, `--counterprogramming`, `--placebo-correction`,
`--interval-calibration`, `--moderated-variances`, `--output`. `--output` is an
output path, not a gate override. `04-training-vs-runs.md:651` itself lists five.
The spec took the larger of two numbers its own source disagrees on, and the
larger one includes a non-gate.

### Finding 3: the people table launders discovery's evidence grades

`03-people.md` graded each persona. Five are explicitly INFERRED with what they
were inferred from: the analyst (`:185`), the data steward (`:438`), the revenue
and yield owner (`:464`), the compliance owner (`:492`), the deployment owner
(`:590`). The spec's section 2 table (`spec.md:38-55`) presents all sixteen with
equal confidence and no grade. The prose does keep "accountabilities"
(`spec.md:30`), which is the right word, but the table is what a builder will
read. Section 10 records five evidence conflicts and does not record this
downgrade of uncertainty.

### Finding 4: owner-blocked marking is inconsistent

P12 is correctly flagged "Owner-blocked for the technical half" (`spec.md:631`).
Two other bars are equally owner-blocked and unflagged:

- P1's bar demands "three answers" (`spec.md:615`), and the third needs the plan
  target that `spec.md:496` and `spec.md:721` both say the owner has not
  supplied.
- P2's bar demands "180 s to a published plan version" (`spec.md:616`), while
  `spec.md:497` and `spec.md:736` say what publishing means is an open owner
  decision.

A builder graded against P1's bar in the loop will either fail or invent a
target. Section 9.2 says the right thing; the build order does not repeat it.

### Finding 5: two evidence conflicts section 10 did not record

Section 10 exists precisely to record these, so their absence is worth naming.

- **The HEAD disagreement.** `05-gaps.md:8`, `03-people.md:9` and
  `04-training-vs-runs.md:9` name `5a80a709`. `06-baseline.md:5` names
  `342a2896`. I verified `git rev-parse HEAD` is `5a80a709` and that `342a2896`
  is an ancestor **31 commits back**. This matters more than the other conflicts
  in section 10 because `06-baseline.md` supplies every "Baseline today" figure
  frozen into the job stories, which Bar 3 grades against. I did not determine
  which tree the baseline browser actually ran against, only that the two
  provenance lines disagree by 31 commits.
- **`dayKeys` at 585 versus 586.** `05-gaps.md:681` says 586, `07-references.md:513`
  says 585. The spec chose 585 and 585 is correct. Choosing right without
  recording the conflict still leaves a critic to rediscover it.

### What is not dishonest

No persona is invented; all sixteen map to `03-people.md`. Every measured figure
I re-checked was accurate. Section 10's five recorded conflicts are handled well,
particularly the test-count row, which refuses to claim a pass count it did not
measure, and the competitor-lane row, which flags a visible removal for the owner
rather than doing it silently. Section 9's twelve declared omissions are the
right instinct and mostly the right list.

---

## The single largest remaining gap

**The build order allocates ownership by surface, and the work is shaped by
layer. Section 8 therefore cannot be executed in parallel as written, and the
five cross-cutting rules the spec is proudest of belong to nobody.**

Concretely, in one pass, section 8 needs three additions.

1. **A module ownership table**, backend as well as frontend, one row per module
   with exactly one owning piece. It must resolve the three measured collisions:
   `insights_api.py` (5 routes, 5 pieces), `dashboard_api.py` (1,827 lines, 8
   routes, 4 pieces), `advertisers.py` (W0-A and W0-B and P4). The honest
   resolution is that wave zero splits `dashboard_api.py` and `insights_api.py`
   by workspace before wave one starts, exactly as it splits
   `TVBreakDashboard.jsx`, and the spec already argues for this at
   `spec.md:594-597` and then does half of it.

2. **A cross-cutting piece, W0-D, that owns what no surface owns**: the `job`
   field and per-job landing; the affiliation wall on the four open read routes
   plus `can_edit` on every walled endpoint plus session affiliation in the
   shell; the competitor scoping of `/api/schedule` and `/api/break-operations`;
   the vocabulary rename across both languages; and lifting the regulatory
   guardrails and `audience_model_activation` out of `KairosSettings`. Each of
   these is a rule the spec states and no builder is told to build.

3. **An engine-performance piece or an explicit deferral**, because P6's 3 s
   preview and P3's 500 ms objective evaluation are measured today at 16.55 s and
   11.3 s, both running `_optimize_one_day` twice, and W0-B's stated scope is
   read caching, which cannot answer a placement that does not exist yet.

Two further one-line fixes belong in the same pass because they are blockers, not
polish: add the advertiser id-to-name mapping to section 10's open decisions and
restate W0-A's bar as "every advertiser that appears in the daily file resolves
to a named record", and add the Hebrew label for each of the seven workspaces,
History and account settings to section 4.6, since the product is Hebrew-first
under Bar 4 and the spec names eleven new destinations in English only.

---

## Verdict

**NOT READY for the build loop.**

Three reasons, in order.

1. **Section 8 is not parallelisable as written.** Three backend modules each
   serve four to six pieces, `dashboard_api.py` is named as the problem and
   assigned to nobody, W0-A and W0-B collide inside the one wave the spec calls
   mutually independent, and five cross-cutting rules have no builder. Ten
   builders launched against this would collide on day one.

2. **W0-A's bar cannot be met from the data.** There is no artifact anywhere
   mapping `ADV_01..ADV_45` to the 41 real Hebrew advertiser names, and 45 cannot
   map onto 41. W0-A gates P3, P4 and P5, and the blocker is not in the
   open-questions list.

3. **Two new training-versus-runs leaks are created by the spec itself**: events
   authoring filed under training against the spec's own taxonomy, and model
   releases merged into an affiliation-blind History timeline reachable from
   every object header.

None of the three requires rethinking the architecture. The people model, the
entity model, the identity-first ordering, the vocabulary and the two dashboards
are strong and well evidenced, and section 5.3's argument is the best reasoning
in the document. What is missing is an ownership map that matches the shape of
the code, one owner question that was never asked, and two seams. That is one
revision pass, not a rewrite.
