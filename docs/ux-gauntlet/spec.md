# Meridian: the architecture the jobs imply

Revision 2, written 2026-07-31 against HEAD `5a80a709` (verified with
`git rev-parse HEAD`), from the seven discovery reports in
`docs/ux-gauntlet/discovery/`, the blind critique in
`discovery/08-spec-critique.md`, and my own measurements on the running
instance at `http://127.0.0.1:8010`. It is a recommendation, not a survey.

A further pass on 2026-07-31 closed the six items the second blind critique
(`discovery/09-spec-critique-2.md`) left open; section 11 lists them. That pass
measured at HEAD `eef9ff91`, two commits later. `git diff --name-only
5a80a709..eef9ff91` filtered to anything outside `docs/ux-gauntlet/` returns
nothing, so every code and data measurement in this document still stands at the
later HEAD, and the ones that did not reproduce failed for their own reasons and
are corrected in place.

Revision 1 was returned NOT READY. The three blockers were the build order
(allocated by surface while the code is shaped by layer), one bar the data
cannot satisfy, and two new training-versus-runs leaks. All three are closed
below, and so are the nine secondary findings. Section 11 lists what changed
and why, so a critic can check the fixes rather than re-derive them.

The companion `docs/ux-gauntlet/job-stories.md` is frozen; its amendment log
records the two additions this revision forced. The choices only the owner can
make are in `docs/ux-gauntlet/decisions-for-owner.md`, five of them, each
blocking a named piece.

---

## 1. What Meridian is

Meridian is where a channel plans its commercial breaks: where each break goes
in the week, what it is worth, and which ads fill it. Every one of those
decisions is priced against a retention model measured on the channel's own
audience, so what a break earns and what it costs in viewers are both numbers,
and both are visible before the decision is made.

That second sentence is the whole product. It is also the thing today's
experience hides: measured on the running app, dragging a break moves it and
changes no figure on the page, because the four schedule editor files contain
zero occurrences of "revenue" or "retention".

---

## 2. The people, and the one door each of them opens

Discovery found sixteen distinct accountabilities and one served persona.
Sixteen accountabilities do not mean sixteen homes. They mean sixteen
questions, and questions cluster.

`03-people.md` graded each persona, and five of the sixteen are explicitly
INFERRED rather than evidenced. Revision 1 dropped those grades. They are
restored here, because a builder reading this table needs to know which
persona is a measured fact and which is a reasoned guess.

**A door is a named landing view, not a destination.** Section 3 ships five
workspaces. Eleven human roles land on eleven different named views of those
five, the way Linear's Inbox and My issues are two named views of one object
family (`07-references.md:86-92`). Two more land outside the five: the
administrator on the account menu, where they already are today
(`TVBreakDashboard.jsx:2288`, so JS-15's passing baseline survives untouched),
and the model steward on the company shell. That is thirteen roles, thirteen
doors, no role sharing a first screen with another and no destination invented
for a role.

The count of ambiguous roles in the blind walk was five, and the count that
could not land at all was three. Every one of the eight is named in the table
below with the view that fixes it, so a critic can check the fix rather than
re-walk the product.

| Person | Evidence grade | Their one question | Door (the named view they land on) | Story |
|---|---|---|---|---|
| General manager, executive viewer | EVIDENCED (`03-people.md:151`) | Is this week on plan and what needs me | **Today** | JS-1 |
| Planner | EVIDENCED (`03-people.md:213`) | What should next week look like | **Plan, week** | JS-2 |
| Scheduler | EVIDENCED (`03-people.md:245`) | Where does this break go | **Plan, the day I am fixing** | JS-3 |
| Traffic operator | EVIDENCED (`03-people.md:380`) | Is what airs tonight correct | **Plan, tonight's breaks** (הברייקים של הערב) | JS-7, JS-8 |
| Programming representative | EVIDENCED (`03-people.md:277`) | When may a break not be placed | **Rules, restrictions** (הגבלות) | JS-4 |
| Compliance owner | INFERRED (`03-people.md:492`, from the `Legal / Ops` report owner at `catalog_api.py:507`) | Are we inside the licence | **Rules, the licence** | JS-14 |
| Revenue and yield owner | INFERRED (`03-people.md:464`, from the two `Revenue` report owners at `catalog_api.py:508-509`) | What is a second of airtime worth | **Rules, the rate card** | JS-13 |
| Account manager | EVIDENCED (`03-people.md:308`) | Who is the client and what did we promise | **Clients, all clients** | JS-5 |
| Campaign manager | EVIDENCED as a need with machinery, unserved (`03-people.md:352`) | Is what we promised being delivered | **Clients, campaigns on air** | JS-6 |
| Analyst | INFERRED as a distinct person (`03-people.md:185`), EVIDENCED as a distinct set of needs | What did we earn, gross and net | **Clients, delivered money** | JS-9 |
| Data steward | INFERRED (`03-people.md:438`, from per-file cadences plus the `Data` report owner at `catalog_api.py:510`) | Is the engine reading true inputs | **Sources, today's inputs** | JS-12 |
| Account administrator | EVIDENCED (`03-people.md:517`) | Who is in the system and what may they do | **Account menu, accounts** (unchanged from `TVBreakDashboard.jsx:2288`) | JS-15 |
| Model steward (company) | EVIDENCED as a role with artifacts and no surface (`03-people.md:539`) | Is the model fit to ship | **Model console** via the context switcher in the account menu, section 4.7 | JS-16, JS-19 |
| New starter, first morning | EVIDENCED as a story, not a role | Which of these is mine | **Today, with the job picker**, section 2.2 | JS-11 |
| Deployment owner | INFERRED (`03-people.md:590`, from the operational assumptions the code states about itself) | Is it up, enforced and secret-safe | Outside the product, plus one honest banner | JS-17 |
| Channel-affiliated account | EVIDENCED as an identity class, not a person (`03-people.md:611`) | Not a person | The wall in section 4 | JS-18 |
| Kai | EVIDENCED (`03-people.md:628`) | Not a person: a delegated actor | Docked everywhere | JS-10 |

**One door has to carry an answer it does not carry today, and it is named here
rather than left to a builder.** The revenue and yield owner's question is "what
is a second of airtime worth", and the product already answers it:
`GET /api/yield-per-second` returns `yield_per_second: 142.7044` with a
`basis.formula` naming every input (I fetched it this session). That endpoint is
served from `insights_api.py:400`, and revision 2 handed the module W0-1 splits
out of it to Plan. So the answer sat on Plan while this person's door was Rules,
and a door that does not answer the question it is chosen for is not a door.
**The rate card carries the figure.** Rules, the rate card opens with what a
second is worth under the card as saved, and the same figure recomputed under the
unsaved edit is JS-13's "money delta on screen before the save". It is one
quantity, computed once, printed with its scope, and it is the same number Plan
shows for the planner's comparison, read from the piece that owns it. Section
8.2 therefore assigns `yield_api.py` to **P5** and P2 consumes it read-only under
the published-contract rule of section 8.8. The person does not move; the answer
does. JS-13's target still says "45 s from opening **Money**", a destination this
document does not ship; amendment 4 in the job stories records that Money reads
as Rules, the rate card, and the 45 s does not move.

### 2.1 Job is a new dimension, orthogonal to role

Today `role` answers "what may this account write" and nothing answers "what
is this person's job". A traffic operator and a planner are both `operator`.
The account gains a `job` field with a safe default, which decides the door and
the order of the sidebar. It does not decide permission. Permission stays with
role and affiliation, so a misconfigured job costs a person a good first screen,
never their access or their safety. `auth_store.normalize_affiliation`
(`kairos_api/auth_store.py:175`) is the pattern to copy: a field that defaults
safely for every existing record.

The enumerated job list is exactly the thirteen door-bearing roles above. The
seventeen-row table also holds the new starter, which is the `unset` state
rather than a job, and three rows that are not jobs at all: the deployment
owner, who works outside the product, the channel-affiliated account, which is
an identity class, and Kai, which is a delegated actor. So the field takes
**thirteen values plus `unset`**, and `unset` is the default for every existing
record. **W0-4 owns this field, the list and the landing map**, section 8.3.

### 2.2 The job picker, so JS-11 does not depend on an admin getting it right

JS-11's target is 300 s with 0 wrong pages opened, for a person who has been in
the building for one morning. Revision 1 made that depend entirely on an
administrator having set `job` correctly beforehand. That is not a design, it is
a hope.

An account whose `job` is `unset` lands on **Today** with one card above the
fold: "Which of these is your job", thirteen rows, each naming the job in the
person's own trade word and the door it opens. Choosing one writes `job` on the
account and lands them. It is reachable afterwards from the account menu, so a
wrong choice is one click to correct. It renders for nobody whose `job` is set,
so it costs an existing person nothing. The model steward's row appears only on
a company-affiliated account, per section 4.7, so a channel account is never
shown that the other side exists.

This is the only adaptive element in the product. **Everything else is
identical for everyone.** Hiding navigation from people is how products become
mysterious. Every operator sees all five workspaces. What changes with `job` is
where they land, what their sidebar puts first, and what Today leads with.

---

## 3. The information architecture

### 3.1 The spine

The single largest structural fact discovery found is that the product has no
object below the programme segment, and no object joining the plan to the
money. Everything else follows from that.

Meridian has two spines and they meet at one object.

```
  the plan spine        WEEK  ->  DAY  ->  BREAK  ->  SPOT
                        (plan)  (board)   (pod)     (one ad)
                                                      ^
  the commercial spine  AGENCY -> ADVERTISER -> CAMPAIGN -> FLIGHT
```

The spot is the join. Today the two are, in the measured words of the API
investigator, two disconnected universes whose only bridge is a shared channel
name and a shared date (`02-api-and-data.md:702`). The weekly universe carries
all the money on the Overview and knows nothing about advertisers; the daily
universe carries real advertisers, real agencies and real rebates over 175
spots and knows nothing about the plan.

### 3.2 One classification axis, and the count it produces

The blind A/B found Linear and Google Ads both read clearer than revision 1,
and named the reason: Google Ads sorts on one axis and states the rule
("There is no fourth layer invented for convenience", `07-references.md:214`),
while revision 1's seven workspaces were sorted on four axes at once, so a
newcomer could not form a rule that predicts where the eighth thing goes.

The axis is: **every destination is a family of objects you can open, except
one home.**

That produces the count rather than asserting it:

| Destination | What family of objects it holds | Why it is a family and not a topic |
|---|---|---|
| **Today** | none: it is the home | The only time scope in the product. It exists because JS-1 requires a landing with zero clicks. It is the single declared exception to the axis and it is named as one. |
| **Plan** | the plan spine: week, day, break, spot | Discovery's first heaviest structural fact (`01-surfaces.md:810`) |
| **Clients** | the commercial spine: agency, advertiser, campaign, flight | The other of the two disconnected universes |
| **Rules** | the things that constrain a plan or a price: restriction, guardrail, rate card, commercial rule, frequency rule | Each is an authored record with a scope and an effect. `02-api-and-data.md:412` measured five rule grammars for one idea |
| **Sources** | the inputs a run reads: seven upload kinds, eight source files, the model version | Each is a record with a state, a row count and an `in_use` verdict (`02-api-and-data.md:513`) |

One home plus four object families. A newcomer learns one sentence and can
predict where a new thing goes: if you can open it and it has its own address,
it belongs to whichever of the four families it is a member of.

Plus four things that are addressable and are **not** workspaces, stated
explicitly so nobody counts five and finds nine:

- **Kai**, docked on the right of every surface, opened with `Cmd J`
  (`07-references.md:413`). It is already a dock; today it also occupies
  navigation slot 15 whose hash resolves to Overview
  (`TVBreakDashboard.jsx:1395-1398`), which is a lie in the URL bar.
- **History**, one attributed timeline, reachable from the account menu and
  from every object's own header. Today this is three separate things:
  `#Versions`, the Settings activity log, and an in-memory bell feed.
- **Account settings**: channel, profile, language, accounts. Reachable from
  the account menu. Not a workspace, because nothing a person does daily lives
  there.
- **The Model console**, a different shell for `affiliation = company` only,
  reached through the context switcher specified in section 4.7.

**Nine addressable places. Five workspaces. Seventeen navigation entries
today.**

And one global control the product does not have: **`Cmd K`**, a grouped
command palette that prints each action's own shortcut on its row, the way
Linear's does (`07-references.md:101`). Zero hits today for command palette,
cmd k, hotkey or shortcut anywhere in the frontend (`05-gaps.md:647`).

### 3.3 Zoom is a control, not a destination

The A/B's third criticism: revision 1 kept grid, daypart and timeline on Plan
and moved the editor to Days, so a timeline of breaks rendered in two
destinations over the same `schedule.rows`. That is the Optimizer-versus-
Schedule duplication (`01-surfaces.md:103-110`) reappearing under two new
names, and `01-surfaces.md:810` names it the first of the three heaviest
structural facts for the rebuild.

Fixed by deleting the second destination. **Plan is one destination with a
zoom control in the content**, `Cmd B` to step it, exactly Linear's mechanism
(`07-references.md:119`):

```
  Plan  [ week | day | break ]        <- a segmented control in the content
        week   the aggregate: objective, run, compare, publish, supply, target
        day    one broadcast day as a timeline: drag, undo, gold, restrictions
        break  one break as a pod: its spots, their durations, their order
```

Five of thirteen roles no longer land on two surfaces that render the same
thing. The scheduler's door is Plan at day zoom on the day they are fixing.
The traffic operator's door is Plan at break zoom, filtered to tonight, named
in their own word: **הברייקים של הערב**. Frame.io's device, where a status is
also a saved place you can navigate to (`07-references.md:328`).

### 3.4 Money lives on exactly one layer

The A/B's second and sharpest criticism: Google Ads puts budget on exactly one
layer, and revision 1 put money on five surfaces. `01-surfaces.md:816` measured
the consequence of that pattern today: one word, revenue, on five surfaces at
five bases, with a contradiction of ₪686,475 between two figures on one page.

Revision 1 wrote a basis rule designed to survive five bases. This revision
reduces them to one.

> **The money layer is the break.** Every money figure in Meridian is a sum
> over breaks. A break carries exactly two money quantities, **projected** and
> **delivered**, and every figure anywhere is one of those two summed over a
> named scope. A figure that cannot resolve to breaks does not render as money.

Concretely:

- **Projected** is what the plan says the break will earn: the optimizer's own
  per-break CPP revenue (`kairos/optimize/objective.py:32`,
  `revenue = cpp * rating_points * (duration / 30) * premium`).
- **Delivered** is the sum of the priced spots that actually sat in that break
  (`kairos/export/spots.py`, the 175-row ledger, 119 priced and 56 dropped).
- A week's money is those two summed over the week's breaks. An advertiser's
  money is delivered summed over the breaks holding that advertiser's spots.
  A daypart's money is the same sum over a filter. There is one quantity per
  column and one drill: **figure -> breaks -> spots**, which is Stripe's
  two-level drill (`07-references.md:186`).
- The scope is printed with the figure, never in a tooltip. Two figures with
  different scopes may sit on one page only if both scopes are printed. That
  is what would have caught the ₪686,475: it is not a contradiction, it is
  ₪10.12M over 7 days beside ₪10.81M over the same 7 days summed from a
  different projection, and neither printed its scope.

This is why the break entity is the keystone rather than one capability among
many: it is the money layer. Nothing in section 3.4 is true until it exists.

**The honest limit, stated here rather than discovered later.** Projected and
delivered are not the same currency today and will not become so in this
rebuild. Projected comes from the weekly plan (8,704 segment rows, no
advertiser); delivered comes from one daily file (175 spots, one day,
2025-04-27). They overlap on zero dates. So the break board shows delivered
where a spot ledger covers the break and an honest empty state everywhere
else, and the two are never summed into one figure. Closing that gap needs a
delivery feed, which is owner decision 4.

### 3.5 What merges, what splits, what disappears, what is new

Fifteen of the eighteen rows below trace to a job story. **Three do not, and
they are marked, because a lead reading a blanket claim will not look for the
exceptions.** Measured duplication and a law are both better reasons than a
story; they just have to be labelled as what they are.

| Today | Verdict | Where it goes | Justification | Traced to |
|---|---|---|---|---|
| Overview | Rebuild, rename | **Today** | It answers two of its three questions today and has no target to answer the third against (`06-baseline.md:91`) | JS-1 |
| Optimizer | Merge | **Plan, week** | Same four tiles, same `ComplianceLedger` with the same prop, same frontier as Overview, same `PlanningCanvas` / `DaypartView` / `TimelineView` over the same `schedule.rows` as Schedule | **Measured duplicate**, not a story (`01-surfaces.md:90-112`) |
| Schedule | Merge | **Plan**, as the week and day zooms | The planner works the week as an aggregate; the scheduler works one day as a timeline. One destination, one zoom control | JS-2, JS-3 |
| Inventory | Merge | **Plan, week**, supply view | JS-2 needs sellable supply beside the objective. Its money column is a dash on all five dayparts because the loaded spots source has no revenue column | JS-2 |
| Break Library | Merge | **Plan, break zoom**, as the day's break list | Its top rows are byte-identical to Overview's priority decisions, and 18 of its 79 adjacent pairs are out of the descending revenue order it claims to rank by | **Measured duplicate and a defect**, not a story (`01-surfaces.md:188-195`) |
| Campaigns | Merge | **Clients** | A campaign is a child of an advertiser, not a peer page. The current page is a historical rollup with revenue a dash on 50 of 50 rows and the advertiser blank on 50 of 50 | JS-5, JS-6 |
| Forecasts | Merge | **Plan, week**, compare view | Comparing is a step of planning. Today the comparison is a separate page reached by a top-bar button that navigates instead of comparing (`TVBreakDashboard.jsx:2406-2417`) | JS-2 |
| Events calendar | Split | read overlay on **Plan**; authoring to **Rules, the calendar** | Event writes are CONFIGURATION (`04-training-vs-runs.md:131`), not training. They stay operator-side, company-gated by affiliation. The model-health panel it currently carries moves to the Model console | JS-4, JS-18 |
| Reports | **Keep, do not dissolve** | **Sources, downloads**, plus each export placed on the object it exports | Revision 1 dissolved a LOAD-BEARING surface (`01-surfaces.md:301`) on the strength of a story about a different person. Its five reports carry four owner departments (`catalog_api.py:506-510`: Traffic, Legal / Ops, Revenue, Revenue, Data), so at least three personas other than the analyst use it. Bar 3 forbids losing it | **Bar 3**, plus JS-9 for the added on-object exports |
| Data | Split | uploads and source files to **Sources**; "Model and parameters" to the **Model console** | The third tab is a model-health dashboard on a run surface, with a "Needs attention" drift chip the operator cannot act on | JS-12, JS-16 |
| Advertisers | Merge | **Clients** | | JS-5 |
| Agencies | Merge | **Clients** | One entity family, one place | JS-5 |
| Pricing | Merge | **Rules, the rate card** | A rate card is a rule that prices a spot. It is not a topic and it is not money: it is what produces money | JS-13 |
| Overrides | Merge | **Plan, day zoom**, as acts on a break | A pin is something you do to a break, not a console you visit. Gold currently has five surfaces for one concept (`01-surfaces.md:430-435`) | JS-3 |
| Kai assistant | Demote to dock | dock on every surface | It already is a dock; the navigation entry resolves to Overview | **Measured defect**, not a story (`01-surfaces.md:51-54`) |
| Restore changes | Merge | **History** | JS-3 needs undo where the work is, not on a separate page measured at 27 to 53 s to load | JS-3, JS-10 |
| Settings | Split | guardrails, protected content, frequency, restrictions to **Rules**; objective and pacing to **Plan, week**; channel, profile, locale, accounts to **account settings** | A programming representative should not register an objection by scrolling past `risk_lambda` and a pace denominator floor (`06-baseline.md:201-204`) | JS-4 |
| Inventory heatmap | **Delete** | nowhere | `InventoryHeatmap` (`TVBreakDashboard.jsx:5159-5172`) is a hard-coded empty state with no data path at all, the one component graded DEAD by construction. Deleting it is the only deletion in this spec and the proof is the fourteen lines themselves, 5159 to 5172, plus its single call site at `:3131` | **Proven dead**, not a story |

**New, and not a rearrangement of anything:**

| New thing | Why | Story |
|---|---|---|
| The break as a first-class object with contents | Nothing below the segment exists. This is the money layer and the largest new capability in the spec | JS-7 |
| The break board (the pod as a timeline) | The physical thing a traffic operator assembles | JS-7, JS-8 |
| Media asset, keyed on House Number | The only media identifier in the data is renamed at `loaders.py:90` and never read again | JS-8 |
| Campaign and flight, with real CRUD | Zero campaign write endpoints exist among 56 | JS-5, JS-6 |
| Make-good as an object | Today `project_make_goods` returns a shortfall fraction. There is no compensating inventory, no approval and no link back to the flight (`05-gaps.md:290`, `:322`) | JS-6 |
| Plan target | "On plan" has no referent. No goal, budget or target key exists anywhere in `/api/overview` | JS-1 |
| Plan version, with publish | The plan is the one thing the version store does not version | JS-2 |
| Restriction as an authored, expiring object | Constraint rows carry no author, no approver, no expiry | JS-4 |
| Model version and release record | The only model identity is a timestamp on a file overwritten in place | JS-16 |
| Candidate adoption with a measured verdict | Five candidate artifacts sit in `models/candidates/` and a script already computes the money each would move. Nothing adopts, rejects or records | JS-19 |
| One history timeline | Three "what happened" surfaces today, none covering the plan | JS-3, JS-10 |
| Command palette and keyboard control | None exists | **Bar 2, Linear**, not a story |

Four rows above are justified by a law or a reference rather than by a story,
and are labelled so a critic does not hunt for a story that does not exist: the
450-line law, the design tokens, `Cmd K` (`07-references.md:101`), `Cmd J`
(`07-references.md:413`), Sunday-first (Bar 4) and RTL (Bar 4).

### 3.6 The rule that removes every dead end

The brief's requirement is absolute and it is cheapest to state as one
invariant, checked by a critic on every surface:

> Every name, number, status and badge on any surface resolves to the thing it
> refers to, in at most one click, and the thing it resolves to is a real
> object with its own address.

Four consequences that are not obvious.

- **A figure without its scope does not render.** Section 3.4's money layer
  plus Stripe's discipline: the scope is attached to the figure, not to a
  tooltip, and where the scope cannot be stated the figure is withheld with a
  named alternative route (`07-references.md:174`).
- **An empty field is an action.** Linear renders `Set estimate` and
  `Add to project` where a value is unset (`07-references.md:143`). Meridian's
  honest empty states are already its strongest asset, measured in at least
  five places (`06-baseline.md:445`), and they currently end in prose. Each
  becomes a control.
- **Prose that names a capability links to it.** Six of the eight cross-surface
  references found in source are prose with no link (`01-surfaces.md:667`).
- **Opening a record keeps its place in the set it came from.** Linear's
  `1 / 31` with up and down arrows (`07-references.md:127`). Revision 1 sent
  people down into objects and never said how they walk the set they came from.
  Every drill in section 3.4 carries the counter and the two arrows.

---

## 4. Training versus runs, made concrete

This is a first-class deliverable, not a naming exercise. Today, of the 113
operations the live app publishes, zero are training, and yet the model's
internals render on four operator pages while one word, recompute, names both
activities 159 times in the UI and 124 in the backend, and a second word,
rebuild, names both 9 times and 85 times (`04-training-vs-runs.md:670-671`).

### 4.1 One classification rule, stated once

Revision 1 defined training as artifact-producing and then filed two
configuration acts under it anyway, because they happened to share an
affiliation gate. The rule below is stated once and applied everywhere, and it
never consults permissions.

> **The training test.** An act is TRAINING if and only if its output is a file
> under `models/`. Nothing else is training.
>
> **The permission rule is separate.** Affiliation decides which side of the
> line you can see. Role decides what you can change on your side. A
> company-only permission is not evidence that an act is training.

Applying the test to every act discovery classified, with the file root each
one writes:

| Act | Writes | Class | Home |
|---|---|---|---|
| `scripts/compute_measured_coefficients.py` | `models/tv_break_coefficients.json` | **TRAINING** | Model console only |
| `scripts/compute_audience_model.py` | `models/audience_model.json` | **TRAINING** | Model console only |
| Adopting a candidate artifact | `models/` | **TRAINING** | Model console only |
| `POST /api/recompute-schedule`, `POST /api/jobs/recompute` | `output/weekly_break_schedule.csv` | RUN | Plan |
| `POST /api/optimizer-plan`, `/api/scenario`, `/api/scenario-compare` | nothing, transient | RUN | Plan |
| `POST|PUT|DELETE /api/events` | `data/calendar_events.csv` | **CONFIGURATION** (`04-training-vs-runs.md:131`) | **Rules, the calendar**, company-gated by affiliation |
| `PUT /api/settings`, `PUT /api/pricing` | `data/kairos_settings.json` | CONFIGURATION | Rules and Plan |
| Throwing `pricing_activation.events` | `data/kairos_settings.json` | **CONFIGURATION**, company-gated | **Rules, the rate card** (this is what the product already does, `pricing_api.py:232-241`) |
| Throwing `audience_model_activation` | `data/kairos_settings.json` | **CONFIGURATION**, company-gated | **Rules**, beside the switch above |
| `POST /api/uploads/{kind}` | `data/` | CONFIGURATION | Sources |
| `POST /api/versions/{id}/restore` | the nine logical files | CONFIGURATION | History |

**Blur 1 is closed.** Event authoring writes `data/`, so it fails the training
test, so it is not on the training side. It stays on Rules, where a calendar
event belongs beside the other things that shape a price, and it keeps its
existing company-only permission through `require_company_editor`
(`events_api.py:378,399,426`, verified as three of the five call sites) plus
the `can_edit` that `GET /api/events` already returns.

**Blur 2 is closed.** History filters by artifact root. A `models/` entry is
visible only to `affiliation = company`. A channel account opening History from
any object header sees plan versions, configuration changes and Kai's actions,
and no model version has ever landed in their timeline. This is a filter on the
read, not a hidden section, so there is no rendered trace that the other side
exists (JS-18's requirement).

**Blur 1 appeared a second time, in a place the first fix did not reach, and it
is closed the same way.** The two rows added to the table above are activation
switches, and revision 2 sent one of them to the Model console because it is
company-only. That is the permission rule being read as evidence of training
again, which the box above forbids in bold. The second blind critique found it
and named it correctly as blur 1 restated. Apply the test literally. `audience_model_activation` is a field on `KairosSettings`
(`kairos_api/core.py:143`) and throwing it writes `data/kairos_settings.json`,
so it is configuration. It is also, by the code's own comment at
`core.py:139-143`, engine input: "flipping it is engine input, so it stays in
the freshness fingerprint and marks the saved schedule stale on a flip". An act
that writes `data/` and makes the plan stale is a run-side act by every part of
the rule at once.

**The artifact and the switch are different things and they live on different
sides.** `models/audience_model.json` is a training output: its coverage, its
gate verdicts and its fitness live in the Model console and nowhere else.
Whether runs consume it is a run-side configuration act with a money
consequence, and it lives on Rules. Section 4.5 states the control, the gate and
what each side sees. **The product already does exactly this for the other
switch**: the event pricing activation is thrown through `PUT /api/pricing` on a
run surface, writes `data/kairos_settings.json`, and is refused to a
channel-affiliated account with its own Hebrew denial at `pricing_api.py:232-241`
by way of `events_access.EVENT_PRICING_COMPANY_ONLY_DETAIL`. So this is a
precedent to follow, not a mechanism to invent.

**Every other switch, checked against the same rule.** Guardrails and the pacing
knobs write `data/`, so configuration, and section 4.5 moves them to their own
store on Rules, which is where they already were. The remaining pricing layers
(position, ad type, specific show) write `data/`, so configuration, and section 9
item 5 keeps them owner-gated and off. `first_break_multiplier` is not a switch
anybody throws: it is read out of the coefficients metadata on every
optimization (`kairos/service.py:102-125`), so it is a training output reaching a
run, which is exactly what the release note of section 4.6 exists to explain.
The five gate-override flags in `scripts/compute_measured_coefficients.py` write
`models/`, so training, Model console only. **Two switches are company-gated and
run-side, three are operator-side and run-side, five are training, and no switch
is filed by its permission.**

### 4.2 The check a critic can run on any surface

Three commands. A run surface passes all three, a training surface fails the
first two by design, and no surface is ambiguous.

1. **The write test.** List every write endpoint the surface calls. For each,
   name the file root it writes. If any writes `models/`, the surface is a
   training surface and must refuse `affiliation = channel` on read as well as
   write. If none does, it is not a training surface and no training content
   may render on it.

2. **The lexicon test.** Fetch every read endpoint the surface calls with a
   channel-affiliated session and grep each JSON response for the training
   lexicon: `gate`, `held_out`, `tau`, `drift`, `coefficient`, `pooling`,
   `p_value`, `training_window`, `wartime`. **A run surface returns zero hits.**
   Today `GET /api/events` returns a `model_context` object whose five keys are
   `training_window`, `weekday_premiums`, `measurement`, `wartime_disclosure`,
   `training_gate` (I fetched it and listed them), so the calendar read fails
   this test today. **Blur 4 is closed by the fix this test forces:** the read
   side of the calendar returns `model_context` only when the caller is
   company-affiliated, and Plan's event overlay never carries it at all.

3. **The verb test.** Read every button and message label on the surface. Each
   must contain exactly one canonical verb from section 4.5 in each language.
   Any label containing a retired word (recompute, rebuild, חישוב מחדש,
   בנייה מחדש) fails. Today this test fails 159 times in the UI.

### 4.3 Which surfaces belong to which side

| Side | Surfaces | Who sees them |
|---|---|---|
| Runs | Today, Plan, Clients, Rules, Sources, History (filtered), account settings, Kai | Any authenticated account, subject to role |
| Training | Model console: Gates, Coverage, Drift, Candidates, Releases | `affiliation = company` only, read and write |

### 4.4 What each side's dashboard answers

**The run dashboard answers six questions, in this order.** Every input already
exists in the payloads; most is not on screen, or sits next to something that
contradicts it.

1. **Is today's plan current, and if not, what made it old?** Today one amber
   banner fuses two different events: the live instance returns
   `{"status": "stale", "changed": ["settings", "coefficients"]}`, which means
   somebody changed a setting and somebody trained the model, and the operator
   sees one sentence, one verb and one button. It becomes two states. "Your
   changes are not in the plan yet" is self-service with one button. "A newer
   model version exists" is informational, names the version and when it landed,
   carries the plain-language release note from section 4.6, and offers the same
   button with a different sentence.
2. **What will change if I run it?** The math exists:
   `/api/constraints/effect` and `/api/overrides/effect` both run the commit
   path's own optimizer twice and diff the result. Nothing joins them to the run
   button.
3. **What did the last run produce, and how does it compare to the one
   before?** `output/run_log.jsonl` holds 489 records (measured) carrying run
   id, engine version, input checksums, guardrails, assumptions and a summary.
   No endpoint serves it and no screen shows it.
4. **What are these numbers the projection of?** Section 3.4's scope on every
   figure.
5. **Am I compliant, and where am I at risk?** The best-served question today.
6. **What is booked, what is unsold, what is a make-good exposure?**

The run dashboard contains **no gate verdicts, no held-out deltas, no tau
squared, no pooling notes, no drift tables and no per-cell coefficients**. It
needs exactly two model facts: which model version this plan was computed with,
as a date and a name, and the release note for it.

**The model dashboard answers six questions.** All six have their ground truth
already written into the artifacts on disk and none of it has a home.

1. **What did each gate decide and why.** The reasons are already stored in
   full sentences, for example "p=0.2034 not < 0.01; multiplier left at 1.0
   (off)" and "series RMSE (fold mean 0.26239) does not beat genre RMSE (fold
   mean 0.24200) by the required 2% margin". Three off-states must be visually
   distinct, because they are three different pieces of news: tested and lost,
   untestable for want of contrast, not yet measured.
2. **How much contrast does the data carry.** Per factor: cells, n per cell,
   contrast ratio. Today each absence is buried in prose inside a gate reason.
   The sharpest coverage fact the product owns belongs at the top: the whole
   30-day training window was wartime, with a post-ceasefire tail of 132 of
   2,532 measured breaks.
3. **What drifted.** `level_drift` is complete and well measured, and currently
   rendered to an operator who cannot act on it. It needs a series across model
   versions, which does not exist because each train overwrites the file.
4. **What would a train change, and did we take it.** `models/candidates/`
   holds five artifacts (measured: afterwindow, calibrated, competitor,
   placebo_corrected, spotclip) and `scripts/estimate_candidate_revenue_movement.py`
   (2,885 bytes) computes the revenue movement if one were adopted. This becomes
   the adoption surface of JS-19, not a display: per candidate, the gate deltas,
   the coefficient deltas, the money the adopted plan would move, and a recorded
   ship or no-ship verdict against a named model version.
5. **What is blocked on data we do not have.** A register, not a prose reason.
   Per blocked factor: the condition that would unblock it and roughly when.
6. **Provenance.** Fingerprints, seeds and method are already recorded.
   Missing: who ran it, with which gate-override flags. **Five** such flags
   exist with environment-variable twins, so a forced gate is today
   indistinguishable from a self-activated one after the fact. Revision 1 said
   six; I enumerated the `add_argument` calls in
   `scripts/compute_measured_coefficients.py` and they are `--series`,
   `--counterprogramming`, `--placebo-correction`, `--interval-calibration`,
   `--moderated-variances` and `--output`, of which `--output` is an output path
   and not a gate override.

### 4.5 The permission rule and the four open doors

One sentence, and it is different from today's.

> **Affiliation decides which side of the line you can see. Role decides what
> you can change on your side.**

Concretely, and each of these is a task with a named owner in section 8:

- Every `/api/model/*` and every training route requires `affiliation =
  company`, on **read as well as write**. Today the wall is three unconditional
  calls plus two conditional ones, which I verified are the only five call
  sites of `require_company_editor` in the codebase, and it covers writes only.
  The four open reads are `GET /api/impact`, `GET /api/model/audience`,
  `GET /api/parameters` and `/api/events`'s `model_context`.
- `audience_model_activation` (`kairos_api/core.py:143`) leaves the free-form
  settings document and becomes a named, company-only control **on Rules, not in
  the Model console**, for the reason section 4.1 gives. Today it decides where
  every forward-dated rating comes from, has no control anywhere in the dashboard
  (measured: zero occurrences of the field name under `tv-break-dashboard/src/`),
  and is settable by any channel operator with one `PUT /api/settings`, because
  that endpoint takes the whole settings model and has no affiliation guard.
  Specified in full, because a switch that moves money and has no surface is how
  this defect happened:
  - **Where it lives.** Rules, in the same activation group as the event pricing
    switch. That is the one place in the product where a company-only, run-side
    activation already lives, and reusing it means one guard rather than two.
  - **Who may throw it.** `affiliation = company` only, enforced by W0-4's
    decorator over the same path `pricing_api.py:232-241` already takes, with a
    Hebrew denial built exactly like the one on disk: "הפעלת מודל הקהל שמורה
    לצוות החברה", beside the existing "הפעלת תמחור אירועים שמורה לצוות החברה".
    Affiliation is the outer gate, so a channel administrator may not throw it;
    role is still the inner one, so a company-affiliated viewer sees the state
    and not the switch. That is the sentence at the head of this section applied
    without an exception.
  - **What a channel account sees.** The switch, rendered as state and not as a
    control, carrying `can_edit: false` with its reason, plus the three-state
    basis label the engine already computes: `off` means forward-dated ratings
    are the historical baseline, `on` names the model version's date, and
    `on_no_artifact` means the switch is on and nothing is trained so the numbers
    are still historical (`kairos_api/audience_api.py:91-113`, which returns
    exactly `{state, computed_at}` and nothing else). No gate verdict, no
    coverage, no coefficient, no p-value. That payload passes section 4.2's
    lexicon test as it stands today, and keeping it that way is a stated duty of
    the piece that owns it.
  - **What a company account sees.** The same switch, throwable, with its
    consequence stated before the click in the same plain language section 4.6
    demands of a release note: flipping it changes the freshness fingerprint, so
    the saved plan goes stale and the operator will be asked to run it, and
    forward-dated ratings change source. In the Model console the
    company account sees the artifact itself and a read-only mirror of the switch
    state that links back to Rules. **The console never carries the control**,
    because throwing it changes a run, and a run-side act on the training side is
    the blur this section exists to prevent.
- The regulatory guardrails leave the same document and become their own store
  with an effective date and a change record, so JS-14's second half becomes
  possible.
- Every surface reads its own session affiliation and renders accordingly.
  Today the dashboard never reads it: `GET /api/auth/me` returns `affiliation`
  and the only frontend uses of the word are the accounts dialog and a label
  helper.
- **A refusal is legible before the click, never a 403 after it.** I fetched
  both: `GET /api/events` returns `can_edit: true`; `GET /api/pricing`'s top
  keys are `currency, units, base, layers, activation, events, has_overrides,
  note` with no `can_edit`, so its identically walled toggle renders enabled and
  fails after the click. Every walled control carries a `can_edit` from its own
  endpoint.

### 4.6 The release note: what closes the silent-money leak

`04-training-vs-runs.md:346` measured that `kairos/service.py:102-125` reads
`first_break_multiplier` from the coefficients metadata on every optimization
and takes `max(assumption, measured)`, currently 1.0 and off at p=0.2034. A
future training that clears p < 0.01 raises the retention cost of every first
break in every plan. Revision 1 gave the operator "a newer model version
exists" and, by rule, no gate verdicts. So revenue would move, section 3.4
would demand a scope, and the training-side rule would forbid the only
explanation.

**A model version does not ship without a release note.** It is authored on the
training side as part of the ship decision (JS-16's done condition), it is
plain language, it names what changed for the operator and in which direction,
and it carries no gate verdict, no p-value and no coefficient. One sentence
plus an optional money direction. It is the only training-authored text that
crosses the line, and it crosses because the alternative is money moving with
no legible cause.

Example, for the case above: "Retention cost is now higher for the first break
in a programme. Plans run after 2026-08-04 will place fewer first-position
breaks." No p-value, no multiplier.

### 4.7 The context switcher

A control that moves a company user between the operator side and the training
side is by definition the one control that can be mis-clicked across the line.
Revision 1 asserted it exists and never specified it. Specified here:

- It lives in the **account menu**, the same menu that already holds Manage
  accounts (`TVBreakDashboard.jsx:2288`), as the last item, separated.
- It reads **"Model console (company)"** / **"קונסולת המודל (חברה)"**. It names
  the destination, never the act, so it cannot be read as a verb.
- It renders **only** when `GET /api/auth/me` reports `affiliation = company`.
  For a channel account the menu item does not exist, there is no disabled
  state, no tooltip and no route: `#Model` returns the operator's Today with no
  message, so nothing tells a channel account the other side exists.
- The Model console is a different shell with its own chrome and a permanent
  marker in the header reading "Company side. Training." Returning is the same
  control, reading "Back to the channel".

### 4.8 The canonical vocabulary

One word per activity, in each language, used everywhere and never for anything
else. Both chosen words are already in the product with exactly this meaning and
no collision, so this narrows rather than invents.

| Class | English | Hebrew | Verb | Output | Never say |
|---|---|---|---|---|---|
| Training | training | אימון | train / לאמן | model version / גרסת מודל | rebuild, recompute, בנייה מחדש, חישוב מחדש |
| Run | run | הרצה | run / להריץ | plan version / גרסת תוכנית | rebuild, recompute, בנייה מחדש, חישוב מחדש |
| Configuration | change | שינוי | save / לשמור | the saved store | apply (reserved for proposals) |
| Publish | publish | הפצה | publish / להפיץ | a published plan version | approve, פרסום |

Why these. **אימון** appears eleven times in the UI and always means model
training, with zero run usages. **הרצה** appears once, in the Run Optimization
button, and never for training. **חישוב מחדש** and **rebuild** are retired from
both activities because they are the collision, and retiring is safer than
reassigning. **הפצה** is chosen for publish because **פרסום** collides with
advertising in an advertising product, and **אישור** collides with the existing
Approve on recommendation cards.

**The two output nouns.** `04-training-vs-runs.md:749` warns that the split
"has to differ in kind, not in degree", and גרסת מודל versus גרסת תוכנית differ
only by the qualifier. They now never appear side by side, because History
filters model versions out of a channel account's timeline entirely (section
4.1) and the Model console is a different shell. For a company account they do
appear together, and there the qualifier is doing exactly the work a qualifier
should: two versioned artifacts, named by what they version.

The renames this forces:

| Today | Becomes |
|---|---|
| "Recompute now" / "הריצו חישוב מחדש" | "Run the plan" / "הריצו את התוכנית" |
| "Recompute weekly schedule" | "Run the weekly plan" / "הרצת הלוח השבועי" |
| "Apply to weekly schedule" | "Save and run" / "שמרו והריצו" |
| "Run Optimization" (which saves nothing) | "Preview" / "תצוגה מקדימה" |
| "recompute the coefficients when new data lands" | "the model needs training when new data lands" / "המודל דורש אימון כשנקלטים נתונים חדשים", with a named owner and no button |
| "Model measurements current, Measured Jul 29" | "Model version 2026-07-29, current" / "גרסת מודל 2026-07-29, עדכנית" |
| staleness label `coefficients` | "a newer model version exists" / "קיימת גרסת מודל חדשה יותר" |

The HTTP paths do not change. `POST /api/recompute-schedule` and
`POST /api/jobs/recompute` keep their addresses; every label and message around
them changes.

**The clocks.** Five timestamps are visible in one session today and the one
labelled "Updated" is a file modification time. Two survive, each named by
activity: **plan version, run at T** and **model version, trained at T**.
Everything else disappears from operator surfaces.

### 4.9 The new object vocabulary, Hebrew and English

Adopted alongside the canonical terms already frozen in the brief (ברייק,
ברייקים, נעיצה, ברייקי זהב, רצועת שידור, הכנסה צפויה, עלות שימור, מפעיל, and
never משתמש). The product is Hebrew-first under Bar 4, so the five workspaces
and the four non-workspace destinations are named in both languages here rather
than in English only.

| Object or place | English | Hebrew | Source of the word |
|---|---|---|---|
| Workspace 1 | Today | היום | new |
| Workspace 2 | Plan | תוכנית | existing |
| Workspace 3 | Clients | לקוחות | the trade's word |
| Workspace 4 | Rules | כללים | new |
| Workspace 5 | Sources | מקורות | new |
| The dock | Kai | קאי | existing |
| The timeline | History | היסטוריה | new |
| The account menu page | Account settings | הגדרות חשבון | existing |
| The company shell | Model console | קונסולת המודל | new |
| The week's plan | weekly plan | תוכנית שבועית | existing |
| A published plan | plan version | גרסת תוכנית | new, mirrors model version |
| A broadcast day | broadcast day | יום שידור | existing |
| A break | break | ברייק | frozen |
| A break's contents | break contents | תוכן הברייק | new. There is deliberately no second word for "pod": the break is the pod |
| One ad occurrence | spot | תשדיר | the trade's own word, already in the daily file (`סוג תשדיר`, `אורך תשדיר`) |
| The media file | House Number | House Number | the trade's own identifier, already an English column in a Hebrew file |
| A programming restriction | restriction | הגבלה | the representative's word, not the engine's |
| A plan target | target | יעד | new |
| A make-good | make-good | פיצוי שידור | new |
| Tonight's break queue | tonight's breaks | הברייקים של הערב | the traffic operator's word (`03-people.md:388`) |

---

## 5. The entity model, and the migration to it

### 5.1 What is wrong today, measured

| Entity | Today | Consequence |
|---|---|---|
| Programme segment | `segment_id = f"{day}\|{channel}\|{index:03d}"`, an `enumerate` position (`kairos/data/transform.py:255`) | Insert one programme in the EPG and every later id on that channel-day shifts |
| Break | Not an entity. A `num_breaks` integer on a segment | "Move the 20:05 break" is inexpressible; the plan can only say a segment now has three breaks instead of two |
| Spot | Not an entity. A free-text version name on a daily-file row | No ad is addressable |
| Media asset | Does not exist. `house_number` renamed at `loaders.py:90`, never read again | Nothing can be verified |
| Advertiser | 45 ids with **zero** intersection with either name space; the two name spaces match each other 41 of 41 (I re-measured all four figures with pandas) | 0 of 45 advertisers have a name or a revenue figure |
| Campaign | Three unjoined things: 478 free-text strings, a pacing key with 0 rows, and a per-spot name | No campaign is an object |
| Make-good | Not an entity. `project_make_goods` returns a projected shortfall fraction | A shortfall cannot be answered, only reported |
| Plan | A CSV plus a sidecar. Not among the nine versioned logical files | No plan history, no undo, no publish |
| Model | A file overwritten in place | No model version identity, no drift series |
| Target | Does not exist | JS-1 is unanswerable |

### 5.2 The model the work implies

```
  Airing            key: hash(date, channel, start_clock, title)
    |                    the anchor triple manual_overrides.csv already carries
    |
    +-- Break         key: airing_id + ordinal
    |     |           state: draft -> assembled -> verified -> locked -> exported
    |     |           carries: planned start, length, position, gold,
    |     |                    projected money, delivered money, retention cost
    |     |
    |     +-- Spot    key: break_id + position
    |           |     joins the plan spine to the commercial spine
    |           |
    |           +-- MediaAsset   key: House Number
    |                            technical facts, owner-supplied
    |
    +-- Restriction   authored, scoped, expiring; compiles to the frozen predicate

  Agency -> Advertiser -> Campaign -> Flight -> (books) Spot
                                        |
                                        +-- MakeGood  shortfall, compensating
                                                      inventory, approval, back-link

  PlanVersion    a run's output, named, dated, publishable, diffable, restorable
  ModelVersion   a training's output, named, dated, with its gates, its release
                 note and its ship or no-ship verdict
  Target         the number a week is measured against
```

The terminal break state is **`exported`, not `aired`**. Revision 1 wrote
`aired`, and nothing in the product can observe that a break aired: a delivery
or as-run feed is owner decision 4 and does not exist. `05-gaps.md:241`
proposed the achievable vocabulary and this adopts it verbatim.

Two design rules carried from what already works.

**Identity is semantic, never ordinal.** The anchor triple is already proven:
`kairos/optimize/overrides.py:175-200` re-binds by `anchor_date, anchor_start,
anchor_title` and reports a stale override rather than silently rebinding. That
discipline is stored today on exactly one of the five rule stores. It becomes
the rule for every operator-facing identity. The ordinal `segment_id` survives
as an internal engine handle that is never stored on an operator record and
never rendered.

**Stores follow the doctrine that already exists.** Module lock, temp file,
`os.replace`, a pre-write backup, and a version snapshot per mutation, as
documented at `kairos_api/agencies.py:1-12`. Every new store registers in the
version timeline.

### 5.3 The identity problems that must be solved first

Nothing else is worth building until these three land, in this order. This is
the best-evidenced ordering in the document and it is unchanged from revision 1.

**1. Advertiser identity.** See section 5.5: the method changed, the priority
did not. This one change unblocks JS-5, JS-6, JS-9 and every delivered-money
figure. Evidence that it works: the same 175 rows resolve the agency to an id
and produce real money (gross ₪699,450, net ₪669,978, 119 spots) while failing
to resolve the advertiser at all.

**2. Airing identity.** Move every operator-facing reference off the ordinal and
onto the anchor hash. Without this, a break entity is built on sand: one EPG
insertion reassigns every pod on the channel-day.

**3. The break entity.** A stable break identity below the airing, with an
ordered list of spots. It is the money layer of section 3.4, so every honest
per-break figure and everything the traffic operator needs rests on it.

Then, in any order: campaign and flight CRUD, the make-good, the media asset,
the plan version, the model version, the target.

### 5.4 `data/Spots.csv`: measured, and what it is and is not good for

Revision 1 did not mention this file once while building both of the things it
appears to carry. I measured it rather than inheriting either the omission or
the critique's reading of it.

**What is in it, measured with pandas:**

| Column | Populated | Distinct | Note |
|---|---|---|---|
| rows x cols | 50,386 x 36 | | 30 dates, 2024-11-01 to 2024-11-30, four channels |
| `break_id` | 50,386 of 50,386 (100%) | **9,492** | integer, unique per (channel, date): 9,492 groups, 9,492 distinct |
| `position_in_break` | 100% | 48 | 1..48 |
| `revenue_ils` | 100% | 7,004 | sums to ₪306,936,788 |
| `advertiser_id` | 100% | 35 | derived garbage: values include `2024` and `פ` (`02-api-and-data.md:571`) |
| `competitor_flag` | 100% | 2 | **True on 26,679 rows, False on 23,707.** It does not partition two channels: the False side is `קשת 12` alone and the True side is `רשת 13`, `כאן 11` and `עכשיו 14` together. Revision 2 read this row as a two-channel split and it is not one |
| `is_target_channel` | 100% | 2 | The exact complement: True on all 23,707 `קשת 12` rows, False on all 18,669 `רשת 13` rows. `include_as_media` is True only for `קשת 12` |

**Who reads it:** nothing reads those columns. The file appears in nine source
locations and every one is either a cache-key signature
(`dashboard_api.py:644,1471,1651`, `catalog_api.py:464,586,601`) or the
uploaded-CSV fallback used only when `data/reference/Spots.xlsx` is absent
(`kairos/data/loaders.py:42`), and that fallback "reads only the shared columns"
by its own docstring at `loaders.py:38`. A repository-wide grep for `break_id`
finds it only in `kairos/optimize/frequency.py` and `kairos/export/spots.py`,
where it is a clock string on the daily path, never this integer column.

**The finding revision 1 and the critique both missed: `revenue_ils` is not
money.** I tested the formula. `revenue_ils == base_rate * TVR * Duration *
total_premium` holds for 99.67 percent of the 50,386 rows, `base_rate` is the
single constant **50** on every row, and `adv_premium` is the constant **1** on
every row. So ₪306.9M is a synthetic price computed from an asserted flat CPP,
not observed revenue, and under the honest-math law it may never render as
money on any surface. The critique's "306.9 million shekels" is arithmetically
correct and commercially meaningless.

**The decision.** Adopt it as a **test corpus and a shape validator, never as a
source of money or of advertisers.** Specifically:

- It is the only artifact carrying a historical break identity with ordered
  contents at scale: **3,055 breaks on the operator's own channel**, median 2
  spots, p95 24, max 47, of which **875 hold seven or more spots**. JS-7's
  target is a seven-ad break and today it is testable against exactly one group
  in one daily file. This gives 875 on the right channel.
- It settles nothing about the pod boundary. **The four gap counts revision 2
  printed here did not reproduce, under eight variants between two people, so
  they are replaced with numbers and a method.** The method: read the file with
  pandas, keep `Channel == רשת 13` (18,669 rows, 3,055 `break_id` values, 30
  dates), parse `Date` plus `Start time` as `%d/%m/%Y %H:%M:%S`, set each spot's
  end to its start plus `Duration` seconds, sort within a date by start then
  `position_in_break`, and take consecutive pairs. A pair is within-break when
  the two rows share a `break_id` and a boundary when they do not.
  - **1,145 of the 18,669 rows carry an Excel `01/01/1900 HH:MM` artifact in
    `Start time`** and do not parse. Each is the only row of its own break, so
    dropping them removes 1,145 whole breaks and no within-break pair.
  - **Within a break: 15,614 consecutive pairs, of which 0 exceed 60 s.** That
    count is method-independent, because it is rows minus breaks
    (18,669 - 3,055), and recovering the 1,145 artifact rows onto their own date
    gives the same 15,614 with the same zero. **Within-break contiguity is real,
    and it is stronger than the "2 of 15,214" this section used to claim.**
  - **Between breaks the count is method-dependent, and that is the honest
    answer rather than a number.** On the 17,524 second-resolution rows: 1,880
    boundaries, 625 at 60 s or less, 33.24 percent. Recovering the 1,145
    minute-resolution rows: 3,025 boundaries, 1,667 at 60 s or less, 55.11
    percent, but those rows resolve only to the minute so a 60 s test on their
    boundaries is not decidable. **Either way, between a third and a half of
    adjacent break pairs sit 60 s or less apart, so a gap rule does not reproduce
    `break_id` and the rule that produced it is not on disk.** The other three
    channels behave the same way and none of them yields the 2,412 boundaries
    previously claimed: `קשת 12` 2,185 boundaries at 23.34 percent, `כאן 11` 892
    at 4.15 percent, `עכשיו 14` 1,153 at 50.48 percent.
  - One structural fact worth keeping, measured while doing this:
    **`break_id` never interleaves in time.** Sorting each date by start and
    counting runs of equal `break_id` gives exactly 3,055 runs for 3,055 ids,
    zero extra. So in this file every break is one unbroken run of rows and its
    spots really are back to back, which is the opposite of what the daily file
    shows below. The two facts are about two different files and they do not
    conflict: the gap rule fails here because breaks sit too close together, and
    it fails there because one declared break has a hole in it.
- **On the daily file a 60 s gap rule does not reproduce the declared grouping,
  and the claim that it does was wrong in both this document and the owner's.**
  It yields 10 groups against the declared 10, which is what the earlier claim
  rested on, but not the same partition. Sizes under the rule are 1, 3, 4, 7, 7,
  22, 28, 30, 35, 38; declared sizes are 1, 1, 3, 3, 7, 28, 29, 30, 35, 38. Two
  counterexample rows: the declared break `21:22:12` holds one 32 s spot starting
  21:22:16 and the next declared break `21:23:10` begins 22 s after it ends, so
  the rule **merges** two declared breaks; the declared break `22:59:40` holds 29
  spots with a 93 s internal gap before the spot at 23:12:58 and spans 642 s
  against 432 s of ad time, so the rule **splits** one declared break and that
  break is not contiguous. Thresholds of 30, 45, 90, 120, 180 and 300 s do not
  reproduce the declared partition either. **The same sentence sits in
  `decisions-for-owner.md:63` and must be corrected there before that document
  goes to the owner; the same paragraph's "13.4 minutes" for the 38-spot group is
  13.93 minutes measured (22:04:16 to 22:18:12, 836 s), which strengthens rather
  than weakens its point.** The owner's decision does not change; it becomes more
  obviously necessary. **The pod boundary stays owner decision 2.**
- Its `advertiser_id` is derived garbage and is never used for identity.
- **Its channel flags are inverted against this operator, and that is a data
  defect the rebuild must not inherit.** In this file `is_target_channel` marks
  `קשת 12` and `competitor_flag` marks the operator's own `רשת 13`, while
  `data/kairos_settings.json` sets `operator_channel: רשת 13`. So the file calls
  a rival the target and the operator a competitor. A builder who applies the
  channel scope of section 4.5 by reading either column selects `קשת 12` and
  breaches Bar 4's competitor law with the product's own data. **The rule: the
  channel scope always comes from `operator_channel` in settings through
  `channel_scope.py` (section 8.3), never from a column in this file, and
  `is_target_channel`, `competitor_flag` and `include_as_media` are read by
  nothing in the rebuild.** They are an artifact of whoever generated the file
  for a different operator. The file is not corrected in place, because it is a
  frozen test corpus and editing it would destroy the only historical break
  corpus the product has. Nothing reads those columns today, as measured above,
  so this is a rule that keeps a true thing true rather than a fix, and it is
  already inside C1's remit: the boundary sweep looks at every competitor name on
  every surface.

Recorded as a decision rather than an omission, which is what the critique
asked for.

### 5.5 Advertiser identity: the bar revision 1 could not meet

Revision 1's bar was "45 of 45 advertisers named". It is impossible from the
data and I confirmed every figure myself.

- `data/advertiser_rules.csv` is 45 rows x 8 columns. The columns are
  `advertiser_id, default_premium, allow_positions, allow_genres,
  prime_time_only, urgency_k, ahead_k, notes`. **There is no name column and no
  alias column, and `notes` is empty in 45 of 45 rows.** Ids are
  `ADV_01..ADV_45`.
- The real advertiser vocabulary is **41 Hebrew names**, in
  `data/agency_advertisers.csv` (41 rows, `source = observed` on all 41) and in
  the daily file's `מפרסם` column (41 distinct). The two name spaces match
  **41 of 41**.
- `advertiser_rules ∩ agency_advertisers = 0`. `advertiser_rules ∩ daily = 0`.

Forty-five cannot map onto forty-one. Meeting the old bar would require
inventing at least four advertisers, which the honest-math law forbids.

The shape of the fix is known, because agencies already work: `agencies.csv`
carries `name` and `aliases` (I confirmed both columns exist), and **9 of 9**
daily agency names are present in `agencies.csv.name`, so resolution is a string
equality that already holds on disk.

**The new bar, which the data can satisfy:**

> **Every advertiser that appears in the daily file resolves to a named record:
> 41 of 41. Zero invented advertisers. Every premium that moves a price is
> traceable to a named advertiser. Engine figures byte-identical.**

Which of the two honest methods to use is **owner decision 1**, because one of
them discards 45 rows of premiums. My recommendation, the evidence for it, and
what is blocked until he answers are in
`docs/ux-gauntlet/decisions-for-owner.md`. Until he answers, W0-3 ships the
`name` and `aliases` columns and the resolver, and the 41 observed names bind;
the disposition of the 45 synthetic rows waits.

### 5.6 Migration, and what is not destroyed

- The five rule grammars collapse to two: a **placement rule** (today's
  constraints plus overrides, which already share a preview endpoint returning
  the same shape) and a **commercial rule** (today's advertiser conditions plus
  agency conditions plus frequency rules, which already share a scope grammar).
  Four of the five stores are empty today and the fifth has one row, so this is
  a consolidation of schemas, not of data.
- **`frequency_rules.csv` is that one row, and it is load-bearing.** Its single
  rule `DEFAULT_ONE_PER_BREAK` (`max_per_break`, scope `default`, value 1) is
  the only rule that fires anywhere in the product, and it drops **56 of the 175
  daily spots** (`status: dropped_frequency` in the exported ledger). It becomes
  a visible, editable commercial rule on Rules with its 56 dropped spots as a
  drill target, because a rule that removes a third of the day's inventory and
  is invisible is the worst kind of dead end.
- `data/kairos_constraints.csv` does not exist on disk while **seven non-test
  modules** reference it: `kairos_api/constraints.py`, `kairos/service.py`,
  `kairos/optimize/constraints_store.py`, `kairos/optimize/_constraints_io.py`,
  `kairos/export/schedule.py`, `kairos/export/schedule_freshness.py` and
  `scripts/export_schedule.py`. Revision 2 said four; I counted with a repo-wide
  grep excluding `tests/`. The new store is created explicitly rather than
  implicitly, and five of those seven modules are frozen, so P5 creates the store
  and changes no reader.
- The version store's entries that point outside the repository are marked
  unrestorable rather than deleted, and the store gains an isolation guard so
  tests cannot write into the operator's history again. **The count, with its
  method, because revision 2's 187 reproduces under neither:** 200 manifests,
  of which **186 carry a file path containing `pytest`** and **2 more carry a
  different temporary path** under `/var/folders/.../T/tmp*`, so **188 of 200
  point outside the repository and 186 of those name a pytest temp directory**.
  Method: parse every `data/versions/*/manifest.json` and test each entry of its
  `files` array's `path`.
- `data/enriched/` (19.7 MB, read by nothing) and `data/Programmes - today.csv`
  (125 rows, zero readers repo-wide) are proven dead before removal, and the
  proof is written down. **`kairos/optimize/agreements.py` is removed from that
  list: it is not dead.** It is imported and re-exported at
  `kairos/optimize/__init__.py:26` and `:44`, and `tests/test_agreements.py`
  exercises `load_agreements`. Revision 2 called it "zero callers" and authorized
  its removal; removing it breaks an import and fails the suite, which Bar 4
  forbids. It stays.
- `AssistantUpload.jsx` and its whole second upload system
  (`kairos_api/assistant_uploads.py`, 245 lines, three routes, its own per-user
  store) is **kept and re-scoped, not deleted**. It is DUPLICATE-OF
  `UploadCenter` for the seven contract-validated kinds, and it is the only path
  for an arbitrary spreadsheet a person wants Kai to read. So the seven kinds
  route to Sources from both places and the dock keeps only the ad-hoc
  read-and-summarise path, labelled as what it is. Deleting it would remove a
  capability, which Bar 3 and the brief's boundaries both forbid.
- **`GET /api/settings/controls` is adopted, not orphaned.** It serves 4,253 B
  of bilingual lever schema with help text and bounds, built precisely so the
  panel would not hardcode, and `TVBreakDashboard.jsx:5454` carries a comment
  claiming the panel stays in sync with it while there is no fetch. Rules and
  Plan drive their lever labels, help and bounds from it. That is one fewer
  hardcoded bilingual string table and it is already written.
- **The upload consequence, blur 3 of the first critique, which revision 2
  claimed was closed here and was not.** Section 11 said "an operator upload
  invalidates the model and the remedy is an act they cannot perform" was closed
  at 5.6 and P6, and nothing in this document said anything about it. It is
  closed now, and the measurement changes the answer. **Today an operator upload
  cannot make the model stale, and the product can already prove it.** `models/tv_break_coefficients.json` carries
  `source_fingerprints` for exactly `data/reference/Spots.xlsx`,
  `Programmes.xlsx` and `Dayparts.xlsx`, while the three matching upload kinds
  write `data/Spots.csv`, `data/Programmes.csv` and `data/Dayparts.csv`. All
  three read `in_use: false` on the live instance with the reason already
  written: the engine resolves the reference workbook first and adopts the upload
  only when that workbook is absent (`kairos/data/loaders.py:30-60`). So the
  upload touches no fingerprinted file and does not reach the engine.
  - **The condition under which it does.** `_source_fingerprints()` in
    `scripts/compute_measured_coefficients.py:178-193` resolves through the same
    `_resolve_reference_path`, so the fingerprints follow the file that actually
    fed the model. Remove a reference workbook and the uploaded CSV becomes both
    the live engine input and the fingerprinted one; the next upload of that kind
    then rewrites a fingerprinted file and the model really is stale. That is one
    condition, it is computable, and the product already computes it.
  - **What Sources shows, and it is three states, not two.**
    `kairos/model/freshness.py:45` returns `fresh`, `stale` with the changed
    paths, or `unknown` when a fingerprinted file is missing on disk, and it
    never invents a fresh. P6 puts that verdict on the source card and on the
    upload confirmation, using the payload it already owns: `GET
    /api/uploads/status` ships `in_use`, `in_use_reason` and `engine_reads` per
    kind today, measured. Before the upload the card states which of the two
    consequences this upload has, in the operator's terms: **stored but not read
    by anything**, or **this is the live input, so uploading changes what the
    model was measured on**.
  - **The remedy names its owner and carries no button.** Training is a company
    act by section 4.1, so the operator's control is a request with a named
    owner, in the words section 4.8 already fixes: "the model needs training when
    new data lands", never a verb the operator cannot perform. The message names
    the changed source files and the model version's date, and carries no gate
    verdict, no coefficient and no p-value, so it crosses nothing.
  - **Owner: P6**, on files P6 owns. `kairos/model/freshness.py` is frozen under
    `kairos/model/**` and is read, never written. JS-12's Bar 3 row already
    requires that every `in_use: false` keeps its honest reason, so this extends
    a floor rather than inventing a surface.
- **The source-file list is corrected.** `GET /api/files` lists
  `models/tv_break_posterior.pkl` (1.2 MB, dated 2026-07-01) which
  `kairos/model/impact.py:289-304` never reads because the measured JSON
  resolves first, and omits `models/tv_break_coefficients.json` (21 KB) which
  drives every retention number. Sources lists the artifact that is read, marks
  the posterior as a fallback that is not currently in use, and says why.
- Every engine number stays byte-identical. Identity work does not touch
  optimization, pricing or retention math. Where a change would move a figure,
  it stops and escalates.

---

## 6. The net-new capabilities

Each with what it needs from the codebase and what the owner must supply.
Anything owner-blocked ships with an honest empty state naming the missing input
and the path forward, never with a placeholder figure.

| Capability | Needs from the codebase | Needs from the owner |
|---|---|---|
| **Plan target** (JS-1) | A small target store, a comparison in Today, a three-state verdict with a published threshold | **Decision 3.** The weekly or monthly revenue or GRP target per channel, and who sets it |
| **Publish** (JS-2) | Plan versions on top of the existing version store, a published state, an author, a diff against the previous version | **Decision 5.** What publishing means here and who may do it |
| **Break entity and break board** (JS-7) | New store, new router, an ordered spot list, a state machine, an engine seam at the daily per-spot path where ads and breaks already meet, plus `kairos/optimize/frequency.py`'s existing `break_id` plus `position` model | **Decision 2.** The pod boundary rule. Measured in section 5.4: neither the daily file nor `Spots.csv` derives it |
| **Media verification** (JS-8) | Media asset store keyed on House Number, a technical probe, a verdict printed on the ad | Media files or a technical metadata feed per House Number, plus the approval state vocabulary. `סטטוס` is empty in 175 of 175 rows so its values are unknown. No media tooling exists in the repository |
| **Campaign and flight CRUD** (JS-5) | A router modelled exactly on `agencies.py`, with deactivate rather than delete | Real flights with dates and goals |
| **Live pacing** (JS-6) | The pacing math is already implemented and honest; it needs a delivered figure that updates and a forecast state | **Decision 4.** A delivery or as-run feed, and a current week. Nothing in the data represents now |
| **Make-good as an object** (JS-6) | A store keyed on the flight, holding the measured shortfall, the proposed compensating inventory drawn from the same break board, an approval state and a back-link to the flight. `project_make_goods` already computes the shortfall | The commercial rules for what a make-good may be offered against and who signs it off |
| **Restriction language and live preview** (JS-4) | A translation layer above the frozen predicate contract, an offset measured from programme end, an occurrence concept, and `/api/constraints/effect` wired with a real latency budget | Which airings are finales. Zero of 418 titles carry any finale marker |
| **Model console** (JS-16) | Company-gated routes wrapping the two existing scripts, a model version identity, a release note, a ship verdict, a drift series across versions | Nothing |
| **Candidate adoption** (JS-19) | A held-out re-measurement per candidate, the revenue movement `scripts/estimate_candidate_revenue_movement.py` already computes, and a recorded verdict against a named model version | Nothing. Approval to move a figure if a candidate is adopted, which the piece escalates rather than assumes |
| **Delivered money drill** (JS-9, JS-13) | Per-spot attribution joined to a named advertiser, and every amount resolving to its breaks and then its spots | Nothing beyond decision 1 |
| **Action-level undo** (JS-3) | Per-action inverses on the day board, on top of the version store | Nothing |
| **One history timeline** (JS-3, JS-10) | Merge three surfaces, add the run log, filter by artifact root per section 4.1 | Nothing |
| **Command palette and keyboard** (all) | New | Nothing |
| **Speed** (all) | See section 8, W0-5. The engine change is named there and it is not caching | Nothing |

Speed is a capability, not a detail. Measured by me on the running instance
during this revision, under the load I was creating: `/api/versions` 53.43 s,
`/api/advertisers` 26.58 s, `/api/constraints/effect` 19.77 s,
`/api/overrides/effect` 15.72 s, `/api/uploads/status` 11.45 s,
`/api/advertisers/stats` 10.88 s, `/api/yield-per-second` 7.42 s. Eight parallel
requests, which is what one page load issues, take 16.98 s wall against a single
uvicorn worker (`06-baseline.md:436`). The target is: **no read on a first paint
slower than 500 ms at p95, no page's first meaningful answer later than 1.5 s,
and no interactive action without either a result or a cancel inside 5 s.**

---

## 7. Kai in the new architecture

Kai's current design is better than its delivery. The structure is right: 31
read tools, 8 propose tools, 0 write tools, plus a system prompt whose fourth
rule states it never changes anything itself. The proof it holds: an action
request in Hebrew ("raise the retention floor to 75 percent") changed nothing,
and the setting was still 0.72 afterwards. What fails is that the answer took
78 s at the backend and never arrived in the browser at all, still "preparing"
at 499 s.

### 7.1 What Kai can do, and from where

- **From every surface**, docked right, opened with `Cmd J`. It already knows
  which page and which record are open, and that context work is real.
- **Read anything the caller may read**, and nothing else. Scope is bounded by
  the caller's identity: their role, their affiliation, and their channel.
- **Propose changes on the object in front of the person.** The proposal renders
  as a concrete diff on the object itself, not only as a card in the dock. That
  is the device to take from Linear's agent, whose panel shows a change card
  with a `Preview` control and a run trace before anything lands.
- **Hand back an addressable restore point** that can be opened and inspected
  before it is used. Cursor's checkpoint model is the reference: the reversal is
  an object you can point at, not a single step.
- **Appear in the same history timeline as people**, attributed, with the model
  named on the surface. The audit schema already stamps `via: assistant`.

New, because discovery found them missing:

- **Grounding for the retention model.** There are eleven keyword-triggered
  grounding sections (`_SECTIONS` at `kairos_api/assistant_keywords.py:354`,
  counted; revision 2 said twelve) and none is triggered by מקדמים, coefficients, אימון,
  training or drift, so an operator asking in Hebrew why the plan moved gets no
  coefficient context, even though a read tool exists that would answer it.
  What it returns to a channel account is the release note of section 4.6 and
  the model version, never a gate verdict.
- **The new object vocabulary.** Kai speaks week, day, break, spot, restriction,
  target, make-good, plan version and model version, the same words the
  interface uses.
- **A voice per job.** The system prompt addresses "the operator" exclusively.
  It should address the person in front of it, using the `job` field of section
  2.1.

### 7.2 What Kai must never do

1. **Never start a training run**, propose one, or offer one as an option. Zero
   of its 39 tools touch a model artifact today and that must stay true. This is
   the training test of section 4.1 applied to the tool surface: no tool may
   write under `models/`.
2. **Never disclose model internals to a channel account.** Gate verdicts,
   held-out deltas, drift, coverage and coefficients follow the same affiliation
   wall as the console. Read tools are open to every authenticated account
   today; three of them return training content.
3. **Never name a competitor channel or carry its data into context.**
4. **Never write without an approval.** Propose, show the diff, wait.
5. **Never answer a money question without its scope.** It already does this
   well, unprompted, and that behaviour is now a requirement rather than a
   habit.
6. **Never fabricate.** Tri-state honesty: real, unavailable, unknown.
7. **Never reach a control the person themselves cannot reach.** If a refusal
   would be legible to them before the click, it is legible to Kai before the
   proposal.

### 7.3 Kai's own bar

JS-10: 45 s from question to applied change, first token within 2 s, a
previewable restore point. Today: 78 s to a backend answer, never in the
browser, no undo control in the product.

---

## 8. The build order

Revision 1 allocated ownership by **surface** while the code is shaped by
**layer**, so ten builders launched against it would have collided on day one.
Three backend modules each served four to six pieces, the file the spec named as
the problem was assigned to nobody, and five cross-cutting rules had no builder.

This section is re-cut so that **every piece owns a vertical slice of files**.

### 8.0 The three ownership rules

1. **One file, one owner, for the whole run.** Not per wave. Every path in the
   table below has exactly one **final** owner. A path may appear a second time
   in a wave-0 create-and-hand-over column, and that second appearance names the
   same final owner in brackets rather than a second claimant. Revision 2 said
   "appears exactly once", which is literally false: I extracted every backticked
   path in this section and 26 appear more than once, 25 of them the declared
   hand-over pattern. The rule is one owner, not one mention. A builder that
   needs to change a path it does not own raises it; it never reaches for it.
2. **Where a shared file must serve several pieces, splitting it is its own
   earlier piece with its own bar.** That is what wave 0 is. Wave 0 creates the
   files wave 1 owns and then hands them over; a created path is listed against
   its wave-1 owner, not against the splitter. **A wave-0 piece that must edit a
   wave-1 file to perform its split carries a bounded, written authorisation
   naming the exact lines, as W0-2 already does for the frontend move.** Wave 0
   closes before wave 1 opens, so an authorisation is a sequence, never an
   overlap.
3. **A frozen file has no owner.** Changing one is an escalation, not a task.
4. **Three classes of path are governed by a rule rather than by a table row**,
   because enumerating them would go stale on the first split: tests, the router
   registration in `server.py`, and the helper modules the 450-line law forces a
   piece to create. Each rule is stated below the tables and is as binding as a
   row.

### 8.1 What wave 0 splits, measured

| File | Lines | Routes | Distinct wave-1 owners it would serve |
|---|---|---|---|
| `tv-break-dashboard/src/TVBreakDashboard.jsx` | 6,236 | n/a, 12 page components inline | 8 |
| `kairos_api/dashboard_api.py` | 1,827 | 8 | 4 |
| `kairos_api/insights_api.py` | 697 | 5 | 5 |
| `kairos_api/catalog_api.py` | 656 | 7 | 4 |
| `kairos_api/version_store.py` | 489 | 5 | 2 |

Every one of those counts I measured this session with a decorator grep and
`wc -l`. `insights_api.py`'s five routes are at `:400` yield-per-second, `:518`
gold-breaks, `:622` scenario-compare, `:630` model/audience, `:646`
make-good-alerts. `dashboard_api.py`'s eight are at `:1633` compliance, `:1638`
overview, `:1695` schedule, `:1711` schedule/segments, `:1728`
schedule/segment/{id}, `:1803` break-operations, `:1817` and `:1823`
break-decisions.

### 8.2 The file-ownership table

Every path a builder may write, either as a row below or under one of the three
rules that follow the tables: tests, the router registration in `server.py`, and
a declared helper module. Nothing else. A path absent from both the table and
those rules is frozen.

#### Wave 0, five pieces, mutually disjoint

| Piece | Paths it may write | Paths it creates and hands over (owner in brackets) |
|---|---|---|
| **W0-1 Router seams** | `kairos_api/dashboard_api.py`, `kairos_api/insights_api.py`, `kairos_api/catalog_api.py`, `kairos_api/version_store.py`, `kairos_api/server.py` | `kairos_api/plan_read.py` **[frozen]**, `overview_api.py` **[P1]**, `week_api.py` **[P2]**, `scenario_compare_api.py` **[P2]**, `yield_api.py` **[P5]**, `day_api.py` **[P3]**, `gold_api.py` **[P3]**, `campaigns_read.py` **[P4]**, `compliance_api.py` **[P5]**, `downloads_api.py` **[P6]**, `model_audience_api.py` **[P7]**, `model_impact_api.py` **[P7]**, `history_api.py` **[P8]**, `pacing_alerts_api.py` **[P11]** |
| **W0-2 Shell seams** | `tv-break-dashboard/src/TVBreakDashboard.jsx`, `App.jsx`, `index.jsx`, `surface-helpers.js`, every existing `src/*.jsx` and `src/*.js` **during the move only**, and all fifteen existing `src/*.css` files (9,306 lines, of which `styles.css` is 6,170) **during the move and the token extraction only** | `src/shell/**` **[frozen after wave 0]**, `src/tokens.css` **[frozen after wave 0]**, plus the destination trees `src/today/**` **[P1]**, `src/plan/week/**` **[P2]**, `src/plan/day/**` and `src/plan/break/**` **[P3]**, `src/clients/**` **[P4]**, `src/rules/**` **[P5]**, `src/sources/**` **[P6]**, `src/model/**` **[P7]**, `src/history/**` **[P8]**, `src/kai/**` **[P9]** |
| **W0-3 Identity** | `kairos_api/advertisers.py`, `kairos_api/advertiser_conditions.py`, `kairos_api/condition_validation.py`, `kairos/optimize/advertiser_rules.py`, `kairos/data/transform.py`, `data/advertiser_rules.csv` | `data/advertiser_names.csv`, `scripts/migrate_advertiser_identity.py`, `kairos_api/spot_ledger.py` **[frozen after wave 0]**, `kairos_api/condition_validation.py` **[frozen after wave 0]** |
| **W0-4 The wall and the words** | `kairos_api/auth.py`, `kairos_api/auth_store.py`, `kairos_api/events_access.py`, `kairos_api/core.py`, `kairos_api/settings_api.py` | `kairos_api/affiliation_wall.py` **[frozen]**, `kairos_api/channel_scope.py` **[frozen]**, `kairos_api/guardrail_store.py` **[P5]**, `kairos_api/model_activation.py` **[P5]**, `data/regulatory_guardrails.json` **[P5]**, `tv-break-dashboard/src/vocabulary.js` **[frozen]**, `tv-break-dashboard/src/session.js` **[frozen]** |
| **W0-5 Evaluation seam and cache** | `kairos_api/overrides.py` and `kairos_api/constraints.py`, **for the bounded extraction below and nothing else** | `kairos_api/preview_inputs.py` **[frozen]**, `kairos/optimize/evaluate.py` **[frozen]**, `kairos_api/read_cache.py` **[frozen]** |

W0-2 is the only piece that may touch a **frontend** file it does not finally
own, and only to move it. Its handover is complete when every `src/*.jsx` at the
top level is either in `src/shell/` or in a destination tree.

**Stylesheets travel with their component.** The third critique found all
fifteen `src/*.css` files frozen by absence while W0-2 owed the design tokens,
which would have stalled the piece on its first line. W0-2 therefore owns them
for the duration of wave zero: it extracts the shared scale, colour and
typography variables into `src/tokens.css`, which freezes at wave-zero close and
is the only place a token may be defined, and it moves every other stylesheet
into the destination tree of the component it styles, where that tree's owner
inherits it. `styles.css` is the shell's and stays with `src/shell/`. No later
piece may define a token; a piece that needs one that does not exist raises it
through the same escalation as a frozen file.

**W0-5 owns `preview_inputs.py` outright, and it carries the one backend
authorisation in the run.** Revision 2 listed the file twice inside a wave whose
heading says its pieces are mutually disjoint, and gave the creation to a piece
that cannot perform it. Measured: the function is `_preview_inputs` at
`kairos_api/overrides.py:244-302`, 59 lines; it is called at `overrides.py:392`
and imported at `kairos_api/constraints.py:328` and `:352` and called at `:330`
and `:355`; `constraints.py:324` names it in a docstring. `overrides.py` is P3's
and `constraints.py` is P5's, both wave 1, and neither file is on W0-1's list. So
W0-1 could not have built it.

- **Owner: W0-5**, from creation. It is W0-5's subject: the cache in section 8.4
  is this function's warm path, and W0-5's first deliverable, the attribution,
  cannot be produced without it.
- **The authorisation, bounded exactly.** W0-5 may write `kairos_api/overrides.py`
  and `kairos_api/constraints.py` for this extraction only: move
  `overrides.py:244-302` into `kairos_api/preview_inputs.py`, delete those lines,
  and rewrite the five referring lines listed above. No other line in either file
  changes. The bar is a diff: `git diff` on those two files touches the moved
  function and the referring lines and nothing else, and every route in both
  modules returns a byte-identical body.
- **The import path every dependent uses:**
  `from kairos_api.preview_inputs import preview_inputs`. The leading underscore
  goes with the move, because a module-private name imported across three modules
  is what created the ambiguity. This is safe to rename: measured, zero files
  under `tests/` reference `_preview_inputs`. The three call sites are
  `overrides.py:392`, `constraints.py:330` and `constraints.py:355`.
- **What does not move.** `_resolved_store_overrides`, `_segment_anchors` and
  `_stored_constraints` stay in `overrides.py` and belong to P3; they are about
  the override store, not about segment construction. `preview_inputs.py`'s only
  `kairos_api` dependency is `kairos_api.core`, imported lazily inside the
  function body today, so the extracted module imports nothing that would create
  a cycle.
- **Sequence.** W0-5 extracts, freezes the module, and hands `overrides.py` and
  `constraints.py` on to P3 and P5 already extracted. Wave 1 never sees a shared
  file.

#### Wave 1, nine pieces, parallel

| Piece | Backend paths it may write | Frontend paths it may write |
|---|---|---|
| **P1 Today** | `overview_api.py`, `kairos_api/target_store.py` (new), `data/plan_targets.csv` (new) | `src/today/**` |
| **P2 Plan, week** | `week_api.py`, `scenario_compare_api.py`, `kairos_api/scenario_api.py`, `kairos_api/recompute_api.py`, `kairos_api/jobs.py`, `kairos_api/plan_version_store.py` (new) | `src/plan/week/**` |
| **P3 Plan, day and break** | `day_api.py`, `gold_api.py`, `kairos_api/overrides.py`, `kairos_api/break_api.py` (new), `kairos_api/break_store.py` (new), `data/breaks.csv` (new), `kairos/export/spots.py` | `src/plan/day/**`, `src/plan/break/**` |
| **P4 Clients** | `campaigns_read.py`, `kairos_api/agencies.py`, `kairos_api/agency_conditions.py`, `kairos_api/campaigns_api.py` (new), `data/campaigns.csv` (new) | `src/clients/**` |
| **P5 Rules** | `compliance_api.py`, `yield_api.py`, `kairos_api/constraints.py`, `kairos_api/_constraint_options.py`, `kairos_api/pricing_api.py`, `kairos_api/events_api.py`, `kairos_api/events_holidays.py`, `model_activation.py`, `guardrail_store.py`, `data/regulatory_guardrails.json`, `data/frequency_rules.csv` | `src/rules/**` |
| **P6 Sources** | `downloads_api.py`, `kairos_api/uploads.py`, `kairos_api/exporters.py` | `src/sources/**` |
| **P7 Model console** | `model_audience_api.py`, `model_impact_api.py`, `kairos_api/audience_api.py`, `kairos_api/model_console_api.py` (new), `kairos_api/model_version_store.py` (new), `models/releases/` (new) | `src/model/**` |
| **P8 History** | `history_api.py`, `kairos_api/activity_log.py` | `src/history/**` |
| **P9 Kai** | `kairos_api/assistant*.py` (**22 modules**, none owned by any other piece; revision 2 said 11 and I counted the glob) | `src/kai/**` |

#### Wave 2, four pieces

| Piece | Backend paths | Frontend paths | Depends on |
|---|---|---|---|
| **P10 Break contents and tonight's breaks** | `break_api.py` extensions inside P3's files, by handover; `kairos_api/break_api_pod*.py` (new); `data/break_pod_order.csv` (new) | `src/plan/break/**`, by handover from P3 | P3 |
| **P11 Pacing and make-good** | `pacing_alerts_api.py`, `kairos_api/makegood_store.py` (new), `data/make_goods.csv` (new) | `src/clients/pacing/**` | P4, decision 4 |
| **P12 Model improvement** | `scripts/adopt_candidate.py` (new), `models/candidates/**`, `models/releases/**` by handover from P7 | `src/model/candidates/**` | P7 |
| **P13 Media verification** | `kairos_api/media_api.py` (new), `kairos_api/media_store.py` (new), `data/media_assets.csv` (new) | `src/plan/break/media/**` | P10, owner media feed |

P10 and P12 take files by handover rather than owning them from the start,
because they extend a thing their predecessor must first prove. A handover is a
declared moment in the run, not an overlap: P3 freezes, then P10 owns.

##### How the orchestration expresses these dependencies

Three of the four wave-two dependencies point back at wave one, which has
closed, so P10, P11 and P12 are mutually independent and start together. The
fourth is inside this wave: P13 writes `src/plan/break/media/**`, which sits
under P10 `src/plan/break/**`. Two agents in one subtree is a collision, not a
scheduling preference, so P13 runs chained behind P10 rather than beside it.

The chain is expressed inside a single concurrent branch, not as a barrier
between two build stages. A barrier would make P13 wait for the slowest of P11
and P12 as well, for no reason: it needs P10 and nothing else. The rule is to
make a dependency exactly as tight as it really is, and no tighter.

##### Durable state: surviving a limit without losing the thread

Wave one lost work to a usage limit three times. What was lost was never the
file edits, which sit on disk and survive anything; it was the thread between
agents. The gap a critic named was interpolated into the next builder's prompt,
so it existed only in the running process, and a process that dies takes it
with it.

The second-order damage was worse than the first. Because the gap text was part
of the prompt, it was part of the resume cache key. One critic that died before
its result was journaled changed the hash of every agent downstream of it, so
agents that had genuinely finished were treated as new calls and ran again.
Measured on the wave-one journal: 187 agent calls started, 128 recorded a
result, and 59 started without ever recording one.

So inter-agent state moves out of the prompt and onto the filesystem, which is
the only thing here that outlives the process. A critic writes its verdict to
`docs/ux-gauntlet/state/<piece>.json` **before** composing its reply, and also
returns it. A builder is told **where to read the gap**, never handed its text.
Three properties follow:

- The prompt now varies only by piece and round number, both deterministic, so
  the cache key is stable and a resume genuinely continues rather than redoing.
- A verdict lost in transit no longer ends the piece. The next builder reads the
  file the dead critic already wrote. Two consecutive losses do stop it, because
  at that point the limit is not letting critics finish at all and continuing
  would only burn builders blind.
- A round that died mid-flight left its edits behind, so every agent is told to
  look at what is already on disk before writing. A resume that rebuilds work
  that is already there is not a recovery, it is damage.

The rule generalises: **state that must survive a process belongs in a file, not
in a prompt.** A prompt is memory, and memory dies with the process holding it.

Two defects in the wave-one script are fixed here, both found by auditing it
rather than by it failing loudly:

- A critic was spawned unconditionally after its builder. `agent()` returns
  null when a run is skipped or dies on a terminal error after retries, so a
  dead builder still bought a full critic round that measured an artifact
  nobody had written. The critic now runs only against a build that exists.
- Verification lived outside the workflow entirely, which is how a
  verification pass once began while builders were still in flight and
  attributed their work in progress to the previous wave. The integration
  critic is now a phase after the barrier, so it structurally cannot start
  before every piece has returned. Attribution against a commit rather than
  against the working tree, recorded in `scripts/gauntlet/README.md`, is the
  other half of that fix.

#### Frozen, no owner

`kairos/service.py`, `kairos/optimize/day_core.py`, `kairos/optimize/optimizer.py`,
`kairos/optimize/objective.py`, `kairos/optimize/_segment_math.py`,
`kairos/optimize/guardrails.py`, `kairos/optimize/pricing.py`,
`kairos/optimize/predicate.py`, `kairos/model/**`, `kairos/data/loaders.py`,
`kairos/data/contracts.py`, `docs/constraint-predicate-contract.md`,
`config/optimization_weights.yaml`, plus every path marked **[frozen]** above.
Changing one is an escalation with a measurement, never a task.

#### Tests

Revision 2 named no test path anywhere, so under its own closure rule no builder
could write a test, while Bar 4 requires the suite green and W0-1's own bar is a
response diff, which is a test. Measured today: **125 Python files under
`tests/`, 122 at the top level and 3 under `tests/validation/`**.

**Correction, 2026-08-01, and it changes every piece's Bar 4.** This section
first recorded 3,102 collected tests, taken with a bare `pytest --collect-only`
from the repository root. That command also collects the vendored Google
Meridian library under `meridian/`, whose twenty test files are not this
product's and whose hundreds of failures are identical on both trees. Two
independent measurements caught it, one in the wave-close gate and one in the
verification harness, both after the gate refused to pass a tree because of
failures nobody here owns. **The product's suite is `tests/` and nothing else**,
which is roughly 1,600 tests: the brief's original figure of 1,438 passing was
about right for the product, and the 3,102 that replaced it was measuring
somebody else's code. Every count in this campaign, and every invocation in the
gate, is scoped to `tests/`. The number a piece must keep green is the product
suite, never the repository.

1. **Adding a test needs no permission and cannot collide.** A piece creates test
   files under its own reserved prefix, `tests/test_<piece>_*.py` with the piece
   id lowercased and hyphens dropped: `tests/test_w0_1_route_identity.py`,
   `tests/test_p3_break_store.py`. The prefix is unique per piece by
   construction, so no table entry and no lead ruling is needed. Every piece is
   expected to use it; a piece that adds behaviour and no test has not finished.
2. **An existing test file's owner is derived, not listed.** A test file belongs
   to the piece that owns every production path it imports. If it imports paths
   owned by two or more pieces, or any frozen path, it has no owner and is
   frozen, like any other shared file. **Measured against the table above: 16 of
   the 125 files resolve to exactly one owner (W0-1 six, P9 five, W0-4 three,
   W0-3 two), 106 span owners or reach a frozen path, and 3 import no production
   module at all (`conftest.py`, `test_rebuild_equivalence.py`, which drives the
   product through a subprocess, and `validation/test_placebo_fast.py`), which
   are frozen as the harness.**
3. **That 106 is the rule working, not the rule failing.** A piece never needs to
   change a shared test in order to add something; it needs to change one only
   when it changes behaviour that a shared test asserts, and behaviour another
   piece depends on is exactly what Bar 3 forbids changing quietly. So the
   expected number of test-change requests is not 106, it is the number of
   deliberate behaviour changes, and this document already enumerates those.
4. **The protocol when a piece must change a test it does not own.** It does not
   edit it. It opens a test-change request naming four things: the test path, the
   assertion that must change, the production change that forces it, and which of
   the two it believes is wrong. **The request goes to C2, not to the lead**,
   because a changed assertion is Bar 3's own subject: a test asserting today's
   behaviour is the machine-readable form of "what works today that this piece
   must not make worse". C2 rules one of three ways, and records the ruling: the
   test encodes a defect this document declares fixed, so C2 makes the edit and
   the piece cites the ruling; or the test encodes behaviour Bar 3 protects, so
   the piece changes its own code; or the test's only other owner has not started,
   so ownership transfers with the ruling attached. **Deleting a test is never a
   resolution unless the module it covers is deleted, and that deletion needs the
   written proof section 5.6 requires.**
5. **Two forced requests are already known, so this is not hypothetical.** The
   version-store isolation guard of section 5.6 changes where a test may write,
   and 186 of the 200 manifests on disk were written by pytest, so every test that
   writes the store is affected. And `tests/test_agreements.py` is why section 5.6
   no longer authorises removing `kairos/optimize/agreements.py`: the file is
   imported and re-exported at `kairos/optimize/__init__.py:26,44` and the test
   exercises it, so the removal revision 2 authorised would have failed the suite
   with no owner able to fix the test.

#### `kairos_api/server.py` after wave 0

Every router is mounted by an explicit line: **20 `app.include_router(...)` calls
between `server.py:112` and `:271`**, interleaved with the auth middleware, the
activity recorder and CORS, whose order is load-bearing and documented in the
file. W0-1 owns that structure and mounts everything wave 0 creates, including
the thirteen route-carrying modules it splits out of the four files in section
8.1. Four later pieces create a router of their own
(`break_api.py` P3, `campaigns_api.py` P4, `model_console_api.py` P7,
`media_api.py` P13), and `pacing_alerts_api.py` needs no append because W0-1
creates and mounts it in wave 0.

`server.py` is not handed over and is not frozen. It becomes **frozen above a
marker and append-only below it**:

- W0-1's last act in wave 0 is to add one marked region after the final existing
  mount and to leave everything above it unchanged. Everything above the marker
  is frozen for the rest of the run, exactly as if it were on the frozen list.
- Below the marker a piece appends **exactly one stanza of two lines**: the
  import of its own router and the `include_router` call for it. It may not
  reorder, edit or remove another piece's stanza, and it may not touch a line
  above the marker. Anything else in this file is an escalation to W0-1's owner.
- Stanzas are appended in piece order. Two pieces appending at once produce a
  conflict whose only correct resolution is to keep both stanzas, so no ruling is
  needed.
- **The bar on an append is an OpenAPI diff, because the real risk is a late
  mount shadowing an existing route.** Baseline measured this session on the live
  instance: **90 paths, 113 operations, 56 of them writes.** After a piece
  appends, `GET /openapi.json` must show those 90 paths unchanged plus exactly
  the paths that piece published and froze under section 8.8, and no path may be
  redefined.

#### The six modules the table did not name, and the ones the 450-line law will create

I diffed every backticked path in this section against the real file list by
exact basename. `kairos_api` holds **51 modules** and six were absent, so rule
8.2 froze them, and five sit on a wave-1 critical path. Each now has an owner,
chosen by who imports it and who must change it:

| Module | Lines | Imported by | Owner, and why |
|---|---|---|---|
| `_constraint_options.py` | 249 | `constraints.py:44` at module level, `scenario_api.py:425` and `assistant_propose_tools.py:44` lazily | **P5.** P5's restriction language translates the option payload this module builds. P2 and P9 read it |
| `audience_api.py` | 114 | `insights_api.py:638`, `core.py:452`, `assistant_audience_model.py:25` | **P7.** It is the single reader of `models/audience_model.json` on the API side by its own docstring, and P7 has to wall that read. **Stated duty:** `audience_model_note` rides run payloads and returns `{state, computed_at}` today (`audience_api.py:91-113`); it may never gain a gate, a coefficient or a p-value, because that would fail section 4.2's lexicon test on every operator surface at once |
| `jobs.py` | 114 | `recompute_api.py:91,126,159` only | **P2.** One importer, one owner |
| `condition_validation.py` | 74 | `advertiser_conditions.py:44` (W0-3), `agency_conditions.py:45` (P4) | **W0-3, frozen at wave-0 close.** Two claimants in two different waves, so the earlier one owns it and the later one reads it, which is rule 8.0.2 |
| `events_holidays.py` | 35 | `events_api.py:51` only | **P5.** One importer, one owner |
| `__init__.py` | 2 | n/a | **Frozen.** One docstring line, no code |

**The rule for the helpers that do not exist yet.** The 450-line law
(`docs/ux-gauntlet-prompt.md:205`, "No source file over 450 lines. Split rather
than compress") will force more of these, and enumerating them now would be a
guess. Measured, it is already binding on owned files before anyone adds a line:
`assistant.py` 775, `uploads.py` 713, `core.py` 698, `assistant_actions.py` 496,
`scenario_api.py` 477 and `overrides.py` 467 all exceed the cap today, on top of
the five files wave 0 exists to split.

- **The frontend needs no rule.** Every wave-1 frontend path is a glob
  (`src/plan/week/**`), so a split inside an owned tree is owned by
  construction. This is a backend problem only, because `kairos_api/` is a flat
  package owned file by file.
- **Naming.** A helper split from an owned module is named
  `<parent stem>_<role>.py` and lives beside its parent, which is what the code
  already does: `events_access.py` says in its own docstring that it was "split
  out of events_api.py to keep that module under the file-size cap".
- **The collision check is the one section 8.8 already runs.** Before its wave
  starts, each piece publishes and freezes the files it will touch. That
  publication now includes the helper names it intends to create. The lead
  rejects a duplicate at publication time, before any code exists, which is the
  cheapest possible moment.
- **A helper declared that way is not frozen by absence.** It is owned by the
  piece that declared it, from the moment the publication is frozen, and it
  inherits that piece's bar.

### 8.3 The five cross-cutting rules, each with a named owner

Revision 1 stated these and assigned them to nobody. Each now has one owner for
the mechanism and a stated adoption duty for every other piece.

| Rule | Owner of the mechanism | What the owner ships | What every other piece must do |
|---|---|---|---|
| **The `job` field and per-job landing** | **W0-4** | `job` on the account record with `unset` default, the **thirteen-value** list (revision 2 said eleven here and thirteen at sections 2 and 2.2; thirteen is right, and eleven is the count of roles landing inside the five workspaces), the door map, `session.js` exposing it, and the job picker card contract | P1 renders the picker. Every piece registers its door name in the map. Nobody else writes the field |
| **The affiliation wall, `can_edit`, session affiliation** | **W0-4** | `affiliation_wall.py`: one decorator that gates a route on `affiliation = company` for read and write, one helper that stamps `can_edit` into any response, `session.js` exposing affiliation to every surface | Every piece applies the decorator to its own walled routes and stamps `can_edit` on its own responses. The four open reads (`/api/impact`, `/api/model/audience`, `/api/parameters`, `/api/events` `model_context`) are closed by **P7, P7, W0-4, P5** respectively |
| **The competitor boundary** | **W0-4** | `channel_scope.py`: one function that takes the operator channel from settings and filters any plan projection to it, plus the unnamed-aggregate form for the model's competitor factor | P2 applies it to `/api/schedule` (measured: 96 `קשת 12`, 73 `כאן 11`, 28 `עכשיו 14`, 3 of the operator's own). P3 applies it to `/api/break-operations` (measured: 12 programmes per channel, all four). P9 applies it to Kai's context. C1 verifies |
| **The vocabulary rename, both languages** | **W0-4** | `vocabulary.js`: the section 4.8 and 4.9 tables as the single string source, with the retired words absent from it | Every piece imports its labels. **The critic's check is a grep:** zero occurrences of recompute, rebuild, חישוב מחדש or בנייה מחדש anywhere under `tv-break-dashboard/src/` outside `vocabulary.js`. Today that grep returns 159 hits |
| **Lifting guardrails and `audience_model_activation` out of `KairosSettings`** | **W0-4** | `guardrail_store.py` with effective date and change record, `model_activation.py` with the company gate built from the same guard `pricing_api.py:232-241` already uses, and a compatibility shim in `core.py` so no reader's import changes | Nobody, and **both stores hand over to P5, not to P7**. The switch is a run-side configuration act by section 4.1 and its surface is Rules. P7 renders a read-only mirror of the state in the console and owns no control. The change is subtractive and its bar is that every other module's imports are unchanged, proven by grep |

### 8.4 The latency bars, and the engine change each one actually needs

Revision 1 attached a 3 s bar and a 500 ms bar to two pieces and gave the work
to a caching piece. The critique said a cache cannot answer a placement nobody
has made. **I measured the components in process, and the critique's diagnosis
is wrong in a way that matters.**

Measured on `רשת 13 / 2024-11-01`, 82 segments, with
`~/.venvs/meridian/bin/python`:

| Component | Measured |
|---|---|
| `_preview_inputs`, building the segments, first call | **6.38 s** |
| `_preview_inputs`, second call in the same process | **0.01 s** |
| `_optimize_one_day`, one leg | **0.98 s**, then 0.80 s |
| `_group_objective_contribution` over all 82 segments | **0.00006 s** |
| Ratio, one optimize to one evaluation | **15,783x** |
| `GET /api/constraints/effect` end to end | **19.77 s** |
| `GET /api/overrides/effect` end to end | **15.72 s** |

Three consequences.

**The double optimize is not the problem.** Two legs cost 1.8 s of a 19.77 s
response. Removing one saves 0.9 s. The critique, revision 1 and
`06-baseline.md` all named it as the cause and it is 9 percent of the cost.

**Segment construction is the largest attributed cost and it is cacheable.**
6.38 s cold, 0.01 s warm, and it is placement-independent, so a cache genuinely
does answer a placement nobody has made yet. That is W0-5's `read_cache.py`
applied to `preview_inputs.py`, which W0-5 extracts itself as its first act under
the bounded authorisation in section 8.2.

**About 13 s of the 19.77 s is unattributed today.** I measured the components
and the endpoint; they do not sum. So **W0-5's first deliverable is an
attribution, not a fix**: a per-stage timing of the request path from route
entry to serialization, published before any bar is set. A builder graded
against a bar whose cause is unmeasured will either fail or fake it.

The engine change, named:

> **`kairos/optimize/evaluate.py`, a pure scoring seam.** Given the built
> segments for one channel-day plus a candidate break count and placement map,
> return `(objective, revenue, retention)` without allocating anything. It is a
> thin wrapper over `_group_objective_contribution`, which is already pure and
> already additive by its own docstring: "summing every group's contribution
> reproduces the global convex-blend objective"
> (`kairos/optimize/_segment_math.py:14-19`). Measured floor: **60
> microseconds** for 82 segments.

That is what answers JS-3's 500 ms bar. A drag does not need a re-optimization;
it needs a score of the placement the person just made, which is an evaluation
and not an allocation. The seam is frozen on delivery and P3 consumes it.

For JS-4's 3 s preview, the composition is: warm segments (0.01 s) plus one
constrained leg (0.98 s), with the baseline leg read from the saved plan on
disk rather than recomputed. That is roughly 1 s of engine work inside a 3 s
budget. **The bar is conditional on W0-5's attribution closing the remaining
13 s.** If it does not, P5's honest bar becomes "the matched segments and their
scored delta inside 3 s, the full re-allocation streamed after", and that
change is recorded rather than quietly dropped.

### 8.5 Bar 3, on every piece

`docs/ux-gauntlet-prompt.md:178`: "This is a rebuild of a living system, so
every critic compares three things, not two: today's Meridian, the new
Meridian, and the reference. If today's version wins on any job story, that is
a gap and it goes back to the builder with the reason."

Revision 1 contained the word zero times while dissolving or merging twelve of
seventeen surfaces. Every piece below carries the specific thing it must not
lose, measured today. **A piece is not done until C2 has run the three-way on
its own regression row.**

| Piece | What works today that this piece must not make worse |
|---|---|
| W0-1 | All 25 split routes return byte-identical bodies. The response diff is the bar, not a smoke test |
| W0-2 | Every one of the 17 current routes renders the same DOM text after the split. The drag in the schedule editor still moves a chip in 2.43 s (`06-baseline.md:172`). The `#Assistant` hash still opens the dock over the current page |
| W0-3 | Agencies still resolve 9 of 9 and still total gross ₪699,450 / net ₪669,978 / 119 spots. Every engine figure byte-identical |
| W0-4 | The three event writes still refuse a channel account with the existing Hebrew denial. `GET /api/events` still returns `can_edit`. The five `require_company_editor` call sites still fire |
| W0-5 | No endpoint gets slower. The saved plan is byte-identical after the cache lands. The extraction moves `overrides.py:244-302` and the five referring lines and nothing else, proven by the diff, and `/api/overrides/effect` and `/api/constraints/effect` return byte-identical bodies before and after it |
| P1 | The amber staleness banner still names what changed and still offers the run. "Priority decisions, 5 actions" still lists the same five with the same figures. The cold answer still lands by 3.59 s |
| P2 | The frontier point can still be clicked and applied as a saved retention floor, which is unique to Overview today (`01-surfaces.md:77`). The plan CSV still downloads 8,704 rows. The four objective templates survive |
| P3 | Drag and resize still work with 30 s and 60 s snap and the zoom scale. The segment inspector still opens from the break list. The override preview still reports rejected overrides verbatim |
| P4 | Agency records keep payment terms, rebate percent, commission percent, credit limit, VAT id and two contacts. Deactivate still beats delete. The 41 observed advertiser links still render |
| P5 | The seven compliance checks still return with profile, `effective_date` 2026-06-14 and `source_url`. The predicate builder's AND/OR grammar still saves the same rows. The six pricing layers keep their honest live and wired-off chips. `GET /api/yield-per-second`, which moves to P5 with the rate card, still returns `yield_per_second` 142.7044 with its `basis.formula` and its named inputs |
| P6 | All five report CSVs still download, all with the same row counts, and Download all still works. The upload validator still refuses a bad file at the door with the contract's own findings |
| P7 | Nothing regresses: the surface does not exist today. The gate table must show every verdict the calendar shows today, and no operator surface may keep showing them |
| P8 | The restore path still restores the same nine logical files and still snapshots first. Viewer write-lock at `AssistantVersions.jsx:296` survives |
| P9 | The propose-only contract holds: 31 read, 8 propose, 0 write. The Hebrew grounded answer with its source, window and unprompted staleness warning still happens |
| P10 to P13 | Nothing regresses: none of these exists today |

The measured honest empty states are a product-wide Bar 3 row of their own.
`06-baseline.md:445` recorded five: campaign revenue dashes with the reason,
advertiser revenue null with the reason, "Net after retention cost: Not
exposed" with the reason, the make-good panel naming the exact missing file,
and the gold-breaks panel. **Every one must survive as a control with a path
forward, never be replaced by a figure.**

### 8.6 The critics

- **C1 The boundary sweep.** JS-18 against the assembled product: a
  channel-affiliated account, every door, every ambiguous button, every
  competitor name, plus the three checks of section 4.2 run on all five
  workspaces.
- **C2 The three-way.** Bar 3, run per piece against the regression rows in
  8.5 and against `06-baseline.md`'s frozen figures. This critic exists in this
  revision and did not exist in revision 1. **C2 also rules on every
  test-change request**, by the protocol in section 8.2, because a request to
  change an assertion is a request to change behaviour something else depends on,
  which is the same question C2 is already answering.
- **C3 The integration critic.** One fresh critic over the whole product:
  vocabulary and interaction consistency across all surfaces, correctness end
  to end, and whether the assembled thing serves the jobs it was decomposed
  from.

### 8.7 Why this order

Identity, the seams, the wall and the evaluation primitive are first because
every other piece is either false, uncoordinated, unsafe or slow without them,
and because after the split they touch disjoint files rather than merely
disjoint surfaces. The break entity sits inside P3 rather than in its own piece
because a break with no board to live on cannot be judged, and its contents move
to P10 because the pod is a second, larger job that needs the entity to exist
first. Media verification is last of the buildable set because it is the only
piece whose bar cannot be met without data the owner has not supplied.

### 8.8 The contract that keeps pieces independent

Before a wave starts, each piece publishes and freezes four things: the endpoints
it owns, the payload shapes it emits, the files from section 8.2 it will touch,
and the helper module names it intends to create. The lead rejects a duplicate
helper name at that moment, which is the collision check section 8.2's helper
rule relies on.

The shared surface is five things and no more: the design tokens and shell from
W0-2, `vocabulary.js` from W0-4, `evaluate.py` and `read_cache.py` from W0-5,
`plan_read.py`, `preview_inputs.py` and `spot_ledger.py` as frozen read layers,
and **the append-only registration region at the foot of `server.py`**, which is
the only shared file any piece writes to and the only one with an append
protocol rather than an owner. Everything else on that list is frozen when wave 0
closes, so a wave-1 builder reads it and never writes it.

---

## 9. What this spec deliberately does not do

Each with its reason, so that a critic does not grade an absence as a miss.

1. **No media probe or transcode pipeline.** There is no ffmpeg, ffprobe or
   MediaInfo dependency and no media file of any kind on disk. Nothing here can
   read a video. P13 builds the arithmetic verification, which is real today,
   and states the technical gap in exact terms.
2. **No invented "on plan".** JS-1's third answer needs a target the owner has
   not supplied. Until then it is an honest empty state with a path forward, not
   a number derived from the plan itself, which would be circular.
3. **No integrations.** Zero exist: no webhook, SFTP, BXF, AdID or inbound API
   anywhere, and the only outbound clients are Anthropic and Gemini. The owner
   has not named the system of record for campaigns or the delivery format for a
   locked break, and both are needed before an integration can be honest.
4. **No change to the frozen predicate contract.**
   `docs/constraint-predicate-contract.md` is frozen. The restriction language
   in P5 is a translation layer above it. Two genuine engine extensions are
   additive: an offset measured from programme end, and an occurrence concept.
5. **No activation of the wired-off pricing layers.** Position carries a 1.30
   first-position multiplier and ad type carries a 0.00 promo multiplier.
   Activating either moves real revenue. They are owner-gated and stay that way.
6. **No renamed API paths.** `POST /api/recompute-schedule` and its siblings keep
   their addresses. Vocabulary changes are labels and messages.
7. **No mobile or responsive redesign.** Every job story is a desk job.
8. **No multi-tenant or multi-channel operation.** The operator owns exactly one
   channel, read from settings.
9. **No general query or BI surface.** JS-9 is served by drilling from a figure
   to its breaks and then its spots. An arbitrary query builder is a different
   product.
10. **No deletion of the version store's polluted history.** 187 of 200 entries
    point at pytest temporary paths. They are marked unrestorable and the store
    gains an isolation guard. Deleting an operator's history to tidy a defect is
    the wrong trade.
11. **No engine number moves without a proof.** Revenue, retention, pricing and
    rating figures stay byte-identical unless a change is proven a genuine fix,
    declared, and measured. P12 is the one piece whose whole purpose is to
    propose such a change, and it escalates rather than ships.
12. **No new persona surfaces for the deployment owner.** JS-17 is served by an
    honest banner and a good bootstrap, not by a page. The work is operational.
13. **No money from `data/Spots.csv`.** Its `revenue_ils` column is
    `50 * TVR * Duration * total_premium` with a constant base rate, verified on
    99.67 percent of 50,386 rows, so the ₪306.9M it sums to is a synthetic price
    and never renders as money. It is adopted as a test corpus only, section 5.4.
14. **No merge of projected and delivered money into one figure.** They come
    from different grains over non-overlapping dates. Section 3.4 states the
    limit; closing it is owner decision 4.
15. **No multi-worker deployment.** `auth_store.py`, `activity_log.py` and
    `jobs.py` each state the single-process assumption in their own docstrings,
    so W0-5's speed work is single-worker work. Raising the worker count would
    need a shared session store and is not in scope.

**The brief's model mandate is in scope and is now a piece.** Revision 1 turned
`docs/ux-gauntlet-prompt.md:94`, "The model is in scope, and it must be the best
one this data can support", into a console that displays gates, which is showing
and not improving. **P12** improves it: `models/candidates/` holds five
artifacts (afterwindow, calibrated, competitor, placebo_corrected, spotclip) and
`scripts/estimate_candidate_revenue_movement.py` already computes the money each
would move. P12 re-measures each candidate's held-out gates, computes the
revenue movement, and records a ship or no-ship verdict against a named model
version, escalating any adoption that would move a figure. That is measured
improvement under the same discipline, not more parameters. It is frozen as
JS-19 in the job stories' amendment log.

---

## 10. Where the evidence conflicted, and what I trusted

Recorded because the lead needs to know which numbers are load-bearing.

| Conflict | What I trusted | Why |
|---|---|---|
| Endpoint count: the brief says 111, discovery says 113 | **113 operations over 90 paths, 56 of them writes** | Counted from the live `openapi.json`. Two investigators and my own count agree |
| **HEAD: `05-gaps.md:8`, `03-people.md:9` and `04-training-vs-runs.md:9` name `5a80a709`; `06-baseline.md:5` names `342a2896`** | **`5a80a709`** | `git rev-parse HEAD` returns `5a80a7098a64c1763fec532f12fd66e7fb0ed824`. `342a2896` is an ancestor 31 commits back. This matters more than the other rows because `06-baseline.md` supplies every "Baseline today" figure frozen into the job stories, which Bar 3 grades against. **C2's first act is to re-measure the `06-baseline.md` figures at `5a80a709` before using any of them as a regression floor.** I did not determine which tree the baseline browser ran against, only that the two provenance lines disagree by 31 commits |
| **Monday-first `dayKeys`: `05-gaps.md:681` says line 586, `07-references.md:513` says 585** | **585** | `const dayKeys = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];` is at `TVBreakDashboard.jsx:585`, and it is consumed at `:956` and `:957`. Recorded because choosing right without recording the conflict leaves a critic to rediscover it |
| **"the same two words name both activities 159 and 124 times"** | **One word, two counts, and a second word with two more** | `04-training-vs-runs.md:670-671` measured `recompute` at 159 in the UI and 124 in the backend, and `rebuild` at 9 and 85. Revision 1 compressed two rows into one sentence. Section 4 now states it correctly |
| **"Six gate-override flags"** | **Five** | I enumerated the `add_argument` calls in `scripts/compute_measured_coefficients.py`: `--series`, `--counterprogramming`, `--placebo-correction`, `--interval-calibration`, `--moderated-variances`, `--output`. `--output` is an output path, not a gate override. `04-training-vs-runs.md:651` says five and is right |
| Assistant allowed settings fields: discovery says 20 | **19** | Enumerated `ALLOWED_SETTINGS_FIELDS` in process. `audience_model_activation` is not among them either way, so the conclusion is unaffected |
| Test count: the brief says 1,438 passing and 4 skipped | **The brief was right and this row was wrong.** The 3,102 recorded here came from a bare `pytest --collect-only` at the repository root, which also collects the vendored Google Meridian library under `meridian/`. Scoped to `tests/`, the product suite is roughly 1,600 tests | Caught on 2026-08-01 by two independent measurements after the wave-close gate refused a tree over failures nobody here owns. Every count and every gate invocation is now scoped to `tests/` |
| `/api/overview` latency: 1.28 s, 5.84 s and 3.5 to 6.7 s reported by three investigators | **The range is real and the variance is the defect** | A read that varies 5x is a UX problem in itself, so W0-5 is graded on p95, not on a best case |
| `run_log.jsonl`: 488 versus 489 records | **489 lines**, re-measured this session | Trivial, recorded for completeness |
| Version store: 200 manifests versus 201 directory entries | **200 manifests** | The extra entry is a directory listing artifact. The load-bearing number is that 187 of them point at pytest paths |
| The competitor lanes in the planning grid: one investigator measured them and explicitly declined to rule | **It is a law breach and it is fixed** | The law reads "No rival channel's name or data reaches an operator surface or the assistant's context." I fetched both myself: `/api/schedule`'s 200-row projection is 96 `קשת 12`, 73 `כאן 11`, 28 `עכשיו 14` and 3 of the operator's own; `/api/break-operations` returns 12 programmes on each of the four channels. The model may continue to use competitor lineup internally (its gate is on, at +2.16 percent held out). The operator surface shows the operator's channel and, where the model uses a competing lineup, an unnamed aggregate. **This removes something visible today, so it is flagged for the owner rather than done silently** |
| **`data/Spots.csv` gap counts: revision 2 printed "2 of 15,214 within" and "702 of 2,412 boundaries, 29.1 percent"** | **15,614 within-break pairs with 0 above 60 s; boundaries are method-dependent, 1,880 with 625 at 60 s or less on the second-resolution rows, 3,025 with 1,667 if the minute-resolution rows are recovered** | Neither of revision 2's counts reproduces under eight variants between the second critic and me, and no channel yields 2,412 boundaries. Section 5.4 now carries the method beside the numbers so a third party can run it. The within count is structural (rows minus breaks, 18,669 - 3,055) and identical under both parses. The conclusion, that a gap rule does not reproduce `break_id`, is supported more strongly by the corrected numbers than by the originals |
| **The daily file: revision 2 said a 60 s gap rule "reproduces those same 10 groups exactly"** | **It does not. Ten groups for ten, but a different partition, with two counterexample rows** | Measured on all 175 rows, all of which parse. Sizes under the rule are 1, 3, 4, 7, 7, 22, 28, 30, 35, 38 against declared 1, 1, 3, 3, 7, 28, 29, 30, 35, 38; `21:22:12` merges forward across a 22 s gap and `22:59:40` splits at a 93 s internal gap. Six other thresholds fail too. **The same sentence, and a 13.4 minute figure that measures 13.93, are in `decisions-for-owner.md:62-63`, which I may not write. Both must be corrected there before it reaches the owner** |
| **`data/Spots.csv` channel flags: revision 2 read `competitor_flag` as a two-channel partition** | **The flag does not partition two channels, and it is inverted against this operator** | `is_target_channel` is True on all 23,707 `קשת 12` rows and False on all 18,669 `רשת 13` rows; `competitor_flag` is the complement, True for the operator's own `רשת 13` and the two other rivals; `include_as_media` is True only for `קשת 12`. Settings set `operator_channel: רשת 13`. Flagged in section 5.4 as a data defect the rebuild must not inherit, with the rule that the scope comes from settings and never from these columns |
| **Version store pollution: revision 2 called "187 of 200" the load-bearing number** | **188 of 200 point outside the repository; 186 of those name a pytest temp directory** | Parsed every `data/versions/*/manifest.json` and tested each `files[].path`. 187 reproduces under neither method, so section 5.6 now states the method with the number |
| **`data/kairos_constraints.csv`: revision 2 said four modules reference it** | **Seven non-test modules** | Repo-wide grep excluding `tests/`, listed in section 5.6. The file is absent from disk |
| **`kairos/optimize/agreements.py` listed as "zero callers" and authorised for removal** | **It has callers and a test** | `kairos/optimize/__init__.py:26` imports it and `:44` re-exports `load_agreements`, and `tests/test_agreements.py` exercises it. Removing it fails the suite. Section 5.6 no longer authorises the removal |
| **`kairos_api/assistant*.py`: revision 2 said 11 modules** | **22** | Counted the glob. It doubles P9's surface, so section 8.2 states it |
| **`data/Spots.csv`: `02-api-and-data.md:301` says "nothing on the engine path", the critique says "read by nothing"** | **Referenced in nine places, none of which reads the three columns in question** | Six are cache-key signatures, one is the uploaded-CSV fallback used only when the xlsx is absent, and that fallback "reads only the shared columns" by its own docstring at `loaders.py:38`. A repo-wide grep for `break_id` outside the daily path returns nothing. Both statements are right about the columns and loose about the file |

### What still needs a human decision

Five things block a design decision and evidence cannot settle them. Each is
written up with its options, its evidence, my recommendation and what is blocked
until he answers, in `docs/ux-gauntlet/decisions-for-owner.md`. In summary:

1. **Advertiser identity.** Which of two honest methods to use, given that 45
   ids cannot map onto 41 names. Blocks W0-3's final shape, and through it P3,
   P4 and P11.
2. **The pod boundary.** Neither the daily file nor `Spots.csv` derives it, and
   I measured both. Blocks P10.
3. **The plan target.** Blocks P1's third answer.
4. **A current week and a delivery feed.** Blocks P11 and the delivered half of
   section 3.4.
5. **What publishing means and who owns the regulatory guardrails.** Blocks P2's
   done condition and P5's second half.

The media feed, previously listed here, is folded into decision 4's family and
is stated in full in P13's row of section 6; it blocks nothing else.

---

## 11. What changed in revision 2, so the fixes can be checked rather than re-derived

| Critique finding | Where it is closed |
|---|---|
| Build order allocated by surface, six named collisions | Section 8.2, a file-ownership table where every path appears exactly once, plus wave 0 splitting five files by owner |
| `insights_api.py` serves five pieces | W0-1 splits it into five modules, section 8.2 |
| `dashboard_api.py` named as the problem, assigned to nobody | W0-1 owns it and splits it into six modules |
| W0-A and W0-B collide on `advertisers.py` | Merged: W0-3 owns the file and its latency, so there is no second claimant |
| Five cross-cutting rules with no builder | Section 8.3, all five owned by W0-4 with a stated adoption duty per piece |
| Latency bars need an engine change, not caching | Section 8.4. The engine change is named (`evaluate.py`, 60 microseconds measured), and the critique's own diagnosis is corrected with measurements |
| "45 of 45 advertisers named" is impossible | Section 5.5, replaced with "41 of 41 that appear in the daily file", and the method is owner decision 1 |
| Events authoring filed under training | Section 4.1, one test, applied. It writes `data/`, so it is configuration and lives on Rules |
| History merges model releases affiliation-blind | Section 4.1, History filters by artifact root; `models/` entries are company-only |
| Six further blur points | Sections 4.1 to 4.7: `model_context` on operator surfaces (4.2 test 2), the silent money move (4.6 release note), the unowned wall (8.3), the two output nouns (4.8), the context switcher (4.7). **The remaining one, blur 3, the upload consequence, was claimed closed at "5.6 and P6" and was not in the document at all. It is closed now, in section 5.6, and this row is corrected rather than left standing** |
| `data/Spots.csv` dropped | Section 5.4, measured myself, adopted as a test corpus, rejected as money with the formula |
| Bar 3 absent | Section 8.5, a regression row per piece, plus critic C2 |
| Model mandate became a console | Section 9, P12, plus JS-19 |
| Three untraced rows in 3.3 | Section 3.5, the claim is fixed and the exceptions are labelled |
| INFERRED grades laundered | Section 2, restored with the source line for each |
| Misattributed counts, HEAD conflict, `dayKeys` line | Section 10 |
| `frequency_rules.csv`, `/api/settings/controls`, the second upload system, `InventoryHeatmap`, the make-good, the posterior in the file list | Sections 5.6, 5.6, 5.6, 3.5, 5.2 and 6, 5.6 |
| A/B: four classification axes | Section 3.2, one axis, one home plus four object families, and the count is derived |
| A/B: money on five surfaces | Section 3.4, one money layer, the break, two named quantities, one drill |
| A/B: Plan and Days duplicate the timeline | Section 3.3, one destination, zoom as a control |
| A/B: no way back from a record | Section 3.6, the `1 / 31` counter on every drill |
| Five-second walk: three roles cannot land | Section 2, thirteen doors; the model steward's switcher is specified in 4.7 and JS-11's picker in 2.2 |
| Five roles ambiguous | Section 2, each ambiguous role now lands on a named view rather than a destination |
| The `job` field has no builder | Section 8.3, W0-4 |
| Owner-blocked marking inconsistent | Section 6, decisions 1 to 5 named in the rows they block, and section 10 |

### What the second blind critique closed, in one further pass

The second critique returned NOT READY narrowly, with blocker 2 fully closed,
blocker 1 closed for all six named collisions, Bar 3 genuinely closed and twelve
of thirteen roles landing clean. Six things remained. They are closed here, and
nothing else in the document moved.

| Finding | Where it is closed |
|---|---|
| Section 8.2 declares itself total and never names `tests/`, while Bar 4 requires the product suite green | Section 8.2, "Tests": a reserved prefix per piece for new tests, a derived owner for existing ones (measured: 16 of 125 files resolve to one owner, 106 are shared, 3 are the harness), and a four-part test-change protocol ruled on by C2 |
| `server.py` is W0-1's, is never handed over, and four later pieces must mount a router in it | Section 8.2, "`kairos_api/server.py` after wave 0": frozen above a marker, append-only below it, one two-line stanza per piece, with an OpenAPI diff against 90 paths and 113 operations as the bar |
| Six of the 51 `kairos_api` modules are absent from the table and therefore frozen, five on wave-1 critical paths | Section 8.2, "The six modules the table did not name": each assigned by its importer, with the dual-consumer duty on `audience_api.py` stated |
| The 450-line law will force helpers nobody owns | Same subsection: naming convention plus the section 8.8 publication as the collision check, with the six owned files already over the cap measured |
| `preview_inputs.py` has two wave-0 claimants and neither can build it | Section 8.2: W0-5 owns it from creation and carries a bounded extraction authorisation over `overrides.py:244-302` and five referring lines, with the import path every dependent uses |
| `audience_model_activation` writes `data/` yet was filed on the training side | Sections 4.1 and 4.5: the artifact stays training, the switch is a run-side company-gated configuration act on Rules built from the guard `pricing_api.py:232-241` already uses, with what each side sees. Every other flag re-checked against the rule in 4.1 |
| The upload-staleness closure claim at section 11 pointed at text that did not exist | Section 5.6, and the row above is corrected. The measurement changes the answer: today an upload cannot make the model stale, and the one condition under which it can is computable and already computed |
| Two measured claims do not reproduce, plus a data defect | Section 5.4 for the gap counts with their method, the daily-file grouping with its two counterexamples, and the inverted channel flags; section 10 records all three plus four smaller corrections |
| The revenue and yield owner's door does not hold the answer to their question | Section 2, and `yield_api.py` moves to P5 in section 8.2 so the rate card can carry the figure and its unsaved-edit delta |
