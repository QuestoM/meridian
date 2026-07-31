# Meridian: the architecture the jobs imply

Written 2026-07-31 against HEAD `5a80a709`, from the seven discovery reports in
`docs/ux-gauntlet/discovery/` plus my own measurements on the running instance at
`http://127.0.0.1:8010`. It is a recommendation, not a survey. Where the
evidence conflicted I say which I trusted, in the last section.

The companion document, `docs/ux-gauntlet/job-stories.md`, is frozen. Every
decision below is justified against a numbered story in it.

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

## 2. The people, and where each one lives

Discovery found sixteen distinct accountabilities and one served persona. Sixteen
accountabilities do not mean sixteen homes. They mean sixteen questions, and
questions cluster.

Each person below gets exactly one home: the surface they open the product on,
where their whole job either happens or begins. Several people share a home.
That is correct and it is the point: a home is a question, not a title.

| Person | Their one question | Home | Story |
|---|---|---|---|
| General manager, executive viewer | Is this week on plan and what needs me | Today | JS-1 |
| Planner | What should next week look like | Plan | JS-2 |
| Scheduler | Where does this break go | Days | JS-3 |
| Programming representative | When may a break not be placed | Rules | JS-4 |
| Account manager | Who is the client and what did we promise | Clients | JS-5 |
| Campaign manager | Is what we promised being delivered | Clients, pacing view | JS-6 |
| Traffic operator | Is what airs tonight correct | Days, break board | JS-7, JS-8 |
| Analyst | What did we earn, gross and net | Money | JS-9 |
| Revenue and yield owner | What is a second of airtime worth | Money | JS-13 |
| Compliance owner | Are we inside the licence | Rules, compliance view | JS-14 |
| Data steward | Is the engine reading true inputs | Sources | JS-12 |
| Account administrator | Who is in the system and what may they do | Account settings | JS-15 |
| Model steward (company) | Is the model fit to ship | Model console | JS-16 |
| Deployment owner | Is it up, enforced and secret-safe | Outside the product, plus one honest banner | JS-17 |
| Channel-affiliated account | Not a person: an identity class | The wall in section 4 | JS-18 |
| Kai | Not a person: a delegated actor | Docked everywhere | JS-10 |

Two consequences.

**Job is a new dimension, orthogonal to role.** Today `role` answers "what may
this account write" and nothing answers "what is this person's job". A traffic
operator and a planner are both `operator`. The account gains a `job` field with
a safe default, which decides the landing surface and the order of the sidebar.
It does not decide permission. Permission stays with role and affiliation, so a
misconfigured job costs a person a good first screen, never their access or
their safety. `auth_store.normalize_affiliation` is the pattern to copy: a field
that defaults safely for every existing record.

**The unit of adaptation is the landing surface, not the menu.** Hiding
navigation from people is how products become mysterious. Every operator sees
all seven workspaces. What changes with `job` is where they land, what their
sidebar puts first, and what their Today screen leads with.

---

## 3. The information architecture

### 3.1 The spine

The single largest structural fact discovery found is that the product has no
object below the programme segment, and no object joining the plan to the money.
Everything else follows from that.

Meridian has two spines and they meet at one object.

```
  the plan spine        WEEK  ->  DAY  ->  BREAK  ->  SPOT
                        (plan)  (board)   (pod)     (one ad)
                                                      ^
  the commercial spine  AGENCY -> ADVERTISER -> CAMPAIGN -> FLIGHT
```

The spot is the join. Today the two are, in the measured words of the API
investigator, two disconnected universes whose only bridge is a shared channel
name and a shared date. The weekly universe carries all the money on the
Overview and knows nothing about advertisers; the daily universe carries real
advertisers, real agencies and real rebates over 175 spots and knows nothing
about the plan.

Every person enters the spine at their own depth. The planner works the week.
The scheduler works a day. The traffic operator works a break. The analyst walks
the commercial spine down to the same spots from the other end. That is why the
navigation collapses: seventeen entries were seventeen guesses at where somebody
might want to be, when there are only four depths and one join.

### 3.2 The navigation, operator side

Seven workspaces. Not seventeen.

```
  Today       is this week on plan, what is broken, what needs a decision
  Plan        the week: objective, run, compare, publish
  Days        a broadcast day: the board, a break, the break's contents
  Clients     agency, advertiser, campaign, flight, pacing
  Money       rate card, gross to net, yield, what an advertiser delivered
  Rules       restrictions, guardrails, compliance, frequency
  Sources     what the engine reads, and whether it is current
```

Plus three things that are not destinations and must stop pretending to be:

- **Kai**, docked on the right of every surface, opened with `Cmd J`. It is
  already a dock; today it also occupies navigation slot 15 whose hash resolves
  to Overview, which is a lie in the URL bar.
- **History**, one attributed timeline of every change by anyone including Kai
  and including runs, reachable from the account menu and from every object's
  own header. Today this is three separate things: `#Versions`, the Settings
  activity log, and an in-memory bell feed.
- **Account settings**: channel, profile, language, accounts. Reachable from the
  account menu. Not a workspace, because nothing a person does daily lives
  there.

And one global control the product does not have: **`Cmd K`**, a grouped command
palette that prints each action's own shortcut on its row. Zero hits today for
command palette, cmd k, hotkey or shortcut anywhere in the frontend.

### 3.3 What merges, what splits, what disappears, what is new

Every row is justified against a story. Nothing is moved for tidiness.

| Today | Verdict | Where it goes | Justification |
|---|---|---|---|
| Overview | Rebuild, rename | **Today** | JS-1. It answers two of its three questions today and has no target to answer the third against. |
| Optimizer | Merge | **Plan** | Measured duplicate: the same four tiles, the same `ComplianceLedger` with the same prop, the same frontier as Overview, and the same `PlanningCanvas` / `DaypartView` / `TimelineView` over the same `schedule.rows` as Schedule. Two of the first three navigation entries are largely one page. |
| Schedule | Split | grid, daypart, timeline to **Plan**; editor, inspector, gold to **Days** | JS-2 versus JS-3. The planner works the week as an aggregate; the scheduler works one day as a timeline. Today one page serves both and the editor is three screens deep. |
| Inventory | Merge | **Plan**, supply view | JS-2 needs sellable supply beside the objective. Its money column is a dash on all five dayparts because the loaded spots source has no revenue column, so the money half belongs to Money. |
| Break Library | Merge | **Days**, the day's break list | It is a segment list called a break list, its top rows are byte-identical to Overview's priority decisions, and 18 of its 79 adjacent pairs are out of the descending revenue order it claims to rank by. |
| Campaigns | Merge | **Clients** | JS-5, JS-6. A campaign is a child of an advertiser, not a peer page. The current page is a historical rollup with revenue as a dash on 50 of 50 rows and the advertiser blank on 50 of 50. |
| Forecasts | Merge | **Plan**, compare view | JS-2. Comparing is a step of planning, not a destination. Today the comparison is a separate page reached by a top-bar button that navigates instead of comparing. |
| Events calendar | Split | read overlay on **Plan** and **Days**; authoring to the **Model console** | JS-18. All three event writes are already company-gated at `events_api.py:378,399,426`. The read side is what an operator needs: which days are special. The page also currently carries the largest model-health dashboard in the product. |
| Reports | Dissolve | each export moves onto the object it exports | JS-9. A download-only page is a dead end by construction, and its five cards duplicate the source-file list that already appears in three other places. |
| Data | Split | uploads and source files to **Sources**; "Model and parameters" to the **Model console** | JS-12 versus JS-16. The third tab is a training dashboard on a run surface, with a "Needs attention" drift chip the operator cannot act on. |
| Advertisers | Merge | **Clients** | JS-5. |
| Agencies | Merge | **Clients** | JS-5. One entity family, one place. |
| Pricing | Merge | **Money** | JS-13. |
| Overrides | Merge | **Days**, as acts on a break | JS-3. A pin is something you do to a break, not a console you visit. Gold currently has five surfaces for one concept. |
| Kai assistant | Demote to dock | dock on every surface | It already is a dock. The navigation entry resolves to Overview. |
| Restore changes | Merge | **History** | JS-3 needs undo where the work is, not on a separate page that takes 27 to 40 s to load. |
| Settings | Split | guardrails, protected content, frequency, constraint builder to **Rules**; objective and pacing to **Plan**; channel, profile, locale, accounts to **account settings** | JS-4. A programming representative should not register an objection by scrolling past `risk_lambda` and a pace denominator floor. |

**New, and not a rearrangement of anything:**

| New thing | Why | Story |
|---|---|---|
| The break as a first-class object with contents | Nothing below the segment exists. This is the largest new capability in the spec. | JS-7 |
| The break board (the pod as a timeline) | The physical thing a traffic operator assembles. | JS-7, JS-8 |
| Media asset, keyed on House Number | The only media identifier in the data is loaded once and never read again. | JS-8 |
| Campaign and flight, with real CRUD | Zero campaign write endpoints exist among 56. | JS-5, JS-6 |
| Plan target | "On plan" has no referent. No goal, budget or target key exists anywhere in `/api/overview`. | JS-1 |
| Plan version, with publish | The plan is the one thing the version store does not version. | JS-2 |
| Restriction as an authored, expiring object | Constraint rows carry no author, no approver, no expiry. | JS-4 |
| Model version and release record | The only model identity is a timestamp on a file overwritten in place. | JS-16 |
| One history timeline | Three "what happened" surfaces today, none covering the plan. | JS-3, JS-10 |
| Command palette and keyboard control | None exists. | All |

### 3.4 The rule that removes every dead end

The brief's requirement is absolute and it is cheapest to state as one
invariant, checked by a critic on every surface:

> Every name, number, status and badge on any surface resolves to the thing it
> refers to, in at most one click, and the thing it resolves to is a real
> object with its own address.

Three consequences that are not obvious.

- **A figure without a basis does not render.** Discovery found the word
  "revenue" on five surfaces at five bases, with one measurable contradiction of
  ₪686,475 between two figures on one page for one channel and one week. The
  rule is Stripe's: the basis is attached to the figure, not to a tooltip, and
  where the basis cannot be stated the figure is withheld with a named
  alternative route.
- **An empty field is an action.** Linear renders `Set estimate` and
  `Add to project` where a value is unset. Meridian's honest empty states are
  already its strongest asset, measured in at least five places, and they
  currently end in prose. Each becomes a control.
- **Prose that names a capability links to it.** Six of the eight cross-surface
  references found in source are prose with no link.

---

## 4. Training versus runs, made concrete

This is a first-class deliverable, not a naming exercise. Today, of the 113
operations the live app publishes, zero are training, and yet the model's
internals render on four operator pages while the same two words name both
activities 159 and 124 times.

### 4.1 The definitions

**Training is ours.** It decides what the model may believe: it fits
coefficients, runs the held-out gates, decides whether a factor is real, and
writes a model artifact. Its output is a **model version**. It is rare, it is
research, and it is company staff only.

**Runs are theirs.** A run computes or publishes a plan, a forecast, a scenario
or an export from the current model plus the current configuration. Its output
is a **plan version**. It is constant, it is operational, and it must feel
routine and safe.

Two further classes exist and must not be confused with either. **Configuration**
changes a stored input that a run reads and changes nothing on screen until a run
happens. **Reading** returns state.

### 4.2 Which surfaces belong to which side

| Side | Surfaces | Who sees them |
|---|---|---|
| Runs | Today, Plan, Days, Clients, Money, Rules, Sources, History, account settings, Kai | Any authenticated account, subject to role |
| Training | Model console: Gates, Coverage, Drift, Releases; plus Events authoring | `affiliation = company` only, read and write |

The Model console is a different shell, not a section of the operator sidebar.
It carries its own chrome and a permanent marker that says this is the company
side. A company account can switch between the two contexts, because company
staff support operators and need to see what they see. A channel account has no
switcher, no route, and no rendered trace that the other side exists.

### 4.3 What each side's dashboard answers

**The run dashboard answers six questions, in this order.** Every input already
exists in the payloads; most is not on screen, or sits next to something that
contradicts it.

1. **Is today's plan current, and if not, what made it old?** Today one amber
   banner fuses two different events: the live instance returns
   `{"status": "stale", "changed": ["settings", "coefficients"]}`, which means
   somebody changed a setting and somebody retrained the model, and the operator
   sees one sentence, one verb and one button. It becomes two states. "Your
   changes are not in the plan yet" is self-service with one button. "A newer
   model version exists" is informational, names the version and when it landed,
   and offers the same button with a different sentence.
2. **What will change if I run it?** The math exists:
   `/api/constraints/effect` and `/api/overrides/effect` both run the commit
   path's own optimizer twice and diff the result. Nothing joins them to the run
   button.
3. **What did the last run produce, and how does it compare to the one
   before?** `output/run_log.jsonl` holds 489 records carrying run id, engine
   version, input checksums, guardrails, assumptions and a summary. No endpoint
   serves it and no screen shows it.
4. **What are these numbers the projection of?** Basis on every figure.
5. **Am I compliant, and where am I at risk?** The best-served question today.
6. **What is booked, what is unsold, what is a make-good exposure?**

The run dashboard contains **no gate verdicts, no held-out deltas, no tau
squared, no pooling notes, no drift tables and no per-cell coefficients**. It
needs exactly one model fact: which model version this plan was computed with,
as a date and a name, and who to ask about it.

**The model dashboard answers six questions.** All six have their ground truth
already written into the artifacts on disk and none of it has a home.

1. **What did each gate decide and why.** The reasons are already stored in full
   sentences, for example "p=0.2034 not < 0.01; multiplier left at 1.0 (off)"
   and "series RMSE (fold mean 0.26239) does not beat genre RMSE (fold mean
   0.24200) by the required 2% margin". Three off-states must be visually
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
4. **What would a train change.** `models/candidates/` holds five alternative
   artifacts and a script computes the revenue movement if one were adopted.
   Nothing surfaces it. This becomes a first-class view: per candidate, the gate
   deltas, the coefficient deltas, and the money the adopted plan would move.
5. **What is blocked on data we do not have.** A register, not a prose reason.
   Per blocked factor: the condition that would unblock it and roughly when.
6. **Provenance.** Fingerprints, seeds and method are already recorded. Missing:
   who ran it, with which gate-override flags. Six such flags exist with
   environment-variable twins, so a forced gate is today indistinguishable from
   a self-activated one after the fact.

### 4.4 The permission rule

One sentence, and it is different from today's.

> **Affiliation decides which side of the line you can see. Role decides what
> you can change on your side.**

Concretely:

- Every `/api/model/*` and every training route requires `affiliation = company`,
  on **read as well as write**. Today the wall is three unconditional calls plus
  two conditional ones, which I verified are the only five call sites of
  `require_company_editor` in the codebase, and it covers writes only.
- `audience_model_activation` leaves the free-form settings document and becomes
  a company-only model-activation control. Today it decides where every
  forward-dated rating comes from, has no control anywhere in the dashboard, and
  is settable by any channel operator with one `PUT /api/settings`, because that
  endpoint takes the whole settings model and has no affiliation guard.
- The regulatory guardrails leave the same document and become their own store
  with an effective date and a change record, so JS-14's second half becomes
  possible.
- Every surface reads its own session affiliation and renders accordingly. Today
  the dashboard never reads it: `GET /api/auth/me` returns `affiliation` and the
  only frontend uses of the word are the accounts dialog and a label helper.
- **A refusal is legible before the click, never a 403 after it.** Today
  `/api/events` returns `can_edit` and the calendar hides its controls, while
  `/api/pricing` returns no such field and its identically walled toggle renders
  enabled and fails after the click. Every walled control carries a `can_edit`
  from its own endpoint.

### 4.5 The canonical vocabulary

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
reassigning: relabelling one act while the other keeps a word the operator has
learned would be worse than the status quo. **הפצה** is chosen for publish
because **פרסום** collides with advertising in an advertising product, and
**אישור** collides with the existing Approve on recommendation cards.

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

### 4.6 The new object vocabulary, Hebrew and English

Adopted alongside the canonical terms already frozen in the brief (ברייק,
ברייקים, נעיצה, ברייקי זהב, רצועת שידור, הכנסה צפויה, עלות שימור, מפעיל, and
never משתמש).

| Object | English | Hebrew | Source of the word |
|---|---|---|---|
| The week's plan | weekly plan | תוכנית שבועית | existing |
| A published plan | plan version | גרסת תוכנית | new, mirrors model version |
| A broadcast day | broadcast day | יום שידור | existing |
| A break | break | ברייק | frozen |
| A break's contents | break contents | תוכן הברייק | new. There is deliberately no second word for "pod": the break is the pod. |
| One ad occurrence | spot | תשדיר | the trade's own word, already in the daily file (`סוג תשדיר`, `אורך תשדיר`) |
| The media file | House Number | House Number | the trade's own identifier, already an English column in a Hebrew file |
| A programming restriction | restriction | הגבלה | the representative's word, not the engine's |
| A plan target | target | יעד | new |

---

## 5. The entity model, and the migration to it

### 5.1 What is wrong today, measured

| Entity | Today | Consequence |
|---|---|---|
| Programme segment | `segment_id = f"{day}\|{channel}\|{index:03d}"`, an `enumerate` position (`kairos/data/transform.py:255`) | Insert one programme in the EPG and every later id on that channel-day shifts |
| Break | Not an entity. A `num_breaks` integer on a segment | "Move the 20:05 break" is inexpressible; the plan can only say a segment now has three breaks instead of two |
| Spot | Not an entity. A free-text version name on a daily-file row | No ad is addressable |
| Media asset | Does not exist. `house_number` renamed at `loaders.py:90`, never read again | Nothing can be verified |
| Advertiser | 45 ids with **zero** intersection with either name space; the two name spaces match each other 41 of 41 (I verified this myself) | 0 of 45 advertisers have a name or a revenue figure |
| Campaign | Three unjoined things: 478 free-text strings, a pacing key with 0 rows, and a per-spot name | No campaign is an object |
| Plan | A CSV plus a sidecar. Not among the nine versioned logical files | No plan history, no undo, no publish |
| Model | A file overwritten in place | No model version identity, no drift series |
| Target | Does not exist | JS-1 is unanswerable |

### 5.2 The model the work implies

```
  Airing            key: hash(date, channel, start_clock, title)
    |                    the anchor triple manual_overrides.csv already carries
    |
    +-- Break         key: airing_id + ordinal
    |     |           state: planned -> assembled -> verified -> locked -> aired
    |     |           carries: planned start, length, position, gold, value, cost
    |     |
    |     +-- Spot    key: break_id + position
    |           |     joins the plan spine to the commercial spine
    |           |
    |           +-- MediaAsset   key: House Number
    |                            technical facts, owner-supplied
    |
    +-- Restriction   authored, scoped, expiring; compiles to the frozen predicate

  Agency -> Advertiser -> Campaign -> Flight -> (books) Spot

  PlanVersion    a run's output, named, dated, publishable, diffable, restorable
  ModelVersion   a training's output, named, dated, with its gates and its release record
  Target         the number a week is measured against
```

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

Nothing else is worth building until these three land, in this order.

**1. Advertiser identity.** Add `name` and `aliases` to `advertiser_rules.csv`,
exactly the columns `agencies.csv` already has, and resolve the daily file's
Hebrew names to ids the way agencies already resolve. This one change unblocks
JS-5, JS-6, JS-9 and half of Money. Evidence that it works: the same 175 rows
resolve the agency to an id and produce real money (gross ₪699,450, net
₪669,978, 119 spots) while failing to resolve the advertiser at all.

**2. Airing identity.** Move every operator-facing reference off the ordinal and
onto the anchor hash. Without this, a break entity is built on sand: one EPG
insertion reassigns every pod on the channel-day.

**3. The break entity.** A stable break identity below the airing, with an
ordered list of spots. Everything the traffic operator needs, and every honest
per-break money figure, rests on it.

Then, in any order: campaign and flight CRUD, the media asset, the plan version,
the model version, the target.

### 5.4 Migration, and what is not destroyed

- The five rule grammars collapse to two: a **placement rule** (today's
  constraints plus overrides, which already share a preview endpoint returning
  the same shape) and a **commercial rule** (today's advertiser conditions plus
  agency conditions plus frequency rules, which already share a scope grammar).
  Four of the five stores are empty today and the fifth has one row, so this is
  a consolidation of schemas, not of data.
- `data/kairos_constraints.csv` does not exist on disk while four modules
  reference it. The new store is created explicitly rather than implicitly.
- The version store's 187 of 200 entries that point at pytest temporary paths
  are marked unrestorable rather than deleted, and the store gains an isolation
  guard so tests cannot write into the operator's history again.
- `data/enriched/` (19.7 MB, read by nothing) and `kairos/optimize/agreements.py`
  (zero callers) are proven dead before removal, and the proof is written down.
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
| **Plan target** (JS-1) | A small target store, a comparison in the operating view, a three-state verdict with a published threshold | The weekly or monthly revenue or GRP target per channel, and who sets it |
| **Publish** (JS-2) | Plan versions on top of the existing version store, a published state, an author, a diff against the previous version | What publishing means here and who may do it |
| **Break entity and break board** (JS-7) | New store, new router, an ordered spot list, a state machine, an engine seam at the daily per-spot path where ads and breaks already meet | The pod boundary rule. Grouping the daily file by break start gives groups of 1 to 38 ads spanning up to fourteen minutes, so the boundary is not derivable |
| **Media verification** (JS-8) | Media asset store keyed on House Number, a technical probe, a verdict printed on the ad | Media files or a technical metadata feed per House Number, plus the approval state vocabulary. `סטטוס` is empty in 175 of 175 rows so its values are unknown. No media tooling exists in the repository |
| **Campaign and flight CRUD** (JS-5) | A router modelled exactly on `agencies.py`, with deactivate rather than delete | Real flights with dates and goals |
| **Live pacing and make-good** (JS-6) | The pacing math is already implemented and honest; it needs a delivered figure that updates and a forecast state | A delivery or as-run feed, and a current week. Nothing in the data represents now |
| **Restriction language and live preview** (JS-4) | A translation layer above the frozen predicate contract, an offset measured from programme end, an occurrence concept, and `/api/constraints/effect` wired with a real latency budget | Which airings are finales. Zero of 418 titles carry any finale marker |
| **Model console and a train trigger** (JS-16) | Company-gated routes wrapping the two existing scripts, a model version identity, a release record, a drift series across versions | Nothing |
| **Money drill** (JS-9, JS-13) | Per-spot attribution joined to a named advertiser, and every amount resolving to its rows at more than one level | Nothing beyond advertiser identity |
| **Action-level undo** (JS-3) | Per-action inverses on the day board, on top of the version store | Nothing |
| **One history timeline** (JS-3, JS-10) | Merge three surfaces, add the run log and the model releases, previewable before restore, separately permissioned | Nothing |
| **Command palette and keyboard** (all) | New | Nothing |
| **Speed** (all) | See section 8, piece W0-B | Nothing |

Speed is a capability, not a detail, and it is worth stating as a target rather
than an aspiration. Measured today on the running instance: `/api/versions` 27
to 40 s, `/api/advertisers` 12 to 24 s, `/api/agencies` 15 to 25 s,
`/api/uploads/status` 9.6 to 11.3 s, `POST /api/scenario-compare` 20 to 25 s,
`POST /api/assistant/ask` 78 s, and eight parallel requests, which is what one
page load issues, taking 16.98 s wall against a single uvicorn worker. Every
page load serializes behind its slowest call. The target is: **no read on a
first paint slower than 500 ms at p95, no page's first meaningful answer later
than 1.5 s, and no interactive action without either a result or a cancel inside
5 s.**

---

## 7. Kai in the new architecture

Kai's current design is better than its delivery. The structure is right: 31
read tools, 8 propose tools, 0 write tools, which I confirmed in process, plus
a system prompt whose fourth rule states it never changes anything itself. The
proof it holds: an action request in Hebrew ("raise the retention floor to 75
percent") changed nothing, and the setting was still 0.72 afterwards. What fails
is that the answer took 78 s at the backend and never arrived in the browser at
all, still "preparing" at 499 s.

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

- **Grounding for the retention model.** There are twelve keyword-triggered
  grounding sections and none is triggered by מקדמים, coefficients, אימון,
  training or drift, so an operator asking in Hebrew why the plan moved gets no
  coefficient context, even though a read tool exists that would answer it.
- **The new object vocabulary.** Kai speaks week, day, break, spot, restriction,
  target, plan version and model version, the same words the interface uses.
- **A voice per job.** The system prompt addresses "the operator" exclusively.
  It should address the person in front of it.

### 7.2 What Kai must never do

1. **Never start a training run**, propose one, or offer one as an option. Zero
   of its 39 tools touch a model artifact today and that must stay true.
2. **Never disclose model internals to a channel account.** Gate verdicts,
   held-out deltas, drift, coverage and coefficients follow the same affiliation
   wall as the console. Read tools are open to every authenticated account
   today; three of them return training content.
3. **Never name a competitor channel or carry its data into context.**
4. **Never write without an approval.** Propose, show the diff, wait.
5. **Never answer a money question without its basis.** It already does this
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

The pieces below are designed to be **genuinely independent**: each owns a
disjoint set of files and a disjoint set of endpoints, so builders can run in
parallel without collision, and a critic can judge one without the others being
finished.

There is one hard prerequisite for that independence. Today
`TVBreakDashboard.jsx` is 6,236 lines carrying the routing, the navigation and
twelve page components inline, and `dashboard_api.py` is 1,827 lines. If ten
builders edit those two files at once, nothing is independent. So wave zero
splits them, and the 450-line law is the mechanism that keeps them split.

### Wave 0: three pieces, mutually independent, everything else waits on them

| Piece | What it does | Files it owns | Bar | Reference |
|---|---|---|---|---|
| **W0-A Identity** | Advertiser name and aliases, name-to-id resolution on the daily path, airing anchor identity | `kairos_api/advertisers.py`, `kairos/optimize/advertiser_rules.py`, `kairos/data/transform.py`, a migration script, the advertiser store | 45 of 45 advertisers named. More than zero with attributed revenue. Every operator-facing reference survives a simulated EPG insertion. Engine figures byte-identical. | Stripe: a figure carries its basis |
| **W0-B Speed** | The eight worst endpoints, a read cache, the parallel-load serialization | the read paths in `kairos_api/`, a cache module | Every read on a first paint under 500 ms at p95. Eight parallel requests under 2 s wall. No endpoint over 5 s anywhere. | Linear: the interface never makes you wait |
| **W0-C Shell** | The seven-workspace IA, the route map, the monolith split, design tokens, `Cmd K`, keyboard, RTL, Sunday-first week | `tv-break-dashboard/src/**` | 17 entries to 7. No file over 450 lines. A first-time person names their own workspace in 5 s. Density measured against Linear's 17 rows at 44 px carrying 9 facts in an 823 px viewport. Monday-first `dayKeys` at line 585 gone. | Linear |

W0-C ships against today's data and today's endpoints. It is judged on
structure, speed and legibility, not on new capability. That is what makes it
safe to run in parallel with W0-A and W0-B.

### Wave 1: ten pieces, parallel, each owns one workspace

| Piece | Bar (the story it must pass) | Reference | Depends on |
|---|---|---|---|
| **P1 Today** | JS-1: 5 s, 0 clicks, three answers, each opening its rows. Includes the target entity and the two-state freshness split from section 4.3. | Google Ads pacing states, Stripe drill | W0-B, W0-C |
| **P2 Plan** | JS-2: 180 s to a published plan version, compare in 5 s, net of retention exposed, every figure with its basis | Stripe basis, Linear speed | W0-B, W0-C |
| **P3 Days and the break entity** | JS-3: 20 s move, money within 500 ms of the drop, 0 dialogs, undo in 1 keystroke, gold in place | Premiere timeline, Figma selection and undo | W0-A, W0-C |
| **P4 Clients** | JS-5: 120 s for agency, advertiser and campaign in one flow, 0 duplicates. JS-11: unaided in 300 s | Google Ads hierarchy, Linear record page | W0-A, W0-C |
| **P5 Money** | JS-9: 30 s, 0 exports, every amount opening its rows at more than one level. JS-13: 45 s with the delta before the save | Stripe reports and audit numbers | W0-A, W0-C |
| **P6 Rules** | JS-4: 30 s, 0 engine words, cost before saving, preview in 3 s. JS-14: 15 s plus 30 s | Google Ads status with the remedy on the row | W0-C |
| **P7 Sources** | JS-12: 60 s to a green in-use state with shadowing explained in place | Frame.io per-file status, Stripe availability | W0-B, W0-C |
| **P8 Model console (company)** | JS-16: 120 s to a recorded ship or no-ship decision, three off-states distinguished, train started from the console | Linear cycle rail, Stripe availability | none, entirely separate file tree |
| **P9 History** | Every change by anyone including Kai and including runs in one attributed timeline, previewable before restore, separately permissioned | Figma version history, Cursor checkpoints, Google Ads change history | W0-C |
| **P10 Kai** | JS-10: 45 s, first token in 2 s, diff on the object, previewable restore point | Linear Agent, Cursor, Figma agent | W0-C |

### Wave 2: three pieces that depend on wave 1

| Piece | Bar | Reference | Depends on |
|---|---|---|---|
| **P11 The break board** | JS-7: 90 s for the real seven-ad break at 22:53:49, 212 s across 7 spots, sum always visible, reorder by dragging, the 36-versus-35 mismatch impossible to miss | Premiere timeline, Frame.io asset grid | P3 |
| **P12 Media verification** | JS-8: verdict printed on the ad, a failing ad blocks the lock. Owner-blocked for the technical half: ships the arithmetic and an honest empty state naming the missing feed | Frame.io | P11, owner data |
| **P13 Pacing board** | JS-6: 60 s, three-state verdict with a published threshold, remedy on the same row | Google Ads budget pacing | P4, owner flights |

### Then

**P14 The boundary sweep.** JS-18 run as an independent critic against the
assembled product: a channel-affiliated account, every door, every ambiguous
button, every competitor name.

**P15 The integration critic.** One fresh critic over the whole product:
vocabulary and interaction consistency across all surfaces, correctness end to
end, and whether the assembled thing serves the jobs it was decomposed from.

### Why this order

Identity, speed and the shell are first because every other piece is either
false, slow or uncoordinated without them, and because they are the three
pieces that touch disjoint layers so they cost nothing to parallelise. The break
entity sits inside P3 rather than in its own piece because a break with no board
to live on cannot be judged. Media verification is last of the buildable set
because it is the only piece whose bar cannot be met without data the owner has
not supplied, and putting it early would burn a wave on an honest empty state.

### The contract that keeps pieces independent

Before a wave starts, each piece publishes and freezes: the endpoints it owns,
the payload shapes it emits, and the files it will touch. A builder that needs
something outside its own list raises it rather than reaching for it. The shared
surface is exactly two things: the design tokens from W0-C, and the vocabulary
in section 4.5 and 4.6.

---

## 9. What this spec deliberately does not do

Each with its reason, so that a critic does not grade an absence as a miss.

1. **No media probe or transcode pipeline.** There is no ffmpeg, ffprobe or
   MediaInfo dependency and no media file of any kind on disk. Nothing here can
   read a video. P12 builds the arithmetic verification, which is real today,
   and states the technical gap in exact terms.
2. **No invented "on plan".** JS-1's third answer needs a target the owner has
   not supplied. Until then it is an honest empty state with a path forward, not
   a number derived from the plan itself, which would be circular.
3. **No integrations.** Zero exist: no webhook, SFTP, BXF, AdID or inbound API
   anywhere, and the only outbound clients are Anthropic and Gemini. The owner
   has not named the system of record for campaigns or the delivery format for a
   locked break, and both are needed before an integration can be honest.
4. **No change to the frozen predicate contract.** `docs/constraint-predicate-contract.md`
   is frozen. The restriction language in P6 is a translation layer above it.
   Two genuine engine extensions are additive: an offset measured from programme
   end, and an occurrence concept.
5. **No activation of the wired-off pricing layers.** Position carries a 1.30
   first-position multiplier and ad type carries a 0.00 promo multiplier.
   Activating either moves real revenue. They are owner-gated and stay that way.
6. **No renamed API paths.** `POST /api/recompute-schedule` and its siblings keep
   their addresses. Vocabulary changes are labels and messages.
7. **No mobile or responsive redesign.** Every job story is a desk job.
8. **No multi-tenant or multi-channel operation.** The operator owns exactly one
   channel, read from settings.
9. **No general query or BI surface.** JS-9 is served by drilling from a figure
   to its rows. An arbitrary query builder is a different product.
10. **No deletion of the version store's polluted history.** 187 of 200 entries
    point at pytest temporary paths. They are marked unrestorable and the store
    gains an isolation guard. Deleting an operator's history to tidy a defect is
    the wrong trade.
11. **No engine number moves.** Revenue, retention, pricing and rating figures
    stay byte-identical unless a change is proven a genuine fix, declared, and
    measured.
12. **No new persona surfaces for the deployment owner.** JS-17 is served by an
    honest banner and a good bootstrap, not by a page. The work is operational.

---

## 10. Where the evidence conflicted, and what I trusted

Recorded because the lead needs to know which numbers are load-bearing.

| Conflict | What I trusted | Why |
|---|---|---|
| Endpoint count: the brief says 111, discovery says 113 | **113 operations over 90 paths, 56 of them writes** | I counted the live `openapi.json` myself. Two investigators and my own count agree. |
| Assistant allowed settings fields: discovery says 20 | **19** | I enumerated `ALLOWED_SETTINGS_FIELDS` in process. `audience_model_activation` is not among them either way, so the conclusion is unaffected. |
| Test count: the brief says 1,438 passing and 4 skipped | **3,102 collected at this HEAD**, pass count unverified | I ran `pytest --collect-only`. I did not run the suite, so I make no claim about passes. The brief's figure appears to predate this HEAD, and the lead should re-baseline before using it as a gate. |
| `/api/overview` latency: 1.28 s, 5.84 s and 3.5 to 6.7 s reported by three investigators | **The range is real and the variance is the defect** | I measured 5.90 s just now under concurrent load. One investigator explicitly stated their load conditions and re-measured. A read that varies 5x is a UX problem in itself, so W0-B is graded on p95, not on a best case. |
| `run_log.jsonl`: 488 versus 489 records | **489 lines**, measured | Trivial, recorded for completeness. |
| Version store: 200 manifests versus 201 directory entries | **200 manifests** | The extra entry is a directory listing artifact. The load-bearing number is that 187 of them point at pytest paths. |
| The competitor lanes in the planning grid: one investigator measured them and explicitly declined to rule | **It is a law breach and it is fixed** | The law reads "No rival channel's name or data reaches an operator surface or the assistant's context." I fetched `/api/schedule` myself: the 200-row projection is 96 `קשת 12`, 73 `כאן 11`, 28 `עכשיו 14` and 3 of the operator's own channel. The model may continue to use competitor lineup internally (its gate is on, at +2.16 percent held out). The operator surface shows the operator's channel and, where the model uses a competing lineup, an unnamed aggregate. **This removes something visible today, so it is flagged for the owner rather than done silently.** |

### What still needs a human decision

Five things block a design decision and evidence cannot settle them.

1. **The plan target.** What number is a week measured against, who sets it, and
   at what grain. JS-1 is unanswerable without it.
2. **The pod boundary.** The daily file's break-start grouping produces groups of
   1 to 38 ads spanning up to fourteen minutes, so a break is not derivable from
   it. Either an explicit break identifier per ad, or the rule that splits a
   group.
3. **The media feed.** Per House Number: exact duration with frames, frame rate,
   container and codec, aspect ratios, audio presence and layout, loudness, plus
   the approval state vocabulary.
4. **A current week.** Nothing in the data represents now. `effective_date` is
   2026-06-14, the plan is November 2024, the one daily file is April 2025.
   Every "on air" and "today" story stands on this.
5. **Who publishes, and what publishing means.** Plus the related question of who
   owns the regulatory guardrails, which are currently ordinary settings
   editable with the same permission as a slider.
