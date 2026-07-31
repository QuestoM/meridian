# Baseline measurement of today's Meridian against the ten seed job stories

Measured 2026-07-31 against the running app at http://127.0.0.1:8010 (uvicorn
`kairos_api.server:app`, single worker, authentication disabled, repo working
tree at `main` 342a2896). These numbers are the floor every future critic must
beat.

## How this was measured, and what the numbers mean

Two browsers were involved and the difference matters for trust.

- The shared Chrome tab group used by the other discovery agents was abandoned
  after it twice navigated tabs out from under a measurement (tab 288064840 was
  moved to `#Break Library` and tab 288064852 to `#Forecasts` between calls).
- Every number below comes from a private headless Chrome 150.0.7871.188 that
  this investigation launched and controlled alone, driven over the DevTools
  protocol with real mouse and key events, viewport 1600x1000, device scale 2,
  a virgin profile. Driver and scripts:
  `/private/tmp/claude-501/-Users-home-Code-questo-meridian/82583077-5cf3-47de-a87d-8f926c5ad678/scratchpad/cdp.py`
  and the `js*.py` scripts beside it.

Three different clocks are reported and never mixed.

1. **App time.** Wall clock the app itself consumes: time from navigation until
   the answer is on screen, and endpoint round trips. Taken in-page from the
   Performance API and from `curl`. No human can be faster than this, so it is
   the honest stopwatch floor.
2. **Interaction cost.** Clicks, keystrokes, distinct screens, dead ends and
   guesses, counted exactly while driving.
3. **Completion verdict.** Whether the job can be finished at all.

Two rules were held. Nothing was mutated: no schedule was published, no
constraint, agency, advertiser or setting was saved, and no assistant proposal
was applied. Where a job ends at a mutating button, the report says so and
stops there. And every claim below is a measurement or a file and line, never
an impression.

### Load contamination, stated up front

The server was concurrently used by other discovery agents during the session.
Evidence: `POST /api/pricing/price-slot` at 16:31:10Z and `PUT /api/settings`
(102,173.6 ms) at 16:52:42Z in `/api/activity-log`, neither issued by this
investigation, and `data/kairos_settings.json` was rewritten at 19:52:42 local
adding `"audience_model_activation": false`, which this investigation did not
do. The uvicorn worker sat at 92 percent CPU throughout. Latency figures are
therefore "under concurrent load", which is also what a real multi-user product
must survive, and several of them were re-measured three or more times to show
they are stable rather than transient.

## The table

App time is the measured floor. Clicks and keystrokes exclude reading time.
"Screens" counts distinct pages traversed. "Guesses" counts moments where the
interface did not say which control does the job.

| # | Person and job | Done | App time floor (measured) | Clicks | Keys | Screens | Dead ends | Guesses / code reads |
|---|---|---|---|---|---|---|---|---|
| 1 | GM: on plan, anything broken, decision today | Partial, 2 of 3 | 3.59 s cold to broken and decisions, 5.17 s to the net figure; warm 2.85 to 3.26 s and 3.78 to 4.27 s | 0 | 0 | 1 | 1 (no plan target exists) | 0 |
| 2 | Planner: objective, optimize, compare on net, publish | No | 12.85 s optimize plus 20.35 to 25.62 s compare, about 45 to 55 s of pure wait before publish | 7 min | 0 | 3 | 2 | 2 |
| 3 | Scheduler: move a break, see cost move, pin gold, respect a constraint | Partial | 5.28 to 8.07 s page, 3.36 to 5.49 s editor, 2.43 s drag | 3 | 0 | 1 (+1 for gold) | 2 | 1 |
| 4 | Programming rep: register a restriction, see what moves and what it costs, before saving | No | preview endpoint 16.5 s for one channel-day, 55.6 s unscoped, and it is not wired to the UI | 6+ | ~20 | 1 (Settings) | 2 | 3 |
| 5 | Account manager: agency, advertiser under it, campaign with flights, rebate, Saturday discount | No | 22.3 to 25.0 s agencies, 23.9 to 40.1 s advertisers | 2 reached | 0 | 3 | 3 | 2 |
| 6 | Campaign manager: every campaign, pacing, under-delivery, action | No | 6.61 to 8.26 s | 1 | 0 | 1 | 1 | 0 |
| 7 | Traffic operator: assemble a pod from ads, verify each, reorder, lock | No | 6.01 s | 2 | 0 | 1 | 1 | 1 |
| 8 | Analyst: top advertiser last month, gross and net of rebates, no export | No | 22.3 to 25.0 s | 2 | 0 | 2 | 2 | 1 |
| 9 | Anyone: ask Kai in Hebrew, act with preview and undo | No in the browser | backend answers in 77.9 s; the dock still said "preparing" 499 s after Enter | 3 | 29 | 1 | 1 | 0 |
| 10 | First day: complete story 5 unaided | No | same as 5 | n/a | n/a | 3+ | 4 | n/a |

Targets in the brief, against the floor: story 1 asks for five seconds and gets
one of its three answers never; story 2 asks for three minutes and spends about
a minute of it waiting on two endpoints; story 4 asks for thirty seconds and its
preview alone costs 16.5 seconds and is not connected; story 7 asks for ninety
seconds for a seven-ad pod and no ad entity exists.

## Story by story

### 1. General manager: five seconds, zero clicks

Decisive screen: `/Users/home/Code/questo/meridian/docs/ux-gauntlet/discovery/baseline/js1-02-at-5s.png`
(also `js1-01-cold-settled.png`).

Measured on a virgin profile, first load ever, then three warm reloads.

| Signal | Cold | Warm run 1 | Warm run 2 | Warm run 3 |
|---|---|---|---|---|
| Any page text | 2461 ms | 2283 ms | 2065 ms | 2086 ms |
| Revenue KPI | 3588 ms | 3264 ms | 2845 ms | 2933 ms |
| "Saved schedule is out of date" | 3588 ms | 3264 ms | 2845 ms | 2933 ms |
| Priority decisions list | 3588 ms | 3264 ms | 2845 ms | 2933 ms |
| Net after retention cost | 5167 ms | 4164 ms | 3784 ms | 4270 ms |
| "Is this week on plan" | never | never | never | never |

`domContentLoadedEventEnd` is 60 ms, so the 2.1 to 2.5 seconds of blank screen
is application boot, not the network. Transferred bytes on cold load: 689,419.

What is answered: "is anything broken" (an amber banner, "Saved schedule is out
of date", naming the settings and coefficient change and the compute time
7/28/2026 11:38:38 AM) and "what needs a decision today" (a "Priority decisions,
5 actions" list, top row "Review the 21:24 Reality breaks on Sat 2024-11-23,
₪393.3K, 80.5%").

What is not answered: **whether the week is on plan**. There is no target
anywhere. `/api/overview` carries 142 keys and not one matches
`goal|target|variance|pace|on_plan|budget|delta`. A 30-second in-page poll for
any "on plan" or "gap versus plan" phrasing never fired. The GM sees
"Weekly projected revenue ₪10.12M" with no number to compare it to.

Honesty defect on the boot path, measured twice: the status chip renders as
`api-state offline`, "אין חיבור ל־API" (no API connection), and only flips to
"API חי" 1278 ms and 1457 ms later. For about one and a half seconds every
session begins by telling the operator the system is offline when it is not.

Two smaller notes. On a virgin profile the app comes up in **English**
(`document.documentElement.dir` is `ltr`), not Hebrew. And the sidebar carries
**17** navigation entries, measured, matching the brief's count.

### 2. Planner: objective, optimizer, two scenarios on net, publish

Decisive screen: `.../baseline/js2-06-scenario-ab-done.png`
(supporting: `js2-01-optimizer.png`, `js2-02-scenario-options.png`,
`js2-03-compare.png`, `js2-07-after-run-optimization.png`).

The path a planner actually walks: Optimizer page, scenario select, Run
Optimization, Compare, Scenario A/B, Apply to weekly schedule. Seven clicks
minimum, three screens.

Measured latencies, all repeated:

- `POST /api/optimizer-plan`, what "Run Optimization" calls
  (TVBreakDashboard.jsx:1861-1868): **12.85 s** by curl; the activity log
  records the calls fired by the click at 9,454.3 ms and 9,877.9 ms.
- `POST /api/scenario-compare`, what the A/B "Compare" calls: **25,479.7 /
  24,866.4 / 25,618.3 / 20,346.0 ms** in the activity log, **25,455 ms**
  end to end in the browser from click to result, **22,639 ms** by curl.

Three findings, each of which alone fails the story.

1. **Net of retention cost is not available for comparison, and the app says
   so.** The A/B result panel prints: "Net after retention cost: Not exposed"
   and "Objective is the optimizer convex-blend score, not literal revenue minus
   retention cost. The API does not expose a net after retention cost figure."
   That is admirable honesty and a complete failure of the job story, which is
   defined on exactly that quantity.
2. **The two scenarios are indistinguishable on every operational number.**
   Scenario A weight 60 and Scenario B weight 85 both return plan revenue
   ₪1.41M, average retention 95 percent, 80 breaks, 9,600 ad seconds. The
   printed delta is Revenue ₪0, Retention 0pp, Breaks 0, Ad seconds 0. Only the
   blended score differs, 0.5 against 0.4. The scenario curve above it shows
   Retention guardrail, Balanced and Revenue priority all at ₪1.41M and 95
   percent.
3. **The comparison is not of next week.** The panel states "Both runs optimize
   one representative channel-day (רשת 13, 2024-11-11), not the weekly total."

Two dead ends on the path. The top-bar "Compare" button does not open a
comparison on the Optimizer page: it silently navigates to the Forecasts page,
where a second, different button also called "Compare" is the real A/B control.
And on first arrival the Forecasts page reads "Daily forecast, 30 days, No
forecast rows were found"; rows appear only after the A/B run.

Publish was not pressed. `Apply to weekly schedule` calls
`handleRecomputeSchedule`, `POST /api/jobs/recompute` with polling
(TVBreakDashboard.jsx:1946-1985), which writes the saved plan. Stopped by
policy, control present and enabled.

### 3. Scheduler: move a break, watch the numbers move, pin gold, respect a constraint

Decisive screens: `.../baseline/js3-03-editor-top.png`,
`js3-04-mid-drag.png`, `js3-05-after-drag.png`.

The Editor tab is a real timeline with per-break chips carrying clock time,
offset and duration ("02:12:00 / 12 m 0 s / 120s"), a snap control at 30s and
60s, a pin scope, and a zoom. Direct manipulation works: a real pointer drag
moved a chip from **02:12:00 to 02:36:00** in 2,430 ms, and afterwards two new
controls appeared, "Save as pin" and "Discard change".

What does not happen is the part the job story is about.

- **No money and no retention move.** Every `₪` figure on the page was byte
  identical before and after the drag (`money_changed_after_drag: false`). This
  is structural, not a timing artefact: `ScheduleEditor.jsx`,
  `ScheduleEditorBreak.jsx`, `ScheduleEditorRow.jsx` and
  `ScheduleEditorToolbar.jsx` contain **zero** occurrences of "revenue" or
  "retention", case insensitive.
- **No undo.** `grep -rniE "\bundo\b" tv-break-dashboard/src/*.jsx` returns
  nothing. The only reversal is the separate "Restore changes" page. The brief
  asks for undo always available.
- **Gold cannot be pinned here.** The editor has no gold control; the page
  footer reads "No gold breaks in the current plan (none configured as gold in
  overrides)", so gold is a trip to another page.
- The instruction under the timeline is "Drag a break to set its offset, then
  save it as a pin", which is the engine's word, not a scheduler's.

`ScheduleEditor.jsx` is 466 lines, over the 450-line law.

### 4. Programming representative: a restriction in their own words, priced before saving

Decisive screens: `.../baseline/js4-01-constraint-builder.png`,
`js4-02-constraint-builder-scrolled.png`.

The constraint builder is rendered inside `SettingsPanel`
(TVBreakDashboard.jsx:5432 and 5797), that is, on the **Settings** page, below
`risk_lambda`, the optimizer balance slider, "Behind-pace strength",
"Over-delivery restraint" and "Pace denominator floor". It sits at character
4,240 of a 9,673-character page.

Its vocabulary, read from the rendered page and from
`ConstraintBuilder.jsx:556-660`: "When (filter conditions)", "Apply effect",
"Effect: Fix offset", "Offset (MM:SS)", "Min offset (MM:SS)", "Max offset
(MM:SS)", "Break count", "Min duration (s)", "Order index (optional)", "Save
constraint", "Recompute weekly schedule".

Three blockers.

1. **The restriction cannot be expressed.** Effects are `fix_offset`,
   `offset_window`, `pin_count`, `duration_range`, `gold`, `forbid`
   (`/api/constraints/options`). All offsets are measured forward from the
   programme start. "No breaks in the last eight minutes" has no representation:
   the representative would have to know the programme's own duration and enter
   `max offset = duration minus 08:00` by hand, and the builder never shows a
   duration.
2. **There is no preview.** `ConstraintBuilder.jsx` fetches exactly four things:
   `GET /api/constraints/options`, `GET /api/constraints`,
   `POST /api/constraints`, `DELETE /api/constraints/{id}`. It never calls
   `/api/constraints/effect`. The only next step after Save is "Recompute
   weekly schedule". Nothing is shown before saving.
3. **Even if it were wired, it would miss the target.** The preview endpoint
   runs the day optimizer twice (`kairos_api/constraints.py:333-400`). Measured:
   **16.55 s** for `channel=רשת 13&day=2024-11-23`, **55.60 s** with no channel,
   and **no response at all** without parameters (two `curl` processes hung for
   over thirty minutes before being killed, and a fresh call still had nothing
   after 15 s). A weekday name is rejected: `day=Sat` returns 404, "No segments
   found for the requested channel-day", so the caller must know the ISO date.

There are currently zero stored constraints (`/api/constraints` returns an empty
list), so nothing on this path has ever been exercised in this environment.

`ConstraintBuilder.jsx` is 690 lines, over the 450-line law.

### 5. Account manager: agency, advertiser, campaign with flights, rebate, Saturday discount

Decisive screens: `.../baseline/js5-03-agencies-loaded.png`,
`js5-04-add-advertiser.png`.

- **Agency: cannot be created.** The Agencies page offers Refresh, a search box
  and status filters, and nothing else. There is no POST to `/api/agencies`
  anywhere in the dashboard source, and `AgencyManager.jsx` and
  `AgencyDetailDrawer.jsx` contain no "Add", "New agency" or "create" control.
  The API supports `POST /api/agencies`; the product does not surface it.
  Existing agencies are rich (AGY_01 "OMD" carries `rebate_percent` 4.0,
  `commission_percent` 15.0, `payment_terms_days` 60, `credit_limit_ils`
  3,000,000, named contacts), so rebate terms are editable, but only for an
  agency that already exists.
- **Advertiser: can be created, but not under an agency.** "Add advertiser"
  opens a form whose fields are Display name, Advertiser ID (prefilled ADV_46),
  Premium (x rate card), Allowed positions, Allowed genres, Prime time only,
  Behind-pace strength, Over-delivery restraint, Notes. There is **no agency
  field**. Linking runs through a different endpoint,
  `POST /api/agencies/{id}/advertisers`, on a different screen. Submit was not
  pressed, by policy.
- **Campaign with flights: does not exist as an entity.** `/api/campaigns` is
  GET only and returns a rollup of historical spots (50 rows, keys
  `Campaign, advertiser_id, channels, last_airing, revenue, seconds, spots`).
  The dashboard references `/api/campaigns` only to read it
  (TVBreakDashboard.jsx:1291, 5476). There is no create control on any screen.
  The Campaigns page states it directly: "campaign_flights.csv has no campaign
  rows yet (header-only seed)".
- **Saturday-only surcharge discount: representable, in the wrong place.** The
  data model has it: `scope_weekdays` as ISO tokens with Saturday=6, and
  `premium_discount` as a percent off the premium surcharge
  (`kairos_api/condition_validation.py:4-6, 21-40, 58-70`), with the option list
  ordered Sunday first for the Israeli week. But it attaches to an advertiser or
  agency condition, never to a campaign, because there is no campaign to attach
  it to.

The two list screens on this path are the slowest in the product. Measured
twice each from navigation to first row, using resource timing rather than
polling: Agencies **24,975 ms** and **22,265 ms**, with `/api/agencies` itself
taking 19,790 ms and 15,916 ms inside the browser; Advertisers **40,067 ms** and
**23,941 ms**, with `/api/advertisers` at 24,437 ms and 11,953 ms and
`/api/advertisers/stats` at 9,470 ms and 7,657 ms. Standalone `curl`, three
consecutive calls each, confirms there is no cache hiding this:
`/api/advertisers` 19.74 s, 21.55 s, 24.50 s; `/api/agencies` 19.25 s, 16.24 s,
15.31 s.

Verdict: the flow cannot be completed. Two of its three entities have no
creation path in the product.

### 6. Campaign manager: pacing against goal and what to do

Decisive screen: `.../baseline/js6-01-campaigns.png`.

One click, one screen, 6.61 s and 8.26 s to content. The page is honest and
empty of the thing the job needs.

- 50 campaigns listed. **Revenue is a dash on all 50** ("The loaded spots source
  carries no revenue column, so campaign revenue shows a dash and campaigns are
  ranked by spot count"; `revenue_available: false` in the payload).
- **The advertiser column is blank on all 50** (`advertiser_id` empty in every
  row).
- There is no goal, budget, flight or pace field anywhere in the payload; the
  row keys are only `Campaign, advertiser_id, channels, last_airing, revenue,
  seconds, spots`.
- The under-delivery panel reads: "No campaign data yet. Under-delivery alerts
  need real campaign flights. Upload campaign_flights.csv with start and end
  dates and delivery goals to start tracking pacing and make-good risk."

So: no pacing, no under-delivery, no recommended action. The screen names the
exact file that would unblock it, which is the right honesty and zero
capability.

### 7. Traffic operator: build a pod from ads and verify each one

Decisive screens: `.../baseline/js7-01-break-library.png`,
`js7-03-break-detail.png`.

The Break Library is a ranked shelf of 80 breaks with status, channel, airing
time, programme type, position, type, length, revenue and retention. It is a
list of breaks, not a pod of ads.

There is no ad-asset entity to assemble. A repository-wide search for
`aspect_ratio`, `has_audio`, `frame_rate` or `codec` across `kairos/` and
`kairos_api/` returns nothing. The only per-ad field in the system is a creative
name, mapped from the Hebrew spots column "שם גרסה" at
`kairos/data/loaders.py:89` and read back at `kairos/export/spots.py:296`.
`kairos/optimize/frequency.py:6` states the precondition in the code itself:
per-ad attribution "exists" only where that data is present.

Durations summing exactly to break length, per-ad duration to the frame, format,
aspect ratio, audio presence, drag reorder and lock: none of these exist. The
job cannot be started, let alone finished in ninety seconds.

### 8. Analyst: top advertiser last month, gross and net of rebates, without exporting

Decisive screens: `.../baseline/js5-03-agencies-loaded.png`,
`js8-01-reports.png`.

The job stops at the first word of the question, "which advertiser".

- `/api/advertisers/stats` returns 45 advertisers. **Named: 0.** Every
  `display_name` is empty with `name_source: "unnamed"`; the cards read
  "Advertiser 1, unnamed, ADV_01".
- **With revenue: 0.** Every `revenue` and `profitability` is null,
  `revenue_source: "source_pending"`. The payload explains why:
  "Spot-revenue attribution is computed on the daily spot-pricing path only; not
  available in this read-only aggregate."

The only gross and net of rebates figures in the product are one portfolio total
on the Agencies page: gross ₪699,450, agency rebates ₪29,472, net after rebates
₪669,978, 119 priced spots. It is not per advertiser, it has no month selector,
and its basis is a third period again: "Basis: the daily ledger
(Wally_Prime_Reshet_Example_2025-04-27.csv)" while the plan on every other
screen is 1 to 30 Nov 2024. The Reports page offers a "Daily spot ledger" of 175
rows, but reading it is exporting, which the story forbids.

### 9. Kai in Hebrew, from a page, with preview and undo

Decisive screens: `.../baseline/js9-05-dock.png`, `js9-06-typed.png`,
`js9-07-reply.png`, `js9-09-still-thinking.png`.

Interaction: one click to switch the interface to Hebrew (the fresh profile
opens in English), one click to open the dock, one click into the box, 28
characters of Hebrew, Enter. The dock opened 42 ms after the toggle.

What is genuinely good, and measured:

- The dock names the page it is on: "אתם בעמוד לוח שידורים" (you are on the
  Schedule page). Page context is real.
- A pending-actions panel exists and states the contract: "אין פעולות ממתינות.
  כשתבקשו מהעוזר שינוי, ההצעות שלו יופיעו כאן לאישור" (no pending actions; when
  you ask for a change, its proposals appear here for approval). The action
  plane is review-first by construction
  (`kairos_api/assistant_actions.py:3`).
- The request was an action request, "העלה את רף השימור ל-75 אחוז" (raise the
  retention floor to 75 percent), and nothing was changed:
  `min_retention_floor` is still 0.72 after the run.
- The backend answers well. A direct `POST /api/assistant/ask` returned Hebrew,
  grounded and self-correcting: it gave ₪10,123,070.8 with its source
  (`overview_summary.week`), its window (2024-11-01 to 2024-11-07), the channel,
  553 breaks and 94.4 percent retention, then warned unprompted that
  `schedule_freshness` says the plan is stale so the number predates the latest
  settings and coefficients.

What fails: **in the browser the answer never arrived.** The dock showed "קאי
מכין תשובה מהנתונים השמורים" continuously, still true at **499 seconds** after
Enter, with no reply, no error and no cancel. The same backend call from `curl`
took **77.9 seconds**. Both numbers fail an in-product assistant bar by a wide
margin, and the browser one fails completely.

Undo: none. There is no undo control in the product, only the separate "שחזור
שינויים" (Restore changes) page.

One law breach caught here. With the interface set to Hebrew, the Schedule page
still renders the untranslated English string "No gold breaks in the current
plan (none configured as gold in overrides)" under the Hebrew heading "ברייקי
זהב".

### 10. First day, no training, no documentation, nobody to ask

This is story 5 attempted without prior knowledge, so it inherits every blocker
above and adds discoverability ones. It cannot be completed for the same
structural reason: two of the three entities have no creation path.

What a newcomer meets before hitting that wall, all measured on the virgin
profile:

- 17 navigation entries, no onboarding, no empty-state guidance pointing at a
  first task. The only welcome affordance is a badge reading "Open access,
  Sign-in is not set up yet".
- The interface opens in English for an Israeli broadcaster's product.
- The one form they can complete, "Add advertiser", asks a first-day account
  manager for "Behind-pace strength" and "Over-delivery restraint" with no
  explanation of what to enter.
- To register a client restriction they must find the constraint builder inside
  Settings, past `risk_lambda` and "Pace denominator floor", and then choose
  between "Fix offset" and "Offset window" in MM:SS.
- Nothing tells them which of two "Compare" buttons compares scenarios, or that
  pressing the top one leaves the page.

## Cross-cutting facts the rebuild will be measured against

**Latency, measured repeatedly under the load described above.**

| Call | Measurements |
|---|---|
| `GET /api/advertisers` | 19.74 s, 21.55 s, 24.50 s (curl); 24.44 s, 11.95 s (browser) |
| `GET /api/agencies` | 19.25 s, 16.24 s, 15.31 s, 18.17 s (curl); 19.79 s, 15.92 s (browser) |
| `GET /api/advertisers/stats` | 9.96 s, 10.85 s, 7.80 s (curl) |
| `POST /api/scenario-compare` | 25.48 s, 24.87 s, 25.62 s, 20.35 s (log); 22.64 s (curl) |
| `POST /api/optimizer-plan` | 12.85 s (curl); 9.45 s and 9.88 s recorded for the click |
| `GET /api/constraints/effect` | 16.55 s scoped, 55.60 s unscoped, no response unparameterised |
| `POST /api/assistant/ask` | 77.95 s |
| `GET /api/overview` | 3.94 s standalone; 3.50 to 6.71 s inside a page load |

Eight requests issued in parallel, which is what one page load does, complete in
16.98 s wall clock against a single uvicorn worker, so every page load
serializes behind its slowest call. This is why `/api/overview` costs 3.5 to 6.7
seconds in the browser.

**Data vintage.** Three different periods are on screen at once and none is
today: the planning week is 1 to 7 Nov 2024, the saved plan is 1 to 30 Nov 2024,
and the agency ledger basis is `Wally_Prime_Reshet_Example_2025-04-27.csv`. The
saved schedule has been stale since 7/28/2026 11:38:38 AM, and the banner
announcing it is on every page of the product.

**Honest empty states, credited.** The product refuses to fabricate in at least
five measured places: campaign revenue dashes with the reason, advertiser
revenue null with the reason, "Net after retention cost: Not exposed" with the
reason, the make-good panel naming the exact missing file, and the gold-breaks
panel saying none are configured. That discipline is the strongest thing in the
current build and must survive the rebuild.

**Law breaches found while measuring.** Files over 450 lines:
`TVBreakDashboard.jsx` 6,236, `ConstraintBuilder.jsx` 690,
`ScheduleEditor.jsx` 466. Untranslated English inside the Hebrew interface on
the Schedule page. A false offline claim for 1.28 to 1.46 seconds on every boot.

## Screenshots

All under `/Users/home/Code/questo/meridian/docs/ux-gauntlet/discovery/baseline/`.

| Story | File |
|---|---|
| 1 | `js1-02-at-5s.png`, `js1-01-cold-settled.png` |
| 2 | `js2-06-scenario-ab-done.png`, `js2-01-optimizer.png`, `js2-02-scenario-options.png`, `js2-03-compare.png`, `js2-04-scenario-ab-result.png`, `js2-05-scenario-ab-zoom.png`, `js2-07-after-run-optimization.png` |
| 3 | `js3-03-editor-top.png`, `js3-04-mid-drag.png`, `js3-05-after-drag.png`, `js3-01-schedule.png`, `js3-02-schedule-editor.png` |
| 4 | `js4-01-constraint-builder.png`, `js4-02-constraint-builder-scrolled.png` |
| 5 | `js5-03-agencies-loaded.png`, `js5-04-add-advertiser.png`, `js5-01-agencies.png`, `js5-02-advertisers.png` |
| 6 | `js6-01-campaigns.png` |
| 7 | `js7-01-break-library.png`, `js7-03-break-detail.png`, `js7-02-inventory.png` |
| 8 | `js8-01-reports.png` (with `js5-03-agencies-loaded.png`) |
| 9 | `js9-05-dock.png`, `js9-06-typed.png`, `js9-07-reply.png`, `js9-08-reply-final.png`, `js9-09-still-thinking.png`, `js9-01-hebrew.png`, `js9-02-dock-open.png` |
| 10 | reuses the story 5 captures |

## What was not measured, and why

- No mutating action was executed: publish, recompute, save constraint, create
  advertiser, apply an assistant proposal. Each was walked to its final control
  and stopped. Where a story's verdict depends on what happens after that
  control, the verdict rests on the API contract and the rendered form state,
  and says so.
- Human seconds are not reported. Reporting app time plus exact interaction
  counts is measurable; converting them into a person's seconds would be an
  estimate dressed as a measurement.
- The single early reading of `/api/advertisers/stats` at 38 ms, taken at
  18:47 local before any optimizer work in this session, could not be
  reproduced later; nine subsequent measurements all landed between 7.7 s and
  10.9 s. The reproducible figure is reported and the outlier is noted rather
  than explained.
- The break-detail panel on the Break Library was clicked but no detail panel
  was captured in the resulting text, so no claim is made about what it does or
  does not show. The story 7 verdict rests on the absence of ad-asset fields in
  the data model, which is independently verifiable.
