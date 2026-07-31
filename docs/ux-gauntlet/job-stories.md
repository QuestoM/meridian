# The frozen job stories

**These are frozen.** Written 2026-07-31 from the seven discovery reports in
`docs/ux-gauntlet/discovery/` and from measurements taken directly against the
running instance at `http://127.0.0.1:8010`. From this point they do not move.
A builder does not renegotiate a target. A critic does not soften one. If a
story turns out to be wrong, that is an escalation to the owner and a written
amendment, not a quiet edit.

Eighteen stories. Ten are the seeds from `docs/ux-gauntlet-prompt.md`, corrected
against what discovery found. Eight are new, because discovery found people the
seeds did not name and one seed turned out to be two different jobs.

## How to read a story

Every story carries the same six things.

- **Person.** One accountability, not one headcount. Several may be the same
  human at a small broadcaster.
- **Trigger.** What makes them open the product.
- **Sequence.** What they do, in order.
- **Done.** The condition that ends the job. If this is not true, the story
  failed no matter how fast it was.
- **Target.** App time in seconds, clicks, screens, and whether a first-time
  person must complete it unaided. App time means wall clock from the trigger
  action until the answer is on screen, including page load. It is the honest
  floor: no human is faster than the app.
- **Baseline today.** What was measured on 2026-07-31, with where it came from.
  This is the number the rebuild has to beat, and Bar 3 says it may never get
  worse.

Three verdict words are used for the baseline. **Passes** means today's product
completes the story. **Partial** means it completes some of the done condition.
**Cannot start** means a capability the story needs does not exist anywhere in
the repository.

## The scoreboard

| # | Person | Story | Baseline verdict | Target |
|---|---|---|---|---|
| 1 | General manager | Read the week | Partial, 2 of 3 answers | 5 s, 0 clicks |
| 2 | Planner | Build and publish next week | Cannot start (no publish) | 180 s, 12 clicks |
| 3 | Scheduler | Move one break | Partial (moves, money does not) | 20 s, 0 dialogs |
| 4 | Programming representative | Register a restriction | Cannot start (not expressible) | 30 s, 0 engine words |
| 5 | Account manager | Onboard a client | Cannot start (2 of 3 entities) | 120 s, 0 duplicates |
| 6 | Campaign manager | Work the pacing board | Cannot start (0 flights) | 60 s, remedy on the row |
| 7 | Traffic operator | Assemble the pod | Cannot start (no break entity) | 90 s, 7 ads |
| 8 | Traffic operator | Verify the media | Cannot start (owner-blocked) | 0 extra clicks |
| 9 | Analyst | Answer a money question | Cannot start (0 named advertisers) | 30 s, 0 exports |
| 10 | Anyone | Ask Kai and act | Partial (backend only) | 45 s, 2 s first token |
| 11 | New starter | First day, unaided | Cannot start | 300 s, 0 questions |
| 12 | Data steward | Land the morning file | Partial (4 of 7 shadowed) | 60 s to green |
| 13 | Revenue owner | Change a price | Partial (no delta before save) | 45 s |
| 14 | Compliance owner | Attest | Partial (cannot prove no change) | 15 s + 30 s |
| 15 | Administrator | Add a teammate | **Passes** | 30 s |
| 16 | Model steward (company) | Decide whether to ship | Cannot start (no surface) | 120 s |
| 17 | Deployment owner | Bring it up enforced | Partial | 180 s |
| 18 | Critic | The boundary holds | Fails on all four doors | 0 breaches |

---

## JS-1. The general manager reads the week

**Person.** The executive who only wants to look. `viewer` role, enforced
read-only at `kairos_api/auth.py:96-121`.

**Trigger.** First coffee, or a question from the board.

**Sequence.** Open Meridian. Read one screen. Click one number if they want the
rows behind it.

**Done.** They can say out loud, without clicking, whether this week is on plan,
whether anything is broken, and what needs a decision today. Each of those three
answers opens what it refers to in one click.

**Target.** 5 s app time to all three answers. 0 clicks, 0 scrolls, 1 screen.
Each answer reachable to its rows in 1 further click. Unaided: yes.

**Baseline today. Partial, 2 of 3.** Measured cold on a virgin profile: any page
text at 2,461 ms, revenue tile and the amber "Saved schedule is out of date"
banner and the "Priority decisions, 5 actions" list all at 3,588 ms, the net
figure at 5,167 ms; warm runs 2,845 to 3,264 ms
(`06-baseline.md` section 1). "Is this week on plan" is never answered: I
fetched `/api/overview` myself and searched its keys for
`goal|target|budget|pace|on_plan|variance`, and the only hit was the unrelated
`workspace`. No plan target exists anywhere in the product. A false
`api-state offline` chip renders for 1,278 to 1,457 ms on every boot.

**What must exist.** A target entity with an owner-supplied number, and a
three-state verdict against it with a published threshold.

---

## JS-2. The planner builds and publishes next week

**Person.** The planner. Named by the owner.

**Trigger.** Thursday. The coming week's programme lineup has landed.

**Sequence.** Choose the objective. Run the plan. Compare two scenarios on
revenue net of retention cost. Publish.

**Done.** A named, dated plan version is published with an author and a
timestamp, everyone downstream is reading it, and every figure on the path
carries the basis it was computed on.

**Target.** 180 s end to end. 12 clicks. 2 screens. Compare returns in 5 s.
Publish writes a version that can be named, diffed against the previous one and
rolled back. Unaided: no, but with no documentation.

**Baseline today. Cannot start.** The word publish appears zero times in
`kairos_api/` and the weekly plan is not one of the nine logical files the
version store captures (`kairos_api/version_store.py:46`, read directly:
settings, constraints, overrides, advertisers, conditions, events, agencies,
agency_links, agency_conditions). Of the rest: `POST /api/optimizer-plan`
measured 12.85 s, `POST /api/scenario-compare` measured 20.35 to 25.62 s across
four recorded calls, so roughly 45 to 55 s of the 180 s budget is pure wait
before the step that does not exist (`06-baseline.md` section 2). The comparison
returns nothing to choose between: weights 60 and 85 both give ₪1.41M, 95
percent, 80 breaks, 9,600 ad seconds, with a printed delta of ₪0 / 0pp / 0 / 0.
The panel prints "Net after retention cost: Not exposed", which is the exact
quantity the story is defined on. It also compares one representative day
(רשת 13, 2024-11-11), not next week. Seven clicks, three screens, two dead ends,
one of which is a top-bar "Compare" button that silently navigates to another
page (`TVBreakDashboard.jsx:2406-2417`).

**What must exist.** A plan version with a published state, a net-of-retention
figure exposed for comparison, and a compare that runs on the week.

---

## JS-3. The scheduler moves one break

**Person.** The scheduler. Named by the owner.

**Trigger.** A programme ran long, or a programming representative called.

**Sequence.** Open the day. Drag the break. Watch the retention cost and the
revenue move as it lands. Pin a gold break. Respect a restriction without
reading anything.

**Done.** The day is valid, the change is saved, the money it cost or earned is
on screen, and it can be undone.

**Target.** 20 s app time from opening the day to the saved move. Money and
retention update within 500 ms of the drop. 0 dialogs for the move. Undo in 1
keystroke. Gold pinned from the same surface. Unaided: yes for the move.

**Baseline today. Partial.** The drag genuinely works: a real pointer drag moved
a chip from 02:12:00 to 02:36:00 in 2,430 ms, and "Save as pin" and "Discard
change" appeared afterwards. Then it stops. Every ₪ figure on the page was byte
identical before and after the drop, and this is structural, not timing:
`ScheduleEditor.jsx`, `ScheduleEditorBreak.jsx`, `ScheduleEditorRow.jsx` and
`ScheduleEditorToolbar.jsx` contain zero occurrences of "revenue" or
"retention". `grep -rniE "\bundo\b" tv-break-dashboard/src/*.jsx` returns
nothing product-wide. Gold is on another page. The instruction under the
timeline reads "Drag a break to set its offset, then save it as a pin", which is
the engine's word (`06-baseline.md` section 3).

**What must exist.** A live objective evaluation on the drop, an action-level
undo, and gold as an act on the break rather than a separate console.

---

## JS-4. The programming representative registers a restriction

**Person.** The programming representative who holds the objections about when
breaks may not be placed.

**Trigger.** The season finale airs Sunday and the last eight minutes must stay
clean.

**Sequence.** Say it in their own words. See exactly which breaks that would
move and what it costs. Save.

**Done.** The restriction is live, attributed to them, with an end date, and the
cost was on screen before they saved.

**Target.** 30 s from first keystroke to saved. Preview returns in 3 s. 0 engine
words anywhere on the path. The saved restriction carries author, reason and
expiry. Unaided: yes.

**Baseline today. Cannot start.** The owner's own example is not expressible.
Offsets run forward from programme start: the frozen contract at
`docs/constraint-predicate-contract.md:95-96` shows
`"effect": "fix_offset", "offset_seconds": 1320`, and a grep across `kairos/`
and `kairos_api/` for `from_end`, `minutes_before_end`, `before_end` or
`tail_minutes` returns nothing. "Season finale" has nothing to match: of the 418
programme titles `GET /api/constraints/options` returns, the count containing
גמר, פרק אחרון, אחרון, finale or Finale is zero. The builder lives inside
Settings, below `risk_lambda` and "Pace denominator floor", at character 4,240
of a 9,673-character page, and offers "Fix offset", "Offset window", "Pin
count", "Duration range", "Forbid" and "matches regex". The preview exists,
works, and reaches no screen: `GET /api/constraints/effect` returned real before
and after revenue (1,542,178.09) when called directly, costs 16.55 s scoped and
55.60 s unscoped, and no frontend file references it, while the near-identical
`/api/overrides/effect` is wired into two screens
(`05-gaps.md` section 6, `06-baseline.md` section 4).

**What must exist.** A restriction vocabulary written from the programming side
that compiles down to the frozen predicate contract, an offset measured from
programme end, an occurrence concept so one airing can be named, and the preview
wired with a real latency budget. Owner must supply which airings are finales.

---

## JS-5. The account manager onboards a client

**Person.** The account manager who enters the agencies, the clients and the
campaigns.

**Trigger.** A signed insertion order.

**Sequence.** Create the agency. Create the advertiser under it. Create the
campaign with its flights, its rebate terms and its Saturday-only surcharge
discount. One flow.

**Done.** All three exist, linked, visible, with no duplicate entity created
anywhere.

**Target.** 120 s for all three. 0 duplicates. 1 flow, meaning the person never
has to leave and come back to link something. Unaided: yes.

**Baseline today. Cannot start.** Two of the three entities have no creation
path in the product. There is no add-agency control anywhere in the dashboard:
`POST /api/agencies` exists at `kairos_api/agencies.py:347` and has no caller in
the frontend, in the tests, or anywhere else. "Add advertiser" opens a form with
no agency field, because linking runs through a different endpoint on a
different screen. Campaigns have no write endpoint at all among the 56 write
operations I counted on the live `openapi.json`. The Saturday-only discount is
representable in the data model (`scope_weekdays` ISO tokens with Saturday=6,
`premium_discount` as a percent, `kairos_api/condition_validation.py:4-6,21-40`)
but it attaches to an advertiser or agency condition, never to a campaign,
because there is no campaign to attach it to. The two list screens are the
slowest in the product: Agencies 22,265 and 24,975 ms to first row, Advertisers
23,941 and 40,067 ms (`06-baseline.md` section 5).

**What must exist.** Campaign and flight entities with real CRUD, an agency
create path, and a single flow that creates and links in one pass.

---

## JS-6. The campaign manager works the pacing board

**Person.** Whoever runs the campaigns that are on air right now.

**Trigger.** Every morning, and any time a client calls.

**Sequence.** See every campaign on air with its pace against goal. Find what is
under-delivering. Do the recommended thing.

**Done.** Every at-risk campaign has an action taken or an explicit decision to
accept the risk, and both are recorded.

**Target.** 60 s to the worst-pacing campaign, with a three-state verdict, a
published numeric trigger, and the remedy on the same row as the diagnosis.
0 derivation by the reader. Unaided: yes.

**Baseline today. Cannot start.** `data/campaign_flights.csv` is header-only,
zero rows, and `GET /api/make-good-alerts` answers
`{"data_available": false, "reason": "campaign_flights.csv has no campaign rows
yet (header-only seed)."}`. The Campaigns page loads in 6.61 to 8.26 s and shows
50 historical campaigns with revenue as a dash on all 50, advertiser blank on
all 50, and no goal, budget, flight or pace field anywhere in the payload. The
pacing math is real and wired (`kairos/optimize/pacing.py`); the data door is a
CSV upload (`05-gaps.md` section 4, `06-baseline.md` section 6).

**What must exist.** Flights with goals, a delivery figure that updates, and a
forecast state that fires before the shortfall. Owner must supply real flights,
a delivery or as-run feed, and a current week: nothing in the data represents
now (`effective_date` 2026-06-14, plan Nov 2024, daily file 2025-04-27).

---

## JS-7. The traffic operator assembles the pod

**Person.** The traffic operator who builds the break out of individual ads.
This is the arithmetic and ordering half of the owner's seed story 7. The media
half is JS-8, because one is buildable today and the other is not.

**Trigger.** Tomorrow's log is in. The 22:53 break needs assembling.

**Sequence.** Pick the ads. See the break as a physical thing with durations
summing exactly to its length. Reorder by dragging. Lock it.

**Done.** The break is locked, its ad durations sum exactly to its length, and
any ad whose booked duration disagrees with its copy is impossible to miss.

**Target.** 90 s for the real seven-ad break at 22:53:49. 0 dialogs for the
reorder. The sum is visible at all times and updates as ads move. Unaided: yes
for the reorder, no for the lock.

**Baseline today. Cannot start.** A case-insensitive search for `\bpod\b` or
`מקבץ` across every source and data file returns exactly two hits, both the
brief itself. The deepest break object is
`{"num_breaks": 4, "break_length_seconds": 120.0, "position": "middle"}`, four
identical breaks with no contents. Break chips in the editor have
`onMovePointerDown`, `onResizePointerDown` and `onKeyDown` and no `onClick`.
Zero hits for `spots_in_break`, `ads_in_break` or `break_contents`
(`05-gaps.md` section 1).

**The real material, verified by me.** In
`data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv`, the break starting
22:53:49 holds exactly 7 rows summing to 212 seconds. Position 3 is booked at 36
seconds while its copy version is named "מחליפה - סרט מלא 35'". That is the
class of discrepancy this person exists to catch, and it is in the shipped
sample data, so this story is testable against truth rather than a fixture.

**What must exist.** A break entity with an ordered list of spots. Owner must
supply the pod boundary rule: grouping the daily file by `שעת התחלת ברייק` gives
groups of 1 to 38 ads spanning up to fourteen minutes, so the boundary is not
derivable from today's data.

---

## JS-8. The traffic operator verifies the media

**Person.** The same traffic operator. Split out because it is blocked on data
the owner has not supplied, and a story that mixes a buildable job with an
unbuildable one cannot be graded.

**Trigger.** The pod from JS-7 is assembled and about to be locked.

**Sequence.** Read each ad's verified duration to the frame, its format, its
aspect ratio and whether it has audio. Fix or replace what fails.

**Done.** Every ad in the break passed, or the lock is refused and the failing
ad is named.

**Target.** 0 clicks beyond JS-7. The verdict is printed on the ad, not behind a
click. A failing ad blocks the lock. Unaided: yes.

**Baseline today. Cannot start, and owner-blocked.** `aspect ratio`, `audio`,
`LUFS` and `loudness` return zero hits anywhere in the repository. `codec` and
`timecode` return two hits, one of which is a test asserting the word does not
appear in an error message. `house_number` appears in exactly one place in the
whole repository, the rename at `kairos/data/loaders.py:90`, and is never read
again, though the shipped daily file carries 76 distinct House Numbers (I
counted them). `אורך תשדיר` is `int64` with 0 of 175 rows fractional, so
duration to the frame is not representable. `סטטוס` is NaN in 175 of 175 rows,
so no ad in the system carries a state. There is no ffmpeg, ffprobe or
MediaInfo dependency and no media file of any kind on disk
(`05-gaps.md` section 2).

**What must exist.** A media asset entity keyed on House Number, and a technical
metadata feed. Owner must supply, per House Number: exact duration including
frames plus frame rate, container and codec, pixel and display aspect ratio,
audio presence and channel layout, loudness, and the approval state vocabulary.
None of it can be derived from what is on disk.

---

## JS-9. The analyst answers a money question

**Person.** The analyst.

**Trigger.** "Which advertiser delivered the most last month, gross and net of
agency rebates."

**Sequence.** Ask it in the product. Read the answer. Open the rows behind it.

**Done.** The number is on screen with its basis, the rows behind it are one
click away, and nothing was exported.

**Target.** 30 s. 0 exports. The amount opens its rows at more than one level:
advertiser to campaigns to spots. Unaided: yes.

**Baseline today. Cannot start.** The job stops at the first word of the
question. All 45 advertisers return `display_name: ""`,
`name_source: "unnamed"`, `revenue: null`, `revenue_source: "source_pending"`. I
verified the cause myself with pandas: `advertiser_rules.csv` is keyed
`ADV_01..ADV_45`, while `agency_advertisers.csv` and the daily file both use
Hebrew trade names, and the intersection between the id space and either name
space is exactly 0, while the two name spaces match each other 41 of 41. The
only gross-versus-net figures in the product are one agency portfolio total on
one day's file: gross ₪699,450, rebates ₪29,472, net ₪669,978, 119 priced spots
(`02-api-and-data.md` section 6.1, `06-baseline.md` section 8). "Last month" is
also not a period the data has: three vintages are on screen at once and none is
today.

**What must exist.** Advertiser identity, per-spot revenue attribution joined to
a named advertiser, and a period selector the data can honour. This story is the
sharpest single argument for fixing identity before anything else.

---

## JS-10. Anyone asks Kai and it acts

**Person.** Any of the above, from whatever screen they are on.

**Trigger.** A question in natural Hebrew.

**Sequence.** Ask. See exactly what will change before it changes. Apply. Undo
if wrong.

**Done.** The change is applied, a restore point exists, and the restore point
can be inspected before it is used.

**Target.** 45 s from question to applied change. First token within 2 s. The
diff is concrete and previewable before anything lands. Kai knows which object
is open. Undo is an addressable restore point, not a single step. Unaided: yes.

**Baseline today. Partial, backend only.** The backend answers well: a direct
`POST /api/assistant/ask` returned Hebrew, grounded, with its source
(`overview_summary.week`), its window (2024-11-01 to 2024-11-07), the channel,
553 breaks and 94.4 percent retention, and then volunteered unprompted that
`schedule_freshness` says the plan is stale so the figure predates the latest
settings and coefficients. It took 77.95 s. In the browser the answer never
arrived: the dock read "קאי מכין תשובה מהנתונים השמורים" continuously, still
true at 499 s after Enter, with no reply, no error and no cancel. The action
plane behaved correctly: an action request ("raise the retention floor to 75
percent") changed nothing, and `min_retention_floor` was still 0.72 afterwards
(`06-baseline.md` section 9). There is no undo control in the product. I counted
the tool surface in process: 31 read tools, 8 propose tools, 0 write tools.

**What must exist.** Streaming that reaches the browser, a bounded first-token
budget, a diff rendered on the object rather than only in the dock, and a
restore point that can be opened before it is applied.

---

## JS-11. The first day

**Person.** A new account manager. No training, no documentation, nobody to ask.

**Trigger.** Their first morning.

**Sequence.** Complete JS-5.

**Done.** The agency, the advertiser and the campaign exist and are correct.

**Target.** 300 s. 0 questions asked. 0 wrong pages opened. No documentation
consulted. Unaided: yes, by definition.

**Baseline today. Cannot start.** It inherits every JS-5 blocker and adds
discoverability ones, all measured on a virgin profile: 17 navigation entries
with no grouping principle and nothing indicating which are theirs, no
onboarding, no empty-state guidance pointing at a first task, and the interface
comes up in English (`document.documentElement.dir` is `ltr`) for an Israeli
broadcaster's product. The one form they can complete asks a first-day account
manager for "Behind-pace strength" and "Over-delivery restraint" with no
explanation of what to enter (`06-baseline.md` section 10).

---

## JS-12. The data steward lands the morning file

**Person.** The data steward. Inferred from three things: the ingestion surface
declares per-file cadences (daily, weekly, reference, config), `UploadCenter.jsx:25`
says the daily file is "loaded each morning for the next broadcast day", and
`GET /api/reports` names a report owner `Data` (`catalog_api.py:510`).

**Trigger.** The daily ad log for tomorrow lands in their inbox.

**Sequence.** Upload it. Confirm it validated. Confirm the engine is actually
reading it.

**Done.** Every input the engine needs is present, valid, and in use, and
anything shadowed says so in place with the consequence named.

**Target.** 60 s from file to a green in-use state. The status page itself
answers in under 1 s. Unaided: yes.

**Baseline today. Partial.** The validation is exemplary: an accepted upload is
parsed with the real engine loader for its kind and refused at the door with the
contract's own findings. The confusion is that four of the seven declared inputs
report `in_use: false` on the running instance (programmes 3,562 rows, spots
50,386 rows, dayparts 43,200 rows, rate_card 96 rows), each with an honest
reason naming the reference file or the YAML the engine reads instead. So more
than half of what this person maintains is not being read.
`GET /api/uploads/status` took 9.6 to 11.3 s across two investigators, because
it row-counts three large CSVs on every call
(`02-api-and-data.md` section 5.5, `03-people.md` persona 9).

---

## JS-13. The revenue owner changes a price

**Person.** The revenue and yield owner. Inferred as distinct from the planner
because a rate-card edit is commercial policy with a settlement consequence
while the planner's levers are operational, and from `GET /api/reports` naming
`Revenue` as the owner of two reports (`catalog_api.py:508-509`).

**Trigger.** A rate-card revision takes effect Sunday.

**Sequence.** Find the layer. Change the value. See what it does to the plan and
to settlement before saving. Save.

**Done.** The new card is live, the change is attributable, and it is reversible.

**Target.** 45 s from opening Money to a saved change, with the money delta on
screen before the save. Unaided: no.

**Baseline today. Partial.** The layered rate card, the price-any-slot tester
and the per-layer activation switches all exist and load fast. Four of the six
layers move no money: position and ad type are wired off with multipliers that
are not 1.0 (1.30 first position, 0.00 promo, which is precisely why they are
gated), the specific-show layer is empty, and the calendar-events layer is live
over zero qualifying events because all 63 events carry `price_multiplier` 1.0.
Six of the tester's nine inputs cannot change the answer. No delta is shown
before saving (`01-surfaces.md` section 13).

---

## JS-14. The compliance owner attests

**Person.** The compliance owner. Inferred from the hardcoded report owner
`Legal / Ops` at `catalog_api.py:507` and from the disclaimer the compliance
endpoint returns.

**Trigger.** A regulator query, or a monthly review.

**Sequence.** Open the verdict. Read the seven checks with observed against
limit. Prove the limits are the current ones.

**Done.** They can attest, with a source, a date, and evidence that no guardrail
changed since the last attestation.

**Target.** 15 s to the seven checks with source and effective date visible.
30 s to prove no guardrail changed. Unaided: yes.

**Baseline today. Partial.** The first half passes: `GET /api/compliance`
returns profile "Israel commercial TV", `effective_date` 2026-06-14,
`source_url` https://www.rashut2.org.il/, seven named checks, status compliant,
zero violations, and an honest disclaimer. The second half is unbuildable: the
regulatory limits (`max_ad_minutes_per_hour` 12.0, `max_breaks_per_hour` 4,
`min_break_spacing_minutes` 7, `protected_program_max_ad_minutes_per_hour` 8.0)
are ordinary settings fields, editable through `PUT /api/settings` with the same
permission as the revenue-weight slider, with no approval, no effective-date
workflow, no alert and no dedicated record (`03-people.md` persona 11).

---

## JS-15. The administrator adds a teammate

**Person.** The account administrator. The only persona the interface currently
recognises.

**Trigger.** Somebody joins.

**Sequence.** Create the account. Choose the role. Choose the affiliation. Hand
over a temporary password.

**Done.** They can sign in and are forced to change it.

**Target.** 30 s per account. Unaided: yes.

**Baseline today. Passes.** The flow exists, guards against deleting the last
admin or your own account (`auth.py:315-320`), and forces a password change.
This is the one seed-adjacent story the current product genuinely serves. Bar 3
applies with full force: it may not get slower or harder.

**The correction discovery forces.** Three roles do not correspond to any of the
jobs above. The administrator's real worst day is picking the wrong role,
because with three roles for sixteen accountabilities that is the normal case.
The story stays at 30 s, and the roles it assigns have to mean something.

---

## JS-16. The model steward decides whether to ship

**Person.** Whoever at the startup judges whether the model is fit to ship.
Company side of the line.

**Trigger.** A month of new data has accumulated, or a factor is suspected.

**Sequence.** Look at coverage. Look at drift. Train. Read every gate verdict
with its held-out delta and its reason. Decide. Record the decision.

**Done.** The artifact is either shipped with its verdicts, or explicitly not
shipped with the reason, and either way the decision is recorded against a
named model version.

**Target.** 120 s from opening the model console to a recorded ship or no-ship
decision, with every gate visible, the three off-states distinguished, and the
training run started from the console. Unaided: no.

**Baseline today. Cannot start.** Of the 113 operations the live app publishes,
which I counted myself from `openapi.json` (90 paths, 113 operations, 56
writes), zero are training. I filtered those 90 paths for
`train|model|gate|drift|coeff|rebuild` and got only `GET /api/model/audience`
plus the four unrelated constraint paths. Training is a shell command that
leaves no trace in any of the three audit systems: not the activity log, not the
version timeline, not the run log. The gate table renders inside the operator's
Events calendar page. There is no model version identity: the only identifier is
a `computed_at` timestamp on a file that is overwritten in place
(`04-training-vs-runs.md` sections 1.2, 4.2).

**The three off-states that must be distinguished.** Today all three render as a
grey "Off" chip. Tested and lost (a real held-out delta below the bar, for
example `calendar_religious_blackout` at +0.057 percent against a 2 percent
bar). Untestable (`held_out_delta_pct: null`, for example `operator_events`,
because all 3,459 observations fall on event days). Not yet measured. Those are
three completely different pieces of news.

---

## JS-17. The deployment owner brings it up enforced

**Person.** The deployment owner. Inferred from the operational assumptions the
code states about itself: `scripts/init_auth.py` seeds the first admin, six
environment knobs control real behaviour, and three modules state the
single-process assumption in their own docstrings.

**Trigger.** A new deployment, or a restart.

**Sequence.** Seed the auth store. Set the TLS cookie flag. Confirm enforcement
is on.

**Done.** `GET /api/auth/me` returns `auth_disabled: false` and the startup log
says authentication is required.

**Target.** 180 s from clone to enforced login. 0 plaintext passwords persisted
unless generated. The bypass state is visible in the product, not only in a log.
Unaided: no.

**Baseline today. Partial.** The bootstrap is honest: it writes a one-time
password to a mode-600 file and logs loudly when auth is bypassed. I confirmed
the instance state myself: `GET /api/auth/me` returns `{"auth_disabled": true}`.
The product reports it in a small sidebar badge reading "Open access, Sign-in is
not set up yet" and nowhere else. A restart signs everyone out, because sessions
are an in-process dict, and nothing in the product says so.

---

## JS-18. The boundary holds

**Person.** A critic, running this against every build. This is not a person's
day. It is the story that guards the product's two hardest laws, and it is
graded on every round like the others.

**Trigger.** A build claims to be done.

**Sequence.** Sign in as a channel-affiliated operator. Try to reach a training
action. Try to read a model internal. Look for a rival channel's name or money
on any operator surface and in the assistant's context. Read every button that
could mean either train or run.

**Done.** All four attempts fail cleanly, with an honest refusal, and no button
on any surface is ambiguous.

**Target.** 0 breaches. Every refusal is legible before the click, not a 403
after it.

**Baseline today. Fails on all four.** Measured in process against an isolated
`KAIROS_AUTH_DIR`: a channel-affiliated operator passes the middleware on every
path tested except `POST /api/auth/users`, including `POST /api/jobs/recompute`,
`POST /api/uploads/spots` and `PUT /api/settings`. The sharp case is
`audience_model_activation`, which decides where every forward-dated rating
comes from, has no control anywhere in the dashboard, is excluded from the
assistant's allowed fields (I confirmed in process: 19 allowed fields, and it is
not among them), and is settable by any channel operator with one `PUT`.
`GET /api/impact`, the full 47-key coefficients metadata including every gate
reason and the drift monitor, is fetched on every dashboard load for every user
at `TVBreakDashboard.jsx:1295`. I fetched `/api/schedule` myself and its
`break_schedule` is 200 rows of which 96 are `קשת 12`, 73 are `כאן 11`, 28 are
`עכשיו 14` and 3 are the operator's own `רשת 13`. "Recompute" means both
activities, 159 times in the UI and 124 in the backend, and on the Data page the
two meanings sit about ten screen-lines apart
(`04-training-vs-runs.md` sections 2, 3.4, 3.5).

I also confirmed the permission gates directly: `require_company_editor` is
called from exactly five places, three unconditional event writes in
`events_api.py:378,399,426` and two conditional pricing checks in
`pricing_api.py:234,241`. `GET /api/events` returns `can_edit`;
`GET /api/pricing` does not, so the identically walled pricing toggle renders
enabled and refuses after the click.

---

## What these stories deliberately do not cover

Named so a critic does not mark their absence as a miss.

- **A general query or BI tool.** JS-9 covers the money question by drilling
  from a figure to its rows. An arbitrary query builder is a different product.
- **Mobile.** Every one of these is a desk job at a broadcaster.
- **Multi-channel operation.** The operator owns exactly one channel, read from
  settings. Multi-tenant is not in any story.
- **Playout integration.** JS-7 ends at a locked break. Where that break goes
  next is an export contract nobody in the repository knows the shape of, and
  the owner has not named the system of record.

## Amendments

None. This section exists so that any future change to a story is visible as an
addition here with its date, its reason and the owner's decision, rather than as
an edit above.
