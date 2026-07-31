# What the owner described that has no implementation

Discovery, read-only. Every claim below carries its evidence: a file and line, an
endpoint with the response actually received, a measured count, or a screenshot
path. Where something is inferred rather than checked, it says INFERRED and from
what.

Method: the repository at /Users/home/Code/questo/meridian on main at 5a80a709,
and the running instance at http://127.0.0.1:8010 with authentication disabled.
Python is ~/.venvs/meridian/bin/python. Nothing was edited, created or deleted
outside this directory. Every request made during this investigation was a GET.

One caveat about the shared instance. `data/kairos_settings.json` shows as
modified in git (`locale` he to en, `direction` rtl to ltr, plus
`audience_model_activation: false`). That write is not mine: the activity log
records `PUT /api/settings` at `2026-07-31T15:46:32.895+00:00`, five minutes
before this session's first browser action at `15:51:47Z`, and further
`POST /api/scenario`, `/api/optimal-plan`, `/api/scenario-compare` and
`/api/pricing/price-slot` writes at 16:17 through 16:31Z that this session did
not issue. Parallel discovery agents are mutating the same instance, so its
locale and activation state drift under everyone reading it.

Verdict summary.

| # | Thing the owner described | Verdict |
|---|---|---|
| 1 | The pod as a visible object | ABSENT |
| 2 | Per-ad verification, and any ad-level media entity | ABSENT |
| 3 | The traffic operator's day end to end | ABSENT |
| 4 | Live campaign management, make-goods, integrations | ABSENT for live and integrations, PARTIAL for the pacing math |
| 5 | The planner's workflow as a first-class flow | PARTIAL as capability, ABSENT as a flow |
| 6 | Programming restrictions in their own words with a preview | PARTIAL, and the owner's own example is not expressible |
| 7 | Role-based views | ABSENT |
| 8 | Other implied things | see section 8 |

---

## 1. The pod as a visible object: ABSENT

The word does not exist in the product. A case-insensitive search for `\bpod\b`
or `מקבץ` across every `.py`, `.jsx`, `.js`, `.json`, `.md`, `.csv` and `.yaml`
in the repository, excluding `node_modules` and `.git`, returns exactly two
files: `docs/ux-gauntlet-prompt.md` and `docs/ux-gauntlet-goal.md`. Both are the
brief itself. There is no third hit anywhere in the engine, the API, the
frontend or the data.

The deepest break object in the system is a programme segment carrying a count.
`GET /api/schedule/segment/2024-11-23|רשת 13|074` returns:

```json
"plan": {"num_breaks": 4, "break_length_seconds": 120.0, "total_break_seconds": 480.0,
         "position": "middle", "break_type": "medium", "is_gold": false}
```

Four breaks, each 120 seconds, all identical, with no contents. That is the
whole break model.

The page named "Break Library" is a list of segments, not breaks. On
http://127.0.0.1:8010/#Break%20Library the header reads "Ranked break
candidates, 80 breaks", but the rows are the same segments: the row for
`2024-11-23 21:24` shows "Length 8 min", which is `num_breaks` 4 times
`break_length` 120 seconds. Clicking a row calls `openBreak` at
`tv-break-dashboard/src/TVBreakDashboard.jsx:3593`, which opens `ScheduleInspector`
on `row.segment_id` (`TVBreakDashboard.jsx:3664-3674`), so the click lands back
on the same segment payload quoted above.

The schedule editor is the one place a single break is drawn, and it is empty.
`tv-break-dashboard/src/ScheduleEditorBreak.jsx` renders a chip whose three
lines are the clock, the offset into the programme, and the length in seconds.
Measured in the browser at http://127.0.0.1:8010/#Schedule with the Editor tab
selected: every chip on every lane reads `120s`, and the panel below is titled
"Break plan rows" with columns Day, Programme type, Position, Break type,
Breaks, Ad minutes, Revenue, Retention. Screenshot:
`docs/ux-gauntlet/discovery/shots/05-schedule-editor.png`. The chip has
`onMovePointerDown`, `onResizePointerDown` and `onKeyDown` and no `onClick`
(grep for `onClick` in that file returns nothing), so there is not even a
gesture that would open contents.

No containment relation exists in code. A search for `spots_in_break`,
`ads_in_break`, `break_contents`, `spot_list`, `adsList` or `breakSpots` across
`kairos/`, `kairos_api/` and `tv-break-dashboard/src/` returns zero hits.

### What building it requires

- A break entity that does not exist today. The weekly plan's atom is
  `ProgramSegment` with an integer `num_breaks` (`kairos/optimize/optimizer.py`),
  and `segment_id` is `date|channel|index`. A pod needs a stable break identity
  below that, something like `date|channel|segment_index|break_index`, plus an
  ordered list of placements inside it. The anchor discipline already used for
  overrides (`kairos/optimize/overrides.py:246`, re-ingest safety by anchor
  rather than by index) is the pattern this must follow, otherwise a re-ingest
  silently reassigns pods.
- A new store, `data/break_pods.csv` or a JSON store, written under the module
  lock plus temp file plus `os.replace` doctrine documented at
  `kairos_api/agencies.py:1-12`, and registered in the version timeline so a pod
  edit is restorable like every other operating change.
- New endpoints in a new router registered beside the others at
  `kairos_api/server.py:205-271`: read a pod, reorder it, add or remove a
  placement, validate it against the break length.
- An engine seam. The daily per-spot path already groups ads by break start:
  `kairos/export/spots.py` reads `break_start` per row (`spots.py:296` area) and
  `price_daily_file` produces the ledger. That is the only place in the engine
  where ads and breaks meet, and it is where a pod assembler would attach.
- Data the owner must supply: a break boundary the data does not carry. In the
  one daily file on disk, grouping by `שעת התחלת ברייק` produces groups of 1 to
  38 ads whose airing times span up to fourteen minutes (measured: the group at
  `22:03:06` holds 38 rows airing from `22:04:16` to `22:18:06`, totalling 803
  seconds). A group of 38 ads over fourteen minutes is not one pod. So the true
  pod boundary is not derivable from today's data, and the owner would have to
  supply either an explicit pod or break identifier per ad, or the rule that
  splits a break-start group into pods.

---

## 2. Per-ad verification and the ad-level media entity: ABSENT

There is no ad-level media entity anywhere. The ad reaches a surface only as a
free-text name on a spot row.

Attribute by attribute, measured by grep across `kairos/`, `kairos_api/`,
`tv-break-dashboard/src/` and `data/`, excluding `node_modules` and `dist`:

- Aspect ratio: zero hits outside `docs/ux-gauntlet-prompt.md`.
- Audio, loudness, LUFS: zero hits. Not one occurrence in any source or data file.
- Codec, MXF, ProRes, frame rate, timecode: two hits, neither of them a
  capability. `tests/test_qa2_uploads.py:120` asserts the word "codec" does NOT
  appear in an error message, and `kairos/optimize/revenue_net.py:324` has a
  pandas variable named `frame`.
- Duration to the frame: not representable. In the only per-ad file on disk,
  `data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv`, the column
  `אורך תשדיר` is `int64` and the count of rows with a fractional value is 0 of
  175. Durations are whole seconds only.
- Approval state: not present. The file has a `סטטוס` column and it is `NaN` in
  175 of 175 rows. No ad in the system carries a status.

The one media identifier in the data is loaded and then discarded. `House
Number` is renamed to `house_number` at `kairos/data/loaders.py:90`. A
repository-wide grep for `house_number`, excluding caches and builds, returns
exactly one hit: that rename line. It is never read, never joined, never
surfaced, never exported. The same is true of the ad `status` mapped at
`kairos/data/loaders.py:99`.

What the ad actually is, end to end: `kairos/export/spots.py:296` sets
`ad = str(getattr(row, "creative", ""))`, where `creative` is the Hebrew column
`שם גרסה`, a version name typed by a human. That string is the only ad identity
the product ever shows. It appears in exactly one place a person can reach:
`GET /api/export/spots.csv`, whose header is

```
status,advertiser,agency,campaign,program,position,genre,daypart,duration_seconds,
planned_tvr,pricing_type,premium,revenue,net_revenue,placement_value,ad,break_id,
rule_id,limit_type,reason
```

with 175 data rows. `status` there is the pricing outcome (priced or dropped),
not the ad's state. `break_id` is a clock string such as `20:24:23`. There is no
JSON endpoint for these rows: a search for `load_daily_input` and
`price_daily_file` in `kairos_api/` shows they are used only by
`kairos_api/exporters.py` (CSV stream), `kairos_api/uploads.py` (validation),
`kairos_api/overrides.py` (preview inputs) and
`kairos_api/agency_conditions.py` (observed pairs).

The raw material that does exist is worth naming precisely, because it is the
foundation any build would stand on. In that one file, each row carries the
advertiser, the campaign, the version name, the House Number, the duration in
seconds, the position in break (`מיקום בברייק`, 0 for sponsorships and 1..N for
commercials, with 99 used as a last-position sentinel), the spot airing time and
the break start time. Measured example, the break group at `22:03:06`:
sponsorships at position 0 airing 22:04:16 through 22:04:48, then commercials at
positions 1 through 6 airing 22:05:01 through 22:07:05. So ordering and
durations are real. Format, aspect ratio, audio and frame precision are not.

### What building it requires

- A media asset entity keyed on House Number, which the industry already treats
  as the media identifier and which the data already carries. New store,
  `data/media_assets.csv` or equivalent, under the same store doctrine.
- New endpoints for asset read, asset technical status, and the pod's per-ad
  verification roll-up.
- A probe that produces the technical facts. Nothing in this repository can
  read a video file: there is no ffmpeg, ffprobe or MediaInfo dependency
  (checked `requirements.txt`, `requirements-api.txt`, `pyproject.toml`), and no
  media file of any kind is on disk. This is the hard blocker.
- Data the owner must supply, and it is unknowable from what is on disk today:
  the actual media files or a feed of their technical metadata. Specifically,
  per House Number, the exact duration including frames and the frame rate, the
  container and codec, the pixel and display aspect ratio, the presence and
  channel layout of audio, and the loudness measurement. None of this can be
  derived, guessed or modelled from the CSVs present. Also needed: the approval
  workflow's states and who owns each transition, since the `סטטוס` column is
  empty in every row and its vocabulary is therefore unknown.
- Honest limit: even with a probe, the product could verify only what the owner
  sends. If the broadcaster's media lives in a separate MAM or playout store,
  nothing here reaches it.

---

## 3. The traffic operator's day end to end: ABSENT

None of the four stations of that day exists.

Approved ads: there is no approval anywhere. No endpoint approves an ad (the
full write surface is 56 endpoints, enumerated from
`http://127.0.0.1:8010/openapi.json`, and none of them names an ad or a media
asset). The only per-ad status column is empty in 175 of 175 rows, as measured
above. `POST /api/break-decisions` exists but it approves a break-count
recommendation and persists an `Override`, per its own comment at
`kairos_api/dashboard_api.py:1825-1826`.

Assembling a break: covered in section 1. There is no pod.

Locking: the word `lock` in the engine means something else. At
`kairos/optimize/overrides.py:56`, `LOCK = "lock"` is documented as "pin this
spot; keep it exactly as-is, never drop/reprice-away", an instruction to the
optimizer and the rule engine, honoured at `kairos/export/spots.py:250` and
`:303`. There is no broadcast lock, no state that says a break is finished and
may not change, and no locked-versus-open distinction on any surface.

Exporting a locked break: the two exports are not break-level.
`GET /api/export/schedule.csv` has the header `channel,date,day,program_type,
start_time,num_breaks,break_length,total_break_time,...`, which is one row per
segment with a break count. `GET /api/export/spots.csv` is the priced ledger of
the single demo day described in section 2. Neither is an assembled break that
could go to air.

The word "traffic" appears in the product only as a label on things that are not
traffic work. `GET /api/reports` returns a report with `"title": "Weekly traffic
plan", "owner": "Traffic"`, which is the segment CSV above, and the Break
Library page copy at `TVBreakDashboard.jsx:3625` says "export the list for the
traffic meeting". There is no traffic surface, no traffic role, no traffic
queue, and no design document: a grep for "traffic operator", "pod", "media
verification", "aspect ratio" or "house number" across `docs/` returns only the
two gauntlet briefs and one reference screenshot filename.

### What building it requires

Everything in sections 1 and 2, plus:

- A day-shaped work surface rather than a page: a queue of the day's breaks with
  their state, which does not exist as a concept.
- A state machine and its store: draft, assembled, verified, locked, exported,
  with who did each transition and when. The activity log
  (`kairos_api/activity_log.py`) and the version timeline
  (`kairos_api/version_store.py`) are the right substrates for the audit trail
  and the undo, and both already exist.
- An export format the playout side accepts. Nothing in the repository knows
  what that is.
- Data the owner must supply: the day's approved ad list with its media, the
  break structure to fill (see the pod boundary problem in section 1), the
  approval vocabulary, and the target format and delivery channel for the
  locked break. All four are unknowable from what is on disk.

---

## 4. Live campaign management: ABSENT for live, PARTIAL for the pacing math, ABSENT for integrations

The pacing and make-good MATH is real and wired. `kairos/optimize/pacing.py`
implements a documented urgency formula and `project_make_goods`, and its
honesty contract is explicit in the module docstring at `pacing.py:11-21`: with
no campaign data the weights are all 1.0 and the schedule is byte-identical.

The data it needs is empty. `data/campaign_flights.csv` is one line long: the
header only, zero rows. Confirmed live:

```
GET /api/make-good-alerts
{"alerts":[],"data_available":false,
 "reason":"campaign_flights.csv has no campaign rows yet (header-only seed).",
 "as_of":"2026-06-14"}
```

Campaigns on air: not representable. `GET /api/campaigns` is a historical
rollup, not a live list. Measured response fields on the top row:
`"spots": 1025, "seconds": 9038, "revenue": null, "advertiser_id": "",
"last_airing": "30/11/2024"`. Every campaign's last airing is in November 2024,
revenue is null for all of them, and the advertiser is blank. The frontend says
so itself at `TVBreakDashboard.jsx:5476-5478`: "the /api/campaigns payload is a
historical-spots rollup that is always non-empty, so it says nothing about
flights".

Screenshot of the whole page: `docs/ux-gauntlet/discovery/shots/05-campaigns.png`.
The Revenue column is a dash on every row, and the panel below reads "Make-good
alerts, Under-delivery risk, No campaign data yet. Under-delivery alerts need
real campaign flights. Upload campaign_flights.csv with start and end dates and
delivery goals to start tracking pacing and make-good risk." That empty state is
honest and it is also the answer: pacing against goal and under-delivery
detection have no working instance.

Make-goods as a thing you do: absent. `project_make_goods` returns projected
shortfall fractions. There is no make-good entity, no compensating spot, no
credit, no re-book, and no endpoint that creates one.

No campaign CRUD exists at all. Across all 56 write endpoints there is no POST,
PUT, PATCH or DELETE for a campaign. Agencies and advertisers have full CRUD
(`POST /api/agencies`, `POST /api/advertisers`, and their conditions), campaigns
have none. The only way a flight can enter the system is a CSV upload of the
eleven columns fixed at `kairos_api/uploads.py:115-127`: `campaign_id,
flight_start, flight_end, target_impressions, target_grp, delivered_to_date,
scope_channels, scope_genres, scope_dayparts, scope_programmes, notes`.

Integrations to a broadcaster's traffic or sales system: none. A search across
`kairos/`, `kairos_api/` and `scripts/` for `webhook`, `sftp`, `ftp`, `s3`,
`external system`, `integration`, `oauth`, `bxf`, `smpte`, `AdID`, `Landmark`,
`WideOrbit`, `Imagine` and `Broadway` returns no integration of any kind. The
only outbound network clients in the repository are the Anthropic client for the
assistant (`kairos_api/assistant.py:282`) and an optional Gemini client for the
programme classifier (`kairos/data/ai_classifier.py:101`). There is no inbound
API, no scheduled sync, no message queue and no partner credential. Every input
arrives as a file a human uploads, through the seven kinds listed at
`kairos_api/uploads.py:139-147`.

### What building it requires

- A campaign and flight entity with real CRUD, in a new router beside the
  advertiser and agency ones, following `kairos_api/agencies.py` exactly: CSV
  store, module lock, temp file plus `os.replace`, version snapshot per
  mutation, deactivate rather than delete.
- A delivered-to-date that updates. Today `delivered_to_date` is a static
  column in the flight file. Live pacing needs actual delivery joined back from
  the spot ledger, which means the ledger has to cover the flight period. It
  covers one day.
- A make-good entity: the shortfall, the proposed compensating inventory, the
  approval, and the link back to the original flight.
- Data the owner must supply, none of which is on disk: real campaign flights
  with start and end dates and delivery goals, a delivery feed or an as-run log
  so delivered-to-date is not stale, and the commercial rules for what a
  make-good may be offered against and who signs it off.
- Honest limit on "live": nothing in this system represents now.
  `settings.effective_date` is `2026-06-14`, the saved plan covers
  `2024-11-01` to `2024-11-30`, and the one daily ad file is `2025-04-27`. There
  is no current or future week in the data, so a campaign cannot be on air. This
  is a data problem before it is a product problem.

---

## 5. The planner's workflow as a first-class flow: PARTIAL as capability, ABSENT as a flow

Every capability the planner's job story names exists. None of them is joined
into a flow, and the last step does not exist at all.

Where the four steps live, measured:

1. Set the objective. The revenue weight slider and the four templates
   (Balanced, Revenue priority, Retention guardrail, Conservative) are inside
   `SettingsPanel`, at `TVBreakDashboard.jsx:5457-5460` and `:5561`. They exist
   only on the Settings page.
2. Run the optimizer. The "Run Optimization" button is in the global topbar at
   `TVBreakDashboard.jsx:2468`, rendered only when `showOptimizationControls` is
   true, which is the Optimizer and Schedule views.
3. Compare two scenarios. `ScenarioCompare` is mounted exactly once, at
   `TVBreakDashboard.jsx:3778`, inside `ForecastsPage`. The topbar "Compare"
   button does not compare anything: at `TVBreakDashboard.jsx:2409-2412` its
   handler is `setActiveView('Forecasts')`.
4. Publish. Does not exist. A grep of `tv-break-dashboard/src/*.jsx` for
   `publish`, `go live`, `approve plan` or `sign off` returns nothing that is an
   action. The nearest button is "Apply to weekly schedule"
   (`TVBreakDashboard.jsx:2482`), whose own tooltip says it "Saves these levers
   and rebuilds the whole weekly schedule", which is a recompute, not a
   publication. There is no published state, no version marked live, and no
   record of what was published to whom.

So the planner crosses three pages plus a topbar, and the flow has no end.

There is no guided flow of any kind in the product. A grep of
`tv-break-dashboard/src/*.jsx` for `wizard`, `Stepper`, `step 1`, `onboarding`,
`guided`, `checklist` or `next step` returns zero hits.

Two of the pages the planner crosses are near-duplicates. Measured in the
browser: `http://127.0.0.1:8010/#Optimizer` and `http://127.0.0.1:8010/#Schedule`
both render the same "Channel / Program / Days" weekly grid with the same rows,
the same programme titles and the same figures. The Optimizer page adds the four
summary tiles above it and the Schedule page adds view tabs including the Editor.

### What building it requires

- A publish concept, which is a genuine new capability: a plan version marked
  published, with who published it, when, and what it superseded. The version
  store at `kairos_api/version_store.py` already holds restore points and diffs
  and is the natural substrate, so this is a state and an endpoint on top of an
  existing store, not a new store.
- A flow surface that carries the objective, the run, the comparison and the
  publish in one place, with the current state visible throughout. This is a
  composition problem, not a missing engine capability: every underlying
  endpoint already works (`POST /api/optimizer-plan`, `POST /api/scenario-compare`,
  `POST /api/recompute-schedule`, `PUT /api/settings`).
- Nothing new is required from the owner in data terms, with one exception: the
  business rule for what publishing means and who is allowed to do it. That is a
  human decision, not something derivable from the repository.

---

## 6. Programming restrictions in a programming person's own words, with a preview: PARTIAL, and the owner's own example is not expressible

Three separate findings here, and they point different directions.

### The surface exists but it is filed under Settings

`tv-break-dashboard/src/ConstraintBuilder.jsx` is 690 lines and it is a real
predicate builder with AND and OR groups. It is mounted at
`TVBreakDashboard.jsx:5797`, inside `SettingsPanel`. Measured heading order on
http://127.0.0.1:8010/#Settings, read from the live DOM:

```
Market and policy settings, Your channel, Optimizer balance, Profile, Guardrails,
Protected content, Commercial policy, Campaign pacing, Constraint builder,
When (filter conditions), Apply effect, Existing constraints, Activity log
```

So a programming representative registers an objection by opening Settings and
scrolling past the pacing denominator floor. There is no Programming entry in
the seventeen-item navigation (enumerated live in section 7).

### It is not in their own words

The effects a person must choose from, at `ConstraintBuilder.jsx:395-402`, are
labelled "Fix offset", "Offset window", "Pin count", "Duration range", "Gold
break", "Forbid". The operator list at `ConstraintBuilder.jsx:38` includes
"matches regex". The fields are "Programme title", "Genre / programme type",
"Daypart", "Weekday", "Date", "Hour". This is the engine's vocabulary, exposed
directly.

### The owner's example cannot be expressed at all

"No breaks in the last eight minutes of a season finale" has two parts and
neither is representable.

Last eight minutes: offsets are measured from programme START, not end. The
frozen contract at `docs/constraint-predicate-contract.md:95-96` shows
`"effect": "fix_offset", "offset_seconds": 1320` with the note "first break at 22
min". There is no from-end concept: a grep across `kairos/` and `kairos_api/`
for `from_end`, `minutes_before_end`, `before_end` or `tail_minutes` returns
nothing. A rep could only approximate it with an `offset_window` if they knew
the programme's duration and did the subtraction themselves, and the builder
gives them no duration to subtract from.

Season finale: there is no episode or occurrence entity. The predicate's
`programme` field matches the raw title string. Measured against the live
options: `GET /api/constraints/options` returns 418 programme titles, and the
count of titles containing any of גמר, פרק אחרון, אחרון, finale or Finale is
zero. So even a `contains` predicate has nothing to match. The series and
episode work that does exist (`kairos/model/series.py`,
`kairos/data/title_features.py`) is a retention-model pooling layer that
deliberately COLLAPSES episodes into one series key
(`title_features.py:11-13`), which is the opposite of what naming one finale
needs.

### The preview exists on the backend and reaches no screen

`GET /api/constraints/effect` is implemented at `kairos_api/constraints.py:333-410`
and it does exactly the right thing: it runs the commit path's own
`_optimize_one_day` twice, once without the constraints and once with them, and
reports the deltas. Measured live:

```
GET /api/constraints/effect?channel=רשת 13&day=2024-11-23
{"summary": {"before_total_breaks": 80, "after_total_breaks": 80,
             "before_revenue": 1542178.09, "after_revenue": 1542178.09,
             "changed_segments": 0, "matched_segments": 0},
 "changed": [], "skipped_constraints": [], "rejected_overrides": []}
```

No frontend file calls it. Extracting every `/api/...` string literal from
`tv-break-dashboard/src/*.jsx` and `*.js` yields `/api/constraints`,
`/api/constraints/` and `/api/constraints/options`, and not
`/api/constraints/effect`. Its only callers in the repository are three tests
(`tests/test_qa_known_bugs_20260706.py:121`,
`tests/test_qa2_preview_commit_identity.py:118`,
`tests/test_api_surface_qa.py:45`). The sibling endpoint
`/api/overrides/effect` IS wired, from `OverrideConsole.jsx:107` and
`ScheduleInspector.jsx:84`, so the pattern is proven and only this one is
unconnected.

The builder's own actions confirm it: "Save constraint"
(`ConstraintBuilder.jsx:653`) then "Recompute weekly schedule"
(`ConstraintBuilder.jsx:657`). Nothing sits between them. Screenshot of the
rendered builder inside Settings:
`docs/ux-gauntlet/discovery/shots/05-constraint-builder.png`, showing the WHEN
group with field "Programme", operator "is", the APPLY EFFECT row with "Fix
offset" and "OFFSET (MM:SS)" and "ORDER INDEX (OPTIONAL)", the two buttons, and
below them "Existing constraints, 0, No constraints yet". Zero constraints have
ever been registered on this instance, which is itself a signal about how
reachable the surface is.

Two further limits on the endpoint as it stands. Its `day` parameter needs an
ISO date: `day=Sun` returns HTTP 404 `{"detail":"No segments found for the
requested channel-day"}`, measured for Sun, Mon, Fri and Sat against the
operator channel. And its `changed` array carries `segment_id` strings such as
`2024-11-23|רשת 13|074`, not break identities or programme names, so it is not
yet an answer a programming person could read.

### What building it requires

- A Programming surface in the information architecture, distinct from Settings.
  No new store: `data/kairos_constraints.csv` and
  `kairos/optimize/constraints_store.py` already exist and work.
- A restriction vocabulary written from the programming side, mapping down to
  the frozen predicate contract. The contract at
  `docs/constraint-predicate-contract.md` is explicitly frozen, so this is a
  translation layer above it, not a change to it.
- Two genuine engine extensions: an offset measured from programme end, and an
  occurrence concept so a specific airing can be named. The first is small and
  needs only the programme's end time, which `Programmes.csv` already carries
  (columns include `End time` and `End_datetime`). The second is not small.
- Wiring `/api/constraints/effect` into the builder as a live preview, plus
  widening its response to name what moved in human terms and to accept a scope
  wider than one channel-day. The endpoint's math is already the commit path's
  own, which is the property that makes the preview trustworthy.
- Data the owner must supply: which airings are finales. Nothing in
  `Programmes.csv` or the 418 title strings marks one, so a rep cannot name a
  finale until the programme feed carries an episode number, a season position
  or an explicit flag.

---

## 7. Role-based views: ABSENT

Nobody sees a different product. Roles exist, but they are access levels, not
job views.

The model: `ROLES = ("admin", "operator", "viewer")` at
`kairos_api/auth_store.py:34`, and
`AFFILIATIONS = ("company", "channel")` at `auth_store.py:38`.

Server enforcement is real and narrow. `WRITE_ROLES = frozenset({"admin",
"operator"})` at `kairos_api/auth.py:50`; a viewer session is refused any
non-GET with 403 (`auth.py:109-111`); only an admin may reach `/api/auth/users`
(`auth.py:104`). Affiliation gates one thing only: event management, per
`kairos_api/events_access.py:3` and `assistant_event_pipeline.py:21`.

The navigation does not branch. `navItems.map` at
`TVBreakDashboard.jsx:2239-2260` renders every entry unconditionally: there is
no role check, no affiliation check and no filter of any kind in that block.
Measured live in the browser, reading the DOM of `.primary-nav .nav-item`, all
seventeen entries render:

```
Overview, Optimizer, Schedule, Inventory, Break Library, Campaigns, Forecasts,
Events calendar, Reports, Data, Advertisers, Agencies, Pricing, Overrides,
Kai AI assistant, Restore changes, Settings
```

The complete set of role-conditional UI in the entire frontend is two things:
the "Manage accounts" menu item, shown when `auth.user.role === 'admin'`
(`TVBreakDashboard.jsx:2288` and `:2337`), and a write flag in the versions page
(`AssistantVersions.jsx:296`). That is all of it.

So a scheduler, a programming representative, an account manager, a traffic
operator and a general manager all land on the same Overview and see the same
seventeen entries in the same order.

### What building it requires

- A job dimension the model does not have. Today's `role` answers "what may this
  account write". A view needs "what is this person's job", which is orthogonal:
  a traffic operator and a planner are both operators.
- The store already supports it cheaply: `auth_store.add_user` writes a dict
  (`auth_store.py:202-222`) and `normalize_affiliation` (`auth_store.py:175`)
  shows the pattern for adding a field that defaults safely for existing
  records. So a `job` field costs one store field plus one migration-free
  default.
- The information architecture is the real work: per-job landing surfaces and a
  navigation that shows a person their own job first. That is the deliverable
  the brief asks for and it does not exist in any form today.
- Nothing needs to come from the owner except the list of jobs and which surface
  each one lands on.

---

## 8. Other things the brief implies that do not exist

Searched rather than assumed. Each item names how it was checked.

### 8a. The training side has no product surface, and training content leaks onto an operator page

The brief makes the company-staff dashboards a first-class deliverable: what
each gate decided and why, data contrast, drift, what a rebuild would change,
what is blocked on missing data.

There is one model endpoint in the whole API. Filtering the 90 paths in
`openapi.json` for train, model, gate, drift, coverage, coefficient or rebuild
returns `GET /api/model/audience` and the four constraint paths, which are
unrelated. `/api/model/audience` is read-only status. Every training act is a
command-line script: `scripts/compute_measured_coefficients.py`,
`scripts/compute_audience_model.py`, `scripts/train_impact_model.py`,
`scripts/validate_classifier.py` and eleven others. There is no drift view, no
coverage view, no rebuild-impact view, no blocked-on-data register and no
surface restricted to company accounts.

And the leak runs the other way. On http://127.0.0.1:8010/#Calendar, which is the
"Events calendar" navigation entry, any account sees, measured from the rendered
page and captured at
`docs/ux-gauntlet/discovery/shots/05-events-calendar-gates.png`:

- "Week-to-week level drift" with a five-row table of Week, Breaks, Mean level
  (log), values from -0.0450 to -0.0235, marked "Binding".
- "Drift per week (log effect): 0.0202".
- "Training window: 2024-11-01 .. 2024-11-30 (30 days, 2,532 measured breaks)".
- "Coefficients computed at: 29/07/2026, 03:12:13".
- "Wartime disclosure", naming the ceasefire date and "132 of 2532 measured
  breaks".
- "Audience model (expected rating)" with a Factor family and Gate verdict table:
  Weekday and slot Active, Series Active, School holidays and Chol HaMoed Off,
  Hanukkah Off, Shabbat and holy days Off, Season Off, Operator events Off,
  Competitor lineup Active.

A gate verdict table is the training dashboard. It is on a runs surface, with no
affiliation gate on the display.

Building it needs: a company-only surface family gated on the existing
`affiliation` field (the gate mechanism already exists at
`kairos_api/events_access.py`), endpoints that read the artifacts the scripts
already write (`models/`, and the coefficient artifacts the freshness guard
already fingerprints), and the removal of the gate table from the operator
calendar. No new data from the owner.

### 8b. There is no campaign to onboard, so the account manager's one flow cannot be built today

Job story 5 asks for an agency, an advertiser under it, and a campaign with
flights, rebate terms and a Saturday-only surcharge discount, in one flow.
Agencies and advertisers are fully served: `POST /api/agencies`,
`POST /api/agencies/{id}/advertisers`, `POST /api/advertisers`, plus conditions
CRUD on both, plus `rebate_percent` on the agency
(`AgencyDetailDrawer.jsx:282`). The campaign is the missing third of it, per
section 4. So this job story is blocked on the campaign entity, not on the flow.

### 8c. The Reports page is download-only, and no page shows per-advertiser gross and net

Measured on http://127.0.0.1:8010/#Reports: five report cards (Weekly traffic
plan, Compliance and guardrails, Revenue forecast, Daily spot ledger, Source file
audit), each with a row count and a Download CSV button, and no on-screen rows
for any of them. The Daily spot ledger, 175 rows, is the only per-ad artifact in
the product and it can only be read by downloading it.

Job story 8 asks the analyst to answer gross and net of agency rebates without
exporting. Today: the Advertisers page cannot answer it, because
`GET /api/advertisers/stats` returns `"revenue": null, "profitability": null,
"revenue_source": "source_pending"` for every advertiser. The Agencies page
shows the aggregate net after rebates only (`AgencyManager.jsx:79-88`). The
assistant CAN answer it: executing the read tool `get_top_advertisers` returns
per-advertiser `gross_revenue_ils` and `net_revenue_ils` with a totals block
(measured: totals gross 699,450.0 and net 669,978.0 over 119 priced and 56
rule-dropped spots), but its basis is the single-day ledger, not last month. So
the capability is one tool call deep and one day wide, and no screen carries it.

### 8d. No command palette, no keyboard control, no general undo

A grep of `tv-break-dashboard/src/*.jsx` and `*.js` for `command palette`,
`cmd k`, `hotkey`, `shortcut` or a meta-key keydown returns zero hits. The only
undo in the product is the version restore-point path
(`AssistantVersions.jsx`, `/api/versions/{id}/restore`), which is a
whole-file rollback, not an action-level undo. Bar 2 names Linear for
keyboard-first control and Figma for direct manipulation with undo, and neither
has an equivalent here.

### 8e. Nothing in the system represents now

Restating the measurement from section 4 because it constrains several jobs at
once: `settings.effective_date` is `2026-06-14`, the saved plan runs
`2024-11-01` to `2024-11-30`, and the single daily ad file is dated
`2025-04-27`. Job stories that say "on air", "today" or "this week" have no data
to stand on. Any build for the campaign manager or the traffic operator needs
the owner to supply a current week before it can be honest.

### 8f. Observation for the boundary auditor, not a verdict of mine

With `settings.operator_channel` = `רשת 13`, the weekly grid on both
http://127.0.0.1:8010/#Optimizer and http://127.0.0.1:8010/#Schedule renders
lanes for כאן 11, עכשיו 14 and קשת 12 with their programme titles and revenue
figures, and the schedule editor timeline shows a כאן 11 lane beside the רשת 13
lane (`docs/ux-gauntlet/discovery/shots/05-schedule-editor.png`). The audience
model gate table on the calendar page names a factor family "Competitor lineup"
as Active. I am recording what I measured and leaving the boundary verdict to
whoever owns that law, since an all-channels planning grid may be deliberate.

### 8g. Confirmed against the brief's own numbers

The brief asked for these to be verified rather than trusted. Measured now:
`tv-break-dashboard/src` totals 20,172 lines across `.jsx` and `.js`, of which
`TVBreakDashboard.jsx` is 6,236. `kairos_api/` holds 51 Python modules. The
running app's `openapi.json` exposes 90 distinct paths carrying 56 write
operations. `const dayKeys = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']` is
still at `TVBreakDashboard.jsx:586` and the rendered weekly grid header still
reads Mon first, so the Israeli-week debt the brief names is confirmed present.

---

## The shape of it

Four of the eight are absent as whole capabilities rather than as missing
screens: the pod, per-ad media, the traffic operator's day, and role-based
views. Two are blocked on data the owner has not supplied rather than on code:
live campaigns need real flights and a current week, and per-ad verification
needs media files or their technical metadata, neither of which exists anywhere
on disk. Two are close: the planner's flow needs composition plus a publish
concept, and the programming preview already computes correctly on the backend
and simply reaches no screen.

The single cheapest real win in this list is wiring `/api/constraints/effect`
into the constraint builder. It already runs the commit path's own optimizer
twice and returns before and after revenue, and the identical pattern is already
wired for overrides in two places.

The single largest genuinely new thing is the break as an object with contents.
Nothing below the segment count exists, and the pod, per-ad verification and the
traffic operator's day all rest on it.
