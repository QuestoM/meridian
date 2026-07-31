# Who actually uses Meridian

Discovery artifact for the experience gauntlet. Read-only investigation, no product code
touched. Every claim below carries its evidence: a file and line, an endpoint plus the
response actually received, a measured count, or a screenshot path. Claims that are not
directly evidenced are labelled INFERRED with the thing they were inferred from.

Investigated on 2026-07-31 against the repository at `/Users/home/Code/questo/meridian`
(HEAD `5a80a709`) and the running instance at `http://127.0.0.1:8010` with authentication
disabled (`GET /api/auth/me` returned `{"auth_disabled": true}`).

## The headline

Sixteen distinct people have a claim on this system. The system knows about three, and
serves one.

- Three roles exist in code: `ROLES = ("admin", "operator", "viewer")` at
  `kairos_api/auth_store.py:36`, plus a second orthogonal dimension
  `AFFILIATIONS = ("company", "channel")` at `kairos_api/auth_store.py:39`.
- One account exists in the deployed store. `data/auth/users.json` holds exactly one
  record: username `admin`, role `admin`, display name `Admin`, created
  `2026-07-05T18:32:23+00:00`. There is no operator account, no viewer account, and no
  channel-affiliated account.
- The interface adapts to who is looking in exactly three places across 18,164 lines of
  frontend: the admin-only Manage accounts menu item (`TVBreakDashboard.jsx:2288`), the
  admin-only accounts dialog (`TVBreakDashboard.jsx:2337`), and a viewer write-lock in the
  restore page (`AssistantVersions.jsx:296`). Everything else renders identically for
  everyone.
- The backend uses the phrase "the operator" 225 times across `kairos_api/` and `kairos/`
  and uses the words scheduler, trafficker, traffic operator, account manager, campaign
  manager, general manager and buyer zero times each. The word planner appears twice, the
  word analyst once.
- The product's own department vocabulary exists, hardcoded, with no behaviour behind it:
  `GET /api/reports` returns five reports whose `owner` values are `Traffic`,
  `Legal / Ops`, `Revenue`, `Revenue`, `Data` (`kairos_api/catalog_api.py:506-510`). Those
  four department names are the closest thing in the codebase to an organisational model,
  and nothing reads them.

The system therefore has one persona, "the operator", who is assumed to do everything from
seeding the auth store to assembling the commercial pod. That single persona is the root
cause of the seventeen flat navigation entries: with one user there is no basis on which to
group anything.

## The evidence base

| Surface | What it yielded | Where |
| --- | --- | --- |
| Auth roles and affiliations | 3 roles, 2 affiliations, 1 live account | `kairos_api/auth_store.py:36,39`; `data/auth/users.json` |
| Enforcement rule | viewer is read-only, admin owns accounts, 4 of 113 operations are affiliation-gated | `kairos_api/auth.py:96-121`; `kairos_api/events_access.py` |
| Endpoint census | 90 paths, 113 operations | live `GET /openapi.json` |
| Entity stores | 11 mutable CSV or JSON stores, 9 of them versioned | `kairos_api/version_store.py:46` |
| Activity log | 5,369 entries, 2026-07-06 to 2026-07-29, schema user/role/event/method/path/status/duration_ms/via | `data/audit/activity.jsonl`; `kairos_api/activity_log.py:96-116` |
| Assistant tool surface | 31 read tools, 8 propose tools, no write tool | `kairos_api.assistant_tools.READ_TOOL_NAMES`, `PROPOSE_TOOL_NAMES` |
| Role vocabulary in data | 11 distinct counterparty job titles in Hebrew | `data/agencies.csv` columns `contact_role`, `contact2_role` |
| Source data | daily ad log carries House Number, copy version, duration, pod position | `data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv` |
| Model artifacts | 8 gated audience factor families, retention gate metadata | live `GET /api/model/audience`; `models/tv_break_coefficients.json` |

Screenshots captured during this investigation (untracked session artifacts,
`.playwright-mcp/` is gitignored at `.gitignore:84`):

- `/Users/home/Code/questo/meridian/.playwright-mcp/people-02-overview.png` (Overview,
  titled "Executive operating view")
- `/Users/home/Code/questo/meridian/.playwright-mcp/people-03-calendar-training-gates.png`
  (the audience model gate table rendered inside the Events calendar page)
- `/Users/home/Code/questo/meridian/.playwright-mcp/people-04-campaigns.png` (Campaigns,
  read-only historical roll-up with no revenue and no flights)
- `/Users/home/Code/questo/meridian/.playwright-mcp/people-01-overview-nav17.png` (the
  seventeen navigation entries)

Two honesty notes about the evidence itself.

1. The activity log is dominated by test traffic, not by human use. The usernames
   `viewer1`, `operator1`, `usera`, `userb`, `admin2`, `chan1`, `comp1` and the path
   `/api/definitely-not-a-route` all originate in `tests/test_auth.py`,
   `tests/test_qa8_permissions.py`, `tests/test_activity_log.py`,
   `tests/test_assistant_actions.py` and `tests/test_version_store.py`. The log carries no
   evidence of real multi-person production use, and I did not find any elsewhere. Every
   persona below is therefore derived from what the system is built to serve, not from
   observed usage.
2. The shared instance is being mutated by parallel work during this investigation.
   `data/kairos_settings.json` flipped `locale` from `he` to `en` and `direction` from
   `rtl` to `ltr` at 2026-07-31 18:46:32 local, and version `048f0c3034ae` records that
   write as actor `auth-disabled`, source `manual_edit`, files `["settings"]`. I issued
   only GET requests. Latency numbers below were taken around that window and should be
   re-measured before being used as a baseline.

## Part 1: the identity model that exists

### Roles

`admin`, `operator`, `viewer` (`kairos_api/auth_store.py:36`). The product's own words for
them, from `tv-break-dashboard/src/Login.jsx:81-85`, are Admin / ניהול, Operator / תפעול,
Viewer / צפייה. The help text at `TVBreakDashboard.jsx:6205` states the contract: "The
viewer role reads only, operator edits and runs, admin also manages accounts."

Enforcement is one rule in one place, `kairos_api/auth.py:96-121`:

- Any `/api/` path other than `POST /api/auth/login` and `GET /api/health` needs a live
  session.
- `/api/auth/users*` needs `admin`.
- Any POST, PUT, PATCH or DELETE outside `/api/auth/` needs `admin` or `operator`.

That is the entire authorisation model. `operator` and `admin` differ only in account
management. There is no permission that distinguishes editing the rate card from moving a
break, or registering a programming restriction from rebuilding the weekly plan.

### Affiliations

`company` and `channel` (`kairos_api/auth_store.py:39`). This is the training versus runs
line, and it is implemented for exactly four of the 113 operations:

- `POST /api/events`, `PUT /api/events/{event_id}`, `DELETE /api/events/{event_id}` via
  `require_company_editor` at `kairos_api/events_api.py:378,399,426`.
- `PUT /api/pricing` when the body carries `pricing_activation.events`, at
  `kairos_api/pricing_api.py:234,241`.
- Plus the assistant apply path, `kairos_api/assistant_actions.py:451-455`.

A missing affiliation reads as `company` (`kairos_api/auth_store.py:139-142`), so every
legacy account keeps full access. The UI states the wall at
`TVBreakDashboard.jsx:6210-6212`: "A channel-affiliated account cannot manage calendar
events or event pricing."

Nothing else is affiliation-gated. In particular, there is no training action anywhere in
the API to gate: a scan of the live OpenAPI document for paths containing train, model,
rebuild, coeff, gate or drift returns only `GET /api/model/audience`, which is a read-only
disclosure. Model rebuilds happen exclusively in a terminal, via
`scripts/compute_measured_coefficients.py` and `scripts/compute_audience_model.py`.

### Accountability

`kairos_api/activity_log.py` records every mutating `/api` request with user, role, method,
path, status, duration and a `via` field that is `assistant` when the path starts with
`/api/assistant` and `dashboard` otherwise (`activity_log.py:96-116`). Visibility is
role-scoped: admin sees everything and may filter by user, every other role sees only its
own entries and the filter is refused with 403 (`activity_log.py:262-300`). The version
store records an `actor` on every snapshot (`data/versions/*/manifest.json`).

So the system already knows the question "who changed what" matters. It just has nobody
distinct to answer it about.

## Part 2: the people

Sixteen. Twelve on the broadcaster's side of the line, two on the startup's side, plus two
cross-cutting identity classes. For each: status, evidence, accountability, what the system
gives them, what it does not.

---

### 1. The executive who only wants to look

**Status: EVIDENCED.** The owner named "someone who only wants to look at data". The `viewer`
role exists and is enforced read-only. The Overview page is titled, in the running app,
"Executive operating view" with the subtitle "A single read on revenue, retention,
compliance, and the next decisions traffic teams need to make"
(`.playwright-mcp/people-02-overview.png`).

**Accountable for:** nothing they change. They answer upward for whether the week is on
plan.

**Has:** one composed payload. `GET /api/overview` returns, live, total breaks 2,391, total
ad seconds 286,920, projected revenue 40,944,759.33 ILS, average retention 94.6 percent
with `retention_basis` `tvr_weighted`, a nested `week` block, five recommendations, the
full compliance verdict, a six-point frontier with a stated basis, and
`schedule_freshness` currently reading `{"status": "stale", "changed": ["settings",
"coefficients"]}`. Measured wall time for that single call: 1.77s, 1.66s, 1.28s on three
consecutive requests.

**Lacks:**

- Any target to be on plan against. There is no budget, goal or quota entity anywhere in
  the data model. `data/campaign_flights.csv` is header-only (1 line, verified), and
  `GET /api/make-good-alerts` answers `data_available: false` with the reason
  "campaign_flights.csv has no campaign rows yet (header-only seed)." So "on plan" is
  currently unanswerable and the page cannot honestly claim it.
- Safety from controls they cannot use. The Overview renders Approve and Approve similar
  on every recommendation (`TVBreakDashboard.jsx:340,342,1695`), which POST to
  `/api/break-decisions` (`TVBreakDashboard.jsx:1223`). A viewer session gets 403 from
  `auth.py:113-119`, and no role gate hides the button.
- A comparison over time. No endpoint returns last week beside this week.

---

### 2. The analyst

**Status: EVIDENCED as a distinct set of needs, INFERRED as a distinct person** from the
existence of export-only surfaces that no other persona's flow uses, and from job story 8
in the brief naming an analyst separately from the general manager.

**Evidence:** `kairos_api/exporters.py` serves `GET /api/export/schedule.csv` and
`GET /api/export/spots.csv`, and its docstring states the spot ledger "is the only surface
that exposes that pipeline; before it, the priced/dropped ledger was reachable from tests
only." `GET /api/agencies/summary` returns gross, net and rebate totals.
`GET /api/yield-per-second`, `GET /api/impact` and `GET /api/reports` exist.

**Accountable for:** answering commercial questions with numbers somebody else will act on.

**Has:** two CSV exports and three aggregate endpoints.

**Lacks:** the ability to answer the question the brief poses. `GET /api/agencies/summary`
returned gross 699,450.00, net 669,978.00, rebate 29,472.00 over 119 spots, with
`basis: "Wally_Prime_Reshet_Example_2025-04-27.csv"`, which is one day. There is no
per-advertiser revenue at all: `GET /api/advertisers/stats` returns `revenue: null` and
`revenue_source: "source_pending"` for every one of 45 advertisers, and it took 3.61s to
say so. `GET /api/campaigns` returns 50 campaigns with `revenue_available: false` and an
empty `advertiser_id` on the top row. So "which advertiser delivered the most last month,
gross and net of agency rebates" cannot be answered by this product today, with or without
an export.

---

### 3. The planner

**Status: EVIDENCED.** Named by the owner. `kairos_api/exporters.py` describes the schedule
export as "genuine planner output". The whole optimizer surface exists for this person.

**Evidence:** `POST /api/optimal-plan`, `GET`/`POST /api/optimizer-plan`,
`POST /api/scenario`, `POST /api/scenario-compare`, `GET /api/optimizer/net-comparison`,
`GET /api/parameters`, `GET /api/forecasts`, `GET /api/settings/controls` (which returns
`levers`, `templates` and `current`, with `revenue_weight` described as "The central lever"
and stating it "drives the weekly schedule, the efficiency frontier, and the forecasts").

**Accountable for:** the shape of next week: how many breaks, where, at what balance of
revenue against retention.

**Has:** four named objective templates (balanced, revenue, retention, conservative, at
`TVBreakDashboard.jsx:5457-5460`), a live optimizer, a two-scenario comparison, and a
frontier with a disclosed basis.

**Lacks:**

- A publish step. The word publish appears zero times in `kairos_api/` and once in the
  entire frontend. There is no plan status, no lock, no approval, and the weekly plan is
  not one of the nine versioned logical files (`version_store.py:46` lists settings,
  constraints, overrides, advertisers, conditions, events, agencies, agency_links,
  agency_conditions). The plan itself has no history.
- A predictable wait. From the activity log, `POST /api/optimal-plan` over 97 recorded
  calls has a median of 982 ms and a maximum of 275,632 ms, which is four and a half
  minutes. `POST /api/scenario` over 212 calls: median 262 ms, maximum 15,422 ms. Nothing
  on the screen tells the planner which of those two worlds they are in.

---

### 4. The scheduler

**Status: EVIDENCED.** Named by the owner. The placement surface exists:
`ScheduleEditor.jsx`, `ScheduleEditorBreak.jsx`, `ScheduleEditorRow.jsx`,
`ScheduleEditorToolbar.jsx`, `ScheduleInspector.jsx`, `schedule-track-view.jsx`,
`GoldBreakManager.jsx`.

**Evidence:** `GET /api/schedule/segments`, `GET /api/schedule/segment/{segment_id}`,
`POST /api/overrides` with `GET /api/overrides/effect` for a with-and-without preview,
`POST /api/break-decisions`, `GET /api/gold-breaks`.

**Accountable for:** where each break actually lands in a real day, and defending that
choice.

**Has:** a preview that runs the optimizer with and without a candidate override and
"reports rejected overrides verbatim from the optimizer (never hiding an infeasible one)"
(`kairos_api/overrides.py` docstring). A gold break concept with a per-day cap of 3
(`data/kairos_settings.json`).

**Lacks:**

- Undo. The strings undo and Undo appear zero times in `ScheduleEditor.jsx`,
  `ScheduleInspector.jsx` and `schedule-track-view.jsx`. Recovery exists only through the
  version timeline, which is a different page called Restore changes.
- Anything to work on. Live state on the shared instance: `GET /api/overrides` returns
  `{"overrides": {"segment": [], "spot": []}}`, `GET /api/break-decisions` returns
  `{"decisions": []}`, `GET /api/gold-breaks` returns count 0 with reason "No gold breaks
  in the current plan (none configured as gold in overrides)." Every store this persona
  owns is empty in the deployment.

---

### 5. The programming representative

**Status: EVIDENCED.** Named by the owner. The constraint layer exists for exactly this
person: `kairos_api/constraints.py` with CRUD plus `/options` plus `/effect`,
`ConstraintBuilder.jsx` at 28,950 bytes, and a frozen contract at
`docs/constraint-predicate-contract.md`.

**Accountable for:** protecting the programme. No break in the last eight minutes of a
finale, nothing inside a memorial broadcast, nothing that breaks a narrative beat.

**Has:** a scoped predicate store with an honest with-and-without preview that "serves the
option lists the dashboard needs to build a scoped rule (real programme Titles, channels,
weekdays, effects, scope types)" (`kairos_api/constraints.py` docstring).

**Lacks:**

- Their own words. The live `GET /api/constraints` returns the column contract
  `constraint_id, scope_type, scope_value, channel, effect, offset_seconds,
  offset_min_seconds, offset_max_seconds, count, duration_seconds, duration_min_seconds,
  duration_max_seconds, order_index, notes, where_json`. Those are the builder's fields.
  A programming representative saying "not in the last eight minutes of a season finale"
  has to translate that into an offset in seconds against a scope type.
- Any identity on the rule. There is no author, no requested_by, no approver, no expiry
  and no review date in that column list. A restriction, once saved, is anonymous and
  permanent.
- Anything to work on. `GET /api/constraints` returned `{"constraints": []}` on the live
  instance. `data/agency_conditions.csv` and `data/advertiser_conditions.csv` are both
  header-only.

---

### 6. The account manager

**Status: EVIDENCED.** Named by the owner. The largest built-out surface in the product
after the optimizer.

**Evidence:** `kairos_api/agencies.py` (9 rows in `data/agencies.csv`),
`kairos_api/advertisers.py` (45 rows in `data/advertiser_rules.csv`),
`kairos_api/agency_conditions.py` (links, 41 rows in `data/agency_advertisers.csv`),
`kairos_api/advertiser_conditions.py`, and the frontend `AgencyManager.jsx`,
`AgencyDetailDrawer.jsx`, `AdvertisersManager.jsx`, `AddAdvertiserForm.jsx`,
`AdvertiserDetailDrawer.jsx`, `AdvertiserConditions.jsx`, `AdvertiserPricingSummary.jsx`.

The trade's own job titles are in the data. `data/agencies.csv` carries `contact_role` and
`contact2_role` with eleven distinct values across nine agencies: מנהלת לקוח (4), מנהל לקוח
(2), בעלים (2), סמנכ"ל מדיה, מנהל תכנון מדיה, מנהלת קניית מדיה, מנהלת קמפיינים, רפרנטית
מדיה, מנהל תחום שידור, מנהלת תכנון, מנהל דיגיטל ושידור. These are the counterparties, not
Meridian users, but they are the closest thing in the repository to a map of who is on the
other end of the phone. All nine agency rows carry `data_source: synthetic` and the note
"פרטי קשר סינתטיים לדוגמה, יש להחליף בנתוני אמת".

**Accountable for:** the commercial record. Who the agency is, what they are owed, which
advertisers sit under them, what each may and may not have.

**Has:** agency records with payment terms, rebate percent, commission percent, credit
limit, VAT id, two contacts and provenance. Advertiser rules with premiums and scoped
conditions. Deactivation instead of deletion, so "a suspended agency keeps resolving on
historic spots while its conditions and rebate go inert on the pricing path"
(`kairos_api/agencies.py` docstring).

**Lacks:**

- The campaign. There is no campaign entity and no campaign write path. `GET /api/campaigns`
  is a read-only roll-up derived from the historical spots file. The only way a campaign
  can enter this system is by uploading `campaign_flights.csv` through
  `POST /api/uploads/campaign_flights`. Job story 5 in the brief, onboard an agency, an
  advertiser under it, and a campaign with flights, rebate terms and a Saturday-only
  surcharge discount, is roughly two thirds unimplemented: the agency and advertiser halves
  exist, the campaign half does not.
- Money on their own records. Every advertiser shows `revenue: null`.
- Speed. `GET /api/advertisers/stats`, which the Advertisers page needs to render, measured
  3.61s.

---

### 7. The campaign manager

**Status: EVIDENCED as a need with real machinery, unserved.** The owner named "whoever runs
the campaigns that are on air".

**Evidence:** `kairos/optimize/pacing.py` implements three-tier pacing strength resolution.
`GET /api/make-good-alerts` and `MakeGoodAlerts.jsx` exist. The flight contract is written
down in the upload schema: `campaign_id, flight_start, flight_end, target_impressions,
target_grp, delivered_to_date, scope_channels, scope_genres, scope_dayparts,
scope_programmes, notes`. The settings carry seven live pacing knobs (`pacing_enabled`
true, `pacing_urgency_k`, `pacing_urgency_max`, `pacing_ahead_k`, `pacing_weight_floor`,
`pacing_epsilon`, `pacing_reference_date`).

**Accountable for:** every campaign currently on air delivering what was sold.

**Has:** nothing usable. The Campaigns page, live, is titled "Campaign allocation" and shows
a table of 50 historical campaigns with Spots, Minutes, Channels, a Revenue column rendering
a dash on every row, and a note "The loaded spots source carries no revenue column, so
campaign revenue shows a dash and campaigns are ranked by spot count." Below it, the
Make-good alerts panel reads "No campaign data yet"
(`.playwright-mcp/people-04-campaigns.png`).

**Lacks:** the flight entity as a first-class thing with a create path, a goal, a delivered
figure, a pace, and a recommended action. The engine is ready and the data door is a CSV
upload.

---

### 8. The traffic operator

**Status: EVIDENCED as a person with real data and zero implementation.** The owner named
them. This is the single largest gap found in this investigation.

**Evidence that the person and their data are real:**
`data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv`, the daily ad log, is ingested
by the product (`GET /api/uploads/status` reports kind `daily`, label "Daily ad log
(Wally)", cadence daily, 175 rows, valid true, `in_use` true). Its eighteen columns are
exactly a traffic department's working set: תאריך, שעה, שעת התחלת ברייק, משרד / MB, סוג
תשדיר, מפרסם, קמפיין, שם גרסה, House Number, אורך תשדיר, תוכנית מוזמנת, שעת התחלת תוכנית,
סוג ברייק, סוג תמחור, מחיר, רייטינג ברייקים מתוכנן, מיקום בברייק, סטטוס.

Measured from that file: 10 distinct pods, pod sizes 1, 1, 3, 3, 7, 28, 29, 30, 35, 38,
totalling 175 spots; 76 distinct House Numbers; 51 campaigns; 41 advertisers; 9 agencies;
124 commercials and 51 sponsorships; 124 CPP and 51 FIX; durations from 6 to 52 seconds
across 27 distinct values. The `סטטוס` column is empty on all 175 rows and the `מחיר`
column is empty on all 175 rows.

A real seven-ad pod exists in the shipped data, which makes job story 7 testable against
truth rather than a fixture. Pod at 22:53:49, inside "המקור - עונה 24 - דיון באולפן",
212 seconds total:

| Pos | Seconds | House Number | Advertiser | Copy version |
| --- | --- | --- | --- | --- |
| 1 | 28 | CID179035 | כלמוביל | מרצדס סטאר ליס 28" |
| 2 | 35 | CGB007548 | בנק הפועלים | סרט ראשי 35" |
| 3 | 36 | CID178977 | קופ"ח מאוחדת | מחליפה - סרט מלא 35' |
| 4 | 34 | CMK022716 | ביטוח ישיר | 34 |
| 5 | 45 | CID178966 | פריסבי | סרט ארוך 45" - מחליפה |
| 6 | 14 | CID179004 | חן ואיתי גינדי | סרט 15 ימי מכירות |
| 99 | 20 | CID178897 | קרסו מוטורס | 20 |

Note position 3: the copy version name claims 35 and the booked duration says 36. That is
precisely the class of discrepancy this person exists to catch, and it is sitting in the
shipped sample data today.

**Evidence that nothing is built for them:** the strings House Number, house_number,
creative, copy_id, pod, aspect, audio and QC appear zero times in
`tv-break-dashboard/src/`. A repository-wide search for aspect_ratio, loudness, LUFS, codec,
frame_rate, mezzanine, audio_track, has_audio, media_check, qc_status and transcode across
Python, JSX, CSV, YAML and JSON returns no media field at all: every hit is the word
"resolution" used to mean rule resolution. The Break Library page, read live from the DOM,
contains no occurrence of ad, House Number, creative, תשדיר or גרסה anywhere on the page;
it is a table of break candidates with Break, Channel, Airing, Programme, Position, Type,
Length, Revenue, Retention.

**Accountable for:** that what airs is what was sold, in the right order, at the right
length, technically clean.

**Has:** nothing. Their input file is read for pricing and agency attribution, and the ad
rows inside it are never shown to anyone.

**Lacks:** the pod as an object, per-ad technical verification of any kind, a status
vocabulary, an ordering tool, and a lock.

---

### 9. The data steward

**Status: INFERRED as a distinct person,** from three things: the ingestion surface declares
per-file cadences that only make sense as somebody's recurring obligation
(`kairos_api/uploads.py:141` and the live status endpoint list seven inputs with cadences
daily, weekly, reference, config); `UploadCenter.jsx:25` says the daily file is "loaded each
morning for the next broadcast day"; and `GET /api/reports` names a report owner `Data`
(`catalog_api.py:510`).

**Accountable for:** the engine reading true inputs.

**Has:** an ingestion surface that refuses bad files at the door. `kairos_api/uploads.py`
parses an accepted upload "with the REAL engine loader for its kind" and checks it against
`kairos.data.contracts`, "so a file that would break or silently empty the optimizer is
refused at the door with the contract's own findings."

**Lacks:** clarity about what is actually live. Of the seven declared inputs, four report
`in_use: false` on the running instance: `programmes` (valid, 3,562 rows, shadowed with the
honest reason "the engine reads data/reference/Programmes.xlsx first and adopts this upload
only when that reference file is absent"), `spots` (50,386 rows), `dayparts` (43,200 rows)
and `rate_card` (96 rows). Only `daily`, `advertiser_rules` and `campaign_flights` are in
use, and `campaign_flights` has 0 rows. The honesty is exemplary and the situation is
confusing: more than half of what this person is asked to maintain is not being read.

---

### 10. The revenue and yield owner

**Status: EVIDENCED as a surface, INFERRED as a distinct person** from the fact that
rate-card edits are commercial policy with a settlement consequence while the planner's
levers are operational, and from `GET /api/reports` naming `Revenue` as the owner of two
reports (`catalog_api.py:508-509`).

**Evidence:** `kairos_api/pricing_api.py` opens with "The operator owns the rate card."
`GET /api/pricing`, `PUT /api/pricing`, `POST /api/pricing/price-slot`,
`PricingManager.jsx`, `PricingSlotTester.jsx`, `AdvertiserPricingSummary.jsx`,
`pricing-layers-lib.js`, `data/rate_card_premiums.csv` with 96 rows, and
`docs/pricing-hierarchy-design.md`. From the activity log,
`POST /api/pricing/price-slot` is the third most-called mutating route on record with 290
calls, and `PUT /api/pricing` has 198.

**Accountable for:** what a second of airtime is worth, and that a change to that is
deliberate.

**Has:** a full layered rate card, a price-any-slot tester with a per-layer breakdown, and
an activation switch per layer with position, ad type and show shipping off because their
multipliers are not 1.0.

**Lacks:** any separation from the planner or the scheduler. All three are the same
`operator` role with the same write permission. A rate-card edit and a break move are
equally reversible and equally unannounced.

---

### 11. The compliance owner

**Status: EVIDENCED as a surface, INFERRED as a distinct person** from the hardcoded owner
string `Legal / Ops` at `catalog_api.py:507` and from the disclaimer the endpoint itself
returns.

**Evidence:** `GET /api/compliance` returns, live, profile "Israel commercial TV",
`effective_date` 2026-06-14, `source_url` https://www.rashut2.org.il/, seven named checks,
status compliant, zero violations, and the disclaimer "Configurable baseline. Validate with
current counsel and broadcaster policy before production use." The guardrails themselves sit
in operator settings: `max_ad_minutes_per_hour` 12.0, `max_breaks_per_hour` 4,
`min_break_spacing_minutes` 7, `protected_program_types` News, Kids, Children at
`protected_program_max_ad_minutes_per_hour` 8.0.

**Accountable for:** the channel not breaching its licence.

**Has:** a verdict with seven checks and a stated regulatory source.

**Lacks:** any protection on the numbers they are accountable for. The regulatory limits are
plain settings fields, editable through `PUT /api/settings` by any operator, with the same
permission as changing the revenue weight slider. There is no approval, no effective-date
workflow, and no alert when a limit changes.

---

### 12. The account administrator

**Status: EVIDENCED.** The only persona the interface actually recognises.

**Evidence:** role `admin` (`auth_store.py:36`); admin-only routes `GET/POST /api/auth/users`,
`DELETE /api/auth/users/{username}`, `PUT /api/auth/users/{username}/affiliation`,
`POST /api/auth/users/{username}/reset-password` (`auth.py:284-350`); the only two
role-conditional renders in the product (`TVBreakDashboard.jsx:2288,2337`); admin-scoped
activity log visibility (`activity_log.py:262-300`); and the accounts dialog copy "Each
teammate signs in with a personal account; the role decides what the account can change."
(`TVBreakDashboard.jsx:6022`).

**Accountable for:** who is in the system and what they may do.

**Has:** create, delete, reset password, flip affiliation, and a guard against deleting the
last admin or your own account (`auth.py:315-320`).

**Lacks:** anything to administer. One account exists. And the roles they can assign do not
correspond to any of the fifteen other people on this list.

---

### 13. The model steward, on the startup's side

**Status: EVIDENCED as a role with real artifacts and no product surface.** The owner named
them: "whoever at the startup judges whether the model is fit to ship, watches the gates,
the drift and the data coverage, and decides when a rebuild is worth running."

**Evidence of the work:** `GET /api/model/audience` returns, live, `activation: false`,
`computed_at` 2026-07-29T06:41:29Z, and eight per-family held-out gate verdicts with
reasons:

| Factor family | Verdict | Held-out delta |
| --- | --- | --- |
| weekday_slot | on | +25.55 percent |
| series | on | +16.08 percent |
| competitor_lineup | on | +2.16 percent |
| calendar_religious_blackout | off | +0.057 percent, under the 2 percent bar |
| calendar_hanukkah | off | null, no Hanukkah days in the window |
| calendar_school_and_chol_hamoed | off | null, no such days in the window |
| operator_events | off | null, all 3,459 observations fall on event days |
| season | off | null, 1 of 1 season cells qualify |

`models/tv_break_coefficients.json` metadata carries the retention side of the same
discipline: `total_breaks_measured` 2,532, `pooling_method` empirical_bayes,
`between_cell_variance_tau2` 0.000173, `series_layer_active` false with
`series_gate_reason` "series RMSE (fold mean 0.26239) does not beat genre RMSE (fold mean
0.24200) by the required 2% margin", `first_break_active` false with `first_break_p_value`
0.2034 and `first_break_reason` "p=0.2034 not < 0.01; multiplier left at 1.0 (off)".

**Evidence of no surface:** there is no training route in the API. The only model path in
the live OpenAPI document is `GET /api/model/audience`. Rebuilds run from
`scripts/compute_measured_coefficients.py` (36,786 bytes) and
`scripts/compute_audience_model.py`.

**Evidence of the boundary blurring:** the gate table renders inside the Events calendar
page, an operator surface, alongside the wartime disclosure, the week-to-week drift table
with a Binding flag, the training window 2024-11-01 to 2024-11-30, and the coefficients
build timestamp (`CalendarEventsModel.jsx:5` imports `AudienceModelBlock` from
`CalendarAudienceModel.jsx`; captured at
`.playwright-mcp/people-03-calendar-training-gates.png`). An operator opening the calendar
to add a holiday is shown eight held-out gate verdicts and a drift measurement.

**Accountable for:** never shipping a coefficient the data did not earn.

**Has:** rigorous artifacts and a terminal.

**Lacks:** a home. No training dashboard, no rebuild trigger, no coverage view, no drift
history, no record of what a rebuild would change before running it, and no separation from
the operator's surfaces.

---

### 14. The deployment owner

**Status: INFERRED as a distinct person,** from the operational assumptions the code states
about itself.

**Evidence:** `scripts/init_auth.py` seeds the first admin and `kairos_api/auth.py`'s
lifecycle docstring says "the operator runs scripts/init_auth.py". Six environment knobs
control real behaviour: `KAIROS_AUTH_DIR`, `KAIROS_AUTH_DISABLED`, `KAIROS_COOKIE_SECURE`,
`KAIROS_ADMIN_PASSWORD`, `KAIROS_AUDIT_DIR`, `KAIROS_VERSIONS_DIR`. Three modules state the
single-process assumption in their own docstrings: `auth_store.py` ("a multi-worker
deployment would need a shared session store instead"), `activity_log.py`, and `jobs.py`
("This deployment is a single uvicorn worker serving one operator").

**Accountable for:** the thing being up, being on TLS, and its secrets not leaking.

**Has:** an honest bootstrap that writes a one-time password to a mode-600 file and logs
loudly when auth is bypassed (`auth_store.py:369-397`, `auth.py:124-150`).

**Lacks:** any in-product surface at all, and any awareness elsewhere that a restart signs
everyone out because sessions are an in-process dict.

---

### 15. The channel-affiliated account

**Status: EVIDENCED as an identity class, not yet a person.** This is the dimension that
carries the training versus runs line.

**Evidence:** `AFFILIATIONS = ("company", "channel")` at `auth_store.py:39`, with the
comment "Company staff manage everything; channel-affiliated accounts are walled off the
event-management surface". `is_company_user` at `auth_store.py:243-262`.
`kairos_api/events_access.py` implements the wall with two Hebrew denials,
"עריכת אירועים שמורה לצוות החברה" and "הפעלת תמחור אירועים שמורה לצוות החברה".

**Reality check:** zero channel accounts exist. The wall covers 4 of 113 operations. Every
persona from 1 through 12 above would be a `channel` account under the intended model, and
personas 13 and 14 would be `company`, but nothing in the product reflects that yet.

---

### 16. Kai, as a delegated actor

**Status: EVIDENCED.** Not a person, listed because the accountability schema already has a
non-human answer to "who changed this".

**Evidence:** `activity_log._entry` stamps `"via": "assistant"` when the path starts with
`/api/assistant` and `"dashboard"` otherwise. Measured over the 5,369 log entries: 4,509
dashboard, 860 assistant. The assistant has 31 read tools and 8 propose tools and no write
tool; `SYSTEM_PROMPT` rule 4 at `kairos_api/assistant.py:85-89` states "you never change
anything yourself. A propose_* tool only records a proposal; the operator reviews and
approves or rejects it". Applying a batch requires the writer role, snapshots a restore
point before the first mutation, and writes into the unified version timeline
(`assistant_actions.py:441-460`).

**Consequence for the people model:** Kai addresses exactly one abstract user. Its system
prompt says "the operator" or "the operator's" seven times in the first thirty lines and
never varies by role, affiliation or job. Whatever personas the rebuild lands on, the
assistant currently speaks to none of them specifically.

---

## Part 3: their day

Frozen job stories. Each has a trigger, a sequence, a done condition, a worst day, and a
target a stopwatch can measure in a browser against the running app. Targets are wall clock
from the trigger to the done condition, including page load, unless stated otherwise.
Where a target is currently impossible because the capability does not exist, that is said
plainly rather than softened.

Baseline measurements taken 2026-07-31 on `http://127.0.0.1:8010`, warm:
`/api/overview` 1.28 to 1.77s, `/api/schedule` 0.44s, `/api/break-operations` 0.45s,
`/api/campaigns` 0.54s, `/api/inventory` 0.57s, `/api/forecasts` 0.36s,
`/api/yield-per-second` 1.34s, `/api/agencies/summary` 1.78s, `/api/advertisers/stats`
3.61s.

### JS-1. The executive reads the week

- **Trigger:** first coffee, or a question from the board.
- **Sequence:** open Meridian. Read one screen.
- **Done when:** they can say out loud whether the week is on plan, whether anything is
  broken, and what needs a decision today, without clicking.
- **Worst day:** the schedule is stale (it is, right now:
  `{"status": "stale", "changed": ["settings", "coefficients"]}`), so every figure on the
  page describes a plan that no longer matches the inputs. Today the banner says so and
  offers Recompute now, which a viewer cannot run. They need to know who can, and reach
  them.
- **Target:** 5 seconds, 0 clicks, 0 scrolls to the three answers. Currently the overview
  payload alone costs 1.28 to 1.77s and "on plan" has no referent because no goal entity
  exists, so this story is not passable today at any speed.

### JS-2. The planner builds next week

- **Trigger:** Thursday, the coming week's programme lineup has landed.
- **Sequence:** choose the objective, run the optimizer, compare two scenarios on revenue
  net of retention cost, publish.
- **Done when:** a named, dated plan is published and everyone downstream is reading it.
- **Worst day:** the optimizer takes minutes instead of a second (measured maximum on
  `POST /api/optimal-plan` is 275,632 ms over 97 recorded calls) and the two scenarios come
  back close enough that the choice is not obvious. They need the difference expressed in
  money with its basis, and they need the run to be cancellable.
- **Target:** under 3 minutes end to end, every number carrying its basis. The publish step
  does not exist: publish appears 0 times in `kairos_api/`, and the weekly plan is not among
  the nine versioned logical files, so the done condition is currently unreachable.

### JS-3. The scheduler moves one break

- **Trigger:** a programme ran long, or a rep called.
- **Sequence:** open the day, drag the break, watch retention cost and revenue move as it
  lands, pin a gold break, respect the constraint without reading anything.
- **Done when:** the day is valid, the change is saved, and it can be undone.
- **Worst day:** they move the wrong break. Today there is no undo in the schedule editor
  (zero occurrences of undo in `ScheduleEditor.jsx`, `ScheduleInspector.jsx`,
  `schedule-track-view.jsx`); recovery means leaving for the Restore changes page and
  finding the right snapshot among entries labelled only `manual_edit`.
- **Target:** under 20 seconds for the move including seeing the money change, 0 dialogs,
  undo reachable in 1 keystroke.

### JS-4. The programming representative registers a restriction

- **Trigger:** the season finale airs Sunday and the last eight minutes must stay clean.
- **Sequence:** say it in their own words, see exactly which breaks that would move and what
  it costs, save.
- **Done when:** the restriction is live, attributed to them, with an end date.
- **Worst day:** the restriction costs more than anyone expected and the planner reverses it
  without telling them. Today nothing on the constraint row records who asked for it: the
  live column contract is `constraint_id, scope_type, scope_value, channel, effect,
  offset_seconds, ..., notes, where_json`, with no author, approver or expiry.
- **Target:** under 30 seconds from first keystroke to saved, with zero engine words on the
  path. Currently the path requires choosing a scope type and expressing eight minutes as
  an offset in seconds.

### JS-5. The account manager onboards a client

- **Trigger:** a signed insertion order.
- **Sequence:** create the agency, create the advertiser under it, create the campaign with
  its flights, its rebate terms and its Saturday-only surcharge discount, in one flow.
- **Done when:** all three exist, linked, visible, with no duplicate entity created
  anywhere.
- **Worst day:** the advertiser already exists under a slightly different spelling and they
  create a second one, splitting the client's history. `data/advertiser_rules.csv` has 45
  rows and the name layer distinguishes `operator`, `observed` and `unnamed` sources
  (`kairos_api/advertisers.py` docstring), which helps, but there is no merge.
- **Target:** under 2 minutes for all three, 0 duplicates. The campaign third of this story
  has no create path at all: `GET /api/campaigns` is read-only and the only write door is a
  CSV upload, so the story cannot complete today.

### JS-6. The campaign manager works the pacing board

- **Trigger:** every morning, and any time a client calls.
- **Sequence:** see every campaign on air with its pace against goal, find what is
  under-delivering, do the recommended thing.
- **Done when:** every at-risk campaign has an action taken or an explicit decision to
  accept the risk.
- **Worst day:** a campaign ends under-delivered and the make-good has to come out of next
  month's inventory. That is exactly what `GET /api/make-good-alerts` exists to prevent, and
  it currently answers `data_available: false`, "campaign_flights.csv has no campaign rows
  yet (header-only seed)."
- **Target:** under 60 seconds to find the worst-pacing campaign and see the recommended
  action on screen without deriving it. Not passable today: 0 flights exist and there is no
  way to create one in the product.

### JS-7. The traffic operator builds the pod

- **Trigger:** tomorrow's log is in, the 22:53 pod needs assembling.
- **Sequence:** pick the ads, see the pod as a physical thing with durations summing exactly
  to the break length, verify each ad's duration to the frame, format, aspect ratio and the
  presence of audio, reorder by dragging, lock it.
- **Done when:** the pod is locked, sums exactly, and every ad passed.
- **Worst day:** a 36-second file is booked as 35 and the pod overruns, or a silent master
  goes to air. The 35-versus-36 case is not hypothetical: it is sitting in position 3 of the
  real 22:53:49 pod in `data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv` today.
- **Target:** under 90 seconds for the real seven-ad pod at 22:53:49 (212 seconds total,
  House Numbers CID179035, CGB007548, CID178977, CMK022716, CID178966, CID179004,
  CID178897), with any failing ad impossible to miss. Not passable today at any speed: the
  pod does not exist as an object anywhere in the product, and no media technical field
  exists in the repository.

### JS-8. The analyst answers a money question

- **Trigger:** "which advertiser delivered the most last month, gross and net of agency
  rebates."
- **Sequence:** ask it, in the product, and read the answer.
- **Done when:** the number is on screen with its basis and the rows behind it are one click
  away, with nothing exported.
- **Worst day:** they export two CSVs, join them in a spreadsheet, and produce a number
  nobody else can reproduce.
- **Target:** under 30 seconds, 0 exports. Not passable today: per-advertiser revenue is
  `null` for all 45 advertisers with `revenue_source: "source_pending"`, and the only
  gross-versus-net figures available (`GET /api/agencies/summary`) are agency-level totals
  on a single day's file.

### JS-9. Anyone asks Kai

- **Trigger:** a question, from whatever screen they are on.
- **Sequence:** ask in natural Hebrew, see exactly what will change before it changes, apply,
  and be able to undo.
- **Done when:** the change is applied and a restore point exists.
- **Worst day:** Kai proposes something confidently that the data cannot support. The system
  prompt's grounding rule and the propose-only tool surface are designed against exactly
  this, and the 31 read tools versus 8 propose tools versus 0 write tools is the structural
  guarantee.
- **Target:** under 45 seconds from question to applied change with a restore point,
  measured on a real proposal. Recorded evidence that the path works end to end: 1 `apply`
  event in `data/assistant/audit.jsonl` and 40 distinct `POST
  /api/assistant/proposals/{batch}/apply` calls in the activity log.

### JS-10. The first day

- **Trigger:** a new account manager, no training, no documentation, nobody to ask.
- **Sequence:** complete JS-5.
- **Done when:** the agency, the advertiser and the campaign exist and are correct.
- **Worst day:** they cannot tell which of the seventeen navigation entries is theirs. Live
  nav, read from the DOM: Overview, Optimizer, Schedule, Inventory, Break Library, Campaigns,
  Forecasts, Events calendar, Reports, Data, Advertisers, Agencies, Pricing, Overrides, Kai
  AI assistant, Restore changes, Settings. Nine of those seventeen are things an account
  manager never touches, and nothing on the screen says so.
- **Target:** under 5 minutes, 0 questions asked, 0 wrong pages opened.

### JS-11. The data steward's morning

- **Trigger:** the daily ad log for tomorrow lands.
- **Sequence:** upload it, confirm it validated, confirm the engine is actually reading it.
- **Done when:** every input the engine needs is present, valid and in use.
- **Worst day:** the file uploads and validates and the engine ignores it, which is the live
  state for four of seven inputs today: programmes, spots, dayparts and rate_card all report
  `in_use: false`, with programmes carrying the honest reason that the engine prefers
  `data/reference/Programmes.xlsx`.
- **Target:** under 60 seconds from file to a green in-use state, with the shadowing
  explained in place.

### JS-12. The revenue owner changes a price

- **Trigger:** a rate-card revision takes effect Sunday.
- **Sequence:** find the layer, change the value, see what it does to the plan and to
  settlement before saving, save.
- **Done when:** the new card is live and the change is attributable and reversible.
- **Worst day:** they activate a layer whose multipliers are not 1.0 and move real revenue
  without meaning to. The product already ships position, ad type and show layers off for
  exactly this reason, with the promo multiplier of 0.00 disclosed as a hazard.
- **Target:** under 45 seconds from opening Pricing to a saved change with the delta shown
  before saving.

### JS-13. The compliance owner checks the licence

- **Trigger:** a regulator query, or a monthly review.
- **Sequence:** open the verdict, read the seven checks with observed against limit, prove
  the limits are the current ones.
- **Done when:** they can attest, with a source and a date.
- **Worst day:** somebody changed `max_ad_minutes_per_hour` in Settings and nobody told
  them. Nothing today prevents or announces that.
- **Target:** under 15 seconds to the seven checks with the source and effective date
  visible, and under 30 seconds to prove no guardrail changed since the last attestation.
  The second half is unbuildable today: guardrail edits are ordinary settings writes with
  no dedicated record.

### JS-14. The administrator adds a teammate

- **Trigger:** somebody joins.
- **Sequence:** create the account, choose the role, choose the affiliation, hand over a
  temporary password.
- **Done when:** they can sign in and are forced to change it.
- **Worst day:** they pick the wrong role and the person either cannot work or can change
  the rate card. With three roles for sixteen jobs, this is the normal case rather than the
  worst one.
- **Target:** under 30 seconds per account. This is the one story the current product
  genuinely serves: the flow exists, guards the last admin, and forces a password change.

### JS-15. The model steward decides whether to ship

- **Trigger:** a month of new data has accumulated, or a factor is suspected.
- **Sequence:** look at coverage, look at drift, run the rebuild, read every gate verdict
  with its held-out delta and its reason, decide, and record the decision.
- **Done when:** the artifact is either shipped with its verdicts or explicitly not shipped
  with the reason.
- **Worst day:** a gate passes on a window that was not representative. The current artifact
  is exactly this case, and the product says so: the whole 30-day training window
  2024-11-01 to 2024-11-30 was wartime, with the ceasefire on 2024-11-27 leaving a
  post-ceasefire tail of 132 of 2,532 measured breaks, and `operator_events` reads off
  because all 3,459 audience observations fall on event days so there is nothing to contrast
  against.
- **Target:** under 2 minutes from opening the training surface to a recorded ship or
  no-ship decision with every gate verdict and reason visible. Not passable today: there is
  no training surface, no rebuild trigger in the product, and the gate table currently
  renders inside the operator's Events calendar page.

### JS-16. The deployment owner brings it up

- **Trigger:** a new deployment, or a restart.
- **Sequence:** seed the auth store, set the TLS cookie flag, confirm enforcement is on.
- **Done when:** `GET /api/auth/me` returns `auth_disabled: false` and the startup log says
  "Auth store initialized: API requests require a signed-in session."
- **Worst day:** the escape hatch is left on in production. The code warns loudly at startup
  ("KAIROS_AUTH_DISABLED is set: API authentication is bypassed for every request. Never run
  a real deployment this way.") and `GET /api/auth/me` reports it honestly, which is exactly
  the state of the instance this investigation ran against.
- **Target:** under 3 minutes from clone to enforced login, 0 plaintext passwords persisted
  unless generated.

## Part 4: what the code believes about people, and why it is the root cause

Three beliefs are baked in, and each one produces a visible symptom.

1. **There is one user, and they do everything.** 225 uses of "the operator";
   `jobs.py` states "one operator" as a design premise; the assistant's system prompt
   addresses "the operator" exclusively. Symptom: seventeen flat navigation entries with no
   grouping principle, because with one user there is nothing to group by.
2. **Permission is about danger, not about job.** The one enforcement rule sorts operations
   into read and write. Symptom: the person who registers a programming restriction and the
   person who sets the rate card hold identical permissions, and the compliance limits are
   editable by the same click that moves a slider.
3. **Training and runs are the same activity done by the same person.** The affiliation
   dimension exists and covers 4 of 113 operations. Symptom: eight held-out gate verdicts, a
   drift table with a Binding flag, a wartime disclosure and a coefficient build timestamp
   all render inside the Events calendar, which is a run surface, while the actual training
   actions live only in a terminal.

Two consequences worth carrying into the architecture work.

- **Every persona except the administrator is currently unaddressed by the interface**, and
  four of them (traffic operator, campaign manager, model steward, analyst) have no surface
  that serves their central task at all.
- **The stores that belong to the least-served personas are the empty ones.** Constraints 0
  rows, overrides 0 rows, break decisions 0 rows, gold breaks 0, campaign flights 0 rows,
  advertiser conditions 0 rows, agency conditions 0 rows. The stores that belong to the
  single served persona, settings and pricing, carry 44 of 50 recent version snapshots. The
  usage pattern in the version store is itself evidence of who this product currently
  belongs to.

## Part 5: what still needs a human answer

These are the questions evidence cannot settle. Each blocks a design decision.

1. **How many of the sixteen are actually distinct people at this broadcaster?** Several may
   collapse in a real newsroom: planner and scheduler could be one person, revenue owner and
   analyst could be one, compliance could be a lawyer who never signs in. The evidence
   establishes sixteen distinct sets of accountability, not sixteen headcount.
2. **Is the traffic operator inside the broadcaster or at a separate playout facility?** It
   changes whether the pod surface is part of this product or an export contract.
3. **Who owns the regulatory limits?** They are currently ordinary settings. If a lawyer
   owns them, they need a different permission and a change record.
4. **What is the campaign's system of record?** If flights live in a sales system, Meridian
   needs an integration rather than a CRUD surface, and the account manager's job story
   changes shape entirely.
5. **Does the broadcaster get channel-affiliated accounts at all?** Today zero exist and
   every account defaults to company, which means the training wall is currently open to
   everyone by default rather than closed.
