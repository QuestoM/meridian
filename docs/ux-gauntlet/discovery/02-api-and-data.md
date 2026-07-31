# API and data inventory

Measured 2026-07-31 against the working tree at `5a80a709` and the browsable
instance at `http://127.0.0.1:8010` (authentication disabled). Every count in
this file was produced by running code or by hitting the live endpoint, not by
reading a docstring. Where I did not check something I say so.

## Method

- Route list: AST walk of every `kairos_api/*.py` decorator, cross-checked
  against the live `GET /openapi.json`. Both give 113.
- Callers: literal-path scan of `tv-break-dashboard/src/**` (65 `.js`/`.jsx`
  files) and `tests/**`, then hand-verified per HTTP method, because the
  dashboard composes several paths dynamically (for example
  `AgencyDetailDrawer.jsx:194`).
- Responses: every one of the 68 distinct GET paths was actually fetched. Two
  needed real parameters (`/api/constraints/effect`, `/api/overrides/effect`);
  supplied `channel=רשת 13&day=2024-11-01` from `/api/schedule/segments`.
- Stores: loaded with `~/.venvs/meridian/bin/python` + pandas, real row counts.
- No POST/PUT/PATCH/DELETE was issued. Write behaviour is described from code
  and from the on-disk artifacts those writes have already produced.
- Orphan claims were hardened against the **shipped bundle**
  (`tv-break-dashboard/dist/assets/*.js`, built 2026-07-29, 1,478,621 chars),
  which is what `127.0.0.1:8010` actually serves. It contains 70 distinct
  `/api/` literals. Diffing bundle against source, the only source-only literals
  are five comment artifacts, so the bundle is not stale and its absences are
  real.

Two caveats on the environment, recorded because they affect reproducibility.
`data/kairos_settings.json` went dirty during this session
(`locale he -> en`, `direction rtl -> ltr`, plus an added
`audience_model_activation: false`) at `2026-07-31T15:46:32Z` via
`PUT /api/settings`, and five POSTs followed between 16:17 and 16:31Z
(`/api/scenario`, `/api/optimal-plan`, `/api/scenario-compare` twice,
`/api/pricing/price-slot`). None of those were mine. Also, `activity_log.py:118`
sets `via` as a static label (`"assistant"` for `/api/assistant*`, `"dashboard"`
for everything else), so the `via` field cannot be used to tell a browser from a
script.

## Counts, corrected

| Brief said | Measured | Evidence |
|---|---|---|
| 111 endpoints | **113** | 113 route decorators via AST; `GET /openapi.json` reports 113 operations over 90 paths |
| 51 modules | **51** | `ls kairos_api/*.py` = 51; **25** declare routes, **26** are support modules |
| 17 csv/json under `data/` | **17** at top level | `ls data/*.csv data/*.json` = 17; plus 8 files in 7 subdirectories and 400 files under `data/versions/` |

The 26 route-less modules are not dead. They are the assistant's whole brain
(`assistant_tools.py`, `assistant_read_tools*.py`, `assistant_context.py`,
`assistant_keywords.py`, `assistant_propose*.py`, `assistant_conversations.py`,
`assistant_memory.py` writes only), plus `core.py` (698 lines, the shared
loaders and the `KairosSettings` model), `auth_store.py`, `jobs.py`,
`events_access.py`, `events_holidays.py`, `condition_validation.py`,
`_constraint_options.py`, `audience_api.py`, `assistant_simulate.py` and
`server.py`.

Note `audience_api.py` (114 lines) declares no route at all. The endpoint
`GET /api/model/audience` lives in `insights_api.py:630` and calls into it.

---

## 1. Endpoint inventory

Legend for the caller column: a `file:line` means a real fetch call in the
dashboard. `TESTS` means tests only. `NOBODY` means no caller anywhere in the
frontend, the tests, the scripts or the assistant.

### 1.1 The plan and its money (dashboard_api.py, catalog_api.py, insights_api.py)

| Method + path | Module:line | What it actually returned | Caller | Duplicate of |
|---|---|---|---|---|
| GET /api/overview | dashboard_api.py:1638 | 14,951 B, keys `brand, workspace, data_freshness, summary, source_counts, recommendations, frontier_scope, settings, compliance, frontier, frontier_status, frontier_net_point, frontier_basis, schedule_freshness`. `settings` is the **full 33-field** model. 5 recommendations, 6 frontier points. 2,969 ms | TVBreakDashboard.jsx:1287, :1359; FrontierScopeChart.jsx:127 | carries `settings` (= /api/settings) and `compliance` (= /api/compliance) verbatim |
| GET /api/compliance | dashboard_api.py:1633 | 1,822 B, `profile, effective_date, source_url, checks, violations, status, disclaimer` | TVBreakDashboard.jsx:1158 | byte-identical block inside /api/overview |
| GET /api/schedule | dashboard_api.py:1695 | 516,470 B. `rows` = 4 channel canvases, `break_operations` = the whole /api/break-operations payload nested, `break_schedule` = **first 200 rows only**, `break_schedule_total_rows` = 8704 | TVBreakDashboard.jsx:1288 | nests /api/break-operations whole |
| GET /api/schedule/segments | dashboard_api.py:1711 | 610,149 B, 2,540 segments, all `רשת 13`, each `{segment_id, channel, day, anchor{date,start_clock,program}, state{num_breaks,is_gold,predicted_revenue,retention}}` | OverrideConsole.jsx:48; schedule-track-view.jsx:40 | same plan as /api/schedule, different projection and different scope |
| GET /api/schedule/segment/{id} | dashboard_api.py:1728 | `segment_id, found, owned_channel, anchor, identity, plan, economics, retention, overrides` for `2024-11-01\|רשת 13\|000` | ScheduleInspector.jsx:61 | the per-row expansion of /api/schedule/segments |
| GET /api/break-operations | dashboard_api.py:1803 | 33,949 B. 48 programs, 30 breaks, one date (2024-11-01), **all four channels, 12 programs each** | TVBreakDashboard.jsx:1297 | also returned whole inside /api/schedule |
| GET /api/break-library | catalog_api.py:591 | 41,874 B, 80 breaks, all `רשת 13`, 28 distinct dates, adds `priority` and `status` to the plan columns | TVBreakDashboard.jsx:1290 | third projection of `weekly_break_schedule.csv` |
| GET /api/break-decisions | dashboard_api.py:1817 | `{"decisions": []}`. Derived from `manual_overrides.csv` rows with `source=recommendation` or `status=dismissed` | **NOBODY** (the dashboard only POSTs) | a filtered view of /api/overrides |
| POST /api/break-decisions | dashboard_api.py:1823 | Writes an Override row | TVBreakDashboard.jsx:1223 | third write path into `manual_overrides.csv` |
| GET /api/impact | catalog_api.py:571 | 8,257 B, `coefficient_impacts` (source `measured_detrended_pooled`, 2,532 breaks, 36 cells) and `drift` (5 weekly levels) | TVBreakDashboard.jsx:1295 | none |
| GET /api/inventory | catalog_api.py:578 | 1,940 B. `summary{spots:18669, revenue:null, seconds:358096}`, `revenue_available:false`, `scope_channel: רשת 13`, 5 dayparts, 24 hours | TVBreakDashboard.jsx:1289 | none |
| GET /api/campaigns | catalog_api.py:596 | 8,505 B, 50 campaigns. **Every `advertiser_id` is `""` and every `revenue` is `null`** | TVBreakDashboard.jsx:1291 | the only "campaign" read; unrelated to `campaign_flights.csv` |
| GET /api/forecasts | catalog_api.py:605 | 3,452 B, `by_day` (per-date revenue/retention/breaks), `scenarios`, `by_day_basis` | TVBreakDashboard.jsx:1179, :1292 | same optimizer sweep as /api/overview.frontier |
| GET /api/reports | catalog_api.py:620 | 496 B, 5 reports, `weekly-plan` reports `rows: 8704` | TVBreakDashboard.jsx:1293 | none |
| GET /api/files | catalog_api.py:641 | 885 B, 8 file records (`path, exists, size, modified`) | TVBreakDashboard.jsx:1294 | overlaps /api/uploads/status, which reports the same files with more truth |
| GET /api/yield-per-second | insights_api.py:400 | 4,123 B, `available, revenue_net_available, basis{formula,inputs,source}, revenue_net_ils, retention_cost_ils, ...` | YieldView.jsx:56; MoneyWaterfall.jsx:167 | shares the retention-cost model with /api/optimizer/net-comparison |
| GET /api/gold-breaks | insights_api.py:518 | 169 B, `count: 0`, reason "No gold breaks in the current plan" | GoldBreakManager.jsx:32 | a filter over the plan's `is_gold` column, also present in /api/schedule/segments |
| GET /api/make-good-alerts | insights_api.py:646 | 134 B, `alerts: []`, `data_available: false`, reason "campaign_flights.csv has no campaign rows yet (header-only seed)" | MakeGoodAlerts.jsx:40 | none |
| GET /api/model/audience | insights_api.py:630 | 2,334 B, `available, computed_at, activation:false, gates{...}, base_summary` | CalendarAudienceModel.jsx:42 | overlaps /api/impact (both are model provenance) |
| POST /api/scenario-compare | insights_api.py:622 | Two optimizer runs at two revenue weights | ScenarioCompare.jsx:91 | fourth optimizer entry point |
| GET /api/export/schedule.csv | exporters.py:107 | 1,217,866 B, 8,704 rows, 21 columns, **all four channels** (`קשת 12` 2727, `רשת 13` 2540, `כאן 11` 2169, `עכשיו 14` 1268) | ScheduleInspector.jsx:190; TVBreakDashboard.jsx:1047 | the unscoped truth behind every scoped view above |
| GET /api/export/spots.csv | exporters.py:244 | 41,114 B, 175 rows, 20 columns, `status` = priced 119 / dropped_frequency 56 | TVBreakDashboard.jsx:1083 | the only place per-advertiser money exists |

### 1.2 The optimizer (scenario_api.py, recompute_api.py)

| Method + path | Module:line | Returned | Caller | Duplicate |
|---|---|---|---|---|
| GET /api/optimizer-plan | scenario_api.py:182 | 53,019 B in 6,112 ms. `channel, day, summary, placements, segments, violations, decisions, weights, controls, guardrails, impact_source, coefficient_freshness` | **NOBODY** | the GET twin of the POST below |
| POST /api/optimizer-plan | scenario_api.py:199 | Same builder with a `ScenarioRequest` | TVBreakDashboard.jsx:1867 | |
| POST /api/scenario | scenario_api.py:229 | One-day optimization for the scenario controls | **TESTS only** (7 test hits) | same engine as POST /api/optimizer-plan |
| POST /api/optimal-plan | scenario_api.py:282 | `optimize_day_plan(...)` under the saved settings | **TESTS only** (5 test hits) | third form of the same run |
| GET /api/optimizer/net-comparison | scenario_api.py:330 | 551 B, `status, basis, current, net_focused, delta` | MoneyWaterfall.jsx:269 | runs the same scenario runner twice |
| GET /api/parameters | scenario_api.py:388 | 2,447 B. `settings` (full 33 fields), `flights_count: 0`, `guardrails`, `assumptions`, `channels`, `operator_channel`, `available_channels`, `pricing`, `coefficient_freshness`, `first_break_active:false` | TVBreakDashboard.jsx:1296 | third carrier of the settings document |
| POST /api/recompute-schedule | recompute_api.py:75 | Synchronous rebuild of `output/weekly_break_schedule.csv`, 409 if a job is running | TVBreakDashboard.jsx:1952 | same work as the job below |
| POST /api/jobs/recompute | recompute_api.py:114 | `{job_id, already_running}`, optional `scope` | TVBreakDashboard.jsx:1946; override-console-lib.js:28 | |
| GET /api/jobs/{job_id} | recompute_api.py:156 | Job record (in-memory, `jobs.py`) | TVBreakDashboard.jsx:1963; override-console-lib.js:38 | none |

### 1.3 Operator rules (constraints.py, overrides.py)

| Method + path | Module:line | Returned | Caller | Duplicate |
|---|---|---|---|---|
| GET /api/constraints | constraints.py:208 | `{"constraints": [], "columns": [15 names]}` | ConstraintBuilder.jsx:492 | |
| POST /api/constraints | constraints.py:218 | Creates a constraint | ConstraintBuilder.jsx:517; ScheduleEditor.jsx:264 | |
| PUT /api/constraints/{id} | constraints.py:255 | Updates one | **NOBODY** (4 test hits) | |
| DELETE /api/constraints/{id} | constraints.py:283 | Deletes one | ConstraintBuilder.jsx:543 | |
| GET /api/constraints/options | constraints.py:294 | 20,059 B, `scope_types, effects, programmes, genres, channels, weekdays, dayparts, predicate_fields, operator_channel, available_channels` | ConstraintBuilder.jsx:474 | near-twin of /api/advertisers/options |
| GET /api/constraints/effect | constraints.py:333 | 9.4 s. `channel, day, summary{before_total_breaks:80, after_total_breaks:80, before_revenue, after_revenue, changed_segments:0, matched_segments:0}, changed, skipped_constraints, rejected_overrides` | **NOBODY** (4 test hits) | **see /api/overrides/effect** |
| GET /api/overrides | overrides.py:161 | `{"overrides": {"segment": [], "spot": []}, "columns": [14 names]}` | OverrideConsole.jsx:33 | |
| POST /api/overrides | overrides.py:172 | Creates an override | OverrideConsole.jsx:164; ScheduleInspector.jsx:133 | also reachable via POST /api/break-decisions |
| PUT /api/overrides/{id} | overrides.py:208 | Updates one | **NOBODY** (9 test hits) | |
| DELETE /api/overrides/{id} | overrides.py:233 | Deletes one | OverrideConsole.jsx:209; ScheduleInspector.jsx:151 | |
| GET /api/overrides/effect | overrides.py:338 | 11.3 s. `channel, day, candidate, summary{before_total_breaks:80, after_total_breaks:80, ...changed_segments:0}, changed, rejected_overrides` | OverrideConsole.jsx:107; ScheduleInspector.jsx:84 | **the same response shape as /api/constraints/effect**, minus `matched_segments`/`skipped_constraints`, plus `candidate` |

### 1.4 Advertisers and agencies

| Method + path | Module:line | Returned | Caller | Duplicate |
|---|---|---|---|---|
| GET /api/advertisers | advertisers.py:211 | 10,652 B, 45 rows, columns `advertiser_id, default_premium, allow_positions, allow_genres, prime_time_only, urgency_k, ahead_k, notes, display_name`. Every row carries **embedded `conditions` and `overlaps` arrays** | AdvertisersManager.jsx:96 | embeds the conditions endpoint |
| POST /api/advertisers | advertisers.py:329 | Creates a rule row | AdvertisersManager.jsx:158 | |
| PUT /api/advertisers/{id} | advertisers.py:293 | Updates | AdvertisersManager.jsx:134 | |
| DELETE /api/advertisers/{id} | advertisers.py:353 | Deletes | AdvertisersManager.jsx:200 | |
| GET /api/advertisers/stats | advertisers.py:229 | 13,707 B, 45 rows. Every row: `display_name:""`, `name_source:"unnamed"`, `rule_count:0`, `revenue:null`, `revenue_source:"source_pending"` | AdvertisersManager.jsx:77 | recomputes over the same store as /api/advertisers |
| GET /api/advertisers/options | advertiser_conditions.py:252 | 19,160 B, `positions, genres, dayparts, programmes, weekdays, effects, modes` | AdvertisersManager.jsx:58; AgencyManager.jsx:124 | near-twin of /api/constraints/options |
| GET /api/advertisers/{id}/conditions | advertiser_conditions.py:273 | `{"conditions": [], "overlaps": []}` | **NOBODY** (the list endpoint already embeds this; 10 test hits) | |
| POST /api/advertisers/{id}/conditions | advertiser_conditions.py:278 | Creates | AdvertisersManager.jsx:216 | |
| PUT /api/advertisers/{id}/conditions/{rule_id} | advertiser_conditions.py:326 | Updates | AdvertisersManager.jsx:236 | |
| DELETE /api/advertisers/{id}/conditions/{rule_id} | advertiser_conditions.py:359 | Deletes | AdvertisersManager.jsx:256 | |
| GET /api/agencies | agencies.py:268 | 8,340 B, 9 agencies (`AGY_01`..`AGY_09`), 24 columns, plus a `boundary` disclosure string | AgencyManager.jsx:144 | |
| POST /api/agencies | agencies.py:347 | Creates | **NOBODY**: there is no add-agency UI (`grep -rn "createAgency\|addAgency\|AddAgency" tv-break-dashboard/src` returns nothing, and `AgencyManager.jsx`/`AgencyDetailDrawer.jsx` contain exactly one `method:` literal, `'PUT'`) | |
| GET /api/agencies/summary | agencies.py:285 | 523 B. `available:true, gross_revenue:699450.0, net_revenue:669978.0, rebate_total:29472.0, spot_count:119, basis:"Wally_Prime_Reshet_Example_2025-04-27.csv"` | AgencyManager.jsx:160 | the only aggregate that resolves real money to a commercial party |
| GET /api/agencies/{id} | agencies.py:334 | Full record + `conditions`, `overlaps`, `links{advertiser_count:6, manual_count:0}`, `boundary` | **NOBODY** (drawer uses the row from the list) | |
| PUT /api/agencies/{id} | agencies.py:364 | Updates, including `status` | AgencyDetailDrawer.jsx:156 | |
| POST /api/agencies/{id}/deactivate | agencies.py:381 | Marks suspended | **NOBODY** | the UI suspends by `PUT ... {status:'suspended'}` at AgencyDetailDrawer.jsx:181 |
| GET /api/agencies/{id}/advertisers | agency_conditions.py:309 | `observed` (6 Hebrew names), `manual` [], `effective` (6), `observed_source_file` | AgencyDetailDrawer.jsx:121 | |
| POST /api/agencies/{id}/advertisers | agency_conditions.py:315 | Creates a manual link | **NOBODY** | the drawer renders links read-only |
| DELETE /api/agencies/{id}/advertisers/{adv} | agency_conditions.py:336 | Removes a manual link | **NOBODY** | |
| GET /api/agencies/{id}/conditions | agency_conditions.py:354 | `{"conditions": [], "overlaps": [], "cross_level": []}` | AgencyDetailDrawer.jsx:133 | note the asymmetry: advertisers embed, agencies fetch |
| POST /api/agencies/{id}/conditions | agency_conditions.py:364 | Creates | AgencyDetailDrawer.jsx:206 | |
| PUT /api/agencies/{id}/conditions/{rule_id} | agency_conditions.py:412 | Updates | AgencyDetailDrawer.jsx:218 (composed path) | |
| DELETE /api/agencies/{id}/conditions/{rule_id} | agency_conditions.py:441 | Deletes | AgencyDetailDrawer.jsx:230 (composed path) | |

### 1.5 Settings, pricing, events, uploads, versions, auth, activity

| Method + path | Module:line | Returned | Caller | Duplicate |
|---|---|---|---|---|
| GET /api/health | settings_api.py:29 | 118 B, `status, project, timestamp, has_schedule, has_model` | **NOBODY** (3 test hits) | |
| GET /api/settings | settings_api.py:41 | 1,053 B, exactly the 33 keys in `data/kairos_settings.json`, 1:1 | **NOBODY** | the same document also ships in /api/overview and /api/parameters |
| PUT /api/settings | settings_api.py:46 | Snapshots then saves | TVBreakDashboard.jsx:1890 | also writes `pricing_overrides`, which PUT /api/pricing owns |
| GET /api/settings/controls | settings_api.py:54 | 4,253 B, `levers, templates, current` (the bilingual lever schema) | **NOBODY** (comment at TVBreakDashboard.jsx:5454 says the panel should be driven by it; the panel hardcodes instead) | |
| GET /api/pricing | pricing_api.py:213 | 2,105 B. `base{value:60.0, overridden:false}`, `activation{show:false, position:false, ad_type:false}`, `events{enabled:true, active_event_count:0}`, 6 layers, `has_overrides:true` | PricingManager.jsx:23; CalendarEvents.jsx:82 | |
| PUT /api/pricing | pricing_api.py:220 | Deep-merges into `settings.pricing_overrides` and calls the same `_save_settings` | PricingManager.jsx:43 | writes a sub-document of the settings file |
| POST /api/pricing/price-slot | pricing_api.py:275 | Priced slot with layer breakdown | PricingSlotTester.jsx:56; AdvertiserPricingSummary.jsx:305 | |
| GET /api/events | events_api.py:357 | 28,159 B. 63 events, 56 holidays, `model_context{training_window, weekday_premiums, measurement, wartime_disclosure, training_gate}`, `can_edit:true` | CalendarEvents.jsx:57; CalendarEventsModel.jsx:86 | |
| POST /api/events | events_api.py:376 | Creates | CalendarEvents.jsx:146, :189 | |
| PUT /api/events/{id} | events_api.py:396 | Updates, including `active` | CalendarEvents.jsx:144, :164 | |
| DELETE /api/events/{id} | events_api.py:424 | Deletes | **NOBODY** (the UI deactivates with PUT `active:false`) | |
| GET /api/uploads/status | uploads.py:552 | 6,524 B in 9,645 ms. 7 input kinds with `exists, rows, valid, in_use, in_use_reason, engine_reads, last_validation, warnings` | UploadCenter.jsx:210 | overlaps /api/files |
| POST /api/uploads/{kind} | uploads.py:627 | Validated ingest | UploadCenter.jsx:81 | |
| GET /api/versions | version_store.py:420 | 9,085 B in **27,480 ms** (slowest endpoint measured). `entries`, `note` | AssistantVersions.jsx:304 | |
| GET /api/versions/{id}/diff | version_store.py:429 | `version_id, created_at, source, diff{settings{changed:[{field,from,to}]}}` | AssistantVersions.jsx:173 | |
| POST /api/versions/{id}/restore | version_store.py:445 | Restores selected logical files, snapshots first | AssistantVersions.jsx:328 | one of three undo systems |
| POST /api/versions/snapshot | version_store.py:468 | Named snapshot of all 9 logical files | AssistantVersions.jsx:346 | |
| PATCH /api/versions/{id} | version_store.py:479 | Renames | AssistantVersions.jsx:318 | |
| GET /api/activity-log | activity_log.py:262 | 17,807 B, `entries, scope` | TVBreakDashboard.jsx:5312 | second audit log (see 5.7) |
| POST /api/auth/login | auth.py:205 | Session cookie | Login.jsx:39 | |
| POST /api/auth/logout | auth.py:241 | | Login.jsx:43 | |
| GET /api/auth/me | auth.py:251 | 22 B: `{"auth_disabled": true}` on this instance | Login.jsx:35 | |
| POST /api/auth/change-password | auth.py:265 | | Login.jsx:47 | |
| GET /api/auth/users | auth.py:284 | 401 on this instance (admin-gated even with auth disabled) | Login.jsx:54 | |
| POST /api/auth/users | auth.py:290 | | Login.jsx:58 | |
| DELETE /api/auth/users/{u} | auth.py:309 | | Login.jsx:62 | |
| PUT /api/auth/users/{u}/affiliation | auth.py:325 | | Login.jsx:73 | |
| POST /api/auth/users/{u}/reset-password | auth.py:341 | | Login.jsx:66 | |

### 1.6 The assistant

The assistant surface has 19 HTTP endpoints and, separately, **39 in-process
tools** (31 read, 8 propose) loaded from `assistant_tools.py:295-296`. The tools
are Python functions, not HTTP calls. I grepped `assistant_read_tools.py`,
`assistant_read_tools_extra.py` and `assistant_read_tools_catalog.py` for
`httpx`, `requests.`, `http://` and `TestClient`: zero hits. So the assistant
reads the same stores the API reads, through a **second, parallel accessor
layer** with its own provenance vocabulary (`SOURCE_BY_TOOL`,
`assistant_read_tools.py:356`).

| Method + path | Module:line | Returned | Caller |
|---|---|---|---|
| GET /api/assistant/status | assistant.py:616 | `available, reason, model, action_plane, auth` | AssistantPanel.jsx:61 |
| POST /api/assistant/ask | assistant.py:733 | Answer + tool trace | AssistantPanel.jsx:162 |
| POST /api/assistant/ask/stream | assistant_stream.py:55 | SSE | assistant-stream.js:55; AssistantConversationsApi.jsx:43 |
| GET /api/assistant/thread | assistant_memory.py:123 | `entries, user, conversation_id` | assistant-panel-state.js:159 |
| DELETE /api/assistant/thread | assistant_memory.py:148 | Clears | AssistantPanel.jsx:92 |
| GET /api/assistant/conversations | assistant_conversations_api.py:51 | `conversations, user` | AssistantConversationsApi.jsx:134 |
| POST /api/assistant/conversations | :58 | Creates | :164 |
| PATCH /api/assistant/conversations/{id} | :65 | Renames | :181 |
| DELETE /api/assistant/conversations/{id} | :75 | Deletes | :194 |
| GET /api/assistant/conversations/{id}/changes | :107 | Per-conversation change list | :28 |
| POST /api/assistant/conversations/{id}/restore | :183 | Reverts that conversation's changes | :32 |
| GET /api/assistant/proposals | assistant_actions.py:179 | 1,286 B, `batches` (1 resolved batch on disk) | assistant-panel-state.js:74 |
| POST /api/assistant/proposals/{batch}/apply | :487 | Applies items | assistant-panel-state.js:89 |
| POST /api/assistant/proposals/{batch}/reject | :493 | Rejects | assistant-panel-state.js:116 |
| GET /api/assistant/audit | assistant_actions.py:100 | 5,175 B, `entries` (17 rows) | **NOBODY** |
| GET /api/assistant/restore | assistant_actions.py:264 | 292 B, `restore_points` | **NOBODY** |
| POST /api/assistant/restore/{id} | assistant_actions.py:271 | Restores a pre-apply snapshot | **NOBODY** (3 test hits) |
| POST /api/assistant/upload | assistant_uploads.py:190 | Parses in memory, stores a summary | AssistantUpload.jsx:61 |
| GET /api/assistant/uploads | :229 | `uploads, count, user` | AssistantUpload.jsx:45 |
| DELETE /api/assistant/uploads/{id} | :237 | Deletes | AssistantUpload.jsx:85 |

---

## 2. Orphaned endpoints

20 of 113 (18 percent) have no caller in the shipped dashboard.

**Never called by anything, anywhere, not even a test (5):**

1. `POST /api/agencies` (agencies.py:347). **The operator cannot create an
   agency.** No add-agency form exists in the dashboard, and no test issues the
   POST (`tests/test_qa6_agencies.py` touches agencies only through direct
   Python calls and one route-ordering assertion at line 352). The 9 rows in
   `agencies.csv` all carry `data_source: "synthetic"`.
2. `POST /api/agencies/{agency_id}/deactivate` (agencies.py:381). The handler
   function is unit-tested directly (`test_qa6_agencies.py:96` calls
   `ag.deactivate_agency`) but the HTTP route is never requested. The UI suspends
   an agency with `PUT /api/agencies/{id}` carrying `status:'suspended'`
   (AgencyDetailDrawer.jsx:181). Two ways to do the same thing, one of them dead.
3. `POST /api/agencies/{agency_id}/advertisers` (agency_conditions.py:315). The
   manual-link write. No UI can create a manual advertiser-agency link, so
   `agency_advertisers.csv` is 41/41 `source=observed`.
4. `DELETE /api/agencies/{agency_id}/advertisers/{advertiser}` (:336). Same.
5. `GET /api/assistant/restore` (assistant_actions.py:264). Returns real restore
   points (one on disk, `8418ebe254ad`); nothing lists them.

**No dashboard caller, tests only (7):**

6. `GET /api/health` (settings_api.py:29)
7. `GET /api/constraints/effect` (constraints.py:333). The constraint preview
   exists and works (9.4 s, real before/after numbers) but no screen shows it,
   while the near-identical override preview is wired into two screens.
8. `PUT /api/constraints/{constraint_id}` (constraints.py:255). The builder can
   create and delete but not edit.
9. `PUT /api/overrides/{override_id}` (overrides.py:208). Same: create and
   delete only.
10. `POST /api/scenario` (scenario_api.py:229)
11. `POST /api/optimal-plan` (scenario_api.py:282)
12. `POST /api/assistant/restore/{restore_id}` (assistant_actions.py:271)

**Superseded by an embedding or a projection (8):**

13. `GET /api/settings` (settings_api.py:41). The dashboard reads settings from
    `overview.settings` (TVBreakDashboard.jsx:1415, :1614) and
    `parameters.settings` (:4190) and writes to `PUT /api/settings`. I verified
    the round trip is safe: `overview.settings` is the full 33-field model, and a
    dry-run `KairosSettings(**overview_settings).model_dump()` differs from the
    live settings in **0 fields**. So this is redundancy, not data loss.
14. `GET /api/settings/controls` (settings_api.py:54). 4,253 B of bilingual lever
    schema with help text and bounds, built precisely so the panel would not
    hardcode. TVBreakDashboard.jsx:5454 has a comment claiming it stays in sync
    with this endpoint; there is no fetch.
15. `GET /api/optimizer-plan` (scenario_api.py:182). 53 KB, 6.1 s, the richest
    single payload in the API (`placements`, `segments`, `violations`,
    `decisions`, `guardrails`, `coefficient_freshness`). Only the POST twin is
    called.
16. `GET /api/agencies/{agency_id}` (agencies.py:334). The drawer reuses the list
    row and separately fetches links and conditions.
17. `GET /api/advertisers/{id}/conditions` (advertiser_conditions.py:273). The
    list endpoint already embeds `conditions` and `overlaps` per row.
18. `GET /api/break-decisions` (dashboard_api.py:1817). Write-only in practice.
19. `DELETE /api/events/{event_id}` (events_api.py:424). Calendar deactivates
    with `PUT ... active:false` (CalendarEvents.jsx:164).
20. `GET /api/assistant/audit` (assistant_actions.py:100).

**Bundle-level confirmation.** Seven of the twenty do not appear anywhere in the
shipped JavaScript, not even as a substring: `/api/health`,
`/api/settings/controls`, `/api/scenario` (bare, as opposed to
`/api/scenario-compare`), `/api/optimal-plan`, `/api/constraints/effect`,
`/api/assistant/restore`, `/api/assistant/audit`. The remaining thirteen are
method-level or dynamic-composition orphans whose path stem is present (for
example `/api/agencies/` covers both the live conditions writes and the dead
link writes), so those were verified from source per method.

---

## 3. Data store inventory

### 3.1 Top-level stores in `data/` (the 17)

| File | Rows today | Entity | Read by | Written by | Overlaps |
|---|---|---|---|---|---|
| `Spots.csv` | 50,386 x 36 cols | Historical aired spot | **nothing on the engine path** | POST /api/uploads/spots | `reference/Spots.xlsx` is the file the loaders actually read |
| `Programmes.csv` | 3,562 x 16 | Programme airing | shadowed by `reference/Programmes.xlsx` | POST /api/uploads/programmes | ditto |
| `Dayparts.csv` | 43,200 x 14 | Minute-level TVR matrix | shadowed by `reference/Dayparts.xlsx` | POST /api/uploads/dayparts | ditto |
| `Spots - inventory.csv` | 994 x 34 | Booked spot demand | `kairos/optimize/inventory.py:46`, `schedule_freshness.py` | nothing | a third `Spots` shape |
| `Spots - sample1day.csv` | 496 x 34 | Sample | `run_optimization.py` (legacy CLI) only | nothing | fourth `Spots` shape |
| `Programmes - today.csv` | 125 x 17 | Programme airing | **nothing** (repo-wide grep, 0 hits) | nothing | second `Programmes` shape |
| `advertiser_rules.csv` | **45** x 8 | Advertiser baseline rule | `advertisers.py`, `advertiser_conditions.py`, `catalog_api.py`, `uploads.py`, `kairos/optimize/advertiser_rules.py:75` | PUT/POST/DELETE /api/advertisers, POST /api/uploads/advertiser_rules, assistant `propose_advertiser_change` | keyed `ADV_01..ADV_45`; see section 6 |
| `advertiser_conditions.csv` | **0** (header only) | Scoped advertiser rule | `advertiser_conditions.py` | the 3 condition endpoints, assistant | same grammar as `agency_conditions.csv` and `kairos_constraints.csv` |
| `agencies.csv` | 9 x 24 | Agency | `agencies.py` | POST/PUT /api/agencies | |
| `agency_advertisers.csv` | 41 x 5 | Agency-advertiser link | `advertisers.py`, `agency_conditions.py` | the 2 orphaned link endpoints; in practice regenerated as `source=observed` from the daily file | |
| `agency_conditions.csv` | **0** (header only) | Scoped agency rule | `agency_conditions.py` | the 3 agency-condition endpoints | duplicate grammar of `advertiser_conditions.csv` plus a `mode` column |
| `calendar_events.csv` | **63** x 9 | Calendar event | `events_api.py`, `pricing_api.py`, `assistant_event_pipeline.py` | POST/PUT/DELETE /api/events, assistant `propose_event_change` | 56 of 63 are imported holidays; `holidays` is also served separately from `kairos/config/israel_holidays.csv` |
| `campaign_flights.csv` | **0** (header only) | Campaign flight and delivery target | `core.py`, `insights_api.py`, `uploads.py`, `kairos/optimize/pacing.py` | POST /api/uploads/campaign_flights | "campaign" also exists as free text in `Spots.csv` (478 values) and in the daily file |
| `frequency_rules.csv` | **1** x 12 | Frequency cap | **no kairos_api module references it** | nothing | a fourth rule grammar |
| `manual_overrides.csv` | **0** (header only) | Manual override with a semantic anchor | `overrides.py`, `dashboard_api.py`, `kairos/optimize/overrides.py` | POST/PUT/DELETE /api/overrides, POST /api/break-decisions, assistant `propose_override` | overlaps `kairos_constraints.csv` |
| `rate_card_premiums.csv` | 96 x 11 | Rate card by channel-hour | `catalog_api.py`, `uploads.py` | POST /api/uploads/rate_card | **the pricing engine reads `config/optimization_weights.yaml` instead**, per `uploads.py:94-96` |
| `kairos_settings.json` | 33 keys | Operator settings + `pricing_overrides` | `core.py`, `_constraint_options.py`, everything downstream | PUT /api/settings, **PUT /api/pricing**, version restore, assistant `propose_settings_change` | |

`data/kairos_constraints.csv` is referenced by `constraints.py:58`,
`kairos/optimize/_constraints_io.py:20`, `kairos/export/schedule.py:156` and
`schedule_freshness.py:185`, and **does not exist**. `GET /api/constraints`
returns `[]` and the plan sidecar records `"constraints": "absent"`. A backup of
a former copy survives at `data/_backups/kairos_constraints_20260704T222759.csv`.

### 3.2 Subdirectories

| Path | Contents measured | Entity | Read/write |
|---|---|---|---|
| `data/reference/` | `Spots.xlsx` 3.4 MB, `Programmes.xlsx` 532 KB, `Dayparts.xlsx` 1.8 MB, `AdvertiserAgreements.csv` **0 rows**, `prime_time_programs.csv` **0 rows** | the files the loaders actually read | `kairos/data/loaders.py:198,211`; upload targets at `uploads.py:81` |
| `data/enriched/` | `Spots_enriched.csv` 50,386 x 37 (17.4 MB), `Programmes_enriched.csv` 3,562 x 19, `Dayparts_enriched.csv` 39,600 x 7. **19.7 MB, 93,548 rows** | derived tables | **read by nothing**: repo-wide grep for each filename returns 0 files |
| `data/daily_input/` | 1 file, `Wally_Prime_Reshet_Example_2025-04-27.csv`, 175 rows x 18 Hebrew columns | the daily ad log | `kairos/data/loaders.load_daily_input`, `kairos/export/spots.py`; POST /api/uploads/daily |
| `data/auth/` | `users.json`, 1 user (`admin`, scrypt) | Operator account | `auth_store.py` |
| `data/audit/` | `activity.jsonl`, **5,370 lines** | HTTP mutation audit | `activity_log.py`; GET /api/activity-log |
| `data/assistant/` | `audit.jsonl` 17 lines, `proposals.json` 1 batch, `threads/admin/` 2 files, `uploads/admin/` 2 files (62 KB each), `restore/8418ebe254ad/` 2 files | assistant state | `assistant_actions.py`, `assistant_conversations.py`, `assistant_uploads.py` |
| `data/versions/` | **200 directories, 400 files**. Captured logical files: settings 110, conditions 86, constraints 2, advertisers 2 | Undo point | `version_store.py` |
| `data/_backups/` | 5 files (advertiser_rules x2, manual_overrides x2, kairos_constraints x1) | pre-write copies | written by the CRUD modules, read by nobody |

### 3.3 Stores outside `data/` that the API depends on

| Path | Measured | Read by |
|---|---|---|
| `output/weekly_break_schedule.csv` | **8,704 rows x 21 cols**, 1.2 MB | 9 API modules (`assistant, catalog_api, core, dashboard_api, exporters, insights_api, recompute_api, scenario_api, server`) |
| `output/weekly_break_schedule.csv.meta.json` | `computed_at 2026-07-28T08:38:38Z` + 9 input fingerprints; `constraints: "absent"`, `classifications: "absent"` | `schedule_freshness.py` |
| `output/run_log.jsonl` | **489 rows**, keys `run_id, channel, day, engine_version, segment_count, summary, guardrails, assumptions, input_checksums` | **only** the assistant tool `get_run_log_summary` (`assistant_read_tools_extra.py:123`). No HTTP endpoint, no screen |
| `output/upload_validation_reports.json` | last validation per kind | `uploads.py:53` |
| `models/tv_break_coefficients.json` | 21 KB | `catalog_api, events_api, insights_api, scenario_api` |
| `models/audience_model.json` | 121 KB | `assistant_audience_model, audience_api, core` |
| `models/tv_break_posterior.pkl` | 1.2 MB | existence check in `GET /api/health` |
| `config/optimization_weights.yaml` | 3.8 KB | the real rate card |

---

## 4. Concepts that exist twice (or more)

### 4.1 The plan, in five incompatible projections

One store, `output/weekly_break_schedule.csv`, 8,704 rows. Five reads, three
different scoping rules, measured:

| Endpoint | Rows returned | Channel scope |
|---|---|---|
| `GET /api/export/schedule.csv` | 8,704 | all four (`קשת 12` 2727, `רשת 13` 2540, `כאן 11` 2169, `עכשיו 14` 1268) |
| `GET /api/schedule` -> `break_schedule` | 200 (`head(200)`, `dashboard_api.py:1503`) | whatever falls in the first 200 CSV rows: **`קשת 12` 96, `כאן 11` 73, `עכשיו 14` 28, `רשת 13` 3** |
| `GET /api/schedule/segments` | 2,540 | `רשת 13` only |
| `GET /api/break-library` | 80 | `רשת 13` only |
| `GET /api/break-operations` | 48 programs, 30 breaks | **all four channels, one date**, 12 programs each |

The operator owns `רשת 13` (`settings.operator_channel`). The screen labelled
"Schedule control" therefore shows a table that is 98.5 percent competitor rows,
and the operations board shows three competitor lanes beside the operator's own.

### 4.2 Two identical effect previews

`GET /api/constraints/effect` and `GET /api/overrides/effect` both run
`kairos.optimize.day_core._optimize_one_day` twice on the same channel-day and
diff the result. Measured on `רשת 13 / 2024-11-01`, both returned
`before_total_breaks: 80, after_total_breaks: 80, before_revenue: 1067845.55,
after_revenue: 1067845.55, changed_segments: 0` and both took roughly 10 s. The
only differences are `matched_segments` + `skipped_constraints` (constraints
side) and `candidate` (overrides side). One is wired into two screens; the other
is wired into none.

### 4.3 Four ways to run the optimizer

`POST /api/optimizer-plan`, `POST /api/scenario`, `POST /api/optimal-plan`,
`POST /api/scenario-compare`, plus `GET /api/optimizer/net-comparison` and
`GET /api/forecasts` which each run the sweep internally. Two of the four POSTs
(`/api/scenario`, `/api/optimal-plan`) have no dashboard caller.

### 4.4 Two ways to rebuild the plan

`POST /api/recompute-schedule` (synchronous, 409 if a job is running) and
`POST /api/jobs/recompute` (async with `job_id` and progress). The dashboard
calls the async one at TVBreakDashboard.jsx:1946 and the sync one at :1952 in
the same function, as a fallback.

### 4.5 Three undo systems, none of which covers the plan

| System | Scope | Storage | Endpoints |
|---|---|---|---|
| Version store | 9 logical files (`settings, constraints, overrides, advertisers, conditions, events, agencies, agency_links, agency_conditions`, `version_store.py:46`) | `data/versions/` | 5 |
| Assistant restore points | pre-apply copies of whatever a proposal batch touched | `data/assistant/restore/` | 2 (both orphaned) |
| Conversation restore | the changes made inside one conversation | `data/assistant/threads/` | 2 |

None of the three versions `output/weekly_break_schedule.csv`. Undoing a
settings change does not undo the plan that change produced. And in the 200
versions actually on disk, only 4 of the 9 logical kinds ever appear: settings
110, conditions 86, constraints 2, advertisers 2. `events`, `agencies`,
`agency_links`, `agency_conditions` and `overrides` have **never** been
versioned.

### 4.6 Four rule grammars for one idea

The idea is "under condition X, do Y to placement or price".

| Store | Key | Scope columns | Effect | Engine |
|---|---|---|---|---|
| `kairos_constraints.csv` (absent) | `constraint_id` | `scope_type, scope_value, channel, where_json` predicate | `offset_seconds`, `count`, `duration_seconds`, `order_index` | weekly optimizer |
| `manual_overrides.csv` (0 rows) | `override_id` + `anchor_date/anchor_start/anchor_title` | `scope, target_id` | `kind, value, gold` | weekly optimizer |
| `advertiser_conditions.csv` (0 rows) | `advertiser_id, rule_id` | `scope_positions, scope_genres, scope_dayparts` | `effect, value` (premium/require/forbid/pressure) | daily per-spot |
| `agency_conditions.csv` (0 rows) | `agency_id, rule_id` | same three plus `scope_programmes` | `effect, value, mode` | daily per-spot |
| `frequency_rules.csv` (1 row) | `rule_id` | `scope, advertiser_id, campaign, ad, competing_group, members` | `limit_type, value, unit` | daily per-spot |

Five stores, five schemas, five UIs, four of them empty. The single populated
row is `DEFAULT_ONE_PER_BREAK`, and it is the rule that drops 56 of the 175
daily spots (`status: dropped_frequency` in the exported ledger).

### 4.7 Two audit logs

`data/audit/activity.jsonl` (5,370 lines, keys `ts, user, role, method, path,
status, duration_ms, via, event`) is HTTP-level. `data/assistant/audit.jsonl`
(17 lines, keys `ts, user, event, model, question, results, batch_id, item_ids,
restore_id`) is assistant-level. `GET /api/activity-log` reads the first, `GET
/api/assistant/audit` reads the second, and the second has no caller. An
assistant-applied change therefore appears in both logs in two different shapes.

### 4.8 Two upload systems

`kairos_api/uploads.py` (7 kinds, writes into `data/`, validated against
`kairos.data.contracts`) and `kairos_api/assistant_uploads.py` (any spreadsheet,
parsed in memory, only a bounded summary kept per user under
`data/assistant/uploads/<user>/`). The two files on disk are 62,806 B and 62,810
B of parsed sheet summaries. Nothing links an assistant upload to an input kind.

### 4.9 The settings document, three carriers

`GET /api/settings`, `GET /api/overview.settings` and `GET
/api/parameters.settings` all return the same 33 fields. I diffed all three
against `data/kairos_settings.json`: identical key sets, no missing fields in
any direction. Two writers own the file: `PUT /api/settings` (whole document)
and `PUT /api/pricing` (deep-merges `pricing_overrides`, `pricing_api.py:250`).

---

## 5. Where the data model reflects the engine, not the work

### 5.1 A break is not an entity

`output/weekly_break_schedule.csv` has 21 columns and no break identifier. The
plan's unit is a **programme segment** with a `num_breaks` integer. The work is
"move the 20:05 break", but the model can only express "this segment now has 3
breaks instead of 2".

The API papers over this in two places, differently:

- `GET /api/break-operations` synthesizes ids like
  `קשת 12-program-3-0200-br-1` from a positional program index plus a counter
  (`dashboard_api.py`).
- `GET /api/break-library` returns `segment_id` and calls the row a "break".

Neither id survives a rebuild. `kairos/optimize/advertiser_rules.py:14-15` states
the consequence plainly: the weekly plan "never attributes a break to an
advertiser or a position, so it cannot consume per-advertiser rules without a
larger redesign".

### 5.2 Segment identity is an ordinal, not a name

`kairos/data/transform.py:255` and `:388`:

```
segment_id=f"{day}|{channel}|{index:03d}"
```

where `index` is `enumerate(...)` over the day's sorted programme rows. Insert or
drop one programme in the EPG and every later `segment_id` on that channel-day
shifts by one. Live example: `2024-11-01|רשת 13|000`.

The system already knows this. `manual_overrides.csv` carries a second key,
`anchor_date, anchor_start, anchor_title`, and
`kairos/optimize/overrides.py:175-200` re-binds by that triple, reporting a
stale override rather than silently rebinding. That anchor triple is the thing
the operator actually means. It is stored on exactly one of the five rule
stores.

### 5.3 The pricing model has premium layers that are switched off

`GET /api/pricing` returned `activation: {show: false, position: false, ad_type:
false}` with `position` carrying 5 values and `ad_type` 3. Turning them on moves
real revenue, so they are correctly gated, but the operator's screen presents
six layers of which three are inert.

### 5.4 The events layer is on and empty

`calendar_events.csv` has 63 rows, all `active=True`, and
`price_multiplier` is **1.0 for all 63** (std 0.0). `settings.pricing_overrides.
pricing_activation.events` is `true`. `GET /api/pricing` reports
`events: {enabled: true, active_event_count: 0}`. The mechanism is live and its
effect is exactly nothing. 56 of the 63 rows are auto-imported holidays whose
note reads "imported from the built-in holiday list, editable".

### 5.5 Four of seven uploadable inputs are stored and unread

Measured from `GET /api/uploads/status`:

| Kind | rows | in_use | reason (verbatim, truncated) |
|---|---|---|---|
| programmes | 3,562 | **false** | "the engine reads data/reference/Programmes.xlsx first" |
| daily | 175 | true | |
| spots | 50,386 | **false** | "the engine reads data/reference/Spots.xlsx first" |
| dayparts | 43,200 | **false** | "the engine reads data/reference/Dayparts.xlsx first" |
| advertiser_rules | 45 | true | |
| rate_card | 96 | **false** | "the rate card is read from config/optimization_weights.yaml" |
| campaign_flights | **0** | true | |

I confirmed the shadowing directly: `kairos.data.loaders.load_spots()` resolves
to `/Users/home/Code/questo/meridian/data/reference/Spots.xlsx` and returns
**12 columns**, while `data/Spots.csv` has 36. The upload panel is honest about
this (`uploads.py:75-96` exists purely to say so), but the effect is that of the
seven things an operator can upload, exactly one (the daily Wally file) reaches
a number on screen. `advertiser_rules` is nominally in use and inert for the
reason in section 6; `campaign_flights` is in use and empty.

### 5.6 `revenue_available: false` is the normal state

`GET /api/inventory` returned `revenue: null` for the summary and for all 5
dayparts and all 24 hours. `GET /api/campaigns` returned `revenue: null` for all
50 campaigns. Both because the engine's `Spots` source (the xlsx) has no
`revenue_ils` column, while `data/Spots.csv` does. The richer file is the one
nobody reads.

### 5.7 The version store is 93.5 percent test exhaust

Of 200 manifests in `data/versions/`, **187** record file paths under
`/private/var/.../pytest-of-home/pytest-NN/...`. `version_store.py:38` provides
`KAIROS_VERSIONS_DIR` for relocation; the tests that produced these entries
relocated the tracked *files* but not the version *store*, so the operator's
undo history is mostly pytest temp state. `POST /api/versions/{id}/restore` on
any of those 187 would write to a path that no longer exists. 188 of the 200
actors are `auth-disabled`. Sequence numbers run 29 to 228, so entries have also
been pruned.

`data/audit/activity.jsonl` shows the same pattern from the other direction: the
top paths include `/api/definitely-not-a-route` 234 times and users `usera`,
`userb`, `comp1`, `chan1`. `activity_log.py:75` honours `KAIROS_AUDIT_DIR` and
`tests/test_activity_log.py:30` sets it, so these came from live traffic against
a real server rather than from the isolated suite.

---

## 6. Identity, and where it is broken

### 6.1 Advertiser: three vocabularies, zero overlap

Measured with pandas:

| Source | Key space | Count | Example |
|---|---|---|---|
| `data/advertiser_rules.csv` `advertiser_id` | synthetic opaque | 45 | `ADV_01` .. `ADV_45` |
| `data/agency_advertisers.csv` `advertiser` | Hebrew trade name | 41 | `בנק הפועלים`, `אסם`, `מגדל` |
| `data/daily_input/Wally_*.csv` column `מפרסם` | Hebrew trade name | 41 | identical set |
| `data/Spots.csv` `advertiser_id` | derived garbage | 35 | `2024`, `פ`, `Reckitt` |

- `advertiser_rules` ∩ `agency_advertisers` = **0**
- `advertiser_rules` ∩ daily file = **0**
- `agency_advertisers` ∩ daily file = **41 of 41**
- `advertiser_rules` ∩ `Spots.csv` = **0**

`data/Spots.csv.advertiser_id` is `Campaign.split(/[-\s]/)[0]` with a **100
percent** match rate across all 50,386 rows. 34,272 rows (68 percent) have
`advertiser_id = "2024"`, the year prefix of the campaign code. In the same file
`adv_premium` is the constant `1` for every row.

The consequence, measured live at `GET /api/advertisers/stats`: all 45
advertisers return `display_name: ""`, `name_source: "unnamed"`, `rule_count: 0`,
`revenue: null`, `revenue_source: "source_pending"`, `avg_effective_premium:
1.0`. Zero advertisers have any attributed spot. The advertiser rules engine
(`kairos/optimize/advertiser_rules.py`) keys `self.baselines` on
`advertiser_id`; the daily path passes the Hebrew name; the lookup can never
hit, and the module's stated honesty rule ("an unknown advertiser yields a
premium of 1.0, never zero") means the miss is silent and revenue-neutral.

The Advertisers page is a fully built CRUD surface over 45 rows that cannot bind
to a single real spot.

### 6.2 Agency identity works, and shows what advertiser identity should look like

`agencies.csv` is keyed `AGY_01..AGY_09` with a `name` column (`OMD`,
`יוניברסל`, ...). The daily file's `משרד / MB` column carries those same names.
Something resolves name to id, and it works: `GET /api/agencies/summary`
returned `gross_revenue: 699450.0, net_revenue: 669978.0, rebate_total:
29472.0, spot_count: 119`, and `GET /api/export/spots.csv` resolves 9 distinct
agencies across 175 rows. `GET /api/agencies/AGY_01/advertisers` returns 6
Hebrew advertiser names with `observed_source_file` naming the daily file.

So the same file, on the same 175 rows, resolves the agency and fails to resolve
the advertiser. The missing piece is a `name` (or `aliases`) column on
`advertiser_rules.csv`, exactly like the one `agencies.csv` has.

### 6.3 Campaign is three things

- `Spots.csv.Campaign`: 478 free-text strings mixing sponsorship codes (`פ-מי
  זאת`) with dated buy codes (`2024-10 - משרד הביטחון - ...`).
- `campaign_flights.csv.campaign_id`: the pacing key. **0 rows.**
- Daily file `קמפיין`: the operator's campaign name per spot.

`GET /api/campaigns` groups on `(Campaign, advertiser_id)` (`catalog_api.py:267`)
and returns `advertiser_id: ""` for all 50. `GET /api/make-good-alerts` returns
`data_available: false` with the reason naming `campaign_flights.csv`. Nothing
joins these three.

### 6.4 Break has no identity at all

See 5.1. Three synthetic id schemes (`segment_id` ordinal, break-operations
composite, override anchor triple), no stored break entity, and `Spots.csv` has
a real `break_id` column (9,492 distinct values) plus `position_in_break` (48
values) that the engine never loads because it reads the xlsx.

### 6.5 User identity

`data/auth/users.json` holds 1 user (`admin`). `data/audit/activity.jsonl` shows
12 distinct `user` values including `auth-disabled` (1,425 rows), `anonymous`
(235) and two malformed entries (`alhost:3000/sign`, an email address). The
assistant's per-user directories are keyed by a sanitize-plus-hash of the
username (`assistant_uploads.py` docstring), so the audit log and the assistant
store key the same person differently.

---

## 7. The entity graph as it exists today

Arrows are labelled with the join key. `[N]` is the measured row count today.
`(x)` marks a join that does not resolve.

```
                      kairos_settings.json [33 keys]
                              |
                              | operator_channel = "רשת 13"
                              v
  reference/Programmes.xlsx --> ProgramSegment  --segment_id--> weekly_break_schedule.csv [8704]
       [3562 airings]         (in memory only)   "date|channel|NNN"        |
            |                      ^                                       | segment_id
            | Title, Channel,      | index = enumerate() position          v
            | Date, Start time     |                          .meta.json fingerprints [9]
            v                      |
  reference/Dayparts.xlsx ---------+                          manual_overrides.csv [0]
       [39600 rows]  tvr by channel-minute                     | override_id
                                                               | target_id ---> segment_id
                                                               | (anchor_date, anchor_start,
                                                               |  anchor_title)  <-- the real key
                                                               v
                                                     kairos_constraints.csv  [FILE ABSENT]
                                                               ^ constraint_id, where_json predicate

  ---------------------------- the daily path (separate universe) ----------------------------

  daily_input/Wally_*.csv [175 spots]
      | מפרסם (Hebrew name)               | משרד / MB (agency name)
      |                                   |
      |  (x) no match                     v  name -> agency_id
      v                              agencies.csv [9]  AGY_01..AGY_09
  advertiser_rules.csv [45]               |  agency_id
      ADV_01..ADV_45                      +--> agency_conditions.csv [0]  agency_id, rule_id
      | advertiser_id                     +--> agency_advertisers.csv [41]  agency_id + advertiser(Hebrew)
      v                                          |
  advertiser_conditions.csv [0]                  | advertiser (Hebrew) --(x)--> advertiser_rules.advertiser_id
      advertiser_id, rule_id                     v
                                          export/spots.csv [175: 119 priced, 56 dropped]
  frequency_rules.csv [1]  ------------------------^  rule_id (the only rule that fires)

  ---------------------------- historical, largely unjoined ----------------------------

  reference/Spots.xlsx [50386 x 12]  -> /api/inventory, /api/campaigns   (no revenue, no advertiser)
  data/Spots.csv       [50386 x 36]  -> nothing            (HAS advertiser_id(garbage), break_id,
                                                            position_in_break, revenue_ils,
                                                            competitor_flag)
  data/enriched/*.csv  [93548 rows]  -> nothing
  Spots - inventory.csv [994]        -> kairos/optimize/inventory.py (demand steer)

  ---------------------------- operator state ----------------------------

  calendar_events.csv [63, all multiplier 1.0] --event_id--> pricing events layer (enabled, 0 active)
  rate_card_premiums.csv [96]  --(x)--> pricing engine (reads config/optimization_weights.yaml)
  campaign_flights.csv [0]     --campaign_id--> pacing, make-good  (no rows, both report unavailable)
  auth/users.json [1]          --username--> audit/activity.jsonl [5370, 12 distinct users]
  versions/ [200 dirs]         --version_id--> {settings, conditions, constraints, advertisers}
                                               (187 of 200 point at pytest tmp paths)
  assistant/proposals.json [1 batch] --batch_id--> item.kind --> the 9 logical files
  assistant/threads/<user>/ [1 conversation]  --conversation_id--> changes, restore
  assistant/restore/<id>/ [1]  --restore_id--> pre-apply file copies
```

Reading the graph: there are **two disconnected universes**. The weekly universe
(programme segments, break counts, retention, the frontier, all the money on the
Overview) knows nothing about advertisers, agencies, campaigns or spots. The
daily universe (175 spots, real advertisers, real agencies, real rebates) knows
nothing about segments, breaks or the plan. The only bridge is a shared channel
name and a shared date.

---

## 8. Defects found while measuring

These are incidental to the inventory but they are measured and they are real.

1. **The "200 of N" honesty banner never renders.** `dashboard_api.py:1506`
   emits `break_schedule_total_rows`; `TVBreakDashboard.jsx:3331` reads
   `schedule.total_rows`. `finiteNumber` (`surface-helpers.js:19-22`) returns
   `null` for `undefined`, so the guard at `TVBreakDashboard.jsx:3471` is always
   false. The operator sees 200 rows with no indication that the plan has 8,704.
   The comment at `:3330` describes the intended behaviour.

2. **`GET /api/versions` took 27.5 seconds** on 200 entries with the local
   filesystem warm. It is the slowest endpoint in the API by 4.5x.

3. **`GET /api/uploads/status` took 9.6 seconds** because it row-counts
   `Spots.csv` (50,386), `Dayparts.csv` (43,200) and `Programmes.csv` (3,562) on
   every call.

4. **`kairos/optimize/agreements.py` is dead.** `load_agreements`, `AgreementSet`
   and `apply_agreements` have zero callers outside the module itself, its
   `__init__` re-export and `tests/test_agreements.py`. Its store,
   `data/reference/AdvertiserAgreements.csv`, is header-only.

5. **`data/enriched/` is 19.7 MB of dead weight.** Repo-wide grep for
   `Spots_enriched`, `Programmes_enriched` and `Dayparts_enriched` returns zero
   source files.

6. `data/Programmes - today.csv` (125 rows) has zero readers repo-wide.

---

## 9. Measured summary for the rebuild

| Question | Answer |
|---|---|
| Endpoints | 113 (not 111) |
| Orphaned endpoints | 20 (18 percent), of which 5 have no caller anywhere |
| Endpoints reading `weekly_break_schedule.csv` | 9 modules, 5 mutually inconsistent projections |
| Rule stores | 5 grammars, 4 empty, 1 row total across all five |
| Uploadable inputs that reach a screen | 1 of 7 |
| Advertisers with a resolvable identity | 0 of 45 |
| Agencies with a resolvable identity | 9 of 9 |
| Money that resolves to a commercial party | 699,450 ILS gross over 119 spots on one day |
| Money in the weekly plan | 8,704 segment rows, attributable to nobody |
| Version entries that point at real paths | 13 of 200 |
| Assistant tools bypassing the HTTP API | 39 (31 read, 8 propose) |
