# Sober dashboard review, 2026-07-06

Scope: every main page of the Kairos dashboard (tv-break-dashboard), checked against the real API payloads fetched in-process (TestClient, KAIROS_AUTH_DISABLED=1) on the live repo data. Method per page: read the full render code, fetch the endpoints it consumes, compare what arrives with what is shown, then check copy, units, dir=ltr and cross-page consistency. Fixes here are UI-only and surgical; the backend was read but not touched. Peer-owned regions (Assistant, Overview money waterfall, FrontierPanel, SettingsPanel engine-focus block, SchedulePage/TimelineView/editor and their component files) were left alone.

No build was run; every touched file was parse-verified with esbuild. Nothing was committed.

## Per-page verdict

| Page | Checked | Defects found | Fixed here / reported |
| --- | --- | --- | --- |
| Overview (סקירה) | render + /api/overview, /api/files | frontier "computing" status arrives but was not passed to the chart, so a cold frontier showed "not enough scenarios" instead of "being computed" | Fixed (prop wired). Waterfall/YieldView are peer-owned; not reviewed here |
| Optimizer (אופטימייזר) | render + /api/overview, /api/schedule, POST /api/optimizer-plan, /api/parameters | Inspector fabricated data (72.3% retention fallback, invented "KAI 1"/"20:00" identity, 3 hardcoded-"compliant" guardrail rows, fabricated "Medium"/₪0 recommendation when none exists); preview run scope (one channel-day) undisclosed next to whole-week metrics; retention-cost rows nameless for live plans; grid cells fabricated a minimum of one break dot; 8 real genres missing Hebrew labels | Fixed (all). Reported: preview optimizes the first channel-day in the source, not the operator channel |
| Inventory (מלאי) | render + /api/inventory | `revenue_available:false` silently dropped (unexplained column of dashes); hourly panel showed dash values with "booked value" framing; decorative arrow-"deltas" made nav words look like data | Fixed (disclosure note, minutes-based hourly view when revenue is absent, deltas removed). Reported: backend by_hour is a fabricated single 00:00 bucket |
| Break library (ספריית ברייקים) | render + /api/break-library | 80 rows spanning 3 channels and 30 dates with no channel/date/time identity shown | Fixed (channel + airing columns). Reported: library mixes competitor channels into "candidates" |
| Campaigns (קמפיינים) | render + /api/campaigns, /api/make-good-alerts | `revenue_available:false` dropped; "advertiser" column empty for every row (loader gap); "טיסות" copy read as airline flights | Fixed (disclosure note, copy). Reported: loader drops advertiser_id; last_airing string-max latent bug |
| Forecasts (תחזיות) | render + /api/forecasts, POST /api/scenario-compare | scenario bars are single channel-day runs shown beside whole-week daily totals (30x scale gap) with no disclosure; payload revenue_weight dropped; daily table arrived in alphabetical day order; ScenarioCompare dropped the channel/day scope it receives | Fixed (basis notes, weight tooltip, week ordering). Reported: Balanced and Revenue-priority produce identical plans, so the weight sweep shows twin bars |
| Reports (דוחות) | render + /api/reports, /api/files | English-only titles/owners in Hebrew UI; `empty`/`attention` statuses (real backend states) would render as raw keys | Fixed (id-mapped bilingual titles/owners, StatusBadge labels) |
| Data (נתונים) | render + /api/files, /api/impact, /api/parameters, /api/uploads/status | decorative metric "deltas"; the model pooling honesty note (coefficient_impacts.pooling_note) silently dropped | Fixed (deltas removed, note surfaced). Reported: uploads in_use_reason/warnings arrive English-only |
| Advertisers (מפרסמים) | render + /api/advertisers, /stats, /options | none material; page is honest ('-' with provenance everywhere revenue attribution is pending) | Reported: Hebrew status banner is a hardcoded translation of one specific backend sentence (brittle) |
| Pricing (תמחור) | render + /api/pricing, POST /api/pricing/price-slot | layer titles and descriptions English-only in the Hebrew UI (Program / Day / "Always applied.") | Fixed (bilingual titles, Hebrew descriptions for the stable layer set, payload fallback). Reported: layer descriptions should carry a Hebrew variant at the API |
| Overrides (עקיפות) | render + /api/overrides (create/list/delete round-trip), /api/schedule/segments, /api/overrides/effect | the store groups overrides by scope and keys records `override_id`; the console expected a flat array with `id`, so the "current overrides" card was permanently blank (no list and no empty state), delete targeted undefined, force-count never showed | Fixed (flatten both scopes, override_id, string value). Reported: needs a contract test |
| Settings (הגדרות) | render + /api/settings, /api/settings/controls, /api/parameters | templates verified in sync with the backend controls endpoint; operator-channel panel honest | Reported: `require_manual_approval` is a dead settings field (no UI control, no engine consumption); pacing help texts are order-coupled to their controls |
| Global | staleness banner, activity feed, status bar | banner called overrides "התאמות ידניות" while the nav canon is "עקיפות" | Fixed (vocabulary aligned) |

Schedule and Assistant were excluded (peer-owned this wave). GoldBreakManager and MakeGoodAlerts were checked as parts of Schedule/Campaigns respectively.

## Fixes shipped (hunk list)

TVBreakDashboard.jsx (line numbers = post-change):

1. L732-735 (impactSegmentLabel) - Data page impact rows: unknown genre keys now fall back to the shared genre map instead of raw English.
2. L840-859 (programTypeLabel) - added the 8 classifier genres observed in live payloads (Digital, Documentary, Lifestyle, Morning Program, Music, Religious, Special Event, Talk Show); they previously rendered as raw English across the Optimizer grid, Break library and Schedule table.
3. L2295-2304 (OptimizationRunSummary) - the preview's real scope (plan.channel + plan.day, fields the payload carries and the UI dropped) is now disclosed: "one channel-day, not the weekly total".
4. L2343-2354 (RetentionCostSegment) - live-plan segments carry only segment_id; confidence rows are no longer nameless blocks of numbers (dir=auto fallback to the id).
5. L2658 (OptimizerWorkspace) - passes the saved retention floor to the Inspector so its guardrail check uses the real setting, not a hardcoded 72.
6. L2702-2703 (StatusBadge) - labels for the real backend statuses `attention` and `empty`, which would have rendered as raw keys on Reports.
7. L2852 (OverviewPage) - FrontierScopeChart now receives overview.frontier_status; a cold "computing" frontier says so instead of "not enough scenarios" (the component itself documented this missing wiring).
8. L3025-3095 (InventoryPage) - consumes `revenue_available`: a bilingual note explains the dash values; the hourly panel switches to booked minutes (real data) instead of a dash column framed as booked value; removed the two decorative arrow-"deltas" ("מקור", nav Schedule) that rendered navigation words as data movements.
9. L3124-3125 (BreakLibraryPage) - added Channel and Airing (date + start time) columns; the payload spans 3 channels and 30 dates and the table showed no identity at all.
10. L3141-3159 (CampaignsPage) - consumes `revenue_available` with a bilingual note (revenue dash + spot-count ranking disclosed).
11. L3186-3230 (ForecastsPage) - daily table sorted into week order (arrived alphabetical: Fri, Mon, Sat...); scenario rows expose the payload's revenue_weight as a tooltip; added the basis note that scenarios are single representative channel-day runs, not weekly totals.
12. L3262-3301 (ReportsPage) - stable report ids mapped to Hebrew titles/owners (payload is English-only), raw payload text kept as fallback for unknown ids.
13. L3391-3394 (DataHubPage) - removed the four decorative metric "deltas" (nav labels with up/down arrows).
14. L3423-3428 (DataHubPage) - surfaces coefficient_impacts.pooling_note (the empirical-Bayes "cells pool to one constant" honesty disclosure) which was silently dropped.
15. L4091-4096 (ProgramCell) - zero planned breaks now shows zero marker dots (was a fabricated minimum of one).
16. L4138-4252 (Inspector) - de-fabricated: retention shows the measured value or a dash (was `|| 72.3`); identity shows real channel/time or an honest empty state (was "KAI 1" / "20:00" / "K1"); the guardrail block keeps only the one check it can actually compute (retention vs the saved floor, with an explicit "not measured" state) plus a pointer to the compliance ledger, instead of three hardcoded-"compliant" rows; the recommendation block renders an honest empty state instead of a fabricated title with "Medium" risk and ₪0 impact; approve/reject/apply-similar disabled when there is no recommendation.

OverrideConsole.jsx:

17. L34-38 - flattens the grouped `{overrides: {segment: [], spot: []}}` payload (tolerates a flat array from older backends). Before this, the list card never rendered anything, including its empty state.
18. L400-434 - records keyed by `override_id` (delete/`key` used a nonexistent `o.id`); force count read from the store's string `value` (was a number-only check that never passed).

PricingManager.jsx:

19. L29-41, L53, L287-288, L333, L427, L433 - bilingual layer titles + Hebrew descriptions for the stable layer set (program/day/show/position/ad_type) in the cards, the tester breakdown and aria-labels; payload English kept as fallback. File held under 450 lines (449).

ScenarioCompare.jsx:

20. L176-184 - basis note naming the channel-day both A/B runs optimize (fields the payload carries and the UI dropped).

MakeGoodAlerts.jsx:

21. L65-66 - "טיסות קמפיין"/"תאריכי טיסה" (read as airline flights) reworded to natural Hebrew; English side tightened to "start and end dates and delivery goals".

ScheduleStalenessBanner.jsx:

22. L32 - overrides named "עקיפות ידניות" to match the nav/page canon (was "התאמות ידניות").

styles.css:

23. L5806-5837 - `.data-basis-note` (shared honest-basis footnote), `.optimizer-run-scope` (full-row placement inside the run-summary grid), `.guardrail-measure` / `.guardrail-footnote` (Inspector).

## Bigger items for the lead (ranked, not fixed here)

1. Backend: the reference Spots loader starves the catalog builders, producing fabricated aggregates. `kairos/data/loaders.py::load_spots` returns only [Campaign, Channel, Date, Duration, Pos. Block 1, Promotion, Spot type, Spots Block 1, Start time, TVR, Title, air_dt]. `kairos_api/catalog_api.py:174` then defaults `hour_of_day` to 0 for every row, so `/api/inventory.by_hour` is a single "00:00" bucket claiming all 50,386 spots (verified in the live payload) and the Inventory hourly chart is built on an invented hour. Same root cause: `campaigns[].advertiser_id` is empty for all 50 rows (dead column on the Campaigns page) and `by_channel[].target_spots` is a fabricated 0 (missing is_target_channel column summed as False). Fix in `_build_inventory`/`_build_campaigns`: derive hour from `air_dt`, carry advertiser_id through the loader, and report target_spots as null when the column is absent.

2. Product/backend: every "scenario" surface optimizes one arbitrary channel-day, and it is not the operator's channel. `kairos/service.py::run_scenario` picks `_first_channel_day` (observed: כאן 11, 2024-11-01) while settings.operator_channel is עכשיו 14. This feeds POST /api/optimizer-plan (the header "Run optimization" button), /api/forecasts scenarios, and /api/scenario-compare. The UI now discloses the scope, but the preview arguably should default to the operator channel. Also measured: Balanced (weight 60) and Revenue priority (weight 90) return byte-identical plans (826,743.62 ILS both; retention-guardrail 823,251.39), consistent with the known result that the weight sweep collapses under the real optimizer; consider replacing the Forecasts weight-scenario bars with a floor sweep like the frontier, or dropping them.

3. Backend/infra: cold-start latency of the optimizer-backed panels. On a cold in-process client, a batch of /api/overrides/effect, /api/gold-breaks, /api/yield-per-second and /api/optimizer/net-comparison did not complete within 33+ minutes (each is a live optimization; yield and net-comparison cover the whole week). After a server restart the Overview yield panel and frontier sit in skeleton/"computing" states for a long time (observed: overview.frontier_status="computing" with an empty frontier). Consider persisting the last computed artifacts with a computed_at timestamp (the weekly CSV pattern) instead of recomputing live.

4. UI contract regression risk: the Overrides page break (fixed here) came from the API's grouped list shape vs the console's flat-array expectation; nothing failed loudly, the card just went blank. Worth a UI-contract test that POSTs an override and asserts the list renders (`overrides.segment` flattening + `override_id` keying), so the next payload change cannot silently blank the decision surface again.

5. Break library mixes competitor channels into "ranked break candidates". /api/break-library returns קשת 12 and רשת 13 rows beside the owned עכשיו 14 with the same "ready" status (`catalog_api.py::_build_break_library` reads the whole weekly CSV). The new channel column makes this visible, but the product question stands: candidates the operator cannot schedule probably need an operator-channel filter or an explicit basis disclosure.

6. English-only backend copy reaching the Hebrew UI (needs *_he fields or stable codes): pricing layers[].description (mitigated in UI for known layers), uploads in_use_reason and warnings[], gold-breaks/make-good/scenario-compare `reason` strings, overrides/effect rejected_overrides[].reason, coefficient_freshness.reason (shown when stale/unknown), drift criterion tooltip. Individually small; together they are the main remaining source of mixed-language surfaces.

7. Dead settings field: `require_manual_approval` exists only as a schema default (kairos_api/core.py:94); no UI control renders it and nothing in kairos/ or kairos_api/ consumes it. Either wire it to the approval flow or drop it; adding a toggle now would fabricate a lever.

8. Latent date bug: campaigns `last_airing` is a string max over dd/mm/yyyy values (catalog_api.py:233); correct only while all data sits in one month. Convert to a real date before aggregating.

9. Vocabulary drift at the API: /api/forecasts scenarios carry name_he values ("ריסון לטובת צפייה", "עדיפות להכנסה") that differ from the dashboard's canonical scenario names ("הגנת שימור", "מקסום הכנסה"). The UI intentionally keeps its own labels for cross-page consistency; align the backend strings when convenient.

10. Advertisers status banner brittleness: AdvertisersManager.jsx:300-311 hardcodes a Hebrew translation of the exact current backend status sentence (advertisers.py:209). If the backend text changes, Hebrew silently keeps the old claim. A stable status code plus client-side label would be sturdier.

11. Minor, conditional: GoldBreakManager renders raw English day tokens in its chips/table when gold breaks exist (none in the current plan); the Inspector's "selected break" framing still conflates program selection (grid) with break selection (timeline) - honest now, but a design-level cleanup someday.

## Verification

- esbuild parse clean on every touched file (TVBreakDashboard.jsx, PricingManager.jsx, OverrideConsole.jsx, ScenarioCompare.jsx, MakeGoodAlerts.jsx, ScheduleStalenessBanner.jsx, styles.css untouched by parse since CSS).
- Added display copy scanned: no em-dash, no exclamation marks, no emoji.
- Touched component files kept under 450 lines (PricingManager 449, OverrideConsole 449); only the monolith exceeds it, as allowed.
- Overrides round-trip (POST -> grouped GET -> DELETE) exercised in-process to validate the new list handling.
- Backend untouched; nothing committed.
