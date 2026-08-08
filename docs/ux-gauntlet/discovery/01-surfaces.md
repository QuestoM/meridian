# Surface inventory

Discovery pass over every front-end surface of the Meridian / Kairos dashboard.
Read-only. No product file was changed.

## Method and instance

- Instance: `http://127.0.0.1:8010` (serves the API and the built dashboard).
  `GET /api/auth/me` returns `{"auth_disabled":true}`, so the app renders with no
  login wall and the sidebar footer reads "Open access / Sign-in is not set up yet".
- Every page below was opened in Chrome on that instance and read from the live DOM.
- Endpoint calls per page were captured with `performance.getEntriesByType('resource')`
  filtered to `/api/`, diffed after each navigation.
- Endpoint latency was measured with `curl -w "%{time_total}"` against the same instance.
- Payload claims were checked against the JSON the server actually returned.
- Source tree: `tv-break-dashboard/src`, 80 files (52 `.jsx`, 13 `.js`, 15 `.css`),
  20,172 lines across `.jsx` + `.js`. `TVBreakDashboard.jsx` alone is 6,236 lines.

Verdict vocabulary used: LOAD-BEARING, DUPLICATE-OF, DEAD, HALF-BUILT.

## The seventeen navigation entries

`navItems` is declared at `tv-break-dashboard/src/TVBreakDashboard.jsx:565-583`.
The route name and the displayed label differ for three of them
(`copy.nav`, `TVBreakDashboard.jsx:291-309`).

| # | Route (`#hash`) | Sidebar label | Renders | Verdict |
|---|---|---|---|---|
| 1 | `#Overview` | Overview | `OverviewPage` (`:3248`) | LOAD-BEARING |
| 2 | `#Optimizer` | Optimizer | `OptimizerWorkspace` (`:2977`) | DUPLICATE-OF Schedule and Overview |
| 3 | `#Schedule` | Schedule | `SchedulePage` (`:3324`) | LOAD-BEARING |
| 4 | `#Inventory` | Inventory | `InventoryPage` (`:3497`) | LOAD-BEARING |
| 5 | `#Break%20Library` | Break Library | `BreakLibraryPage` (`:3586`) | DUPLICATE-OF Overview "Priority decisions" |
| 6 | `#Campaigns` | Campaigns | `CampaignsPage` (`:3679`) | HALF-BUILT |
| 7 | `#Forecasts` | Forecasts | `ForecastsPage` (`:3725`) | HALF-BUILT |
| 8 | `#Calendar` | Events calendar | `CalendarEvents.jsx` | LOAD-BEARING |
| 9 | `#Reports` | Reports | `ReportsPage` (`:3800`) | LOAD-BEARING |
| 10 | `#Data` | Data | `DataPage` (`:3942`) | LOAD-BEARING |
| 11 | `#Advertisers` | Advertisers | `AdvertisersManager.jsx` | HALF-BUILT |
| 12 | `#Agencies` | Agencies | `AgencyManager.jsx` | HALF-BUILT |
| 13 | `#Pricing` | Pricing | `PricingManager.jsx` | HALF-BUILT |
| 14 | `#Overrides` | Overrides | `OverrideConsole.jsx` | LOAD-BEARING |
| 15 | `#Assistant` | Kai, AI assistant | opens `AssistantDock` beside the current page | LOAD-BEARING |
| 16 | `#Versions` | Restore changes | `VersionsPage.jsx` | LOAD-BEARING |
| 17 | `#Settings` | Settings | `SettingsPanel` (`:5432`) | LOAD-BEARING |

Route-versus-label mismatch: `Calendar` shows as "Events calendar", `Assistant` as
"Kai, AI assistant", `Versions` as "Restore changes". A shared or bookmarked URL
does not name the page the reader lands on.

`#Assistant` is not a workspace. `viewFromLocation` accepts it, `useState` maps it
to `Overview` (`:1395-1398`), and the hash listener opens the dock instead of
switching views (`:1592-1600`). Verified live: navigating to `#Assistant` while on
Overrides left the Overrides workspace mounted and opened the dock.

### 1. Overview

Landing page. Sections read from the live DOM: Executive operating view, From gross
to net, Priority decisions, Control room, Compliance ledger, Revenue vs retention,
Yield per second.

Endpoints on cold load (captured from resource timing, one page load):
`/api/overview`, `/api/inventory`, `/api/break-library`, `/api/schedule`,
`/api/campaigns`, `/api/forecasts`, `/api/reports`, `/api/files`, `/api/impact`,
`/api/break-operations`, `/api/parameters`, `/api/events`, `/api/auth/me`,
`/api/yield-per-second` (twice), `/api/overview?scope=channel%3A%D7%A8%D7%A9%D7%AA%2013`.

The double `/api/yield-per-second` is real: `YieldMoneyPanel` (`MoneyWaterfall.jsx:167`)
and `YieldView` (`YieldView.jsx:56`) each fetch it independently, and both render on
Overview (`TVBreakDashboard.jsx:3270` and `:3319`).

What a person can do here: read the week's revenue, retention, ad minutes and risk;
read the gross-to-net waterfall; open one of five priority decisions into the
Optimizer; read the compliance ledger; click a frontier point and apply its
retention floor; read yield per second by daypart and programme.

Verdict: LOAD-BEARING. It is the only surface where the frontier point can be
applied as a saved retention floor (`onApplyFrontierFloor`, `:3313`).

### 2. Optimizer

Renders, in order: the same four stat tiles as Overview, an optimization run
summary, a coefficient freshness chip, a retention cost panel, a four-mode planner
(Grid View / Timeline / Daypart / Inventory), a break inspector, and then
`FrontierPanel` + `InventoryHeatmap` + `ComplianceLedger` (`:3128-3134`).

Endpoints called on entering the page: none. It re-renders the bulk payload the
shell already fetched.

Measured duplication against other pages:

- The four headline tiles are byte-identical to Overview's: `10.12M`, `94.4%`,
  `1,106 min`, `Low 0/100`, with the same basis line. `SummaryMetrics` (`:2660`) is
  rendered on both.
- `ComplianceLedger` is the same component with the same `compliance` prop as
  Overview's (`:3132` vs `:3306`). Both print the identical 7 rows including
  "Viewer retention floor Compliant 78.6% / 72%".
- `FrontierPanel` titled "Revenue vs retention" (`:3130`) prints the same
  `94.4% / 100.6% / 1.6M / 1.41M / 95%` as the Forecasts copy (`:3776`) and the same
  numbers as the Overview `FrontierScopeChart` (`:3307`).
- Grid View calls `PlanningCanvas` with `schedule.rows` (`:3069`). Schedule's Grid
  View calls `PlanningCanvas` with `schedule.rows` (`:3428`). Same component, same
  data, same rendered numbers (verified: both show `749.56K / 95.1%` for the same
  cell).
- Daypart calls `DaypartView(schedule.rows)` (`:3090`); Schedule's Daypart calls
  `DaypartView(schedule.rows)` (`:3459`). Identical.
- Timeline calls `TimelineView(schedule.break_operations)` (`:3080`); Schedule's
  Timeline calls the same (`:3438`) but additionally passes `notify`, `zoom` and
  `onGlobalRefresh`. The Optimizer copy therefore drops the shared zoom scale and
  the "not on your owned channel" message path (`TimelineView`, `:4385-4390`).

Verdict: DUPLICATE-OF Overview (tiles, compliance ledger, frontier) and
DUPLICATE-OF Schedule (grid, daypart, timeline). What is unique to Optimizer is the
break inspector with approve / reject / apply-similar / open-in-overrides, and the
`Run Optimization` preview plan.

Sub-panels:

- "Daypart inventory heatmap" renders "No daypart heatmap data yet" on this live
  install. `InventoryHeatmap` (`:5159-5172`) is a hard-coded empty state with no
  data path at all. Verdict: DEAD.
- The inspector export dropdown offers three scopes ("Break detail",
  "Weekly traffic plan", "Guardrail report", `:4886-4890`) but every one calls the
  same handler, which always writes `kairos-break-detail.json` with only an
  `exportScope` string changed (`:2071-2074`). Verdict: HALF-BUILT.
- The inspector footnote says schedule-wide checks "live in the compliance ledger
  below" (`:4849`). The ledger is below on this page, and also on Overview.

### 3. Schedule

Title "Schedule control". Four modes: Grid View, Daypart, Timeline, Editor
(`:3376-3410`). Then a "Break plan rows" table and `GoldBreakManager` (`:3492`).

Endpoints on entry: `/api/gold-breaks`. Entering Editor adds `/api/overrides`,
`/api/schedule/segments`, `/api/constraints/options`, `/api/constraints`.

What a person can do: read the weekly plan four ways, download the plan CSV, print
it, drag and resize breaks in the Editor with 30s/60s snap and a zoom scale, build
a constraint, open a segment inspector, recompute a single day.

Findings:

- The toolbar's "Weekly traffic plan" button writes
  `kairos-weekly-plan-first-200.json` from `schedule.break_schedule`, which the API
  caps at 200 rows out of `break_schedule_total_rows: 8704` (verified in the
  payload). The Reports page has a card also called "Weekly traffic plan" that
  downloads 8,704 rows as CSV. Two controls, one name, a 43x difference in content.
- The Editor renders lanes for `קשת 12`, `עכשיו 14`, `כאן 11` (competitor channels),
  while the Overrides page states "Pick an owned-channel segment".
- In the Editor at 1.0x zoom, break chips clip their own labels: observed
  `02:12:`, `05:16:0`, `16 m ...`, with the seconds figure `120s` on a second line.

Verdict: LOAD-BEARING. The Editor is the only place a break can be moved.

### 4. Inventory

Title "Inventory yield". Reads `/api/inventory` from the bulk load. Shows
"Your channel's inventory: רשת 13", four tiles (Inventory spots 18,669, Booked value
`-`, Booked minutes 5,968 min, Retention risk Low 0/100), a five-row daypart table
(Morning, Noon, Night, Prime time, Evening) with Revenue `-` on every row, and an
hourly booked-minutes bar list.

The page states its own blocker: "The loaded spots source carries no revenue column,
so money figures on this page show a dash."

Findings:

- "Retention risk Low 0/100" is the same tile as Overview and Optimizer.
- The Revenue column is `-` for all five dayparts, while the Overview "Yield per
  second" card prints revenue for the same five daypart names: Prime time 10.77M,
  Evening 13.01M, Noon 6.53M, Night 5.56M, Morning 5.08M. Same taxonomy, two pages,
  one shows money and one shows a dash.
- The daypart vocabulary here (Morning, Noon, Night, Prime time, Evening) is not the
  one hard-coded in the dashboard (`daypartKeys = ['Morning','Daytime','Access',
  'Primetime','Late night']`, `:586`).

Verdict: LOAD-BEARING (it is the only sellable-supply view), with a money column
that cannot be filled from the current sources.

### 5. Break Library

Title "Break library", subtitle promises "The ranked shelf of the strongest breaks".
Table "Ranked break candidates", 80 rows, columns Status / Channel / Airing /
Programme type / Position / Type / Length / Revenue / Retention. CSV export button.
Clicking a row opens `ScheduleInspector` and calls
`/api/schedule/segment/2024-11-23%7C%D7%A8%D7%A9%D7%AA%2013%7C074`.

Measured findings:

- The table is not sorted by the Revenue column it is ranked on. In the 80 rows
  returned by `/api/break-library`, 18 of the 79 adjacent pairs are out of
  descending order. First violation is at index 3: `260,121.30` sits above
  `263,007.38`. Visible in the UI as `260.12K` printed above `263.01K`.
- Overview "Priority decisions" prints the same records: `393,297.19`,
  `287,284.99`, `281,215.58`, `263,007.38`, with the same retention `80.5%`.
  `/api/overview` `recommendations[0].impact` is `393297.19`;
  `/api/break-library` `breaks[0].predicted_revenue` is `393297.19`.
- The drawer opened by clicking a break is headed "PROGRAMME INSPECTOR" and titled
  `Reality` (a programme class, not a programme). Its "Class" row shows the same
  value the table column calls "Programme type".
- Its money row is "Plan revenue (this segment) 393,297 ₪". The Optimizer's
  inspector, headed "SELECTED BREAK", shows "Plan revenue ₪16,141" for a
  37-minute programme carrying 3 breaks. Same label, two granularities.

Verdict: DUPLICATE-OF Overview "Priority decisions" for the top of the list, plus a
ranking that does not rank.

### 6. Campaigns

Title "Campaign allocation". Table "Advertiser demand", 50 campaigns, columns
Campaign / Advertiser / Spots / Minutes / Channels / Revenue / Last airing. Below it
`MakeGoodAlerts` (`/api/make-good-alerts`).

Findings:

- The Advertiser column is blank on every visible row (10 of 10 checked).
- The Revenue column is `-` on every visible row. The page says why: "The loaded
  spots source carries no revenue column, so campaign revenue shows a dash and
  campaigns are ranked by spot count."
- "Make-good alerts / Under-delivery risk" renders "No campaign data yet" plus
  "Upload campaign_flights.csv with start and end dates and delivery goals".
  `/api/uploads/status` confirms a `campaign_flights` input exists with `rows: 0`
  and `in_use: true`, so the upload slot is real and empty.
- This page is titled around "Advertiser demand" while a separate nav entry named
  Advertisers holds the advertiser records, and a third named Agencies holds the
  agencies that book them.

Verdict: HALF-BUILT. Two of seven columns and the entire make-good panel have no
data path on the shipped sources.

### 7. Forecasts

Title "Forecast scenarios", promise "Compare revenue-forward, balanced, and
retention-protected plans before committing inventory". Sections: Scenario curve,
Revenue vs retention, Scenario A/B, Daily forecast.

Measured findings:

- The Scenario curve prints `₪1.41M / 95%` on all three rows with three
  visually identical bars. `/api/forecasts` `scenarios` gives revenue_weight 20 ->
  1,414,315.81; 60 -> 1,414,695.20; 90 -> 1,414,695.20. A 4.5x change in the lever
  moves revenue by ₪379.39 (0.027 percent) and retention by 0.0pp.
- Scenario A/B ran live at weights 60 and 85. The button stayed "Running..." for
  more than 30 seconds. `POST /api/scenario-compare` measured at 20.96s and returned
  `a.projected_revenue = 1414695.2`, `b.projected_revenue = 1414695.2`,
  `delta.revenue = 0.0`, `delta.retention = 0.0`, `delta.breaks = 0.0`. Only the
  internal `objective` scalar differs (0.5404 vs 0.3697).
- "Revenue vs retention" here is the same `FrontierPanel` as the Optimizer's, with
  the same numbers, minus the Overview version's scope selector and apply action.
- "Daily forecast" lists 30 rows whose Day column reads `Mon, Mon, Mon, Mon, Tue,
  Tue, Tue, Tue, Wed, Wed`. The payload carries a `date` on every row
  (`{"day":"Fri","date":"2024-11-01",...}`) but the table sorts by weekday index
  and renders `row.day` only (`:3728-3730`, `:3789`). Thirty dated days are shown as
  seven indistinguishable labels.
- Basis clash on one page: the Scenario curve and the frontier are a single
  representative day (₪1.41M), the Daily forecast is 30 saved-plan days, and the
  top-bar tiles that a person just left on Overview said ₪10.12M for a week. The
  page discloses this in prose but the three figures sit within one scroll.

Verdict: HALF-BUILT. The comparison capability is fully wired and returns no
distinguishable answer on this data.

### 8. Events calendar (`#Calendar`)

Two views, Calendar and List. Endpoints: `/api/events`, `/api/pricing`,
`/api/model/audience`. Sections in List view: Operator events, Bundled holidays
(read only), What the model relies on today, Event overlaps: training window and
current plan. 63 active events, list paginated at "Showing 15 of 63".

Findings:

- The page is honest by construction: "Events do not change retention numbers until
  an effect is measured on richer history."
- It carries a real cross-page link, "Open the Pricing page"
  (`CalendarEvents.jsx:330-332`, `setActiveView('Pricing')`), which is the only
  cross-page link of its kind found in the product.
- "What the model relies on today" prints "Weekday pricing premiums (rate-card
  assertions) Sun 1.00 Mon 1.00 Tue 1.00 Wed 1.00 Thu 1.05 Fri 1.15 Sat 1.20". The
  Pricing page's "Day of week" layer holds the identical seven values under a
  different name.
- The month grid opens on the current month (July 2026) which contains none of the
  events, while the "Coming up" strip lists September and October 2026 and the List
  view opens on 2027 holidays.

Verdict: LOAD-BEARING.

### 9. Reports

Five report cards (Weekly traffic plan 8,704 rows, Compliance and guardrails 7 rows,
Revenue forecast 2,391 rows, Daily spot ledger 175 rows, Source file audit 8 rows),
a "Download all" button, and a "Source package" table of 8 files with state, size
and modified time.

Findings:

- "Weekly traffic plan" collides by name with the Schedule toolbar's 200-row JSON
  export (see Schedule).
- "Source file audit" (report) plus "Source package" (table) plus Data > Source
  files plus Overview > Control room are four presentations of the same eight files.

Verdict: LOAD-BEARING. This is where the real CSVs come from.

### 10. Data

Title "Data and model". Three tabs: Upload (default), Source files, Model and
parameters.

Findings:

- Upload tab measured: `.page-header` bottom sits at y=404 while its visible text
  ends near y=240, so roughly 164px inside the header is empty, and another blank
  band separates the tab strip from the intro copy. During that time the tab body
  reads "Loading data inputs...". `/api/uploads/status` measured at 11.34s, so the
  blank-plus-spinner state lasts over eleven seconds on every visit.
- Source files tab prints four tiles: Programmes, Spots, Plan rows, Sources online
  (`:3992-3995`). Overview "Control room" prints the same four values under
  Programmes, Spots, Planned break rows, Available source files (`:3298-3301`).
  Same numbers, two label sets, two pages.
- `/api/overview` `source_counts` is `{"programmes":8704,"spots":50386,
  "planned_break_rows":8704}`. Programmes and planned break rows are the same
  number, 8,704, printed as two separate facts.
- Model and parameters tab renders four panels side by side that visibly collide.
  Observed overlapping runs in the DOM text: "Retention delta per bre**Position
  impact**", "-9.3pp / -1**Length impact**", "Retention delta per brea**Audience
  level stability**".
- That tab uses raw engine keys as user-facing labels: `PrimeShow1`, `PrimeShow2`,
  `pp`, `n=654`, "Drift per week".

Verdict: LOAD-BEARING. Upload is the only ingestion path.

### 11. Advertisers

45 advertiser cards. Endpoints: `/api/advertisers/options` (2.09s),
`/api/advertisers` (23.91s), `/api/advertisers/stats` (12.18s).

Findings:

- The page shows "Loading advertisers..." for about 36 seconds on a cold load.
  `loadAdvertisers` awaits `/api/advertisers` and then awaits `loadStats` inside the
  same `setLoading(true)` block, so `setLoading(false)` cannot fire until both
  serial requests finish (`AdvertisersManager.jsx:94-107`). Verified: after 27
  seconds of waiting across two navigations the page still read "Loading
  advertisers...". A clean load with 50 seconds of waiting resolved.
- Every advertiser is named `Advertiser 1` ... `Advertiser 45` with an `unnamed`
  chip and an id like `ADV_01`.
- Summary tiles read 45 Advertisers, 0 With scoped rules, 0 Scoped rules total,
  0 Conflicts flagged.
- The page carries its own admission: "Honest status: The weekly optimizer does not
  consume advertiser rules; only the daily spot-pricing path prices against them.
  Revenue and profitability are therefore shown as '-' until the daily path
  attribution lands." Revenue and Profitability are `-` on every card.
- The detail drawer is headed "MANAGEMENT AREA / ADV_01", using the raw id as the
  title. Its "ALLOWED POSITIONS" control offers `Any / first / middle / last / gold`,
  lowercase internal enum values shown verbatim.
- The drawer holds "BEHIND-PACE STRENGTH" and "OVER-DELIVERY RESTRAINT", which is a
  third pacing surface alongside Settings > Campaign pacing and Campaigns >
  Make-good alerts.

Verdict: HALF-BUILT. The rules can be authored and saved, and the page states they
do not reach the weekly optimizer.

### 12. Agencies

Endpoints: `/api/advertisers/options`, `/api/agencies` (22.74s),
`/api/agencies/summary` (4.96s). Loads in roughly 10 to 23 seconds behind
"Loading agencies...".

Findings:

- Every agency card carries a "Demo data" chip (AGY_01 through AGY_08 and beyond).
- Totals strip: Net revenue after agency rebates ₪669.98K, Gross revenue ₪699.45K,
  Agency rebates ₪29,472, Priced spots 119, with the basis "the daily ledger
  (Wally_Prime_Reshet_Example_2025-04-27.csv)".
- "Gross revenue ₪699.45K" here versus "Gross revenue ₪40.94M" on Overview. Same
  two words, two bases (one CSV day of the daily ledger versus 30 saved-plan days),
  a 58x difference.
- A real cross-page control exists: "Open the ledger on the Reports page"
  (`AgencyManager.jsx:96-99`, `setActiveView('Reports')`).
- The page states its own reach: "Agency rules affect daily spot pricing, reported
  net revenue, and placement preferences. They do not change the weekly plan, viewer
  retention, or quarter hour settlement."

Verdict: HALF-BUILT. Real records, demo data, no effect on the weekly plan.

### 13. Pricing

Base CPP field plus six premium layers with state chips, and a "Price any slot"
tester. Endpoints: `/api/pricing`, `/api/pricing/price-slot`.

Layer states read from the live chips:

| Layer | Chip | Note printed on the card |
|---|---|---|
| Programme type | Live | "Always applied." |
| Day of week | Live | "Always applied." |
| Specific show | Empty | "No values yet; defaults to 1.0 (no effect)." |
| Position in break | Wired off | multipliers 1.3 / 1.15 / 1.05 / 1 / 1.2 present |
| Ad type | Wired off | warns that activating it zeroes פרומו |
| Calendar events | Live, On | "0 active events carry a price multiplier other than 1.0." |

Findings:

- Four of the six layers move no money today: two are wired off with non-1.0
  multipliers loaded, one is empty, one is live over zero qualifying events.
- The "Price any slot" tester exposes nine inputs (Program class, Weekday, Date,
  Show, Position, Break size, Ad type, Advertiser base, Advertiser, Campaign) but
  the breakdown it returns has three lines: `Base CPP 60.00`,
  `x Programme type (rate card) 1.150`, `x Day of week (rate card) 1.000`,
  `= Final CPP (ILS) 69.00`. Six of the inputs cannot change the answer while their
  layers are off. The card discloses this: "Wired-off layers show struck-through,
  never multiplied into the live total."
- The field is called "Program class" here, "Programme type" on Break Library and
  Schedule, "Class" in the segment inspector, and "Programme type impact" on
  Data > Model.
- The Programme type values here are `News / PrimeShow1 / PrimeShow2 / Other`. The
  Programme type values on Schedule, Optimizer and Break Library are `Reality,
  Documentary, News, Talk Show, Lifestyle, Comedy, Drama, Promo, Special Event,
  Morning Program, Music, Digital, Other`. Two disjoint taxonomies under one name.

Verdict: HALF-BUILT.

### 14. Overrides

Endpoints: `/api/overrides`, `/api/schedule/segments`, `/api/overrides/effect`,
`/api/jobs/recompute`. Left column creates an override (segment search, segment
select, decision select defaulting to "Pin current plan", notes, projected effect);
right column lists current overrides. Live state: "No overrides yet."

Findings:

- The decision list includes marking a segment gold. Gold breaks also have a
  dedicated manager at the bottom of the Schedule page (`GoldBreakManager`,
  `:3492`), a compliance row "Gold breaks per day 0 / 3", a "Gold: No" field in the
  segment inspector, and a `gold` value in the advertiser drawer's allowed
  positions. Five surfaces for one concept.
- The page says "Saving here never triggers the recompute on its own", while the
  segment inspector reached from Break Library and from the Schedule Editor offers a
  "Recompute this day" button in the same flow (`ScheduleInspector.jsx:168`).

Verdict: LOAD-BEARING.

### 15. Kai, AI assistant (`#Assistant`)

A right-hand dock, not a workspace. Endpoints on open: `/api/assistant/status`,
`/api/assistant/thread`, `/api/assistant/conversations`, `/api/assistant/proposals`,
and `/api/assistant/ask/stream` on send.

Findings:

- The header prints the raw model id `claude-opus-5` next to "Connected", and
  "Acting user: auth-disabled".
- The "Conversation" sub-panel clips its own content in the observed layout: the
  heading is cut at the top edge of the scroll area and "Acting user: auth-disabled"
  is half-visible under it.
- The proposal card's restore control writes `window.location.hash = 'Versions'`
  directly (`AssistantPanel.jsx:240` and `:393`) rather than going through
  `setActiveView`.

Verdict: LOAD-BEARING.

### 16. Restore changes (`#Versions`)

`VersionsPage.jsx` (27 lines) wrapping `AssistantVersions.jsx` (400 lines).
Endpoint: `/api/versions`, measured at 40.07s.

Findings:

- The page reads "Loading restore points" for the whole 40 seconds. Observed at 8s
  and still loading.
- Roughly 280px of empty space sits between the intro paragraph and the next line
  of copy while loading.
- This is the third "what happened" surface: `ActivityFeed` (bell icon, in-memory
  toasts only, from `notifications` React state, `:2509-2519`),
  `ActivityLogPanel` (Settings page, `GET /api/activity-log`, `:5299-5330`), and
  this restore-point history.

Verdict: LOAD-BEARING.

### 17. Settings

Sections read live: Market and policy settings, Your channel, Optimizer balance,
Profile, Guardrails, Protected content, Commercial policy, Campaign pacing,
Constraint builder, Existing constraints, Activity log. Endpoints: `PUT
/api/settings`, `/api/optimizer/net-comparison`, `/api/constraints`,
`/api/constraints/options`, `/api/activity-log`.

Findings:

- "What changes with net focus" (`NetComparisonCard`, `:5611`) prints
  `Gross revenue ₪0`, `Retention cost ₪0`, `Net after retention cost ₪0`,
  `Breaks 0`, then states "Switching to net focus will lower the displayed gross and
  raise the net, per the numbers here." `GET /api/optimizer/net-comparison` returns
  `current` and `net_focused` as identical objects (gross 1,414,695.2, retention_cost
  141,224.8, net 1,273,470.4, breaks 80) with `delta` all zeros. The card's own data
  contradicts the sentence next to it.
- "Optimizer balance" is a 0-100 Retention-to-Revenue slider at 60. It is the same
  `revenue_weight` the top bar exposes as "Scenario", the Forecasts page exposes as
  "Scenario A revenue weight" and "Scenario B revenue weight", and the frontier note
  calls "your saved revenue weight". Four names, one lever.
- A "Constraint builder" lives here and a second `ConstraintBuilder` lives inside
  the Schedule Editor (`ScheduleEditor.jsx:3`).

Verdict: LOAD-BEARING.

## Every component under tv-break-dashboard/src

Import graph resolved by grepping every `from './<name>'`. Verdicts are about the
component's role in the shipped app, not about code quality.

### Shell and entry

| File | Role | Reached from | Verdict |
|---|---|---|---|
| `index.jsx` | React root | `index.html:11` | LOAD-BEARING |
| `App.jsx` (8 lines) | renders `TVBreakDashboard` and nothing else | `index.jsx` | LOAD-BEARING, but a pure pass-through |
| `TVBreakDashboard.jsx` (6,236) | shell, top bar, 17 routes, and 12 page components inline | `App.jsx` | LOAD-BEARING |
| `surface-helpers.js` | `API_BASE`, `pageText`, `finiteNumber`, formatters | 29 files | LOAD-BEARING |
| `Login.jsx` (301) | login form | `TVBreakDashboard:2194` only when `auth.status === 'login'` | DEAD on this instance (`/api/auth/me` returns `auth_disabled: true`, so the branch never runs) |

### Overview and money

| File | Role | Endpoint | Verdict |
|---|---|---|---|
| `MoneyWaterfall.jsx` | default export is the presentational waterfall; `YieldMoneyPanel` ("From gross to net") on Overview; `NetComparisonCard` on Settings | `/api/yield-per-second`, `/api/optimizer/net-comparison` | LOAD-BEARING; `NetComparisonCard` HALF-BUILT (renders an all-zero delta under a sentence that promises change) |
| `YieldView.jsx` | "Yield per second" card on Overview | `/api/yield-per-second` | DUPLICATE-OF the same fetch `YieldMoneyPanel` already made on the same page |
| `FrontierScopeChart.jsx` | "Revenue vs retention" with a scope selector and apply-floor action, Overview only | `/api/overview?scope=` | LOAD-BEARING (the only applying copy) |
| `ActivityFeed.jsx` | bell-icon panel over in-memory notifications | none | LOAD-BEARING, but holds no server history |

### Schedule and planner

| File | Role | Verdict |
|---|---|---|
| `ScheduleEditor.jsx` (466) | drag and resize breaks, snap, pin scope, recompute | LOAD-BEARING |
| `ScheduleEditorRow.jsx`, `ScheduleEditorBreak.jsx`, `ScheduleEditorToolbar.jsx` | editor parts | LOAD-BEARING |
| `schedule-track-view.jsx`, `schedule-track.js`, `schedule-editor-format.js` | shared time axis, zoom, segment anchors | LOAD-BEARING |
| `ScheduleInspector.jsx` (353) | segment drawer, used by Break Library and by the Editor | LOAD-BEARING |
| `ScheduleStalenessBanner.jsx` | the "Saved schedule is out of date" band | LOAD-BEARING |
| `BreakChip.jsx` | break chip in timeline and editor | LOAD-BEARING |
| `GoldBreakManager.jsx` | gold-break list at the bottom of Schedule | DUPLICATE-OF the gold decision in Overrides |
| `ConstraintBuilder.jsx` (690) | AND/OR predicate builder, mounted twice (Settings and Schedule Editor) | LOAD-BEARING |

### Advertisers, agencies, pricing

| File | Role | Verdict |
|---|---|---|
| `AdvertisersManager.jsx` (445) | card grid, filters, sort, add | HALF-BUILT (36s serial load; rules do not reach the weekly optimizer) |
| `AdvertiserCardGrid.jsx`, `AdvertiserStatCard.jsx`, `AddAdvertiserForm.jsx` | grid parts | LOAD-BEARING |
| `AdvertiserDetailDrawer.jsx` (385) | per-advertiser workspace | HALF-BUILT |
| `AdvertiserConditions.jsx` (430) | scoped rule editor, also used by the agency drawer | HALF-BUILT (0 scoped rules exist across 45 advertisers) |
| `AdvertiserPricingSummary.jsx` (391) | per-advertiser price preview | LOAD-BEARING |
| `advertiser-*-helpers.js`, `advertisers-helpers.js` | shared naming and stat logic, 19 importers | LOAD-BEARING |
| `AgencyManager.jsx` (316), `AgencyDetailDrawer.jsx` (384), `agencies-helpers.js` | agency records | HALF-BUILT (all cards flagged "Demo data") |
| `PricingManager.jsx` (275) | rate card and six layers | HALF-BUILT (four layers move no money) |
| `PricingSlotTester.jsx` (240) | price any slot | HALF-BUILT (six of nine inputs inert) |
| `PricingEventsLayer.jsx` (128) | calendar-events price layer card | HALF-BUILT (live over 0 qualifying events) |
| `pricing-layers-lib.js` | layer state logic | LOAD-BEARING |

### Calendar

| File | Role | Verdict |
|---|---|---|
| `CalendarEvents.jsx` (344) | page shell, Calendar / List switch | LOAD-BEARING |
| `CalendarMonthGrid.jsx` (209), `CalendarEventsList.jsx` (362) | the two views | LOAD-BEARING |
| `CalendarHolidays.jsx` (103) | read-only bundled holidays | LOAD-BEARING |
| `CalendarEventsModel.jsx` (405) | "What the model relies on today", plus `PlanEventBadges` and `usePlanEvents` used by the shell | LOAD-BEARING |
| `CalendarAudienceModel.jsx` (162) | audience-model disclosure, `/api/model/audience` | LOAD-BEARING |
| `calendar-events-lib.js` | date and overlap math | LOAD-BEARING |

### Assistant

| File | Role | Verdict |
|---|---|---|
| `AssistantDock.jsx`, `AssistantPanel.jsx` (398), `AssistantThread.jsx` | the dock | LOAD-BEARING |
| `AssistantProposalCard.jsx` (361) | apply or reject proposed changes | LOAD-BEARING |
| `AssistantUpload.jsx` (149) | upload from inside the dock | DUPLICATE-OF `UploadCenter` on the Data page |
| `AssistantConversations{Api,Sidebar,Rail,Changes}.jsx` | conversation list, restore, changes view | LOAD-BEARING |
| `AssistantVersions.jsx` (400) | the restore-point list rendered by `VersionsPage` | LOAD-BEARING |
| `assistant-stream.js`, `assistant-panel-state.js`, `assistant-page-context.js` | SSE and page-context plumbing | LOAD-BEARING |

### Other

| File | Role | Verdict |
|---|---|---|
| `OverrideConsole.jsx` (450), `override-console-lib.js` | Overrides page and the shared day-recompute job runner | LOAD-BEARING |
| `UploadCenter.jsx` (325) | Data > Upload, always mounted with `embedded` | LOAD-BEARING |
| `VersionsPage.jsx` (27) | thin wrapper around `AssistantVersions` | LOAD-BEARING |
| `ScenarioCompare.jsx` (215) | Forecasts Scenario A/B | HALF-BUILT (21s round trip, zero delta) |
| `MakeGoodAlerts.jsx` (141) | Campaigns under-delivery panel | HALF-BUILT (no campaign flight rows exist) |
| `DateField.jsx` (25) | shared date input | LOAD-BEARING |
| 15 `.css` files | styling | not evaluated |

## The same concept under different names

Each row is one concept and the names it wears on different surfaces.

| Concept | Names in the product |
|---|---|
| Optimizer revenue weight | "Optimizer balance" (Settings slider), "Scenario: Balanced / Revenue priority / Retention guardrail" (top bar, maps to 85 / 35 / saved at `:1852`), "Scenario A revenue weight" and "Scenario B revenue weight" (Forecasts), "your saved revenue weight" (frontier note), "Engine focus: Balanced / Net focused" (Settings) |
| `risk_lambda` | "Caution level 0/100" (top bar), "Conservative / Reports at the worst plausible retention cost" (Settings), `risk_lambda` in the retention-cost prose on Overview |
| Programme class | "Programme type" (Break Library, Schedule, Optimizer), "Program class" (Pricing tester), "Class" (segment inspector), "Programme type impact" (Data > Model, with a disjoint value set) |
| Inventory | "Inventory" nav page (spot supply, money is `-`), "Inventory" tab inside Optimizer (per-channel plan revenue), "Daypart inventory heatmap" (Optimizer, empty), "Inventory by broadcast daypart" (Inventory page) |
| Weekly traffic plan | Schedule toolbar button (200-row JSON), Reports card (8,704-row CSV), inspector export dropdown option (produces a break-detail JSON) |
| Source-file inventory | Overview "Control room", Data > "Source files", Reports > "Source package", Reports > "Source file audit" |
| Plan row count | "Programmes 8,704" and "Planned break rows 8,704" (Overview), "Plan rows 8,704" (Data), "8,704 rows" (Reports) |
| Gold break | Overrides decision, Schedule "Gold breaks" manager, compliance row "Gold breaks per day", inspector field "Gold: No", advertiser allowed-position `gold` |
| Change history | "Restore changes" (`#Versions`), "Activity log" (Settings), bell-icon notification feed |
| Pacing | "Campaign pacing" (Settings), "Make-good alerts / Under-delivery risk" (Campaigns), "Behind-pace strength" and "Over-delivery restraint" (advertiser drawer) |
| Revenue vs retention frontier | Overview `FrontierScopeChart`, Optimizer `FrontierPanel`, Forecasts `FrontierPanel`, all titled "Revenue vs retention" |
| Day-of-week price premium | Pricing "Day of week" layer, Calendar "Weekday pricing premiums (rate-card assertions)" |
| Advertiser | Campaigns "Advertiser demand" (column blank), Advertisers page, Agencies "the advertisers it books", Pricing "Advertiser base" and "Advertiser (optional)" |
| Constraint builder | Settings section, Schedule Editor panel (same component) |
| Upload | Data > Upload tab, `AssistantUpload` inside the dock |

## Numbers that differ across pages or carry different bases

All figures below were read on the same instance within one session, with no
recompute in between.

| Figure | Overview | Optimizer | Forecasts | Agencies | Inventory |
|---|---|---|---|---|---|
| "Weekly projected revenue" | ₪10.12M (7 days, רשת 13) | ₪10.12M (same tile) | not shown | not shown | not shown |
| "Gross revenue" | ₪40.94M (30 days, רשת 13) | not shown | not shown | ₪699.45K (one daily ledger CSV) | not shown |
| רשת 13 plan revenue | not shown | ₪10.81M (Inventory tab) | not shown | not shown | "Booked value -" |
| "Current plan revenue" | ₪1.41M (frontier) | ₪1.41M (frontier) | ₪1.41M (frontier and scenario curve) | not shown | not shown |
| Retention | 94.4% (tile), 78.6% (compliance floor row) | 94.4%, 78.6%, 85.4% (inspector) | 95% (frontier) | not shown | not shown |

Two of these are measurable contradictions rather than disclosed basis shifts.

1. On the Optimizer page, the headline tile says the channel's week is ₪10.12M and
   the page's own Inventory tab says רשת 13 is ₪10.81M. Verified against the API:
   `/api/overview` `summary.week.projected_revenue = 10,123,070.80` for
   `2024-11-01` to `2024-11-07`; summing `program.revenue` across the `רשת 13` row
   of `/api/schedule` gives `10,809,546` over the same seven dates
   (`2024-11-01` to `2024-11-07`, 524 programmes). A gap of ₪686,475, or 6.8
   percent, between two figures on one page for one channel and one week, with no
   basis note on the sub-view.
2. The Inventory page prints `-` for daypart Revenue while Overview prints revenue
   for the same five daypart names, summing to ₪40.95M against Overview's own
   "Gross revenue ₪40.94M".

Related bases worth flagging even where disclosed:

- The Optimizer and Schedule grids show competitor-channel money (`קשת 12` sums to
  ₪28.23M, `כאן 11` to ₪6.23M, `עכשיו 14` to ₪9.86M in the same payload), while
  every headline figure and the Inventory page are scoped to `רשת 13` only.
- The Optimizer's default selected break is on `כאן 11` and shows "Plan revenue
  ₪16,141" for it, while Overrides states that only owned-channel segments can be
  steered.

## Controls that lead nowhere

| Control | Where | What it actually does | Evidence |
|---|---|---|---|
| "Nov 1" with a chevron, styled as a date picker | top bar on Overview, Optimizer, Schedule | navigates to `#Schedule`, offers no date choice | clicked live: hash went `#Overview` to `#Schedule`; `TVBreakDashboard.jsx:2352-2362` |
| "Compare" with a compare icon | top bar on the same three pages | navigates to `#Forecasts` | clicked live: hash went `#Overview` to `#Forecasts`; `:2406-2417` |
| Inspector export scope dropdown, 3 options | Optimizer inspector | all three write `kairos-break-detail.json`; only an `exportScope` string changes | `:4886-4890` calling `:2071-2074` |
| Position, Break size, Ad type, Show inputs in "Price any slot" | Pricing | cannot change the result while their layers are wired off or empty | live breakdown showed only Base CPP x Programme type x Day of week |
| "Position in break" multipliers 1.3 / 1.15 / 1.05 / 1.2 | Pricing | loaded, chip says "Wired off" | live chip |
| "Ad type" multipliers | Pricing | loaded, chip says "Wired off", and the card warns activating it would zero פרומו | live chip and warning |
| "Specific show" layer | Pricing | "No values yet; defaults to 1.0 (no effect)" | live copy |
| "Calendar events" layer, live and On | Pricing | "0 active events carry a price multiplier other than 1.0" | live copy |
| Scenario A/B "Compare" | Forecasts | 21s round trip returning `delta.revenue = 0.0` at weights 60 vs 85 | measured `POST /api/scenario-compare` |
| Scenario select (3 options) | top bar | at weights 20 / 60 / 90 the forecast payload differs by ₪379.39 total | `/api/forecasts` `scenarios` |
| "Daypart inventory heatmap" | Optimizer | hard-coded empty state, no data path in code | `:5159-5172` |
| Advertiser scoped-rule authoring | Advertisers | the page states the weekly optimizer does not consume these rules | live banner |

## Text that references a capability with no link

Six of the seven cross-surface references found in source are prose only. Two are
real controls.

| Text | File | Linked |
|---|---|---|
| "Full net in the from gross to net card." | `YieldView.jsx:117` | no |
| "Not channel data: advertiser terms (also editable in the Advertisers screen)..." | `UploadCenter.jsx:37` | no |
| "Set your owned channel in Settings to scope the frontier to a single channel." | `FrontierScopeChart.jsx:286` | no |
| "Schedule-wide checks (ad minutes, spacing, protected content) live in the compliance ledger below." | `TVBreakDashboard.jsx:4849` | no |
| "An operator assertion... it affects the forecast only while the events layer is activated on the Pricing page." | `CalendarEventsList.jsx:109` and `:166` (tooltips) | no |
| "Upload campaign_flights.csv with start and end dates and delivery goals" | Campaigns make-good panel | no; the upload slot is on Data > Upload |
| "Open the Pricing page" | `CalendarEvents.jsx:330-332` | yes, `setActiveView('Pricing')` |
| "Open the ledger on the Reports page" | `AgencyManager.jsx:96-99` | yes, `setActiveView('Reports')` |

## Pages that require the engine's vocabulary

| Surface | Terms a reader must already know |
|---|---|
| Overview, gross-to-net card | "risk_lambda", "ci_low", "ci_high", "retention_cost_high", "calibrated 95 percent coefficient interval bounds", "the same interval seam" |
| Overview and Optimizer, frontier | "sweeps the retention floor at your saved revenue weight", "refined single representative-day optimum", "Net focused" |
| Optimizer top bar | "Caution level 0/100" with no unit, sitting next to a separate 0-100 "Optimizer balance" in Settings |
| Data > Model and parameters | "pp", "Retention delta per break", "Drift per week", "n=654", "PrimeShow1", "PrimeShow2", "Audience level stability" |
| Break Library segment inspector | "Baseline audience (TVR) 16.2", "Base rate 57.6", "Likely range (credible interval)", "Measurement confidence: Medium" |
| Break Library table | "Position: middle", "Type: medium" (raw enum values as cell text) |
| Advertiser drawer | title is the raw id "ADV_01"; "ALLOWED POSITIONS: Any / first / middle / last / gold"; "Behind-pace strength"; "Over-delivery restraint" |
| Overrides | "carrying the segment anchor so the decision survives a re-ingest", "An override changes a fingerprinted input" |
| Assistant dock | raw model id "claude-opus-5", "Acting user: auth-disabled" |
| Settings | "Engine focus", "objective_mode", "retention_basis: tvr_weighted" implied by the basis lines |
| Calendar | "Retention measurement mode: Global baseline, calendar-blind" |

## Reachability and screens to traverse

Cold start is `http://127.0.0.1:8010/`, which lands on Overview with the sidebar
already rendered. The sidebar is persistent on every workspace, so all seventeen
entries are one click deep. Depth below counts screens a person must land on,
starting from that cold Overview.

| Capability | Path | Screens |
|---|---|---|
| Read the week's revenue, retention, ad minutes, risk | Overview | 0 |
| Read the gross-to-net waterfall | Overview | 0 |
| Read yield per second by daypart and programme | Overview | 0 |
| Apply a frontier retention floor | Overview, click a point, Apply | 0 |
| Open a priority decision as a break | Overview, click a row (lands on Optimizer) | 1 |
| Approve or reject a break recommendation | Optimizer | 1 |
| Read the weekly plan as a grid, daypart or timeline | Schedule or Optimizer | 1 |
| Download the weekly plan CSV | Schedule, or Reports | 1 (two independent paths) |
| Move or resize a break | Schedule, Editor tab, click a break | 3 |
| Build a scoped constraint | Settings (section), or Schedule, Editor tab, Constraint builder | 1 or 3 |
| Create an override | Overrides | 1 |
| Read spot supply and hourly pressure | Inventory | 1 |
| Read the ranked break shelf | Break Library | 1 |
| Open a break's full detail | Break Library, click a row | 2 |
| Compare two revenue weights | Forecasts, Scenario A/B, Compare | 1 |
| Manage a calendar event | Events calendar, List, expand a row | 2 |
| Download any report CSV | Reports | 1 |
| Upload a source file | Data (Upload is the default tab) | 1 |
| See what the model learned | Data, Model and parameters tab | 2 |
| Edit an advertiser rule | Advertisers, wait about 36s, click a card | 2 |
| Edit an agency rebate | Agencies, wait about 10 to 23s, click a card | 2 |
| Edit the rate card | Pricing | 1 |
| Price a single slot | Pricing (right column) | 1 |
| Roll back to an earlier state | Restore changes, wait about 40s | 1 |
| Ask the assistant | bot icon, sidebar entry, or `#Assistant` | 0 (dock overlays the current page) |
| Change the operating profile | Settings | 1 |
| Read the server-side activity log | Settings, scroll to the last section | 1 |

Surfaces reachable only by knowing a URL: none. The only URL-driven state besides
the seventeen hashes is `?axis=day|daypart|hour|type` (`:596-602`), and all four
values are reachable from the Days / Dayparts / Hours / Formats control
(`GridAxisControl`, `:4282-4300`). The `?bl=2` query string seen on a stale tab is
read by nothing in `src`.

Surfaces unreachable on this instance: `Login.jsx`, `ChangePasswordDialog`,
`AccountsDialog`, the operator card and its user menu, all gated on
`auth.status`, which cannot leave `ready` while `/api/auth/me` returns
`auth_disabled: true`.

Depth caveat: the top-bar command group (date button, Scenario select, Caution level
slider, Compare, Run Optimization, Apply to weekly schedule) renders only on
Overview, Optimizer and Schedule (`showOptimizationControls`, `:1632`). On the other
fourteen surfaces the primary actions are absent, so a person mid-task on Pricing or
Advertisers must navigate back to one of three pages to run or apply anything.

## Measured endpoint latency

Single `curl` per endpoint against `http://127.0.0.1:8010`, warm server. Every one
returned HTTP 200 except where noted.

| Endpoint | Time | Size |
|---|---|---|
| `/api/versions` | 40.07s | 9,085 B |
| `/api/advertisers` | 23.91s | 10,652 B |
| `/api/agencies` | 22.74s | 8,340 B |
| `POST /api/scenario-compare` | 20.96s | 794 B |
| `/api/advertisers/stats` | 12.18s | 13,707 B |
| `/api/uploads/status` | 11.34s | 6,524 B |
| `/api/overview` | 5.84s | 14,951 B |
| `/api/yield-per-second` | 5.01s | 4,123 B |
| `/api/agencies/summary` | 4.96s | 523 B |
| `/api/parameters` | 2.96s | 2,447 B |
| `/api/advertisers/options` | 2.09s | 19,160 B |
| `/api/files` | 2.00s | 885 B |
| `/api/reports` | 1.93s | 496 B |
| `/api/compliance` | 1.93s | 1,822 B |
| `/api/events` | 1.71s | 28,159 B |
| `/api/optimizer/net-comparison` | 1.66s | 551 B |
| `/api/forecasts` | 1.46s | 3,452 B |
| `/api/break-operations` | 1.38s | 33,949 B |
| `/api/make-good-alerts` | 1.36s | 134 B |
| `/api/break-library` | 1.31s | 41,874 B |
| `/api/pricing` | 1.29s | 2,105 B |
| `/api/activity-log` | 1.28s | 17,869 B |
| `/api/gold-breaks` | 1.20s | 169 B |
| `/api/overrides` | 1.16s | 198 B |
| `/api/inventory` | 1.08s | 1,940 B |
| `/api/schedule` | 1.06s | 516,470 B |
| `/api/campaigns` | 0.98s | 8,505 B |
| `/api/model/audience` | 0.92s | 2,334 B |
| `/api/impact` | 0.77s | 8,257 B |
| `/api/constraints` | 0.75s | 259 B |
| `GET /api/scenario-compare` | 1.90s | HTTP 404 (POST only) |

The shell fires eleven of these in parallel on every cold load
(`TVBreakDashboard.jsx:1287-1297`).

## Verdict roll-up

Navigation entries: 10 LOAD-BEARING, 2 DUPLICATE-OF, 5 HALF-BUILT, 0 DEAD.

Components under `tv-break-dashboard/src`: 80 files. Of the 52 `.jsx`, one is DEAD
on this instance (`Login.jsx`, auth disabled), three are DUPLICATE-OF another
surface (`YieldView.jsx`, `GoldBreakManager.jsx`, `AssistantUpload.jsx`), and ten
are HALF-BUILT (`AdvertisersManager`, `AdvertiserDetailDrawer`,
`AdvertiserConditions`, `AgencyManager`, `AgencyDetailDrawer`, `PricingManager`,
`PricingSlotTester`, `PricingEventsLayer`, `ScenarioCompare`, `MakeGoodAlerts`).
One function inside `TVBreakDashboard.jsx` is DEAD by construction:
`InventoryHeatmap` (`:5159-5172`).

The three heaviest structural facts for the rebuild:

1. Optimizer and Schedule render the same three planner views over the same data
   from the same components, and Optimizer additionally re-renders Overview's
   headline tiles, compliance ledger and frontier. Two of the first three nav
   entries are mostly the same page.
2. One number, "revenue", appears on five surfaces at five bases (₪10.12M weekly,
   ₪40.94M monthly, ₪10.81M grid-summed, ₪1.41M representative day, ₪699.45K daily
   ledger) with one measurable internal contradiction of ₪686,475 on a single page.
3. Five surfaces (Campaigns, Forecasts, Advertisers, Agencies, Pricing) each state,
   in their own copy, that the thing they are for does not currently move the plan.
