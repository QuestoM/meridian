# Campaign reference notes: digital ads model vs television

Source read: `/Users/home/Code/experiments/ads/convex/schema.ts`, `campaigns.ts`, `delivery.ts`, `deliveryStore.ts`.

The source is a digital display/video campaign tracker for id-x (an Israeli DSP). Most of its measurement vocabulary (impressions, clicks, CTR, CPM, viewability, completion rate) is digital-only and does not transfer. What does transfer is the structural thinking: how a campaign record is split into a cheap summary and a heavy detail, how a flight is bounded in time with computed running-state fields, how delivery-to-date is expressed as a time-series of actuals against a goal, and how creative assets attach to a campaign via line items rather than directly.

---

## 1. Split model: summary vs detail

**Digital pattern.** `campaigns` holds a slim summary row (name, status, dates, budget cap, brand/company/category denormalised strings, aggregated predictions). `campaignDetails` holds the full nested JSON, loaded only when the operator opens a single campaign. The list page never reads `campaignDetails`. This keeps list pagination cheap and avoids hitting document-size limits.

**Television equivalent.** The same split applies and for the same reasons. A campaign list page needs: name, status, advertiser, agency, flight dates, total budget, spots booked vs spots delivered so far. The heavy payload (per-break placement constraints, creative versions, per-daypart targets, historical pacing curve) is loaded only on the detail drawer. Implement as two database rows or as a summary JSON plus a details JSON, keyed by the same `campaign_id`.

---

## 2. Identity hierarchy: brand, company, agency

**Digital pattern.** The source carries `brandName` (the advertising brand), `companyName` (the legal entity behind the brand), `userName` (the campaign manager's name inside the buying platform, denormalised for facet filtering), and a `customerSuccessName` for the operator's own CS contact. There is no agency object; the buyer is an internal platform user.

**Television requirement (owner-stated).** Mako builds campaigns for clients who have no agency. In our product a client must never appear as agency-less: every client gets an agency record, and when the client is buying direct, that record is labelled "משרד עצמי" (in-house). The identity hierarchy for television is therefore:

| Level | Field | Notes |
|---|---|---|
| Agency | `agency_id`, `agency_name` | Mandatory. Use "משרד עצמי" when client buys direct. |
| Client | `client_id`, `client_name` | The brand or legal entity paying for the campaign. |
| Brand | `brand_name` | The advertised brand, which may differ from the legal client name (e.g. Ferrero / Kinder). |
| Category | `category_name` | Advertising category (food, automotive, finance, ...). Used for audience and competitor analysis. |

The `company` / `brand` / `category` objects in the digital source (loaded from `campaignDetails.data`) are a good model: the summary row denormalises the names as strings for fast filtering; the detail row carries the full objects with ids and metadata.

---

## 3. Flight: time bounds and running-state fields

**Digital pattern.** Every campaign carries:

- `startDate`, `endDate` — unix-ms epoch boundaries of the flight.
- `campaignDays` — total length in days, pre-computed.
- `daysFromStart` — days elapsed since start (computed at sync time, not stored as a permanent fact).
- `daysToEnd` — days remaining.
- `relativePeriod` — fraction elapsed (0–1), used for pacing rate calculations.

These are denormalised from the live API at sync time and stored so the dashboard can show "X days into a Y-day campaign" without a live calculation.

**Television equivalent.** Carry the same fields. In television the flight is anchored to broadcast weeks (Sunday–Saturday in the Israeli market). Add:

- `flight_start_date` — ISO date (YYYY-MM-DD), Sunday of the opening week.
- `flight_end_date` — ISO date, Saturday of the closing week.
- `flight_weeks` — integer count of broadcast weeks.
- `weeks_elapsed`, `weeks_remaining` — computed at last recompute, stored for list display.
- `relative_period` — fraction of flight elapsed; the pacing engine uses this to compare delivery rate to goal rate.

Do not store `daysFromStart` / `daysToEnd` as permanent facts in the database; recompute them from `flight_start_date` and today's date when serving the list. They are display helpers, not booking data.

---

## 4. Budget and goal

**Digital pattern.**

- `budgetCap` — the primary spend ceiling for the campaign.
- `compositeBudgetCap` — the total ceiling across all distributors when the campaign runs on multiple publishers; may exceed `budgetCap` if bonus is included.
- `bonusCap` — bonus (make-good or added-value) budget above the paid cap.
- `bookedImpressions` — the contracted delivery goal in impression units.

The distinction between `budgetCap` (what the client pays) and `compositeBudgetCap` (what can actually be delivered including bonus) is important: the delivery engine targets the composite, but billing stops at the paid cap.

**Television equivalent.**

- `budget_ils` — agreed spend in ILS (the primary contractual figure). Mandatory.
- `bonus_ils` — make-good or added-value spend above the contract; shown separately in yield reporting.
- `total_budget_ils` — `budget_ils + bonus_ils`; this is what the scheduler targets.
- `spots_booked` — contracted number of spots (the television equivalent of `bookedImpressions`).
- `grp_target` — target GRP (gross rating points) when the contract is GRP-based rather than spot-count-based; optional, omit when not applicable.

Do not carry a single `budgetCap` that conflates paid and bonus. Keep them separate from day one; bonus spots must be labeleld as such in the break board.

---

## 5. Price model and delivery type

**Digital pattern.** `priceModel` (CPM, CPC, flat, ...), `deliveryType` (standard, accelerated), `blockType` (display, video, ...), `priorityMode` (guaranteed, house, remnant).

**Television equivalent.**

- `price_model` — `"cpp"` (cost per rating point) or `"flat"` (fixed price per spot). These are the only two models in Israeli broadcast.
- `priority` — `"guaranteed"` (booked into the plan and protected) or `"preemptible"` (can be displaced by a higher-priority campaign). Equivalent to `priorityMode`.
- `spot_type` — `"commercial"` (standard 30 s), `"sponsorship"` (programme sponsorship bumper), `"promo"` (operator's own promotion). Do not conflate; they have different pricing multipliers.

`deliveryType` (accelerated vs standard pacing) transfers directly: call it `pacing_mode` with values `"even"` or `"front_loaded"`.

`blockType` / `audienceType` / `audienceId` — do not transfer. In television the audience is defined by the programme and daypart; there is no digital-style audience segment targeting.

---

## 6. Creative assets

**Digital pattern.** `blockCreativeSets` is a nested array on `campaignDetails.data`, loaded lazily. Each creative set contains one or more line items, each with a `creativeSize` (banner dimensions), a `lineItemName`, a DFP link, and delivery data (impressions, cost). Creative metadata lives on the detail, not the summary.

**Television equivalent.** A television creative is a spot: a video file with a declared duration (15 s, 20 s, 30 s, 45 s, 60 s). The campaign carries a list of spot versions:

```
spot_versions: [
  { version_id, title, duration_sec, material_code, status }
]
```

`material_code` is the broadcast-house asset identifier (equivalent to `lineItemDfpLink`). `status` is `"approved"` | `"pending"` | `"rejected"`. Creative approval is a hard gate; a spot cannot be placed until approved.

Keep `spot_versions` on the detail row, not the summary. The summary row carries only `spot_count` and `has_pending_approval` (boolean) for the list badge.

The digital `byCreativeSize` delivery breakdown has no television equivalent. Spots do not have sizes; they have durations. If you need a delivery breakdown by creative, key it by `version_id` and `duration_sec`.

---

## 7. Delivery-to-date: structure and granularity

**Digital pattern.** The `campaignDelivery` table holds a campaign-level summary (totals + three pre-aggregated breakdowns: by creative size, by distributor, by date). The `campaignDeliveryRows` table holds line-item-by-day detail rows (one row per line-item × day × creative-size × distributor). The summary is written on every sync; rows are written only when the user opens the expanded delivery view. The summary is sufficient for charts; the rows power drill-down tables.

**Television equivalent.** In television, delivery is measured in spots aired and GRPs delivered. The delivery record should carry:

```
campaign_delivery (summary, one row per campaign):
  spots_planned       -- from the current approved plan
  spots_aired         -- confirmed by the traffic log
  spots_remaining     -- spots_planned - spots_aired
  grp_delivered       -- sum of rating points for aired spots (from viewership data)
  grp_remaining       -- grp_target - grp_delivered (when target exists)
  spend_to_date_ils   -- cost of aired spots at agreed rates
  last_updated_at

campaign_delivery_rows (detail, one row per aired spot):
  campaign_id
  break_id
  broadcast_date      -- ISO date
  broadcast_time      -- HH:MM
  channel             -- operator's own channel only
  programme_title
  daypart
  duration_sec
  version_id          -- which creative ran
  rating_achieved     -- actual viewership % from measurement house
  cost_ils            -- settled cost for this spot
  is_bonus            -- boolean; bonus spots must be flagged
  source              -- "traffic_log" | "manual" | "estimated"
```

The `byDate` breakdown from the digital source transfers directly: aggregate `spots_aired` and `grp_delivered` by `broadcast_date` to draw a pacing chart.

The `byDistributor` breakdown has no television equivalent (a broadcast spot runs on one channel, the operator's own). Drop it.

`impressions`, `viewableImpressions`, `clicks`, `ctr`, `cpm`, `cpv`, `viewabilityRate`, `completionRate` — all digital-only. Drop all of them.

---

## 8. Pacing and progress

**Digital pattern.** `relativePeriod` (fraction of flight elapsed) is the anchor. The dashboard compares `totalPredictedImpressions` (the plan's forecast) against actuals from `campaignDelivery.totals.impressions`. The delivery status `internalDeliveryStatus` is a separate field from the booking status (`status`) because a campaign can be `"Active"` (booking) but `"Underdelivering"` (delivery).

**Television equivalent.** Maintain the same two-status pattern:

- `booking_status` — `"draft"` | `"confirmed"` | `"paused"` | `"cancelled"` | `"completed"`. This is the CRM state.
- `delivery_status` — `"on_track"` | `"ahead"` | `"behind"` | `"at_risk"` | `"unknown"`. Computed from `spots_aired / spots_planned` vs `relative_period`. Do not compute this on the fly in the list query; materialise it at each delivery sync.

Pacing rate: `(spots_aired / spots_planned) / relative_period`. A rate below ~0.85 triggers `"behind"`; above ~1.15 triggers `"ahead"`. Thresholds are owner-configurable.

---

## 9. Agency-less client rule

The owner stated explicitly: Mako builds campaigns for clients who have no agency. In the television product, every campaign must have an agency record. When the client is buying direct, create an agency record with:

```
agency_id:   <auto>
agency_name: "משרד עצמי"
agency_type: "in_house"
client_id:   <the client's id>
```

This record must be labelled visibly wherever it appears in the UI so that an operator can distinguish a direct-client campaign from a genuinely agency-represented one. The label "משרד עצמי" should appear as-is in the agency column; do not suppress it or replace it with the client name.

---

## 10. Fields that do not transfer at all

| Digital field | Reason excluded |
|---|---|
| `bookedImpressions` | Impression-based buying does not exist in Israeli broadcast. Use `spots_booked`. |
| `distributorNames`, `distributorShares` | A broadcast spot runs on one channel. No distributor split. |
| `audienceId`, `audienceName`, `audienceType`, `externalAudienceId` | Segment targeting is digital. Television audience is a programme + daypart attribute, not a campaign attribute. |
| `impressions`, `viewableImpressions`, `clicks`, `ctr`, `cpm`, `cpv`, `viewabilityRate`, `completionRate` | Digital engagement metrics. |
| `creativeSize` (banner dimensions) | No banner sizes in television. Use `duration_sec`. |
| `lineItemDfpLink` | DFP is a digital ad server. Use `material_code` for broadcast asset reference. |
| `totalUniqueUsers` | Reach in digital sense. In television use GRP or net reach from the measurement panel. |
| `blockType` (display, video, somplo) | Digital channel types. |
| `domainIntel`, `landingIntel`, `trackers`, `gtmIntel` | Web / performance marketing intelligence. Irrelevant in television. |
| `adAccounts` (Google / Meta transparency) | Competitive digital intelligence. Not part of broadcast campaign management. |
| `providerMetrics` (AdClarity) | Digital spend estimates. |

---

## 11. Fields worth carrying verbatim or with minor renaming

| Digital field | Television field | Change |
|---|---|---|
| `campaignId` | `campaign_id` | Rename to snake_case. |
| `revision` | `revision` | Keep; use for optimistic locking and change history. |
| `name` | `campaign_name` | Rename. |
| `status` | `booking_status` | Rename; separate from delivery status. |
| `internalDeliveryStatus` | `delivery_status` | Rename. |
| `startDate`, `endDate` | `flight_start_date`, `flight_end_date` | Rename; store as ISO date string, not epoch ms. |
| `campaignDays` | `flight_weeks` | Convert to weeks; Israeli broadcast plans in weeks. |
| `relativePeriod` | `relative_period` | Keep concept; recompute at each delivery sync. |
| `budgetCap` | `budget_ils` | Rename; always ILS. |
| `bonusCap` | `bonus_ils` | Rename. |
| `brandName` | `brand_name` | Keep as denormalised string on summary row. |
| `companyName` | `client_name` | Rename; "company" is ambiguous. |
| `categoryName` | `category_name` | Keep. |
| `priceModel` | `price_model` | Keep concept; values change (cpp / flat). |
| `priorityMode` | `priority` | Rename; values change (guaranteed / preemptible). |
| `isTest` | `is_test` | Keep; essential for filtering demo and test campaigns out of real reporting. |
| `lastSyncedAt` | `last_updated_at` | Rename. |
