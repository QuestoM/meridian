# Campaign and clients current state

Survey date: 2026-08-02. Read-only. Nothing was changed.

---

## 1. The failure the owner sees

### What the code does

`ClientsWorkspace.jsx` lines 120-142 run:

```js
Promise.allSettled([loadClients(), loadMoney(), loadCampaigns(), loadAdvertiserRules()])
```

The four calls are:

| slot | function | endpoint |
|------|----------|----------|
| `clients` | `loadClients()` | `GET /api/clients` |
| `ledger` | `loadMoney()` | `GET /api/clients/money` |
| `booked` | `loadCampaigns()` | `GET /api/clients/campaigns` |
| `rules` | `loadAdvertiserRules()` | `GET /api/advertisers` |

When any slot has `status === 'rejected'`, line 134 sets the banner:

> "Part of this page could not load. What is missing is a failure, not an empty result."

The banner carries none of the four names. A person reading it cannot know which call failed or why.

### Which one actually fails, and why

Probed all four endpoints against the running server at `:8000` without a session cookie.

Every single one returns:

```json
{"detail": "A signed-in session is required."}
```

with HTTP 401.

The source is `kairos_api/auth.py` lines 88-103: the `enforce_request` middleware intercepts every `/api/` path when `auth_active()` is true and returns a `JSONResponse(status_code=401)` before any route handler runs. The only paths that are exempt are `/api/auth/login` and `/api/health` (`PUBLIC_API_PATHS`, line 48).

So: when a session cookie expires or is absent, all four slots are rejected simultaneously. The banner fires correctly but names nothing.

### The defect

The defect is not that the banner fires; it is that `broken` (the filtered list of rejected results, line 132) is counted but never inspected. The names of the failed slots (`clients`, `ledger`, `booked`, `rules`) are already available in the destructured result at line 127; none of them reach the error string. A person cannot fix what the product will not name.

The path to fixing it: collect the slot names of every rejected result and include them in the `setFailed` call. The slot names are known at the call site because the results array is positional.

---

## 2. The advertisers: names file versus what the surface shows

### What the owner sees

The Advertisers surface shows entries labelled "ADV_01" through "ADV_45" (45 rows). The owner reports these appear as numbers 1 to 50.

### Where the numbering comes from

`data/advertiser_rules.csv` (45 data rows plus one header = 46 lines). Every row's `advertiser_id` column contains `ADV_01` through `ADV_45`. The `name` and `display_name` columns are empty for all 45 rows.

The Advertisers tab (`AdvertiserRecordsPanel.jsx` line 120) calls `GET /api/advertisers`, which is the rules store read (`kairos_api/advertisers.py` line 266, `list_advertisers()`). That read returns each row's `advertiser_id`, `name`, `display_name` and a computed `name_source`.

On the frontend, `displayNameOf()` (`advertiser-name-helpers.js` line 59) resolves the shown name:
1. Uses `display_name` if present.
2. Uses `name` (the bound advertiser name) if present.
3. Falls back to `prettifyRawId(raw, locale)`.

`prettifyRawId()` (line 41) tests the id against `SEED_ID_PATTERN = /^ADV[_-]?0*(\d+)$/i`. When it matches, it returns the raw id unchanged. So `ADV_01` is shown as `ADV_01`, not as "Advertiser 1" or any real company name.

The "numbers 1 to 50" the owner describes is this: the rows are rendered as their seed ids `ADV_01`...`ADV_45` because no name has been bound to any of them.

### Where the names file is

`data/advertiser_names.csv` exists and holds 41 Hebrew advertiser names (all `source=observed`, `first_seen=2025-04-27`). These names are the real advertisers observed in the daily spot data.

This file is read only by:
- `kairos/optimize/advertiser_rules_identity.py` via `load_advertiser_names()`
- `kairos_api/advertisers_identity.py` via `identity_report()`
- `kairos_api/campaigns_read_clients.py` via `_identity_index()`

None of those paths feed `GET /api/advertisers`. The names file is not wired into the Advertisers surface. The Advertisers surface is the rules store; the names+rules join is `GET /api/advertisers/identity`, which the Advertisers tab does not call.

`AdvertiserRecordsPanel.jsx` line 107-115 does call `loadAdvertiserIdentity()` (`GET /api/advertisers/identity`) in a separate `loadIdentity()` callback, and the result feeds `identityIndex`. This index is merged onto each row via `mergeRowWithIdentity()` before the cards render. So the identity read is wired. The gap is downstream: `displayNameOf()` reads from `row.display_name` and `row.name` (the rules-store fields), not from the identity record's `shown_name`. The `identityIndex` is used by `mergeRowWithIdentity()` in `advertiser-stats-helpers.js`; that merge must be confirmed to write the name fields the card will read.

### Summary of the gap

The 41 observed Hebrew names exist in `data/advertiser_names.csv`. They are read by the identity endpoint. The identity endpoint is called from the Advertisers panel. Whether the merge lands the `shown_name` on the field `displayNameOf()` reads (`name` or `display_name`) determines whether the card shows a real name or `ADV_##`. If it does not, the card falls back to the seed id.

---

## 3. The campaign stores

### data/campaigns.csv

`data/campaigns.csv` exists. It is header-only: one line containing the column names, zero data rows. The columns declared in `kairos_api/campaigns_api_store.py` (line 45-64):

```
record_type, campaign_id, flight_id, name, advertiser, agency_id, status,
starts_on, ends_on, goal_kind, goal_value, rebate_percent,
surcharge_discount_percent, surcharge_weekdays, notes, created_at, created_by,
data_source
```

Two record kinds share the file, distinguished by `record_type` (`campaign` or `flight`). A flight is a line of a campaign and is never written without one.

### campaigns_api.py

`kairos_api/campaigns_api.py` is the HTTP layer over the store. It provides:

- `GET /api/clients/campaigns` - every campaign with its flights and the two honest limits (delivery unavailable, terms not priced by engine)
- `POST /api/clients/campaigns` - create a campaign
- `PUT /api/clients/campaigns/{campaign_id}` - update a campaign
- `POST /api/clients/campaigns/{campaign_id}/deactivate` - end a campaign (never delete)
- `POST /api/clients/campaigns/{campaign_id}/flights` - add a flight
- `PUT /api/clients/campaigns/{campaign_id}/flights/{flight_id}` - update a flight
- `DELETE /api/clients/campaigns/{campaign_id}/flights/{flight_id}` - remove a flight
- `GET /api/clients/onboarding/options` - form options for the one-flow onboarding
- `POST /api/clients/onboarding` - agency, advertiser link, campaign, flights, terms in one pass

Two honest limits are stated on every payload:
- `delivery.available = false`: no as-run feed exists, so what a flight delivered is unknown
- `terms.priced_by_engine = false`: campaign terms have no scope in the pricing path

### campaigns_read.py

`kairos_api/campaigns_read.py` hosts three routers:

- `GET /api/campaigns` (line 93) - the legacy rollup read (see section 4 below)
- `GET /api/clients` (line 102) - the client tree
- `GET /api/clients/money` (line 111) - the priced ledger by client

### campaigns_api_store.py

The store module (`kairos_api/campaigns_api_store.py`) contains the full persistence layer: module lock, timestamped backup, temp file plus `os.replace`, version snapshot before every manual edit. The column set, the vocabulary tables (`STATUS_VOCABULARY`, `GOAL_KIND_VOCABULARY`), the id generators (`next_campaign_id`, `next_flight_id`) and all validators are here. The design mirrors `agencies.py` deliberately.

The `data_source` column is written as `"manual"` on every campaign created through the API. There is no automated import path for campaigns.

---

## 4. The "campaigns observed in source data" table

### What it is and where it comes from

The heading "קמפיינים שנצפו בנתוני המקור" / "Campaigns seen in the source data" appears in `CampaignRollupPanel.jsx` at line 51.

The panel calls `loadRollup()` (`clients-api.js` line 65), which hits `GET /api/campaigns`. That route is `campaigns()` in `campaigns_read.py` line 93, which calls `_campaigns_cached()` then `_build_campaigns()`.

`_build_campaigns()` reads `data/Spots.csv` (or `data/reference/Spots.xlsx` if present) via `_load_spots()`. It groups by the `Campaign` column in that file. The result is a list of up to 50 campaign strings extracted from the raw uploaded spot data, sorted by revenue descending (or by spot count when revenue is unavailable), each carrying `spots`, `seconds`, `revenue`, `channels` and `last_airing`.

### What it means

This is a diagnostic read of what the daily spots export carries. It answers: "Which campaign labels does the uploaded file already name?" It does not represent booked campaigns. It does not read `data/campaigns.csv`. The `advertiser_id` column in the result is the raw advertiser string from the spots file, not a resolved name.

The panel is rendered inside the `campaigns` view alongside `CampaignBoard` (the booked campaigns). Both appear when `active === 'campaigns'` in `ClientsWorkspace.jsx` line 317-330.

### Is it useful to show?

It is useful as a data-reconciliation tool: an operator importing a new spots file can confirm which campaign labels the file carries and compare them to what was booked. It is not a campaign management surface.

It is potentially confusing because it lives side-by-side with the booked campaigns board. A person who has booked zero campaigns but uploaded a spots file sees 50 rows under "campaigns seen in the source data" with no explanation that these are file artefacts rather than bookings. The heading in Hebrew is accurate but not self-explanatory to someone who does not know what "source data" refers to.

The panel is an internal diagnostic that has been exposed on a surface that is otherwise about commercial bookings. The data is real and honest, but the context it is rendered in (next to a booking board) invites the reader to confuse it with a booking list.

---

## File inventory

| file | role |
|------|------|
| `tv-break-dashboard/src/clients/ClientsWorkspace.jsx` | page shell, four-slot load, unnamed error banner |
| `tv-break-dashboard/src/clients/clients-api.js` | all fetch calls for the clients destination |
| `tv-break-dashboard/src/clients/AdvertiserRecordsPanel.jsx` | rules store panel, calls /api/advertisers and /api/advertisers/identity |
| `tv-break-dashboard/src/clients/AdvertiserCardGrid.jsx` | named/unnamed partition and grid render |
| `tv-break-dashboard/src/clients/AdvertiserStatCard.jsx` | one card, calls displayNameOf() |
| `tv-break-dashboard/src/clients/advertiser-name-helpers.js` | displayNameOf(), prettifyRawId(), SEED_ID_PATTERN |
| `tv-break-dashboard/src/clients/CampaignBoard.jsx` | booked campaigns table |
| `tv-break-dashboard/src/clients/CampaignRollupPanel.jsx` | source-data rollup panel |
| `kairos_api/campaigns_api.py` | campaign and flight CRUD, /api/clients/campaigns/* |
| `kairos_api/campaigns_api_store.py` | persistence layer, COLUMNS, validators, id generators |
| `kairos_api/campaigns_read.py` | /api/campaigns, /api/clients, /api/clients/money routers |
| `kairos_api/campaigns_read_clients.py` | client tree assembly |
| `kairos_api/advertisers.py` | /api/advertisers CRUD, name_source classification |
| `kairos_api/advertisers_identity.py` | /api/advertisers/identity, names+rules+ledger join |
| `kairos_api/auth.py` | enforce_request middleware, 401 on all /api/ paths without session |
| `data/campaigns.csv` | header only, zero data rows |
| `data/advertiser_rules.csv` | 45 rows ADV_01-ADV_45, name and display_name empty on all |
| `data/advertiser_names.csv` | 41 Hebrew observed names, not wired to /api/advertisers |
| `data/agency_advertisers.csv` | observed names used by _observed_names() in advertisers.py |
