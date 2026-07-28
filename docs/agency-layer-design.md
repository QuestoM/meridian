# Agency layer: design

Owner ask (2026-07-28): manage agencies beside advertisers, for the cases where
an advertiser buys through one: agency-level rules, rebates (החזרים), and a
clear map of what an agency layer affects and what it does not.

Ground truth this design is built on (recon 2026-07-28, all measured):

- The daily Wally file DOES carry an agency column ('משרד / MB', mapped to
  `agency` by the loader): 175 of 175 rows populated, 9 agencies, 41
  advertisers, a clean 1:1 advertiser-to-agency mapping in the measured file.
  It is consumed NOWHERE: the loader mapping is the single occurrence of
  'agency' in the whole codebase.
- The weekly Spots.csv (50,386 rows) and the reference workbook carry NO agency
  column, so agency attribution for the weekly plan, historical analytics and
  retention training is not extractable from source data.
- Money composes per spot only on the daily path (`price_daily_spots`), and the
  daily ledger `/api/export/spots.csv` is the only priced-spot money surface.
  Invoicing does not exist anywhere (qh_billing is rating settlement).
- CRITICAL pre-existing defect: the three advertiser vocabularies are disjoint.
  `advertiser_rules.csv` keys (ADV_01..ADV_45) overlap ZERO with the daily
  Hebrew advertiser names and ZERO with the weekly advertiser_id tokens (daily
  and weekly overlap on just 3). Every real daily spot therefore prices at the
  default premium 1.0 and the 45 baseline rows are inert seed data. Any agency
  layer inherits this unless identity resolution ships first.

## The boundary, stated plainly

An agency layer CAN honestly affect:

- The daily per-spot pricing path: agency-scoped conditions composing with
  advertiser conditions (forbid wins; premiums compose; advertiser CPP-absolute
  overrides an agency percent), and the drop/keep decisions with reasons.
- The daily ledger: a `rebate_percent` yielding a reporting-only
  `net_revenue` column beside the gross figure, clearly labeled; nothing is
  invoiced because no invoicing exists.
- Placement steering: agency pressure rules folding into the demand weights,
  which rank placements and never change charged money (provably neutral).

An agency layer CANNOT affect, given the current engine and data, and the UI
must say so: the weekly break plan and its revenue (no per-break advertiser or
agency attribution exists, and the weekly data carries no agency column), the
optimizer objective and retention math, quarter-hour settlement, and invoicing.

## Prerequisite zero: identity resolution (its own deliverable)

A canonical advertiser registry with aliases: `data/advertiser_identity.csv`
(canonical_id, display_name, aliases pipe-joined, source). Seeded by matching
the daily Hebrew names against the weekly first-word tokens, reviewed by the
operator in the Advertisers page (a new 'זיהוי' section listing unmatched
names). The rules engine resolves through it before baseline lookup. Without
this, both the existing advertiser layer and the new agency layer stay inert
on real spots; with it, both come alive at once.

## The agency build

1. Stores: `data/agencies.csv` (agency_id, name = the exact Wally string,
   aliases, rebate_percent, active, notes) and `data/agency_advertisers.csv`
   (agency_id, advertiser canonical_id, source observed|manual). The observed
   links auto-refresh from each daily file (spot-level column wins on
   disagreement); manual links override. Atomic writes, locks, version
   snapshots with new logical files so edits are restorable.
2. Conditions: `data/agency_conditions.csv` reusing the existing condition
   shape and engine (premium, require, forbid, pressure; the same five scope
   dimensions), evaluated agency-first then advertiser, forbid-wins, and the
   existing overlap detector extended across the two levels.
3. Daily pricing: `price_daily_spots` reads the spot's agency (column first,
   link table fallback), applies agency conditions in composition, applies
   `rebate_percent` to produce `net_revenue` in the ledger and the export, and
   records the agency and the applied rule ids on each priced row.
4. API and UI: an Agencies tab beside Advertisers mirroring its manager
   (list, drawer, conditions builder, stats from the daily ledger), plus the
   boundary note rendered on-page: what agency rules do and do not touch.
5. Assistant: read tool `get_agencies` and condition proposals extend the
   existing propose path later; not in the first build.

## Evidence caveats to re-measure as data lands

The 1:1 advertiser-to-agency mapping and the 9-agency vocabulary come from ONE
daily file (one date, prime only). The link model supports many-to-many; the
UI should not assume exclusivity.

## Non-goals

Invoicing and credit notes; agency commissions inside the weekly plan money;
any effect on retention coefficients.
