# Agency layer: design and contract

Owner ask (2026-07-28, approved): manage agencies beside advertisers, for the
cases where an advertiser buys through one. Agency-level rules, rebates
(החזרים), a full record per agency (contacts, terms, identifiers), and a clear
map of what the agency layer affects and what it provably cannot. The owner
explicitly approved synthesizing seed data where real data lacks (contacts,
terms and more), provided every synthesized value is labeled synthetic in the
data itself.

## Ground truth (measured recon, 2026-07-28)

- The daily Wally file carries an agency column ('משרד / MB', mapped to
  `agency` by `kairos/data/loaders.py`). In the measured file: 175 of 175 rows
  populated, 9 distinct agencies, 41 advertisers, a clean advertiser-to-agency
  mapping (1:1 in this one file). Until this build it was consumed nowhere.
- The weekly Spots.csv and the reference workbook carry NO agency column, so
  the weekly plan, historical analytics and retention training cannot honestly
  carry agency effects. This is a data boundary, not a code choice.
- Money composes per spot only on the daily path (`price_daily_spots` in
  `kairos/export/spots.py`); the daily ledger is the only priced-spot money
  surface. Invoicing does not exist anywhere in the system.
- The 9 real agencies (exact Wally strings): OMD, יוניברסל, יוניון, ישירים,
  לפמ, מדיהקום, פובליסיס, פיתוח עסקי, רואים קונים.

## The boundary, stated plainly

The agency layer CAN honestly affect:

- Daily per-spot pricing: agency-scoped conditions evaluated agency-first,
  composing with advertiser conditions. Forbid wins across both levels: an
  agency forbid drops the spot even when every advertiser rule allows it, and
  vice versa. Premium effects compose multiplicatively across the two levels.
- Daily ledger reporting: `rebate_percent` yields a `net_revenue` figure
  BESIDE gross on every priced spot. Reporting only. Gross is unchanged, and
  nothing is invoiced because no invoicing exists.
- Placement pressure: agency pressure rules fold into `placement_value`
  exactly like advertiser pressure, ranking placements without ever being
  charged.

The agency layer CANNOT affect, and the UI must say so:

- The weekly break plan and its money: no per-break advertiser or agency
  attribution exists, and the weekly data carries no agency column.
- The optimizer objective and retention math: retention is trained on weekly
  data without agency identity.
- Quarter-hour settlement (`qh_billing`): settlement restates observed spot
  starts; agency identity plays no role in the round-QH rule.
- Invoicing and credit notes: none exist; `net_revenue` is a report line, not
  a billing artifact.

## Entity model: data/agencies.csv, field by field

One row per agency. Every field exists for a stated operational reason.

| field | type | rationale |
| --- | --- | --- |
| agency_id | str, unique, required | Stable key (AGY_NN). Conditions and links reference it, so a display rename never orphans rules. |
| name | str, unique, required | The EXACT Wally string from 'משרד / MB'. This is the observed join key against the daily file; never editable casually, because editing it silently detaches the agency from its spots. |
| display_name | str | What the dashboard shows. Decouples presentation from the join key. |
| aliases | str, pipe-joined | Alternate spellings seen in future daily files (Latin/Hebrew variants). Resolution tries name, then display_name, then aliases, so a respelled upload still lands on the right record. |
| agency_type | enum | מדיה מלא / קריאייטיב / בוטיק. Segments the book of business; a boutique with one advertiser is negotiated differently than a full media shop. |
| contact_name | str | Primary account contact. Who the sales desk calls when a campaign breaks. |
| contact_role | str | The contact's role (e.g. מנהלת לקוח), so escalation goes to the right level. |
| contact_phone | str | Direct line, Israeli format. |
| contact_email | str | Email for ledgers and confirmations. |
| contact2_name | str | Secondary contact, for coverage when the primary is out. |
| contact2_role | str | Secondary contact's role. |
| contact2_phone | str | Secondary direct line. |
| contact2_email | str | Secondary email. |
| address_city | str | Registered office city, needed on any future paperwork. |
| address_street | str | Street address, same reason. |
| vat_id | str | מספר עוסק (9 digits). The legal identifier for any future settlement export; stored as text to preserve leading digits. |
| payment_terms_days | int >= 0 | שוטף+N payment terms. 60 means שוטף+60. Feeds future cash-flow reporting; consumed by nothing that moves money today. |
| rebate_percent | float 0..100 | The agency's negotiated rebate. The ONLY terms field consumed by the engine today: net_revenue = gross x (1 - rebate/100), reporting only. |
| commission_percent | float 0..100 | Standard agency commission. Recorded for the record; NOT applied anywhere, because commission settlement is an invoicing concern and invoicing does not exist. |
| credit_limit_ils | float >= 0 | Exposure ceiling for future credit control. Consumed by nothing today; recorded so the record is complete when credit control ships. |
| status | enum | active / suspended. A suspended agency's conditions and rebate go inert (spots still record its name); deactivation is reversible, deletion is not offered, so history keeps resolving. |
| onboarded_at | date str | When the relationship started. Synthetic in seeds (set to the first observed daily date). |
| notes | str | Free text. Every synthetic seed row carries the Hebrew replacement note here. |
| data_source | enum | observed / synthetic / manual. Row-level provenance: seeds are `synthetic` (their names are observed but their contacts and terms are invented, and the row is only as real as its weakest field); operator-created rows are `manual`; `observed` is reserved for rows fully backfilled from real source data. |

## Link model: data/agency_advertisers.csv

Columns: agency_id, advertiser, source (observed/manual), observed_date, notes.

- Observed links are auto-derived from the LATEST daily file at read time: the
  API recomputes them on every request, so a new upload refreshes the map with
  no migration step. The seeded rows freeze the measured 2025-04-27 mapping so
  the layer works before any request-path derivation runs.
- Manual links override observed links per advertiser: when both exist for one
  advertiser, the manual row wins. One manual link per advertiser; linking an
  already-manually-linked advertiser to a second agency is a 409, remove first.
- The model is many-to-many capable across time (an advertiser can move
  between agencies); the measured 1:1 comes from ONE daily file (one date,
  prime only) and the UI must not assume exclusivity.
- Per-spot resolution order in pricing: the spot's own agency column first
  (the file is the freshest truth), the link table only as fallback for daily
  files that lack the column.

## Conditions: data/agency_conditions.csv

Same shape as advertiser conditions, keyed by agency_id instead of
advertiser_id: rule_id, scope_positions, scope_genres, scope_dayparts,
scope_programmes, effect (premium/require/forbid/pressure), value, mode
(multiplier/percent/cpp_absolute/cpp_add/cpp_discount), notes. The same
`AdvertiserRuleEngine` evaluates them (baselines empty, conditions keyed by
agency_id), so scope semantics, mode math and the overlap detector are one
engine, not two.

Evaluation order and precedence on the daily pricing path:

1. Resolve the spot's agency (spot column, then link table).
2. Agency level first: an agency forbid drops the spot (reason prefixed
   `agency`); agency require rules must be satisfied at their level.
3. Advertiser level second, exactly as before.
4. Forbid wins across levels: any forbid at either level drops the spot. A
   locked (נעיצה) spot is never dropped by either level, same as today.
5. Premiums compose: total premium = agency effective premium x advertiser
   effective premium. CPP-absolute stays authoritative WITHIN its level.
6. Pressure composes the same way into placement_value only, never charged.

The overlap detector runs within the agency's own rules (the engine's existing
pairwise findings) and ACROSS levels: every agency condition is intersected
against the conditions of each linked advertiser, and a require/forbid pair
across levels is flagged as a cross-level conflict with the note that forbid
wins.

## Ledger integration (kairos/export/spots.py)

Each priced spot gains: agency (resolved name, empty when unresolved),
agency_premium (1.0 when no agency condition matched), rebate_percent (0 when
no active agency terms), net_revenue (gross x (1 - rebate/100)). Each dropped
spot gains the agency and a reason that names the level that dropped it.
`DailyPricingResult.total_net_revenue` sits beside `total_revenue`.

Identity guarantees, proven by tests:

- With no agency conditions and rebate 0, gross revenue is byte-identical to
  the pre-agency ledger (premium x 1.0 is exact in floating point).
- The shipped seed (conditions file header-only) moves NO gross revenue.
- net_revenue never feeds the optimizer, the weekly plan, or any charged
  figure. It is a reporting column.

## Lifecycle and permissions

- Create/update/deactivate go through /api/agencies, snapshot-before-write
  into the unified version timeline (logical files `agencies`, `agency_links`,
  `agency_conditions`), atomic temp-file writes under a module lock, timestamped
  backups in data/_backups. Same auth seam as every other store: writes require
  a writer role when auth is active.
- Deactivate (status suspended) instead of delete: spots keep resolving to the
  record, conditions and rebate go inert, reactivation is a status change.
- Note: the version_store logical registry is owned elsewhere; until
  `agencies`/`agency_links`/`agency_conditions` are registered there, the
  snapshot hooks are safe no-ops and the backup files are the recovery path.

## Seed data honesty contract

- data/agencies.csv ships 9 rows, one per REAL agency name from the measured
  daily file. Names are the observed Wally strings. Every other detail
  (contacts, phones, emails, addresses, vat ids, terms, limits) is SYNTHETIC:
  data_source=synthetic on the row, emails on the reserved `.example` domain,
  and every row's notes carry: פרטי קשר סינתטיים לדוגמה, יש להחליף בנתוני אמת.
- data/agency_advertisers.csv ships the 41 measured advertiser links,
  source=observed, observed_date=2025-04-27 (the daily file's date).
- data/agency_conditions.csv ships header-only: no invented rules, so the
  shipped layer prices nothing differently until an operator writes a rule.

## API surface

- GET /api/agencies: all records, each with its conditions, overlap findings
  and link counts, plus the boundary note for the UI.
- GET /api/agencies/{id}, POST /api/agencies, PUT /api/agencies/{id},
  POST /api/agencies/{id}/deactivate.
- GET /api/agencies/{id}/advertisers: observed (derived live from the newest
  daily file), manual, and the effective merge.
- POST /api/agencies/{id}/advertisers, DELETE
  /api/agencies/{id}/advertisers/{advertiser}: manual link management.
- GET/POST/PUT/DELETE /api/agencies/{id}/conditions[/{rule_id}]: the condition
  engine shape, validated exactly like advertiser conditions.

## UI vocabulary (for the dashboard build, not this backend increment)

משרד (the Wally column's own word) for agency, מפרסם for advertiser, הכנסה
צפויה for gross, הכנסה נטו אחרי החזר for net, and the boundary note rendered
on-page. Suspension confirm copy uses מפעיל vocabulary, never משתמש.

## Non-goals

Invoicing, credit notes and commission settlement; agency effects inside the
weekly plan money; any effect on retention coefficients or QH settlement; the
canonical advertiser identity registry (its own deliverable; this layer joins
on the daily file's own strings, which are internally consistent).
