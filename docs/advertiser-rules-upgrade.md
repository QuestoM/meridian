# Advertiser rules upgrade: programmes, placement preference, coefficient modes, real dayparts, gold break

This is the characterization and build plan for the Advertisers-page upgrade. It
traces where each piece of data comes from today and defines the changes, so the
backend and the dashboard move together and nothing is faked.

## 0. Where each thing comes from today (the seam map)

- Advertiser rules: `kairos/optimize/advertiser_rules.py` reads two CSVs.
  - `data/advertiser_rules.csv` -> one `Baseline` per advertiser (default_premium,
    allow_positions, allow_genres, prime_time_only).
  - `data/advertiser_conditions.csv` -> scoped `Condition`s (effect = premium |
    require | forbid; scoped by position, genre, daypart token-sets).
  - CRUD: `kairos_api/advertisers.py` (baselines) and
    `kairos_api/advertiser_conditions.py` (conditions). UI:
    `tv-break-dashboard/src/AdvertisersManager.jsx` and `AdvertiserConditions.jsx`.
- Where rules bite: `kairos/export/spots.py` `price_daily_spots` only (the daily
  Wally per-spot path). Each spot's revenue is multiplied by
  `effective_premium(advertiser, position, genre, daypart)`; a spot failing
  `is_allowed(...)` is dropped with a reason. The weekly break-count optimizer does
  NOT attribute breaks to advertisers, so it does not consume these rules.
- Coefficient (the "premium" the operator sets) = a pure multiplier on revenue.
- Genres: `ProgramClassifier` (`kairos/data/classifier.py`) assigns a category to a
  programme title. The classifier's category vocabulary is the genre source.
- Programmes: `kairos/data/loaders.py` `load_programmes` (the EPG: Title, Channel,
  start/end). Present, but not yet a rule dimension.
- Dayparts: a coarse clock bucket in `spots.py` (`prime` 20-23, `daytime` 6-19,
  `overnight`); the UI lets the operator type a free token, which is the "All /
  write-something" problem.
- Positions: the engine vocabulary is `first/middle/last` (`kairos/model/prepare.py`
  `position_bucket`), plus 1-based integer positions on the daily spot path. No
  "gold break" today.
- Pricing: Israeli CPP (Cost Per rating Point). `kairos/optimize/objective.py`
  `break_revenue` = planned break rating x duration-in-30s-units x channel base
  price (the CPP), then the advertiser premium. `PricingModel.base_price` is the CPP.

## 1. Real Israeli dayparts (replaces "All / write-something")

A spot or a slot must map to exactly one real daypart, the same way Israeli TV is
sold, and the mapping must be identical across training, the weekly plan and the
daily spot file. New module `kairos/data/dayparts.py` defines the canonical
taxonomy (bilingual he/en, full 24h coverage, aligned to the broadcast day that
starts 02:00), overridable from `config/dayparts.yaml`:

| key      | he          | clock window | notes                          |
|----------|-------------|--------------|--------------------------------|
| morning  | בוקר         | 06:00-12:00  |                                |
| noon     | צהריים       | 12:00-17:00  |                                |
| evening  | ערב          | 17:00-20:00  | early evening / pre-prime      |
| prime    | פריים טיים   | 20:00-23:00  | matches the engine prime window |
| night    | לילה         | 23:00-06:00  | wraps midnight                 |

`daypart_for_hour` / `daypart_for_timestamp` give the canonical key; the API
exposes the list so the dashboard renders a fixed multi-select, never free text.
`spots.py` switches its coarse 3-bucket derivation to this taxonomy so a
daypart-scoped rule means the same thing everywhere. `prime_time_only` continues to
match the `prime` key.

## 2. Programmes as a rule dimension (multi-select + filter)

Add a fourth scope dimension `scope_programmes` to `Condition` (token set of
programme titles), parallel to position/genre/daypart, with the same ANY/empty
semantics. Genres stay `scope_genres` but the UI upgrades both to real multi-select
dropdowns with search/filter:

- Programme options come from `load_programmes().Title` (distinct, the real EPG).
- Genre options come from the classifier's category set.
- A new read endpoint serves both option lists so the dashboard never invents a
  name. On the daily spot path a spot's programme is the `program` column
  (`DAILY_COLUMN_MAP`), matched against `scope_programmes`.

## 3. Placement preference (steers placement, never charged)

A second, parallel lever to the coefficient. Copy: "Placement preference" (he:
העדפת שיבוץ). Earlier draft copy "לחץ שיבוץ" was dropped: with a kamatz the word
לחץ reads as the imperative "press", which is confusing on a control label. The
engine token stays `pressure` internally; only the customer-facing copy changes.
Distinct from the premium because it changes WHERE the model wants to place an ad
without adding money to the real total.

- New effect `pressure` on `Condition`, value = a percent (e.g. +10 means the model
  behaves as if the slot pays 10% more).
- New engine method `placement_multiplier(advertiser, scope...)` = `effective_premium
  x product(1 + pressure/100)` over matching pressure rules. The optimizer/ranking
  uses `placement_multiplier` to decide placement; the revenue total uses only
  `effective_premium`. So a +10% pressure makes the slot rank as if richer, but the
  reported revenue is unchanged: honest money, biased placement.
- Surfaced separately in outputs: `placement_value` (with pressure) vs `revenue`
  (without), so the operator sees the steer and the truth side by side.

## 4. Coefficient modes (percent or CPP, absolute / add / discount)

The premium effect gains a `mode` (column on `advertiser_conditions.csv`):

- `multiplier` (default, the original behaviour): value IS the multiplier, so 1.15
  means +15%. Every legacy row with no mode column keeps this meaning exactly, so
  nothing already priced changes.
- `percent`: value is a signed percent; multiplier = 1 + value/100 (so +15 -> 1.15,
  -15 -> 0.85).
- `cpp_absolute`: the spot's CPP is SET to `value` (an absolute cost-per-point).
- `cpp_add`: CPP becomes `base_cpp + value`.
- `cpp_discount`: CPP becomes `base_cpp - value` (floored at 0, never negative).

The cpp_* amounts are in the same units as the engine's configured point price
(`PricingModel.base_price`), which is the price the daily pricing path actually
multiplies by. `effective_premium` and `placement_multiplier` take an optional
`base_cpp`; the daily spot path passes `pricing.base_price`. With no base_cpp known,
a cpp_* rule leaves the premium unchanged rather than guess a conversion.

Because CPP modes need the base CPP, `effective_premium` gains an optional
`base_cpp` and returns the effective CPP factor; the pricing path passes the
channel base price. Percent rules keep working with no CPP. The UI offers a toggle
(percent vs amount) and, for amount, a sub-toggle (absolute / add to point /
discount from point).

## 5. Gold break (ברייק זהב) position

"Gold break" is a premium placement class (typically the first in-show break around
prime). Add `gold` to the position vocabulary so it is selectable as a scope and as
an allow_position, and recognized on the daily path (first break inside a prime
programme). It carries no automatic premium by itself; the operator sets the
coefficient/pressure for it like any other position, but it is now a first-class,
nameable position rather than a raw integer.

## 6. Backend correctness (the part that must be insane)

- Every new field is optional with an honest default, so old CSVs load unchanged.
- Pure, deterministic engine methods; no hidden constants; full unit tests for the
  daypart taxonomy, the programme dimension, the pressure split, the four coefficient
  modes and the gold-break position.
- The pricing path reports `revenue` (real) and `placement_value` (with pressure)
  separately, and never lets pressure leak into a revenue total.
- The daypart taxonomy is the single source of truth across training, weekly and
  daily, so a rule scoped to `prime` means the same minutes everywhere.

Build order: (1) daypart taxonomy, (2) data-model extensions (programmes, pressure,
coefficient modes, gold break) in the engine with tests, (3) pricing-path wiring,
(4) API option/CRUD endpoints, (5) dashboard UI. Each step ships green.

## 7. Binding the rules store to the advertisers that actually air (2026-08-09)

### The defect, measured

`data/advertiser_rules.csv` held 45 rows keyed `ADV_01`..`ADV_45`, each carrying a
commercial rule. Its `name`, `display_name` and `aliases` columns were empty in all
45. The resolver in `kairos/optimize/advertiser_rules_identity.py` binds a spot to a
rule BY NAME, so the two key spaces did not intersect:

| measurement | before | after |
|---|---|---|
| advertisers airing on `data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv` | 41 | 41 |
| rules rows | 45 | 45 |
| rows bound to an observed advertiser | 0 | 41 |
| **advertisers that resolve to a rules row** | **0** | **41** |
| daily gross revenue | 699,450.00 ILS | 699,450.00 ILS |
| priced spots / dropped spots | 119 / 0 | 119 / 0 |

The whole advertiser premium layer bound to nothing. Not "usually 1.0": dead. No
premium anyone set on that page could ever have priced a spot.

### What was done

The 41 rows that could bind are re-keyed on the observed advertiser names, which is
option A of `docs/ux-gauntlet/decisions-for-owner.md` decision 1, approved by the
owner on 2026-08-01. The names come from `data/advertiser_names.csv`, which holds
the operator's own traffic vocabulary with `source=observed` and
`first_seen=2025-04-27`. Nothing here invents an advertiser; this store points at
that name space and is never a source of names itself, which is its own test.

Re-keying rather than filling the `name` column is deliberate and was measured
both ways. Both shapes bind 41 of 41 and both leave the money identical, but with
the name in `advertiser_id` the API reports `name_source: observed` (literally
true: the id is a name seen in the data), while filling `name` reports
`name_source: operator`, which claims a person named these rows when a seed did.
The key-on-the-name shape also breaks no existing test, and the resolver was built
for it: `build_name_index` indexes `advertiser_id` itself, and
`kairos_api/advertisers.py::_name_source` already has an `observed` branch for a
Hebrew id.

### Provenance: why every row says `synthetic`

The premiums in this store are seed values. Attaching one to a real bank without
saying so fabricates a commercial term about a real company, which is the
campaign's fabrication law. So the store gains a `data_source` column reading
`synthetic` on every row, the same column `data/agencies.csv` already publishes,
plus a Hebrew note on every row saying the commercial detail is a seed and not
something agreed with the advertiser.

### The four rows left unbound, and why

`ADV_02` carries the only seed terms in the store with teeth: `default_premium`
1.27 and `allow_positions` `first,last`. Both were invented. Binding them to a
named advertiser would assert a 27 percent premium and a placement restriction
about a real company, and would also drop that company's spots that fall in the
middle of a break. It stays unbound and says so in its notes.
`ADV_43`, `ADV_44` and `ADV_45` are the arithmetic remainder: 45 rows, 41 observed
advertisers. They stay as neutral 1.0/ANY seed rows bound to nobody rather than
have a company invented to fill them.

That is also the rule the guard enforces: a row carrying a premium other than 1.0,
a narrowed allow list, or `prime_time_only`, may not be bound to a named
advertiser until the term is real.

### Why the money did not move

Every row that took a name carries `default_premium` 1.0 and allows ANY position
and genre, so consulting it is arithmetically identical to the unknown-advertiser
fallback of 1.0 it replaced. Gross stays 699,450.00 ILS across 119 priced spots
with zero drops, every applied premium still exactly 1.0. What changed is that the
layer is now live: a premium the operator sets on any of those 41 rows will price
that advertiser's spots, where before it could not.

`output/weekly_break_schedule.csv` is unmodified, checked by MD5 and by
`tests/test_plan_artifact_fingerprint.py` before and after. The weekly break-count
plan does not attribute breaks to advertisers, so it cannot see this store.

### Known gap, in `kairos_api/advertisers.py` (not fixed here)

`_write_frame` writes `frame[COLUMNS]` and `COLUMNS` does not list `data_source`.
Measured: one `PUT /api/advertisers/{id}` erases the `data_source` column from all
45 rows. The Hebrew note in `notes` survives, because `notes` is in `COLUMNS`, so
the provenance is not lost entirely; but the machine-readable half is. Adding
`"data_source"` to that list is the one-line fix.
`tests/test_advertiser_rules_bind.py::test_every_row_declares_its_commercial_terms_synthetic`
fails loudly with that instruction if it ever happens.

### The guard

`tests/test_advertiser_rules_bind.py`, nine checks over binding, provenance and
money. Six of the nine fail against the pre-fix file, including the headline
"41 of 41 advertisers resolve to no rules row". One of them,
`test_a_row_returned_to_the_old_shape_fails_the_bar`, rebuilds an unbound row from
the shipped store and runs the same check over it, so the guard cannot quietly
stop guarding.
