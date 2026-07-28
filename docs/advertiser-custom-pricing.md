# Advertiser custom pricing: weekday scopes and premium-surcharge discounts

Status: shipped. This document is the contract for the two custom-pricing
dimensions added to the conditions engine (the weekday scope and the
premium_discount mode), where each kind of custom price lives, how everything
composes, and the honest boundary of the whole feature.

## The owner ask

Real per-advertiser price personalization:

* a discount on the extra cost (the premium) a specific programme carries,
* custom prices scoped to Saturdays only,
* many custom prices on very specific scopes.

All three land in the existing conditions engine, which already carries scoped
premium rules per advertiser and per agency. Nothing new was invented beside
the two missing dimensions: WHEN a rule applies (weekday) and a discount that
targets only the surcharge part of a price (premium_discount).

## Where each custom-price kind lives

All custom prices are rows in the two condition stores:

* `data/advertiser_conditions.csv`, keyed by `advertiser_id`, CRUD in
  `kairos_api/advertiser_conditions.py`, engine
  `kairos/optimize/advertiser_rules.py`.
* `data/agency_conditions.csv`, keyed by `agency_id`, CRUD in
  `kairos_api/agency_conditions.py`, consumed through
  `kairos/export/agency_layer.py` with the SAME engine class, so scope
  semantics and mode math exist exactly once.

| Custom price kind | Row shape |
| --- | --- |
| Discount on a programme's premium | `effect=premium`, `mode=premium_discount`, `value=0..100`, `scope_programmes=<title>` |
| Saturday-only custom CPP | `effect=premium`, `mode=cpp_absolute` (or `cpp_add` / `cpp_discount`), `scope_weekdays=6` |
| Percent uplift or reduction on a scope | `effect=premium`, `mode=percent`, signed value |
| Plain multiplier on a scope | `effect=premium`, `mode=multiplier` (the legacy default) |
| Many very specific prices | many rows, each scoped by any combination of positions, genres, dayparts, programmes, campaigns and weekdays |

A rule matches a spot only when EVERY non-ANY scope dimension matches. The
overlap detector (`AdvertiserRuleEngine.overlaps` and the cross-level agency
view) understands weekday scopes: two rules on disjoint weekday sets, for
example Saturday-only versus Sunday-only, cannot describe the same spot and are
not reported as overlapping.

## The weekday scope

Both condition stores carry a `scope_weekdays` column: `ANY` or a comma-joined
list of ISO weekday numbers 1..7 (Monday=1 .. Sunday=7; שבת is 6). The column
is read tolerantly at both levels: a store written before this feature has no
such column, and every row then reads as ANY, byte-identical to its old
meaning.

Matching is driven by the spot's REAL date. `price_daily_spots`
(`kairos/export/spots.py`) derives each spot's ISO weekday from the daily
file's parsed date column and threads it into `allow_decision`,
`effective_premium`, `pressure_multiplier` and `placement_multiplier` at both
the agency and the advertiser level. A consumer that has NO date passes
`weekday=None`, and a weekday-scoped rule then never matches; nothing guesses
a day.

### The Israeli week convention (owner directive, frozen)

The week starts Sunday and ends Saturday; the weekend is Friday and Saturday
(ISO 5 and 6); Sunday is a regular workday, never weekend. Data-layer weekday
numbers stay ISO everywhere (Monday=1 .. Sunday=7); only presentation order,
week windows and weekend semantics follow the Israeli convention. The
condition builders' weekday vocabulary (`/api/advertisers/options`,
`weekdays`) is therefore ISO-keyed but ordered Sunday first:
א=7, ב=1, ג=2, ד=3, ה=4, ו=5, שבת=6.

## The premium_discount mode

`mode=premium_discount` reads `value` as a percent 0..100 taken off the
premium SURCHARGE only, the part of the composed premium above 1.0:

```
final_premium = 1 + (final_premium_before_rule - 1) * (1 - value / 100)
```

Frozen semantics, both levels:

* It composes AFTER every other premium mode, whatever its CSV row order: the
  other matching rules and the baseline premium compose first, then each
  matching discount takes its percent off the running surcharge.
* Sequential discounts stack MULTIPLICATIVELY on the surcharge: two 50s leave
  25 percent of the surcharge, not zero.
* The result never falls below 1.0 and never rises above the pre-discount
  premium. A premium at or below 1.0 has no surcharge, so the rule is then a
  no-op rather than a guess. `value` is clamped to 0..100 at apply time and
  rejected outside 0..100 at the API.
* A premium_discount row never carries a `target_layer`: it is defined only on
  the composed whole-stack premium, so the per-layer override resolver
  (`kairos/optimize/layer_overrides.py`) never sees it. Its standalone factor
  (`compute_premium_factor`) is 1.0, so every one-rule-at-a-time consumer
  (segment demand, layer overrides) treats it as an honest no-op.
* A discount is not demand: `segment_demand` never counts premium_discount
  rules toward the placement steer.

## Composition order on the daily pricing path

For each spot in the daily Wally file, in `price_daily_spots`:

1. Resolve the spot's agency (the spot's own column first, the link table as
   fallback). A suspended agency's terms and conditions are inert.
2. Agency level first: an agency forbid drops the spot before the advertiser
   rules run (forbid wins across levels; a locked spot is never dropped).
3. The agency premium composes inside the agency engine: baseline (none at
   agency level, so 1.0), then matching premium rules by mode, then the
   agency's own premium_discount rules on the agency stack's surcharge.
4. The advertiser premium composes the same way inside the advertiser engine:
   the advertiser baseline `default_premium`, then matching premium rules,
   then the advertiser's premium_discount rules on the advertiser stack's
   surcharge.
5. The spot premium is the product: `advertiser_premium x agency_premium`.
   Each level's discounts bite that level's own surcharge only; a discount at
   one level never reaches into the other level's stack.
6. Revenue is the CPP math times that premium (per-second basis,
   `unit_seconds=1.0`), or the stated price times the premium for a FIX spot.
   The agency `rebate_percent` then yields the reporting-only `net_revenue`
   beside unchanged gross.

Relation to the rate-card layers (`docs/pricing-hierarchy-design.md`): the
rate-card breakdown (base CPP times program/prime/day/show/position/ad_type
layers) is a separate composition that prices SLOTS; the conditions engine
prices an advertiser's SPOTS on top of the channel base price. A targeted rule
(`target_layer` set) replaces a named rate-card layer through
`layer_overrides`; everything in this document is the untargeted whole-stack
path. The layered resolver carries no date, so weekday-scoped targeted rules
do not match there today; give a targeted rule a weekday scope only when the
layered path learns a date.

## Worked examples (ILS)

The shipped base price is 60 ILS per second per rating point. Take a 30 second
spot with planned TVR 2.0: base value `2.0 x 30 x 60 = 3600 ILS`.

### A discount on a programme's premium

Advertiser baseline `default_premium=1.3`; a premium rule `multiplier 1.2`
scoped to programme "חדשות הערב" composes to `1.3 x 1.2 = 1.56`. A
premium_discount rule `value=50` on the same programme leaves half the
surcharge: `1 + 0.56 x 0.5 = 1.28`. The spot earns `3600 x 1.28 = 4608 ILS`
instead of `5616 ILS`.

### A Saturday-only custom CPP

Rule: `effect=premium`, `mode=cpp_absolute`, `value=75`, `scope_weekdays=6`.
On a Saturday spot the effective CPP is SET to 75, a factor of `75/60 = 1.25`,
so the spot earns `3600 x 1.25 = 4500 ILS`. On a Sunday spot (ISO 7, a regular
workday) the rule does not match and the spot earns `3600 ILS`.

### Stacked discounts and the floor

Two premium_discount rules of 50 each on a premium of 1.4:
`1 + 0.4 x 0.5 x 0.5 = 1.1`. A discount of 100 lands exactly on 1.0. A
premium already at 1.0 (or below, after a cpp_discount) is untouched: the
mode can only shrink a surcharge, never push a price below the base.

### Cross-level composition

Agency premium 1.2 (from an agency condition) and advertiser premium 1.28
(the first example) price the spot at `3600 x 1.28 x 1.2 = 5529.60 ILS`. An
agency premium_discount of 50 would discount only the agency surcharge:
`1 + 0.2 x 0.5 = 1.1`, giving `3600 x 1.28 x 1.1 = 5068.80 ILS`.

## The honest boundary

Custom prices bite on the DAILY per-spot pricing path only
(`price_daily_spots` and everything downstream of it: the spot ledger, the
break operations money, net revenue). The weekly break-count plan carries no
advertiser attribution (it decides break counts per programme segment, never
which advertiser fills them), so no per-advertiser custom price can honestly
move the weekly plan's revenue, and none does. The weekly optimizer sees
advertiser rules only as the placement-preference steer (`segment_demand`),
which is never charged; that steer is computed for a generic week with no
dates, so weekday-scoped rules do not participate in it (`weekday=None` never
matches a weekday-scoped rule).

Identity guarantee, test-proven (`tests/test_qa7_custom_pricing.py`): with no
rule using the new fields, every number is byte-identical to the pre-feature
engine, spot for spot on the real daily file, and the golden weekly freeze is
untouched.

## Freshness

The condition stores are already part of the schedule-freshness fingerprint
(the `advertiser` group hashes both advertiser CSVs). The calendar-events
store joins the fingerprint as the `events` group ONLY while the event pricing
layer (`pricing_activation.events`) is active; with the layer off the engine
never reads the store and the sidecar stays byte-identical to the pre-events
stamp, so an events edit never invites a pointless recompute.
