# Pricing hierarchy and per-advertiser overrides: design

Status: design, owner-requested 2026-06-19. This document is the single source of
truth for how Kairos should price an ad break slot. It records the current reality
(two disconnected engines), the target layered model, the override semantics, the
implementation plan, and the operator UX. No revenue-math code changes ship from
this document; implementation is a separate, carefully staged wave because it
touches the money path.

## 1. Current reality (ground truth, verified in code)

Two pricing engines exist and never compose with each other.

### Engine A: the rate card (`kairos/optimize/pricing.py` PricingModel)
Used by the weekly optimizer and the live dashboard.
- Base: `base_price_per_second_per_tvr_point = 60.0` (config/optimization_weights.yaml).
  Units: ILS per second per rating point.
- Premium tables defined: program_type, day_of_week, position_in_break, ad_type.
- Actually applied in live revenue:
  - Dashboard (`server.py` break-ops): program_type ONLY.
  - Optimizer/export segment math: program_type x day_of_week (`segment_premium`).
  - position_in_break premium: DEFINED in config, NEVER multiplied into any revenue
    number. Dead hook.
  - ad_type premium: DEFINED, NEVER applied. Dead hook.
- "Prime" is not a layer. It is encoded into the program_type class enum
  (PrimeShow1 / PrimeShow2), assigned by air-order after the news. Consequence: a
  specific show (for example Big Brother) cannot be priced above its class; there
  is no per-show premium on the rate card.

### Engine B: advertiser rules (`kairos/optimize/advertiser_rules.py`)
Used ONLY by the daily spot-export path (`kairos/export/spots.py`), not by the
optimizer or the dashboard.
- Modes (effect = premium):
  - multiplier: value is the multiplier. STACKS (premium *= factor).
  - percent: 1 + value/100. STACKS.
  - cpp_add: (base_cpp + value) / base_cpp. STACKS.
  - cpp_discount: (base_cpp - value) / base_cpp, floored at 0. STACKS.
  - cpp_absolute: value / base_cpp. REPLACES the running premium (authoritative,
    last-absolute-wins). This is the only replace semantic today.
- Condition scopes: advertiser_id, scope_positions, scope_genres, scope_dayparts,
  scope_programmes (title-level). NO campaign field. NO date range. NO channel.
- Precedence: rules fold in CSV row order. There is no most-specific-wins.
- base_cpp passed in is the flat channel base price, not position-adjusted.

### The gap versus the owner's model
The owner wants: base CPP, then premiums that stack (prime, then show on top, then
position), and a per-advertiser or per-campaign override that REPLACES one specific
layer (for example, advertiser X gets a low special price on position 2 that
replaces the general position-2 premium, while prime and show still stack on top).

Today that is not expressible because:
1. On the path where advertiser rules run, there is no general position-2 premium to
   replace (the rate-card position premium is dead).
2. cpp_absolute wipes the entire running premium, not a single layer.
3. There is no campaign scope.
4. There is no most-specific-wins precedence.
5. The two engines never compose, so a layered price is not consistent across the
   optimizer, the dashboard, and the spot export.

## 2. Target model

Final CPP for a slot = Base x (product of named premium layers), with per-scope
overrides that replace exactly one named layer (not the whole stack) or adjust the
final price.

Canonical layer order (each separately configurable, each defaults to 1.0 identity):
1. Base CPP per rating point. Channel default; a per-advertiser negotiated base is
   allowed and is a base override, not a premium.
2. Daypart / prime premium. Split out of the program-type class into its own layer.
3. Show premium. Per-title (for example Big Brother) or per-program-type. Distinct
   from prime, stacks on top of it.
4. Position-in-break premium (1 / 2 / 3 / last). Applied for real, not dead.
5. Ad-type premium (regular / sponsorship / promo).
(Optional future layers: day-of-week, gold, seasonality. Same mechanism.)

Worked example: base 60, prime x1.10, Big Brother x1.25, position-2 x1.15,
sponsorship x1.05 -> 60 x 1.10 x 1.25 x 1.15 x 1.05 = 99.6 ILS per second per rating
point.

### Override layer (per advertiser, optionally per campaign)
A scoped rule targets one named layer. Two kinds:
- REPLACE-LAYER: swap exactly that layer's multiplier (or set an absolute CPP for
  that layer) for the matched scope. Other layers still stack. This is the owner's
  position-2 case.
- ADJUST-FINAL: a percent or multiplier applied to the whole computed price
  (a blanket discount or surcharge).

Precedence: most-specific-wins per layer. Specificity = count of matched scope
dimensions (advertiser + campaign + position + show beats advertiser + position
beats advertiser-wide). Per layer, the most-specific matching REPLACE-LAYER override
wins; ties broken by an explicit priority field, then by row order. ADJUST-FINAL
overrides compose after the layer stack, most-specific applied last.

This is strictly more expressive than today and matches the owner's mental model.

## 3. Implementation plan (staged, money-path safe)

Each stage is independently shippable and identity-preserving until data enables it.

S1. Unify composition. [SHIPPED 2026-06-19, commit 8546f8b] Built
    `PricingModel.price_slot(...) -> PriceBreakdown` (kairos/optimize/pricing.py) plus
    the `PriceLayer` / `PriceBreakdown` types. Returns the final CPP plus a per-layer
    breakdown with provenance. Identity proven: with only the program and day layers
    active (the default), `price_slot(...).total_premium` equals `segment_premium`
    across all 35 class/day combinations (test_pricing.py). The position and ad-type
    layers are wired but OFF by default, because their configured multipliers are not
    1.0, so no revenue number moves until a later, owner-approved stage activates them.
    Remaining in S1: converge the existing call sites (transform.py segment math,
    server.py dashboard, export/spots.py) onto price_slot so the breakdown is the live
    path everywhere. Deferred until a surface consumes the breakdown, to avoid adding
    indirection with no behavior gain.
S2. Make position premium live. Apply position_in_break in `price_slot` so the
    dead hook becomes a real layer. Default values already in config; revenue moves
    only where position premiums differ from 1.0.
S3. Split prime into its own layer. Add a daypart/prime premium table; stop folding
    prime into the program_type class. Reassign program_type to a true content
    category so show/prime are independent.
S4. Add per-show premium. New show-premium table keyed on title (or a stable
    show id), distinct from program_type. Identity until populated.
S5. Per-layer override. Add `target_layer` to advertiser Condition plus a
    REPLACE-LAYER resolution that swaps only that layer. Keep cpp_absolute as a
    whole-stack escape hatch for back-compat.
S6. Campaign scope. Add `scope_campaign` to Condition and match on the spot's
    campaign. Identity for rules that leave it ANY.
S7. Most-specific-wins resolver. Specificity scorer + per-layer winner selection +
    explicit priority tiebreak. Diagnostics report shadowed rules.
S8. Guardrails. Final-CPP floor/ceiling, below-cost warning, explicit-zero rule for
    promo, no silent collapse.

Each stage ships with: real-data parity test proving identity when the new layer is
empty, a unit test for the new layer, and the existing export/service/optimizer
suites green. Competitor boundary untouched (pricing never reads competitor signals).

Owner gate. S1 is identity-preserving and shipped autonomously. S2 (activate the
position premium: a 1.30 first-position multiplier, 0.00 promo rate) and any other
layer activation MOVE REAL REVENUE the moment they go live, so they do not ship
without the owner's explicit yes on the intended numbers. They are built behind the
already-wired `enable_position` / `enable_ad_type` flags (default OFF) so the
mechanism can land and be tested without changing a single live price; flipping the
default to ON is the owner's call. The three open decisions in Section 5 gate S5-S6
(override target-layer, campaign scope); their documented default assumptions hold
until the owner says otherwise.

## 4. Operator UX (error-proof, comfortable)

A Rate Card and Overrides workspace. The implementation-ready detail (layout, components,
interaction states, error-proofing flows, per-surface data contract, and build order) is
in docs/pricing-hierarchy-ux-spec.md. The summary below is the shape; that doc is the
build spec.

1. Layer-stack panel. Vertical stack of named layer cards (Base, Prime, Show,
   Position, Ad-type), each with its current multiplier inline-editable. Beside it, a
   live worked example: pick a sample slot and watch the running price compute
   layer by layer with each step labeled. Stacking becomes visually obvious; the
   operator never does mental math.
2. Override builder. A predicate row: WHEN advertiser = X [AND campaign = Y] [AND
   position = 2] [AND show = Big Brother] THEN REPLACE [position] layer with [value]
   OR ADJUST final by [+/- percent]. The builder shows in real time which base layer
   it overrides and the rate-card-versus-override delta for a matching slot.
3. Precedence preview. When several overrides could match one slot, show the
   resolved winner and grey out the shadowed rules, each with a one-line reason (for
   example: the position-2 override for X wins over the advertiser-wide X discount
   because it is more specific). The operator sees the outcome before saving. This is
   the core error-proofing.
4. Price-any-slot tester. One form: pick advertiser, campaign, show, daypart,
   position. Instant full breakdown: every layer and every applied override with
   per-line provenance. Zero ambiguity.
5. Guardrails surfaced inline: final-CPP floor and ceiling, a below-cost warning, an
   explicit-zero confirmation for promo, and an honest "rate card only" empty state
   when no overrides exist.

Law 9: every number on every surface traces to base x named layers x named
overrides, each line labeled with its source.

## 5. Decisions (owner delegated, decided 2026-06-20)

The owner handed the reins with one standing requirement: everything reflects in the
dashboard, and everything that needs tuning is tunable there at every level. These are
the resolved decisions, not open questions.

- Base per advertiser: DECIDED shared channel base plus a per-advertiser negotiated base
  override. `price_slot(base_cpp=...)` already accepts the override; the dashboard sets
  the channel base and, per advertiser, an optional base override.
- Show identity: DECIDED literal title string for v1. Israeli prime titles (for example
  אח הגדול) are stable within a season; a stable show id is a later migration, not a v1
  blocker.
- Campaign granularity: DECIDED a campaign always belongs to one advertiser (campaign
  implies advertiser). The dashboard disables the campaign field until an advertiser is
  chosen.
- Layer activation (position, ad-type): DECIDED ship them fully tunable and toggleable
  from the dashboard at every level (edit each multiplier, flip activation per layer).
  Default activation stays OFF, because flipping unvalidated multipliers live would
  silently restate every historical revenue figure. Turning a layer ON is a one-click
  dashboard action that visibly restates the price in the tester, so the operator owns a
  real-money rate-card decision with full sight of its effect. This is the honest form of
  "tunable at every level": real control in the dashboard, no silent revenue change.
