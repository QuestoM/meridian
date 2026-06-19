# Pricing hierarchy operator UX: implementation-ready spec

Status: spec, owner-requested 2026-06-19. Companion to docs/pricing-hierarchy-design.md
(the engine design and staged plan). This document specifies the operator-facing Rate
Card and Overrides workspace in implementation detail: layout, components, interaction
states, the error-proofing flows, and the data contract each surface reads. The goal the
owner set is a workspace that is comfortable, efficient, and error-proof, where the
operator never does mental math and never ships a price they did not intend.

Every number on every surface traces to base x named layers x named overrides, each line
labeled with its source (Law 9). No surface fabricates a value; an absent value shows an
honest empty state with the next step, never a placeholder number.

## 1. Mental model the UX must make obvious

The operator thinks in one sentence: "Base price per rating point, times the premiums
that stack on top, unless a specific advertiser or campaign replaces one of them." The
workspace makes that sentence literally visible. The price for any slot is:

    final CPP = base CPP x prime x show x position x ad-type   (each layer defaults to 1.0)

with per-scope overrides that either replace exactly one named layer or adjust the final
price. The operator never sees a single opaque premium; they see the stack, layer by
layer, with the running product beside it.

## 2. Workspace layout

A single page, three regions, no modal-only flows (everything is inspectable in place):

- Left rail: the layer stack (the rate card). Vertical list of named layer cards.
- Center: the override list and the override builder.
- Right rail: the price-any-slot tester, pinned, always live.

A top strip shows the channel base CPP and the active currency (ILS) and units (per
second per rating point). The base is the only number that is not a multiplier; it is
labeled "base, not a premium" so it is never confused with a layer.

## 3. The layer stack (left rail)

One card per named layer, in canonical order: Base, Prime/daypart, Show, Position,
Ad-type. Each card shows:

- the layer name and a one-line description of what it prices,
- its current multiplier(s), inline-editable, with the configured default shown as a
  ghost value when unedited,
- a state chip: Live (multiplied into revenue today), Wired-off (defined but not yet
  activated, owner-gated), or Empty (no values configured yet).

The Position and Ad-type cards currently render Wired-off, because their multipliers are
not 1.0 and activating them moves real revenue (see design doc, owner gate). The chip is
not decorative: it is the honest statement of what the engine does today. A Wired-off card
shows its configured values greyed, with a one-line note: "configured, not yet applied to
revenue; activation is an owner decision."

Editing a multiplier updates the worked example (Section 7) in real time. There is no
Save-and-pray: the operator sees the price move as they type.

## 4. The override list (center, top)

A table of every override rule, one row each, columns:

- Scope: the predicate in plain language (advertiser X, optionally campaign Y, position 2,
  show Big Brother), rendered as labeled chips, not raw ids.
- Target: which named layer it replaces, or "final" for an adjust-final rule.
- Effect: the replacement multiplier or the absolute CPP for that layer, or the percent
  for an adjust-final rule.
- Status: Active, or Shadowed (a more specific override wins for every slot this one could
  match), or Draft (saved but not enabled).

Shadowed rows are greyed with a one-line reason and a link to the winning rule. The
operator sees dead rules without having to reason about precedence themselves. This is the
first error-proofing surface: a rule that can never fire is visible at a glance.

## 5. The override builder (center, bottom)

A single predicate row that reads as a sentence:

    WHEN advertiser = [X]  [AND campaign = [Y]]  [AND position = [2]]  [AND show = [Big Brother]]
    THEN  REPLACE the [position] layer with [value]
          OR    ADJUST the final price by [+/- percent]

Rules for the builder, each chosen to prevent a class of mistake:

- Channel is never a scope field. The operator owns one channel; it is sourced from
  settings.operator_channel and shown read-only. A competitor channel can never be
  entered, so a pricing rule can never be redirected off the owned channel (competitor
  boundary, enforced in the predicate engine, mirrored here so the field does not exist).
- The Target dropdown lists only named layers that exist plus "final". Choosing REPLACE
  on a layer shows, live, the rate-card value being replaced and the delta for a matching
  sample slot, so the operator sees exactly what their rule changes before saving.
- Absolute-CPP replacement is a distinct entry mode from multiplier replacement, labeled
  clearly, because an absolute value is authoritative and overrides the running stack for
  that layer. The builder spells out the consequence inline.
- Campaign implies advertiser (the default assumption in design Section 5). The campaign
  field is disabled until an advertiser is chosen, so an orphan campaign scope is
  impossible.
- A rule that would shadow an existing rule, or be shadowed by one, surfaces the
  precedence preview (Section 6) before the Save control enables.

## 6. Precedence preview (the core error-proofing)

Whenever the operator builds or edits a rule, or whenever several rules could match one
slot, the workspace shows the resolved outcome before anything is saved:

- It picks a representative matching slot (or the slot in the tester, if set) and resolves
  the full stack against all active overrides.
- It shows the winning override per layer and greys the shadowed ones, each with a
  one-line reason in plain language, for example: "the position-2 override for advertiser
  X wins over the advertiser-wide X discount because it matches more scope dimensions
  (more specific)."
- Ties are broken by an explicit priority field, then row order; the preview states which
  tiebreak applied, so a tie is never silent.

The operator sees the outcome, not the rule soup. Saving a rule that produces a surprising
winner requires a deliberate confirm with the reason shown. This is the single most
important surface for preventing a wrong price from shipping.

## 7. Price-any-slot tester (right rail, pinned)

One form: pick advertiser, campaign, show, daypart, position, ad-type. The result is the
full breakdown, top to bottom:

    base CPP                         60.00   (channel base)
    x prime                          1.10    (rate card: prime daypart)
    x show (Big Brother)             1.25    (rate card: per-show)        [wired-off today]
    x position 2                     1.15    (rate card: position-in-break) [wired-off today]
    override: position 2 for X       0.90    (replaces position layer)   [active]
    = final CPP                      ...

Every line names its source: rate card, or a specific override, or wired-off (shown for
transparency but not multiplied into the live total). The live total only multiplies the
layers the engine actually applies today, so the tester never overstates the price. A
wired-off line is rendered struck-through-light with its value visible, so the operator
sees the full intended model and the honest current behavior side by side.

The tester is the operator's sandbox: change any input and watch the breakdown recompute,
with zero ambiguity about why the number is what it is.

## 8. Guardrails surfaced inline

- Final-CPP floor and ceiling: if a composed price falls below the configured floor or
  above the ceiling, the line is flagged inline with the bound it crossed, not silently
  clamped.
- Below-cost warning: a price under the configured cost basis shows a warning chip; the
  operator can still save, but the warning is logged with the rule.
- Explicit-zero for promo: a zero ad-type multiplier (promo) requires a one-click confirm
  so a zero price is never an accident.
- No silent collapse: an override that would zero or invert the whole stack is blocked with
  a clear message, never applied silently.

## 9. Empty and honest states

- No overrides configured: the override list shows "Rate card only. No per-advertiser or
  per-campaign overrides yet," with a one-click path to the builder. It does not invent a
  rule.
- A layer with no configured values: the layer card shows Empty with "no values yet;
  defaults to 1.0 (no effect)," so an empty layer reads as a true no-op, not a missing
  number.
- A scope that matches no slots in the current schedule: the builder warns "this scope
  matches no slots in the loaded schedule" before saving, so a rule that can never fire is
  caught at authoring time.

## 10. Data contract each surface reads

- Layer stack: the rate-card tables (base_price, program_type/prime, show, position,
  ad_type) plus each layer's Live/Wired-off/Empty state derived from whether the engine
  multiplies it today (position and ad_type are Wired-off until owner activation).
- Override list and builder: the advertiser/campaign override rules with their target
  layer, effect, scope, and computed Active/Shadowed/Draft status from the most-specific
  resolver.
- Precedence preview and tester: a single price_slot-style composition call that returns
  the per-layer PriceBreakdown (base + named layers + applied overrides, each with its
  source), so the UI renders exactly what the engine computes, never a re-implemented
  formula. This is why the engine exposes PriceBreakdown (design doc S1): the UI binds to
  the breakdown, so a number on screen and a number in the optimizer can never drift.

## 11. Build order (UX, after the engine stages it depends on)

1. Layer stack read-only view bound to the rate card, with Live/Wired-off/Empty chips.
   Depends on engine S1 (PriceBreakdown). No money-path change.
2. Price-any-slot tester bound to the breakdown. Depends on S1. Read-only, safe.
3. Override list (read-only) bound to the existing advertiser rules. Safe.
4. Override builder + precedence preview. Depends on engine S5-S7 (target-layer override,
   campaign scope, most-specific resolver) and the 3 open decisions.
5. Inline editing of rate-card multipliers and override effects. Each edit is a real
   pricing change, so it ships behind the same owner gate as the layer activations.

Steps 1-3 are safe to build before the revenue-changing engine stages land, because they
only read the breakdown. Steps 4-5 follow the engine stages and the owner sign-off.
