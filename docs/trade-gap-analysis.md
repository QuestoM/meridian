# Trade gap analysis

Measured against docs/media-domain-from-the-trade.md (2 August 2026 transcript
extraction), file and line, as of this survey. Nothing in this document changes
code. Ordered as the source document orders its nine gaps; the three that carry
money (positions, creatives, goal-based order) are covered in the most depth.

## 2. Positions: the product is wrong today

The transcript is explicit: preferred positions are **1, 2, 3, 4, 5 and L**,
where L is a distinct thing (not "the last ordinal"), and which positions count
as preferred is per client/per agreement. The product models **1, 2, 3 and a
"last" that is defined as "whatever position equals break_size"** — i.e. it
never models 4 or 5 at all, and it never models L as an independent concept.

Every place a position is defined, priced, constrained, counted or displayed:

- `config/optimization_weights.yaml:22-29` — `premiums.position_in_break` has
  keys `1`, `2`, `3`, `default_middle`, `last`. No `4`, no `5`, no `L`.
- `kairos/optimize/pricing.py:313-326` — `PricingModel.position_premium(position,
  break_size)`. Positions 1-3 use their explicit premium; **"last" is computed
  as `position == break_size and position > 3`**, i.e. last is derived from
  break length, not stored or requested as its own thing. There is no branch
  for 4 or 5 at all: a spot at position 4 or 5 in a break longer than 5 silently
  falls through to `default_middle` (line 326).
- `config/optimization_weights.yaml:49` / `kairos/optimize/pricing.py:195,238` —
  the whole `position_in_break` layer ships **activation-off**
  (`pricing_activation.position: false`), so today none of this even reaches
  revenue; it is wired but dormant.
- `kairos_api/pricing_api.py:59-60,73-74,151,194,280,311,343-347` — the pricing
  API's `position` layer and the `PriceSlotRequest.position` field
  (`ge=1`, line 194) are a plain integer with no upper bound and no `L` token;
  `price_slot` (line 343) calls the same `position_premium(req.position,
  req.break_size)` above.
- `kairos/optimize/_rule_models.py:18-19,38,50-53,80,95,107-136` — the
  constraint/advertiser-rule engine's position scoping (`GOLD_POSITION`,
  `allow_positions`, `scope_positions`) matches on free-text tokens
  (`dimension_matches`), so a token like `"4"` or `"L"` is not rejected by the
  type system, but nothing in the product ever writes or offers such a token
  (see next two files).
- `data/advertiser_rules.csv:1-3` — the `allow_positions` column holds values
  like `"first,last"` (row `ADV_02`), i.e. the advertiser-rule vocabulary is
  the **word set first/second/third/last**, not 1-5/L, and is a second,
  parallel vocabulary from the pricing layer's `1/2/3/last` integer keys.
- `kairos_api/advertiser_conditions.py:195-209` — `_position_options()` builds
  the dropdown an operator scopes a premium/pressure rule to. It reads
  `DEFAULT_BREAK_POSITIONS` (`kairos/model/spec.py:42`, value
  `("first", "middle", "last")`) plus an appended `"gold"` token
  (`GOLD_POSITION`). **This is the narrowest vocabulary in the codebase: no
  numeric positions, no 4, no 5, no L, no distinction between "last" and "any
  position beyond a fixed count."**
- `tv-break-dashboard/src/rules/pricing-layers-lib.js:28-31` —
  `POSITION_NAMES` mirrors the YAML: `1, 2, 3, default_middle, last`. This is
  what `PricingManager.jsx` and `RateCardEffect.jsx` render.
- `tv-break-dashboard/src/rules/PricingSlotTester.jsx:14,45,124-125` — the
  price-any-slot tester's position field is `<input type="number" min="1">`
  (line 125), so an operator cannot even type `L` to test it; it is coerced to
  `Number(slot.position)` at line 45.

**Preferred-position percentage / two counting methods**: a repo-wide search
for `preferred_position`, `preferred position` (case-insensitive) across
`kairos/`, `kairos_api/` and `tv-break-dashboard/src/` returns **zero
matches**. Neither the "agency method" (breaks appeared in, counting a Top+Tail
break twice) nor the "channel method" (out of total broadcasts) exists in any
form. No figure on any surface computes a preferred-position percentage at
all, so there is no confusion between the two methods today only because
neither is built.

**Distance from the trade**: the pricing layer's five-slot table is a data
change plus a new "L" concept threaded through `position_premium`,
`PriceSlotRequest`, `pricing-layers-lib.js` and `PricingSlotTester.jsx`. The
constraint/advertiser-rule vocabulary (`first/middle/last` + `gold`) is a
separate, narrower surface that needs the same 1-5/L vocabulary independently,
plus the per-client/per-agreement "which positions are preferred" concept,
which does not exist anywhere yet. The percentage metric and its two counting
methods are new, from zero.

## 3. Creatives: what data/campaign_assets.csv and its API already model

`data/campaign_assets.csv` (schema at `kairos_api/campaigns_assets.py:42-68`,
80 lines of demo data) already carries, per row: `asset_id`, `campaign_id`,
`advertiser`, `channel`, `house_number`, `version_name`, `spot_type`,
`length_class`, `duration_seconds`, plus a set of tri-state unknowns (`media`,
`video_format`, `aspect_ratio`, `loudness`, `clearance`) and provenance
(`first_observed_on`, `last_observed_on`, `airings_observed`,
`identity_source`, `source_file`, `is_demo`).

What is real today (module docstring, `kairos_api/campaigns_assets.py:1-27`):
house number, version name, spot type and duration are read from the traffic
log, one row per house-number-per-campaign. `assets_by_campaign()` (line 211)
groups by campaign; `campaigns_api_detail.py:51-66` exposes
`GET /campaigns/{id}/assets`.

What is missing, measured against the transcript:

- **Validity window.** `COLUMNS` (`campaigns_assets.py:42-68`) has no
  "valid until" / expiry field at all. The transcript's "each creative also
  carries a validity window: until when it may be scheduled. That is a
  constraint too" is entirely unmodelled; nothing in the constraint engine
  (`kairos/optimize/_frequency_rules.py`, `kairos/optimize/constraints_store.py`)
  reads an asset expiry.
- **Top and Tail adjacency.** A repo-wide search for `top.*tail`, `top_tail`
  (case-insensitive) across `kairos/`, `kairos_api/`, `data/` and
  `tv-break-dashboard/src/` returns no matches. There is no representation of
  "these two creatives in this campaign must air in the same break, separated
  by exactly one or two other ads." `kairos/optimize/_frequency_rules.py:49-50`
  has `MAX_CONSECUTIVE` (no same target in N adjacent positions) and
  `MIN_SEPARATION` (min gap in minutes or positions between two named things),
  which are the right primitive shape for a "separated by exactly 1-2" rule,
  but they operate on advertiser/campaign frequency, not on named creative
  pairs within one campaign, and nothing calls them with a same-break,
  same-campaign, cross-creative pairing today.
- **Twenty creatives per campaign.** No cap is enforced or even counted
  anywhere in `campaigns_assets.py`; `summarise()` (line 222) counts assets but
  never checks against a ceiling. Not a hard blocker, just unenforced.
- **House number differs per channel, for the same creative.** The schema
  stores `channel` and `house_number` on the same row
  (`campaigns_assets.py:46-47`), and each row belongs to exactly one campaign
  which itself carries one `channel` (`data/campaigns.csv:1`, column list).
  There is **no creative-level identity that spans channels** — no
  `creative_id` or similar that says "this is the same underlying spot, filed
  under house number X on channel A and house number Y on channel B." Today a
  creative is only ever seen through the traffic log of the one channel that
  aired it, so the cross-channel binding the transcript describes (Owner
  issues the house number, operator pastes it into Jumbo) has no home in the
  data model to attach to.
- **Dashboard surfacing.** A repo-wide search of `tv-break-dashboard/src` for
  `campaigns_assets`, `campaign_assets`, `CampaignAssets` and `house_number`
  finds only `tv-break-dashboard/src/vocabulary.js:183` (a label string,
  `object.house_number`). There is no dedicated creatives/assets panel
  component; the API (`campaigns_api_detail.py:51-66`) is unconsumed by any
  found `.jsx` file in this search.

## 1. The goal-based order: what exists

The transcript names this the destination: the agency sends a TRP/GRP goal
against a named audience instead of a spot list, and the channel owns
placement end to end.

What exists today is a **reporting field, not an order type or a placement
mode**:

- `data/campaigns.csv:1` (header) carries `goal_kind`, `goal_value`,
  `rating_goal_points`, `rating_goal_audience` at both campaign and flight
  record types (row 2, `flight` record: `goal_kind=grp`, `goal_value=65.00`).
- `kairos_api/campaigns_api.py:86-87,110-111,122-123,134-135` — these fields
  are plain request/response attributes on `CampaignCreate`/`CampaignUpdate`.
  `kairos_api/campaigns_api_store.py:79-80,95-96,111,115-116,188-199,243-244`
  stores and validates them (`GOAL_KINDS`, `GOAL_KIND_VOCABULARY`) and attaches
  a human label plus a `rating_goal_measurable` flag
  (`campaigns_commitment.py:229-234`).
- `kairos_api/campaigns_delivery.py:276-306` — the only consumer.
  `delivery_for()` computes `rating_progress` by comparing
  `aired["rating_points_planned"]` (a figure derived from days that already
  have a schedule) against `terms["rating_goal_points"]`. This is **pacing
  measurement against a target the schedule was built without reference to**,
  not placement driven by the goal.

What does not exist:

- No order type where a goal is the only input. Every campaign in
  `data/campaigns.csv` still has `starts_on`/`ends_on` and is placed through
  the ordinary break-board/schedule-editor path
  (`kairos_api/break_api.py`, `tv-break-dashboard/src/plan/day/ScheduleEditor.jsx`)
  driven by the optimizer working from spot-shaped inventory, not solving "hit
  this TRP for this audience, choose the breaks yourself."
  `rating_goal_audience` is validated against a fixed `AUDIENCE_VALUES`
  vocabulary (`campaigns_commitment.py:232-234`) used only for the label shown
  next to the goal, never fed into the optimizer's objective
  (`kairos/optimize/objective.py`) or into break selection.
- A repo-wide, case-insensitive search for `TRP` and `goal_based`/`goal-based`
  across `kairos/` and `kairos_api/` returns no matches beyond the
  `rating_goal_*` reporting fields above; the term "TRP" from the transcript
  does not appear in the engine at all (the engine uses `grp`/`rating_goal_points`
  and a generic named-audience string).

**Distance from the trade**: this is the largest gap of the nine. The reporting
skeleton (a stored goal, a named audience, a measurable/unmeasurable flag) is a
usable foundation, but there is no order type that omits the spot list, no
solver that treats the goal as the objective, and no UI path that lets an
agency submit only a goal.

## 4. Make good: not a ledger

The transcript: make good is managed at three levels at once (campaign,
advertiser, agency) as an **accrual and utilisation ledger** (an agency
accrues, e.g., 10% of spend and may spend that credit on a different
campaign later).

What exists: `kairos/optimize/pacing.py:139-145,416` —
`project_make_goods()` returns `ShortfallCampaign` records: a campaign
"projected to finish under target," carrying `projected_shortfall = max(0, 1 -
projected_frac)`. `kairos_api/pacing_alerts_api.py:28-54` exposes this as
`GET /api/make-good-alerts`: a **forward-looking, per-campaign risk flag**,
not a ledger. `data/campaigns.csv:1` has a `bonus_ils` column (bonus spend),
but nothing accrues it at advertiser or agency level, and nothing tracks
utilisation (spending an agency's accrued credit on a different campaign).
A repo-wide search for "accrual", "utilis", "ledger" scoped to make-good
context finds no such structure. This is a naming collision more than a
partial build: the product's "make good" is a pacing alert; the trade's
"make good" is a three-level balance sheet. They share nothing but the name.

## 5. As Run ingestion: absent

A repo-wide case-insensitive search for `as_run`, `as run`, `asrun` across
`kairos/` and `kairos_api/` returns only unrelated matches (the English word
"run" inside phrases like "no plan was run"). There is no ingestion of a
second-by-second broadcast JSON file, and billing/delivery
(`kairos_api/campaigns_delivery.py`) is computed from the traffic-log-derived
schedule/day state, not from an As Run feed. The transcript's core point ("the
schedule is not what happened... billing and delivery must be computed from As
Run, never from the plan") is unaddressed: today's delivery figures are
computed from the same planned/observed traffic-log path the transcript says
is not truth.

## 6. Run parameters as an explicit per-run priority set: partially present, wrong shape

The transcript wants a **per-run priority set**: large media companies, direct
clients, success deals, new-campaign de-prioritisation, competitive
separation, regulatory-check switch — changed between runs.

What exists is a `PRESSURE` effect on the advertiser/agency condition engine,
which is close in mechanism but wrong in scope and shape:

- `kairos/optimize/_rule_models.py:25-29` — `PRESSURE` is one of four effects
  (`PREMIUM, REQUIRE, FORBID, PRESSURE`), documented as "a placement-only
  lever: steers WHERE the optimizer wants to place" (line 25).
- `kairos/optimize/advertiser_rules.py:225-270,402-423` —
  `pressure_multiplier()` composes matching `PRESSURE` rules multiplicatively
  and `rank_value()` (line 260) multiplies the real premium by it for ranking.
  This is functionally identical to the transcript's "pressure level 1.3 means
  treat it as though it paid 1.3 times its budget."
- **Scope mismatch, exactly as the transcript flags it.** The condition
  dataclass (`_rule_models.py:80-155`) already supports `scope_campaigns`
  (line 86), so campaign-level scoping is technically possible in the model.
  But the data file and API that actually create these rules are advertiser-
  keyed only: `data/advertiser_conditions.csv:1` header is
  `advertiser_id,rule_id,scope_positions,scope_genres,scope_dayparts,effect,value,notes`
  — **no `scope_campaigns` column** — and
  `kairos_api/advertiser_conditions.py:114-360` (`_load_frame`,
  `create_condition`, `update_condition`) never writes or reads one. The
  transcript's exact complaint — "the owner had built this at advertiser
  level; the media professional was explicit that it belongs at campaign
  level" — is unresolved: the engine's data class is campaign-ready, the
  storage layer is not.
- **No run-level priority set at all.** There is no place in
  `kairos/optimize/objective.py`, `config/optimization_weights.yaml`, or any
  scenario-parameters API (`kairos_api/scenario_api_parameters.py`) that lets
  an operator set, per run, priority levels for "large media companies,"
  "direct clients," "success deals" (`עסקאות הצלחה` — no Hebrew or English
  match found for this term anywhere in the repo), or de-prioritise "new
  campaigns." Competitive separation exists as a hard boundary
  (`kairos/optimize/_frequency_rules.py` frequency/separation rules,
  documented elsewhere as "airtight" per prior QA), but as a constraint, not a
  per-run priority weight. Regulatory checks as an explicit on/off switch for
  a run were not found in `kairos/optimize/objective.py` or the scenario
  parameters API.

## 7. Rate card: date-specific and seasonal layers partially present, rest as documented missing

The transcript's nine-layer rate card, checked layer by layer:

1. Base price/hour — `config/optimization_weights.yaml:6`
   (`base_price_per_second_per_tvr_point`). Present.
2. Day-of-week — `optimization_weights.yaml:31-38`. Present, live.
3. Hour — not found as a distinct layer (only day and programme-class/segment
   premiums; `kairos/optimize/pricing.py:328-334` composes
   `program_premium x day_premium` for `segment_premium`, no hour-of-day
   table).
4. Programme category — `optimization_weights.yaml:10-14`
   (`premiums.program_type`). Present, live.
5. Specific programme — the "show" layer,
   `kairos_api/pricing_api.py:47,59` ("show, position ... are wired but off"),
   `kairos/optimize/pricing.py` `enable_show`. Present, wired, off by default.
6. **Specific date** — the owner confirmed this is missing in the transcript.
   Measured: `data/calendar_events.csv:1-3` and
   `kairos/optimize/pricing.py:34-35,69-74,204-210` (`event_day_multipliers`,
   `load_event_day_multipliers`, `enable_events`) give a genuine per-date (or
   per-date-range) `price_multiplier`, gated by
   `pricing_activation.events: false`. This is close to the transcript's ask
   but framed as "calendar events," not raw specific-date pricing, and ships
   off.
7. Break position — the `position_in_break` layer covered in gap 2. Present,
   wired, off, and wrong-shaped (1/2/3/last only).
8. Seasonal/periodic premiums (e.g. pre-Passover) — the same calendar-events
   mechanism as (6) can express this (an event with a date range and
   `intensity`/`price_multiplier`, `data/calendar_events.csv` row 3 is a Pesach
   holiday entry with `price_multiplier=1.0`), also gated off.
9. Per-agency/per-advertiser/per-campaign adjustability of all the above — the
   advertiser condition engine (`kairos/optimize/advertiser_rules.py`,
   `data/advertiser_conditions.csv`, `data/agency_conditions.csv`) gives
   premium/require/forbid/pressure adjustability per advertiser and per agency
   today; campaign-level is the same gap flagged in item 7 above
   (`scope_campaigns` modelled, not exposed).

Gold-break separate rate card: `config/optimization_weights.yaml:58`
(`max_gold_breaks_per_hour`) and `GOLD_POSITION`
(`kairos/optimize/_rule_models.py:19`,
`kairos_api/advertiser_conditions.py:208`) mark gold breaks as a distinct
scope operators can condition on, but no separate gold rate table (base price,
premiums) was found; gold breaks price through the same `segment_premium`
path as any other break, only tagged for constraint/condition purposes.

## 8. The block booking stage, and an email approval state machine: absent

The transcript's near-term ask is a media-company portal covering two stages
(a name-optional block booking, then the actual buy about two days out) with a
visible status such as "awaiting approval" that can be actioned from inside an
email.

Measured: `kairos_api/campaigns_api_store.py:76,108,217,265` — `status` is a
single free-text column (`data/campaigns.csv:1` header includes `status`),
defaulting to `"active"` when blank (line 217) and used only to sort active
campaigns first (line 265). A repo-wide search of
`kairos_api/campaigns_api*.py` and `kairos_api/campaigns_commitment.py` for a
status vocabulary (`STATUS_VALUES`, `CAMPAIGN_STATUS` or similar validated
enum, the pattern used elsewhere for e.g. `GOAL_KINDS`,
`campaigns_api_store.py:46-47`) finds none: `status` is not validated against
any list, so nothing stops or names an "awaiting approval" or "block booking"
state today. There is no two-stage order model (reserve-capacity-without-a-
named-advertiser, then the flighted buy), no approval state machine, and no
email-actionable link anywhere in `kairos_api/` or
`tv-break-dashboard/src/`. This gap is unbuilt from zero.

## 9. Owner and Jumbo integration seams: absent

A repo-wide, case-insensitive search for `owner` (as the Israeli traffic
system, distinct from the many unrelated uses of the English word "owner" in
this codebase) and `jumbo` across `kairos/`, `kairos_api/` and `docs/` (other
than the transcript itself) returns no matches. There is no database-link
seam, no house-number-issuance workflow, and no Jumbo API client anywhere in
the repository. `house_number` (gap 3) is read-only from the traffic log
(`kairos_api/campaigns_assets.py`); nothing writes one, receives one from
Owner, or pushes one to Jumbo.

## Summary table

| # | Gap | State |
|---|---|---|
| 1 | Positions 1-5+L, preferred-position % | Wrong shape (1/2/3/last only, "last" derived not distinct); percentage entirely absent |
| 2 | Creatives as real objects | House number/version/duration real; validity window, Top-and-Tail adjacency, cross-channel identity all absent |
| 3 | Goal-based orders | Reporting field only; no order type, no goal-driven placement |
| 4 | Make good ledger | Naming collision: today it is a pacing risk alert, not an accrual/utilisation ledger |
| 5 | As Run ingestion | Absent; billing computed from plan, not from broadcast truth |
| 6 | Run parameters, per-run priority set | Pressure lever exists but advertiser-scoped not campaign-scoped; no run-level priority set; "success deals" concept absent |
| 7 | Rate card by date, seasonal | Calendar-events layer gives a close approximation, wired off by default; hour-of-day layer absent |
| 8 | Block booking + email approval state machine | Absent; no status vocabulary beyond a free-text field defaulting to "active" |
| 9 | Owner / Jumbo integration | Absent |
