# Trade re-audit: what the gap analysis missed

Adversarial re-read of `docs/media-domain-from-the-trade.md` sentence by sentence
against the code, auditing `docs/trade-gap-analysis.md` rather than trusting it.
Read-only. Every claim below carries the file and line it was measured at on
2026-08-09, working tree at `fc144daa`.

## Why things were missed, structurally

The transcript ends with its own summary, "What the product is missing, drawn
from the above", nine numbered items (lines 214-227). The gap analysis has
exactly nine sections, in exactly that order. **It audited the transcript's
summary, not the transcript's body.** Everything in the body that the summary
did not lift became invisible: the rating currency, the order's own fields, the
time tolerance, cancellation, withheld inventory, the direct-versus-agency
distinction, the daily-amendment lifecycle. That is where most of what follows
comes from.

A second cause: the gap analysis was committed in `cdc3935f` (2026-08-03), the
same commit that shipped `kairos/optimize/positions.py`. It has not been touched
since. **Two of its nine were superseded, one of them by its own commit**, and
one line of its evidence is now false (below). It reads as a live register and
is not one.

---

## 1. The preferred-position percentage is built, tested, bilingual, and unreachable

**The trade's sentence:** "They give different percentages for the same
schedule. The product must state which method a figure uses, because the two
parties audit each other with it."

**What the product would need:** an operator path to agree a preferred set per
client, and one surface that quotes the percentage under a named method.

**What it has:**
- `kairos/optimize/positions.py:322` — `preferred_position_rate(...)`, both
  methods implemented, agency and channel, Top-and-Tail double-counting handled
  explicitly at lines 364-367.
- `config/optimization_weights.yaml:68-70` — `preferred_positions:
  channel_default: null`, `per_advertiser: {}`. Shipped unset, deliberately, and
  the comment says a guessed percentage is worse than none. Correct.
- `tv-break-dashboard/src/rules/PricingPreferredPositions.jsx:11-60` — the card
  that renders the set. **It has no input, no onChange and no save.** It is
  display only.
- `kairos_api/pricing_api.py:180` returns `preferred_positions` on GET. There is
  no PUT field for it: `PricingUpdate` (`pricing_api.py:187-197`) documents
  `base_price_per_second_per_tvr_point`, `premiums` and `pricing_activation`.
  The key would survive the deep merge, but no schema names it and no screen
  offers it.
- `grep -rn preferred_position_rate` outside `tests/`: **zero callers.** Only
  `tests/test_positions.py` invokes it.

**What it blocks:** the number the channel and the agency audit each other with
cannot be produced by anybody using the product. Not "computed wrongly" —
uncomputable, because the configuration it is gated on has no way in.

**Previous audit:** called the metric "new, from zero" (line 76). It was built in
the same commit, and left with no operator path. The register never recorded it.

---

## 2. Two contradictory answers to "which positions are preferred", and one of them badges every spot

**The trade's sentence:** "Which of them count as preferred is per client and per
agreement, so it is configurable, not fixed."

**What it has, in two places at once:**
- `kairos_api/break_api_pod_spots.py:31` —
  `PREFERRED_POSITIONS = ("1", "2", "3", "4", "5", "L")`, a hardcoded constant.
  `break_api_pod_spots.py:132,147` set `"preferred": str(code) in
  PREFERRED_POSITIONS`. Every real ordinal 1-5 and every Last is therefore
  `preferred: true`.
- `tv-break-dashboard/src/plan/break/PodBoard.jsx:373` applies
  `pod-pos-preferred` from that flag; the class is styled at
  `tv-break-dashboard/src/plan/break/break-pod.css:285`.
- `tv-break-dashboard/src/rules/PricingPreferredPositions.jsx:37`, two screens
  away, prints "No preferred set is configured, so no preferred-position
  percentage is computed anywhere."

**What it blocks:** on a five-spot break every spot carries the preferred badge,
so the badge distinguishes nothing and an operator reading position quality off
the pod board reads noise. Meanwhile the pricing screen tells the same operator
that nothing is configured. Both sentences are shipped; they cannot both be
acted on.

`break_api_pod_spots.py:32-33` is honest in its own basis string ("this is a
default and not a reading of any agreement") — but the basis string is on the
API payload, not on the badge, and it does not reconcile with the other surface.

**Previous audit:** missed entirely. It surveyed the position *vocabulary* across
eight files and did not ask whether two of them answer the same question
differently.

---

## 3. The rating currency is not the trade's currency, and the vocabulary cannot name it

**The trade's sentence:** "The trading currency is **Jewish households,
quarter-hour rating, overnight plus one**, where plus one is deferred viewing."

**What the product would need:** the universe of every TVR it holds recorded,
and the market's universe expressible as a goal base.

**What it has, measured in three parts:**

*Universe.* `kairos_api/campaigns_commitment.py:37-51` — `ALL_VIEWERS` is the
one audience marked `measurable: True`, on the stated grounds that "the ratings
on the traffic log and in the reference month are the general-audience TVR with
no demographic split". The full vocabulary
(`campaigns_commitment.py:42-88`) is: all viewers, adults 18+, women 25-54, men
25-54, children 4-14. **"Jewish households" is not in it.** The list is sourced
from `docs/campaign-rate-card-research.md` section 11
(`campaigns_commitment.py:90`), not from the transcript. `data/Programmes.csv:1`
carries a bare `TVR` column with no universe qualifier anywhere in the file.

*Quarter-hour.* Present and correct: `kairos/optimize/qh_billing.py:1-19`,
owner-gated off at `kairos/optimize/pricing.py:206`. This half of the currency
is modelled honestly.

*Overnight plus one.* `grep -rniE "overnight|consolidat|deferred"` over `kairos`,
`kairos_api`, `config`: the only `overnight` is a daypart band name
(`kairos/model/audience_frame.py:54`). **Nothing anywhere records whether the
TVR the product holds is an overnight figure or a consolidated one**, and
nothing models the day-after revision.

**What it blocks:** every ILS figure in the product is
`base_price_per_second_per_tvr_point x TVR x premiums`
(`docs/campaign-rate-card-research.md:52`), denominated in a rating unit the
market does not settle in. And an operator whose client bought on the market's
actual currency cannot state that goal: the nearest expressible base is "all
viewers", which the product then reports progress against as measurable. The
module's own promise at `campaigns_commitment.py:20-22` — "never as a silently
substituted base" — holds only for the four audiences it happens to list; the
one the trade actually trades on is not refused, it is absent.

**Previous audit:** missed entirely. Its gap 1 was about goal-based *orders* as
an order type. It never asked what unit the goal, or the price, is denominated
in.

---

## 4. As Run would not fix delivery, because the rating is a second, later, revisable feed

**The trade's sentences:** "The final rating is only known the day after
broadcast, and it moves: one, two, even three points can be added. On a
programme rating ten, three points is a thirty percent revision. Any figure the
product shows on the night is provisional and must say so."

**What it has:** `data/campaign_delivery.csv:1` carries `rating_points_planned`
and no delivered-rating column. A row with `air_state=aired`
(`data/campaign_delivery.csv:2`) still reports `rating_points_planned=8.7000`.
`kairos_api/campaigns_delivery.py:106-109` labels it "Planned break rating from
the traffic log, on the all-viewers base." Honest, and structurally terminal.

**What it blocks:** the audit's gap 5 says As Run ingestion fixes billing. It
does not. As Run is a broadcast log — it gives spots, times and durations. It
never gives a rating point. Delivered TRP comes from the measurement panel, a
different source, arriving a day later, and revisable by up to 30 percent on a
low-rating programme. **Two feeds are missing and the register counts one.** A
team that builds As Run to the letter of gap 5 will still be unable to say what
a campaign delivered, and will not discover that until the feed is wired.

The provisional caveat is also missing as a distinct thing. The product's
caveats all say "planned, not delivered"; none says "even the delivered figure
moves tomorrow".

**Previous audit:** covered at the wrong altitude.

---

## 5. A promise the line beneath it removes: campaign amendments are not versioned

**The trade's sentence:** "The campaign is a living object updated daily, at
best. Mid-day re-planning is normal."

**What it has:**
```
kairos_api/campaigns_api_store.py:155-159
def snapshot_before_write(request: Any) -> None:
    """Version the campaigns store before a manual edit writes it."""
    from kairos_api import version_store
    version_store.snapshot_manual_edit(request, "campaigns")
```
`kairos_api/version_store.py:48-49` — `_LOGICAL_ORDER = ("settings",
"constraints", "overrides", "advertisers", "conditions", "events", "agencies",
"agency_links", "agency_conditions")`. `"campaigns"` is not in it.
`version_store.py:198-200` — `wanted = [...]; if not wanted: return None`.

Six callers believe they are versioning: `kairos_api/campaigns_api.py:248, 284,
351, 381, 410, 422`. All six are no-ops. Failure mode: silence.

`kairos_api/activity_log.py:11-14` records metadata only, never bodies, so the
content of an amendment is unrecoverable from there either. The only survivor is
a timestamped file copy in `data/_backups/` (`campaigns_api_store.py:147`) — not
addressable, no actor, no diff, no restore route.

`kairos_api/target_store.py:264-265` performs the *identical* no-op and
**documents it as one**. That is the house style, and it makes the campaigns
docstring the outlier rather than an oversight of convention.

**What it blocks:** the single most emphasised claim in the transcript is that
the campaign changes every day. Nobody can see what changed.

**Previous audit:** missed entirely.

---

## 6. There is no order, so the trade's "the time is a range" cannot even be violated

**The trade's sentences:** "`שם ערוץ, תאריך, שעה, שם תוכנית, אורך תשדיר`" and
"The time is approximate and everyone knows it... It is a range, not a
commitment."

**What it has:** no order, booking or reservation entity anywhere in `kairos/`,
`kairos_api/` or `data/`. The closest is the campaign/flight pair,
`data/campaigns.csv:1` and `kairos_api/campaigns_api_models.py:66-73`. Of the
trade's five order fields it holds channel (at campaign level) and a flight
*window*; it holds no time, no programme and no spot length. `data/campaign_flights.csv`
declares `scope_programmes`/`scope_dayparts` and is **header-only with no
writer**.

`grep` for `tolerance|requested_time|time_window|drift_minutes` across `kairos`,
`kairos_api`, `data`, `tv-break-dashboard/src`: the only `tolerance` hits are
floating-point epsilons (`kairos/optimize/refiner.py:285`,
`tv-break-dashboard/src/plan/day/day-board-settlement.js:28`).

**What it blocks:** an account manager holding a signed insertion order still has
nowhere to put it — which `kairos_api/campaigns_api.py:6` says in those words.
And the design question the transcript is warning about, whether a requested
20:40 is a point or a band, cannot be answered because there is no requested
time to be a band around.

**Previous audit:** its gap 8 covered the *portal stages* over the order. It did
not check the order's own five fields.

---

## 7. Direct clients are invisible, and their return runs the wrong way

**The trade's sentences:** "direct clients" as a placement priority, and "Direct
buying carries around a **twenty percent return**, which buyers routinely ask to
take as added media rather than cash."

**What it has:**
- No direct-versus-agency flag. `kairos/export/agency_layer.py:98-103` —
  `resolve()` returns `None` when no agency resolves, giving premium 1.0 and
  rebate 0 (`agency_layer.py:11-12`). **A direct client is byte-identical to a
  failed agency lookup.**
- `kairos_api/campaigns_read_clients.py:38-41` treats a client with no agency
  link as a data defect and prints a remedy: "Link it on the agency record."
  The product asserts every client must have an agency.
- `data/agencies.csv:2` — `AGY_01` carries `rebate_percent=4.0`,
  `commission_percent=15.0`. So the agency-bought client gets a 4 percent cash
  rebate and the direct client gets zero, where the trade says direct carries
  roughly twenty percent, usually taken as **added media**.
- `bonus_ils` (`data/campaigns.csv:1`) is a shekel amount, not added inventory.
  `kairos_api/campaigns_commitment.py:14` already records that a bonus shekel
  and a paid shekel are different things.

**What it blocks:** pricing a direct deal at all, and one of the six run
priorities the transcript names.

**Previous audit:** gap 6 listed "direct clients" as a missing *priority*. It
missed that the distinction does not exist, and that the money runs opposite to
the trade.

---

## 8. Cancellation, and the billing-started boundary: absent, with a two-state campaign

**The trade's sentence:** "Cancellation rules are demand-dependent: a day before
broadcast billing has started and cancellation is refused, but in high-demand
periods a channel may welcome a cancellation because it can resell."

**What it has:** `kairos_api/campaigns_api_words.py:29` —
`STATUSES = ("active", "ended")`. `"ended"` means the flight finished
(`campaigns_api_words.py:40-48`), carries no refusal and no date boundary.
`data/campaign_delivery.csv:1` `air_state` takes `aired|scheduled|unknown` — no
cancelled value. An exhaustive case-insensitive sweep for
`cancel|cancellation|ביטול` across `kairos`, `kairos_api`, `data`,
`tv-break-dashboard/src` returns only dialog dismiss buttons, edit-undo, JS
stream teardown and prose. `resell|resale|resold` appears only in
`docs/media-domain-from-the-trade.md:194-195` and a dossier quoting it.

No code anywhere compares a date against broadcast-minus-one-day.

**What it blocks:** the operator cannot record the single most common commercial
event after a booking, and cannot tell a refused cancellation from a welcomed
one.

**Previous audit:** not one of the nine.

---

## 9. No break can be held back

**The trade's sentence:** "The channel does not release every break, and the
schedule is not known a week out. Both sides are working under deliberate
opacity."

**What it has:** the complete break record is `data/breaks.csv:1` —
`break_id, segment_id, ordinal, channel, day, programme, offset_seconds,
duration_seconds, is_gold, constraint_id, actor, saved_at, note`. The engine's
own is `kairos/optimize/guardrails.py:36-47`. The constraint effect vocabulary
is closed at `kairos/optimize/constraints_store.py:68-75`:
`fix_offset, offset_window, pin_count, duration_range, gold, forbid` — and
`forbid` *deletes* the break (line 74), it does not hold one back.
`kairos/optimize/inventory.py:63-79` has a `SlotDemand.available`, consumed only
by `inventory_hard_cap` (`inventory.py:214-241`), which its own docstring calls
"HOOK (not yet enforced)... intentionally side-effect-free and unused by the
live path".

**What it blocks:** every break in the plan is implicitly fully offered to the
market. There is no way to model the withheld inventory that both sides of this
trade negotiate around.

**Previous audit:** not one of the nine.

---

## 10. The engine documents its own demand weights backwards

`kairos/optimize/advertiser_rules.py:26` — segment demand is "supplied to
optimize_breaks as `demand_weights`... **Off by default**."
`kairos/optimize/demand.py:8-12` — "The weights are always computed (never gated
behind a flag)". `kairos/optimize/day_core.py:95-102` calls the builder
unconditionally on every optimize path.

The behaviour is safe (identity when no rules match, which is today's state
because `data/advertiser_conditions.csv` is header-only), so nothing is wrong
in the numbers. The docstring is wrong, and it is the docstring a reader
consults before deciding whether activating advertiser conditions is a live
money change. It is.

**Previous audit:** missed.

---

## 11. Two more stores that model a thing nothing reads

Same shape as the `ADV_01..ADV_45` dead premiums the owner found.

- **`price_model`.** Validated against `PRICE_MODEL_VALUES = cpp | flat`
  (`kairos_api/campaigns_commitment.py:98-137,236`), stored
  (`campaigns_api_store.py:200`), labelled
  (`campaigns_api_words.py:83`). `grep price_model kairos/` → **no hits.** It
  never branches in `kairos/optimize/pricing.py`. Shipped data: 27 cpp, 10 flat,
  15 blank — and the flat ones price identically to the CPP ones.
  Separately, there is no revenue-share value, so the transcript's **success
  deals (`עסקאות הצלחה`)** cannot be stored even as a label.
- **`priority`.** Validated `guaranteed | preemptible`
  (`campaigns_commitment.py:115-130,239`), stored, labelled, and read by no code
  under `kairos/`. The optimizer never sees whether a campaign is preemptible.
  All 52 shipped campaigns leave it blank.

---

## 12. The regulatory switch runs the wrong way round

**The trade's sentence:** regulatory checks are "not a priority but a switch:
does this run also enforce them".

Measured: enforcement is **unconditional** inside the optimizer
(`kairos/optimize/guardrails.py:75-103`, `dp_refine.py:166`, `refiner.py:345`);
no flag skips it. So the switch the trade wants does not exist. Worse, the
*values* enforced come from `data/kairos_settings.json` via
`kairos/service.py:151-158`, **not** from the attested
`data/regulatory_guardrails.json`. The declared cutover,
`kairos_api/guardrail_store.py:409-423` (`settings_overlay`, described at line 23
as "the cutover onto the engine"), has **no production caller** — only tests. The
only bridge is a manual button, `POST /api/rules/guardrails/apply`
(`kairos_api/compliance_api_licence.py:188-218`). Until an operator presses it, a
licence change on disk changes nothing the engine does. Today the two agree only
because `tests/test_w0_4_guardrails.py:44` pins them together.

**Previous audit:** said the switch "was not found". Correct, and it stopped
there. It did not find that the compliance surface and the engine read different
files.

---

## 13. Lower confidence, stated as such

`kairos_api/assistant_prompt.py:140`, rule 24, names the product's objects as
"the week, the day, the break, the spot, the restriction, the target, the
make-good, the plan version and the model version. Use those words and no others
for them." Kai holds `get_agencies`, `get_agency_detail`,
`get_advertiser_pricing` and `get_campaign_pacing`, so the campaign, the
advertiser, the agency and the pod are objects its own tools return and rule 24
does not list. Read strictly the rule governs only how those nine are named, so
this may be harmless; I did not drive the assistant to test whether it refuses
the noun "campaign". Reported as a suspicion with the ambiguity named, not as a
measured fault.

---

## Where the previous audit was RIGHT, and where it has gone stale

Being right about a non-finding is worth as much as a finding.

**Still right, still open:**
- **Campaign-scoped pressure.** `data/advertiser_conditions.csv:1` is still
  `advertiser_id,rule_id,scope_positions,scope_genres,scope_dayparts,effect,value,notes`
  — no `scope_campaigns` column, and no `kairos_api/` hit for the name. The
  engine's dataclass is campaign-ready and the storage layer is not, exactly as
  gap 6 said.
- **Gold breaks have no separate rate card.** Confirmed; gold is a constraint
  scope, not a price table.
- **Hour-of-day rate-card layer absent.** Confirmed.
- **Owner and Jumbo:** no seam. Confirmed.
- **Block booking / approval state machine:** absent. Confirmed, and now
  reinforced by finding 8 — the status vocabulary is not merely unvalidated, it
  is a closed two-value tuple with no room for either state.

**Superseded — the register is stale and should be marked so:**
- **Gap 2, positions.** `kairos/optimize/positions.py`,
  `kairos_api/break_api_pod_spots.py:31` and
  `config/optimization_weights.yaml:68` shipped 1-5 plus L, L as its own token,
  and both counting methods — in the same commit as the gap analysis. What
  remains is findings 1 and 2 above, which are different problems.
- **Gap 3, creatives.** `kairos_api/campaigns_assets_constraints.py` now models
  both the validity window (`campaigns_assets.py:225`) and the Top-and-Tail
  pair, the latter as a `pair_separation` rule row
  (`kairos/optimize/_frequency_rules.py:61,98-99,119-133`) enforced by
  `kairos/optimize/_pair_placement.py` and surfaced through
  `GET /campaigns/{id}/assets` (`campaigns_api_detail.py:51-66`). Two residuals:
  `data/frequency_rules.csv` holds **one row and it is not a pair**, and there is
  no authoring surface, so the enforcement has nothing to enforce. Cross-channel
  creative identity is still absent.
- **Gap 4, make good.** No longer a naming collision. `kairos_api/makegood_store.py`
  plus `data/make_goods.csv:1` is a real decision ledger with states and actors.
  But it is **campaign-keyed only**: the header carries `campaign_id`,
  `campaign_name`, `advertiser` and `channel` and **no `agency_id`**, no accrual
  balance, and no utilisation of credit on a different campaign. The trade's
  "three levels at once" is one level with two labels on it, and the agency
  level — the one the transcript says accrues ten percent of spend — is missing.

---

## Ranked, by what it blocks

| # | Finding | Blocks | Previous audit |
|---|---|---|---|
| 1 | Preferred-position % unreachable: no operator can set the preferred set | The number channel and agency audit each other with | Named as absent; then built and left inert |
| 2 | Two answers to "which positions are preferred"; pod board badges every spot | Reading position quality off the board | Missed |
| 3 | Rating currency is all-viewers, not Jewish households; overnight+1 nowhere | Every ILS figure's unit; stating a goal on the real currency | Missed |
| 4 | Delivered rating is a second feed As Run cannot supply | Gap 5 as specified will not produce a delivered TRP | Wrong altitude |
| 5 | `snapshot_before_write("campaigns")` is a silent no-op | Seeing what changed in a campaign amended daily | Missed |
| 6 | No order entity; no time, programme or length on it | Putting a signed insertion order anywhere | Partly, at the portal altitude |
| 7 | Direct clients invisible; return inverted (0% vs agency 4%) | Pricing a direct deal; one of six run priorities | Named the priority, missed the entity and the money |
| 8 | No cancellation, no billing-started boundary, two-state campaign | Recording the commonest post-booking event | Missed |
| 9 | No break can be held back | Modelling withheld inventory | Missed |
| 10 | `advertiser_rules.py:26` says demand weights are off; they are always on | A reader deciding whether activation moves money | Missed |
| 11 | `price_model` and `priority` stored, validated, read by nothing | Flat-price and preemptible campaigns price identically | Missed |
| 12 | Guardrails unconditional; compliance and engine read different files | The per-run regulatory switch; attestation matching behaviour | Named the switch, missed the split |
| 13 | Assistant rule 24 omits campaign/advertiser/agency (unverified) | Possibly nothing | Missed |

## What I could not determine cheaply

- Whether the `TVR` column in `data/Programmes.csv` and `data/Spots.csv` is in
  fact an all-viewers figure. `kairos_api/campaigns_delivery.py:107` asserts it
  is; I found no provenance record, no source-file header and no ingestion check
  that establishes it. The assertion may be right — it is not evidenced in the
  repo, and finding 3 stands either way, because the universe is unrecorded.
- Whether the assistant actually refuses the noun "campaign" under rule 24. That
  needs a live turn against the running assistant, which I did not drive.
