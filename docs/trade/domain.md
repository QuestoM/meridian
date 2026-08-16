# Israeli television trade agreements — the working domain document

Written 2026-08-16 for the trade-agreement engine. Everything the engine builds
is judged against this document. Its authority chain, strictest first:

1. `docs/media-domain-from-the-trade.md` — the owner and the media professional.
   Where anything here disagrees with it, THIS file is wrong.
2. The Israeli primary-source research: `docs/research/israeli-rating-currency.md`,
   `docs/research/israeli-goal-selling.md`, `docs/research/hebrew-trade-vocabulary.md`,
   `docs/campaign-rate-card-research.md`, `docs/quarter-hour-billing.md`.
3. Standard television-trading practice that has NOT been attested in Israel.
   Every claim of this rank is marked **[standard practice — owner to confirm]**
   and ships in the taxonomy as a representable term, never as an assumed fact
   about this operator. The ruling in `docs/audits/research-scope-ruling.md`
   governs: foreign structure becomes behaviour only with Israeli evidence or an
   owner's word.

The companion catalogue of individual commercial terms is
`docs/trade/term-taxonomy.md`. This file explains the world those terms live in.

---

## 1. Who contracts with whom

Three parties, two contracting distances:

- **The channel** (here: the operator, רשת 13 in the live configuration) sells
  airtime. Its sales function is the counterparty on every agreement.
- **The media agency** (סוכנות מדיה / חברת מדיה) buys on behalf of many
  advertisers. The big agencies concentrate most of the market's money, and the
  channel's most important annual negotiations are with them. The repository
  already carries nine real agencies and 41 observed advertiser links
  (`data/agencies.csv`, `data/agency_advertisers.csv`).
- **The advertiser** (מפרסם) is the brand whose money it ultimately is. Some
  buy **direct** (לקוח ישיר) with no agency in the middle — the trade transcript
  lists "direct clients" as a distinct placement priority, and direct buying
  carries its own economics (a ~20% return customarily taken as added media
  rather than cash).

An agreement can therefore bind at three levels, and real ones do:

- **Agency framework** (הסכם מסגרת עם סוכנות): the agency commits volume across
  ALL its clients; terms cascade to every advertiser it brings, unless a
  client-specific agreement overrides.
- **Advertiser agreement** (הסכם מפרסם): one client's terms, whether direct or
  through an agency; overrides the agency framework where they conflict.
- **Campaign schedule / order terms**: the short-lived commercial terms of one
  campaign, arriving as orders do — one to two days ahead, revised daily.

This three-level shape is not incidental; the make-good ledger the trade
describes runs at exactly these three levels (campaign, advertiser, agency), and
the rule model must resolve precedence across them (§8).

**Success deals** (עסקאות הצלחה) are revenue-share arrangements named in the
trade transcript as their own placement-priority class. Their terms live in
agreements too: a share percentage, a measurement basis, a settlement cycle.
The transcript names them; their internal mechanics are
**[trade transcript names the class; internals owner to confirm]**.

## 2. The instruments

- **The annual framework agreement** (הסכם מסגרת שנתי / חבילה שנתית): the
  channel and an agency or large advertiser fix the year's commercial
  architecture — committed budget, CPP levels per daypart strip (רצועה),
  discount ladder, guarantee terms, positioning entitlements, payment terms.
  The Israeli market attests the noun חבילה שנתית for a negotiated annual
  volume-discounted CPP commitment (`campaign-rate-card-research.md` §10).
- **The campaign order** (הזמנה): under the framework, per campaign. An order
  is a negotiating position, not a plan — it arrives late, over-asks for prime
  deliberately, and is revised daily. The agreement is the stable object; the
  order is the volatile one. The engine extracts agreements, not orders.
- **Block booking**: capacity reserved ahead, often without naming the client
  ("ten minutes in prime every day through August"). Named directly by the
  owner as an Israeli practice. An agreement may grant block-booking rights or
  price them.
- **Amendments and appendices** (תיקונים ונספחים): signed mid-term changes and
  attached schedules (rate tables, brand lists, daypart definitions). They
  layer on the base agreement with their own effective dates and must not
  erase the history of what they replaced.
- **Side letters**: short signed clarifications. Same layering behaviour as an
  amendment. **[standard practice — owner to confirm the Israeli habit]**.

## 3. The commercial spine: what a shekel buys

The traded currency and settlement mechanics, all Israeli-attested:

- **CPP per daypart strip** (עלות לנקודת רייטינג לרצועה): the price of one
  rating point for a 30-second spot in a named רצועה. The trading basis is
  **Jewish households** (בתי אב יהודיים), **quarter-hour rating**
  (רייטינג רבעי שעה), **overnight +1** (צפייה נדחית של 24 שעות מרגע השידור).
  One published sentence carries all three plus the length factor
  (`israeli-goal-selling.md` §1).
- **Settlement per spot**: actual payment = CPP × the quarter-hour rating of
  the quarter-hour containing the spot × the length factor. A spot at 8:03
  bills on the 8:00–8:14 average. Already shipped as
  `docs/quarter-hour-billing.md` (owner-gated activation).
- **The length factor** (מקדם אורך): pro-rata against a 30-second base —
  the market prices a 30" unit and scales, it does not quote per-second.
  Shorter spots are commonly priced ABOVE linear pro-rata
  (15" at ~60–75% of 30" — `campaign-rate-card-research.md` §6), so an
  agreement may carry its own length-coefficient table.
- **Planned on TRP, paid on household CPP**: campaigns commit to TRP against a
  named target audience (e.g. נשים 25-54) while money settles on the household
  base; the ratio is the trade's named efficiency measure **יחס המרה**
  (TRP ÷ GRP). An agreement can fix the audience, the TRP quantity, and
  sometimes a CPP against the TARGET audience — the model must never conflate
  the two bases.
- **The final rating is only known the day after** and can move by up to three
  points. Every delivery figure computed on the night is provisional and the
  agreement's measurement clauses decide WHICH vintage settles (§7).

## 4. What the buyer commits: budgets, shares, mixes

The commitment classes that anchor an annual agreement:

- **Budget commitment** (התחייבות תקציב): a money floor for the period —
  quarterly or annual, sometimes split per brand. The discount ladder hangs off
  it (§5). Falling short of a committed budget is itself an outcome-dependent
  event: ladders re-rate, rebates claw back.
- **Share commitment** (נתח / share of investment): the agency or advertiser
  commits a PERCENTAGE of its television spend to this channel, not just an
  absolute sum. Verification needs a denominator the channel cannot see alone
  (the client's total market spend), so agreements pair the share term with a
  declaration/audit mechanism. **[standard practice worldwide; the Israeli
  variant is closed-door — owner to confirm the usual denominator source]**.
- **Mix obligations**: how the money must spread — per daypart strip
  (רצועות), per season, weekday/weekend, programme genres, spot lengths.
  These bound both sides: they protect the channel's off-prime inventory
  (the classic prime-share cap: no more than X% of a budget in prime) and
  the buyer's audience goals.
- **Flighting obligations**: weeks on-air, continuity floors ("no dark weeks
  during Q4"), campaign-count minimums. Their planning consequence is a
  calendar constraint, not a price.

## 5. What the price does: ladders, commissions, surcharges

- **The rate card is nine layers deep** (trade transcript, in order): base
  price per hour; day-of-week; hour; programme category; specific programme;
  specific date; break position; seasonal/periodic premium (e.g. the three
  weeks before Passover); all adjustable per agency, per advertiser, per
  campaign. Gold breaks carry a separate rate card. The engine's pricing
  hierarchy already models most layers with activation flags
  (`docs/pricing-hierarchy-design.md`); the agreement supplies the per-client
  values.
- **Discount ladder** (מדרגות הנחה): discount % as a step function of
  committed or actual spend. Two bases exist and agreements must say which:
  off rate-card (הנחה מהמחירון) or off net-after-commission. Ladders can be
  **retroactive** (hitting a tier re-rates the whole period) or **marginal**
  (each tier prices its own tranche) — a distinction worth millions that a
  shallow model erases. **[ladder mechanics are standard practice; Israeli
  publication is nil — owner confirms per agreement]**.
- **Agency commission** (עמלת סוכנות): the agency's cut, historically 15%
  worldwide; in Israel the observed agency economics in this repository are
  rebate rows on the agency layer (net_revenue beside gross). An agreement
  states the %, its base, and whether it is a deduction on invoice or a
  periodic rebate (החזר).
- **Cash/payment terms** (תנאי תשלום): current+30/60/90 (שוטף+), early-payment
  discount (הנחת מזומן), linkage/interest on late payment. These price money,
  not airtime, and belong to settlement rather than planning.
- **Seasonal surcharges/discounts**: high-demand windows (September–December,
  pre-Passover) publish up to ±35% seasonal movement
  (`campaign-rate-card-research.md` §8). In an agreement these appear either
  as month coefficients or as blackout-from-discount windows.
- **Position surcharges**: first/second/third/last premiums, the gold-break
  surcharge (up to 25% published), Top-and-Tail as a combined-length product
  (תשדירי T&T priced on combined length — `israeli-goal-selling.md` §8).
- **Success-deal terms**: revenue share in place of (or blended with) rate
  card. Measurement basis and audit rights follow it.

## 6. What placement must honour

The constraint classes an agreement imposes on the schedule itself:

- **Preferred positions**: 1–5 plus **L** (Last — its own thing, never the
  fifth ordinal; one campaign can hold Top AND Tail of one break). WHICH
  positions count as preferred is per client per agreement. Any percentage
  commitment must state its counting method, because two live methods disagree:
  the **agency method** (preferred positions obtained ÷ break appearances,
  counting a double appearance twice) and the **channel method** (÷ total
  broadcasts). Both already have a seam in `kairos_api/preferred_rate.py`.
- **Competitive separation** (הפרדה תחרותית): rival advertisers not in the
  same break (or separated by N spots/minutes). COMMERCIAL, not regulatory —
  the 1992 rules contain no such clause; presenting it as law is a fail.
- **Category exclusivity** (בלעדיות קטגוריה): stronger than separation — sole
  advertiser of a category in a programme/daypart/period, priced accordingly.
- **Content adjacency** (סמיכות תוכן): keep-outs from named content classes
  (news adjacency on request, exclusion from disaster coverage, children's
  programming rules), and positive adjacency purchases (the news-adjacent
  premium breaks the market names as the strong ones).
- **Programme/daypart restrictions**: allow-lists or forbid-lists of
  programmes, genres, strips; the אתנחתה legality of mid-programme breaks is
  regulatory background the channel owns regardless of agreement.
- **Creative constraints**: validity windows per creative, Top-and-Tail
  same-break separation (exactly one or two spots between), up to ~20
  creatives per campaign, house numbers per channel. Already largely modelled
  (`data/campaign_assets.csv`, Top-and-Tail rules in the frequency engine).
- **Spot-length rules**: allowed lengths, length mix, the regulatory 90-second
  single-spot cap as an outer bound.

## 7. Delivery, measurement, and the day after

- **As-Run is the only truth about what aired.** Billing and delivery are
  computed from the broadcast system's after-the-fact record, never from the
  plan. The control room moves things by telephone and none of it passes
  through the order system.
- **Measurement windows**: a delivery guarantee names its window (campaign,
  month, quarter, year) and its grain (per spot QH settlement rolling up to
  period TRP totals).
- **Rating vintage**: overnight+1 is the trading vintage; the final figure
  arrives the next day and moves. An agreement's measurement clause fixes
  vintage, audience, and source; a stored rating without its vintage cannot be
  reconciled (the settlement schema shipped 2026-08-10 already refuses
  vintage-less segments).
- **Reconciliation cycle**: weekly reconciliation of booked-vs-delivered is
  the market's stated practice (`campaign-rate-card-research.md` §1), with
  period-end true-up under the agreement's shortfall clauses.

## 8. When the outcome misses: the compensation machinery

The hard class the client named explicitly. Israeli reality, per the trade
transcript, is LARGER than the tidy foreign model:

- **Make-good** (מייק גוד): compensation in airtime for a spot that did not
  air, aired wrong, or for audience shortfall. Managed as an
  **accrual-and-utilisation ledger at three levels at once** — campaign,
  advertiser, agency. An agency accrues (e.g. 10% of spend) and may SPEND the
  credit on a DIFFERENT campaign later. Foreign systems bind each cure to its
  own deal; importing that shape would ship the smaller object.
- **Bonus inventory** (מדיה נוספת / בונוסים): added airtime instead of cash —
  the default cure in this market. Direct buying customarily carries ~20%
  return taken as media.
- **Shortfall against a TRP guarantee**: the channel adds spots at no charge
  until the points arrive, or credits. The agreement fixes: threshold
  (tolerance band), evaluation moment (mid-flight checkpoints vs period end),
  cure window, cure inventory quality (like-for-like daypart or the published
  1:3 prime↔non-prime anchor from the 1992 rules' restitution clause), and
  what happens if the cure itself under-delivers.
- **Over-delivery**: either banked to the buyer's favour, charged (rare),
  or absorbed — the agreement says. **[treatment varies; owner confirms]**.
- **Under-spend by the buyer**: the mirror image — ladder re-rating, committed
  budget true-up, lost exclusivity. A complete model tracks obligations in
  BOTH directions.
- **Cancellation rules are demand-dependent**: a day out, billing has started
  and cancellation is refused; in high-demand periods the channel may welcome
  it to resell. Cancellation windows and fees are agreement terms.

The engineering consequence, and the requirement the client stated: these are
**living obligations**, each with a measurement basis, a window, a threshold,
an evaluation moment, a consequence, and a settlement path — tracked
continuously, projected forward, and alarmed EARLY, not reported after the
period closes.

## 9. Time, versions, and precedence inside one agreement

- **Effective windows**: every term carries validity dates; the agreement has
  its own; amendments open and close term versions mid-flight. Two versions of
  a term never both bind on one day.
- **Precedence**: contracts contradict themselves. Real resolution rules, in
  descending strength: explicit precedence clauses ("this appendix prevails");
  later-dated over earlier; specific over general (a programme-level term over
  a daypart term; an advertiser agreement over its agency framework); and
  where genuine ambiguity remains, the system must refuse to guess and ask.
  Determinism about WHICH term wins, and why, is a product feature the
  reviewer sees.
- **Language**: agreements are Hebrew, right-to-left, salted with English
  brand names, Latin numerals, tables, and scans. The extraction layer's whole
  design answers this (`docs/trade/extraction-design.md`).

## 10. What already exists in this product (so the engine extends, never forks)

Measured on the tree at 2026-08-16; the deep map lives with the design docs.

| Domain object | Where it lives today |
|---|---|
| Advertiser/agency conditions (premium/require/forbid/pressure) | `data/advertiser_conditions.csv`, `data/agency_conditions.csv`, engine `kairos/optimize/advertiser_rules.py` |
| Planning constraints (AND/OR predicates) | `kairos/optimize/predicate.py`, frozen contract `docs/constraint-predicate-contract.md` |
| Pricing hierarchy with activation flags | `kairos/optimize/pricing.py`, `pricing_from_settings`, `kairos_api/pricing_api.py` |
| Campaigns, flights, goals, delivery, pacing | `kairos_api/campaigns_*.py`, `data/campaigns.csv` |
| Make-good projection (risk flag, not yet a ledger) | `kairos/optimize/pacing.py::project_make_goods` |
| Preferred-position % with both counting methods | `kairos_api/preferred_rate.py` |
| Quarter-hour settlement restatement (owner-gated) | `kairos/export/qh_billing.py` |
| Rating-currency settlement gates (vintage/basis refusals) | shipped 2026-08-10, activation off |
| Expected-TVR audience model with honesty gates | `kairos/model/audience_*.py`, `models/audience_model.json` |
| Versioned stores + approval-creates-a-version | plan version store, conditions versioning, History domain |
| Human-approval pattern for AI proposals | Mabat propose→approve appliers |

The trade engine's job is to make a SIGNED DOCUMENT the source that fills,
versions, and explains these existing mechanisms — and to add the missing
objects (the agreement entity, the obligation tracker, the accrual ledger,
clause-level provenance) around them.

## 11. Honesty rules this document imposes on the engine

1. **The taxonomy marks provenance per term**: Israeli-attested / trade-doc /
   standard-practice-unconfirmed. A term of the third rank is representable and
   extractable but its UI copy never asserts it as "how the Israeli market
   works".
2. **No number from research is a default.** Published percentages (25% gold,
   5–20% position, 15% commission) never seed a rule; values come from the
   document under review or from the operator's own configuration.
3. **A rating without vintage, an amount without basis, a share without
   denominator — refused, not defaulted.**
4. **Nothing extracted binds without human approval** (the mission's hard
   rule), and nothing approved binds outside its effective window.
5. **A clause the model cannot map is surfaced as such.** Being honest about
   an unmapped term is a success; silence is the only failure.
