# The trade-term taxonomy

The exhaustive catalogue of commercial terms a television trade agreement in
this market can carry. Written 2026-08-16 against `docs/trade/domain.md`; the
machine-readable twin is `kairos/trade/taxonomy.py` and a test pins the two to
the same term list. **This catalogue is the contract for completeness**: the
extraction pipeline classifies every clause against it, the rule model must
represent every term whose behaviour class demands representation, and the
review screen shows, per document, what landed where.

## How to read an entry

- **Behaviour** — what the term does inside the product. One or more of:
  - `prices` — changes what airtime costs (rate, discount, surcharge, commission).
  - `constrains-hard` — placement the schedule must not violate.
  - `constrains-soft` — placement the schedule should avoid or prefer; warns, steers.
  - `obliges` — creates a living obligation tracked over time (commitment,
    guarantee, cure), with basis/window/threshold/evaluation/consequence/settlement.
  - `settles` — governs how money is reconciled after broadcast (measurement
    basis, vintage, true-up), without steering placement.
  - `process` — legal/administrative; stored, displayed, deadline-tracked; no
    planning or settlement arithmetic.
  - `meta` — governs other terms (precedence, definitions, layering, effective
    windows).
- **Data** — the parameters a complete instance carries. The strict extraction
  schema per term lives in `kairos/trade/taxonomy.py`.
- **Interacts** — the terms it composes with, overrides, or contradicts, and
  the resolution rule.
- **Measured** — how standing is computed over time, where relevant.
- **Status** — honesty about today's product:
  - `BINDS` — representable AND changes behaviour through existing machinery.
  - `REPRESENTABLE` — the model can hold it faithfully; binding path being wired
    in this campaign.
  - `TRACKED` — stored, measured and surfaced; consequences stay human-driven.
  - `RECORDED` — stored and displayed with deadlines/alerts only.
  - Anything outside these four is flagged loudly at review as *understood but
    not yet supported* — silence is never an option.
- **Rank** — provenance: `IL` = Israeli-attested (primary source);
  `TRADE` = the owner/media-professional transcript;
  `STD` = standard trading practice not yet attested for this operator — the
  UI never asserts STD terms as local market fact.

The two counting-method and basis warnings recur enough to state once: **every
percentage names its counting method, every money amount names its basis
(gross / net-of-commission / rate-card), every rating names its audience and
vintage.** An instance missing one of these is extracted as INCOMPLETE, not
defaulted.

---

## Family A — Identity, scope and instrument (meta)

### A1 `agreement-parties` — הצדדים להסכם
The channel and the counterparty: media agency (framework), advertiser
(direct or via agency), or both named. Binds the agreement to the existing
agency/advertiser entities; a name that matches no known entity goes to review
as a proposed new entity, never silently created.
**Data**: counterparty type, entity refs, signatories, direct-client flag.
**Interacts**: level decides precedence (A5). **Status**: REPRESENTABLE. **Rank**: TRADE.

### A2 `brand-scope` — היקף מותגים
The advertiser brands/products the agreement covers; a framework may carve
brands out (a client with a separate direct deal). Placement and money terms
apply only inside the scope.
**Data**: included brands, excluded brands, per-brand budget splits (D1 link).
**Interacts**: narrows every other term's applicability. **Status**: REPRESENTABLE. **Rank**: STD.

### A3 `channel-scope` — היקף ערוצים
Which channels/platforms the agreement covers. This operator is one channel;
agreements still name it (and sometimes digital assets, which this product
does not schedule — such clauses classify here and are RECORDED with a reason).
**Data**: channel list; non-TV assets flagged out-of-product.
**Status**: REPRESENTABLE. **Rank**: STD.

### A4 `effective-window` — תקופת תוקף
Start and end dates of the agreement, auto-renewal (התחדשות אוטומטית), notice
period for non-renewal. Every term inherits the agreement window unless it
carries its own (meta rule; see H-family for per-term windows).
**Data**: start, end, renewal terms, notice days.
**Interacts**: bounds everything; expiry raises a renewal alert.
**Measured**: days-to-expiry surfaced. **Status**: REPRESENTABLE. **Rank**: STD.

### A5 `agreement-level` — רמת ההסכם
Agency framework / advertiser agreement / campaign schedule. Sets default
precedence: campaign-specific over advertiser over agency framework, unless an
explicit precedence clause (A6) says otherwise. This mirrors the market's
three-level make-good reality.
**Data**: level enum, parent-agreement ref where layered.
**Status**: REPRESENTABLE. **Rank**: TRADE.

### A6 `precedence-clause` — סעיף עדיפות
An explicit statement of which document or term wins on conflict ("the appendix
prevails over the body", "this agreement supersedes the 2025 framework").
Compiles to edges in the deterministic precedence graph; the resolver's
explanation quotes this clause when it decides a conflict.
**Data**: winner ref, loser ref, scope of precedence.
**Interacts**: outranks the default A5 ordering. **Status**: REPRESENTABLE. **Rank**: STD.

### A7 `definitions` — הגדרות
The agreement's own defined terms ("Prime" means 20:00–24:00; "Quarter" means
the calendar quarter). Definitions re-scope other terms' vocabulary — a
document defining prime as 21:00–23:00 changes every prime-scoped term in it.
Extraction resolves defined terms before parameterising clauses that use them.
**Data**: term → definition map, incl. daypart boundary overrides.
**Interacts**: feeds parameter resolution of every family. **Status**: REPRESENTABLE. **Rank**: STD.

### A8 `amendment-layer` — תיקון / נספח
A signed amendment or appendix layering onto the base: it opens new term
versions and closes old ones at its effective date, never erasing history.
Extraction must attach an amendment to its base agreement and mark which
clauses it modifies (cross-reference resolution).
**Data**: base ref, effective date, modified-term refs, added terms.
**Interacts**: A6/A4; versioning machinery. **Status**: REPRESENTABLE. **Rank**: TRADE (appendices/amendments named as routine).

---

## Family B — Money basis (prices / settles)

### B1 `cpp-daypart-table` — טבלת CPP לפי רצועה
The heart of the deal: cost per rating point for a 30-second spot, per daypart
strip (רצועה), against a named audience basis. The Israeli basis is Jewish
households, quarter-hour rating, overnight+1. A table may carry per-strip
audiences (rare) — each row states its own.
**Data**: rows of {daypart, CPP, audience basis, base length 30"}.
**Interacts**: discount ladder (C1) applies to it per its stated basis; length
factor (B3) scales it; definitions (A7) may re-bound the strips.
**Measured**: effective CPP delivered vs table (E2).
**Status**: REPRESENTABLE (feeds pricing seams; effective-CPP tracking with the
obligation layer). **Rank**: IL (basis attested; per-strip CPP structure IL).

### B2 `target-cpp` — CPP לקהל יעד
A CPP quoted against the TARGET audience (e.g. ₪X per women-25-54 point)
instead of households. Planned-on-TRP/paid-on-households is the market norm
(יחס המרה names the gap), so a target-CPP clause changes the settlement
basis and must never be conflated with B1.
**Data**: audience, CPP, window, which spots it covers.
**Interacts**: mutually exclusive with B1 per scope; E1 guarantees measure
against the same audience. **Status**: REPRESENTABLE. **Rank**: IL (the two bases and the
ratio are attested; target-CPP deals STD).

### B3 `length-factor-table` — מקדמי אורך
Pricing per spot length relative to the 30" base. Pro-rata is the published
default; short lengths commonly price ABOVE pro-rata (15" at 60–75% of 30").
An agreement may fix its own table; absence means the operator's configured
factors.
**Data**: length → factor rows; rounding rule.
**Interacts**: B1/B2, settlement (H1). **Status**: REPRESENTABLE. **Rank**: IL
(מקדם האורך attested in the settlement sentence).

### B4 `ratecard-index` — הצמדה למחירון
Pricing expressed as a percentage of the operator's rate card ("client pays
82% of rate card"), instead of absolute CPPs. Needs the rate-card version
pinned (which card, which date) — an index against a floating card is an
incomplete instance.
**Data**: index %, rate-card version ref, scope.
**Interacts**: C1 (a ladder may deepen the index), rate-card layers.
**Status**: REPRESENTABLE. **Rank**: STD.

### B5 `fixed-spot-pricing` — מחיר קבוע לספוט
Flat prices for named programmes/slots (the smaller-advertiser pattern, and
the pattern for special properties: finals, event television). No rating risk:
airtime, not audience, is what's bought.
**Data**: rows of {programme/slot scope, length, price}.
**Interacts**: overrides B1 for its scope (specific-over-general); E1
guarantees do not attach to flat buys unless stated.
**Status**: REPRESENTABLE. **Rank**: IL (flat-spot market attested).

### B6 `sponsorship-terms` — תנאי חסות
Sponsorship notices (הודעות חסות): 6-second billboards at fixed per-airing
prices, sold per programme/season, regulatory text obligations attached.
Distinct inventory from spots — priced flat, not by position.
**Data**: programme, airings count/period, price per airing, notice length.
**Interacts**: separate from break placement terms (F-family does not apply);
regulatory caps live with the channel, not the agreement.
**Status**: REPRESENTABLE. **Rank**: IL.

### B7 `gold-break-rates` — מחירון ברייק זהב
The separate gold-break rate card the trade names: per-break or surcharge
pricing for the 1–3-spot premium break. May appear as a % surcharge (up to 25%
published) or absolute prices.
**Data**: surcharge % or price rows; allocation link (E4).
**Interacts**: stacks with position premiums per the market's two-multiplier
structure. **Status**: REPRESENTABLE. **Rank**: IL + TRADE.

### B8 `payment-indexation` — הצמדה וריבית
Linkage of prices to an index (CPI) across a multi-year term, VAT treatment,
late-payment interest. Settlement arithmetic, not planning.
**Data**: index, linkage formula, VAT basis, interest rate.
**Status**: RECORDED (finance-side; surfaced on the agreement, deadlines
tracked). **Rank**: STD.

---

## Family C — Discounts, commissions, incentives (prices)

### C1 `volume-discount-ladder` — מדרגות הנחת היקף
Discount % as a step function of spend. The three parameters that move real
money and are silently erased by shallow models:
**basis** (rate-card gross / net-after-commission; committed vs actual spend),
**mechanics** (RETROACTIVE — crossing a tier re-rates the whole period; or
MARGINAL — each tranche keeps its tier), and **period** (annual/quarterly,
which calendar). A ladder on committed budget interacts with D1: under-spend
triggers re-rating (E7).
**Data**: tier rows {threshold, discount %}, basis, mechanics, period, scope.
**Interacts**: C4 commission (order of application must be stated — discount
off gross then commission, or commission then discount), B4 index, D1.
**Measured**: current tier, distance to next tier, projected year-end tier.
**Status**: REPRESENTABLE. **Rank**: STD mechanics; חבילה שנתית volume-CPP
commitment IL-attested.

### C2 `share-bonus` — תמריץ נתח
Extra discount or bonus media awarded for hitting a share-of-investment
commitment (D2). Distinct from the ladder: the trigger is a percentage of the
client's total TV spend, not an absolute amount.
**Data**: share threshold %, award (discount % or bonus media %), denominator
source, period. **Interacts**: D2 supplies the measurement; E6 if paid in media.
**Measured**: with D2's declared denominator. **Status**: TRACKED (denominator
is external; standing computed when declared spend supplied). **Rank**: STD.

### C3 `seasonal-coefficients` — מקדמי עונתיות
Month/period price coefficients (the ±35% seasonal movement the market
publishes; pre-Passover rises). In an agreement: either explicit coefficients
per period or discount-blackout windows ("ladder does not apply in December").
**Data**: period → coefficient rows, or blackout windows per term ref.
**Interacts**: composes onto B1/B4; the events pricing layer already gives
per-date multipliers operator-side. **Status**: REPRESENTABLE. **Rank**: IL
(seasonality published; exact shape per agreement).

### C4 `agency-commission` — עמלת סוכנות
The agency's percentage, its base, and its form — deduction on invoice or
periodic rebate (החזר). The product already carries agency rebates as
reporting-only net_revenue beside unchanged gross; an agreement's commission
clause is the SOURCE for that number, per agency per period.
**Data**: %, base (gross/net-of-discount), form, payment cycle.
**Interacts**: C1 order-of-application; H1 settlement.
**Status**: BINDS (agency rebate layer exists; agreement becomes its source).
**Rank**: TRADE + repository-observed agencies.

### C5 `cash-discount` — הנחת מזומן
Early-payment discount tied to payment terms (G1): pay within N days, take X%.
Settlement-side money.
**Data**: %, qualifying terms. **Status**: RECORDED. **Rank**: STD.

### C6 `success-deal` — עסקת הצלחה
Revenue-share arrangement: the channel participates in the advertiser's
outcome (sales uplift, response) instead of, or blended with, rate-card
pricing. Named by the trade transcript as its own placement-priority class —
so a success-deal flag also STEERS placement priority in runs.
**Data**: share %, measurement basis, settlement cycle, blend terms, priority
class flag. **Interacts**: run-priority parameters; H1.
**Status**: TRACKED (flag + terms stored, priority steer via pressure lever;
outcome measurement is external). **Rank**: TRADE.

### C7 `added-value-media` — מדיה נוספת קבועה
A fixed bonus-media grant independent of any shortfall: "12% added value on
all spend", the direct-buy ~20% return taken as media. Distinct from cure
bonuses (E6): this one is EARNED by buying, not owed for missing.
**Data**: %, basis, delivery window, inventory quality constraints.
**Interacts**: accrues into the same make-good/bonus ledger currency (E5) but
with its own reason code. **Measured**: accrued vs utilised.
**Status**: REPRESENTABLE. **Rank**: TRADE (the ~20% direct return).

### C8 `new-business-incentive` — תמריץ לקוח חדש
First-year discount or bonus for a new advertiser / returning-after-absence
clause. Time-bounded by nature.
**Data**: award, qualification rule, window. **Status**: REPRESENTABLE. **Rank**: STD.

### C9 `package-bundle` — חבילה משולבת
Bundled buys (TV + the channel's digital inventory unlocks; named-programme
sponsorship seasons). Out-of-product assets are RECORDED with reason; the TV
side compiles normally.
**Data**: components, bundle price/discount, allocation between components.
**Status**: REPRESENTABLE (TV components) / RECORDED (non-TV). **Rank**: IL
(combined TV-digital deals named in market research).

---

## Family D — Buyer commitments (obliges)

### D1 `budget-commitment` — התחייבות תקציב
The buyer commits money for a period, possibly split per brand/quarter. The
anchor for ladders (C1) and the mirror of the channel's guarantees. A living
obligation: standing = actual (As-Run-based) billed spend vs committed curve.
**Data**: amount, period, currency basis, per-brand/per-quarter splits,
tolerance, true-up rule ref (E7).
**Interacts**: C1 basis; E7 under-spend consequences; pacing surfaces.
**Measured**: billed-to-date vs time-proportional or contract-curve pacing;
projection to period end.
**Status**: REPRESENTABLE. **Rank**: TRADE + IL (annual commitment attested).

### D2 `share-commitment` — התחייבות נתח
A committed % of the buyer's total TV spend. The denominator (total market
spend) is invisible to the channel, so the term carries its declaration
mechanism: periodic declared spend, audit rights (G3).
**Data**: share %, period, denominator source (declaration cadence), audit ref.
**Measured**: channel-billed spend ÷ declared total; UNKNOWN when declaration
missing — never guessed. **Status**: TRACKED. **Rank**: STD (mechanism), TRADE
(share deals implied by market structure — owner to confirm phrasing).

### D3 `daypart-mix` — תמהיל רצועות
Bounds on how spend/points spread across strips: prime caps ("≤35% of budget
in prime"), off-prime floors. Protects channel inventory and buyer goals both.
**Data**: rows of {daypart scope, min %, max %, basis (money/points/spots)}.
**Interacts**: planning steer + delivery measurement; B1 strips via A7.
**Measured**: mix-to-date vs bounds, projection.
**Status**: REPRESENTABLE. **Rank**: STD (prime-share mechanics), IL (prime
1:3 value anchor exists in law).

### D4 `flighting-obligation` — התחייבות רציפות
Calendar-shape commitments: minimum weeks on air, no-dark-weeks windows,
launch-date obligations. Constrains the plan's calendar rather than a day's
break.
**Data**: window rules, min/max consecutive dark days, campaign-count minimums.
**Measured**: calendar coverage vs rule. **Status**: TRACKED. **Rank**: STD.

### D5 `length-mix` — תמהיל אורכים
Commitment on spot-length distribution (e.g. ≥70% of units at 30"), because
short spots at above-pro-rata factors change yield.
**Data**: length buckets, min/max %, basis. **Measured**: mix-to-date.
**Status**: TRACKED. **Rank**: STD.

### D6 `cancellation-terms` — תנאי ביטול
Notice windows and fees for cancelling booked activity; demand-dependent in
practice (a day out is refused; high-demand periods may welcome resale). The
agreement fixes windows/fees; the demand-dependence is channel discretion.
**Data**: notice days per window, fee %, force-majeure carve-outs (G5).
**Interacts**: G5 war clauses; planning lock windows.
**Status**: RECORDED (deadlines tracked; enforcement is a human act).
**Rank**: TRADE.

---

## Family E — Channel guarantees and cures (obliges / settles)

### E1 `trp-delivery-guarantee` — התחייבות נקודות רייטינג
The channel guarantees TRP delivery: N points against a named audience over a
window, per campaign or per period, sometimes per daypart. THE living
obligation: basis (audience + vintage), window, threshold/tolerance,
evaluation moments (checkpoints + period end), consequence (cure path E5/E6),
settlement. GRP variant when the audience is households.
**Data**: points, audience, vintage, window, scope (campaign/period/daypart),
tolerance band, checkpoint cadence, cure ref.
**Interacts**: B2 if target-CPP; forecast layer supplies projection; E5/E6
consume breaches; the Channel-24-style budget-for-GRP deal is this term with
budget attached.
**Measured**: delivered points (As-Run × settled ratings) vs committed curve;
projected end-state from the rating forecast; alarm BEFORE the window closes.
**Status**: REPRESENTABLE (delivery pacing exists per campaign goal; the
guarantee object with tolerance/cure is this campaign's build). **Rank**: IL
(GRP commitment attested for a named channel) + TRADE.

### E2 `effective-cpp-cap` — תקרת CPP אפקטיבי
Guarantee that the period's effective CPP (billed money ÷ delivered points)
does not exceed X — equivalently a delivery-or-discount promise at period
level. Often the annual deal's true economic core.
**Data**: cap value, audience basis, window, computation basis (which spend
counts), true-up form (credit/bonus points).
**Interacts**: E1 (points side), C1 (money side), H1 settlement.
**Measured**: rolling effective CPP + projection vs cap.
**Status**: REPRESENTABLE. **Rank**: STD (mechanism; CPP-per-strip trading IL).

### E3 `preferred-position-guarantee` — התחייבות מיקומים מועדפים
A committed % of appearances in preferred positions, where the agreement
defines WHICH positions are preferred (from 1–5 + L) and WHICH counting
method audits it — agency method (÷ break appearances, double-appearance
counts twice) or channel method (÷ total broadcasts). The two disagree on
real schedules; a figure without its method is not a figure.
**Data**: preferred set, target %, counting method, window, scope.
**Interacts**: F6 position entitlements (the per-break mechanics), planning
steer; `kairos_api/preferred_rate.py` computes both methods today.
**Measured**: preferred-rate to date + projection, method-labelled.
**Status**: BINDS for measurement (seam live), REPRESENTABLE for the
guarantee object. **Rank**: TRADE (1–5+L, both methods) + IL (position set
"first, second, third and last" published).

### E4 `gold-break-allocation` — הקצאת ברייקי זהב
Entitlement to N gold breaks per period/programme, or first-refusal on gold
inventory. Scarce-inventory allocation: over-commitment across agreements is
itself a conflict the model must detect (aggregate feasibility).
**Data**: count/period, scope, first-refusal flag.
**Interacts**: B7 pricing; F-family placement; cross-agreement feasibility.
**Measured**: allocated vs entitled. **Status**: REPRESENTABLE. **Rank**: IL
(gold breaks trade in counted units — "רק 10 ברייקים זהב" reporting).

### E5 `makegood-accrual-policy` — מדיניות צבירת מייק גוד
The three-level accrual-and-utilisation ledger the trade describes: cures and
bonuses accrue at campaign, advertiser and agency level; agency credit may be
SPENT on a different campaign later. The policy term fixes accrual triggers,
rates, levels, utilisation rules (who may spend, on what, until when), and
expiry of credit.
**Data**: accrual rules {trigger, rate/quantity, level}, utilisation rules,
expiry, inventory-quality constraints on utilisation.
**Interacts**: E1/E7 breaches feed it; C7 added-value feeds it with its own
reason; F-family constraints apply when credit is spent as placements.
**Measured**: per-level balances: accrued, utilised, expiring.
**Status**: REPRESENTABLE (ledger is this campaign's build; projection flag
exists today as pacing alert). **Rank**: TRADE (the ledger's three-level
shape is the transcript's own correction of the smaller foreign model).

### E6 `shortfall-cure` — מנגנון השלמה
What actually happens at a breach of E1/E2: bonus spots until points arrive,
credit notes, carry-forward to next period — with cure-inventory quality
("like-for-like daypart", or the law's own 1 prime ≡ 3 non-prime substitution
anchor), cure window, and the recursion rule (what if the cure under-delivers).
**Data**: cure form, quality rule, window, valuation basis, recursion rule.
**Interacts**: consumes E1/E2 breaches, writes E5 ledger entries, spends as
F-constrained placements. **Measured**: open cures, cure delivery, residue.
**Status**: REPRESENTABLE. **Rank**: TRADE + IL (substitution anchor in law).

### E7 `underspend-true-up` — התחשבנות חוסר ניצול
The mirror obligation: the buyer misses D1's committed budget → ladder
re-rates retroactively, rebates claw back, or a shortfall fee applies.
**Data**: trigger (D1 ref), re-rating rule, fee, waiver conditions.
**Interacts**: C1 mechanics decide the arithmetic; H1 settles.
**Measured**: projected spend vs commitment, projected re-rating exposure —
surfaced EARLY, while buying more is still possible.
**Status**: REPRESENTABLE. **Rank**: STD (mechanics; the commitment IL).

### E8 `overdelivery-treatment` — טיפול בעודף אספקה
What happens when delivery EXCEEDS the guarantee: banked to buyer, charged
(rare), or absorbed. Silence in the document means absorbed — but the model
records the silence as a decision, not a default fact.
**Data**: treatment enum, cap on banking, valuation.
**Status**: REPRESENTABLE. **Rank**: STD.

### E9 `preemption-compensation` — פיצוי על הקדמת שידור
When the control room pulls or moves a spot (news pre-emption, breaking
events), the buyer's remedy: reschedule window, like-for-like quality,
make-good accrual. As-Run is the trigger's source of truth. In this market's
wartime reality this clause does real work (G5 interacts).
**Data**: qualifying events, remedy form, window, quality rule.
**Interacts**: E5/E6 machinery; As-Run ingestion when it lands.
**Status**: REPRESENTABLE (as obligation); trigger automation waits on
As-Run (owner-blocked input). **Rank**: TRADE (control-room reality) + STD
(clause form).

---

## Family F — Placement constraints (constrains)

### F1 `competitive-separation` — הפרדה תחרותית
Rival advertisers/categories not in the same break, or separated by N spots /
minutes. COMMERCIAL practice, not Israeli regulation — the UI must not call it
law. Engine primitives exist (separation rules, competitor boundary).
**Data**: category/named-rival scope, separation unit + quantity, hard/soft.
**Interacts**: F2 exclusivity (stronger); aggregate feasibility across
agreements. **Status**: BINDS (compiles to existing separation/frequency
rules). **Rank**: TRADE + IL-negative (not in regulation — attested absence).

### F2 `category-exclusivity` — בלעדיות קטגוריה
Sole-category presence in a scope (programme / daypart / period / event).
Priced accordingly; breaches are loud. Cross-agreement conflict detection is
mandatory (two exclusivities on one scope cannot both be approved —
the approval screen must catch it, deterministically).
**Data**: category, scope, period, carve-outs.
**Interacts**: F1; aggregate feasibility; E-family if breached (cure).
**Status**: REPRESENTABLE. **Rank**: STD (mechanism; exclusivity vocabulary
searched, closed-door in Israel — owner confirms phrasing).

### F3 `content-adjacency-exclusion` — הרחקה מתוכן
Keep-outs: not adjacent to named content classes (disaster coverage, competitor
brand content, children's programming per advertiser policy), not inside named
genres. The regulatory אתנחתה/content rules are channel-side background; this
term is the ADVERTISER'S keep-out list.
**Data**: content classes/genres/programmes, adjacency radius (same break /
adjacent break / same programme), hard/soft.
**Interacts**: F4 purchases (opposite sign); genre vocabulary must use the
product's canonical genre map. **Status**: BINDS (compiles to predicate
constraints on genre/programme). **Rank**: STD (form) + TRADE (practice).

### F4 `adjacency-purchase` — רכישת סמיכות
The positive twin: bought adjacency to premium content — news-adjacent breaks
(the market's named strong inventory), first break after kickoff, named-show
adjacency. A placement PREFERENCE with money attached.
**Data**: target content, break relation, per-instance premium ref (B-family).
**Interacts**: B1/B5 pricing; F6 positions within the bought break.
**Status**: REPRESENTABLE. **Rank**: IL (news-adjacency strength attested).

### F5 `programme-daypart-restrictions` — הגבלות תוכניות ורצועות
Allow-lists/forbid-lists of programmes, genres, strips, weekdays for the
advertiser's spots. The bread-and-butter constraint class; compiles directly
to the existing predicate engine.
**Data**: scope expressions (programme/genre/daypart/weekday), allow/forbid,
hard/soft. **Status**: BINDS. **Rank**: TRADE.

### F6 `position-entitlements` — זכויות מיקום בברייק
Per-campaign/per-agreement position rights: which of 1–5 + L the client may
hold, per-break maxima, Top-and-Tail rights (both ends of one break with the
exact 1-or-2-spot separation creative constraint), gold-break positions.
**Data**: position set, per-break limits, T&T flag, scope.
**Interacts**: E3 measures the outcome; F7 creative mechanics; existing
position vocabulary (1–5, L) and T&T pair rules in the frequency engine.
**Status**: BINDS (position/T&T primitives live). **Rank**: TRADE.

### F7 `creative-constraints` — אילוצי חומרים
Creative-level terms the agreement fixes: validity windows (עד מתי מותר
לשדר), version rotation shares, house-number binding per channel, the
~20-creative ceiling, QC preconditions (no air before clearance).
**Data**: per-creative windows, rotation %, QC gate refs.
**Interacts**: campaign assets store + media QC path (P13) hold the
mechanics; the agreement supplies windows/rotations.
**Status**: BINDS (validity/T&T/QC seams live), rotation REPRESENTABLE.
**Rank**: TRADE.

### F8 `spot-length-constraints` — אילוצי אורך
Allowed lengths for the client's spots, length-per-daypart rules, the
regulatory 90-second outer bound as background.
**Data**: allowed lengths, scope rules. **Status**: BINDS (length is a
first-class spot property). **Rank**: TRADE (whole-second law) + IL.

### F9 `frequency-caps` — תקרות תדירות
Max spots per break / per programme / per hour for the advertiser (viewer-fatigue
protection, sometimes mutual). Existing frequency-rule primitives carry it.
**Data**: unit, cap, scope, window. **Status**: BINDS. **Rank**: STD.

---

## Family G — Process and legal (process)

### G1 `payment-terms` — תנאי תשלום
שוטף+30/60/90, billing cycle, invoice currency. Deadline-tracked; feeds C5.
**Status**: RECORDED. **Rank**: STD (shape) — the Israeli שוטף+ convention IL.

### G2 `reporting-obligations` — חובות דיווח
What the channel must send and when: delivery reports, position reports,
make-good balances, As-Run extracts. Each is a deadline with an artifact —
and this product IS the reporting machine, so these compile to scheduled
report surfaces where they exist.
**Data**: report type, cadence, recipients, format.
**Status**: RECORDED (tracked deadlines; report surfaces exist for delivery/
pacing/preferred-rate). **Rank**: STD.

### G3 `audit-rights` — זכויות ביקורת
The buyer's right to audit delivery/settlement (and the channel's to audit
share declarations, D2). Window, scope, who pays.
**Status**: RECORDED. **Rank**: STD.

### G4 `termination` — סיום ההסכם
Breach/convenience termination, notice, survival clauses (accrued make-good
credit survives!). The survival rule matters to the ledger: E5 balances
outlive the agreement per this clause.
**Data**: grounds, notice, survival list.
**Interacts**: E5 expiry rules. **Status**: RECORDED. **Rank**: STD.

### G5 `force-majeure` — כוח עליון
War and emergency clauses — in this market, load-bearing, not boilerplate:
extended news pre-emption, campaign pauses (advertisers going dark in war
phases), rate/commitment relief windows, re-planning duties. Interacts with
E9 compensation and D-family relief. The repository's own training data is
war-shaped; the clause class is real here.
**Data**: qualifying events, relief per term class, invocation mechanics.
**Status**: RECORDED (+ relief windows surfaced against commitments).
**Rank**: TRADE-adjacent (wartime reality measured in the data) + STD form.

### G6 `confidentiality` — סודיות
Terms secrecy (the market's rate cards are closed-door for a reason).
**Status**: RECORDED. **Rank**: IL (the closed-door market itself).

### G7 `credit-security` — בטחונות ואשראי
Credit limits, guarantees (ערבות), prepayment demands for weak covenants.
**Status**: RECORDED. **Rank**: STD.

### G8 `dispute-resolution` — יישוב מחלוקות
Governing law, arbitration, the reconciliation-dispute path for rating
disagreements (which vintage, whose report — H2 pins the facts that make
disputes decidable). **Status**: RECORDED. **Rank**: STD.

---

## Family H — Measurement and settlement meta (settles / meta)

### H1 `settlement-mechanics` — מנגנון התחשבנות
The clause fixing HOW money reconciles: quarter-hour per-spot settlement
(the 8:03→8:00–8:14 rule), weekly reconciliation cadence, period true-up
order (discounts, then commission, then cures — or as stated), rounding.
**Data**: settlement grain, cadence, application order, rounding.
**Interacts**: everything in B/C/E; the QH billing engine (owner-gated) is
its computational home. **Status**: REPRESENTABLE (engine exists; per-
agreement parameters bind it). **Rank**: IL.

### H2 `measurement-source` — מקור מדידה
Which figures settle: the rating committee's panel, audience basis (Jewish
households), vintage (overnight+1), and the revision rule (final next-day
figures move up to ~3 points — which vintage is FINAL for money). The
settlement gates shipped 2026-08-10 already refuse basis-less segments.
**Data**: source, audience, vintage, revision/final rule.
**Interacts**: E1/E2/B1/B2 all measure through it. **Status**: REPRESENTABLE
(gates live, activation owner-held). **Rank**: IL.

### H3 `delivery-truth-source` — מקור אמת לשידור
As-Run as the billing truth (never the plan); fallback order when As-Run is
absent; the discrepancy pairing (planned vs aired) that makes disputes
decidable. As-Run ingestion itself is an owner-blocked input today; the term
is representable and its automation waits honestly.
**Data**: source order, discrepancy rules.
**Status**: REPRESENTABLE (term + discrepancy model), automation TRACKED on
input arrival. **Rank**: TRADE (As-Run is the only truth).

### H4 `term-effective-windows` — חלונות תוקף לסעיפים
Per-term validity inside the agreement window: a summer-only discount, a
Q4-only exclusivity, an amendment-opened CPP. The versioning meta-rule: two
versions of one term never both bind on one day; amendments (A8) open/close
versions; history is never erased.
**Data**: per-term {from, to, source-version ref}.
**Status**: REPRESENTABLE. **Rank**: STD (mechanics) / TRADE (amendments).

---

## The completeness rule this taxonomy serves

Every clause of every ingested document lands in exactly one of:

1. **Mapped** — classified to a term above and parameterised under its schema.
2. **Commercially irrelevant** — with a stated reason (boilerplate class:
   signatures, notices addresses, severability…). The reason list is itself
   closed and reviewable.
3. **Understood, not supported** — classified to a term whose status today
   cannot honour its behaviour (or to no term at all), flagged loudly,
   blocking approval until a human disposes of it.

Nothing else exists. A document's coverage state is the count of its clauses
in each bucket, and approval is impossible while bucket 3 or unreviewed
bucket 1 items remain unseen.

## Support summary (as designed; the build tracks this table)

- **BINDS today** through existing machinery: C4, E3 (measurement), F1, F3,
  F5, F6, F7 (core), F8, F9.
- **REPRESENTABLE — bound by this campaign's build**: A1–A8, B1–B7, C1, C3,
  C7, C8, C9(TV), D1, D3, E1, E2, E4, E5, E6, E7, E8, E9, F2, F4, H1–H4.
- **TRACKED** (standing computed, consequences human): C2, C6, D2, D4, D5.
- **RECORDED** (stored, displayed, deadline-tracked): B8, C5, D6, G1–G8.
- **Not applicable in this market**, kept for honest refusal: regional-feed
  splits (single national feed), co-op invoicing (no Israeli evidence),
  barter/per-inquiry buys (no Israeli evidence) — a clause matching these
  classifies as understood-not-supported with the market note, never dropped.
