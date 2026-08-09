# NOT APPLIED

Foreign documentation only. Nothing here has been built into the product and
nothing in the product changes because of it. `docs/media-domain-from-the-trade.md`
outranks every source on this page, and the owner's ruling in
`docs/audits/research-scope-ruling.md` governs: a foreign field name is not a
reason to add a field here.

Recorded 2026-08-09. Every claim below was fetched with plain curl or WebFetch,
no cookies and no login, unless flagged otherwise.

---

## Why this file exists at all

The other research files answer "what does the trade call this". This one answers
a narrower question that turned out to be answerable: **which of these systems
publish an actual machine-readable schema, and what fields does it carry.**

The answer is that four do and four do not, and the ones that do publish field
names for exactly the concept this product is building a seam for right now: an
order that names an audience goal rather than a spot list.

**Read it as vocabulary and as evidence that the shape is real, never as a
specification to implement.** Field names are the weakest possible kind of
foreign evidence: they tell you a system holds a value, not what a trader means
by it, and not whether anyone in Israel means the same thing.

## What is public and what is gated

| Vendor | Verdict |
|---|---|
| Imagine Communications | PUBLIC. Full OpenAPI 3.0 JSON, no login. The strongest find. |
| Salesforce Media Cloud | PUBLIC. Object and field reference. |
| FreeWheel | PUBLIC. Publisher and advertiser reference. |
| Marketron | PUBLIC OpenAPI 3.0, and ZERO audience or rating fields in 124 schemas. |
| WideOrbit | GATED. Developer portal behind single sign-on. Two public PDFs, neither a business schema. |
| Operative.One | GATED. No public API reference or schema found. |
| Mediaocean / Prisma | GATED. Explicit HTTP 403. |
| VideoAmp | NOT FOUND. No public developer portal or schema. |

## Imagine Communications, the one that answers the question

Swagger JSON reachable by plain curl, found through `SwaggerUIBundle({url: ...})`
embedded on the public pages. Host `https://imaginecommunications.com`:

- `.../content/uploads/2024/10/ImagineComms-xGLinearOrderAPI-1.json`
- `.../ImagineComms-Landmark-Sales-API-2.49.003.json`
- `.../ImagineComms-GamePlan_API-5.1.7.json`
- `.../SureFire_management-api-1.5.1.json`

The vendor's own caveat, quoted from those pages: "Our documentation is updated
regularly. For the most comprehensive and current APIs request access to our API
Developer Portal." So this is a published snapshot, not necessarily current.

**The goal-bearing structure, `xgLineAudience`.** Field names and types only; the
specification's own description strings are EMPTY, so no meaning is quoted and
none should be inferred:

    SegmentCode (string)
    RatingGoal (double)      RatingToDate (double)      RatingForecast (double)
    ImpressionGoal (int32)   ImpressionToDate (int32)   ImpressionForecast (int32)

Three tenses of the same quantity, in two currencies. Goal, to date, forecast.
That triple is the interesting part, not the names.

**The order line around it**, `xgOrderLine`: StartDate, ThruDate, Daypart,
ExcludeDaypart, RunPattern (WeeksOn / WeeksOff), WeeklyQty, FlightQty, DailyQty
(MondayQty through SundayQty), ValidDays, BillingMode, LineRate, RNUnitRates,
Length, Priority, EffectivePriority, Audience, GenerateBonus, BonusQtyType
(PercentOrCount), BonusQty, IsMakegood (boolean), LineAutoMakegoodMode, xgStatus.
The header adds FrontLoadPercent; the summary adds GrossAmount and SpotCount.

**Landmark Sales**, where descriptions DO exist and are quoted verbatim:

- `Spot.makeGoodSpotNumber` - "Make Good Spot Number for the Spot"
- `Campaign.numberOfRatings` - "Fixed Ratings related Ratings specification. Must be >0, if Delivery Currency is '6'..."
- `Campaign.revenueBooked` - "Revenue Booked. Sum of Spot PSD Price (Spot Status is \"S\") and any Payment Schedule amount"
- `CampaignMonthBudget.deliveryPercentage` - "Spots delivery percentage for year month. Only Applicable for Fixed Ratings and Fixed Schedule Delivery Currencies"
- `DeliveryCurrencyPricing.premiumCPT` - "Defines the optional PremiumCPT Index. Normally related to delivery currency of CPT index base ratings (2)"
- `Payback.paybackAmount` - "Amount of Payback required. A negative value implies Clawback"
- `Campaign.demographNumber` - "Landmark Demograph Number"

**GamePlan**, an optimiser, and its under and over delivery vocabulary. Names
only, descriptions empty:

    CampaignModel: demoGraphic, startDateTime, endDateTime, revenueBudget,
      targetRatings, actualRatings, revenueBooked, revenueRequired,
      revenueAchieved, achievedPercentageTargetRatings,
      achievedPercentageRevenueBudget, ratingsDifferenceExcludingPayback,
      valueDifference
    RSDeliverySettingsModel: daysToCampaignEnd, upperLimitOfOverDelivery,
      lowerLimitOfOverDelivery
    ScenarioCampaignResultModel: targetAchievedPct,
      deliveryCappingGroupPercentage, differenceValueDelivered,
      differenceValueDeliveredPercentage, passThatDelivered100Percent
    StrikeWeightModel: startDate, endDate, desiredPercentageSplit,
      currentPercentageSplit, revenueRequired, revenueAchieved

`upperLimitOfOverDelivery` and `lowerLimitOfOverDelivery` as a PAIR is worth
noticing: over-delivery is bounded on both sides rather than being free.

## Salesforce Media Cloud

`developer.salesforce.com/docs/atlas.en-us.media_developer_guide.meta/...`

A correction to something an earlier note assumed: **"AdSalesOrder" and
"MediaAdSalesGoal" DO NOT EXIST** in the published object list. The real objects
are AdOpportunity, AdQuote, AdQuoteLine and AdOrderItem, 63 in total.

Quoted verbatim:

- `ImpressionLimit` - "The number of impressions that must be delivered for the ad order item."
- `GrossRatingPoint` - "The gross rating point calculated on the basis of AdSpaceSpecification.AudienceSizeRating * Paid Commercial Time per 'Linear Commercial Time Slot Unit of the Org'."
- `CostPerRatingPoint` - "The cost per rating point calculated on the basis of QuoteLineItem.ImpliedRate / AdSpaceSpecification.AudienceSizeRating."
- `PrimaryDemographicCodeId` - "The ID of the primary demographic code associated with the ad order item."
- `PaidAdTime` - "Indicates total commercial time slots customer are paying for in seconds."
- `AdLinearAvailability` - "Represents the daily, weekly, or monthly view of offered, available, booked, and forecasted units for the Linear media type calendar view."

**NEGATIVE FINDING, and it is the useful one.** No delivered-to-date, no pacing,
no make-good anywhere in the published Media Cloud model. Delivery actuals are
deferred to a separate integration API whose published guide carries no
field-level schema at all.

## FreeWheel

`api-docs.freewheel.tv/publisher/reference/placement-v3`, public. Quoted:

`budget_model` takes CURRENCY_TARGET, IMPRESSION_TARGET, ALL_IMPRESSION, SOV,
SOP, SOI, EVERGREEN, DEMOGRAPHIC_IMPRESSION_TARGET, DEMOGRAPHIC_CURRENCY_TARGET,
CUSTOM_EVENT_TARGET, CUSTOM_CURRENCY_TARGET.

- `on_target_impressions` and `gross_impression_cap` - "Valid when budget_model = DEMOGRAPHIC_IMPRESSION_TARGET"
- `demographic_on_target_calculation` takes DIRECT, COMPOSITIONAL, COMPOSITIONAL_INCLUDE_UNKNOWN_BANDS
- `cold_start_on_target_rate` - "Value -2 means using the ratio of on_target_impressions to gross_impression_cap. Value -1 means using RON level prediction model. A positive value means using a custom rate. For example, 120 stands for 120%."
- `pacing` - "The pacing of the placement." Values SMOOTH_AS, FAST_AS, FORECAST_INFORMED_DELIVERY_OPTIMIZATION, SMOOTH_OVER_LIFE_BUT_FAST_AS_WITHIN_A_DAY, CUSTOM_PACING, PRE_DEFINED
- `pacing_point` - "A set of pacing points, for each pacing point you must have a date and a percentage"
- `over_delivery_value` - "To use the network default, the node should be empty in the XML"

**NEGATIVE:** zero occurrences of "makegood" on the Placement V3 page.

`cold_start_on_target_rate` deserves a note because it is an honesty mechanism in
another product's clothes: a single field that says WHICH of three methods
produced a rate, one of them being a model and one being a hand-set value. That
is the same shape as this product's real / unavailable / unknown discipline,
arrived at independently.

## Marketron, and its negative result

`developers.marketron.com`, public OpenAPI 3.0, 124 schemas.

- `Spot.isMakeGood` - "Is Make Good enabled."
- `Spot.scheduleStatus` - "Status of the spot scheduled e.g. `Scheduled`, `Reconciled`, `Bumped`."
- `TraficOrderSummary.projectedSpotCount` / `.actualSpotCount`

**NEGATIVE FINDING across all 124 schemas: no rating, GRP, CPP, CPM, demographic,
audience-goal or pacing field anywhere.** It is a traffic, billing and receivables
system, radio-first. Worth recording because it shows the two concerns are
genuinely separable, and a system can be complete at one of them and hold nothing
of the other.

## The gated four, and one trap

**WideOrbit.** The developer gateway redirects to a single sign-on page. The
public data-API guide states its own model verbatim: "DAPI uses the closed
authentication model, where client authentication and usage patterns are managed
by WideOrbit." A separate public sales training PDF names user-interface fields
verbatim, "adjust the GRP %, GRP, CPP and Station Budget in the individual
cells", which is domain evidence and not a schema.

**Operative.** No public API reference found. **THE TRAP, and it is worth the
line:** `developer.operations1.com`, which search engines surface for this query,
is **operations1**, an unrelated German manufacturing-software company. It is not
Operative Media. Do not cite it. One quoted endpoint from an older release note
could not be verified against the current document and is treated as unverified.

**Mediaocean / Prisma.** HTTP 403, verbatim: "You don't have permission to read
prisma_integration_api. Not a registered user? Contact us to register."

**VideoAmp.** No public developer portal or schema; only third-party integration
notes referencing it.

## What this does and does not mean for us

It means the shape this product is building a seam for is a shape other systems
hold explicitly: a goal, a to-date and a forecast, in a named audience currency,
with bounded over-delivery.

It does not mean any of these names, defaults or enumerations belong here. Every
one of them encodes a market's assumptions, and this market differs from those
markets in the ways the trade document records, including a lead time roughly
thirty times shorter. A field copied without its market is a guess wearing a
specification's clothes.

Nothing here is a build item. The questions it raises for someone who trades in
Israel are recorded as decision 10 in `docs/ux-gauntlet/decisions-for-owner.md`.
