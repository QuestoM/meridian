# NOT APPLIED

Foreign documentation only. Nothing here has been built into the product and
nothing in the product changes because of it. `docs/media-domain-from-the-trade.md`
outranks every source on this page, and the owner's ruling in
`docs/audits/research-scope-ruling.md` governs.

Recorded 2026-08-09. Companion to `ad-sales-api-schemas.md`, which covers vendor
APIs. This file covers published STANDARDS, and two of them answer a question the
vendor APIs do not.

---

## The one finding worth the whole file

**The only published field list anywhere that models audience goal, then
guarantee, then shortfall, then make good, then discrepancy END TO END is a
document from 2001.**

TVB's *Data Elements for the Electronic Transmission of Local Broadcast
Television Business*, issued 2001-10-02 with the American Association of
Advertising Agencies' Local TV and Radio Committee. Recovered as a live Excel
workbook from the Internet Archive: `web.archive.org/web/2006id_/http://www.tvb.org/xls/arc/edi_standards.xls`,
109 KB, seven sheets. The landing page states the goal verbatim: "Our goal is for
the local broadcast television industry to electronically conduct business via a
set of open, Internet-based (XML) standards."

Its seven transactions are Avail Request, Avail Submission, Orders, Broadcast
Instructions, Makegoods, Invoices, Discrepancy.

**The request carries the goal.** `Demographics`, `Book(s)` meaning the rating
book, `Target GRP`, `Target CPP`, `Target CPM`, `% of GRP`, `Daypart Budget`.

**Every buy line carries a `Rating`.**

**The miss is its own record.** Under `DETAIL RECORDS - MISSED`: `Total Units
Missed`, `Total Cost of Missed Spots`, `Makegood/Credit Flag` with the documented
values "M=Makegood; C=Credit", and `Reason for Miss` as fifty characters of free
text.

**And the discrepancy pairs the two ratings on one line:** `Estimated Rating`
against `Achieved Rating`, beside `Ordered Not Run` and `Run Not Ordered` totals
in both money and spots.

That last pair is the industry's canonical under-delivery primitive and it lands
directly on the open question in `israeli-rating-currency.md`. A rating held
without recording WHICH rating it is cannot be reconciled against anything. This
standard solved that by keeping two fields.

**None of this is a build instruction.** It is a twenty-five-year-old US local
spot-television standard whose XML schemas were on a members area and are gone.
What it is worth is the SHAPE, and the confirmation that a product holding one
rating per line has fewer fields than the trade needs.

## SMPTE BXF, and its honest absence

Full official schema set, free and public: `github.com/SMPTE/st2021-4`, 33 XSD
files. Companion prose at `pub.smpte.org/pub/eg2021-4/eg2021-4-2017.pdf`.

Quoted from the schemas' own documentation:

- `MakeGoodFlag` - "Used to indicate the spot is being used as a make good for a previous ordered spot that did not air properly. Default is 0."
- `PreemptionWarning` - "Do not preempt the spot"
- `TrafficCautionFlag` - "Warning indicating that the operator should think twice before preempting or changing the spot."
- `AsRunStatusType` enumerations: "Aired Without Discrepancy", "Technical Difficulty", "Did not air", "Aired with Duration Discrepancy", "Aired with Content Discrepancy", "Preempted", "Joined in Progress", "Inserted by Operator", "Unknown", "Missing Content"
- `audienceDeficiencyUnit` - "If True indicates that it is an ADU."
- `DayOfWeek` - "A 7 element binary representation of the days of the week in Monday-Sunday order where a 1 includes the day and a 0 excludes the day"

**THE HONEST NEGATIVE, and it is the useful part: BXF has NO rating, impression
or guarantee field.** A grep of all 33 schemas for those concepts returns only
`audienceDeficiencyUnit`. BXF is a traffic and automation protocol, not an
audience-guarantee protocol. The two concerns are genuinely separable, and a
complete system can hold one and none of the other.

One documented cross-reference worth noting: BXF's product-category attribute
carries the note "Recommend use of TVB EDI Value", pointing at the same 2001
standard above.

Also note the day-of-week encoding is MONDAY-FIRST. Israeli week law here is
Sunday to Saturday. Any borrowed bitmask would be silently wrong by one day.

## X12 has nothing for broadcast

Searched the full X12 4060 segment and transaction dictionary. There is **no
broadcast-advertising-specific transaction set**. What exists:

- Transaction set 290, "Cooperative Advertising Agreements". Print and retail
  co-op, with no broadcast, daypart, GRP or make-good construct.
- Segment `ADV`, "Advertising Demographic Information", whose shape is a
  qualifier plus `Range Minimum`, `Range Maximum` and `Measurement Value`. That
  min-max-plus-value shape is how a demographic band and its rating would be
  carried. Which transaction sets actually use it could NOT be confirmed from a
  free source and is not asserted here.

The researcher's inference, flagged as an inference: no public broadcast
implementation guide exists for the generic X12 sets because the US spot-TV
industry deliberately went to XML instead, which the TVB material above
corroborates.

## The 4A's and IAB XML, and what it reveals by its examples

`github.com/jpych/iab` carries the IAB e-business schema set including
`AAAA_Common.xsd`, whose root type is documented as "THE STANDARD HEADER FROM THE
AAAA". Its own examples name a sibling family: `SchemaName` "For example:
spotRadioOrder", `Media` "For example: SpotRadio".

So the modern broadcast spot schemas exist and are members-only. Their existence
and naming convention is measured; their contents are not obtained.

From the digital half, which IS public:

- `IsGuaranteed` - "Default value is True. False would correspond to a pre-emptible line item."
- `MakegoodNotes` - "Used when IsMakegood is True to reference what item(s) are being made good on."
- `ZeroCostTypeEnumeration` = `MakeGood`, `AddedValue`
- `RateCardPrice` - "Commonly available unit price equivalent to a published rate card."

## Pacing, from the vendors that publish it

Recorded here rather than in the API file because it is one comparison. Four
platforms, and the shape recurs across all of them.

**The goal and the cap are different things, everywhere.** Stated most plainly by
one vendor: caps "do not regulate the frequency of impressions to meet a goal
over time. They are simple limits, shutting off a flight as soon as it reaches
its goal." The same split appears as goal versus catch-all cap, as goal volume
versus daily volume cap, and as goal mode versus saturation limiter.

**Under-delivery catch-up, the best-documented instance:** a field whose
documentation reads "Dictates how the system deals with an underdelivered daily
budget. Use `evenly` if you'd like the unspent portions of your budget to be
spent evenly throughout the rest of flight, or `ASAP` if you'd like the unspent
budget to be spent as soon as possible", paired with a second field for "the
number of days across which the underspent amount will be distributed".

**Pacing intensity as a number rather than a mode:** one platform expresses it as
a value from 50 to 150 where 50 is behind schedule, 100 is even, and 150 is ahead.

**A THIRD NEGATIVE THAT MATTERS MORE THAN THE POSITIVES.** Of four platforms, only
ONE exposes delivered-to-date on the write object at all. The other three force
the reader to a separate reporting layer. Any design that assumes a
delivered counter sitting on the order line breaks against three of four.

And one honest tolerance statement, quoted, because almost nobody publishes one:
"Kevel's pacing algorithm will attempt to deliver 100% of goal over the desired
time period. The actual goal achieved can vary by +/- 5%, due to varying amounts
of request traffic."

## What could not be obtained, named so nobody re-hunts it

The 2002-era TVB XML schemas (members area, gone from the archive). The current
4A's spot TV and radio schemas (members only). SCTE 130-3 and its siblings
(members only; message names recovered only from a patent, and no child element
names are asserted). CableLabs ADI 3.0. Ad-ID specifications. Anything named
"TIP" as a Traffic Interface Protocol, which was searched for and **no published
standard by that name was found**, so it is not asserted to exist.

## What this means for us

Nothing, directly, and that is the ruling working. Not one field name here
belongs in this product because it appears here.

Two things are worth carrying into a conversation with someone who trades in
Israel, and both are recorded as questions in decision 10 rather than as work:
whether an estimated rating and an achieved rating need to be separate stored
values here as they are there, and whether delivered-to-date belongs on the order
or only in a report.
