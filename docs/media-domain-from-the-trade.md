# How Israeli television advertising is actually bought and sold

Source: a recorded and transcribed working conversation between the owner and
the media professional behind this project, 2 August 2026. The transcript is
imperfect and Hebrew-first; what follows is the extraction, and where the
transcript was ambiguous it says so rather than smoothing it over.

This is domain knowledge that cannot be derived from the codebase, the data, or
any public source. Most of it contradicts what a reasonable engineer would
assume. Read it before designing anything that touches orders, placement,
pricing or creatives.

## The single most important correction: an order is not a plan

Large campaigns arrive **one to two days ahead, not as a whole flight**. The
agency deliberately withholds the rest because it wants to keep control and
re-optimise daily. Sponsorships are the exception and do come further ahead.

Three consequences the product must absorb:

- **The campaign is a living object updated daily**, at best. Mid-day re-planning
  is normal. A design that assumes a campaign is entered once and then executed
  is modelling a business that does not exist.
- **Agencies over-order prime deliberately.** They ask for more prime minutes
  than they need because they know they will not get them, so the remainder
  lands where they actually wanted it. An order quantity is a negotiating
  position, not a demand forecast.
- **The channel does not release every break, and the schedule is not known a
  week out.** Both sides are working under deliberate opacity.

The break schedule itself is laid out roughly at the **end of the previous
year** as a schematic gantt of programmes and break counts. Launches move,
programmes get swapped, and rival channels' moves change it during the year.

## Placement is prioritised, and the priorities are a run parameter

Placement is not a single objective. The operator sets **a priority level per
parameter** before each run, and changes them between runs:

- large media companies
- direct clients
- **success deals** (`עסקאות הצלחה`), which are revenue-share arrangements
- new campaigns, which can be deprioritised deliberately: "this run I am not
  favouring new campaigns, I want to split what I have evenly across the market"
- competitive separation
- regulatory checks, which are not a priority but a switch: does this run also
  enforce them

**Pressure level** (`רמת לחץ`) is a per-campaign multiplier: 1.3 means treat it
as though it paid 1.3 times its budget. The owner had built this at advertiser
level; the media professional was explicit that **it belongs at campaign level**,
with the option to default from the advertiser. Campaign level subsumes
advertiser level, because doing every campaign of an advertiser covers the
advertiser.

## What an order actually contains

`שם ערוץ, תאריך, שעה, שם תוכנית, אורך תשדיר` — channel, date, time, programme
name, spot length.

- **The time is approximate and everyone knows it.** An order says 20:40 and the
  break may open at 20:30 or 20:50, or the spot may land in a different break
  entirely. It is a range, not a commitment.
- **Length is in whole seconds. There are no milliseconds.** Frames exist (the
  transcript disputes 24 against 25 FPS and settles on 25 at channel level, 24
  common in digital) but the trade unit is the second.
- **Positions are requested but not binding at order time.** Position is settled
  the day before broadcast. A media buyer placing an order does not discuss
  position at all.

## Positions: the product is wrong today

Preferred positions are **first, second, third, fourth, fifth and Last**, where
Last is `L` and is a distinct position, not a number. Which of them count as
preferred is per client and per agreement, so it is configurable, not fixed.

The product currently models 1, 2, 3 and last. **It must model 1 to 5 plus L**,
and it must treat L as its own thing rather than as the last ordinal.

There is a further subtlety with real money attached. A campaign can hold both
the **Top and the Tail** of the same break: the first spot and the last spot.
That is two positions in one break. Counting then becomes contested, and the
transcript names two live methods:

- **Agency method (preferred by the trade):** numerator is the number of
  preferred positions obtained; denominator is the number of breaks the campaign
  appeared in, counting a break twice if it appeared twice.
- **Channel method:** measured out of total broadcasts.

They give different percentages for the same schedule. The product must state
which method a figure uses, because the two parties audit each other with it.

## Top and Tail is a creative constraint, not a position preference

A campaign carries **many creatives**, up to twenty versions. A common structure
is a 10 second spot plus a 6 second closer, with the constraint that they air in
the same break separated by **exactly one or two other advertisements**. These
are hard placement constraints and the optimiser has to honour them.

Each creative also carries a **validity window**: until when it may be
scheduled. That is a constraint too.

## The rating currency, and why the price is not final on the night

The trading currency is **Jewish households, quarter-hour rating, overnight plus
one**, where plus one is deferred viewing. GRP is gross rating points against any
audience; **TRP is against a named target audience** and is what campaigns
commit to, for example rating points for women 35 and over.

**The final rating is only known the day after broadcast**, and it moves: one,
two, even three points can be added. On a programme rating ten, three points is
a thirty percent revision. Any figure the product shows on the night is
provisional and must say so.

## As Run is the only truth about what aired

`As Run` is a JSON file from the broadcast system, **second by second, produced
after the fact**. It is not the schedule and not the EPG.

It matters because **the schedule is not what happened**. On air, the control
room delays a break because a presenter has not finished a sentence, or someone
telephones and has a campaign pulled, and none of that passes through the order
system. Billing and delivery must be computed from As Run, never from the plan.

## Make good is a ledger, and it is not optional

`Make good` covers bonuses and compensation, for a spot that did not air or
aired wrong. It is managed at **three levels at once**: campaign, advertiser and
agency. An agency accrues, for example ten percent of its spend, and may spend
that credit on a different campaign later. So it is an **accrual and utilisation
ledger**, not a per-campaign flag.

Direct buying carries around a **twenty percent return**, which buyers routinely
ask to take as added media rather than cash.

## The rate card is deeper than the product models

Layers named in the conversation, in order:

1. a base price per hour
2. adjustment per day of the week
3. adjustment per hour
4. adjustment per programme category
5. adjustment per specific programme
6. **adjustment per specific date** — the owner confirmed this is missing today
7. adjustment per break position
8. seasonal or periodic premiums, for example a rise in the three weeks before
   Passover
9. all of the above adjustable per agency, per advertiser, and per campaign

Gold breaks carry a separate rate card.

## The systems this has to live beside

- **Owner** (`עונר`) is the incumbent Israeli traffic system each channel runs.
  Closed, no public API, changes require a vendor request, but a database link
  for synchronisation is possible. The media professional's stated ambition is
  to replace it.
- **Jumbo Media** is the Israeli hub where creatives are uploaded. The agency
  uploads and selects the channel; the channel receives automatically. Jumbo
  performs the quality control, including frame rate. It has an API and there is
  an existing relationship.
- **House number** is the channel-side identifier for one creative version,
  issued by Owner. **The same creative has a different house number per
  channel.** The workflow is: create the version in Owner, Owner issues the house
  number, the operator pastes that number into Jumbo to bind the file to the
  booking. Today this is manual copy and paste.

## Where this is going, and what to build toward

The stated direction is that the channel takes over placement: the agency sends
**a GRP or target-audience goal instead of a spot list**, and the channel is
accountable for delivering it under all its constraints. The media professional
believes large agencies will resist, because a chief executive wants to see his
own advertisement tonight, but that even thirty to fifty percent of the market
moving is transformative.

**This is the product's real thesis**, and the goal-based order is therefore not
a secondary mode. It is the destination.

The near-term shape asked for is a **media company portal**, because today the
entire negotiation happens over email:

- **Stage one, a block booking.** An agency reserves capacity ahead, often
  without naming the client: "ten minutes in prime every day through August".
  Channels push back and demand the advertiser's name, because they fear
  cancellation and want to know whether the advertiser is worth the inventory.
- **Stage two, the actual buy**, about two days out, with the flight.
- Every step needs a visible status such as awaiting approval, and approval
  should be possible **from inside the email** rather than by forcing everyone
  onto a new platform, with a link through to the full view.

Cancellation rules are demand-dependent: a day before broadcast billing has
started and cancellation is refused, but in high-demand periods a channel may
welcome a cancellation because it can resell.

## What the product is missing, drawn from the above

Ordered by how much of the business they block:

1. Goal-based orders (TRP against a named audience) as a first-class order type.
2. Positions 1 to 5 plus L, with the counting method stated.
3. Creatives as real objects: many per campaign, house number per channel,
   validity windows, Top and Tail adjacency constraints.
4. Make good as a three-level accrual and utilisation ledger.
5. As Run ingestion, with delivery and billing computed from it.
6. Run parameters as an explicit, per-run priority set.
7. Rate card by specific date, and seasonal premiums.
8. The block booking stage, and an approval state machine that works over email.
9. Owner and Jumbo integration seams.
