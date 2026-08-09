# Break shape and capacity: what the model gets wrong, and what fixing it requires

Written 2026-08-09 in answer to the owner's question, which was not a feature
request but a diagnosis: break length and break count should be dynamic within
sensible bounds, the model itself should work out how many and how long, it
should follow the load of the orders, and something should coordinate the two so
the channel does not accept more than it can carry.

He is right on every count, and the product is further from it than the code
suggests. This document establishes how far, why the obvious fix is a trap, and
what has to be true before any of it can ship honestly.

Everything numbered below was MEASURED on this tree on 2026-08-09, not recalled.

---

## 0. CORRECTED 2026-08-09 BY THE MEASUREMENT THIS DOCUMENT COMMISSIONED

Stage A ran and it corrected four things below, including two numbers I
published. `docs/break-shape-measured.md` is the authority; this section is the
diff so nobody reads a superseded figure.

**1,145 of the 3,055 "breaks" ARE NOT BREAKS.** They carry an Excel epoch-zero
start time of 01/01/1900, a null parsed timestamp and `hour_of_day = -1`, and
each is its own single-row break at position 1. That is 37.5 percent of the
count, all of it below the median. They are what drags the median to 39. Drop
them and change nothing else: **median 39 becomes 108**, p75 186 becomes 310, and
the share above 120 seconds goes from 29.4 to 47.1 percent. Whether they are real
airtime or an export defect the data cannot say, and it is the first question for
the owner.

**MY "40 BREAKS PER HOUR" IS AN ARTIFACT AND MUST NOT BE USED.** The real maximum
in a clocked hour is SEVEN. Forty is reachable only through the minus-one bucket,
one cell per day holding that day's orphans. Anyone sizing a breaks-per-hour
guardrail against 40 is sizing against an export defect. The hourly block in
section 1 did not reproduce under any of four groupings; only the 24.3-minute
maximum survives every method.

**MY SHARPEST LINE WAS WRONG, AND STAGE A SAID SO FIRST, IN THE CONSTANT'S
FAVOUR.** I wrote that 120 seconds is "the one summary statistic that describes
none of it". Against COMMERCIAL airtime, 120 seconds is exactly the 25th
percentile. What survives is narrower and still real: the median commercial pod
is 58 percent longer, 72.8 percent exceed 120 seconds, and only 2.4 percent sit
at it.

**STAGE C AS WRITTEN IS CONTRADICTED.** I proposed that the length a segment gets
should be the length its programme carries. Programme duration does NOT carry a
length: correlation with a pod's length is **+0.002**. It carries a COUNT,
correlation **+0.843**, and the count is already a decision the optimizer makes.
Stage C has to be rewritten around what actually varies.

**AND THE FINDING THAT REFRAMES STAGE B.** Spot count explains 85 percent of a
pod's length variance, and the mean spot inside a pod is about 21 seconds flat at
every pod size above three. A break is long BECAUSE MORE SPOTS WERE SOLD INTO IT.
So stage B's question is not "does an extra thirty seconds cost audience", it is
**"does an extra SPOT cost audience"**, and the two are the same question only if
the cost is per second rather than per interruption. The trap in section 2 holds
and sharpens: revenue is priced per second, retention per break, and the ratio
between them is a spot count nobody has modelled.

Two further corrections of mine. The commercial-versus-promotional split I
flagged as open is measured: sponsorship billboards have a median AND mode of
exactly 6 seconds, 1,973 of 3,055 breaks carry no commercial spot at all, and a
commercial pod count is 760 rather than 3,055. And my suspicion about the
35.6-minute hour was right: a break starting 22:59:40 runs 432 seconds past the
boundary, and excluding it gives 28.4.

The prime-window gap between 322 and 39 seconds was never a prime effect. It is a
DEFINITION ladder: 39 for everything, 108 without the orphans, 340 for breaks
carrying a commercial, 389 in prime.

---

## 1. The measurement, and it is not close

**Every break in the plan is exactly 120 seconds.** Not a default that varies in
practice: `kairos/optimize/_types.py:17` sets `DEFAULT_BREAK_LENGTH_SECONDS =
120.0`, `kairos/data/transform.py:270` and `:405` fill every segment with it, and
the shipped plan carries exactly ONE distinct value of `break_length` across all
2,540 operator rows.

**A whole month of the operator's own spots says the constant is the mean of a
distribution that looks nothing like it.** `data/Spots.csv`, 18,669 spot rows on
the operator's channel across all 30 days of November 2024, grouped by the file's
own `break_id`: **3,055 real breaks.**

| | seconds of advertising |
|---|---|
| minimum | 6 |
| 10th percentile | 6 |
| 25th percentile | 12 |
| **median** | **39** |
| 75th percentile | 186 |
| 90th percentile | 373 |
| maximum | **747** |
| mean | **117** |

**Ten of 3,055 breaks are 120 seconds. That is 0.33 percent.** 29.4 percent are
longer. Spots per break: median 2, 90th percentile 19, maximum 47.

Look at the mean. **117 seconds, against a modelled constant of 120.** The
constant is almost exactly the average of a distribution running from 6 to 747
whose median is 39. Somebody picked the mean of a wildly skewed distribution and
froze it, which is the one summary statistic that describes none of it.

**Against the same month, per clock hour, on the operator's own channel:**

    median 8.1 minutes    90th percentile 14.7    maximum 24.3
    hours above the model's 8.0 ceiling:  352 of 682, FIFTY-TWO PERCENT
    hours above the 12.0 setting:         157, 23 percent
    hours above the cited regulatory 10:  241
    breaks per hour: median 3, MAXIMUM 40, against a cap of 4
    hours above 4 breaks: 148, 22 percent

**The model cannot represent half the hours the channel actually broadcasts.**

ONE CAVEAT THAT STEP A MUST SETTLE AND I WILL NOT SETTLE HERE. This file carries
promotions as well as commercials, and a 6-second break at the 10th percentile
looks like a billboard or a promo slot rather than a commercial pod. The split
between commercial and non-commercial airtime changes every number above, and
which of them a regulatory cap even counts is a question for the owner. What does
NOT change under any split is that the shape is a wide distribution and the model
holds one number.

**The single prime window in `data/daily_input/` agrees, and shows the tail.**
Ten breaks from the operator's own traffic file for 2025-04-27, 20:24 to 22:59:

| | seconds of advertising |
|---|---|
| shortest | 18 |
| lower quartile | 32 |
| median | **322** |
| upper quartile | 579 |
| longest | **803** |

Zero breaks at 120s. Six longer, four shorter. **The longest real break is 6.7
times the only length the model can express.** Spots per break: median 17,
maximum 38, minimum 1. The 17 sits inside the trade's own published range of
twelve to fifteen advertisements a break, and the minimum of 1 is a gold break.

**Per clock hour, the file is outside what the model can represent, in every
hour it covers.** The traffic file is one prime window, 20:24 to 22:59, so the
comparison is prime against prime and not day against day:

    20:00   10.1 minutes of advertising in 2 breaks
    21:00   10.8 minutes in 4 breaks
    22:00   35.6 minutes in 4 breaks

The model's hard ceiling is four breaks an hour at two minutes each, so **8.0
minutes**. It cannot represent a single one of those three hours. The settings
cap is 12.0 and the regulation cited for this regime is 10, and reality sits
above all three, which is a measured corroboration of the owner's own ruling
that the commercial channels do not always work to the regulation.

**And the plan's shape is the mirror image of the real one.** The plan supplies
160 minutes a day in about 80 breaks of 2.0 minutes. The real evening carries its
advertising in a handful of breaks averaging 5.6 minutes. The model plans many
short breaks; the channel airs few long ones. The totals are not the problem.
The SHAPE is the problem, and the shape is what every position, pod, pairing and
premium in this product is defined on.

## 2. Why the obvious fix is a trap, and this is the part that matters

The obvious fix is to let the optimizer choose the length. It must not be done
first, and the reason is arithmetic rather than engineering.

Revenue for a break is computed from its length. `_segment_math.py:96-105` passes
`length` into `break_revenue(effective_tvr, length, cpp, unit_seconds=...)`, so a
break earns in proportion to its seconds.

Retention cost is computed from the COUNT and never from the length.
`_segment_retention(segment, k)` takes `k` and nothing else. A break costs the
programme a fixed slice of audience whether it runs eighteen seconds or eight
hundred.

So the moment length becomes a free variable, **every additional second is pure
revenue at zero modelled cost.** The optimizer would set every break to the
maximum the bounds allow, everywhere, always, and it would be right to, given
what it has been told. The result would not be a model of break length. It would
be a constant with a larger number, dressed as a decision, and it would move real
money on the strength of a coefficient nobody fitted.

**This is a measurement problem before it is a code problem.** Nothing may vary
the length until the retention model prices seconds. The good news is that the
data to fit it exists: the audience series and the daily traffic files carry real
breaks spanning 18 to 803 seconds, which is a range wide enough to identify a
length term if one is there. The honest possible outcomes are three, and all
three are useful:

1. Length has a measurable audience cost, in which case fit it and the lever is
   safe to open.
2. Length has no measurable cost beyond the interruption itself, in which case
   the trade's own break-length practice is driven by something other than
   audience shedding, and the bound has to come from the regulator, the schedule
   and the sales policy rather than from the model.
3. The data cannot answer it, in which case the honest state is a length that
   remains fixed and a product that says so, and the estimate is not invented.

Anything that ships before that question is answered is a guess with a revenue
number attached.

## 3. What already exists, so nothing is rebuilt

The anti-duplication check, before any design:

- **`break_length_seconds` is ALREADY a per-segment field.** It is not a global
  constant in the type system. Every segment carries its own and every segment is
  handed the same value.
- **The value function ALREADY honours a per-break length.** With explicit pins,
  `_segment_math.py:96-98` values the k-th break at that pin's own duration
  rather than the segment's. Variable-length breaks are a path the revenue
  arithmetic already supports and only the pinned path uses.
- **The count is ALREADY a decision.** The optimizer chooses `k` per segment
  against the guardrails; that half of the owner's question is built.
- **The guardrails ALREADY carry the right shape**: breaks per hour, minutes per
  hour, minutes per day, spacing, protected content. What they lack is anything
  that varies underneath them.
- **The pod surface ALREADY reads real variable-length breaks** from the traffic
  file, with positions, pairings and arithmetic defined on them.

So the gap is narrower than it looks and worse than it looks at the same time.
The engine can express a variable-length break. Nothing has ever given it one,
and nothing knows what one costs.

## 4. The second half of the question: coordinating load and capacity

The owner's second point is separate from the first and is not solved by it: do
not load orders in when there is no room, and have a mechanism that reconciles
the load against the quantities.

**What exists today: nothing.** There is no point anywhere in this product at
which an order is checked against remaining capacity. Campaigns are stored, the
plan is optimised, and the two never meet. The research sweep found this is the
concept international systems call an avail, and separately found that no Israeli
source uses that word, which is a vocabulary problem and not an absence of the
thing.

**What it requires, and each piece is a real question:**

- **A supply figure per unit of time**, derived from the plan rather than
  asserted: for a given day, daypart or hour, how many advertising seconds does
  the schedule actually offer, under the guardrails in force. This is computable
  today and computed nowhere.
- **A demand figure over the same unit**, derived from the orders: how many
  seconds are committed, by whom, and with what latitude. The campaign store
  holds flights and goals; it does not hold a seconds demand curve.
- **The comparison, with three states.** Room, full, oversold. Never a silent
  zero and never a green screen because nobody looked.
- **A decision about what "full" DOES.** This is where the trade document has to
  win over instinct. It says orders arrive one to two days out, are revised
  daily, and that **agencies deliberately over-order prime knowing they will not
  get it**. A product that refuses an order because a capacity number says full
  would be modelling a business that does not exist. The mechanism is a
  DISCLOSURE and a RANKING, not a gate: the channel must be able to see it is
  oversold, and to know which commitments give way, which is exactly what
  `priority: preemptible` already models on a campaign.

**And the coordination runs in both directions, which is the owner's real
point.** Capacity is not fixed and demand is not passive. Given a heavy evening,
the schedule should be able to carry more, by more breaks or by longer ones,
within bounds. Given a light one, it should not manufacture inventory nobody
bought. That loop is the thing being asked for, and it cannot close until the
first half is honest, because a loop that shapes capacity to demand with a free
revenue lever inside it will simply saturate.

## 5. Where the bounds come from, since "sensible bounds" is doing a lot of work

Not from a constant, and not from me. Four sources, in order of authority:

1. **The owner and the trade document.** What lengths the channel actually airs,
   and what its own commercial policy allows. This outranks everything.
2. **The regulator**, off by default per the owner's ruling of 2026-08-09, with
   the ability to turn a cap on when somebody asks.
3. **The schedule itself.** A break cannot exceed the programme that carries it,
   and spacing constrains how many fit.
4. **The measured audience cost**, once section 2 has an answer.

The real traffic file supplies the empirical envelope, and it is 18 to 803
seconds. That is a description of what happened, not a permission.

## 6. What has to happen, in order, and nothing may jump the queue

**A. Establish the empirical shape, and the data is already here.**
`data/Spots.csv` carries 50,386 spot rows across all 30 days of November 2024
with the file's own `break_id`, `position_in_break`, `Duration` and `Spot type`.
The first pass is above. What Step A owes beyond it: the split between commercial
and promotional airtime, the distribution BY daypart, programme type and day of
week rather than pooled, and the same by programme duration, since a break's
length plausibly follows the programme that carries it.

Note what this dataset is NOT for. Its revenue column is a synthetic price from a
constant base rate, verified on 99.67 percent of rows in an earlier audit. It is
usable for TIME and STRUCTURE and it is not usable as money.

**B. Fit the length term in the retention model, or fail honestly.** Does an
extra thirty seconds of advertising cost audience beyond the interruption
itself? The three outcomes in section 2 are all publishable. Nothing about
length ships before this returns.

**C. Make the length a real per-segment value, still not a free variable.** Fed
from A: the length a segment gets is the length that programme, daypart and
channel actually carries. This alone would move the plan closer to reality than
anything else on this list, and it can be measured against the golden before it
is believed.

**D. Only then, let the optimizer choose within bounds**, with the cost from B
priced in, and prove it does not simply saturate.

**E. The capacity ledger, in parallel with all of the above**, because it does
not depend on them: supply from the plan, demand from the orders, three states,
disclosure rather than a gate.

**F. The loop last.** Shape capacity toward demand, within bounds, once every
number in it is one somebody measured.

## 7. What I am NOT claiming

That the plan's totals are wrong. 160 minutes a day against a real evening is not
a comparison anybody should draw from one 2.5-hour window, and I have not drawn
it. What is established is the SHAPE: many short breaks where the channel airs
few long ones, and a hard ceiling of 8 minutes an hour against three real hours
that all exceed it.

That the picture is complete. The month of spot data settles the SHAPE beyond
argument, and it leaves the commercial-versus-promotional split open, which is
the first thing Step A must resolve because several of the numbers above move
under it.

That 35.6 minutes in the 22:00 hour is a clean figure. It is what the file says
and it deserves its own look, since the file ends at 22:59 and a long break at
the boundary would inflate it.

That any of this is a regulatory finding. The owner has ruled that the caps are
off by default because the commercial channels do not always work to the
regulation, and reality sitting above the cited limits is consistent with exactly
that.
