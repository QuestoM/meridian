# Decisions only you can make

> **Owner ruling, 2026-08-01: full approval given for everything.**
>
> That resolves every decision here that was waiting on a choice, and each one
> proceeds on the recommendation written below it: decision 1 takes option A and
> re-keys advertiser identity on the 41 observed names, archiving the 45
> synthetic rows as demo data; decision 2 takes option A, an explicit break
> identifier per ad, with the stated splitting rule as the fallback for
> historical data that will never carry one; decision 5 makes publishing an
> internal freeze with a named version performed by the planner alone, and moves
> the regulatory guardrails into their own store with an effective date, a change
> record and a company-only permission until a real owner is named.
>
> Two of the five were never blocked on permission and approval cannot unblock
> them, so they are recorded honestly rather than marked resolved. Decision 3
> needs a number: what a week is measured against, its unit, its grain and its
> owner. Decision 4 needs data: a current or near-future broadcast week, real
> campaign flights, and ideally a delivery feed. Neither is a choice anyone can
> approve; they are facts that exist outside this repository. Both surfaces ship
> with an honest empty state that names the missing input and offers the path to
> supply it, which is what the specification already committed to, so nothing
> waits on them and nothing pretends to know them.
>
> This approval covers the decisions on this page and the build that follows from
> them. It is not read as standing authorisation for anything irreversible that
> was never proposed here: deployments, purchases, outbound messages, credential
> use or destructive data operations still get asked for individually, because a
> blanket yes to a page of design choices is not a yes to an action nobody
> described.

Written 2026-07-31 against HEAD `5a80a709`. Five decisions. Every other question
raised during discovery I answered myself from the code and the data; these five
cannot be answered that way, and each one blocks a named build piece.

Each carries the options, the evidence I measured, my recommendation, and what
stops until you answer.

---

## 1. Advertiser identity: which of two honest methods

**The situation, measured with pandas this session.**

- `data/advertiser_rules.csv` is 45 rows and 8 columns. There is no name column
  and no alias column, and `notes` is empty in 45 of 45 rows. The ids are
  `ADV_01` through `ADV_45`.
- The real advertiser vocabulary is **41 Hebrew names**. They are in
  `data/agency_advertisers.csv` (41 rows, `source = observed` on all 41) and in
  the daily file's `מפרסם` column (41 distinct). The two name spaces match each
  other **41 of 41**.
- The intersection between the 45 ids and either name space is **zero**.
- Consequence today: all 45 advertisers return `display_name: ""`,
  `revenue: null`, `rule_count: 0`. Not one of them has ever priced a spot.

Forty-five cannot map onto forty-one. There is no artifact anywhere on disk
that says which `ADV_xx` is `בנק הפועלים`.

**Option A. Re-key the store on the observed names.** Give
`advertiser_rules.csv` a `name` and an `aliases` column, exactly the shape
`agencies.csv` already has, and key it on the 41 observed advertisers. The 45
synthetic rows are archived to `data/_backups/` and declared demo data. Their
premiums go with them, including `ADV_02`'s `default_premium` of 1.27.

**Option B. You supply the mapping.** Send a list of which `ADV_xx` is which
Hebrew name. The 45 rows keep their premiums, 41 get names, and the 4 that map
to nothing stay visibly unnamed.

**Recommendation: A.** The premiums in those 45 rows have never moved a shekel.
I measured it: `advertiser_rules ∩ daily = 0`, `rule_count: 0` on all 45,
`avg_effective_premium: 1.0` on all 45, and the rules engine's own honesty rule
means an unknown advertiser yields a premium of 1.0, so every lookup has always
missed silently. Re-keying orphans nothing that was ever live. Option B is
strictly better only if those 45 premiums are real commercial terms somebody
negotiated, in which case say so and I will do B.

**Blocked until you answer:** W0-3's final shape, and through it P3 (the break's
delivered money), P4 (Clients, and with it JS-9's money question) and P11
(pacing). W0-3 can start regardless: the columns and the resolver are the same
either way, and the 41 observed names bind under both options. Only the
disposition of the 45 synthetic rows waits.

---

## 2. The pod boundary: what makes one break one break

**The situation, measured on both files that could answer it.**

- In the daily file, grouping by `שעת התחלת ברייק` gives 10 groups of 1, 1, 3,
  3, 7, 28, 29, 30, 35 and 38 spots. The 38-spot group airs continuously from
  22:04:16 to 22:18:12, which is 836 seconds, or 13.93 minutes of unbroken
  commercial time.
- A 60 second gap rule does NOT reproduce those groups. Measured, it yields 1,
  3, 4, 7, 7, 22, 28, 30, 35 and 38, and two rows show why in both directions:
  the 21:22:12 group merges across a 22 second gap, and the 22:59:40 group
  splits at a 93 second internal gap while spanning 642 seconds against 432
  seconds of actual ad time. So the file's own declared grouping and any gap
  rule disagree, which is precisely why this decision cannot be inferred.
- I then checked whether `data/Spots.csv` settles it, because it carries a
  `break_id` column with 9,492 distinct values over 30 days. It does not.
  Within-break contiguity is real and method-independent: **0 of 15,614**
  consecutive within-break gaps exceed 60 seconds. But the boundaries between
  breaks are not: on second-resolution rows, **625 of 1,880 boundaries, 33.24
  percent, have a gap of 60 seconds or less**, and if the 1,145 rows that
  resolve only to the minute are recovered the figure is 1,667 of 3,025, or
  55.11 percent. Either way a gap rule does not reproduce `break_id`, and
  whatever rule produced it is not on disk. Both figures were measured three
  times by two independent readers; the method is stated here because the
  answer depends on it.

A 13.93-minute block is also above the 12 minutes per hour your own compliance
profile enforces, so calling it one break would put every plan in breach.

**Option A. An explicit break identifier per ad**, carried on the daily file as
a new column. Unambiguous, and it makes the traffic operator's surface exact.

**Option B. A splitting rule you state**, for example "a new break starts after
a gap of N seconds, or at a maximum of M seconds of ad time, whichever comes
first". I can implement any rule you can state in one sentence.

**Recommendation: A, with B as the fallback for historical data.** A costs your
traffic department one column on a file they already produce, and it is the only
option that is exact. B is derivable and cheap but it will disagree with what
the traffic department believes on some days, and disagreements about which
break an ad sat in are exactly the errors this product exists to catch.

**Blocked until you answer:** P10, the break contents and the traffic operator's
door, which is JS-7 and JS-8. P3 can still build the break entity, because the
break identity comes from the plan side (airing plus ordinal), not from the
daily file. Only filling a break with real historical ads waits.

---

## 3. The plan target: what is a week measured against

**The situation.** "Is this week on plan" is the first of the three answers JS-1
asks for and it is the only one the product cannot give. I searched
`/api/overview`'s 111 distinct keys for `goal`, `target`, `budget`, `on_plan`
and `variance` and the only hit was the unrelated `workspace`. There is no
budget, goal or quota entity anywhere in the data model.

I will not derive a target from the plan itself. That is circular: the plan
would always be exactly on plan.

**What I need from you, in one line:** the number, its unit, its grain and its
owner. For example, "₪9.5M of projected revenue per week for רשת 13, set by the
revenue owner each quarter". Revenue, GRP, ad minutes or breaks all work; I need
to know which one you actually manage against, and the threshold that separates
on plan from at risk from behind.

**Recommendation:** weekly projected revenue per channel, with a two-sided
threshold you set once, because it is the only quantity the plan already
computes at that grain and it needs no new measurement to be honest. If you
manage against GRP instead, say so; the machinery is the same and the store is a
column either way.

**Blocked until you answer:** P1's third answer. Until then Today ships an
honest empty state that names the missing input and offers the path to set it,
which is what section 9 item 2 of the spec commits to. It is not a placeholder
figure and it never will be.

---

## 4. A current week, and where delivery comes from

**The situation.** Nothing in this system represents now. `effective_date` is
2026-06-14, the saved plan covers 2024-11-01 to 2024-11-30, and the single daily
ad file is 2025-04-27. Three vintages on screen at once and none of them is
today. Every story that says "on air", "this week" or "tonight" stands on this.

`data/campaign_flights.csv` is header-only, zero rows, and
`GET /api/make-good-alerts` answers `data_available: false` with the reason
naming that file. The pacing math behind it is real, implemented and honest; it
has nothing to run on.

**What I need from you, three things:**

1. A current or near-future broadcast week: the EPG for it, so a plan can be
   about a week that has not happened yet.
2. Real campaign flights with start dates, end dates and delivery goals. The
   eleven-column contract already exists at `kairos_api/uploads.py:115-127` and
   the upload door is already built.
3. A delivery or as-run feed, so `delivered_to_date` updates instead of sitting
   static. Without it, pacing compares a goal against a number that never moves.

**Recommendation:** send 1 and 2 first, even without 3. With a current week and
real flights, the pacing board becomes honest for everything except live
delivery, and P11 can ship the goal, the forecast state and the make-good object
with delivery shown as unavailable rather than guessed. That is most of the
value and it does not wait on an integration.

**Blocked until you answer:** P11 entirely, JS-6, and the delivered half of the
money layer. The spec's section 3.4 states the limit explicitly rather than
papering over it: projected and delivered are never summed into one figure while
they cover non-overlapping dates.

---

## 5. Publishing, and who owns the regulatory limits

These are two questions and they have one shape: both are about which acts need
an authority that today's three roles cannot express.

**5a. What publishing means.** The word `publish` appears zero times in
`kairos_api/`, and the weekly plan is not among the nine logical files the
version store captures. So today there is no published state, no author, no
record of what superseded what. JS-2's done condition is "a named, dated plan
version is published with an author and a timestamp, and everyone downstream is
reading it". I need to know what "everyone downstream is reading it" means in
your operation: is publishing an internal freeze, or does it emit something to
somebody, and may a planner do it alone or does it need a second person.

**5b. Who owns the regulatory limits.** `max_ad_minutes_per_hour` 12.0,
`max_breaks_per_hour` 4, `min_break_spacing_minutes` 7 and
`protected_program_max_ad_minutes_per_hour` 8.0 are ordinary settings fields
today, editable through `PUT /api/settings` with exactly the same permission as
the revenue-weight slider. There is no approval, no effective-date workflow and
no alert when one changes. Your compliance owner is accountable for numbers
anybody can move without telling them.

**Recommendation on 5a:** publish is an internal freeze plus a named version,
performed by the planner alone, with the previous version one click away. That
is the smallest thing that satisfies JS-2 and it emits nothing, so it cannot be
wrong about a downstream system nobody has named yet. If publishing must notify
or export, tell me to whom and in what format and it becomes a second increment.

**Recommendation on 5b:** the guardrails move into their own store with an
effective date, a change record and a distinct permission, and I gate them on
`affiliation = company` as an interim owner until you name a real one.
`kairos_api/events_access.py` already implements exactly this kind of gate for
calendar events, so the mechanism is proven rather than new. If your compliance
owner should hold it instead, that is a fourth role and I will add it.

**Blocked until you answer:** P2's done condition for 5a, and the second half of
JS-14 for 5b. Both pieces can build everything else in the meantime; the
guardrail store lands either way and only its permission waits.

---

## 7. What does EB mean in your traffic file?

**One question, and it is only yours to answer.**

`סוג ברייק` in your Wally file takes two values. Counted on the shipped example,
`Wally_Prime_Reshet_Example_2025-04-27.csv`: **Regular 111 rows, EB 64 rows.**

The product prints `EB` on the pod board exactly as your file writes it, labelled
"from the file", because nothing in `media-domain-from-the-trade.md` says what it
stands for and inventing a Hebrew word would put a term on your screen that your
own vendor does not use.

**What I need:** the words a trader uses for each, in Hebrew and in English. If EB
is a real category with different commercial behaviour, say that too, because
then it is not only a label: a break type that prices or places differently is a
lever the optimizer does not currently know about.

**What happens until you answer:** nothing breaks. The board shows your own file's
word and says where it came from, which is honest and readable to anybody who
knows the file. It just is not yet the product's own vocabulary.

---

## 8. Three regulatory caps: ANSWERED by the owner, 2026-08-09

Raised and answered the same day. Kept here because the reasoning is worth
having, not because anything is still open.

**What a research agent found.** The Second Authority's 1992 placement rules,
`כללי הרשות השניה לטלויזיה ורדיו (שיבוץ תשדירי פרסומת בשידורי טלויזיה), תשנ"ב-1992`,
at https://www.nevo.co.il/law_html/law00/4941.htm, cap commercial time three
ways: clause 3(א), ten minutes in any hour; clause 3(ב)(1), forty minutes IN
TOTAL across 20:00 to 24:00; clause 3(ג), ten percent of the whole broadcast day.
Clause 10(ב) additionally allows a mid-programme cluster only at an אתנחתה.

Three Israeli regimes carry three different sets of numbers. The public
broadcaster runs nine minutes an hour and up to four pods; designated cable and
satellite channels run up to ten pods, eight when drama or film exceeds half the
hour. Our operator is in the Second Authority regime, so those two are context.

**What the product has today.** One of the three. `max_ad_minutes_per_hour` is a
setting at 12.0, not 10. The evening-window cap and the whole-day cap do not
exist anywhere: no field, so nothing is set wrongly, there is simply no way to
express them.

**THE OWNER'S RULING.** Do not turn these on by default, because the commercial
channels do not always work to the regulation. Build the technical ability to
turn them on when somebody asks.

**What that means, precisely, and it is a better answer than the question.** The
caps become constraints the product CAN express and does not apply. Off is the
default and off is honest: a cap that is not enforced must not appear as though
it were, and a plan produced without it must not carry a compliance badge earned
by a rule nobody ran. Three states, as everywhere else: enforced, available and
off, and absent.

It also settles what I would have got wrong on my own. I was preparing to ask
which numbers legally bind us, as though the answer were a value. The answer is
that the value is not the product's to hold: the operator decides whether a rule
applies, and the product's job is to be able to apply it and to say plainly
whether it did.

**Still not changed:** `max_ad_minutes_per_hour` stays at 12.0. The ruling is
about capability, not about moving a number that is already in use.

---

## 9. The hour past midnight, and a cap that cannot bind

Both found 2026-08-09 while checking whether the plan breaches its own hourly
limit. It does not. The check turned up two things that matter more.

**9a. Hour 24 exists and no rule was written for it.**

A break's hour is its seconds-from-midnight divided by 3600, and that division is
not bounded at 23. A programme starting at 23:xx carries breaks past midnight,
which come back as hour 24 or 25 of the SAME broadcast day, because the day does
not roll with them. Measured: 9,026 breaks on the shipped plan, 143 of them past
midnight, 30 on your own channel.

The day not rolling is defensible and probably right: a programme belongs to the
broadcast day it started in. But the hourly cap is then applied to a bucket the
cited regulation does not describe, since it caps commercial time "in any hour",
which reads as a clock hour. Bucket (day D, hour 24) and (day D+1, hour 0) are
the same sixty minutes of real time and two different keys, so nothing is ever
added up across them.

I changed nothing. Moving which bucket a break falls in moves the plan and
therefore money, and the source listing overlaps heavily around midnight, so the
excess the merge would reveal may be an artifact of the listing rather than real
concurrent airtime. It is measured, named in the code where the hour is defined,
and pinned by `tests/test_the_hour_past_midnight.py`.

**What I need:** does a programme that starts at 23:40 and runs to 00:30 belong,
for the purpose of an hourly limit, to the day it started in or to the clock?

**9b. The hourly minutes cap CANNOT BIND, and lowering it would change nothing.**

Every break in the plan is 120 seconds, and that length is a hardcoded constant
with no settings key. So `max_breaks_per_hour` at 4 gives 4 x 120s = 8 minutes as
a HARD CEILING, and `max_ad_minutes_per_hour` at 12 sits two thirds above a
ceiling nothing can reach. It would take 7 breaks in an hour before the minutes
cap could ever bite.

This changes the answer to decision 8. Setting the hourly limit to the ten
minutes the regulation cites WOULD CHANGE NOTHING AT ALL, because the plan cannot
reach even eight. The lever that actually shapes ad load is the break count, and
404 of 713 hours sit exactly at 4 breaks and exactly 8 minutes.

Two things follow that you should know:

  * The protected-content cap is 8 minutes, which is EXACTLY the ceiling. It has
    never been violated only because the test is a strict greater-than. One
    second lower and it would bind on those 404 hours.
  * The daily cap is 160 minutes and 28 of 30 days sit exactly on it, minimum
    146. There is no headroom. And 160 minutes is more than ten percent of a
    24-hour day, which is 144, so against a ten-percent rule every single day
    would be over.

**What I need:** whether the break length should be configurable at all. Today it
is one constant for every break on every channel, and it is the number that
silently decides what your hourly limit means.

**A correction to what I told you in decision 8.** I said the whole-day cap does
not exist anywhere. That was wrong. `max_daily_ad_minutes` exists, defaults to
160 and is enforced. What it is not is a percentage, and it is deliberately held
as sales policy rather than as a licence limit.

---

## 10. Seven questions for the person who works in this market

These came out of the broadcast research, 2026-08-09, and they are a different
kind of question from the ones above. Each one is a mechanism that is DOCUMENTED
AND REAL somewhere else, quoted from a regulator rather than from marketing, and
that has NO Israeli evidence either way. Under the research ruling they stop at
documentation. The only route by which any of them becomes real work is you or
Tal saying "yes, we do that".

They are listed because a concrete question gets a better answer than an open
one. "Does an Israeli deal ever contract a minimum weekly spread" is answerable.
"What else should we build" is not.

**First, the part that is NOT a question, because the research settled it.** The
goal-based order is not a bet on a market that might arrive. The UK regulator has
published the mechanics in detail: sales houses run an optimiser over aggregated
demand, re-optimise continually, and the sales house's optimisation "generally
overrides the vast majority of specific contractual terms". An entire national
market has traded this way for decades. Whatever else is uncertain, the direction
is not.

And one thing it CONFIRMED about our own naming: the international concept for
what a channel owes when it under-delivers is already in this product under the
name `deficit`, in the make-good ledger. Nothing is renamed.

**1. A contracted weekly spread.** Elsewhere a buyer can contract a minimum
spread of ratings per week, so the pacing curve is a TERM OF THE DEAL rather than
something we watch internally. This product has a pacing board and no contracted
curve. Does an Israeli deal ever fix one?

**2. A spot that cannot be moved.** Elsewhere there are clauses that forbid the
broadcaster from moving an advertisement out of a fixed slot. The trade document
says the ordered time here "is a range, not a commitment", which suggests we have
no such thing. Is any spot ever sold as immovable?

**3. Length factors, and whether they are linear.** Elsewhere a published table
converts a spot's duration to thirty-second equivalents and it is deliberately
NOT proportional: five seconds is priced at 0.300 of a thirty, not 0.167. This
product prices per second, which is linear by construction. Does the Israeli
trade apply any non-linear length factor?

**4. Withholding breaks.** The trade document says "the channel does not release
every break". Elsewhere that is a licence problem: the regulator explicitly
prohibits a dominant broadcaster from withholding airtime to push prices up. Is
there any Israeli restriction on this, or is holding inventory back entirely the
channel's business?

**5. How many audiences do you actually TRADE?** Elsewhere over a hundred
audiences are measured and only about twenty are traded. That number decides what
values a goal-based order's audience field may take, so it is a build input and
not trivia.

**6. Does the advertiser's real target get mapped to a tradeable one?** Elsewhere
a buyer picks the traded audience closest to the real target and accepts the
mismatch, and gets the spill for free. Do we need to model that mapping, or does
an Israeli order name the audience it is settled on directly?

**7. Is a gold break inside a goal-based order or outside it?** Elsewhere the
highest-rating programmes are carved out as "specials", priced at a premium and
traded spot by spot OUTSIDE the optimised pool. Our gold breaks carry a separate
rate card, which is the same shape. Are they also excluded from a goal order, or
can a goal be delivered partly out of gold inventory? This one changes the design
of the goal seam and it is the most load-bearing of the seven.

**Two things the research could NOT corroborate anywhere, so they rest on the
trade document alone and that is fine.** Deliberate over-ordering of prime as a
negotiating position, which no regulator or industry source describes; and the
one-to-two-day lead time, where the same foreign market books eight weeks ahead.
The second is a factor of thirty and it means no foreign assumption about a known
order book may be imported here. Our daily re-optimisation is right for Israel
BECAUSE of that difference, not in spite of it.

---

## 11. The plan is a function of machine load, and the fix moves money

Measured 2026-08-09, full write-up in `docs/audits/the-plan-is-not-reproducible.md`.

**What was found.** The exact DP tier aborts on a WALL CLOCK deadline, five
seconds per channel-day, so a group that finishes on an idle machine and times
out on a busy one adopts a DIFFERENT PLAN. The plan this product exports depends
on how loaded the computer was when it ran.

**The measurement.** Six real channel-days, varying nothing but that budget:
five of six produced a different plan, and the starved plan was WORSE every
single time, by between 2,318 and 122,886 shekels on a single day. The one day
that did not move is exactly the day where the DP had nothing better to offer,
so the exception predicts itself.

On an idle machine the worst real day uses 1.89 of its 5 seconds, so there is
about 2.6x headroom. A full export is 120 independent groups each with its own
budget, so any subset can flip on any run.

**This is almost certainly what has been rewriting your plan file.** Four times
in one session the committed artifact changed underneath us with the same
signature every time: identical rows, identical break count, identical total ad
seconds to the second, breaks redistributed inside a day, and revenue slightly
lower. That is exactly what this mechanism produces. I say almost certainly
because the load conditions at those moments are gone and cannot be re-measured.

**The fix, and why I did not make it.** Replace the wall clock with a
deterministic work budget. There is already a deterministic gate beside it
counting states, and it already bounds the compute; counting work instead of
seconds would make the same inputs give the same plan on any machine.

I did not make it because ON THE DAYS WHERE THE DP CURRENTLY TIMES OUT, THE PLAN
WOULD CHANGE, and it would change in the direction of more revenue. That is a
real money movement on my judgement rather than yours, and it is exactly the kind
of change this campaign has agreed I do not make alone.

**What I need:** whether to make the budget deterministic. My recommendation is
yes, and to measure the revenue difference across all 120 groups before and after
so you can see what it is worth rather than take my word for it.

**One thing that follows either way, and you should know it.** Until this is
fixed, a single green golden run is weaker evidence than it looks, because if the
golden exercises the DP near its budget then the golden is load-sensitive too.
Every measurement in this repository inherits that caveat.

**A smaller finding alongside it:** the budget is not settable by any caller. It
is a definition-time default, so it cannot be raised for a production export or
lowered for a fast preview without editing the module.

---

## 12. Your committed plan is out of date by 17,966.31 shekels

Measured 2026-08-09, and it corrects four things I told you today.

**What I got wrong.** Four times today `output/weekly_break_schedule.csv` changed
under us and four times I called it pollution and restored it from git. I was
restoring the OLD file over the correct one.

**The measurement that settled it.** I ran a fresh export to a scratch path,
touching nothing. It produced bytes IDENTICAL to the file I had been calling
polluted, and different from the one in git. Five independent occurrences now
agree: four rewrites plus one deliberate export.

**So the engine has been right every time and the committed plan is stale.**

**The drift, measured exactly:**

| | |
|---|---|
| committed plan revenue | 221,891,590.23 |
| what the current tree produces | 221,873,623.92 |
| difference | **-17,966.31** |
| on your own channel | -9,350.68 |
| segments with a different break count | 68 of 8,704 |
| total breaks | IDENTICAL, 9,026 both ways |

Same number of breaks, distributed differently, slightly less money.

**Why it drifted, and I was wrong about this too.** I assumed this session's own
engine work. It is not. I extracted the exact tree from the commit that COMMITTED
the plan and ran a full export inside it: it produced the SAME file today's tree
does. **The artifact committed in that commit was never what that commit's own
code produced.**

It has been carried forward by RESTORE rather than by EXPORT, for several
commits. Its content even appears twice in the file's own history with different
content in between, which is the signature of a restore rather than a rebuild.

That is possible only because THE GOLDEN AND THE ARTIFACT ARE DIFFERENT THINGS.
The golden asserts against its own embedded baseline, not against this file, so
it can be green while the shipped artifact matches nothing the engine produces.
It was, and it has been for longer than this session.

So the 17,966.31 is not the price of anything done today. It is the accumulated
distance between a file nobody rebuilt and the engine that was supposed to have
built it.

**What I need from you, and there are two questions.**

1. **Do I commit the fresh plan as the new baseline?** Right now the file the
   dashboard reads and the file in version control disagree, and every
   measurement anyone takes against the committed one is against a fiction. My
   recommendation is yes, because an artifact that does not match what the code
   produces is worse than a slightly different artifact. But it changes what your
   dashboard shows, so it is your call.

2. **Do you want the 17,966.31 attributed before I do?** It is 0.008 percent, and
   it is the price of three corrections that were each independently right. I can
   isolate which change costs what, one at a time, if it matters to you. It is a
   day's compute, not a day's argument.

**Related but separate: decision 11**, that the plan is not reproducible across
machines because the DP tier stops on a wall clock. That is a real defect and it
is NOT what happened here; the same bytes came back five times, which is the
opposite of a race.

---

## What I did not ask you

For completeness, so you can see the line I drew. I answered these myself from
the code and the data rather than sending them to you: how many of the sixteen
accountabilities are distinct people (the doors work whether one human holds
three of them or three humans hold one each), which surfaces merge, what the
vocabulary should be, whether the competitor lanes are a law breach (they are,
and the removal is flagged in the spec's section 10 because it removes something
visible today), whether to keep the Reports page (yes, Bar 3), whether to delete
the second upload system (no, it is the only ad-hoc spreadsheet path), and
whether `data/Spots.csv` is usable as money (no, its revenue column is a
synthetic price computed from a constant base rate of 50, verified on 99.67
percent of 50,386 rows).
