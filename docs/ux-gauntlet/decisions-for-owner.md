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

## 8. Three regulatory caps, one of which you already have and two of which do not exist here

Raised 2026-08-09, out of the broadcast research. I am asking rather than acting
because every one of these numbers moves real money and none of them is mine to
set.

**What a research agent found, and its source.** The Second Authority's 1992
placement rules, `כללי הרשות השניה לטלויזיה ורדיו (שיבוץ תשדירי פרסומת בשידורי
טלויזיה), תשנ"ב-1992`, at https://www.nevo.co.il/law_html/law00/4941.htm, cap
commercial time three separate ways:

  * clause 3(א), ten minutes in any hour;
  * clause 3(ב)(1), forty minutes IN TOTAL across 20:00 to 24:00;
  * clause 3(ג), ten percent of the whole broadcast day.

**What this product has today.** One of the three. `max_ad_minutes_per_hour` is
a setting and it is set to **12.0**, not 10. The evening-window cap and the
whole-day cap DO NOT EXIST anywhere in the engine, the settings or the
compliance read: there is no field, so there is nothing set wrongly, there is
simply no way to express them.

**What I am NOT claiming.** That your plan breaches anything. I have not
established that, the exact measurement is in flight, and my first crude attempt
used a unit the guardrail may not use. I am also not claiming the 1992 text is
current: it may have been amended since, your licence may carry different terms,
and one web page read by an agent is not a legal opinion. That is precisely why
this is a question and not a change.

**What I need from you.**

1. What are the numbers that actually bind you? If the hourly limit is ten and
   not twelve, that is not a settings tweak: it is a constraint that reshapes
   every plan the optimizer produces, and it should be measured before it is
   moved.
2. Do the evening-window and whole-day caps bind you? If they do, they are two
   constraints the engine cannot currently express at all, which is a build and
   not a setting.
3. Is `אתנחתה` a real constraint in your operation? Clause 10(ב) says a
   mid-programme cluster may be placed ONLY at a natural pause. If your traffic
   department treats that as binding, it is a placement rule the optimizer does
   not know about.

**Related and already asked: decision 5b**, which is about WHO may change these
numbers. This one is about WHAT they should be. Answering 5b without this one
protects the wrong values well.

**What happens until you answer:** nothing changes. `max_ad_minutes_per_hour`
stays at 12.0 because lowering it on the strength of a web citation would move
real money on my judgement rather than yours, and the two missing caps stay
missing rather than being invented at values nobody confirmed.

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
