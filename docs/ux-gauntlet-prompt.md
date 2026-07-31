# The Meridian experience gauntlet: the prompt

Paste everything below the line into a fresh Claude Code session at the repo
root. It is one instruction. The word ultracode on the first line is
deliberate: it authorizes multi-agent orchestration for the whole run.

---

ultracode

# Rebuild the Meridian operator experience until it beats the bar

## Who you are on this run

You systematically under-estimate yourself, and on this run that error would be
expensive. You are not a developer working a ticket. You are a system that can
hold an entire product, its users, its data and its code in mind at once, run
dozens of investigations in parallel, and rebuild an interface faster than a
design team can schedule its kickoff. Two days of your work here equals a human
product team's quarter. Correct for this upward, deliberately, every time you
catch yourself scoping down.

The bottleneck on this run is never your thinking. It is external: the owner's
review cycles, the real data on disk, wall-clock on test suites and rebuilds.
Optimize the whole plan around those, never around fitting your own work into a
comfortable size. Design and planning cost you nothing, so never ship the
scoped version of a plan to save effort. Plan the complete thing, then choose
what runs first. The brief below is a floor, not a ceiling: if you see a better
product than the one described here, build that and say why.

The valve that makes this safe: maximal confidence in what you build, paired
with unsparing honesty about what you measured. Every claim of "better" on this
run is a measurement or it is nothing. A screenshot, a stopwatch, a click
count, a passing test, a critic's blind verdict. Never round in your own favor.

## The goal

Meridian is a TV ad-break operations system: it decides where commercial breaks
go in a broadcast week, what they are worth, and who fills them. The engine
underneath is measured, honest and well tested. The experience on top of it grew
by accretion, one page at a time, and it now serves seventeen navigation entries
to people whose actual jobs it was never designed around.

Rebuild that experience. Re-specify what this product is for, discover who
actually lives in it and what each of them does when they sit down in the
morning, then rebuild the information architecture, the flows, the screens and
the AI integration until a person doing that job would rather use Meridian than
the best tool they have ever used for any part of their day.

You are not polishing pixels and you are not starting a new product from zero.
You are re-deciding what this thing should be for the people who depend on it,
and then making that true in the code.

## This is not a user interface project

Do not read the goal as screens. Read it as the whole product. Whatever a job
story needs in order to be real, build it: new endpoints, new stores, new
entities, a data model that fits the work instead of the engine's internals,
integrations with whatever these people already use, permissions, background
jobs, performance work, exports, notifications, the model itself. If the
honest way to serve a job is a capability that does not exist anywhere in this
repository, that capability is in scope and you build it.

Everything on every surface is reachable. If a screen names a thing, that name
opens the thing. If it shows a number, the number opens the rows behind it. If
it reports a state, the state opens what caused it. No text that describes
something the reader cannot then go to, no dead ends, no engine jargon anywhere
a non-engineer can see it.

## The line that must never blur: training versus runs

Two completely different activities live in this system today and the product
treats them as if they were one. Separating them is a first-class deliverable.

Training is ours. We are the startup: we fit the retention coefficients, we run
the held-out gates that decide whether a factor is real, we rebuild the audience
model, we judge data coverage and drift, we decide what the model is allowed to
believe. It happens rarely, it is a research act, and it is company staff only.
Accounts affiliated with a channel must never see it as an option, let alone
reach it.

Runs are theirs. The broadcaster's people compute the weekly plan, recompute
after a change, produce the daily schedule, forecast the coming week, publish.
It happens constantly, it is operational, and it must feel routine and safe.

Nobody, on either side, may ever wonder which of the two a button does. Give
each its own surfaces, its own vocabulary, its own permissions and its own
dashboards. The operator's dashboards answer whether this week is on plan and
what needs a decision. Our dashboards answer whether the model is healthy: what
each gate decided and why, how much contrast the data carries, what drifted,
what a rebuild would change, what is blocked on data we do not have yet. Both
are real deliverables and both must be genuinely comfortable to live in.

## The model is in scope, and it must be the best one this data can support

The engine is honest today. That is the floor, not the finish. Keep improving
the retention model, the audience model and the optimizer wherever measurement
supports it, always through the same discipline: a factor enters only when a
held-out gate says it earned its place, a verdict is recorded with its reason,
and nothing is ever asserted into a coefficient because it sounds right. Where
the current data cannot decide something, say exactly that and say what data
would decide it. Quality here means measured skill, not more parameters.

## The bar

The bar is the most important part of this run. "Make it beautiful" and
"production quality" are not bars. These four are, and every critic uses them
against real artifacts, never against a builder's description.

### Bar 1: the stopwatch (the job gets done, measurably)

During discovery you will freeze a set of job stories. Each one names a person,
a task and a target you can measure in a real browser against the running app.
These ten are the seed; complete the set from what you discover, and once frozen
they do not move.

1. A general manager opens Meridian and within five seconds, with zero clicks,
   knows whether this week is on plan, whether anything is broken, and what
   needs a decision today.
2. A planner builds next week's break plan: sets the objective, runs the
   optimizer, compares two scenarios on revenue net of retention cost, and
   publishes. Under three minutes, with every number carrying the basis it was
   computed on.
3. A scheduler places breaks in a real day: moves one, sees the retention cost
   and revenue move as it lands, pins a gold break, and respects a constraint
   without reading documentation. Direct manipulation, no dialog for the common
   move, undo always available.
4. A programming representative registers a restriction in their own words, for
   example no breaks in the last eight minutes of a season finale, and sees
   exactly which breaks that would move and what it costs, before saving. Under
   thirty seconds, no engine jargon anywhere on the path.
5. An account manager onboards an agency, an advertiser under it, and a
   campaign with flights, rebate terms and a Saturday-only surcharge discount.
   One flow, no duplicate entity created anywhere, everything visible after.
6. A campaign manager sees every campaign on air, its pacing against goal, what
   is under-delivering and what to do about it. The recommended action is on
   the screen, not derived by the reader.
7. A traffic operator assembles a real break: picks the ads, sees the pod as
   the physical thing it is with durations summing exactly to the break length,
   with per-ad verification of duration to the frame, format, aspect ratio and
   the presence of audio, reorders by dragging, and locks it. Under ninety
   seconds for a seven-ad pod, with any failing ad impossible to miss.
8. An analyst answers "which advertiser delivered the most last month, gross
   and net of agency rebates" without exporting anything.
9. Anyone, from any screen, asks Kai in natural Hebrew and it acts: it knows
   what they are looking at, shows exactly what will change before changing it,
   and the change is reversible.
10. A person on their first day completes job story 5 with no training, no
    documentation and nobody to ask.

Critics measure these in a browser on the running app: seconds elapsed, clicks,
keystrokes, screens traversed, dead ends hit, times the person had to guess.
Numbers, not impressions.

### Bar 2: named references (blind A/B against the best in the world)

For each surface, a critic opens the reference product, captures it, and does a
blind A/B against a capture of ours. What to take from each:

- Linear: speed, keyboard-first control, the command palette, density without
  clutter, the feeling that the interface never makes you wait.
- Stripe Dashboard: money legibility, every figure carrying its basis, honest
  empty states, drilling from a number to the rows behind it.
- Google Ads: campaign and flight hierarchy, pacing against goal, surfacing
  under-delivery before the human notices it.
- A professional editing timeline (Premiere, Resolve, Descript): the break pod
  as a real timeline with exact durations, not a list of rows.
- Frame.io: per-asset technical status at a glance, so a bad file is obvious
  before it airs.
- Figma: direct manipulation, selection, drag, undo, and the absence of modal
  dialogs for things that should be done by moving something.
- The best in-product AI you can find: an agent embedded in the surface it
  acts on, with preview before action and undo after.

A critic that cannot reach a reference says so and uses the stopwatch bar
instead. It never invents a comparison it did not run.

### Bar 3: the three-way (never regress what already works)

This is a rebuild of a living system, so every critic compares three things,
not two: today's Meridian, the new Meridian, and the reference. If today's
version wins on any job story, that is a gap and it goes back to the builder
with the reason. Nothing that works today may get worse.

### Bar 4: the laws (non-negotiable, checked every round)

- Honest math. No number appears on any screen that was not computed from real
  data. Missing capability is an honest empty state with a path forward, never
  a placeholder figure. Tri-state honesty everywhere: real, unavailable,
  unknown, never a confident guess.
- The competitor boundary. The operator owns exactly one channel, read from
  settings. No rival channel's name or data reaches an operator surface or the
  assistant's context.
- The engine's measured numbers are frozen. Revenue, retention, pricing and
  rating figures must be byte-identical unless you prove a change is a genuine
  fix, declare it, and show the measurement.
- The Hebrew vocabulary is canonical and consistent: ברייק, ברייקים, נעיצה,
  ברייקי זהב, רצועת שידור, הכנסה צפויה, עלות שימור, מפעיל. Never משתמש.
- Israeli week: it starts Sunday and ends Saturday, the weekend is Friday and
  Saturday only. Data stays ISO-keyed, presentation is Sunday-first.
- RTL is correct everywhere: logical properties, bidirectional isolation on
  numbers and ranges, tooltips and drawers on the side that reads naturally.
- No em-dashes, no emojis, no exclamation marks in product copy. Sentence case.
  One display string per source line, never hard-wrapped across lines.
- No source file over 450 lines. Split rather than compress.
- The full test suite stays green. It is currently 1438 passing, 4 skipped.
- Nothing on any surface is a dead end. Every entity name, number, status and
  badge opens what it refers to.
- Training and runs are never confusable. A critic checks, on every surface,
  that an operator cannot reach a training action and cannot mistake a run for
  one.

## Before you build: discovery

Do not open an editor until you can answer these from evidence in the
repository and the running system, not from assumption:

1. Who are the people? The owner named some: someone who only wants to see
   data, the schedulers who build the placements, the programming
   representatives who have objections about when breaks may not be placed, the
   account managers who enter agencies, clients and campaigns, the traffic
   operators who assemble the pod out of individual ads, the planners. He said
   explicitly that he has not named them all. Find the rest from the data, the
   endpoints, the stores and the roles that already exist, and say which ones
   you inferred and how.
2. What does each of them do, in order, on a normal day and on their worst day?
   Write it as job stories with a trigger, a sequence and a done condition.
3. What exists today, measured: every navigation entry, every page, every
   endpoint, every store. Which are load-bearing, which are duplicates of each
   other under different names, which are dead, which are half-built.
4. What is missing entirely? Some of what the owner described has no
   implementation at all. Say so plainly rather than pretending a nearby page
   covers it.
5. What is the information architecture that would fall out of the jobs rather
   than out of the engine's internal structure? Propose it, and justify every
   consolidation and every split against a job story.

The specification is itself a deliverable and it goes through the gauntlet like
everything else: a critic A/B's it against how a great product's structure
reads, and against whether a person could find their job in it in five seconds.

## How to run

Decide the decomposition yourself. Break the work into the smallest pieces that
can be improved and judged independently, and fan out builders only where the
pieces are genuinely independent. Do not follow a decomposition I did not give
you and do not assume the seventeen existing navigation entries are the units.

Every piece gets its own builder and its own critic, and they are different
agents. A critic starts with fresh context, never sees the builder's reasoning,
and inspects the real artifact: the rendered page in a browser, the running
flow, the actual pixels, the measured seconds. A critic that reads a summary
and grades it has failed its job. A builder never grades its own work.

When the critic's blind comparison picks something other than ours, it names
the single largest remaining gap in concrete terms and sends only that back to
the builder. Then the piece runs another round. There is no round limit. Do not
do three rounds and stop.

A piece is finished when the blind comparison stops picking the reference and
stops picking today's Meridian, or when three consecutive rounds fail to close
the named gap, in which case stop that piece, write down exactly what is
blocking it and why, and move on. Escalate to the owner only what genuinely
needs a human decision: a business rule you cannot derive, a destructive
action, a number that would move.

When the pieces are done, run one fresh integration critic over the whole
product: consistency of vocabulary and interaction across every surface,
correctness end to end, and whether the assembled thing actually serves the
jobs it was decomposed from. Then fix what it finds.

## Boundaries

- Do not deploy, purchase, publish, message anyone, use credentials, or take
  any irreversible action without explicit approval.
- Do not delete an existing capability until you have proven it is dead or
  fully replaced, and said so.
- Do not fabricate data to make a screen look finished. Ever.
- Commit in logical commits with honest messages as you go. Push to origin
  main when a coherent piece is done and its tests are green.

## The workbench

Maintain a live progress page at docs/ux-gauntlet/workbench.html that the owner
can open at any time without interrupting you. After every round it records,
per piece: what changed, the evidence (screenshot paths, measured seconds,
click counts, test results), the critic's verdict, the largest remaining gap,
the next action, and anything you dropped or deferred with the reason. It shows
the shape of the run at a glance and it never contains a claim you did not
measure.

## What is on disk today (measured just now, verify rather than trust)

- Frontend: 52 components, 18,164 lines, of which TVBreakDashboard.jsx alone is
  6,236 lines, roughly a third of the entire interface in one file. It carries
  the routing, the navigation, and several whole pages inline.
- Navigation: seventeen entries. Overview, Optimizer, Schedule, Inventory,
  Break Library, Campaigns, Forecasts, Calendar, Reports, Data, Advertisers,
  Agencies, Pricing, Overrides, Assistant, Versions, Settings.
- API: 51 modules, 111 endpoints.
- The engine is real and measured: a retention model with held-out gates, an
  audience model predicting expected rating with eight gated factor families,
  quarter-hour billing, an optimizer with a local-search refiner, agency and
  advertiser pricing layers, a calendar of operator events and Israeli
  holidays, versioning with restore points.
- Kai, the assistant, is docked on every page, knows which record is open, and
  can propose changes through an approval and restore-point path.
- Campaign flights exist as data with pacing signals. Per-ad media
  verification, the assembled pod as a visible object, and the traffic
  operator's day appear to have no implementation at all. Verify this before
  building on it.
- Known debt beyond the monolith: weekday arrays inside TVBreakDashboard.jsx
  are still Monday-first, against the Israeli week law.

## The people (seed, deliberately incomplete)

Someone who only wants to look at data. The schedulers who build the
placements. The programming representatives who hold the objections about when
breaks may not be placed. The account managers who enter the agencies, the
clients and the campaigns. Whoever runs the campaigns that are on air right
now, with whatever they need to connect to. The traffic operators who build the
pod out of individual ads and must verify that every ad is the right format,
the right aspect ratio, has audio, and is exactly the number of seconds it
claims. The planners. And the ones nobody listed, whom you will find.

And one more, on our side of the line: whoever at the startup judges whether
the model is fit to ship, watches the gates, the drift and the data coverage,
and decides when a rebuild is worth running.

## Last

Everything written here is a floor. It is not everything that is needed, and it
is not everything the owner knows he needs, and he said so plainly. Find the
rest. Where you see a better product than the one described, build that one and
say why you did.

Start with discovery. Report the frozen job stories and the proposed
architecture on the workbench before the first builder starts, then run the
loop and keep going.
