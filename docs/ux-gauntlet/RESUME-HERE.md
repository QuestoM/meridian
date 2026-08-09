# Resume here

Rewritten 2026-08-09 at the end of a very long session. If you are a new session
picking this up, read this file first and trust it over any summary you were
handed. Everything below was measured, not remembered.

---

## The one-line state

Waves two, three and six are closed. Four and five are in flight with agents
live. The suite went from 111 failures to 3 over the session, and every one of
the three was closed. Six frontend guards are green and all their quarantines are
empty.

## What is running RIGHT NOW, and do not duplicate it

Check `find . -newermt "-20 minutes" -not -path "./.git/*"` before touching
anything. Agents were live at the end of this session on:

- **Top and Tail** — `kairos/optimize/frequency.py`, `_frequency_rules.py`,
  `data/campaign_assets.csv`, `data/frequency_rules.csv` (adding `pair_lead`,
  `pair_closer`, `value_max`). It briefly made `load_frequency_rules`
  unimportable mid-edit, which broke the daily pricing pipeline and produced
  transient failures ANOTHER agent mis-attributed to itself. Expect that shape.
- **The goal-based order (wave five)** — `campaigns_api_store.py`,
  `campaigns_commitment.py`, a new `kairos/optimize/goal_seam.py`.
- **The mention picker (wave four, R1)** — `assistant_mentions.py`,
  `MentionPicker.jsx`, its own stylesheet.

## The rules this session paid for, in the order they cost the most

1. **A named gap is a defect CLASS, not a site.** Measured repeatedly: the accent
   bar was reported at one site and found at 23 across 11 files; the Latin `s`
   for seconds at one site and found at 8; the direction override at one and
   found at 68.
2. **A check whose failure mode is silence is not a check.** Met seven times in
   one day. The worst: `npm run test:smoke` died on an ENOENT from wave zero
   until yesterday, and it was the only thing banning native controls, so the
   tree drifted to 384 unseen.
3. **A number nobody re-derives is a number nobody checks.** Contract line counts
   were stale in 55 of 79 rows. My own dossier was refused by my own gate on five
   counts within a day of writing the gate.
4. **Measure before fixing.** Two queue items marked "serious, open" were already
   closed, one under the queue's own words "left open deliberately".
5. **A wave's size is unknown until its class is COUNTED**, and reading a backlog
   is not counting it. An 80-item backlog collapsed to a handful of real classes.
6. **Restore, then verify. Never trust a report over git.** An agent said a plan
   artifact's content was unchanged; git said 69 rows.
7. **A guard that counts literal substrings cannot tell code from a comment about
   code.** Two explanatory comments moved the counts they explained.
8. **A test that says what it measured and when is worth more than one that only
   says what it expects.** Twice this session a test failed because the product
   got BETTER, and both times the docstring made that legible.

## The four shared writable stores an agent has now polluted

Settings (twice), the override store, the agency layer. Each cost real money or
real time, and each guard was written after the fact and too narrow:

- `data/kairos_settings.json` — revenue_weight and min_retention_floor, then
  locale and direction. **15,844,833 ILS** and a declared licence breach.
- `data/manual_overrides.csv` — one gold mark. **131,878.70 ILS**, and it
  survived the settings restore because nothing guarded this file.
- `data/agencies.csv` + `agency_advertisers.csv` + `campaigns.csv` — twelve rows
  named "סוכנות ביקורת" at `critic.example`, with campaigns marked `is_demo`
  FALSE so seeded rows presented as real bookings. One pollution, five failing
  tests across three files, none of which named the cause.

Guards now: `tests/test_plan_artifact_fingerprint.py` (settings + an active-override
digest) and `tests/test_shared_stores_are_not_agent_leftovers.py` (the class).

**The gap that remains:** the fingerprint cannot catch an in-memory pollution,
because the exporter re-stamps the fingerprint in the same call, so the pair stays
consistent while the plan is wrong. What catches that is the golden's own
committed baseline hash.

## The six frontend guards, and every quarantine is empty

    npm run test:card       the card, its inset, off-scale padding
    npm run test:direction  isolation lives in shell/bidi.jsx and nowhere else
    npm run test:dates      a calendar day is read in shell/dates.js and nowhere else
    npm run test:accent     one-sided accent bars, AT ZERO, down only
    npm run test:smoke      native controls, at 350, down only
    node scripts/verify-card-rules.mjs

A directory is quarantined ONLY while an agent is actively holding it. Anything
longer is a budget with no number.

## What the owner ruled, and what still waits on him

**Ruled by the owner 2026-08-09:** the assistant runs the newest Opus
(`claude-opus-5`). That was his call because it spends his money.

**Ruled by me, and I should have ruled sooner:** ruling 009. The plan FILE keeps
every channel; the export ROUTE serves the operator's own. The two tests that
looked contradictory agreed about the file and differed about the route.

**Still waiting on the owner:**
1. `data/campaign_flights.csv` is header-only, owner decision 4.
2. What `EB` means in his traffic file. `סוג ברייק` takes Regular (111) and EB
   (64) on the shipped example, and nothing in the trade document says what EB
   is. Question 7 in `decisions-for-owner.md`, and the part that matters more
   than the label: whether EB prices or places differently.
3. The Cursor admin credential for cortex-lens.

## The adversarial re-audit, and its structural finding

`docs/audits/trade-reaudit-2026-08-09.md`. The finding that matters is a
mechanism, not a list:

> The transcript ends with its own nine-item summary. `trade-gap-analysis.md` has
> exactly nine sections in exactly that order. **It audited the summary, not the
> body.** Everything the summary did not lift became invisible.

Its top three, all confirmed:

1. **The preferred-position percentage is built, tested, bilingual and
   unreachable.** `preferred_position_rate` has zero callers outside its own
   test. The number the channel and the agency audit each other with cannot be
   computed by anyone using the product.
2. **Two contradictory answers to which positions are preferred.** CLOSED TODAY:
   the pod hardcoded the trade default and marked every ordinal preferred, while
   the pricing screen said a guessed percentage is worse than none. The pod reads
   the configured set now and answers UNKNOWN when it is unset.
3. **The rating currency is not the trade's currency.** The trade settles on
   Jewish households, quarter-hour, overnight plus one. The five modelled
   audiences do not include Jewish households, and nothing records whether a held
   TVR is overnight or consolidated. Every shekel figure is in a unit the market
   does not settle in.

## The engine, and the one number worth remembering

The last regression was **not a code bug**: one row in `data/manual_overrides.csv`
written by a browser walk. With it inert the engine reproduces the committed
golden byte-exact across all 120 channel-days.

Then the exact DP tier learned to plan WITH an operator's own overrides instead
of declining the whole channel-day. **Recovered 124,806.66 of 131,878.71.** The
residual 7,072.05 is real: the plan emits 3 gold breaks against a cap of 3.

**I proposed the wrong fix and the agent refused it with a runnable
counterexample.** I wanted to split the day at `_window_ends`; that closes the
three LOCAL guardrails and not the two DAILY ones, and the split doubles the
reported objective by shipping a plan that breaches the daily cap.

## What is open, ranked

1. **Wave five, the goal-based order.** Dossier at `dossiers/G1.md`. The headline:
   the goal is ALREADY modelled on 51 to 55 campaigns and reaches the optimizer
   ZERO times. Not "add a goal": carry it across one seam.
2. **Wave four R2**, the mention drill-down. Dossier at `dossiers/K2.md`.
3. **Top and Tail**, in flight, and the owner asked for full handling at every
   level.
4. **The rating currency**, re-audit finding 3. Probably the largest correctness
   gap left in the product.
5. **The preferred-position percentage**, built and unreachable.
6. **Kai's remaining coverage**: no propose tool records a remedy, so it can read
   a problem it cannot act on. Also the campaign record, what a restriction would
   cost, and the account-administrator persona with zero coverage.
7. **`kairos_api/advertisers.py::_write_frame`** writes `frame[COLUMNS]` and
   COLUMNS omits `data_source`, so one PUT erases provenance from all 45 rows.
   One-line fix, and a test already fails with that instruction.
8. **`campaigns_api_store.py:155-159`** documents "Version the campaigns store"
   and passes a logical name the version store does not know, so six callers are
   silent no-ops.

## Before writing any new wave script

`docs/ux-gauntlet/campaign-plan.md` now opens with an honest table of which of
its prescriptions are ENFORCED and which are only written down, plus a section on
what measurement did to its own ordering. `scripts/gauntlet/wave_preflight.py`
refuses a launch without a complete dossier and re-counts every line number in
it. `workflow-fixes.md` holds seven orchestration defects with the incident that
proved each.

Do not apply them by editing a running script: that invalidates the resume cache
for every agent in the run, which is the most expensive lesson this campaign has
paid for.
