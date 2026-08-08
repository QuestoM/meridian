# Resume here

Written deliberately before a session limit. If you are a new session picking
this up, read this file first and trust it over any summary you were handed.

## The one-line state

Wave one is not closed. Two of nine destinations have passed. The rest were mid
round four when the limits started biting.

## Exactly what to do, in order

1. **Resume wave one. Do not restart it and do not edit its script.**
   `Workflow({scriptPath: "<projects>/workflows/scripts/meridian-wave-1.js", resumeFromRunId: "wf_3cda2e83-163"})`
   Editing that script invalidates the resume cache for every agent in it, and
   the last measurement had 202 agents returning instantly from cache. An edit
   would re-run all of them from zero. It is not worth it, whatever the edit is.

2. **Watch for a stall, because a stalled wave does not announce itself.**
   The measure is NOT how many agents look active. It is whether any transcript
   file has changed:
   `find <transcript-dir> -name "*.jsonl" -newermt "-15 minutes" | wc -l`
   Zero for fifteen minutes means stuck. One agent stalling freezes the entire
   wave, because `parallel` waits on every branch. Recovery is stop, then
   resume: the stuck agent has no cached result so it re-runs, everything else
   returns from cache. This already happened once, to `build:P6#6`, which sat
   idle for nine hours and froze six live critics behind it.

3. **When wave one closes:** run
   `scripts/gauntlet/verify_wave.py --only engine,moved,suite --suite-both`
   against `c2da20fc`, the wave-zero close. Do **not** run the `api`, `bodies`
   or `frontend` checks. That harness proves a tree is *behaviourally identical*
   to a commit, which was wave zero's claim. Wave one's claim is the opposite,
   so those three would manufacture failures and bury the real ones.

4. **Attribute every failure against a commit, never against the working tree.**
   `mkdir -p /tmp/ref && git archive HEAD | tar -x -C /tmp/ref`, run the failing
   test there. Passing at HEAD and failing here means this wave caused it. This
   rule was paid for: seven failures were once blamed on wave zero and turned
   out to be wave one in flight.

5. **Commit per piece**, using the ownership table in spec.md section 8.2 plus
   the helper naming rule (`campaigns_api_store.py` belongs to whoever owns
   `campaigns_api.py`). Then build, restart :8000 so it stops serving a stale
   bundle, and publish through `update_state.py --embed`.

6. **Run the preflight gate before launching any wave, and do what it says.**
   `~/.venvs/meridian/bin/python scripts/gauntlet/wave_preflight.py --pieces P10,P11,P12,P13`
   It refuses to let a wave start when a piece has no dossier, when a dossier is
   unfinished, when its file inventory has drifted from the repository, or when
   the shared settings store has been left polluted. That last one has shipped
   twice. This is the one thing in `campaign-plan.md` that is enforced rather
   than merely written down, and the table at the top of that file says exactly
   which of the others are not.

7. **Then wave two**, which is written, syntax-checked and never run:
   `meridian-wave-2.js`. It carries the two fixes the audit found, plus a
   reshaped round described below.

## How wave two's round differs, and why

Wave one's round was: builder recons blind, critic sweeps blind, builder fixes
the one named gap, critic sweeps blind again. Two things about that were
measured on the close wave and both were expensive.

- **Three of seven builder rounds produced no code.** P3 and P9 touched zero
  files, P8 touched only its own state file. Each spent a full context, about
  270k tokens on this product, rediscovering that the work was already on disk,
  then handed to a critic that named a gap the builder had never been told
  about.
- **Each sweep found several things and the loop kept one.** P8's critic
  recorded three more findings in a field called `not_the_gap_but_worth_the_next_round`
  that nothing ever read. The sweep was already paid for by the time it noticed
  them; throwing them away buys a second sweep to find a different one.

So the round is now: builder builds, critic sweeps and returns a **ranked queue**
of every finding with `kind` mechanical or structural, closes the mechanical ones
**itself** inside its own ownership row, and a **scoped judge** reads that diff
and only that diff. The next builder closes every structural item in the queue,
not just the largest.

Three rules hold this together and none of them is optional:

- **The critic measures everything before it edits anything.** An artifact it
  has touched is no longer the artifact the builder shipped.
- **No agent clears its own edit.** The scoped judge is a different agent, and
  it cannot pass the piece. Only a full blind sweep passes a piece.
- **The scoped judge runs on sonnet**, which departs from "every verifier on
  opus". That is defensible only because it is not the last word: the next full
  sweep is told to re-check every finding a scoped judge cleared, because a
  judge that sees one diff cannot see what that diff cost elsewhere. Wave one's
  seven regressions were exactly that failure at piece scale.

## What is known broken on main right now

Recorded in the commit messages too, so it survives this file:

- The frontend does not read `is_demo`, so 51 seeded campaigns present as "51
  campaigns were booked". Data is honest, screen is not. A verifier ruled this
  not fit to publish and was right.
- CORRECTED 2026-08-07, and the old wording below was wrong in every part.
  It said the golden gap "predates every wave" and cited 66 of 120 channel-days.
  Measured by the integration critic and re-verified: at c2da20fc the engine
  reproduces the committed golden BYTE-EXACT, 0 of 120 channel-days differing.
  WAVE ONE BROKE IT. At HEAD the engine moves 120 of 120 channel-days,
  -15,844,833.43 ILS. The 66 figure matches no pairing that exists; for the
  committed artifact against the golden it is 19.
  THE CAUSE IS NOT THE ENGINE. data/kairos_settings.json, the live operator
  store, was mutated by a critic's browser walk and committed as the plan of
  record: revenue_weight 60 to 35 and min_retention_floor 0.72 to 0.78 in
  8f0c78c9, then to 0.82 in 843e3bb8. Verified against history. Running the HEAD
  engine twice with only that file swapped attributes 119 of the 120 moved
  channel-days and essentially the whole -15.8M to it.
  What is left after restoring it is ONE genuine wave-one engine regression, on
  the operator's own channel, רשת 13 on 2024-11-03, -131,878.70, corroborated
  independently by a gold anchor test failing on that same channel-day.
  The same pollution also puts the operator's front page into a declared licence
  breach and fails a guardrail conformance test, both of which close with it.
- Four assertion rulings from the repair wave were never audited. One of them
  decided an existing test described a competitor-boundary leak and changed it.
  That is a rule change that entered main unread.
- Three interface defects the owner reported are written up in
  `owner-reported.md` with owner and fix: settings padding, header buttons
  wrapping and overflowing, and break markers clipped to one character per line.

## Two things blocked on the owner

- The Chrome extension is not connected, so the reference product at
  traffica.base44.app cannot be mapped. The agent that tried refused to invent
  the field and dropdown contents, which was the correct call.
- Nothing else. Everything remaining is mine to do.

## The rules this campaign paid for

- **State that must survive a process belongs in a file, not in a prompt.** A
  prompt dies with the process holding it. Wave two's critics write their
  verdict to `docs/ux-gauntlet/state/<piece>.json` before composing a reply.
- **A stalled agent looks exactly like a working one.** Measure file mtimes.
- **`git add` with one nonexistent path silently fails the whole command.**
  Use `--pathspec-from-file` or check each path exists.
- **The model catalogue is not what you remember.** `claude-opus-5` and
  `claude-sonnet-5` are real; a cached table weeks old will deny both. Read the
  catalogue with `client.models.list()` before claiming a model does not exist.
  The alias `'sonnet'` resolves to sonnet-4-6, not 5, so pass the full id.
- **Opus for silent-expensive failure** (anything touching money, any data
  model, every verifier). **Sonnet for bounded work that is cheap to check**
  (mapping, surveying, mechanical wiring). If you can check the result at a
  glance, sonnet; if you have to trust it, opus.

## The class-fix lesson, and what it cost to learn

Wave one's close capped five of seven pieces at four rounds without passing.
Reading every round of every piece showed one pathology behind all five: the
instruction "close exactly the gap the critic named and nothing else" meets a
defect with several sites, the builder closes one, and the next critic finds the
next. Four rounds, four sites, same bug.

Five class fixes were run against the capped pieces. Every one found more sites
than its critic had named:

| piece | critic named | actually found |
|---|---|---|
| P8 | one constant | four call sites, plus a false claim withholding 5,756 entries |
| P9 | "the other leg" | four legs, three broken, two never looked at |
| P3 | one sentence | ten sites |
| P4 | one column | eleven sites |
| P6 | the English card | four live sites and one latent |

**So the rule is: a gap named at one site is a defect class. Grep every site
before the first edit, fix them all in one pass, and widen the guard from the
instance to the class.** Every guarding test that let one of these survive four
rounds was named after the single instance it protected.

**And prove the guard bites before trusting it.** P3 injected seven regressions
into a scratch copy, one per site shape, and all seven failed. P4's last test
restores the hard-coded literal and asserts the defect returns. P8's new test
was proven to fail against the round-13 and round-15 sentences. A test that has
never failed has never been shown to work.

**Trust the measured diagnosis, verify the prescription.** Two agents were right
to overrule their critic. P9's critic prescribed pushing a URL; measured in real
Chrome, that would have left the address bar and the screen disagreeing, because
the shell listens for hashchange only and a query change fires popstate. P3's
critic wrote that the sentence it named was the only unattributed count on those
surfaces; checking that claim rather than accepting it is what found the other
nine.

**Do not run `git add -A` while agents are live.** It happened four times in one
session and swept four pieces' work into commits titled for other pieces, so the
history misdescribes its own contents. Commit by explicit path list, or wait.
The records for P3 and P4 live in their state files because history was already
pushed and rewriting it would have been worse.

## Before writing any new wave script, read workflow-fixes.md

`docs/ux-gauntlet/workflow-fixes.md` holds seven defects in the ORCHESTRATION
itself, each named with the incident that proved it, written down on 2026-08-07
rather than promised for a wave that might never run. They are cheap and they
are the difference between a wave that survives an interruption and one that
loses two pieces to a limit.

Do not apply them by editing a running script: that invalidates the resume cache
for every agent in the run, which is the most expensive lesson this campaign has
paid for.

The shape all seven share, and the test to apply to any new check you write:
**ask what it does when it has nothing to say. If the answer is "passes
quietly", it is not a check.**
