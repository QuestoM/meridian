# The campaign, everything open, and how the next wave is built

Written 2026-08-08. The owner asked why wave two is still wasteful, what would
let an agent finish correctly on its first round, and for a list of everything
still to do including the waves not yet run. This is that.

## What in here is enforced, and what is only written down

The owner then asked the better question: what makes a document like this
actually get used? Nothing, on its own. A document whose failure mode is being
unread is the same shape as a check whose failure mode is silence. It reports
nothing, so it looks like it is working.

So one prescription in this file is now a gate rather than a paragraph, and the
rest are honestly marked as not.

| Prescription | Enforced by | What happens if you ignore it |
|---|---|---|
| A dossier per piece, complete, before the wave (2.1) | `scripts/gauntlet/wave_preflight.py` | The launch is refused, per piece, by name |
| The dossier's file inventory is current | the same gate, which re-counts every file | Refused, naming the file and both counts |
| `data/kairos_settings.json` is unpolluted at launch | the same gate | Refused before a wave writes on top of it |
| Evidence carries forward instead of re-measuring (2.2) | nothing yet | It quietly does not happen |
| Blocking versus sweep (2.3) | `sweep-protocol.md`, by convention | A critic keeps one finding of four again |
| The seven orchestration fixes (2.4) | nothing yet, they are script edits | The next wave repeats a paid-for defect |

The gate is proven to bite: `tests/test_wave_preflight.py` constructs each shape
of a dossier that would have wasted a round, including one whose line counts have
drifted, and asserts the refusal. Run it before launching anything:

    ~/.venvs/meridian/bin/python scripts/gauntlet/wave_preflight.py --pieces P10,P11,P12,P13

---

# Part one: why a round is wasteful, measured

Not opinion. Every number here was measured on this campaign.

| Waste | Measured | Cost |
|---|---|---|
| A builder round that produces no code | 3 of 7 in one wave-one round | ~270k tokens each |
| A builder that reads and never writes | 200 tool calls, 54 minutes, 0 edits | 964.8k tokens across retries |
| The same piece, built directly with a tight brief | 78 tool calls | the whole piece |
| A critic finding several things and the loop keeping one | 4 found, 1 kept | a second full sweep to refind |
| Retries against a session limit | 4 of 5 attempts | a limit no retry can fix |
| Wrong interpreter in a brief | 4 of 5 attempts hit it | calls burned on `ModuleNotFoundError` |

**The single biggest lever is the first round.** A builder starting cold spends
its whole context discovering what a critic will later tell it in one paragraph.

## What actually made a piece succeed

P10 failed five workflow attempts and produced nothing, then one direct agent
built it in 78 calls. Only three things differed in the brief:

1. **The diagnosis was handed over, not re-derived.**
2. **The exact command lines were pasted in**, not the interpreter's name.
3. **"Recon is not the job. If you are thirty calls in with no edit, stop
   reading and start writing."**

That is the whole difference, and it is cheap to reproduce.

---

# Part two: how the next wave is built

## 2.1 A dossier per piece, written BEFORE the wave

The round-one builder should never discover anything a previous round already
knew. Write one dossier per piece first, and hand it in the prompt as a path:

- The job stories it serves, in full, with their done conditions.
- The baseline numbers from `discovery/06-baseline.md` for those stories.
- **A file inventory with current line counts**, so it knows what exists and
  what is near the 450 cap before it opens anything.
- The API surface it owns: routes, payload shapes, what each field means.
- The reference product for its blind A/B, and what specifically to compare.
- The trade facts from `media-domain-from-the-trade.md` that bind it, quoted.
- **What is already built**, from its contract and its state file.
- The exact command lines: interpreter, test invocation, build, its own port.

A dossier is cheap to write once and is read by every round of that piece. The
alternative is paying for the same discovery four times.

## 2.2 Evidence carries forward; measurement does not repeat

A critic that measured the stopwatch on round one should hand that number
forward. Round two's critic re-measures only what the round-two diff could have
moved. Today every critic re-measures everything, which is a full sweep per
round per piece.

The state file already carries a queue. Extend it to carry the bar evidence, so
a later critic reads "stopwatch 42ms, measured round 1, unchanged files" rather
than driving a browser again.

## 2.3 Blocking versus sweep, which is already written

`sweep-protocol.md` holds this and it worked the first time it ran: P10's critic
returned 19 findings, closed 6 mechanical ones itself, and handed over 13 ranked.
Compare wave one: 4 found, 1 kept. Keep it, and keep the axis, which is NOT
importance but whether closing it needs the piece's own context.

## 2.4 The seven orchestration fixes

In `workflow-fixes.md`, each with the incident that proved it. The cheapest and
most valuable: the loop must read its starting round from the state file rather
than a fresh counter, because a machine restart resets the counter while the
history survives on disk.

## 2.5 The rule to apply to every new check

Ask what it does when it has nothing to say. If it passes quietly, it is not a
check. Met six times in one day: a poller green over zero work, a smoke test
reading deleted files, a freshness sidecar excluded from git, a stall detector
that cannot see a limit, a guard scoped to the incident rather than the risk, and
a critic with nowhere to put a small finding.

---

# Part three: everything open

## A. In flight now

- **Wave two**: P10, P11, P12, P13, then the integration critic. P10 is on
  builder round two against a 19-item queue with two blocking findings.
- **The card inset sweep**: one home for card padding, a rule for content inside
  a card, and a ratcheting guard. 330 hardcoded padding values against 417 token
  uses, and no card primitive exists.

## B. Owed and specified, not started

1. **The one real engine regression.** רשת 13 on 2024-11-03, -131,878.70, the
   single channel-day left after the settings restore closed 119 of 120.
   Corroborated by a gold anchor test failing on that same day.
2. **The sweep backlog**, `state/sweep-backlog.jsonl`: three accent bars in
   frozen directories, the pacing separator, the pacing button group, and the
   direction sites inside frozen trees.
3. **Kai's coverage gap**, and this is the largest product gap open. 31 read
   tools and 8 propose tools, and everything built in the last two waves is
   invisible to all of them. Worst: the prompt gives Kai the words for the pod
   and the spot while no tool can read them, and the guardrail permission
   refuses proposals over four keys no tool can list.
4. **The mention system**, build order R1 to R5 in
   `audits/kai-mentions-and-coverage-2026-08-07.md`. R1 is the smallest
   shippable: `@` over the four kinds that already resolve.
   **CORRECTED 2026-08-09, and the earlier note here was too harsh.** It said
   the graphical surface had not been studied and the icon and drill-down
   questions were open. Re-read: Part four of that audit IS the design for the
   graphical surface. It specifies the trigger, one glyph per KIND taken from the
   rail's own icons so the glyph is navigational identity rather than decoration,
   a server-side candidate index because a client-side one would put rival rows
   in the browser, three named boundary rules, Hebrew prefix stripping that
   neither reference product has, and the drill-down BOTH reference products
   verifiably declined to build, with the leading-edge arrow resolved from
   documentDirection rather than hardcoded.
   What is genuinely missing is narrower and the audit says so itself: no
   screenshots, because taking one required opening an interactive session of
   another product, which it was told not to do. So every rendering claim in it
   is sourced to code and the visual gaps are listed rather than guessed.
   That is a good document with one hole, not a document that answered the wrong
   question.
5. **The eight remaining trade gaps**, `trade-gap-analysis.md`. The goal-based
   order is the thesis of the product and the largest single piece of work left.
6. **Three owner-reported interface defects** in `owner-reported.md`.
7. **Four assertion rulings** audited and written into `rulings.md`; ruling 009,
   whether the schedule export may carry rival channels, is escalated and waits
   on the owner. One test is deliberately red until he rules.
8. **cortex-lens**, three fixes ranked by the cost of being wrong: the resolver
   that books unknown vendors to Anthropic, the two budget paths that price an
   unknown model at zero, and health that means delivered rather than ran. Plus
   the Cursor admin credential, which only the owner can create.

## C. The waves, in the order they should run

## What actually happened, 2026-08-09, against the order below

The order below was written on 2026-08-08 and it is kept because the reasoning in
it still holds. What follows is what MEASUREMENT did to it, which is the more
useful document.

**Wave six finished first.** The remaining engine regression was ranked last
because it looked like a hard optimizer bug. It was a DATA restore: one row in
`data/manual_overrides.csv`, written by an agent's browser walk on 2026-08-01,
the same walk that polluted the settings. With it inert the engine reproduces the
committed golden byte-exact across all 120 channel-days. Ranking work by how hard
it LOOKS is how the cheapest thing on the list stays last.

**Wave three shrank on contact.** The direction class was 68 violations and the
sweep closed it to zero, but the accent bar the owner reported had never been
COUNTED at all: 23 across 11 files. The card quarantine hid only 9. A backlog of
80 items collapsed to a handful of real classes once each was measured instead of
being read.

**Wave four split honestly.** Coverage before mentions was right, and coverage
turned out to be five gaps rather than one. The pod landed; the break itself, the
pacing it can name a remedy for but not read, the plan version and the
account-administrator persona are still open. The mention system has not started
and its research still answers the wrong surface, which is recorded below.

**Wave five has a dossier now**, `dossiers/G1.md`, and writing it moved the piece
before any code was written. The headline measurement: the goal is ALREADY
modelled, on 51 to 55 campaigns, and reaches the pacing board, the delivery
ledger and the commitment check. It reaches the optimizer ZERO times. So the
piece is not "add a goal", it is "carry it across the seam and settle against
it", which is a different and much smaller thing than the roadmap assumed.

**The rule this session earned.** A wave's size is unknown until its class is
counted, and reading a backlog is not counting it. Four items on that backlog
were already closed, three more were one measurement each, and the two that
looked smallest, the accent bar and the seconds mark, were the two that turned
out to be classes spanning eleven and eight files.

---

**Wave three, the sweep wave.** Close the backlog by class, not by site. Each
class gets a widened guard proven to fail on the old behaviour. This is where
the owner's reported defects land, because every one of them turned out to be a
class rather than a site.

**Wave four, Kai.** Close the coverage gap first, since a mention system over
tools that cannot read the object is worth nothing. Then R1 of the mention
design. Study the graphical mention surface before designing its icons.

**Wave five, the goal-based order.** The trade's own destination: the channel
takes placement against a target-audience goal instead of a spot list. This
changes a data model rather than a screen and deserves its own wave.

**Wave six, the engine.** The remaining regression, and whatever the golden test
still disagrees about after it.

## Closure update, 2026-08-10

All four post-gauntlet waves are now closed in code. Wave 6 had already finished
through the data restore and 120-day golden reproduction. Wave 5 is now wired at
the shared day core: real goal orders affect both demand ranking and the scalar
used by greedy, F1 and DP, while the demo-only store leaves the shipped plan
byte-identical. Wave 4 now includes the full scoped campaign record in addition
to the mention, drill-down, break, pod, pacing, restriction-cost and account
coverage. Wave 3's four owner-reported classes have focused guards, and the
dashboard build plus card, direction and accent guards pass.

The measured commands and the remaining external P13 dependency are recorded in
`audits/remaining-waves-completion-2026-08-10.md`.

## D. Standing, not a wave

- Check `data/kairos_settings.json` and the plan fingerprint before every commit.
  Both are guarded now; the guard is not a reason to stop looking.
- Never `git add -A` while agents are live. Commit by explicit path list.
- Record what is known broken in the commit message, so a known defect never
  becomes a hidden one.
