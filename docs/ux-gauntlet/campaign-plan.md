# The campaign, everything open, and how the next wave is built

Written 2026-08-08. The owner asked why wave two is still wasteful, what would
let an agent finish correctly on its first round, and for a list of everything
still to do including the waves not yet run. This is that.

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
   **KNOWN GAP IN THAT RESEARCH:** it studied the terminal products, where there
   are no icons and no drill-down. The owner meant the GRAPHICAL typing
   experience: a floating panel on `@`, a list that narrows as you type, an icon
   per kind, mouse drill-down into a container, and `@` alone showing categories.
   That surface has not been studied and the icon and drill-down questions are
   therefore still open.
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

## D. Standing, not a wave

- Check `data/kairos_settings.json` and the plan fingerprint before every commit.
  Both are guarded now; the guard is not a reason to stop looking.
- Never `git add -A` while agents are live. Commit by explicit path list.
- Record what is known broken in the commit message, so a known defect never
  becomes a hidden one.
