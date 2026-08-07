# The sweep protocol

How a critic disposes of what it finds, so that a piece is not failed for a
label while a real defect is discarded.

Written 2026-08-07 after a wave in which six of seven pieces capped at four
rounds without passing. Every rule below is a response to something measured,
and the measurement is named.

## The axis is NOT importance

The obvious split is major against minor, and it is wrong. The owner has cared
intensely about things a naive reading would call cosmetic: a rule down one edge
of a card, three buttons at three sizes, a number pulled to the wrong side of a
Hebrew column. Those are not unimportant. Calling them minor and deferring them
on that basis would be exactly backwards.

**The axis is whether closing it needs the piece's own context.**

- **BLOCKING.** Only the builder who holds this piece can close it, because
  closing it requires knowing what this surface is for, what its payload
  carries, and what its job story is. It fails the piece.
- **SWEEP.** A cross-cutting agent closes it better than the piece's builder
  would, because the defect is an instance of a class that appears across many
  surfaces, and fixing one site is the failure mode this campaign has paid for
  six times. It does NOT fail the piece.

Measured support for that split: the accent-bar defect was reported at one site
and was present at twenty-six across seventeen files. The direction override was
reported in one table and is present in over five hundred places. Neither was
fixable by the piece that surfaced it, and both were closed correctly by one
agent working the class.

## What is always blocking

These fail a piece regardless of how small the instance looks, because each one
makes the product lie or lose money:

- A number on a screen that was not computed from real data.
- A figure without its basis, or a percentage that is a floor presented as a
  total.
- A rival channel name or figure on an operator surface.
- A training act reachable from an operator surface.
- A job story that cannot be completed, or one that got slower.
- A regression in a capability that worked before.
- A dead end where a named object should open.
- A claim that is false on the payload rendering it.

## What is sweep

Real defects, none of them acceptable, all of them better closed as a class:

- A design rule broken: an accent bar, a state colour, a button group, a
  separator, spacing.
- A direction or bidi override where isolation was needed.
- Copy that is frozen in one language at the point of production.
- A raw internal key, an enum value or a repr reaching a screen.
- A test named after one instance of a rule it was written to protect.
- A vocabulary word that drifts from the agreed one.

## Where it goes

`docs/ux-gauntlet/state/sweep-backlog.jsonl`. One JSON object per line, appended
and never rewritten.

**Append-only JSONL, not a JSON file, and that is deliberate.** Several critics
write concurrently. A shared JSON document means read, modify, write, and the
last writer erases the others. One line appended per finding cannot clobber a
neighbour. This session lost work to concurrent writes more than once; this is
the fix.

Each entry carries exactly what the sweep agent needs so it never re-derives
anything. A whole builder round was spent tonight rediscovering what a critic
had already measured, and three of seven builders in one round produced no code
at all for that reason.

```json
{"class":"one-sided-accent-bar","what":"a 3px rule down the inline-start edge of the risk card","where":"tv-break-dashboard/src/clients/pacing/pacing-row.css:18","rule":"design-rules.md section 1","symptom":"reads as an unfinished frame, and inverts under RTL","evidence":"docs/ux-gauntlet/evidence/p11-c1-01-board-he.png","found_by":"P11","round":1,"locale":"he"}
```

- `class` is the join key. The sweep agent groups by it and fixes a class in one
  pass, never a site at a time.
- `where` is file and line. Not a description of where.
- `rule` names the line in `design-rules.md` that was broken. If no rule covers
  it, say `none` and the sweep agent's first job is to write one.
- `symptom` is what the reader experiences, not what the code does.
- `evidence` is a path, and it is what makes the fix checkable without rerunning
  the measurement.

## How the wave ends

1. Every piece finishes on blocking findings only. A piece with fourteen sweep
   entries against it still passes.
2. One sweep agent reads the backlog, groups by `class`, and closes each class
   across every surface in one pass. It is a direct agent, not a workflow
   branch, because a workflow branch that stalls is indistinguishable from one
   that hit an API limit, and four retries were burned on that distinction.
3. For every class it closes it must **widen a guard to the class and prove the
   guard fails on the old behaviour first.** A rule that is not tested is not a
   rule, and every test that let a defect survive four rounds was named after
   the single instance it protected.
4. One judge, on the sweep diff only. Not on the pieces, which have already
   passed. It reads the backlog and the diff and answers three questions: is
   each class genuinely closed everywhere, did anything break, and does each new
   guard actually fail on the old behaviour.

## Calibrating the critic

A critic that never passes is as useless as one that always does. Say so in the
brief, and give it the disposal route so it has somewhere to put a real finding
that is not worth a round. Without that route it faces a false choice between
failing a piece over a label and discarding the label, and this campaign proved
it will discard: one critic recorded three findings in a field it named "not the
gap but worth the next round" and nothing ever read that field.

The queue is the point. A sweep is the expensive part and it is already paid for
by the time the critic notices the fourth thing.
