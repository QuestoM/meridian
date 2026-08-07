# Workflow fixes owed, each paid for by a measured failure

Written 2026-08-07 rather than promised for a future wave, because a fix that
lives only in a plan is a fix that does not exist. Every item below is a defect
in the orchestration itself, not in any piece, and each one is named with the
incident that proved it.

Apply these when writing the next wave script. Do NOT edit a running script to
apply them: editing invalidates the resume cache for every agent in the run, and
that lesson cost this campaign more than any item on this list.

## 1. The stall detector cannot tell an API limit from a stuck agent

**What happened.** A builder was reported as "stalled on all 6 attempts, no
progress for 180000ms each". Its transcripts showed normal recon: reading files,
running greps, nothing pathological. What actually happened is that a session
limit refused every call, which produces exactly the same signature as a stall.
Four retries were spent on a condition no retry could fix, and the piece
consumed 964,800 tokens and 200 tool calls to produce nothing.

**The fix.** Distinguish the two before retrying. An API error carrying a limit
or quota signal is not a stall and must not consume a retry; it should back off
until the limit lifts, or fail the branch immediately and cleanly so the lead
can resume later. Only genuine silence, where calls succeed but no progress is
made, should count against the retry budget.

## 2. A dependent branch dies with its dependency, for a reason that is not its own

**What happened.** P13 was chained behind P10 because P13's frontend path lives
inside P10's tree, and two agents writing one tree collide. That reasoning was
correct. The consequence was not: when P10 failed for reasons entirely unrelated
to P13, P13 never ran at all and the wave lost two pieces instead of one.

**The fix.** The dependency is on P10 RELEASING THE TREE, not on P10 SUCCEEDING.
If the dependency fails, the dependent should still run against whatever is on
disk, which the disk-first rule already tells it to read. Express the chain as a
mutex over a path, not as a success precondition.

## 3. The integration critic needs every piece, so one dead branch silences it

**What happened.** The integration critic never ran, because it waits on all
pieces and one branch had died. The wave therefore closed with no cross-piece
judgment at all, which is the one judgment no individual critic can provide.

**The fix.** It should run on whatever returned, and say explicitly which pieces
it could not see. A partial integration verdict that names its own gaps is worth
far more than no verdict.

## 4. The loop counter lives in the process; the history lives on disk

**What happened.** The machine restarted mid-wave. The durable state survived
exactly as designed: P11's file carried four rounds of history and P12's carried
six, and the builders read them and continued rather than restarting. But the
loop's own round counter is an in-process integer, so it restarted at one. The
prompt said ROUND 1 while the state file said round five. Measured on the resume:
15 of 36 started agents had a cached result, so 21 re-ran.

**The fix.** The loop should read its starting round from the piece's state file,
not from a fresh counter. The information is already on disk; nothing new needs
recording. This is a one-line change and it is the cheapest item here.

## 5. A builder can burn a whole context without writing a line, and nothing notices

**What happened.** The same failed piece read for fifty-four minutes across 200
tool calls and never reached an edit. No check anywhere in the loop asks whether
a builder has produced anything.

**The fix.** Count edits. If a builder has made N tool calls with zero writes,
say it out loud in the prompt or fail it early. The direct agent that later built
the same piece did it in 78 calls, and the only difference in its brief was an
explicit instruction that recon is not the job.

## 6. Exact command forms, not named interpreters

**What happened.** Four of five attempts at one piece burned calls on
`ModuleNotFoundError: No module named 'pandas'`, because the brief named the
interpreter as `~/.venvs/meridian/bin/python` and agents reach for bare `python3`
out of habit.

**The fix.** Put the literal command lines in the brief, in full, ready to paste.
Naming the tool is not the same as showing the command.

## 7. Nothing guards a shared writable artifact against a concurrent agent

**What happened.** `output/weekly_break_schedule.csv`, the plan of record, was
silently overwritten twice with a stale copy from a temp mirror. Both times it
was caught only because a person hashed the file before committing.
`data/kairos_settings.json` was polluted twice as well, once with economics that
moved 15,844,833 ILS and once with a locale that would have shipped a Hebrew
right-to-left product booting in English.

**Fixed on 2026-08-07** by a committed fingerprint and
`tests/test_plan_artifact_fingerprint.py`, proven to fail on both real incidents
before being trusted. Recorded here because the general rule still stands: any
shared writable file an agent may touch needs a guard that fails, not a habit
that remembers. The settings file still has no such guard.

## The shape all seven share

Every one is a check whose failure mode is silence. A stall that looks like a
limit. A branch that dies quietly with its neighbour. A critic that simply does
not run. A counter that resets without saying so. A builder that produces
nothing. A guard that answers "unknown". This repository met the same shape three
more times in one day outside the orchestration: a poller that ran thirty times
reporting success over zero work, a smoke test reading two deleted files, and a
freshness sidecar excluded from version control.

**When adding any check, ask what it does when it has nothing to say. If the
answer is "passes quietly", it is not a check.**
