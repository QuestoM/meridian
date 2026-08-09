# What every critic needs, discovered once

Written 2026-08-10, before the wave-one judging round. It exists because nine
critics would otherwise each rediscover the same five things, and three of them
would have stopped dead at the first.

**Do not re-derive anything on this page. Measure everything else.**

---

## 1. The app is running and you must not restart it

    API   http://127.0.0.1:8000
    UI    http://127.0.0.1:3000

Started by the lead. Do NOT start or kill anything on 8000, 3000 or 8010.

**Auth is disabled** via `KAIROS_AUTH_DISABLED=1`, which is the documented escape
hatch at `kairos_api/auth.py:53`. Without it the UI shows a login wall and no
critic gets past it. Nobody entered a credential and nobody should.

The app renders an open-access badge because of this. **That is deliberate for
this run and is NOT a Bar 4 violation.** Do not report it.

## 2. The four bars, and which you can actually reach

Verbatim at `docs/ux-gauntlet-prompt.md`, lines 110 to 200. In short:

- **Bar 1, the stopwatch.** The job gets done, measured in a real browser against
  the running app. **REACHABLE. This is your main instrument.**
- **Bar 2, named references.** A blind A/B against a reference product.
  **PROBABLY NOT REACHABLE**: opening another product's interactive session is
  out of scope. The protocol's own instruction is that a critic who cannot reach
  a reference SAYS SO and uses the stopwatch instead. Say so; do not invent one.
- **Bar 3, the three-way.** Today's Meridian, the new one, and the reference.
  Running the old version is expensive and probably out of reach. **The half you
  CAN do is the regression half:** does anything that worked before work worse
  now. Do that half and name the half you did not.
- **Bar 4, the laws.** **REACHABLE.** Honest math with no fabricated figure;
  tri-state real, unavailable, unknown; the competitor boundary, where the
  operator owns exactly ONE channel read from settings and no rival name or
  figure appears on an operator surface; Hebrew first; Sunday-first weeks;
  right-to-left.

## 3. The rail, so nobody has to find the pages

Seventeen destinations, from `tv-break-dashboard/src/shell/nav.js`:

    Overview · Optimizer · Schedule · Inventory · Break Library · Campaigns
    Forecasts · Reports · Data · Advertisers · Agencies · Overrides
    Assistant · Versions · Settings · Calendar · Pricing

## 4. Which piece owns what

| piece | contract | subject |
|---|---|---|
| P1 | no contract file | Today — **UNDER FIRST JUDGEMENT**, see below |
| P2 | `contracts/P2.md` | Plan, the week — **UNDER FIRST JUDGEMENT**, see below |
| P3 | `contracts/P3.md` | Plan: the day and the break |
| P4 | `contracts/P4.md` | Clients |
| P5 | `contracts/P5.md` | Rules |
| P6 | `contracts/P6.md` | Sources |
| P7 | `contracts/P7.md` | The model console — **PASSED**, do not re-judge |
| P8 | `contracts/P8.md` | History |
| P9 | `contracts/P9.md` | Kai |
| P10 | no contract file | the pod board |
| P11 | `contracts/P11.md` | Pacing and make-good |
| P12 | `contracts/P12.md` | Model improvement |

Each piece also has `state/<piece>.json` and some have `state/<piece>-critic.json`.
**Read those for the OPEN GAPS ONLY.** You are blind to the builder's claims of
success: a claim is not evidence, and you re-measure anything you rely on.

**AND EVERY ONE OF THOSE FILES IS A HISTORY, NOT A STATE.** Measured 2026-08-10:
every `<piece>-critic.json` is dated 2026-08-07, and 103 commits touching 663
files, 320 of them under `kairos/`, `kairos_api/` or `tv-break-dashboard/src/`,
have landed since the newest of them was written. Two were class fixes that
closed named blockers outright.

This already cost a wrong verdict. P4's critic file names a hard-coded `unknown`
literal in `CampaignFlights.jsx`; the literal is not in that file, the eleven-site
class fix shipped `tests/test_p4_delivery_on_screen.py`, and that file's last test
restores the literal and asserts the defect returns, so the other twelve are not
passing vacuously. The verdict was one publish away from failing a piece for a
defect fixed three days earlier.

So an inherited finding is a LEAD, and it is only a finding again once you have
reproduced it today. Say which of the two you did. **A finding you cannot
reproduce is a closed finding, and reporting it closed is a real result rather
than an empty round.** P10, P11 and P12 have no critic file at all and nothing
to inherit.

**P1 AND P2 WERE RECORDED AS PASSED AND THE CLAIM HAS NO ARTIFACT.** This table
said so until 2026-08-10 and it was wrong. There is no `state/P1.json`, no
`state/P2.json`, no P1 contract, and P2's contract carries no verdict; neither
file was ever committed and no state file has ever been deleted, so the record
does not exist and was not lost. The claim traces to a commit where the lead
asserted it while correcting a different error. Both pieces ARE built — 17 files
under `src/today/`, 46 under `src/plan/week/`, 19 test files between them — so
they are built and unjudged, exactly like P3 to P9. Both are now under their
first blind judgement. **The standard the campaign holds the product to is that a
claim is not evidence, and it does not get an exemption when the claim is ours.**

## 5. Known state, so nobody reports it as a finding

- **`output/weekly_break_schedule.csv` differs from its committed version.** It is
  STALENESS, not corruption: the artifact has not been rebuilt since before
  2026-07-07, and a fresh export reproduces the on-disk file exactly. Owner
  decision 12. Do not restore it, do not re-export, do not report it.
- **Six frontend guards are green** with empty quarantines: card, direction,
  dates, accent at 0, colour at 67, native controls at 350. `npm run test:all`
  runs the build first. A guard being green is not a verdict on a piece.
- **The suite is 3,969 passing** with one deliberate failure, the artifact guard
  above.

## 6. How to measure, from the workbench's own rules

`README.md` in this directory carries six. The three that cost the most today:

1. **Measure it, do not argue it.** Run the thing, count the rows, read the
   rendered geometry.
2. **Run the counter-check and say it came back empty.** An investigation that
   only confirms has not been tested.
3. **Attribute against a COMMIT, never the working tree.**

And one this session earned four separate times:

**A probe that cannot fail loudly is not a probe.** Four measurements today
returned comfortable answers because the thing being tested was not connected: a
fixture that did not bind, a knob patched at the wrong binding, a grep with a
malformed flag, and a harness measuring against a poisoned baseline. **Before you
believe a null result, prove the instrument moves.**

## 7. What a critic does not do

Does not build. Does not fix. Does not publish to `state.json` — report to the
lead and the lead publishes. Read-only on `data/`, `output/` and `models/`. No
git write commands. Never invent a term, a number or a screen. Never report a
measurement you did not take.

The interpreter is `~/.venvs/meridian/bin/python` and no other.
