# Completion re-audit, 2026-08-10

## Scope and result

This was a cold re-audit of the complete gauntlet, not an acceptance of the
previous completion note. It covered wave 0 and P1 through P13 on the current
tree. The code waves are complete. P1 through P12 pass. P13's implementation
passes and its live state remains owner-blocked because the repository contains
neither a real media QC report nor owner-approved playout standards.

No test wrote to `data/`, `config/` or `models/`. The two modified weekly
schedule output files predated this audit and remain deliberately unstaged.

## Executed gates

| Scope | Result | Notes |
|---|---:|---|
| Wave 0 | 183 passed | One stale assertion expected two genre tables after P1 had correctly reduced them to one. The guard now requires delegation to the shared table and forbids a local copy. |
| P1 to P4 | 739 passed, 3 skipped | Re-run after the P4/P11 locale seam was repaired. |
| P5 to P8 | 593 passed, 1 skipped | Run with Chrome and localhost access. The sandbox-only Chrome and socket denials did not reproduce. |
| P9 to P13 | 667 passed, 2 skipped | One continuous regression run across the dependent wave-two pieces. |
| Total disjoint gauntlet collection executed | 2,182 passed, 6 skipped | The four rows above use disjoint test prefixes. |

The focused P13 contract also passed 18 tests, and the combined P10/P13 focus
passed 66 tests before the broad run. Dashboard build and guards are recorded by
the final `npm run test:all` gate in the completion commit.

## Piece verdicts

| Piece | Re-audit verdict | Evidence of the previously open seam |
|---|---|---|
| W0-1 to W0-5 | passed | `tests/test_w0_*.py`; frozen router, shell, identity, wall, vocabulary, evaluation and cache seams remain intact. |
| P1 | passed | `tests/test_p1_genre_localization.py`; all programme genres use the one shared translation table. |
| P2 | passed | `tests/test_p2_plan_versions.py`, `tests/test_p2_plan_week_surface.py`, `tests/test_p2_run_control.py`; a collapsed plan needs explicit confirmation in both server and surface, and a zero-break run is not reported as routine success. |
| P3 | passed | `tests/test_p3_gold_settlement.py`, `tests/test_p3_surface_readout.py`, `tests/test_p3_day_board.py`; unsaved and stale edits are refused explicitly and negative zero is normalised. |
| P4 | passed | `tests/test_p4_onboarding_refuses_cleanly.py`, `tests/test_p4_makegood_reason.py`; malformed onboarding returns 422 and the empty make-good reason follows the reader's locale. |
| P5 | passed | `tests/test_p5_constraint_draft_gate.py`; an incomplete or zero-match draft cannot be saved and a later edit invalidates its preview. |
| P6 | passed | `tests/test_p6_sources.py`; every engine-read input is named and read-state is distinct from a pool that yields usable items. Cache tests were split without changing their contract. |
| P7 | passed | Full `tests/test_p7_*.py` browser and localhost run; candidate, verdict, gate and training-watch paths remain green. |
| P8 | passed | `tests/test_p8_history_modules.py`; `airtime_caps` has a translated real, partial or not-set rendering and never becomes an object string. |
| P9 | passed | `tests/test_p9_immediate_first_token.py`; the grounded local first line precedes any provider call while apply and restore regressions remain green. |
| P10 | passed | `tests/test_p10_surface_closure.py`; shown order is the recorded order, preferred-position state reaches the surface and pair status is recomputed from the shown order. |
| P11 | passed | `tests/test_p11_regression.py`; thresholds are visible before disclosure, and the P4/P11 empty projection seam now carries both language reasons. |
| P12 | passed | `tests/test_p12_browser_evidence.py`; browser and command-line verdicts preserve the same complete common-basis evidence and the adoption guard discriminates in both directions. |
| P13 | implementation passed, live owner-blocked | `tests/test_p13_media_verdict.py`; House Number is the join key, all eight QC fact families are represented, import is atomic and a measured failure blocks lock in both API and UI. Live verification awaits real owner inputs. |

## Findings from this re-audit

Four findings were acted on:

1. P13 had two material defects in the earlier implementation: it joined on the
   copy version rather than House Number, and its `blocks_lock` value did not
   reach the lock endpoint. Both were fixed and pushed before this audit.
2. P4's empty make-good state carried one English-only `reason`. The API now
   publishes `reason_en` and `reason_he`, retains `reason` for compatibility,
   and the surface selects the locale pair.
3. P6 had a 466-line test file. Its cache contract moved to a separate test file
   without production changes. Under the owner's later ruling, future existing
   files below 500 lines do not warrant work solely for size.
4. A wave-zero guard still required the duplicate genre table removed by P1.
   The guard now protects the stronger invariant: one table and one delegating
   helper.

No other material code defect reproduced. Deprecation and parser warnings were
left as non-blocking maintenance signals because the exercised contracts passed
and the compatibility aliases were separately counter-probed.

## External completion boundary

P13 is the final piece of wave 2. Its software path is complete, but a real
asset cannot honestly be marked verified until an owner supplies both:

1. a QC, ingest, MAM or transcoder report keyed by House Number;
2. approved playout rules and approval-state vocabulary in
   `config/media_standards.json`.

The importer and unavailable state are the completed product behavior while
those inputs are absent. Inventing a seed row or a standard would turn the last
external dependency into a false pass and is explicitly outside completion.
