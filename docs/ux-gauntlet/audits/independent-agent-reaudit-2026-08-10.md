# Independent agent re-audit and repair wave

Measured on `main` on 2026-08-10 after the first waves 3 through 6 completion
claim. Three independent read-only agents were asked to disprove that claim from
separate directions. They did. The first verdict was PASS for waves 5 and 6,
PARTIAL for waves 3 and 4, and FAIL for the broader statement that no code work
remained. The findings were repaired, then rechecked by the agents that found
them.

## Findings that changed the verdict

1. The exact DP used a five-second wall-clock deadline. Identical inputs could
   therefore keep the exact result on an idle machine and fall back to a
   different greedy result under load.
2. A pytest recompute had written the shipped schedule and fingerprint. The
   resulting plan differed on 18 of 120 channel-days and by 17,966.31 ILS while
   keeping the same 9,026 breaks. A new in-memory run returned the committed
   plan, proving that the dirty pair was test output from the load-sensitive
   path, not a user edit.
3. The wave 3 tests asserted CSS text but did not measure overflow. The narrow
   timeline marker also did not implement the owner contract to hide partial
   text.
4. Kai still lacked real plan-version and model-adoption reads. The documented
   `day -> programme -> break` drill stopped at programme.
5. `/api/parameters` published rival channel names, and a missing affiliation
   normalized to company instead of failing closed.
6. The spot-ledger layer was frozen without a published contract.
7. Quarter-hour settlement had math and an activation flag but no provenance
   fields proving that `baseline_tvr` was Jewish-household, overnight+1 data.

## Repairs and independent results

| Area | Repair | Verification |
|---|---|---|
| Deterministic DP | Replaced the wall-clock decision with deterministic state and transition budgets. Elapsed time is telemetry only. | 69 focused DP tests passed. The depth-13 real day used 40,485 peak states and 328,877 of 5,000,000 work units. All 120 channel-days reproduced hash `1b9d4298...` and aggregate hash `4d77669c...`. |
| Plan safety | pytest now forces `KAIROS_PLAN_READONLY=1`; successful recompute tests redirect writes to `tmp_path`. The golden also compares its rebuilt bytes directly with the shipped CSV and fingerprint. | 44 focused export/guard tests passed; the product export returned 8,704 rows, 9,026 breaks and the committed `1b9d4298...` hash. `output/` is clean. |
| Wave 3 | The header becomes yielding rows below 1500px. Timeline chips keep their real duration; text disappears below the measured readable width while tooltip and aria-label remain. | Real Chrome layout passed at 1600, 1500, 1280, 1000, 861, 860 and 700px. The 35-test wave group, dashboard build and all design guards passed. |
| Wave 4 | Added scoped reads for live/frozen plan versions and company-walled model candidates/adoption. Completed `day -> programme -> break` with contextual drill IDs and send-time scoped resolution. | 176 focused tests and a 61-test follow-up passed. Registry is 43 READ and 9 PROPOSE tools. Rival and typo absence remain indistinguishable. |
| Boundary | `/api/parameters` returns only the operator channel. Missing or malformed affiliation becomes an unresolved, fail-closed state that an admin can repair without a password reset. | 188 broad permission/regression tests, 3 scope/legacy probes and the dashboard build passed. |
| Frozen seam | Published the W0-3 spot-ledger read contract with signatures, semantics and consumers. | Contract validation is included in the passing regression groups. |
| Rating currency | Added all-or-none rating provenance to schema, ingestion, segment type, pricing config, API state and runtime. Activation requires `jewish_households`, `overnight_plus_1` and a matching named source on every billed segment. Model-rebased ratings clear provenance. | 45 focused QH/schema/transform tests passed. Current unlabelled data remains blocked and the default stays off. |

The focused groups overlap and are not summed into one inflated test count.

## Adversarial pass after the first repair

The agents were sent back over the repaired tree instead of being asked only
whether their first findings were gone. They found more places where a green
signal could still lie, and the record keeps them rather than rounding the first
repair up to a pass:

1. The gauntlet compared rebuilt hashes but ignored the golden script's non-zero
   exit code. It now fails when either the shipped plan coupling or the embedded
   golden fails, even when reference and work happen to rebuild equal hashes.
2. A frozen plan version could expose a historical operator channel and its
   totals after the configured operator changed. Every frozen CSV is now scoped
   again at read time and every delta is recomputed from those scoped bytes.
3. The backend failed closed for an unresolved affiliation, while the frontend
   still mapped every non-channel value to company. A Node regression now drives
   the shipped session policy and proves that the model route, job and wall stay
   hidden for `unknown`.
4. The spot ledger selected the newest daily file twice, allowing money from one
   upload to be labelled with another. It now resolves one path and passes that
   same path through pricing and attribution.
5. QH activation accepted an empty segment set, and its revenue currency did not
   reach the API or committed artifact. Empty data is now refused; the result,
   API, frame and fingerprint carry the basis and verified source fields.
6. A settings edit during recompute could stamp new settings onto money built
   under old settings. The build now carries stable settings, pricing, override
   and input snapshots to publication. Shipped publication refuses explicit
   programme/model/weight inputs that are not represented by that snapshot. A
   partial recompute runs only from a hash-valid committed base with the same
   economic snapshot; an absent or mismatched fingerprint forces a full rebuild
   instead of laundering untouched rows into a new stamp.
7. CSV and fingerprint publication could split if the second write failed. Both
   are now prepared as a pair; an injected fingerprint publish failure rolls the
   CSV back and raises.
8. Clock-only marker coordinates wrapped after midnight. Programme and break
   coordinates are now seconds from the broadcast-day midnight, so a 00:10 break
   inside a 23:30 programme remains at 24:10 on the same timeline.

The counterexample tests for these cases are
`tests/test_gauntlet_engine_gate.py`, `tests/test_golden_gate_coupling.py`,
`tests/test_assistant_version_tools.py`, `tests/test_qa8_permissions.py`,
`tests/test_w0_3_advertiser_identity.py`, `tests/test_qh_billing.py`,
`tests/test_incremental_recompute.py`, `tests/test_plan_write_guard.py` and
`tests/test_wave3_owner_defects.py`.

## Honest remaining boundary

No known code defect from the numbered waves or either adversarial pass remains. External
facts and owner choices still remain external: the P13 House Number QC feed and
broadcast standards; the real rating provenance needed before QH activation;
the current-week, campaign-flight and as-run inputs; and the unresolved trade
questions recorded in `decisions-for-owner.md`. A future answer to one of those
questions may authorize new code. It is not treated as code already completed,
and no missing value is filled with a guess.
