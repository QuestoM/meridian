# Completion audit: campaign waves 3 through 6

Measured on `main` on 2026-08-10. This closes the ambiguity between the P1-P13
gauntlet, whose implementation ended at wave 2, and the broader campaign plan,
which defined four additional waves.

## Outcome

| Wave | Intended result | Current result | Evidence |
|---|---|---|---|
| 3, sweep | Close defect classes and owner-reported interface defects | Passed. Navigation duplication, workspace gutter, header collision and timeline marker legibility are guarded. The dashboard builds; card, direction and accent guards pass; accent count is zero. | `tests/test_wave3_owner_defects.py`; `34 passed` across the owner/shell/design group; dashboard build and three named guards |
| 4, Kai | Close read coverage, then ship mention R1 and drill-down | Passed. Mentions, typed references, drill-down, break/pod/pacing reads, pacing proposals, restriction cost and account coverage were already present. The last general coverage gap, one complete campaign record, now has a scoped read tool. | `164 passed` in the coverage/mention group; `46 passed` in the campaign/security group |
| 5, goal order | Carry the already-modelled rating goal into placement | Passed in code. The shared day core now folds the goal into ranking and supplies one commitment-aware net scalar to greedy, F1 and DP. Weekly export preloads the stores once. A scoped API and Kai read expose the pre-flight basis. All stored goals are demo rows, so the current plan stays byte-identical. | `43 passed` in the goal group; `71 passed` in the engine adoption and golden gate |
| 6, engine | Remove the remaining regression and reproduce the golden | Passed before this audit. The regression was a polluted manual-override row, not an optimizer defect. The restored plan reproduced all 120 channel-days byte-exact. | `docs/ux-gauntlet/campaign-plan.md`; `tests/test_refiner_net.py`; the 71-test gate rerun in this audit |

## Wave 5 integration contract

All live optimizer routes converge at `kairos/optimize/day_core.py`. That core now
calls `prepare_goal_inputs` once after the ordinary demand fold. With no real
applicable order, it preserves the objective mode and does not send a custom
`net_of` argument at all. With a real applicable order, it:

1. multiplies the two-sided rating-efficiency lean into demand weights;
2. uses `revenue_net` as the effective mode so refinement cannot erase the goal;
3. passes the same adjusted scalar into greedy, F1 and DP through the optimizer's
   optional `net_of` seam;
4. leaves reported revenue untouched;
5. scopes an order to its own channel and treats an absent or rival channel as
   inapplicable.

The weekly exporter reads real goal orders once for the full multi-day run and
reads the delivery ledger only if at least one real order exists. Single-day live
and simulation paths use the same core and may load lazily.

## What remains external, not another code wave

P13's live media verification remains honest and incomplete until the owner
supplies a real House Number QC report and approved broadcast standards. The code
contract, importer, lock gate and tests are complete. No placeholder QC verdict
or guessed standard was added to make that dependency look closed.

The two pre-existing modified schedule artifacts were not staged or rewritten:
`output/weekly_break_schedule.csv` and its fingerprint sidecar.
