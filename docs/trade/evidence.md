# Visual evidence — the surfaces, photographed

Captured from the production build against the live store by
`tv-break-dashboard/scripts/capture-trade-evidence.mjs` on 2026-08-17. The
full run covers 22 route×size combinations at 1728px and 1280px with
console, network, hard-break, target-size and overflow measurement per
route; the last full run reported **zero console errors, zero pages
scrolling sideways, zero manual line breaks** on every route. The seven
frames below are the demonstration's spine.

| # | Frame | What it proves |
|---|---|---|
| 1 | `evidence/01-agreements-shelf.png` | Six agreements in every lifecycle state; the סנו card carries live standing chips (בסיכון · 2); the open-ended window says ללא מועד סיום in words. |
| 2 | `evidence/02-review-gate.png` | The completeness meter (1/50 clauses seen, 0/41 decided) and the server's refusal naming every blocker — clause ids and term ids, not a generic error. |
| 3 | `evidence/03-review-precedence-tail.png` | The BOTTOM of the 16,700px review page: precedence terms with citation chips, the STD-provenance honesty chip (נוהג מקובל, לא מאומת בשוק המקומי), and the closing conflicts line. The harness once photographed beige here and called it a page. |
| 4 | `evidence/04-obligations-live.png` | Terms by family with per-term honesty chips, the honest empty bound-rules state that says WHY, and the live obligations header (בסיכון · 2). |
| 5 | `evidence/05-obligation-cards-simulation.png` | The flagship's measured standing: 14,580 of 480,000 ILS and 22.6 of 500 points against real As-Run, the projection-method box, and the default-bands disclosure ("זה אינו רף שמישהו סיכם"). |
| 6 | `evidence/06-day-versions-compare.png` | Pillar 2 live: three authored versions of one day, each side with its decision headline, the revenue delta WITH its by-daypart reasoning, net of retention, the honest commitments line, and adopt/reject. |
| 7 | `evidence/07-forecast-stage.png` | Pillar 3 live: 82 programmes with expected rating, honest band, drivers (including families held out by name), the historical mean beside the model, and the measured verdict printed above the table. |

To regenerate the full set:

    node tv-break-dashboard/scripts/capture-trade-evidence.mjs \
         --base-url http://127.0.0.1:3001 --out /tmp/trade-evidence

The harness embeds three laws it paid for: the viewport must be grown to the
true page height before any capture (captureBeyondViewport does not paint far
past the viewport, and worse, resizes the surface internally — which crossed
the desktop gate's threshold and photographed a freshly remounted app);
liveness is awaited before AND after the growth by reading the shell's own
connection chip; and section slices spread over the whole page so a tall
page's tail is never unseen.
