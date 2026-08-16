# Production-preview request audit

- Captured: 15 August 2026 at 23:52 UTC
- Runtime: Vite production preview on `127.0.0.1:3002`, backed by the isolated read-only API runtime
- Build: 3,627 modules; entry JavaScript 348.74 kB / 114.53 kB gzip; entry CSS 86.86 kB / 16.02 kB gzip
- Verdict: **PASS — 0 active defect records**

This audit exists to separate production request behavior from React development
`StrictMode` remount noise. It used the same dependency-free CDP harness and the
same HE/RTL × EN/LTR × 1280×720/1728×900 contract as the canonical matrix, but on
one representative route in each operator domain rather than all 34 routes.

## Coverage

| Domain | Route |
| --- | --- |
| Today | `today` |
| Plan | `plan-objective` |
| Broadcast | `broadcast-day` |
| Commercial | `commercial-clients` |
| Sources | `sources-inputs` |
| Governance | `governance-restrictions` |
| History | `history` |
| Model | `model-gates` |

- 8/34 canonical routes represented
- 32/32 requested desktop captures succeeded
- 128/128 requested full/top/middle/bottom PNGs were written during the run
- Hebrew and English at both desktop sizes
- No gate capture; the gate contract belongs to the full canonical matrix

## Results

| Signal | Result |
| --- | ---: |
| Capture, settle, route identity, H1, locale/direction defects | 0 |
| Horizontal-overflow and content-edge defects | 0 |
| Active target-size defects | 0 |
| Active computed-contrast defects | 0 |
| Console errors / warnings | 0 / 0 |
| HTTP errors / fetch failures / cancellations | 0 / 0 / 0 |
| Duplicate successful GET groups / extra requests | **0 / 0** |
| Active reduced-motion violations | 0 |

The classifier retained 12 measured native-wrapper exceptions, eight inactive
control contrast exceptions, and four route-owned History address enrichments.
Those records are non-defects and remain distinct from the zero active-defect
verdict.

The first production-preview pass exposed a genuine 22px History restore-option
target. The label was corrected to the canonical 44px control height, the bundle
was rebuilt, and all four History locale/viewport captures were replaced before
the result above was recorded. The same correction was then recaptured into the
34-route canonical matrix.

## Boundary

This proves that the representative production bundle does not repeat the GETs
that appeared twice under the Vite development `StrictMode` runtime. It is not a
claim that every write flow, role, browser engine, or non-route state was tested
in production mode. The complete read-only visual/layout contract remains the
[final canonical matrix](./evidence/after/final-cream-matrix-v2/aggregate.md).
