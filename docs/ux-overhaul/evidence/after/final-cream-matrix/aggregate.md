# Kairos final-cream UX evidence

Generated 2026-08-15T21:42:51.487Z. Verdict: **PASS** (0 defect records).

The flow under test is: each canonical address loads → the first meaningful operational screen settles → static layout, accessibility-adjacent, network and screenshot evidence is recorded without invoking a write control. The repository-mandated dependency-free CDP harness was used instead of the in-app browser so the same isolated, repeatable matrix could cover every route.

## Coverage

- Canonical routes: 34/34
- Desktop captures: 136/136; successful 136
- Gate captures: 2/2
- Screenshots present: 552/552
- Viewports: 1280×720, 1728×900; gate 1024×768
- Locales: Hebrew/RTL and English/LTR

## Defect ledger

| Category | Records |
| --- | ---: |
| captureErrors | 0 |
| settleTimeouts | 0 |
| pageIdentity | 0 |
| h1 | 0 |
| localeDirection | 0 |
| horizontalOverflow | 0 |
| targetFailures | 0 |
| contrastFailures | 0 |
| consoleErrors | 0 |
| consoleWarnings | 0 |
| httpErrors | 0 |
| requestFailures | 0 |
| edgeFailures | 0 |
| reducedMotionActiveViolations | 0 |
| missingScreenshots | 0 |

Successful duplicate GET groups (440) are informational and are not defects. Request cancellations (4) are recorded separately from failures. Native-wrapper target exceptions: 80. Allowed full-bleed edge contacts: 0.

## Route matrix

| Route | Captures | H1 | Overflow | Targets | Contrast | Console E/W | HTTP/Fetch/Cancel | Edge | Duplicate GET groups | PNGs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [today](./today/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 8 | 16 |
| [plan-objective](./plan-objective/report.json) | 4/4 | 0 | 0 | 0 fail / 16 except | 0 fail / 4 except | 0/0 | 0/0/0 | 0 | 20 | 16 |
| [plan-run](./plan-run/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 20 | 16 |
| [plan-compare](./plan-compare/report.json) | 4/4 | 0 | 0 | 0 fail / 32 except | 0 fail / 0 except | 0/0 | 0/0/4 | 0 | 20 | 16 |
| [plan-publish](./plan-publish/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 4 except | 0/0 | 0/0/0 | 0 | 20 | 16 |
| [plan-supply](./plan-supply/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 20 | 16 |
| [plan-board](./plan-board/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 20 | 16 |
| [broadcast-day](./broadcast-day/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 4 except | 0/0 | 0/0/0 | 0 | 8 | 16 |
| [broadcast-pods](./broadcast-pods/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 12 | 16 |
| [broadcast-library](./broadcast-library/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 0 | 16 |
| [broadcast-decisions](./broadcast-decisions/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 4 except | 0/0 | 0/0/0 | 0 | 8 | 16 |
| [commercial-clients](./commercial-clients/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 16 | 16 |
| [commercial-money](./commercial-money/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 16 | 16 |
| [commercial-campaigns](./commercial-campaigns/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 28 | 16 |
| [commercial-pacing](./commercial-pacing/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 24 | 16 |
| [commercial-advertisers](./commercial-advertisers/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 28 | 16 |
| [commercial-agencies](./commercial-agencies/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 28 | 16 |
| [sources-inputs](./sources-inputs/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 4 | 16 |
| [sources-files](./sources-files/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 4 | 16 |
| [sources-reports](./sources-reports/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 4 | 16 |
| [governance-restrictions](./governance-restrictions/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 4 except | 0/0 | 0/0/0 | 0 | 0 | 16 |
| [governance-licence](./governance-licence/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 0 | 16 |
| [governance-rate-card](./governance-rate-card/report.json) | 4/4 | 0 | 0 | 0 fail / 16 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 8 | 16 |
| [governance-calendar](./governance-calendar/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 8 | 16 |
| [governance-channel](./governance-channel/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 8 | 16 |
| [governance-levers](./governance-levers/report.json) | 4/4 | 0 | 0 | 0 fail / 16 except | 0 fail / 4 except | 0/0 | 0/0/0 | 0 | 4 | 16 |
| [history](./history/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 4 | 16 |
| [model-gates](./model-gates/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 12 | 16 |
| [model-coverage](./model-coverage/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 8 | 16 |
| [model-drift](./model-drift/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 12 | 16 |
| [model-candidates](./model-candidates/report.json) | 4/4 | 0 | 0 | 0 fail / 2 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 32 | 16 |
| [model-training](./model-training/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 12 | 16 |
| [model-versions](./model-versions/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 12 | 16 |
| [model-provenance](./model-provenance/report.json) | 4/4 | 0 | 0 | 0 fail / 0 except | 0 fail / 0 except | 0/0 | 0/0/0 | 0 | 12 | 16 |

Classification ledger: 80 native controls delegate to measured 44px-or-larger wrappers; 2 targets are within the explicit 0.1 CSS-pixel rendering tolerance; 24 low-ratio records belong to inactive controls; 8 final URLs retain the canonical address while adding route-owned selection state. Reduced media retained 388 configured motion-sensitive styles but 0 active nontrivial animations. The 440 duplicate-successful-GET groups contain 488 requests beyond the first; all 4 cancellations are 200-status event streams aborted after use.

## Gate contract

| Capture | Exact 1024×768 | ResourceTiming API list empty | Network API list empty |
| --- | --- | --- | --- |
| desktop-gate/he/1024x768 | pass | pass | pass |
| desktop-gate/en/1024x768 | pass | pass | pass |

## Shell geometry

Values are min / median / p95 / max in CSS pixels, followed by sample count.

| Element | Height distribution |
| --- | --- |
| topBar | 57 / 57 / 57 / 57 (108) |
| topBarPrimary | 56 / 56 / 56 / 56 (68) |
| shellLocalNav | 56 / 56 / 56 / 56 (40) |
| workspaceLocalNav | — |
| routeHeader | 53.28 / 98.16 / 205 / 205 (96) |

Main padding top/right/bottom/left: top 0 / 0 / 20 / 20 (136); right 16 / 16 / 24 / 24 (136); bottom 32 / 40 / 40 / 40 (136); left 16 / 16 / 24 / 24 (136).

Route-root padding top/right/bottom/left: top 12 / 20 / 24 / 24 (136); right 0 / 0 / 24 / 24 (136); bottom 24 / 32 / 40 / 40 (136); left 0 / 0 / 24 / 24 (136).

## Motion

- View Transition API available in 136/136 desktop captures.
- View-transition rule count: 9 / 9 / 9 / 9 (136).
- Normal configured effects: 0 / 24 / 200 / 200 (136).
- Reduced-motion configured effects: 0 / 15 / 79 / 200 (136).
- Motion-sensitive effects remaining under reduced motion: 0.

## Fonts actually painted

| Locale | Sample | Captures | Platform fonts observed |
| --- | --- | ---: | --- |
| he | hebrew | 69 | IBM Plex Sans Hebrew, IBM Plex Sans Hebrew SemiBold, Noto Sans Hebrew Thin |
| he | latin | 69 | IBM Plex Mono, IBM Plex Sans Hebrew SemiBold, IBM Plex Sans SemiBold, Noto Sans Hebrew Thin |
| he | figure | 68 | Courier New, IBM Plex Mono Medium, IBM Plex Sans, IBM Plex Sans Hebrew, IBM Plex Sans Hebrew Medium, IBM Plex Sans Hebrew SemiBold, IBM Plex Sans Medium, IBM Plex Sans SemiBold, Noto Sans Hebrew Thin |
| en | hebrew | 55 | Arial, IBM Plex Sans, IBM Plex Sans Medium, IBM Plex Sans SemiBold, Noto Sans Hebrew Thin |
| en | latin | 69 | IBM Plex Sans, IBM Plex Sans SemiBold |
| en | figure | 68 | IBM Plex Mono Medium, IBM Plex Sans, IBM Plex Sans Medium, IBM Plex Sans SemiBold, Noto Sans Hebrew Thin |

Load-to-capture timing in ms, min / median / p95 / max: 1628 / 1746 / 2398 / 2818 (138). Full records, screenshots, console/network payloads, contrast calculations, target rectangles, font glyph counts and edge insets are in [aggregate.json](./aggregate.json) and each linked route report.
