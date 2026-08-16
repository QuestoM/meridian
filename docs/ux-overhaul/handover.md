# Kairos UX overhaul — implementation handover

- Implementation: 15 August 2026; production-direction/source snapshot synchronized 16 August 2026
- Status: implementation and guarded structural convergence complete; independent final visual/accessibility evidence is not claimed by this document

## Delivered product shape

Kairos now uses the selected Studio Ledger direction after production calibration: warm cream and bone working surfaces, near-black ink/chrome, restrained mineral sage for action/focus/state, Noto Sans Hebrew for Hebrew glyphs, IBM Plex Sans for Latin and figures, IBM Plex Mono for identifiers, one MUI theme, and one route-owned application shell. The single shipping product mark is a transparent two-path frame-splice SVG; generated logo directions and raster alternates do not ship.

The previous fifteen visible entrances are expressed as seven durable domains:

| Domain | Owned work |
| --- | --- |
| Today | Money, health, priority decisions, Economics, Guardrails, Yield |
| Plan | Objective, Run, Compare, Publish, Supply, Week board |
| Broadcast | Day timeline, Traffic pods, Break library, Manual decisions |
| Commercial | Clients, Money, Campaigns, Delivery, Pricing rules, Agencies |
| Sources | Inputs, Files, Reports |
| Governance | Restrictions, Licence, Rate card, Calendar, Channel/model, Planning levers, company Model |
| History | Changes, runs, restores, accounts, restore points |

Mabat remains a contextual dock rather than an eighth destination. Legacy hashes resolve to the canonical capability, local task state is URL-backed and scoped, user navigation pushes browser history, and only the active route/panel mounts. The company Model is a lazy, permission-gated Governance context inside the one live React application root. The shell header is one 57px band: Broadcast and Governance local navigation share the row rather than creating a tall second tier. Rail/header geometry persists while route and addressable-tab changes acknowledge only the workspace through a feature-detected View Transition or restrained fallback; reduced motion preserves the update/focus behavior without animation.

The Plan board now opens on a one-day operating workbench. It automatically captures an immutable, exact browser-local optimizer baseline for the current channel/day/week identity; named manual arrangements retain their placement snapshot, edit map, engine-scored totals and compliance result. Operators can compare any two current variants, open one locally, or return to the baseline without a server write. The distinct official path remains consequence-reviewed: it writes placement restrictions and re-plans the day. Browser-local drafts are deliberately not represented as server versions or cross-device records.

The detailed code record is [`phase-3-implementation.md`](./phase-3-implementation.md); the product and acceptance surface are [`product-model.md`](./product-model.md) and [`phase-0-product-map.md`](./phase-0-product-map.md).

## Explicit support policy

The user chose not to adapt the operator console for mobile or tablet. Below the supported desktop condition, `App` branches before session/authentication and data hooks and mounts a localized “continue on desktop” gate. It supports Hebrew/RTL and English/LTR, states the 1,200px requirement, and does not leave a hidden operational tree authenticating or fetching behind it.

The exception for zoomed desktop users combines viewport, pointer, physical-width, outer-window, and ratio signals. It is deliberately documented as a heuristic in [`assumption-ledger.md`](./assumption-ledger.md), not as a guaranteed browser/device classification API. Multi-browser zoom and hybrid-device validation remains a release-owner responsibility.

## Design-system adoption boundary

The authoritative new-work contract is [`design-system.md`](./design-system.md). Its implemented sources are the role tokens, Studio CSS layers, canonical primitives, MUI theme/RTL cache, shell, and desktop gate. All stylesheet colour literals outside `tokens.css` are eliminated, the one-sided-accent guard remains at zero, local Noto/IBM files ship without remote font requests, and Lucide remains the only runtime interface-icon source. Runtime and generated-artifact details are in [`asset-inventory.md`](./asset-inventory.md).

This is a unified rendered system with a completed guarded structural-control migration, not a completed CSS/framework rewrite. The measured after-state in [`divergence-resolution.md`](./divergence-resolution.md) records:

- 108 stylesheets and 24,777 CSS lines; the former 6,224-line shell sheet is split by responsibility, and every JS/JSX/CSS file is at or below 450 lines;
- 452 JS/JSX/CSS source files totaling 87,331 lines in the exact 16 August worktree boundary;
- zero raw screen button/input/select/textarea tags, with exactly four canonical native bridges in `src/shell/dom-controls.jsx`;
- zero hand-built-card recipe violations and no card-guard exception budget;
- 159 unique JS/JSX modules consuming Studio entry points (123 actions, 75 structural controls, 13 modal, 13 aggregate readout/layout; overlapping sets);
- zero direct MUI action imports outside `src/studio/actions.js` and zero feature imports from shell primitive implementations;
- deleted `Surface`, `CardHead`, `LocalNav`, `LinkButton`, and `VisuallyHidden` React wrappers; and
- one interface-icon library, with direct Lucide imports still a migration boundary.

New work must consume the specialized Studio entry points and preserve the zero structural budgets; it must not interpret feature override sheets or compatibility aliases as permission to create another design dialect. Residue remains explicit: 47 direct non-action MUI consumers, 121 direct Lucide consumers, and two screen-level modal implementations outside `studio/modal`.

## Capability and compatibility boundary

The overhaul consolidates capability rather than deleting it. Legacy addresses remain aliases, and active-only compositions replace hidden/stacked work. [`deprecation-ledger.md`](./deprecation-ledger.md) is the deletion authority.

Nine zero-bundle compatibility modules total 776 lines. Three are quarantined Model mount shims—`console-bridge.jsx`, `console-mount.js`, and `candidates/board-mount.jsx`—totaling 262 lines. They are unreachable from the live application but remain for downstream standalone mounts and regression/integration harnesses. Five additional known legacy candidates total 888 lines and are also outside emitted chunks. Delete none of these by filename age: first migrate the named external imports, guards, contracts, and capability assertions.

Old hashes likewise require observed usage and an explicit redirect/migration window before removal.

## Verification snapshot

The current structural source snapshot was verified with:

| Check | Result |
| --- | --- |
| `npm run test:all` | Pass: production build plus card, direction, date, accent, colour, and smoke guards |
| Production build | Pass with Vite 8.0.16 and 3,621 transformed modules |
| Initial JavaScript | 348.74kB raw / 114.53kB gzip; baseline was 1,830.9kB / 514.2kB |
| Initial CSS | 86.86kB raw / 16.02kB gzip |
| Plan route | 179.17kB raw / 51.97kB gzip |
| Model route | 183.83kB raw / 37.38kB gzip |
| Commercial route | 283.17kB raw / 74.38kB gzip |
| Data Grid chunk | 479.93kB raw / 140.30kB gzip |
| CSS colour guard | 0 literal colours outside `tokens.css` |
| One-sided accent guard | 0 prohibited patterns |
| Native-control boundary | 0 screen tags; exactly 4 canonical bridges |
| Hand-built-card boundary | 0 recipes; no exceptions |
| `npm audit --offline --json` | 0 vulnerabilities across 145 resolved dependencies |
| [Representative production-preview request audit](./production-preview-audit.md) | Pass across eight domains × HE/EN × two desktop sizes; 0 duplicate GET groups, console errors/warnings, HTTP/fetch failures, or cancellations |
| `git diff --check` | Pass |

The initial-bundle reduction is a build-artifact comparison. It does not prove total route transfer, caching, interaction latency, or real-user performance.

The final uncontended broad repository replay is green: 4,067 passed, 27 skipped, and 110 deselected in 839.64 seconds, with zero failures or errors. The independent frontend aggregate is also green: `npm run test:all` passed with 3,621 transformed modules and every guard at its frozen budget.

This handover does not claim final visual, keyboard, screen-reader, console/network, authenticated permission-matrix, or destructive-write certification. The generated [`final-cream-matrix-v2`](./evidence/after/final-cream-matrix-v2/aggregate.md) and the probes behind [`friction-ledger-after.md`](./friction-ledger-after.md) remain evidence under their own stated harnesses; this source handover does not elevate them into release certification. The baseline runtime used open authentication, read-only plan protection, and blank assistant-provider credentials; write-flow proof requires an authorized isolated data copy.

## Blocking data-owner decision: optimizer inventory

The optimizer placement-input parser currently rejects all **994** shipped CSV
rows because none supplies a usable hour. Authoritative Plan Run, scenario,
forecast, comparison, and model-measurement boundaries now fail closed on that
condition instead of computing a money-moving result with neutral weights. This
is not an empty business state and must not be presented or repaired as zero by
the interface.

The Plan Supply read uses a separate reference-spots pipeline, so the 994-row
failure does **not** prove that the current Supply payload is empty. Before
production planning can run against inventory steering, the data owner must
choose and validate one of these contract-level outcomes:

1. correct the source timestamps/row shape to the current parser contract; or
2. change the parser contract with fixtures, migration evidence, and an explicit definition of valid hour bucketing.

The UX already preserves unknown/unavailable semantics. It cannot safely infer which of 994 rejected rows should move money. This owner decision is the most important non-frontend release dependency.

## Known limits and release gates

| Area | Current truth | Required closure |
| --- | --- | --- |
| Optimizer inventory | 994/994 placement-input rows rejected; authoritative planning refuses to run | Data-owner contract decision and validated ingest fixture; do not infer the separate Supply payload from this file |
| Auth and permissions | Open-access runtime used for overhaul QA | Configured admin/operator/viewer and company/channel matrix |
| Writes | Plan read-only and providers blank during evidence capture | Isolated writable copy; review, refusal, persistence, undo, and recovery checks |
| Desktop gate | HE and EN passed at exact 1024×768 inner/outer/screen/available metrics with no shell/auth tree and empty API ResourceTiming/network lists | Chrome/Safari/Firefox zoom 200–400%, touch laptop, split-screen, display scaling, remote desktop |
| Visual/a11y | Generated route-matrix evidence exists; this document makes no final independent-pass claim | Review it alongside independent keyboard, screen-reader, device, and release evidence before certification |
| Source convergence | Screen controls, cards, action imports, and feature-to-shell primitive imports are converged; CSS, non-action MUI/Lucide imports, and two modal exceptions remain | Preserve zero structural budgets; follow the ordered residual migration in the divergence ledger |
| Compatibility | Zero-bundle shims and old hashes remain intentionally | Migrate downstream consumers/tests, observe alias use, then remove with explicit evidence |

## Developer operation

Frontend root: `tv-break-dashboard/`

```sh
cd tv-break-dashboard
npm run dev
npm run test:all
npm audit --offline --json
```

The primary architecture and migration documents are:

- [`phase-1-audit.md`](./phase-1-audit.md) — ranked baseline findings and acceptance criteria
- [`phase-2-direction.md`](./phase-2-direction.md) — three generated directions and Studio Ledger selection
- [`phase-3-implementation.md`](./phase-3-implementation.md) — code and bundle record
- [`design-system.md`](./design-system.md) — normative tokens, components, state, interaction, and layout laws
- [`friction-ledger-after.md`](./friction-ledger-after.md) — comparable task-density after-state
- [`divergence-resolution.md`](./divergence-resolution.md) — exact system adoption and remaining debt
- [`deprecation-ledger.md`](./deprecation-ledger.md) — aliases, shims, legacy candidates, and deletion conditions
- [`asset-inventory.md`](./asset-inventory.md) — fonts, icons, marks, and generated direction artifacts
- [`decision-ledger.md`](./decision-ledger.md) and [`assumption-ledger.md`](./assumption-ledger.md) — autonomous decisions and explicit inference boundaries

## Non-regression rules for the next owner

1. Preserve the seven-domain model; add work as an addressable local context, not a new global entrance.
2. Mount only the active panel and keep route state scoped and Back/Forward-safe.
3. Branch to the desktop gate before auth/data; never hide a live console behind it.
4. Keep the shell header one 57px row; local navigation may scroll inline but must not wrap into a second tier.
5. Preserve workspace-only causal continuity, fallback behavior, reduced-motion behavior, and post-transition focus.
6. Keep unknown, missing, refused, stale, modeled, observed, and zero distinct.
7. Use Studio roles and the specialized `studio/actions`, `studio/dom-controls`, `studio/modal`, and aggregate Studio entry points; keep literal CSS colours and one-sided accents at zero.
8. Let `.workspace` own the route gutter and Card own its inset; use named bleed rather than erasing child padding.
9. Keep primary actions, icon actions, tabs, date selectors, and proxy targets at 44px; keep data rows at 48px.
10. Preserve consequence review, Cancel-first destructive confirmation, announced outcomes, and Undo where recovery exists.
11. Keep screen-native controls and hand-built-card recipes at zero, keep exactly four native bridges, and keep direct MUI action and feature-to-shell primitive imports at zero; never raise a budget to make a regression pass.
12. Keep quarantined compatibility modules outside the live import graph until their consumers are deliberately migrated.
13. Do not claim release readiness from build success. Close data, auth, write-flow, independent visual/a11y, console/network, and device-class evidence explicitly.
