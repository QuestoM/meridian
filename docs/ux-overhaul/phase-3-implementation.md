# Phase 3 — implementation record

Implementation date: 15 August 2026. Production-direction and source-count synchronization: 16 August 2026.

Scope: frontend architecture, design foundation, shell, all seven domains, Mabat, company Model, routing, and the user-approved desktop-only gate.

Evidence boundary: this file records code and build facts. It does not claim final browser, visual, accessibility, console/network, or write-flow QA; those require their own evidence and three-pass report.

## Outcome in code

The application now expresses the Studio Ledger direction through one token/theme layer and one seven-domain shell. Runtime routing mounts the active workspace lazily, local task state is addressable, Browser Back/Forward is part of the contract, and phone/tablet operation is replaced before authentication or data loading by a localized desktop gate. The frozen production calibration is warm cream/near-black/mineral sage, script-specific Noto/IBM typography, a transparent two-path frame-splice mark, and a persistent one-row shell whose workspace—not its rail or header—acknowledges navigation.

The capability hierarchy is now:

| Domain | Addressable contexts | Runtime component |
| --- | --- | --- |
| Today | Overview plus Economics, Guardrails, Yield detail | `OverviewPage` |
| Plan | Objective, Run, Compare, Publish, Supply, Week board | `PlanWeek` |
| Broadcast | Day timeline, Traffic pods, Break library, Manual decisions | `DayPage`, `PodPage`, `BreakLibraryPage`, `OverrideDecisions` |
| Commercial | Clients, Money, Campaigns, Delivery, Pricing rules, Agencies | `ClientsWorkspace` |
| Sources | Inputs, Files, Reports | `SourcesPage` |
| Governance | Restrictions, Licence, Rate card, Calendar, Channel/model, Planning levers, company Model | `RulesWorkspace`, lazy `ModelConsole` |
| History | Changes, runs, restores, accounts, restore points | `HistoryPage` through `VersionsPage` |

Mabat is a persistent dock rather than an eighth destination.

## Foundation implementation

[`tokens.css`](../../tv-break-dashboard/src/tokens.css) now owns the complete Studio roles: material/ink, semantic ramps, contrast-on-strong roles, Hebrew/Latin/mono stacks, the eight-role type scale, 4/8px spatial rhythm, desktop/rail/control/data-row dimensions, radii, two elevations, motion timings/easings, and named layers. Compatibility names resolve to the new roles rather than carrying a competing palette.

[`shell/theme.js`](../../tv-break-dashboard/src/shell/theme.js) maps the same roles into MUI 9 for both Emotion LTR and RTL caches. [`shell/emotion-cache.js`](../../tv-break-dashboard/src/shell/emotion-cache.js) contains the narrow MUI Data Grid RTL selector correction while preserving Emotion diagnostics for every unrelated unsafe selector. The theme normalizes Button, IconButton, fields, checkbox/radio/switch, tabs, toggle buttons, menus, paper/dialog/drawer/backdrop, Data Grid, tables, tooltip/popover, skeleton, and progress. Focus-visible is a 3px mineral-sage ring with a 2px offset; primary and icon actions have 44px minimums; data rows and headers are 48px. All MUI contained semantic palettes explicitly use the cream `surface` foreground rather than inheriting a dark contrast value.

[`studio/actions.js`](../../tv-break-dashboard/src/studio/actions.js) is the only source import authority for MUI `Button`, `ButtonBase`, and `IconButton`. [`studio/dom-controls.js`](../../tv-break-dashboard/src/studio/dom-controls.js) exposes dependency-light `Pressable`, `InputControl`, `SelectControl`, and `TextAreaControl`; their only four native tags are implemented in [`shell/dom-controls.jsx`](../../tv-break-dashboard/src/shell/dom-controls.jsx). [`studio/modal.js`](../../tv-break-dashboard/src/studio/modal.js) exposes Dialog/Sheet and focus helpers. [`studio/index.js`](../../tv-break-dashboard/src/studio/index.js) exposes Card/CardBody/CardBleed, Status, Metric, PageHeader, empty/loading/error states, and the bidi-aware lazy DataTable from the shell implementation layer. [`studio/studio.css`](../../tv-break-dashboard/src/studio/studio.css) and [`studio/studio-workspaces.css`](../../tv-break-dashboard/src/studio/studio-workspaces.css) supply their material, workspace, state, reduced-motion, and forced-colour layers.

The structural import migration is complete at its guarded boundary. Across overlapping sets, 159 JS/JSX modules import at least one Studio entry point: 123 import actions, 75 import structural controls, 13 import modal mechanics, and 13 import the aggregate readout/layout API. Six modules use `IconButton` through `studio/actions`. No feature imports `shell/primitives`, `shell/dom-controls`, or `shell/modal-primitives`, and no module outside `studio/actions.js` imports MUI `Button`, `ButtonBase`, or `IconButton` directly. `Surface`, `CardHead`, `LocalNav`, `LinkButton`, and `VisuallyHidden` were deleted rather than retained as aliases.

The remaining framework residue is narrower and explicit: 47 modules still import non-action MUI APIs directly, 121 import Lucide directly, and two screen-level modals remain outside `studio/modal`—the MUI suspension dialog in `AgencyDetailDrawer` and the native command-palette dialog. Those are follow-up migrations, not alternate approved entry points.

The local type bundle loads a Hebrew-only Noto Sans Hebrew Variable 100–900 subset, IBM Plex Sans Hebrew 400/500/600 as compatibility fallback, IBM Plex Sans Latin 400/500/600, and IBM Plex Mono Latin 400/500. Noto Hebrew plus IBM Plex Sans regular and semibold WOFF2 faces are preloaded. Asset details are in [`asset-inventory.md`](./asset-inventory.md).

## Shell and routing

[`shell/nav.js`](../../tv-break-dashboard/src/shell/nav.js) is the canonical information architecture and address contract. It:

- defines seven global domains and their local items;
- maps every old visible hash to a canonical capability;
- owns scoped query parameters for Today, Plan, Broadcast, Commercial, Sources, Governance, History, and Model;
- validates enumerated values and removes parameters owned by another domain;
- preserves a Model permission wall; and
- creates shareable relative URLs.

[`TVBreakDashboard.jsx`](../../tv-break-dashboard/src/shell/TVBreakDashboard.jsx) uses push history for user navigation, replace history only for address normalization, and listens to traversal and local address changes. It provides one localized skip link and one named `main`. [`workspace-router.jsx`](../../tv-break-dashboard/src/shell/workspace-router.jsx) lazy-loads each domain and passes the recommendation/approval actions into the live Plan surface rather than discarding them.

The top-level bundle no longer imports the heavy workspace tree eagerly. The shell lazy-loads Activity Feed, Mabat, user administration, and Model; the router lazy-loads Today, Plan, four Broadcast contexts, Commercial, Sources, Governance, and History.

The shell header is one persistent 56px content row with a 1px block-end boundary, for a 57px rendered band. Domains with local navigation—including Broadcast and Governance—use a three-column title / scrollable-local-nav / status composition in that same row. The local rail cannot wrap or shrink its 44px controls into a second tier. The rail and header receive stable View Transition names; they do not animate or remount when the workspace changes.

[`shell/workspace-continuity.js`](../../tv-break-dashboard/src/shell/workspace-continuity.js) makes route, Back/Forward, and addressable-tab continuity causal. Where supported, `document.startViewTransition` atomically commits the React update and animates only the outgoing/incoming workspace. If the API is absent or throws, a short workspace-only CSS acknowledgement runs after the update. Destination focus is applied at the transition boundary. `prefers-reduced-motion` bypasses both animation paths while retaining the update and focus behavior.

The former 6,224-line `shell/styles.css` body is split by responsibility into 13 `styles*.css` sheets; the base file is now 313 lines and its largest sibling is 400. `studio-shell.css` is 442 lines and `shell-continuity.css` is 120. Every JavaScript, JSX, and CSS source file in the current tree is at or below the 450-line maintenance cap. The exact source snapshot is 108 CSS files / 24,777 lines, 344 JS/JSX files / 62,554 lines, and 452 combined files / 87,331 lines.

## Spacing and inset correction

The shell now has one owner for page gutters and bounded-surface insets. `.workspace` supplies 24px inline gutters on expanded desktop, 16px on compact desktop, and 40px at the block end. A direct `.page-workspace` keeps its vertical padding but removes duplicate inline padding. Broadcast pods receive their missing route-leading space.

[`shell/card.css`](../../tv-break-dashboard/src/shell/card.css) makes the card own `--card-inset`; `CardBleed` is the named opt-in for full-width row/table bands, and the band still aligns its first and last content columns to the card inset. The status Data Grid correction removes block padding only instead of erasing both axes, and the Today decision/revenue disclosures restore the inherited inline/card inset that later feature rules had zeroed.

The reusable evidence harness measures shell/header/navigation height, main and route padding, four logical edge insets, and under-12px edge contacts with a named full-bleed allowlist. Its generated matrix is stored in [`final-cream-matrix-v2`](./evidence/after/final-cream-matrix-v2/aggregate.md); this implementation record documents the instrumentation and code corrections without treating that artifact as independent QA or release certification.

## Desktop support policy

The user explicitly chose a desktop-only console. [`shell/App.jsx`](../../tv-break-dashboard/src/shell/App.jsx) evaluates support before mounting the session boundary, so an unsupported device does not authenticate or begin protected/data requests behind a hidden interface.

[`shell/desktop-gate.jsx`](../../tv-break-dashboard/src/shell/desktop-gate.jsx) renders one localized main/H1, product explanation, 1200px requirement, and language switch. It supports Hebrew RTL and English LTR. A width-only check would reject low-vision desktop zoom, so the implementation also considers hover/fine pointer, absence of a coarse primary pointer, available physical width, outer window width, and an outer-to-layout ratio. This is intentionally recorded as a heuristic in [`assumption-ledger.md`](./assumption-ledger.md), not as a proven browser API.

The rail is 96px at 1400px and wider, and 88px between 1200px and 1399px. Below the supported condition, the console does not reflow: the gate replaces it.

## Today

Today is reorganized around the three first questions—money, health, and decisions—before secondary analysis. Its independent primed read can answer before the rest of the shell data; when that endpoint is unavailable it distinguishes a real shared-payload fallback from no answer. Target creation remains an inline edit next to the value it governs. Health findings lead to their owned context, and stale-plan findings focus the one shared rebuild action rather than creating a duplicate run control.

Economics, Guardrails, and Yield are addressable active-only tab panels with full RTL-aware Arrow/Home/End behavior. Recommendation links now carry the recommendation ID into Plan, where `RecommendationDecisionPanel` can approve, reject, apply similar, or hand an incomplete forced-count decision to Manual Decisions with provenance.

## Plan

`PlanWeek` is the single weekly workspace. Its six steps read and write `?plan=`, push user navigation, restore on `popstate`, mount only the active panel, and focus the newly active region. Objective recommendations now share the shell's real decision actions instead of disappearing at the route seam.

The shell overview verdict seeds Plan freshness on the first load, eliminating a duplicate `/api/overview` read. Plan owns every subsequent revalidation after a write or whenever no trustworthy seed exists, so the deduplication does not reintroduce the former stale-header defect.

The command palette is a native modal dialog with initial focus, focus containment, Escape, focus restoration, localized name, 44px close target, combobox/listbox semantics, `aria-activedescendant`, and visible focus. It remains one of the two explicitly recorded screen-level modal exceptions; new modal work uses `studio/modal`. Publish fields use the current MUI `slotProps.htmlInput` API rather than forwarding obsolete `inputProps` to the DOM.

Plan controls and panels consume Studio target/type/colour rules. Objective, run, compare, publish, supply, and board keep the existing API capability, while hidden sections no longer perform their own work.

## Broadcast

Broadcast no longer concatenates the daily timeline, override composer, traffic pod, day table, and ranked library into one document.

- Day uses `DayBreakNavigator`: a select, previous/next walk, and 44px open/edit proxy for every true-scale timeline break. The visual chip remains available to pointer direct manipulation but is not the only keyboard target.
- Date, toolbar, hourly load, readout, editor, and action targets are normalized to 44/48px. The hour strip scrolls on its intended axis rather than shrinking targets.
- `ScheduleInspector` and `BreakInspector` are named nonmodal detail asides with initial focus, Escape behavior, busy/status/error semantics, and 44px actions. Opening programme detail makes the underlying Break inspector inert and hidden rather than leaving two interactive layers.
- Manual Decisions progressively reveals 12 records, previews with/without consequences, preserves recommendation provenance, and replaces one-click delete with an inline named review whose Cancel control receives focus.
- Break Library mounts one of Library, Day, or Pod. The ranked board reveals 20 rows at a time and keeps whole-set totals explicit; pod/day navigation pushes and restores URL state.
- Pod spot rows use `content-visibility: auto` with a 48px intrinsic row size to limit off-screen layout work.

## Commercial

Commercial is one workspace over agency → advertiser → campaign → flight and the money/delivery records that connect them. Its six local views use a full ARIA tab pattern and mount only the current panel. Related record drills retain the entity rather than sending the operator back to search.

High-volume surfaces now reveal stable windows: 12 campaigns, 16 pacing campaigns, and 18 client/agency/advertiser records at a time. Totals and matching counts remain visible. Campaign provenance distinguishes demo-seeded data from operator booking. Pacing retains unknown-source days as unknown and keeps risk acceptance, make-good creation, offer/close/withdraw, undo, and decision-ledger capability.

Commercial's four locale-neutral reads depend only on explicit refresh and reload keys. Switching Hebrew/English rerenders labels without refetching clients, money, campaigns, or advertiser rules; a focused regression contract pins that ownership.

Onboarding is a four-step full-height workflow—Identity, Commercial terms, Flights, Review—over the existing one-submit transaction. Background shell nodes become inert/`aria-hidden`; focus is contained and restored; each step focuses its title; the final review names the object before creation.

## Sources, Governance, History, Model, and Mabat

Sources unifies Inputs, Files, and Reports under one addressable active panel. Source truth states, check/commit distinction, finding consequences, file/row preview, and downloads remain. Failed/unreachable reads render as failures, not empty lists.

Governance unifies Restrictions, Licence, Rate card, Calendar, Channel/model, and Planning levers. Section choice is URL-backed, Back-safe, job-aware on first landing, active-only, and keyboard navigable. Constraint, calendar, and pricing actions adopt the Studio target/focus/material layer without changing their API meaning.

History now exposes the combined timeline through the canonical History domain. Kind filters are addressable and keyboard navigable; selection remains list/detail; link targets do not silently fall back to the newest entry; late requests cannot overwrite a newer filter read; refused, paged-out, unreachable, and empty states remain distinct.

Model is a lazy, permission-gated route with seven addressable sections, a vertical tab pattern, numeric shortcuts, active-only reads, and explicit exits to Governance Rules and Calendar. The Candidate Board is mounted directly in Candidates. The former bridge, side-effect mount, and standalone board root are quarantined as unreachable compatibility shims; none is imported by the live application.

Mabat remains a resizable, persisted dock with conversation, proposal review/apply, changes, upload, restore, and undo capability. Its Studio feature layer normalizes material, controls, focus, and dock width without warming an external provider merely because a phone/tablet gate is on screen.

## Build and import-graph result

A fresh production `npm run build` completed successfully on 16 August 2026 with Vite 8.0.16 and 3,621 transformed modules.

| Artifact | Raw | Gzip |
| --- | ---: | ---: |
| Initial `index` JavaScript | 348.74 kB | 114.53 kB |
| Initial `index` CSS | 86.86 kB | 16.02 kB |
| Lazy Data Grid chunk | 479.93 kB | 140.30 kB |
| Plan route | 179.17 kB | 51.97 kB |
| Model route | 183.83 kB | 37.38 kB |
| Commercial route | 283.17 kB | 74.38 kB |
| Governance route | 132.89 kB | 38.01 kB |
| Mabat dock | 117.34 kB | 35.45 kB |

The Phase 1 baseline initial bundle was 1,830.9 kB raw / 514.2 kB gzip. The current initial JavaScript artifact is 81.0% smaller raw and 77.7% smaller gzip. This is a build-artifact comparison, not a claim about total route transfer, cache behavior, interaction latency, or real-user performance.

The import graph leaves nine compatibility modules (776 lines) and five known legacy candidates (888 lines) outside all emitted chunks. The details and deletion conditions are in [`deprecation-ledger.md`](./deprecation-ledger.md).

## Explicit non-claims and handoff boundary

- Build success proves compilation and chunking only.
- This record does not assert zero console errors, zero failed requests, complete keyboard traversal, screen-reader behavior, final contrast under every state, or visual quality across every route/locale.
- The desktop zoom heuristic still needs multi-browser and device-class probes.
- Authenticated permission matrices and mutating flows need an isolated writable data copy; the audit runtime used open auth, read-only plan protection, and blank assistant-provider credentials.
- Compatibility token/class aliases, two modal-path exceptions, direct non-action MUI/Lucide consumers, old feature CSS, and unreferenced source still exist. The guarded screen-control, card-recipe, action-import, and feature-to-shell-primitive budgets are at zero; that structural result is not a claim of whole-framework or CSS elimination.
- Final release proof must link the before/after gallery, remeasured friction/divergence report, and all three independent QA passes rather than converting this implementation record into evidence it does not contain.
