# Phase 1 — Quantitative UX, interaction, accessibility, and frontend audit

Date: 2026-08-15

Baseline: current `main`, Hebrew/RTL, 1280×720 unless noted
Evidence: [`evidence/before`](./evidence/before) and [`phase-0-product-map.md`](./phase-0-product-map.md)

## Executive finding

Kairos has a strong operational model and unusually rich domain detail, but its current interface makes that model harder to operate than necessary. The core issue is not a missing feature. It is fragmentation: fifteen top-level entrances, several overlapping navigation systems, multiple generations of UI primitives, long unvirtualized workspaces, state that is not represented in the URL, and dialogs that do not isolate the task from the background.

The rebuild therefore has four non-negotiable outcomes:

1. Consolidate the product into a small, stable information architecture without removing any capability.
2. Turn the three critical jobs into focused, resumable paths with explicit state and consequences.
3. Replace the accumulated visual and interaction dialects with one measured system.
4. Make keyboard, screen-reader, loading, error, history, and destructive-action behavior part of the product contract rather than per-screen exceptions.

Mobile and tablet operation are deliberately out of scope. Below the supported desktop condition, the operational application must be unmounted or inert and replaced by a localized, accessible “continue on desktop” gate.

## Method

- Exercised all primary routes, every Rules and Model section, the assistant dock, authentication and error states, and the three critical end-to-end jobs.
- Captured more than 145 before-state screenshots, including Hebrew, English, narrow viewport, overlays, inspectors, and long-page sections.
- Measured rendered document length and interactive density at 1280×720.
- Counted target dimensions from the live DOM.
- Audited keyboard behavior, dialog isolation, ARIA patterns, headings, tables, announcements, bidi behavior, and destructive actions.
- Inspected the React route/state/data-loading paths, production bundle, CSS corpus, primitives, inline styling, icon use, fetch call sites, and unreachable modules.
- Ran the repository's existing guard suite and a production build. Both pass; the gaps below are largely outside what those guards test.

## Friction and density

### Core jobs

| Job | Minimum observed task friction | Rendered density |
|---|---|---|
| Week plan | About 7 activations from Overview through Run, Compare, and Freeze, plus two publish fields. Objective exposes 4 presets, 4 sliders, and 2 focus choices; Compare exposes 8 sliders and 4 focus choices. | Objective: 45 visible interactives, 6 fields, 1,096px. 39 targets are under 44px; 6 are under 24px. |
| Broadcast day / pod | Day detail is approximately rail → break → Enter. Pod truth is Break Library → covered day → pod. The day combines 30 dates, 80 break controls, 96 time labels, 24 hour controls, inspectors, and the override composer. | Overrides: 255 visible interactives, 5 fields, 1,633px. 223 targets are under 44px; 81 are under 24px; about 112 interactives appear per viewport. |
| Commercial delivery | Campaigns → row expansion → pacing → day evidence → onboarding is 5 activations and at least 4 local context shifts. Existing-agency onboarding contains 14 fields plus 7 weekday toggles; new-agency mode contains 24 controls. | Campaigns: 355 interactives, 52 rows, 7,488px; 352 under 44px and 222 under 24px. Pacing: 302 interactives, 51 repeated upload links, 15,226px. Opening onboarding leaves 381 interactives in the document and increases it to 9,740px. |

### Route-level baseline

| Surface | Document height | Buttons | Inputs / fields | Primary signal |
|---|---:|---:|---:|---|
| Overview | 2,628px | 35 | 2 | 13 distinct computed button styles on one screen |
| Objective | 1,078–1,096px | 79 total / 45 visible | 18 total / 6 visible | Hidden Plan sections remain mounted |
| Week board | 1,282px | 79 | 18 | Hidden DataGrids measure a zero-width parent |
| Supply | 2,158px | 85 | 16 | Same workspace, different local-only step |
| Compare | 1,008px | 67 | 16 | State is not addressable or Back-safe |
| Break library | 4,666px | 85 | 1 | Plan and traffic truth cover different dates |
| Overrides | 1,751px | 88 | 5 | Selected state produces 253–255 controls |
| Campaigns | 7,454–7,488px | 364 total / 355 visible | 1 | 52 unvirtualized rows and dense row actions |
| Campaign pacing | 15,226px | — | — | 51 repeated upload links; only 1 of 7 days has traffic truth |
| Advertisers | 5,278px | 33 | 2 | Separate management dialect inside Clients |
| Agencies | 1,233px | 40 | 1 | Separate management dialect inside Clients |
| Reports | 1,458px | 35 | 0 | Shared data family exposed as a top-level route |
| Data | 2,107px | 58 | 7 | Shared data family exposed as a top-level route |
| History | 1,125px | 34 | 5 | Confirmation quality is stronger here than elsewhere |
| Assistant | 6,145px | 151 | 72 | Persistent dock behaves like a full workspace |

### Target-size samples

| Surface | Visible interactives | Height under 44px | Width under 44px | Both dimensions under 44px | Minimum observed |
|---|---:|---:|---:|---:|---:|
| Overview | 43 | 35 | — | 10 | 8px chart activation |
| Objective | 45 | 39 | — | 8 | 12px-tall range tracks |
| Overrides | 255 | 67 | 195 | 39 | 16px in root sample; 81 controls under 24px in detailed audit |
| Campaigns | 355 | 352 | — | 206 | 8–20px row-action controls |

## Ranked findings

### P0 — Cold boot starts work before authentication resolves

`useKairosData` begins 11 requests before the authentication state is known. Events, model, and today requests raise the cold-boot minimum to at least 15. This explains the observed 401 burst and delayed hydration and makes login state, loading state, and data state race one another.

**Required correction:** establish a single authenticated application boundary; no protected data request may start before it resolves. Deduplicate shared requests and initiate independent, permitted requests in parallel.

### P0 — Route and state ownership are fractured

Navigation uses `replaceState`, preserves unrelated query parameters, and therefore prevents browser Back from restoring the user's previous workspace. Four week-plan entrances remount one `PlanWeek`; its six sections are local state rather than addressable locations. The workspace router still passes 27 legacy optimizer/recommendation/axis/inspector props that `OptimizerWorkspace` does not consume, so shell recommendation and approval state never reaches the live planning surface. Some state setters run during render.

**Required correction:** one canonical route/state schema, `pushState` for user navigation, explicit replace only for normalization, scoped query serialization, addressable plan steps, and a single owner for recommendation/approval state. Only the active route subtree mounts.

### P0 — The current system cannot be maintained as one visual language

The frontend contains 65 CSS files and 20,669 lines of CSS. Approximately 3,409 rule blocks, 3,844 selector occurrences, 2,329 unique classes, 217–223 repeated selector names, 2,077 raw pixel literals, 311 hard dimensions, 31 z-index declarations, and 64 inline `style` props have accumulated. Passing guards allow this structure rather than constraining it.

**Required correction:** a small named token set, documented layers, canonical primitives, feature-scoped composition, and migration guards that forbid reintroducing raw dialects.

### P1 — Dialogs and drawers do not isolate focus or context

The onboarding dialog has no accessible name, `aria-modal`, initial focus, focus containment, inert background, or focus restoration; all 355 background campaign controls remain reachable. Break and programme inspectors similarly leave the board active and can stack. Thirteen `role="dialog"` render points exist; only three declare `aria-modal`. Only the command palette explicitly moves focus, and it still does not trap or restore it.

### P1 — Critical controls are too small

The day board has 81 targets under 24px. Two-minute breaks are intentionally drawn at roughly 12×44px and the resize handle is no wider than 8px. Campaign actions commonly render at 8–26×18–20px. Keyboard movement and resizing exist, but a nested non-focusable `role="separator"` incorrectly advertises an independent adjustable control.

### P1 — Navigation creates context churn without reliable return or bypass

Fifteen rail controls precede `main`; there is no skip link. The week job crosses the global rail, top controls, six plan steps, and state-only drawers. Clients adds another local navigation model. Back cannot reliably restore any of these transitions.

### P1 — Consequence and recovery patterns are inconsistent

“Apply to weekly schedule” persists settings and rebuilds the week immediately. Override removal is a one-click DELETE in the composer and programme inspector. Plan restore and campaign ending use clearer interlocks. A user cannot infer from presentation whether a similar-looking action is immediate, reversible, or destructive.

### P1 — Unsupported narrow devices currently receive a broken partial reflow

At 390×844, `main` begins exactly one viewport below the top and the body is 5,182px tall. The initial viewport contains clipped navigation and blank space. This is replaced by the product decision for a complete desktop-only gate, not by further responsive adaptation.

### P1 — Production delivery is monolithic

The production build passes but ships a 1,830.90kB minified main bundle (514.19kB gzip), a 477.21kB DataGrid chunk (139.53kB gzip), and 346.19kB CSS (49.14kB gzip) across 3,556 modules. Routes are eagerly imported; a second Model root adds inactive DOM and bundle cost.

### P1 — All PlanWeek sections mount even when hidden

Five of six sections remain mounted. Hidden MUI DataGrids consequently report zero-width parents, do work the user cannot see, and leak component state across conceptual steps.

### P1 — Campaign and pacing surfaces do not scale

Campaigns is an unvirtualized 52-row application with hundreds of always-mounted actions. Pacing expands into 15,226px and repeats day-level actions 51 times. `DataTable` also recomputes normalized rows and columns. This is an information-architecture and rendering problem, not a request for smaller typography.

### P2 — Composite-widget semantics are incomplete

Clients and Pacing tabs lack roving focus, Arrow/Home/End behavior, `aria-controls`, and associated tab panels. The command palette changes `aria-selected` while focus remains in its input but supplies no `aria-activedescendant`; the pacing `j/k` list similarly changes descendant state without moving DOM focus.

### P2 — Document and table semantics do not survive density

Overrides renders two H1s because `DayPage` and the full override application are concatenated. Twenty-six native table render points exist but only two captions. Campaign row actions such as “Amend” and “End” omit the affected campaign from their accessible names.

### P2 — Loading, failure, and completion announcements vary by feature

The weekly rebuild and several commercial failures use appropriate live semantics. Global loading/offline/action toasts are plain `div`s, while Campaign, Pacing, Day, and inspector loading messages are unannounced prose. The same event type therefore changes accessibility behavior by route.

### P2 — Visible focus is removed from important inputs

Clients search and the command-palette input remove the native outline without a replacement. Other search components already demonstrate a usable `:focus-within` treatment, proving the correction is compatible with the current stack.

### P2 — Localization infrastructure is strong but incomplete at control level

`lang="he" dir="rtl"` is correct, central mixed-script primitives are a genuine strength, and the English baseline preserves Hebrew entity names safely. Remaining gaps include an English-only rail landmark and English MUI pagination names in Hebrew because only part of `localeText` is supplied.

## System divergence inventory

### Paint and typography

- 2,545 paint declarations use 72 written values; 94.2% reference variables.
- There are 48 normalized opaque colors: 21 central values and 27 off-token values, including near-duplicate neutrals and three spellings of white.
- 1,132 font-size declarations use 11 sizes; 1,129 are tokenized.
- 440 weight declarations use 10 CSS values; 425 are tokenized, while JavaScript introduces additional 620 and 700 weights.
- Inter is declared but never loaded. Three mono stacks compete.
- Eighteen literal line heights remain.

### Spacing, borders, elevation, dimensions, and layers

- 2,390 spacing declarations contain 216 compound values; 1,409 are tokenized and 750 still contain raw pixels.
- The most frequent raw atoms are 10px (138), 2px (125), 8px (91), 6px (80), 12px (59), 4px (52), and 16px (40).
- 456 radius declarations use 12 values; 441 are tokenized.
- 726 border declarations use 26 recipes; `1px solid var(--line)` appears 411 times.
- 36 shadow declarations use 15 recipes.
- 444 dimension declarations use 123 values; 237 are fixed pixels.
- 31 z-index declarations use 14 values. Twenty-two are raw and only nine reference named layers.
- Twenty-one media-query declarations use 13 conditions; 18 are width-based with 11 different breakpoints.

### Primitive fragmentation

- Buttons: 182 native buttons, 145 MUI Buttons, and one MUI IconButton.
- Fields: 28 native inputs, 14 selects, 4 textareas, 12 MUI TextFields, and 7 MUI Selects.
- Tables: 24 native tables and four uses of the shared `DataTable`.
- Overlays: 12 dialog surfaces, nine inline alert-dialog patterns, three explicit modal semantics, and one MUI Dialog.
- CSS naming exposes at least 55 card classes, 19 panel classes, 25 table classes, 24 drawer classes, 86 chip classes, seven badge classes, 12 status classes, and 36 tab classes.
- The shared Card abstraction is used once. `.page-panel` is used 28 times but does not define a complete primitive contract.
- Control heights compete at 28, 30, 34, 36, 40, 44, and 46px.
- Lucide is the sole live icon source, but 336 imports across 111 statements use 97 glyphs and ten explicit sizes from 10–22px.

### Runtime and maintenance

- Fifteen obsolete MUI `inputProps` usages emit React warnings; MUI 9 expects `slotProps.htmlInput`.
- Emotion emits an unsafe `:first-child` warning, likely through MUI X/theme interaction.
- Ninety fetch call sites across 44 files duplicate request behavior.
- Eight high-confidence unreachable modules total 1,438 lines: `ModelView`, `board-mount`, `GoldBreakManager`, `FrontierPanel`, `Inspector`, `OptimizerInventoryView`, `OptimizerRunPanels`, and `ScenarioCompare`. Two rules style modules are also likely unreachable.
- Approximately 87 CSS tokens are high-confidence unused; 272 more are potential dynamic/dead candidates and require guarded removal.
- Current guards accept 350 native controls across 97 files, 67 literal colors, and 54 hand-built cards across 24 files.
- There are no unit/component/E2E tests, ESLint rule set, type checker, or automated accessibility runner for these behaviors.

## What is already worth preserving

- The domain model is deep and specific: break placement, pricing derivation, retention confidence, licence, As Run truth, pod coverage, pacing, and makegood risk are not generic placeholders.
- Timeline breaks already support useful keyboard movement and resize operations.
- Plan restore and campaign ending show that careful review/confirmation patterns already exist.
- The long weekly rebuild uses a proper status announcement; several failures use alerts.
- Hebrew/English document direction and mixed-script primitives are thoughtfully centralized.
- Lucide already provides one coherent icon family.

## Measurable acceptance criteria

### Navigation and state

- A first-focusable skip link targets one localized, named `main`.
- Each workspace has exactly one H1.
- Browser Back/Forward restores the previous workspace, Clients view, selected entity, and week-plan step.
- Shareable URLs restore meaningful task state without leaking unrelated query parameters.
- Only the active route subtree mounts; inactive Model and Plan sections contribute no DOM, fetch, or DataGrid work.

### Dialogs and drawers

- Every modal has a localized accessible name, initial focus, contained Tab/Shift+Tab, Escape close, focus restoration, and inert/hidden background.
- While onboarding is open, the accessibility tree exposes only the onboarding task and necessary global announcement regions.
- Inspectors have a defined stacking policy and cannot leave an obscured layer interactive.

### Controls and keyboard

- Every interactive target is at least 24×24 CSS px with adequate separation.
- Primary controls, icon actions, tabs, and date selectors are at least 44×44px.
- Narrow timeline visuals use non-overlapping accessible activation proxies rather than tiny hit targets.
- Every control works by keyboard and has a non-clipped focus indicator with at least 3:1 contrast.
- Custom keyboard commands are discoverable through localized `aria-describedby` instructions.
- Tabs implement the complete ARIA pattern with one tab stop, Arrow/Home/End navigation, controlled panels, and correct associations.

### Data and consequence

- Every operational table has a caption or equivalent accessible name, scoped headers, and row-action names that include the affected entity.
- Every asynchronous container exposes `aria-busy`; progress and completion use `role="status"`; failures use `role="alert"`; failures never render as empty results.
- Every irreversible or broad-scope action has a localized review step naming the object, scope, and consequence; Cancel receives initial focus.
- Reversible actions provide an announced Undo.

### Localization and desktop support

- Both locales have correct document language/direction, isolated mixed-script values, and no English-only accessible names in Hebrew.
- The unsupported-device rule is documented and tested independently of content.
- Below the supported desktop condition, operational UI is unmounted or inert and the accessibility tree contains one localized H1 and a concise desktop explanation.
- The gate has no horizontal overflow on phone or tablet, portrait or landscape.
- The gate does not trigger solely because a low-vision desktop user zooms to 200–400%; the implementation must combine available viewport/feature evidence rather than punish zoom.

### Performance and regression

- No protected network request starts before authentication resolves.
- Shared data requests deduplicate; independent requests run concurrently.
- Campaign and pacing row rendering is windowed or progressively revealed, with stable selection and focus.
- Route-level code splitting prevents heavy inactive workspaces from entering the initial path.
- Overview, Objective, Overrides, Campaigns, Pacing, and onboarding have zero serious/critical automated accessibility findings, supplemented by manual keyboard and screen-reader passes through all three core jobs.
- Console is free of application-caused React, MUI, DataGrid, Emotion, accessibility, and unhandled-request warnings in the QA routes.
