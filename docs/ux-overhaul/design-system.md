# Studio Ledger design system

Status: implemented foundation, synchronized to the production direction and current structural-control boundary on 16 August 2026. This document is normative for new Kairos UI. It describes the system present in the application; known migration residue is called out explicitly rather than presented as finished work.

## Sources of truth

- Role tokens and compatibility aliases: [`tokens.css`](../../tv-break-dashboard/src/tokens.css)
- Canonical component styles: [`studio.css`](../../tv-break-dashboard/src/studio/studio.css) and [`studio-workspaces.css`](../../tv-break-dashboard/src/studio/studio-workspaces.css)
- Canonical React entry points: [`studio/actions.js`](../../tv-break-dashboard/src/studio/actions.js), [`studio/dom-controls.js`](../../tv-break-dashboard/src/studio/dom-controls.js), [`studio/modal.js`](../../tv-break-dashboard/src/studio/modal.js), and [`studio/index.js`](../../tv-break-dashboard/src/studio/index.js)
- React implementations behind that boundary: [`shell/dom-controls.jsx`](../../tv-break-dashboard/src/shell/dom-controls.jsx), [`shell/primitives.jsx`](../../tv-break-dashboard/src/shell/primitives.jsx), [`shell/data-table.jsx`](../../tv-break-dashboard/src/shell/data-table.jsx), and [`shell/modal-primitives.jsx`](../../tv-break-dashboard/src/shell/modal-primitives.jsx)
- MUI theme and RTL caches: [`shell/theme.js`](../../tv-break-dashboard/src/shell/theme.js)
- Seven-domain shell and desktop gate: [`shell/studio-shell.css`](../../tv-break-dashboard/src/shell/studio-shell.css) and [`shell/desktop-gate.css`](../../tv-break-dashboard/src/shell/desktop-gate.css)
- Direction rationale: [`phase-2-direction.md`](./phase-2-direction.md)

New work imports actions from `src/studio/actions`, structural native semantics from `src/studio/dom-controls`, modal mechanics from `src/studio/modal`, and readout/layout primitives from `src/studio`. It does not create a second button, status, overlay, table, or empty-state vocabulary. Feature CSS may own layout, but it must consume the role tokens below rather than redefining colour, type, radius, elevation, target size, or motion.

## Design premise

Studio Ledger is a warm material control surface for high-stakes broadcast and commercial operations. Near-black chrome gives stable orientation; cream and bone surfaces support long reading; mineral sage identifies focus, selection, and primary action without becoming decorative. Data remains dense, factual, and calm. Hierarchy comes from type, spacing, alignment, and material—not ornamental rules.

The canonical working composition is:

`domain rail → one-row shell header → list or board → detail/edit surface`

The shell header owns orientation and local navigation together. Its content row is 56px high with a 1px block-end boundary, so the rendered shell band is 57px. A domain with local navigation—including Broadcast and Governance—places that navigation in the bounded middle of the same row; it does not create a second header tier. The rail and header persist while only the operational workspace changes.

Mabat is a contextual dock. The company Model is a permission-gated Governance context with its own clearly marked shell. Phone and tablet layouts are intentionally replaced by the localized desktop gate; the operational console does not reflow onto them.

## Colour roles

| Token | Value | Use |
| --- | --- | --- |
| `--canvas` | `#f2eee4` | Application background |
| `--surface` | `#fbf8f0` | Primary working surface; never pure white |
| `--surface-muted` | `#eae4d7` | Recessed groups and disabled context |
| `--surface-raised` | `#ded6c7` | Focused records, sheets, menus |
| `--ink` | `#1d1b17` | Primary text; warm near-black |
| `--ink-muted` | `#5d574d` | Secondary text |
| `--ink-subtle` | `#625c52` | Lowest-emphasis text that remains text-safe on raised material |
| `--line` | `#d0c7b7` | Sparse structural separation |
| `--line-strong` | `#8f8572` | Input and selected boundaries |
| `--chrome` | `#1d1e1a` | Domain rail and high-authority surfaces |
| `--chrome-hover` | `#31312b` | Chrome hover/active material |
| `--accent` | `#526d62` | Mineral-sage identity, focus, and text on light surfaces |
| `--accent-strong` | `#344f47` | Filled primary action and high-emphasis selection |
| `--positive` / `--positive-soft` | `#376b50` / `#dfeadf` | Healthy, complete, compliant |
| `--warning` / `--warning-soft` | `#8c5b18` / `#f1e3c4` | Attention and operational risk |
| `--danger` / `--danger-soft` | `#9e3f38` / `#f2dcd7` | Refusal, destructive action, error |
| `--info` / `--info-soft` | `#3f6274` / `#dce7e9` | Read-only, modeled, informational |

Additional paired roles—`*-line`, `*-wash`, `--sport*`, on-strong fills, scrims, and heat-map fills—live in `tokens.css`. A semantic family owns foreground, boundary, and wash together; do not take one isolated colour from a family and use it decoratively.

Current measured pairings include 14.84:1 for `ink` on `canvas`, 6.17:1 for `ink-muted` on `canvas`, 5.71:1 for `ink-subtle` on `canvas`, 4.59:1 for `ink-subtle` on `surface-raised`, 15.79:1 for `surface` on `chrome`, 5.32:1 for `accent` on `surface`, and 8.40:1 for `surface` on `accent-strong`. MUI semantic fills use `surface` as their explicit contrast text; a component may not infer a text-safe combination merely because both values are tokens.

Rules:

- Colour never carries state alone. Pair it with copy, an icon, a shape, or native state semantics.
- Do not introduce `#000`, `#fff`, stock framework blue, decorative gradients, glows, or dark uniform shadows.
- Do not use a coloured edge, one-sided accent stripe, or ornamental border to create hierarchy.
- Prefer space to a rule. A border must describe a real boundary, selection, input, or table structure.

## Product mark

The shipping `KairosMark` is a transparent inline SVG on a 32×32 grid. It contains exactly two filled `currentColor` paths: two opposed frames interrupted by an off-axis negative-space splice. It has no background plate, raster fallback, gradient, shadow, letterform, play symbol, or equalizer bars. The same component is used in the rail, authentication/loading state, login, and desktop gate; only its size and inherited foreground change. It is a product mark, not a member of the 24px interface-icon family.

## Typography

Hebrew UI uses the local Hebrew-only `Noto Sans Hebrew Variable` subset, with IBM Plex Sans Hebrew as its compatibility fallback. Latin copy and figures use IBM Plex Sans; identifiers and code use IBM Plex Mono. Noto's Hebrew rhythm is calibrated independently while its subset deliberately contains no Latin glyphs, so mixed-script amounts, dates, channel names, and identifiers retain IBM's operational numeral and Latin voice.

| Role | Token | Size / line | Weight | Use |
| --- | --- | ---: | ---: | --- |
| Micro | `--type-micro-*` | 12 / 16px | 500 | Eyebrows and compact hierarchy; never a primary action |
| Data | `--type-data-*` | 12 / 18px | 500 | Table metadata, provenance, compact status |
| UI | `--type-ui-*` | 13 / 20px | 500–600 | Controls and dense table copy |
| Body | `--type-body-*` | 14 / 22px | 400 | Explanations, forms, detail evidence |
| Emphasis | `--type-emphasis-*` | 16 / 24px | 600 | Record and surface leads |
| Section | `--type-section-*` | 18 / 24px | 600 | Local section titles |
| Page | `--type-page-*` | 30 / 36px | 600 | Exactly one workspace H1 |
| Metric | `--type-metric-*` | 40 / 44px | 500 | Primary financial and operational figures |

Typography laws:

- Use the Hebrew stack in RTL and the Latin stack in LTR; do not set direction to fix alignment.
- Money, percentages, counts, dates, times, durations, table columns, and identifiers use tabular lining numerals.
- Identifiers, hashes, keyboard keys, and code use IBM Plex Mono.
- Isolate mixed-script values with the shared bidi primitives (`Figure`, `Name`, `Code`, `Prose`, `Numeric`), not Unicode spacing tricks.
- Body measure stays between 42ch and 68ch. Headings may use `text-wrap: balance`; prose must wrap naturally.
- Do not use `<br>`, non-breaking spaces, or fixed pixel widths to art-direct line breaks.
- A page has one H1. Subsequent levels follow the document hierarchy; visual size does not determine semantic rank.

## Spatial system

The system uses a 4px subgrid and an 8px primary rhythm.

| Token | Value |
| --- | ---: |
| `--space-half` | 2px |
| `--space-1` | 4px |
| `--space-1-5` | 6px |
| `--space-2` | 8px |
| `--space-3` | 12px |
| `--space-4` | 16px |
| `--space-5` | 20px |
| `--space-6` | 24px |
| `--space-8` | 32px |
| `--space-10` | 40px |
| `--space-12` | 48px |
| `--space-16` | 64px |

Structural dimensions:

| Token | Value | Contract |
| --- | ---: | --- |
| `--content-max` | 1680px | Maximum working canvas |
| `--desktop-min` | 1200px | Supported console threshold |
| `--rail-compact` | 88px | Rail from 1200–1399px |
| `--rail-expanded` | 96px | Rail at 1400px and wider |
| `--control-height` | 44px | Primary minimum control height |
| `--icon-action-size` | 44px | Icon action hit target |
| `--data-row-height` | 48px | Compact operational row |
| `--surface-inset-dense` | 20px | Dense bounded surface |
| `--surface-inset` | 24px | Default bounded surface |
| `--surface-inset-major` | 32px | Major work surface |

Do not invent intermediate gaps or target heights because a local composition is awkward. Change the composition or add a named role token only when the need repeats across the product.

The shell `.workspace` owns the route's outer inline gutter: 24px at 1400px and wider, 16px from 1200–1399px, and 40px at the block end. A direct `.page-workspace` therefore keeps its block padding but does not add a second inline gutter. Bounded surfaces own their own inset through `--card-inset`; a table or row band may reach the edge only through the named `CardBleed`/`.card-bleed` contract, and its first and last content columns still align to that inset. Resetting a data cell to `padding: 0` on both axes is prohibited when only its block padding needs removal.

## Shape, elevation, motion, and layers

- Controls use `--radius-control` (7px).
- Cards and bounded data groups use `--radius-surface` (10px).
- Sheets and dialogs use `--radius-overlay` (14px).
- `--radius-pill` is restricted to compact statuses; ordinary buttons are not pills.
- Elevation zero is the default. `--shadow-1` is for a raised working surface; `--shadow-2` is for sheets, menus, dialog frames, and drag state. Both share the same warm overhead light source.
- Motion uses 110ms (`--motion-fast`) for immediate response, 180ms (`--motion-standard`) for state or position, and 260ms (`--motion-continuity`) for a sheet or larger spatial transition.
- Standard motion uses `--ease-standard`; emphasized exit uses `--ease-exit`. Motion must explain cause or continuity, never decorate.
- Route and addressable-tab changes preserve the rail and header and transition only the workspace. When `document.startViewTransition` is available, the state update is committed atomically and the old/new workspace receives the causal out/in treatment; the root, rail, and header do not animate. Other browsers receive the same small workspace acknowledgement through the CSS fallback.
- Focus moves to the destination only after the transition boundary, so visual continuity never races semantic navigation. Failed or unavailable View Transitions fall back without delaying the update.
- `prefers-reduced-motion` removes both View Transition and fallback animation; the update and focus behavior remain intact.
- Named layers run from `--z-raised` through sticky, popover, drawer, feed, dialog, and toast. Feature CSS must not introduce an arbitrary z-index.

## Canonical component contracts

The current canonical API spans the four Studio entry points listed above; `src/studio/index.js` remains the aggregate readout/layout surface.

| Purpose | Canonical API | Contract |
| --- | --- | --- |
| Text action | `Button` from `studio/actions` | MUI variants under the Kairos theme; 44px minimum; loading disables action and announces status |
| Composite press mechanic | `ButtonBase` from `studio/actions` | Use only when a composite widget needs low-level press behavior; ordinary actions remain `Button` |
| Icon action | `IconButton` from `studio/actions` | Requires an accessible `aria-label`; 44×44px; icon is presentation only |
| Dependency-light pressable | `Pressable` from `studio/dom-controls` | Native button semantics behind the shared bridge; use for structural rows, timeline bands, and harness-sensitive controls |
| Specialized native field | `InputControl`, `SelectControl`, `TextAreaControl` | Native semantics behind the shared bridge for password, range, file, and other cases where a MUI field is not the right abstraction |
| Navigation link | Semantic `<a>` | Use a link for navigation and an action component for in-place work; there is no link-looking-button wrapper |
| Bounded material | `Card`, `CardBody`, `CardBleed` | `Card` owns material and inset; `CardBody` applies the inset; `CardBleed` lets bands reach the boundary while keeping first/last content aligned |
| State | `Status` / `StatusBadge` | Positive, warning, danger, info, or neutral; copy remains visible beside the mark |
| Headline value | `Metric` | Tabular figures, label, optional evidence/subline; never a freestanding decorative KPI tile |
| Workspace header | `PageHeader` | One H1, bounded measure, optional action |
| Persistent shell header | `renderTopBar` + `context-local-nav` | One 57px band for domain title, bounded local navigation, connection state, and utilities; never a stacked local-nav row |
| Local navigation | `renderTopBar` + `context-local-nav` + canonical actions | Addressable item, `aria-current`, 44px targets, horizontal overflow when necessary |
| Empty, loading, failure | `EmptyState`, `LoadingState`, `ErrorState` | Empty teaches; loading holds meaning and uses status; failure is an alert with recovery when available |
| Focused overlay | `Dialog` from `studio/modal` | Native modal dialog, localized name, initial focus, Escape/backdrop policy, focus restoration |
| Contextual overlay | `Sheet` from `studio/modal` | Same modal mechanics, logical start/end placement, task-sized width |
| Operational data | `DataTable` | Lazy MUI Data Grid, 48px row/header, bidi-aware cells, localized pagination, honest empty label |
| Direction and values | `DirectionRoot`, `Figure`, `Name`, `Code`, `Prose`, `Numeric` | Establish direction once and isolate mixed-direction values at their boundary |

`Surface`, `CardHead`, `LocalNav`, `LinkButton`, and `VisuallyHidden` are deleted React wrappers. `Card` is the canonical bounded-material primitive; semantic headers remain ordinary markup inside it, local navigation is owned by the shell, links retain link semantics, and live-region/status copy uses ordinary semantic markup rather than a component wrapper.

Fields currently use the themed MUI `TextField`, `Select`, checkbox, radio, switch, toggle, menu, and tab contracts, plus the four structural bridges above. There is not yet a standalone `Field` export. Do not create one ad hoc inside a feature; extend the Studio layer when a reusable field abstraction is actually required. Screen modules contain no raw `<button>`, `<input>`, `<select>`, or `<textarea>` tags; the only four such tags are the implementations in `shell/dom-controls.jsx`.

## State matrix

Every component must implement the states relevant to its purpose. Absence of a state is a design defect, not permission to improvise locally.

| State | Required behavior |
| --- | --- |
| Default | Uses role tokens and a truthful semantic element |
| Hover | Changes material or boundary without moving layout |
| Focus-visible | 3px `--focus-ring`, 2px offset, never clipped or removed |
| Active/pressed | Gives immediate causal feedback; no decorative animation |
| Selected/current | Uses semantic state plus surface/boundary; never colour alone |
| Disabled/refused | Is not actionable; names why when the refusal affects task completion |
| Loading | Preserves the expected structure where known; container uses `aria-busy`; status is announced |
| Success | Names what changed and the affected object; reversible work offers Undo |
| Warning/partial | Distinguishes missing, unknown, stale, and partial from zero |
| Error | Uses `role="alert"`; states what failed and the available recovery |
| Empty | States why there is no data and offers the next valid action when one exists |
| Long/overflow | Text wraps; data controls progressively reveal or scroll on the intended axis; no clipping of meaning |
| RTL/LTR | Logical properties and bidi primitives preserve order and alignment in both locales |
| Reduced motion/forced colours | Remains operable and visibly focused under user preferences |

## Interaction laws

- Nothing important is small. Primary controls, icon actions, tabs, and date selectors are at least 44×44 CSS px; compact data rows are at least 48px high.
- One conceptual action stays in one view. Micro-edits are inline or in a popover; contextual record work uses a sheet/inspector; a single focused transaction uses a dialog; a genuinely multi-step transaction uses a workflow.
- Do not stack live interactive inspectors. If one contextual record opens above another, the underlying record becomes inert and hidden from the accessibility tree.
- Destructive or broad-scope actions enter an explicit review state that names object, scope, and consequence. Cancel receives initial focus. Reversible actions announce Undo.
- Active-only panels mount. Hidden tabs must not fetch, run Data Grid, or remain interactive.
- Long operational lists progressively reveal stable windows and keep totals visible. Current implemented windows include 12 campaigns, 16 pacing campaigns, 18 agency/advertiser record rows, 20 break rows, and 12 manual decisions.
- Timeline visuals may render at true scale, but tiny geometry is never the only activation target. Use a non-overlapping 44px selection/editing proxy.
- Browser Back and Forward are part of the interaction model. User navigation pushes history; normalization alone replaces it.
- Keyboard behavior is taught where it applies. Tabs implement one tab stop plus Arrow/Home/End. Command-style lists use `aria-activedescendant` or move DOM focus consistently.

## Layout and navigation laws

- The seven global domains are Today, Plan, Broadcast, Commercial, Sources, Governance, and History.
- Domain navigation does not duplicate local sections. Plan steps, Broadcast views, Commercial records, Source views, Governance sections, and Model sections remain contextual and addressable.
- The shell header remains a single 57px band. Local navigation scrolls on its own inline axis when necessary; it never wraps, creates a second tier, or reduces a target below 44px.
- Route and local-tab transitions preserve the rail/header spatial frame. Only the workspace acknowledges the cause, and Back/Forward uses the same continuity contract.
- Use CSS logical properties (`inline`, `block`, `start`, `end`) for structural layout. Physical left/right is acceptable only where the data itself represents a physical direction.
- The workspace canvas is desktop-only. Below the supported condition, mount `DesktopGate` before session and data trees; do not leave hidden operational UI authenticating or fetching behind it.
- The zoom exception in the desktop gate is a support heuristic, not permission to build a narrow desktop composition. The console itself still assumes the desktop working canvas.

## Content and data laws

- Every consequential figure states scope, basis, freshness, and modeled-versus-observed status close enough to prevent misreading.
- Unknown is not zero. Unreachable is not empty. Partial is not complete. Never synthesize a number or a successful state to fill a gap.
- Row actions include the affected entity in the accessible name.
- Operational tables have a caption or equivalent accessible name, scoped headers, stable row identity, and visible totals.
- Errors say what happened and what the operator can do next. Success messages name the object changed.
- Hebrew and English copy are authored as product copy, not machine-shaped word swaps. Dynamic names and identifiers are isolated at insertion.

## Building the next screen

Before a screen is accepted, verify all of the following:

1. It belongs to one of the seven domains and uses an addressable local state instead of adding a global entrance.
2. It imports the appropriate Studio entry point and uses role tokens only.
3. It has one H1, a localized named `main`/region, correct tab semantics, and 44px primary targets.
4. Empty, loading, partial, error, success, permission-refused, long-content, RTL, and LTR states are designed.
5. A write shows scope and consequence before commitment and announces the outcome afterward.
6. Dense rows are progressively revealed or virtualized; inactive panels do not mount.
7. Mixed-direction values use shared bidi primitives; figures use tabular numerals.
8. Back/Forward restores task state and unrelated query parameters do not leak into the URL.
9. No one-sided decoration, pure black/white, arbitrary radius/shadow/z-index, manual line break, emoji icon, or second icon source has been introduced.
10. The screen is inspected in the browser and included in the relevant visual, keyboard, console/network, and accessibility evidence before a release claim is made.

## Migration boundary

The guarded structural migration is complete at its declared boundary:

- zero screen-native button/input/select/textarea tags, with exactly four native bridge tags in `shell/dom-controls.jsx`;
- zero hand-built-card recipe violations;
- 159 JS/JSX modules importing at least one Studio entry point: 123 action consumers, 75 structural-control consumers, 13 modal consumers, and 13 aggregate readout/layout consumers (the sets overlap);
- zero direct MUI `Button`, `ButtonBase`, or `IconButton` imports outside `studio/actions.js`; and
- zero feature imports from `shell/primitives`, `shell/dom-controls`, or `shell/modal-primitives`.

That is not a claim that every framework or stylesheet dependency has disappeared. Forty-seven modules still import non-action MUI APIs directly, 121 import Lucide directly, and two screen-level modal implementations remain outside `studio/modal`: the MUI suspension dialog in `AgencyDetailDrawer` and the native command-palette dialog. The CSS corpus and quarantined compatibility modules also remain. These facts are recorded in [`divergence-resolution.md`](./divergence-resolution.md) and [`deprecation-ledger.md`](./deprecation-ledger.md); they must not be mistaken for permission to add another local primitive or raw value.
