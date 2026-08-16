# Divergence resolution — current source snapshot

- Snapshot: 16 August 2026, after structural control/card convergence; browser evidence remains separately governed
- Baseline: `c03dd34128ef` (`main`), measured in [`phase-1-audit.md`](./phase-1-audit.md)
- Scope: static frontend source under `tv-break-dashboard/src`; rendered-route results belong in the QA report

## Outcome

The runtime now has one deliberate product direction: warm cream and bone work surfaces, near-black ink/chrome, restrained mineral sage for focus/action/state, Noto Sans Hebrew for Hebrew glyphs, IBM Plex Sans for Latin and figures, and IBM Plex Mono for identifiers. The same role tokens govern MUI, the shell, authentication, the desktop gate, and feature correction layers. The shipping mark is one transparent two-path frame-splice SVG.

The source has moved through the earlier “primitives exist but nobody imports them” state and the later limited-consumer snapshot. Actions now enter through `src/studio/actions.js`, dependency-light native semantics through `src/studio/dom-controls.js`, modal mechanics through `src/studio/modal.js`, and readout/layout primitives through `src/studio/index.js`. Across those overlapping entry points, 159 JS/JSX modules are consumers. Screens contain zero raw button/input/select/textarea tags; their four native implementations live only in `src/shell/dom-controls.jsx`. The card guard likewise has no exceptions and reports zero hand-built-card recipes.

CSS is more maintainable but not smaller than the Phase 1 baseline. The former 6,224-line `shell/styles.css` has been divided by responsibility, and every JavaScript, JSX, and CSS file is now at or below 450 lines. The current CSS corpus contains 108 files and 24,777 lines. The earlier claim of a fixed “20 canonical files / 64 residue files” is retired: after the shell split, filename age is not a defensible proxy for whether a live rule owns layout, adapts a legacy surface, or makes a normative system decision. Exact whole-corpus counts are reported; deletion still requires reachability and capability evidence.

The honest state is:

- **Resolved at the product-system boundary:** one palette, script-specific local typography, one MUI theme, one master mark, one shell geometry, causal navigation continuity, role-owned padding, no off-token CSS colours, and no banned one-sided accent bars.
- **Resolved at the guarded structural boundary:** zero raw screen button/input/select/textarea tags, exactly four canonical native bridges, zero hand-built-card recipes, zero direct MUI action imports outside `studio/actions.js`, zero feature imports from shell implementation primitives, deleted obsolete wrappers, and a source-wide 450-line cap.
- **Still open as migration work:** two screen-level modal implementations outside `studio/modal`, 47 modules with direct non-action MUI imports, 121 direct Lucide consumers, compatibility modules, and 24,777 CSS lines whose safe consolidation must be proved route by route.

## Measurement boundary and reproducibility

The exact file and line totals were read from the working tree with `find` plus `wc -l`. Counts include imported, lazy, and unreferenced source. “Source” in this document means `.js`, `.jsx`, and `.css`; it does not include font binaries, licence text, screenshots, generated evidence, or build output.

From `tv-break-dashboard/`:

```sh
find src -type f -name '*.css' -print0 | sort -z | xargs -0 wc -l
find src -type f \( -name '*.js' -o -name '*.jsx' \) -print0 | sort -z | xargs -0 wc -l
find src -type f \( -name '*.js' -o -name '*.jsx' -o -name '*.css' \) -print0 | xargs -0 wc -l
rg -l "studio/actions" src --glob '*.{js,jsx}'
rg -l "studio/dom-controls" src --glob '*.{js,jsx}'
rg -l "studio/modal" src --glob '*.{js,jsx}'
rg -l "from ['\"][^'\"]*studio(?:/index(?:\\.js)?)?['\"]" src --glob '*.{js,jsx}'
npm run test:guards
npm run build
```

The earlier lightweight rule/selector scanner was not retained as an executable repository tool. Its historical figures remain in the Phase 1 audit, but this current snapshot does not copy parser-derived counts forward without a reproducible implementation. File and line totals plus executable guard results are the authoritative source boundary here.

The 16 August build and guards completed before this document was updated: Vite transformed 3,627 modules; card, direction, date, accent, colour, and smoke guards passed. Final route/browser evidence is deliberately not inferred from those static/build results.

## Current source corpus

| Source family | Files | Lines | Reading |
| --- | ---: | ---: | --- |
| CSS | 108 | 24,777 | Whole authored stylesheet corpus |
| JavaScript | 118 | 18,268 | `.js` only |
| JSX | 226 | 44,286 | `.jsx` only |
| JavaScript + JSX | 344 | 62,554 | Runtime and compatibility code |
| **JS + JSX + CSS** | **452** | **87,331** | Exact source boundary used here |

No `.js`, `.jsx`, or `.css` file exceeds 450 lines in this snapshot. These values are a worktree snapshot: evidence-script or production edits made after 16 August must rerun the commands rather than copy the numbers forward.

## Before → current exact CSS boundary

| Metric | Phase 1 published | 16 August worktree | Reading |
| --- | ---: | ---: | --- |
| CSS files | 65 | 108 | +43 / +66.2%; responsibility splitting improved ownership without reducing the file count |
| CSS lines | 20,669 | 24,777 | +4,108 / +19.9%; source-size convergence is not complete |
| Largest CSS file | 6,224 | 450 | The monolith is gone and the source-wide cap holds |

The Phase 1 rule-block, selector, class, repeated-selector, raw-pixel, inline-style, and “hard dimensions” predicates are not reconstructed here. Replacing them with new definitions would create false comparability.

## CSS architecture after the shell split

The shell split is a maintainability result, not a line-count reduction:

| Sheet | Lines | Responsibility |
| --- | ---: | --- |
| `shell/styles.css` | 313 | Base shared shell/legacy compatibility rules |
| Largest `styles*.css` sibling | 400 | Commercial shell layout; every other sibling is smaller |
| `shell/studio-shell.css` | 442 | Canonical rail, workspace, header, and desktop-shell material |
| `shell/shell-continuity.css` | 120 | One-row local-nav composition and causal transition layer |

The 13 `styles*.css` files together replace the old monolith by named responsibility. `tokens.css`, Studio primitives/styles, font roles, card inset ownership, shell geometry, continuity, desktop gate, and named feature layers are normative decision sources. Older and newer feature sheets may both remain live layout providers. A rule becomes removable only when import reachability, rendered-route inspection, and capability regression evidence agree.

## Colour, type, mark, and shell

| Contract | Current fact | Resolution reading |
| --- | --- | --- |
| CSS colour literals outside `tokens.css` | 0 | Resolved and guarded |
| JavaScript colour literals | 31, all in `shell/theme.js` | Controlled MUI duplication remains; feature JS has none |
| Hebrew type | Local Hebrew-only Noto Sans Hebrew Variable 100–900 | Primary Hebrew voice |
| Latin and figures | IBM Plex Sans 400/500/600 | Deliberate mixed-script/numeral voice |
| Identifiers | IBM Plex Mono 400/500 | Deliberate code/identifier voice |
| Font output | 17 local font files; three critical WOFF2 preloads | No runtime font CDN |
| Product mark | One transparent, two-path, 32px-grid frame-splice SVG | Same component at rail/loading/login/gate sizes; generated raster explorations do not ship |
| Shell header | 56px content row + 1px boundary | 57px rendered, including Broadcast/Governance local navigation |

The palette contract remains clean in CSS: all stylesheet colours originate in [`tokens.css`](../../tv-break-dashboard/src/tokens.css). [`shell/theme.js`](../../tv-break-dashboard/src/shell/theme.js) repeats the values and interaction alpha recipes for MUI, so token/theme synchronization remains a source-level follow-up even though feature code no longer carries colour literals.

## Navigation continuity and padding ownership

[`shell/workspace-continuity.js`](../../tv-break-dashboard/src/shell/workspace-continuity.js) feature-detects `document.startViewTransition`, commits route updates atomically, and focuses the destination after the transition boundary. [`shell/shell-continuity.css`](../../tv-break-dashboard/src/shell/shell-continuity.css) suppresses root/rail/header animation and gives only the old/new workspace a small causal out/in cue. Browsers without the API receive a workspace-only fallback. Reduced-motion preference removes both animations while preserving update and focus semantics.

Padding now has explicit owners:

- `.workspace` supplies the route's outer desktop gutter and block-end space;
- a direct `.page-workspace` retains block padding but does not double the shell's inline gutter;
- `Card`/`.card` owns `--card-inset`;
- `CardBleed` reaches a boundary only as a named opt-in while retaining first/last-column alignment;
- status-grid cells remove block padding only, so the leading column no longer runs flush to the card edge; and
- the Today decision scope and revenue disclosure restore insets that feature overrides had erased.

The evidence harness measures main/route padding, four logical edge insets, header/navigation heights, and under-12px edge contacts with a named full-bleed allowlist. Its current output is stored separately in [`final-cream-matrix-v2`](./evidence/after/final-cream-matrix-v2/aggregate.md); this structural document does not restate that matrix as independent QA or release certification.

## Public Studio API and control adoption

The current source import topology is:

| Entry point | Consumer modules | Responsibility |
| --- | ---: | --- |
| `src/studio/actions.js` | 123 | `Button`, `ButtonBase`, and `IconButton` |
| `src/studio/dom-controls.js` | 75 | `Pressable`, `InputControl`, `SelectControl`, and `TextAreaControl` |
| `src/studio/modal.js` | 13 | `Dialog`, `Sheet`, focus-first, and focus-return mechanics |
| `src/studio/index.js` | 13 | Card/readout/state/data-table primitives |
| **Unique consumers across the four paths** | **159** | Sets overlap; this is not their arithmetic sum |

The specialized entry points are now the source-wide import contract. Six modules consume `IconButton` through `studio/actions`. `Surface`, `CardHead`, `LocalNav`, `LinkButton`, and `VisuallyHidden` have no React declaration, export, or import in current source. `Card`, `CardBody`, and `CardBleed` are the canonical bounded-material composition.

| Guarded debt | Baseline | Current | Reading |
| --- | ---: | ---: | --- |
| Screen-native control render points | 350 across 97 files | 0; four bridge tags in one boundary file | Guarded screen migration complete |
| Budgeted hand-built card recipes | Guard debt present across 24 files | 0 | Guard has no quarantine or exception budget |
| Studio consumer modules | 0 | 159 unique | Specialized source-wide boundary adopted |
| Direct MUI action imports outside `studio/actions.js` | Not measured | 0 | `Button`, `ButtonBase`, and `IconButton` have one import authority |
| Feature imports from shell primitive implementations | Not measured | 0 | Features consume Studio entry points |
| Off-token CSS colours | 67 | 0 | Resolved |
| Banned one-sided accents | 0 | 0 | Zero budget preserved |

`npm run test:card` now proves that the card recipe and inset live only in `shell/card.css`; there is no exception budget to raise. `npm run test:smoke` proves screens have zero raw button/input/select/textarea tags and the whole source tree has exactly the four canonical tags in `shell/dom-controls.jsx`.

## Resolution ledger

| Finding | Status | Current evidence | Closure condition |
| --- | --- | --- | --- |
| Warm cream / near-black / mineral-sage direction | Resolved | Tokens, MUI theme, shell, feature layers | Keep feature colours at zero outside tokens |
| Script-specific local typography | Resolved at runtime | Noto Hebrew + IBM Latin/Mono; 17 local outputs | Remove legacy stack declarations with their owning selectors |
| Master mark | Resolved | One two-path transparent SVG used in four product contexts | Keep generated/raster alternates out of runtime |
| Tall/multi-row shell header | Resolved in code | One 57px band; local nav shares the row | Preserve the invariant; rendered evidence remains separately governed |
| Causal page/tab continuity | Resolved in code | View Transition path, fallback, reduced-motion path, focus boundary | Multi-browser navigation/motion release evidence remains separate |
| Missing/inconsistent padding | Resolved by ownership; instrumented for evidence | Workspace/card owners plus edge-contact scan | Preserve the zero-budget edge contract without widening the allowlist casually |
| CSS file maintainability | Improved | Every JS/JSX/CSS file ≤450; shell monolith split | Continue responsibility-based consolidation |
| CSS corpus size | Unresolved | 108 files / 24,777 lines | Delete only with reachability, capability, and visual evidence |
| Studio action/control boundary | Resolved in source | 159 consumers; zero direct MUI action imports; zero feature-to-shell primitive imports | Keep specialized imports and the zero budgets guarded |
| Native screen controls | Resolved in source | Zero screen tags; four canonical bridge implementations | Keep raw tags confined to `shell/dom-controls.jsx` |
| Hand-built cards | Resolved in source | Zero recipes and no guard exceptions | Keep material/inset ownership in Card and `shell/card.css` |
| Modal boundary | Canonical path established; residue remains | 13 Studio modal consumers; one direct MUI Dialog and one direct native command-palette dialog | Migrate the two named exceptions without weakening their focus/consequence behavior |
| Cross-language palette source | Partial | 31 JS colour literals in `theme.js` | Generate both outputs or add exact synchronization coverage |

## Final interpretation

The overhaul establishes one authoritative rendered system and a source-wide structural import boundary. It also replaces the tall/staged shell with a compact persistent frame, gives navigation a causal motion contract, and makes padding ownership testable rather than anecdotal.

It does not erase the historical implementation. Release language may say that screen-native controls and hand-built card recipes are at zero, action imports are canonical, feature code no longer reaches shell primitive implementations, and every source file is bounded. It must not say that CSS, all MUI/Lucide use, modal implementations, or compatibility source are fully consolidated, and it must not convert a build/static scan into the final browser-evidence claim.
