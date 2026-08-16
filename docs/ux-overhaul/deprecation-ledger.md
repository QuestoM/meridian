# UX overhaul deprecation ledger

This ledger records routes, source modules, mounting patterns, and interaction compositions that were folded away or quarantined. “Not in the bundle” means the current application import graph does not reach the module and the 16 August 2026 Vite build emitted no matching implementation string; it does not mean the source file has been deleted.

## Removed runtime architecture

| Deprecated item | Previous behavior | Replacement | Disposition and reason |
| --- | --- | --- | --- |
| `src/model/console-bridge.jsx` (211 lines) | Created a second React root over the operator shell, observed auth DOM mutations, and manually muted the underlying root | Lazy `ModelConsole` route owned by `TVBreakDashboard` | Retained as an unreachable compatibility export for downstream standalone mounts. The live shell does not import it |
| `src/model/console-mount.js` (5 lines) | Side-effect entry that can mount the compatibility bridge | Shell route-level lazy import | Retained only as an unreferenced downstream entry. Ordinary app evaluation never reaches it |
| `src/model/candidates/board-mount.jsx` (46 lines) | Standalone `createRoot` helper used to expose CandidateBoard outside the console | Direct `CandidateBoard` import inside the Model candidates section | Retained as a side-effect-free regression/integration harness; it has no inbound application import |
| Duplicate application roots | `index.jsx` plus the Model bridge/board helper could create independent roots | One reachable `ReactDOM.createRoot` call in `src/index.jsx` | Removed from the live import graph. Compatibility mount functions still exist but are unreachable unless an external integrator imports them explicitly |
| All Plan panels mounted hidden | Six weekly sections existed in the DOM/fetch lifecycle simultaneously | `PlanWeek` mounts only the addressed active section | Removed to reduce inactive work, ambiguity, and accessibility noise |
| Model overlay over a live operator shell | Two shells could be simultaneously present and required manual inerting | Model replaces the operator route subtree and provides explicit ways back | Removed; company context remains visually distinct without a duplicate root |

The live application removes the duplicate-root lifecycle without removing Model gates, coverage, drift, candidates, training, versions, provenance, recording, or links to governing Rules/Calendar surfaces. The three quarantined mount files total 262 lines and contribute zero live-bundle weight; their explicit deletion condition is migration of the remaining external harnesses and integrators.

## Deleted component wrappers

The source-wide structural migration removed five obsolete React wrappers rather than preserving aliases:

| Deleted wrapper | Current contract |
| --- | --- |
| `Surface` | `Card`, `CardBody`, and `CardBleed` own bounded material and inset behavior |
| `CardHead` | Semantic `header`/heading markup sits inside Card; `.card-head` remains a class-level inset contract where legacy markup needs it |
| `LocalNav` | `renderTopBar` owns `context-local-nav` and renders canonical Studio actions with address/current semantics |
| `LinkButton` | Navigation remains a semantic link; in-place work uses `Button` or `Pressable` |
| `VisuallyHidden` | Live-region/status markup is authored directly; no React wrapper remains |

A current source scan finds no declaration, export, or import of those wrapper identifiers. Their names must not be reintroduced as compatibility shortcuts.

## Legacy addresses preserved as aliases

These visible destinations are deprecated as global information architecture, not as capabilities. [`shell/nav.js`](../../tv-break-dashboard/src/shell/nav.js) resolves them into seven canonical domains and applies a scoped default only when a more specific valid query value is absent.

| Legacy hash | Canonical destination |
| --- | --- |
| `#Overview` | `/#Today` |
| `#Optimizer` | `/?plan=objective#Plan` |
| `#Schedule` | `/?plan=board#Plan` |
| `#Inventory` | `/?plan=supply#Plan` |
| `#Forecasts` | `/?plan=compare#Plan` |
| `#Break Library` | `/?broadcast=library#Broadcast` |
| `#Overrides` | `/?broadcast=decisions#Broadcast` |
| `#Campaigns` | `/?clients=campaigns#Commercial` |
| `#Advertisers` | `/?clients=advertisers#Commercial` |
| `#Agencies` | `/?clients=agencies#Commercial` |
| `#Data` | `/?sources=inputs#Sources` |
| `#Reports` | `/?sources=downloads#Sources` |
| `#Settings` | `/#Governance` |
| `#Calendar` | `/?rules=calendar#Governance` |
| `#Pricing` | `/?rules=rate_card#Governance` |
| `#Versions` | `/#History` |
| `#Assistant` | Last safe workspace plus open Mabat dock |
| `#Model` | Permission-gated company Model inside Governance context |

Removal condition: collect evidence that no supported bookmark, integration, test, or user workflow depends on an alias; then ship an explicit redirect/migration notice before deleting it.

## Zero-bundle compatibility source

The following six files remain in source but are not imported by the runtime application. They contain 514 lines; together with the three Model mount shims above, the nine-module compatibility set contains 776 lines. The production build emitted no `OptimizerWorkspace`, `SchedulePage`, `InventoryPage`, `ForecastsPage`, `OverrideConsole`, `ModelView`, or compatibility-mount implementation string.

| Module | Lines | Why it remains | Live replacement | Removal condition |
| --- | ---: | --- | --- | --- |
| `plan/week/OptimizerWorkspace.jsx` | 32 | Preserves the old exported entry shape and current smoke-guard fragment | Direct lazy `PlanWeek`, `plan=objective` | Update downstream imports/guards to canonical routes |
| `plan/week/SchedulePage.jsx` | 26 | Preserves old schedule entry contract | Direct lazy `PlanWeek`, `plan=board` | Same |
| `plan/week/InventoryPage.jsx` | 25 | Preserves old inventory entry contract | Direct lazy `PlanWeek`, `plan=supply` | Same |
| `plan/week/ForecastsPage.jsx` | 30 | Preserves old forecasts entry contract | Direct lazy `PlanWeek`, `plan=compare` | Same |
| `plan/day/OverrideConsole.jsx` | 114 | Preserves the former two-tab component/API and named export used by older integration code | Broadcast router mounts `DayPage` or `OverrideDecisions` directly | Confirm no external import/test needs the wrapper and update guard fixtures |
| `model/ModelView.jsx` | 287 | Retains the old explainability/parameter-ledger implementation as a quarantined compatibility/reference module | Company `ModelConsole` plus Governance channel/rules/levers | Verify every parameter-ledger capability has an owned Governance home, then delete |

Because these modules are outside the import graph, they add source maintenance cost but not initial or async production chunk weight. New work must not import them to “reuse” the old entrance; it must target the canonical workspace.

## Known unreferenced legacy candidates

Phase 1 identified additional modules that remain unreferenced. A current import scan and production build keep the following 888 lines out of emitted chunks:

- `plan/break/GoldBreakManager.jsx` — 157 lines
- `plan/week/FrontierPanel.jsx` — 270 lines
- `plan/week/Inspector.jsx` — 173 lines
- `plan/week/OptimizerInventoryView.jsx` — 59 lines
- `plan/week/OptimizerRunPanels.jsx` — 229 lines

They were not deleted in this pass because removal must be tied to a capability/test audit rather than filename age. Their live behaviors are largely represented by the Plan surface, Frontier chart, active inspectors, and Break/Override flows, but that equivalence still needs a named assertion before deletion.

`plan/week/ScenarioCompare.jsx` was retired on 2026-08-16 after a zero-inbound import scan. Its live capability is the guarded Plan Compare flow; `test_retired_direct_compare_surface_has_no_live_inbound_path` keeps the legacy direct POST from returning.

## Folded interaction compositions

| Deprecated composition | Replacement | Capability treatment |
| --- | --- | --- |
| Four global Plan entrances plus six local sections | One Plan domain with six addressable steps | Objective, run, compare, publish, supply, board, keyboard commands, recommendations, and day drill remain |
| Break Library page stacking pod, day table, and ranked library | Broadcast context plus active-only Library/Day/Pod views | All views remain addressable; only one mounts at a time |
| Day editor concatenated with full manual-override application | Separate Broadcast Day and Manual Decisions contexts | Day direct manipulation and manual pin/forbid/force/gold capabilities remain |
| Three Commercial global entrances | One Commercial domain with Clients, Money, Campaigns, Delivery, Pricing rules, Agencies | Records and write paths remain; related entity drills open in context |
| Reports and Data as separate global destinations | One Sources domain with Inputs, Files, Reports | Checks, findings, commit, preview, and download remain |
| Calendar and Pricing as redirect-like global destinations | Governance local sections | Calendar events/activation/import and staged rate-card editing remain |
| Restore-point-only destination | One History timeline over change, run, restore, account, and restore-point records | Restore points, diff, rename, create, and selective restore remain |
| Long one-page onboarding form | Four-step workflow with final review | Agency/client/campaign/terms/flights remain one submit transaction |
| Tiny timeline geometry as the only target | True-scale visual plus `DayBreakNavigator` proxy | Pointer manipulation remains; keyboard/select/open target is 44px |
| Immediate destructive row action | Inline review with named consequence and Cancel-first focus | Delete/end/remove capability remains, with a safer commitment boundary |

## Migration residue, not a second approved system

- `tokens.css` intentionally carries compatibility aliases such as `--bg`, `--muted`, `--teal`, and legacy type/radius names. They resolve onto Studio roles and exist to prevent feature breakage during migration.
- `Card`, `CardBody`, and `CardBleed` are the bounded-material contract. The card guard has no quarantines or exception budgets and reports zero hand-built-card recipes.
- The smoke guard reports zero raw screen button/input/select/textarea tags and exactly four native bridge tags in `src/shell/dom-controls.jsx`.
- Across overlapping sets, 159 JS/JSX modules consume Studio entry points: 123 actions, 75 structural controls, 13 modal mechanics, and 13 aggregate readout/layout consumers.
- Direct MUI `Button`, `ButtonBase`, and `IconButton` imports outside `src/studio/actions.js` are zero. Feature imports from `shell/primitives`, `shell/dom-controls`, and `shell/modal-primitives` are also zero.
- Two screen-level modal implementations remain outside `studio/modal`: `AgencyDetailDrawer` uses a MUI Dialog for suspension review, and `CommandPalette` owns a native dialog. They are named migration residue, not alternate modal authorities.
- Forty-seven modules still import non-action MUI APIs directly and 121 import Lucide directly.
- Feature-specific Studio sheets coexist with older feature CSS. The new sheets own the corrected role/token layer; dead selector removal still requires guarded reachability work.

Do not add new compatibility aliases, old-route imports, raw screen button/input/select/textarea tags, hand-built card recipes, or duplicate CSS recipes. Preserve the zero structural budgets and reduce the remaining ledgers in the same change that removes their final dependency.
