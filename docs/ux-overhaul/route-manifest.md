# Kairos route manifest

Captured from the current frontend source on 16 August 2026. This is the acceptance map for the UX overhaul: 34 addressable operational surfaces, plus authentication, the desktop-support gate, and the contextual assistant. Every canonical row below is covered by the [final-cream read-only browser matrix](./evidence/after/final-cream-matrix-v2/aggregate.md).

## Product route contract

- Kairos is a desktop broadcast-revenue operations console. Operational workspaces mount only on a supported desktop canvas.
- The canonical shell has seven domains: Today, Plan, Broadcast, Commercial, Sources, Governance, and History. Company Model is permission-gated inside the Governance domain.
- The shell router is hash-based. Query parameters own the state inside a domain; switching domains removes parameters owned by other domains.
- Hebrew/RTL and English/LTR use the same capability map. Locale is persisted as `kairos.locale`, not encoded in the route.
- The persistent operator-shell header is 57px in all 108 measured operator-route captures; Governance and Broadcast keep their 56px local navigation in that one row instead of stacking a second header. Company Model uses its own permission-gated console shell.
- The assistant is contextual. `#Assistant` opens its dock over the current non-Model workspace instead of mounting an eighth operator workspace.
- Existing legacy hashes continue to resolve into the canonical map. They are compatibility entrances, not parallel information architecture.

The router contract is implemented in [`src/shell/nav.js`](../../tv-break-dashboard/src/shell/nav.js), and workspace mounting is implemented in [`src/shell/workspace-router.jsx`](../../tv-break-dashboard/src/shell/workspace-router.jsx).

## Supported-canvas states

| State | Trigger | Mounted content | Evidence status |
| --- | --- | --- | --- |
| Desktop console | CSS viewport at least 1,200px, or the conservative desktop-zoom exception | Session boundary followed by the operational shell | Read-only matrix pass: 34/34 routes, 136/136 HE/EN desktop captures, 544/544 route PNGs |
| Desktop-required gate | Unsupported viewport below 1,200px | One localized `main`, H1, explanation, requirement, and locale control; no operational shell | HE and EN pass at an exact 1024×768; 8/8 PNGs and no API resource or network request |
| Signed out | Supported desktop with configured authentication and no session | Hebrew-first controlled login | Targeted auth/network and native-dialog probes pass; full role matrix remains pending |
| Open-access deployment | Authentication deliberately disabled server-side | Shell labels the state honestly; it is not treated as a successful sign-in | Read-only QA mode, not permission-matrix proof |

The gate is defined by [`src/shell/desktop-gate.jsx`](../../tv-break-dashboard/src/shell/desktop-gate.jsx). The authentication boundary is in [`src/shell/Login.jsx`](../../tv-break-dashboard/src/shell/Login.jsx).

## Canonical operational surfaces

URL examples are relative to the application root. Query order is not significant.

### Final canonical evidence status

The [aggregate report](./evidence/after/final-cream-matrix-v2/aggregate.md) and [machine-readable ledger](./evidence/after/final-cream-matrix-v2/aggregate.json) cover all 34 addresses in Hebrew/RTL and English/LTR at 1280×720 and 1728×900. The matrix completed 136/136 desktop captures and 544/544 route screenshots with zero capture, settle, H1, locale/direction, horizontal-overflow, active under-44 target, active computed-contrast, console, HTTP, fetch, cancellation, edge-inset, or active reduced-motion defects. Eight final URLs add route-owned selection state while retaining their canonical path/hash/query contract. Each route report and its full/top/middle/bottom screenshots live under `evidence/after/final-cream-matrix-v2/<route-slug>/`.

### Today — 1 surface

| Surface | Canonical address | Operator job | Runtime module | Current evidence |
| --- | --- | --- | --- | --- |
| Broadcast status | `/#Today` | Read the plan of record, review queue, operational checks, economics, guardrails, and yield | [`today/OverviewPage.jsx`](../../tv-break-dashboard/src/today/OverviewPage.jsx) | [Full structural capture](./evidence/after/today-wide-full.png); [current cream viewport](./evidence/after/today-cream-1280.png); [current HE type](./evidence/after/cream-typography-he-today-1280.png); [current EN type](./evidence/after/cream-typography-en-today-1280.png) |

Addressable Today detail state uses `todaySection=economics|guardrails|yield` without changing the owning route.

### Plan — 6 surfaces

| Surface | Canonical address | Operator job | Runtime module | Current evidence |
| --- | --- | --- | --- | --- |
| Objective | `/?plan=objective#Plan` | Set the planning objective and review the recommendation/provenance seam | [`plan/week/PlanWeek.jsx`](../../tv-break-dashboard/src/plan/week/PlanWeek.jsx) | [Full structural capture](./evidence/after/plan-objective-wide-full.png); [current cream viewport](./evidence/after/plan-objective-cream-1280.png); [current HE type](./evidence/after/cream-typography-he-plan-1280.png); [current EN type](./evidence/after/cream-typography-en-plan-1280.png) |
| Run | `/?plan=run#Plan` | Confirm scope and start or inspect a weekly planning run | [`plan/week/RunPanel.jsx`](../../tv-break-dashboard/src/plan/week/RunPanel.jsx) | [Full structural capture](./evidence/after/plan-run-wide-full.png) |
| Compare | `/?plan=compare#Plan` | Prepare and compare A/B planning outcomes, then adopt a draft | [`plan/week/ComparePanel.jsx`](../../tv-break-dashboard/src/plan/week/ComparePanel.jsx) | [Full structural capture](./evidence/after/plan-compare-wide-full.png) |
| Publish | `/?plan=publish#Plan` | Publish, inspect, diff, restore, or freeze a version | [`plan/week/PublishPanel.jsx`](../../tv-break-dashboard/src/plan/week/PublishPanel.jsx) | [Full structural capture](./evidence/after/plan-publish-wide-full.png) |
| Supply | `/?plan=supply#Plan` | Review inventory pressure and yield by planning dimension | [`plan/week/SupplyPanel.jsx`](../../tv-break-dashboard/src/plan/week/SupplyPanel.jsx) | [Full structural capture](./evidence/after/plan-supply-wide-full.png) |
| Week board | `/?plan=board#Plan` | Inspect and edit the saved week at day zoom | [`plan/week/BoardPanel.jsx`](../../tv-break-dashboard/src/plan/week/BoardPanel.jsx) | [Full structural capture](./evidence/after/plan-board-wide-full.png) |

Plan-owned state includes `recommendation=<id>` and `axis=day|daypart|hour|type`. Only the active Plan panel is mounted.

### Broadcast — 4 surfaces

| Surface | Canonical address | Operator job | Runtime module | Current evidence |
| --- | --- | --- | --- | --- |
| Day timeline | `/?broadcast=day#Broadcast` | Select, inspect, move, resize, price, save, undo, and reverse breaks against the true-scale day | [`plan/day/DayPage.jsx`](../../tv-break-dashboard/src/plan/day/DayPage.jsx) | [Full structural capture](./evidence/after/broadcast-day-wide-full.png) |
| Traffic pods | `/?broadcast=pods#Broadcast` | Reconcile traffic truth, inspect pod evidence, and manage pod ordering | [`plan/break/PodPage.jsx`](../../tv-break-dashboard/src/plan/break/PodPage.jsx) | [Full structural capture](./evidence/after/broadcast-pods-wide-full.png) |
| Break library | `/?broadcast=library#Broadcast` | Rank saved-plan breaks, inspect a schedule record, and export the library | [`plan/break/BreakLibraryPage.jsx`](../../tv-break-dashboard/src/plan/break/BreakLibraryPage.jsx) | [Full structural capture](./evidence/after/broadcast-library-wide-full.png) |
| Manual decisions | `/?broadcast=decisions#Broadcast` | Create and manage pin, forbid, forced-count, and gold-break decisions with consequence preview | [`plan/day/OverrideDecisions.jsx`](../../tv-break-dashboard/src/plan/day/OverrideDecisions.jsx) | [Full structural capture](./evidence/after/broadcast-decisions-wide-full.png) |

Broadcast-owned drill state includes `breakView=library|day|pod`, `day=<ISO date>`, and `pod=<id>`. Those states preserve the owning Broadcast domain.

### Commercial — 6 surfaces

| Surface | Canonical address | Operator job | Runtime module | Current evidence |
| --- | --- | --- | --- | --- |
| Clients | `/?clients=clients#Commercial` | Navigate agency → client → campaign/flight relationships and begin controlled onboarding | [`clients/ClientsWorkspace.jsx`](../../tv-break-dashboard/src/clients/ClientsWorkspace.jsx) | [Full structural capture](./evidence/after/commercial-clients-wide-full.png); [current cream viewport](./evidence/after/commercial-clients-cream-1280.png) |
| Money | `/?clients=money#Commercial` | Reconcile gross and net value against the priced traffic ledger | [`clients/MoneyBoard.jsx`](../../tv-break-dashboard/src/clients/MoneyBoard.jsx) | [Full structural capture](./evidence/after/commercial-money-wide-full.png) |
| Campaigns | `/?clients=campaigns#Commercial` | Review booked windows, commitments, counted delivery, and campaign state | [`clients/CampaignBoard.jsx`](../../tv-break-dashboard/src/clients/CampaignBoard.jsx) | [Full structural capture](./evidence/after/commercial-campaigns-wide-full.png) |
| Delivery pace | `/?clients=pacing#Commercial` | Prioritize delivery risk, inspect evidence quality, and record make-good decisions | [`clients/pacing/PacingWorkspace.jsx`](../../tv-break-dashboard/src/clients/pacing/PacingWorkspace.jsx) | [Full structural capture](./evidence/after/commercial-pacing-wide-full.png) |
| Pricing rules | `/?clients=advertisers#Commercial` | Maintain advertiser identities, aliases, and pricing conditions | [`clients/AdvertiserRecordsPanel.jsx`](../../tv-break-dashboard/src/clients/AdvertiserRecordsPanel.jsx) | [Full structural capture](./evidence/after/commercial-advertisers-wide-full.png) |
| Agency records | `/?clients=agencies#Commercial` | Maintain agency terms, contacts, and client relationships | [`clients/AgencyRecordsPanel.jsx`](../../tv-break-dashboard/src/clients/AgencyRecordsPanel.jsx) | [Full structural capture](./evidence/after/commercial-agencies-wide-full.png) |

`client=<id>` addresses a selected client inside the Commercial master/detail composition. The four-step onboarding workflow is an overlay state owned by Clients, not a separate route.

### Sources — 3 surfaces

| Surface | Canonical address | Operator job | Runtime module | Current evidence |
| --- | --- | --- | --- | --- |
| Inputs | `/?sources=inputs#Sources` | Verify exactly which source the engine reads and why | [`sources/InputsView.jsx`](../../tv-break-dashboard/src/sources/InputsView.jsx) | [Full structural capture](./evidence/after/sources-inputs-wide-full.png); [current cream viewport](./evidence/after/sources-inputs-cream-1280.png) |
| Files | `/?sources=files#Sources` | Inspect stored source files and row/field evidence | [`sources/SourceFilesView.jsx`](../../tv-break-dashboard/src/sources/SourceFilesView.jsx) | [Full structural capture](./evidence/after/sources-files-wide-full.png) |
| Reports | `/?sources=downloads#Sources` | Preview and download reports with their exact data basis | [`sources/DownloadsView.jsx`](../../tv-break-dashboard/src/sources/DownloadsView.jsx) | [Full structural capture](./evidence/after/sources-reports-wide-full.png) |

Inputs can be filtered with `source=all|in_use|shadowed|not_read|empty|invalid|missing`. `sourceView` is retained as an internal compatibility state while the canonical shell parameter is `sources`.

### Governance — 6 operator surfaces

| Surface | Canonical address | Operator job | Runtime module | Current evidence |
| --- | --- | --- | --- | --- |
| Restrictions | `/?rules=restrictions#Governance` | Author scoped future-plan rules and inspect cost before save | [`rules/RestrictionsPage.jsx`](../../tv-break-dashboard/src/rules/RestrictionsPage.jsx) | [Full structural capture](./evidence/after/governance-restrictions-wide-full.png); [current cream final viewport](./evidence/after/governance-restrictions-cream-final-1280.png); [current HE type](./evidence/after/cream-typography-he-governance-1280.png); [current EN type](./evidence/after/cream-typography-en-governance-1280.png) |
| Licence | `/?rules=licence#Governance` | Review regulatory limits, compliance evidence, and attestations | [`rules/LicencePage.jsx`](../../tv-break-dashboard/src/rules/LicencePage.jsx) | [Full structural capture](./evidence/after/governance-licence-wide-full.png) |
| Rate card | `/?rules=rate_card#Governance` | Stage, test, save, discard, or reset pricing assumptions | [`rules/PricingManager.jsx`](../../tv-break-dashboard/src/rules/PricingManager.jsx) | [Full structural capture](./evidence/after/governance-rate-card-wide-full.png) |
| Events calendar | `/?rules=calendar#Governance` | Maintain dated demand, price, and availability events | [`rules/CalendarEvents.jsx`](../../tv-break-dashboard/src/rules/CalendarEvents.jsx) | [Full structural capture](./evidence/after/governance-calendar-wide-full.png); [evidence overlay state](./evidence/after/governance-calendar-evidence-open-wide-full.png) |
| Channel & model | `/?rules=channel#Governance` | Verify and change the declarations that scope every channel/model figure | [`rules/ChannelPage.jsx`](../../tv-break-dashboard/src/rules/ChannelPage.jsx) | [Full structural capture](./evidence/after/governance-channel-wide-full.png) |
| Planning levers | `/?rules=levers#Governance` | Maintain saved engine parameters and recompute with explicit state | [`rules/settings-levers.jsx`](../../tv-break-dashboard/src/rules/settings-levers.jsx) | [Full structural capture](./evidence/after/governance-levers-wide-full.png) |

### History — 1 surface

| Surface | Canonical address | Operator job | Runtime module | Current evidence |
| --- | --- | --- | --- | --- |
| Changes & restore | `/#History` | Filter the audit timeline, inspect exact effects, create restore points, and selectively restore | [`history/VersionsPage.jsx`](../../tv-break-dashboard/src/history/VersionsPage.jsx) | [Full structural capture](./evidence/after/history-wide-full.png); [current cream viewport](./evidence/after/history-cream-1280.png) |

History-owned state includes `entry=<id>` and `historyKind=<kind>`.

### Company Model — 7 permission-gated surfaces

Company Model belongs to the Governance domain and is available only when `canAccessModel` is true. Otherwise `#Model` resolves to Governance.

| Surface | Canonical address | Steward job | Runtime module | Current evidence |
| --- | --- | --- | --- | --- |
| Gates | `/?modelSection=gates#Model` | Read the release gates and their measured states | [`model/console/GatesPanel.jsx`](../../tv-break-dashboard/src/model/console/GatesPanel.jsx) | [Full structural capture](./evidence/after/model-gates-wide-full.png); [current cream viewport](./evidence/after/model-cream-1280.png) |
| Coverage | `/?modelSection=coverage#Model` | Inspect model/data coverage and missing evidence | [`model/console/CoveragePanel.jsx`](../../tv-break-dashboard/src/model/console/CoveragePanel.jsx) | [Full structural capture](./evidence/after/model-coverage-wide-full.png) |
| Drift | `/?modelSection=drift#Model` | Inspect drift signals and their measurement basis | [`model/console/DriftPanel.jsx`](../../tv-break-dashboard/src/model/console/DriftPanel.jsx) | [Full structural capture](./evidence/after/model-drift-wide-full.png) |
| Candidates | `/?modelSection=candidates#Model` | Inspect candidates, run measurements, and record verdicts | [`model/console/CandidatesPanel.jsx`](../../tv-break-dashboard/src/model/console/CandidatesPanel.jsx) | [Full structural capture](./evidence/after/model-candidates-wide-full.png) |
| Training | `/?modelSection=training#Model` | Start and monitor a controlled training job | [`model/console/TrainingPanel.jsx`](../../tv-break-dashboard/src/model/console/TrainingPanel.jsx) | [Full structural capture](./evidence/after/model-training-wide-full.png) |
| Versions | `/?modelSection=versions#Model` | Record and inspect model versions | [`model/console/VersionsPanel.jsx`](../../tv-break-dashboard/src/model/console/VersionsPanel.jsx) | [Full structural capture](./evidence/after/model-versions-wide-full.png) |
| Provenance | `/?modelSection=provenance#Model` | Trace artifacts and source provenance | [`model/console/ProvenancePanel.jsx`](../../tv-break-dashboard/src/model/console/ProvenancePanel.jsx) | [Full structural capture](./evidence/after/model-provenance-wide-full.png) |

## Non-route application surfaces

| Surface | Address/state | Contract | Evidence status |
| --- | --- | --- | --- |
| Login | Session probe returns signed-out on a supported desktop | Hebrew-first sign-in with busy and honest 401/429/503/offline states | [Before](./evidence/before/login-wide.png); [current cream](./evidence/after/login-cream-1280.png); final auth/network audit pending |
| Login error | Failed sign-in result | Error remains inside the sign-in task, with no shell behind it | [Before error state](./evidence/before/login-error-wide.png); current error-state recapture pending |
| Desktop gate | Unsupported canvas | Operational app does not mount; locale remains switchable | [Before narrow failure](./evidence/before/overview-mobile-390x844.png); [final cream gate with canonical mark](./evidence/after/desktop-gate-cream-final-1024.png); cold 1024×768 resource list was empty |
| Mabat assistant | `#Assistant` over current operator workspace | Contextual dock, conversation/proposal/upload errors, explicit apply, restore point, and undo | [Before unavailable state](./evidence/before/assistant-dock-unavailable-wide-full.png); [final connected Max/OAuth state](./evidence/final-goal/mabat-connected-1728-he.png) |

## Legacy entrance resolution

| Legacy hash | Canonical destination |
| --- | --- |
| `#Overview` | `#Today` |
| `#Optimizer` | `?plan=objective#Plan` |
| `#Schedule` | `?plan=board#Plan` |
| `#Inventory` | `?plan=supply#Plan` |
| `#Forecasts` | `?plan=compare#Plan` |
| `#Break%20Library` | `?broadcast=library#Broadcast` |
| `#Overrides` | `?broadcast=decisions#Broadcast` |
| `#Campaigns` | `?clients=campaigns#Commercial` |
| `#Advertisers` | `?clients=advertisers#Commercial` |
| `#Agencies` | `?clients=agencies#Commercial` |
| `#Data` | `?sources=inputs#Sources` |
| `#Reports` | `?sources=downloads#Sources` |
| `#Settings` | `#Governance` with the role-derived landing section |
| `#Calendar` | `?rules=calendar#Governance` |
| `#Pricing` | `?rules=rate_card#Governance` |
| `#Versions` | `#History` |
| `#Model` | Model when authorized; Governance otherwise |

## Acceptance boundary

This manifest now has read-only rendered proof for every canonical address in both languages: mount identity, quiet console, successful HTTP/fetch completion, basic solid-background text contrast, target geometry (including explicit native-wrapper exceptions), layout edges, fonts, and screenshots. It does not certify keyboard order, focus behavior after interaction, accessibility-tree semantics, dialog traps/return, forced colours, zoom behavior above the desktop gate, permissioned identities, destructive writes, or non-route error/loading/assistant states. Those boundaries remain explicit in [`qa-report.md`](./qa-report.md).
