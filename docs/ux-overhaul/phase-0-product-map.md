# Phase 0 — product, route, state, and task map

Captured on 15 August 2026 against the local FastAPI/Vite product with authentication disabled, plan writes disabled, and all assistant-provider credentials blanked. No stylesheet was inspected before this map and the baseline capture were complete.

## Product in one sentence

Kairos is the control surface for an Israeli commercial TV channel to plan advertising breaks, protect retention and licence constraints, verify what will air, manage campaign delivery, and recover every material decision.

The fuller product model is in [product-model.md](./product-model.md).

## The operating model

| Cadence | Primary operator question | Current surfaces | Stakes |
| --- | --- | --- | --- |
| Daily | What needs action now? | Overview, Sources, Campaign pacing, Break Library, Day plan, Overrides | Missed revenue, delivery liability, media failure, on-air mistakes |
| Weekly | What plan should we commit to? | Objective, Run, Compare, Publish, Supply, Week board | Roughly ₪40.9M modeled gross value in the sample plan, retention, licence compliance |
| Episodic | What rules, identities, and models govern the plan? | Clients, Rules, History, Model, account administration | Commercial terms, legal policy, auditability, model release safety |

Thirteen job profiles collapse into three authority roles. Job should determine the first useful context; role and company/channel affiliation should determine what a person may change.

## Current route topology

The app is a hand-written hash router, not a collection of independent pages. Its visible navigation overstates the number of actual workspaces.

| Visible entrance | Actual workspace | Internal surface |
| --- | --- | --- |
| Overview | Today | Money, health, priority decisions, revenue/retention evidence |
| Optimizer | Week plan | Objective |
| Schedule | Week plan | Week board |
| Inventory | Week plan | Supply |
| Forecasts | Week plan | Compare |
| Break Library | Break operations | Traffic pod, broadcast-day break table, ranked plan library |
| Campaigns | Clients | Campaigns |
| Advertisers | Clients | Pricing rules |
| Agencies | Clients | Agency records |
| Reports | Sources | Reports/downloads |
| Data | Sources | Inputs/files |
| Overrides | Day control | Day timeline plus manual override composer |
| Restore changes | History | Audit timeline and restore controls |
| Settings | Rules | Restrictions, licence, rate card, calendar, channel/model, planning levers |
| Mabat, AI assistant | Persistent dock | Conversation, context, proposals, apply/undo |
| Model | Separate overlay root | Gates, coverage, drift, candidates, training, versions, provenance |

Compatibility-only `#Calendar` and `#Pricing` bookmarks rewrite into Rules. The Assistant item does not navigate. Model mounts a second React root over an inert operator shell.

### Material wiring seams

- Four rail items mount the same six-section Week plan. The active section is local component state and is not addressable in the URL.
- The shell still fetches optimizer previews and stores recommendation, approval, grid-axis, and inspector state, but the live Week plan discards those props. Overview can therefore “open” a recommendation that never appears in the destination.
- The shell stores an `axis` query parameter, while the live board always initializes its own axis to day.
- Navigation uses `replaceState`; ordinary workspace changes do not create useful browser Back history.
- Query parameters from unrelated workspaces accumulate. A real return to Overview retained `clients`, `entry`, `rules`, `day`, and `pod` simultaneously.
- Job selection does not globally route a returning person to their operating surface, and several picker labels map to different destinations than they promise.
- Several old components are dead while other visually “legacy” components remain live. Removal must follow reachability, not filename or appearance.

## Complete state families

This inventory is the acceptance surface for the redesign. Individual visual variants may be recomposed, but none may disappear without an explicit removal decision.

### Global shell and identity

- Session check, login, failed login, forced password change, authenticated, and honest open-access mode.
- Admin/operator/viewer authority; company/channel/unknown affiliation.
- Hebrew RTL and English LTR; channel/date/scenario/caution context; partial and offline API states.
- Global staleness, background rebuild, toast, persistent activity, account/password/user administration.
- Assistant closed/open/resized; connected/checking/unavailable; conversation/proposal/upload errors; explicit apply, restore point, and undo.

### Today

- Reading, answered, partially degraded, and unavailable.
- Revenue target absent, present, editing, saving, and refused.
- Healthy/degraded plan, licence, and sources.
- Empty or ranked decisions; day and decision detail drills.
- Modeled gross/net, retention, compliance, frontier, and yield evidence.

### Week plan

- Objective presets, four continuous levers, engine focus, clean/dirty/save/revert/failure.
- Run freshness, progress, elapsed time, failure, zero-break warning, and completed plan.
- Compare A/B edit, preparation, streaming day results, cancellation, error, decisive day, and adopt-as-draft.
- Publish name/reason, permission wall, frozen versions, diff, restore, and restore confirmation.
- Supply totals, daypart/hour pressure, yield-per-second, unscoped/empty.
- Board grid, strips, timeline, and one-day editor; loading/error/empty/stale/no-channel.
- Command palette and keyboard navigation.

### Broadcast day and traffic pods

- Date selection across all covered days; snap, zoom, fit, programme focus.
- Select/move/resize/gold a break, keyboard controls, exact-time and duration edit.
- Live revenue/retention/regulatory effect, preview, unsaved count, save, undo, reverse saved placement.
- Break/programme detail drawers; previous/next navigation; modeled value, retention confidence, hourly licence use, As Run, pod contents.
- Traffic coverage fallback; time/attention sorting; expanded pod.
- Verification/media/duration/gap/preferred-position evidence.
- Drag/Alt-arrow reorder, dirty/save/revert/traffic-order reset, lock/unlock/stale saved order.
- Manual pin/forbid/count/gold override, effect preview, save, optional single-day run, dismissed/stale/delete.

### Commercial delivery

- Client, money, campaign, pacing, pricing-rule, and agency-record views.
- Campaign list; demo versus operator-booked provenance; inline campaign/flight expansion; add/edit/end/remove flows.
- One-submit onboarding of agency, client, campaign, terms, weekday discount, and one or more flights; existing/new identities, loading/saving/done/refused/open-existing.
- Pacing behind/at-risk/unknown/on-pace; worst-first list; planned versus measured truth; missing-source days; day drill.
- Accept risk, raise make-good, offer/close/withdraw, undo, and decision ledger.
- Advertiser and agency search, details, commercial conditions, add/save/suspend, guarded delete.

### Sources, Rules, History, and Model

- Source in-use/shadowed/not-read/empty/invalid/missing; choose, check, findings/consequence, commit; file and row preview.
- Report present/empty/unavailable, exact row basis, single/all download, preview.
- Restrictions compose/preview/save/delete/expired/advanced; programme/airing lookup and scope.
- Licence compliant/review/unknown/unreachable; attestation; admin guardrail editing.
- Rate-card staged base/premium/event/position/daypart edits; tester, effect, save/discard/reset.
- Calendar grid/list, edit/activate/import/overlap, company wall.
- Channel and audience-model activation confirmations; planning-lever dirty/recompute/activity states.
- History filters, pagination, missing/paged-out detail, run/change/account/restore-point/restore/preview entries, diff and selective restore.
- Model gates/coverage/drift/candidates/training/versions/provenance; loading/refused/unreachable; measurement, verdict, training job, and activation mirror.

## Observed core journeys

### 1. Plan the week

Observed path: rail `Optimizer` → Objective → Run → Compare → Publish.

- The operator meets 15 global navigation choices before the six local Week-plan choices.
- Objective exposes four presets, four sliders, two engine-focus choices, save, and revert. A slider's visible track is only about 12 px high.
- Run is one paragraph, one destructive/high-consequence action, and a status, but it inherits the same large shell as every other planning state.
- Compare places eight sliders and four focus buttons on one surface before any result exists.
- Publish mixes version creation with four historical versions and eight diff/restore actions.
- No revenue target is set, so the plan repeatedly explains why it cannot declare success.
- The four weekly rail entrances and six internal sections make the same mental model appear as two unrelated navigation systems.

### 2. Control a broadcast day and pod

Observed path: rail `Overrides` → select first break → Enter for details; then `Break Library` → covered traffic day → first pod.

- The day page initially exposed 30 date buttons, 96 quarter-hour labels, programmes, 80 break controls, 24 hourly-load controls, save/undo/effect controls, and a second override application in one document.
- With one break selected, the accessibility tree contained 253 buttons.
- The break drawer is strong evidence—exact window, modeled revenue, pricing derivation, retention confidence, hourly licence use, As Run, and pod coverage—but it competes with the still-live day board and the override composer behind it.
- Traffic truth covers 27 April 2025 while the saved plan covers November 2024. The product honestly explains the mismatch but forces the operator to reconcile two time contexts inside one stacked page.
- A pod expands inline inside a page that continues into a separate day table and whole-plan ranked library. Even the one-spot pod presents four save/revert/lock actions.

### 3. Manage commercial delivery

Observed path: rail `Campaigns` → expand campaign → `Delivery pace` → expand seven-day evidence → `Onboard a client`.

- The Campaigns page renders 52 campaigns in a 7,454 px document and exposed 364 buttons in the baseline.
- Fifty-one of 52 rows are demo-seeded. Each row repeats a long provenance explanation, making the exceptional real booking harder to find.
- Campaign details expand inside the table, increasing density and preserving every surrounding row.
- Pacing correctly ranks 13 at-risk cases first, but each card repeats method, missing-source, remaining-goal, upload, accept-risk, and evidence controls.
- The seven-day drill proves that only one of seven days has traffic truth; six are unknown, not zero. This is the key decision evidence and should be primary rather than a nested disclosure.
- Onboarding is one modal containing agency identity, client identity, campaign terms, weekday pricing logic, and repeatable flight goals. It is conceptually one transaction but visually one long form with no progressive hierarchy.

## Quantitative baseline

Wide viewport: 1440 × 900, Hebrew RTL unless marked otherwise.

| Surface | Document height | Buttons | Inputs | Primary finding |
| --- | ---: | ---: | ---: | --- |
| Overview | 2,628 px | 35 | 2 | Seven evidence sections under a dense global toolbar |
| Week Objective | 1,078 px | 79 | 18 | Two levels of planning navigation |
| Week Board | 1,282 px | 79 | 18 | Same workspace presented as a separate route |
| Supply | 2,158 px | 85 | 16 | Optimizer placement parser rejects all 994 shipped steering rows; Supply itself uses a separate reference pipeline |
| Break Library | 4,666 px | 85 | 1 | Three distinct levels stacked on one page |
| Campaigns | 7,454 px | 364 | 1 | Maximum repeated-row action density |
| Compare | 1,008 px | 67 | 16 | Form density before results |
| Reports | 1,458 px | 35 | 0 | Separate entrance into Sources |
| Data | 2,107 px | 58 | 7 | Check/commit distinction is present but visually buried |
| Advertisers | 5,278 px | 33 | 2 | Cards produce excessive page length |
| Agencies | 1,233 px | 40 | 1 | Same Clients workspace, separate rail entrance |
| Overrides | 1,751 px | 88 | 5 | Day editor and override composer collide |
| History | 1,125 px | 34 | 5 | Entry state leaks into unrelated URLs |
| Rules: restrictions | 900 px | 28 | 7 | One of six internal Rules applications |
| Rules: rate card | 1,926 px | 28 | 36 | High-consequence staging needs clearer grouping |
| Rules: calendar | 8,371 px | 88 | 0 | Longest governance surface |
| Rules: planning levers | 6,006 px | 36 | 21 | Older system nested inside new Rules chrome |
| Assistant open | 6,145 px | 151 | 72 | Dock plus underlying page remain one huge document |

The baseline evidence folder contains 147 PNGs: full-page and sectional captures for every primary workspace, all six Rules sections, all seven Model sections, the Assistant state, login/error, the three core journeys, an English control, and a mobile probe.

### Responsive baseline

At 390 × 844, the Overview `main` element began at `y = 844`—exactly one full viewport below the top—while the first screen contained only brand, a truncated horizontal navigation strip, blank space, and the fixed model-console chip. The user must scroll an entire screen before seeing page context or content. The body measured 5,182 px tall.

**Product decision, 15 August 2026:** mobile and tablet are intentionally unsupported for this dense operator console. The redesign will replace the broken partial layout below the desktop breakpoint with a complete, accessible “continue on desktop” gate. Responsive acceptance therefore means that the gate is correct at phone, tablet, portrait, landscape, RTL, and LTR sizes; it does not mean that the operational workspaces reflow on those devices.

## Interaction and runtime findings carried into Phase 1

- Initial protected API bootstrap causes many 401 requests before the login screen settles.
- First meaningful Overview data took roughly 20 seconds on the local dataset.
- React warns about an unrecognized `inputProps` DOM property and unsafe `:first-child` selectors.
- MUI Data Grid repeatedly reports a zero-width parent.
- The optimizer placement parser rejects all 994 shipped steering rows at server startup. The separate Supply read may still be populated, so the release defect is the unavailable money-moving placement input rather than a proven-empty Supply payload.
- Opening the Assistant automatically warms an external provider when credentials exist. Baseline testing therefore blanked all provider credentials.
- `KAIROS_PLAN_READONLY=1` protects the weekly plan but not the other flat-file stores. Destructive interaction testing requires an isolated data copy.

## Baseline evidence

- Root: [`evidence/before`](./evidence/before)
- Representative wide captures: `overview-wide-section-01-top.png`, `optimizer-wide-section-01-top.png`, `flow-day-03-break-inspector.png`, `flow-commercial-03-pacing.png`
- Responsive failure: `overview-mobile-390x844.png`, followed by `overview-mobile-390x844-scrolled.png`
- English control: `overview-en-wide-full.png` and its three sectional captures

## Phase 0 conclusion

The product does not lack capability; it lacks a stable hierarchy for that capability. The redesign should consolidate the shell into durable task domains, make entity and time context persistent, turn long mixed-purpose pages into list/detail or focused-decision compositions, and surface evidence quality next to every consequential number. The weekly plan, broadcast day/pod, and commercial-delivery loop are the primary architecture. Sources, Rules, History, and Model support them. Assistant should remain contextual.
