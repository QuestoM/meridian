# UX friction ledger — measured after-state

- Measurement date: 15–16 August 2026; final rendered heights recaptured after the Noto Hebrew, cream-palette, compact-header, and inset corrections
- Document refresh: 16 August 2026, against the final route/task model
- After-state: final UX-overhaul worktree used for the named browser probes
- Baseline: [`phase-1-audit.md`](./phase-1-audit.md), Hebrew/RTL at 1280×720 unless noted

> **Measurement integrity note:** the task boundaries and interaction counts below are measured results, not estimates. The final matrix remeasured the six rendered heights at the same 1280×720 Hebrew boundary after the type, header, and inset changes. Interaction counts use the same named task states; the visual correction did not add or remove their controls.

## Result

The measured information architecture materially reduces simultaneous work on the highest-density Broadcast and Commercial paths. Manual Decisions exposes 90.6% fewer interactive elements than the old selected Overrides composition, Campaigns exposes 67.9% fewer, and the controlled Pacing comparison exposes 56.7% fewer. On the final rendered build those documents are respectively 54.5%, 63.5%, and 70.8% shorter than their controlled baselines. The Day surface still draws the broadcast timeline at true scale, but no longer makes its smallest geometry the only way to select or edit a break.

The result is not uniformly shorter. Plan Objective is 58.5% taller and exposes three more controls in the measured state because the recommendation, provenance, and decision seam that previously disappeared between Today and Plan is now present in the destination. That is an intentional capability and evidence repair, not a density win. Only the active Plan panel is mounted, so the old total of 79 controls across visible and hidden sections is no longer the relevant runtime composition.

## Measurement method and limits

- After-state probes used the local Chrome runtime with authentication disabled, plan writes disabled, and assistant-provider credentials blank. This protects production-like data while permitting deterministic read-path inspection.
- Desktop probes used a 1280×720 viewport and Hebrew/RTL. Document height is the rendered scrolling document. “Interactive elements” means DOM-visible controls and actionable elements across the whole document, not only the first viewport.
- Target sizes were read from rendered rectangles. MUI widgets may expose nested implementation nodes, and timeline geometry intentionally remains true-scale. The ledger reports an unresolved small target only when the small rectangle is itself the operator's actionable target without a compliant proxy.
- Counts are state- and data-dependent. They are a reproducible snapshot of the same named task state, not a claim that every possible record count produces the same document length.
- The baseline and after probes use the same task name, viewport, locale, selector boundary, and rendered-state definition for every percentage in the comparable table. Where that was not possible, the row is excluded or explicitly reconciled rather than presented as a percentage.
- Broadcast and Commercial were deliberately decomposed into active-only contexts. Their comparisons measure the immediate task presented to the operator, not identical simultaneous DOM capability: the displaced capabilities remain reachable in adjacent addressable contexts.
- Percentage changes use unrounded source measurements and are rounded to one decimal place. A negative value means less rendered height or fewer simultaneous interactives.
- The Pacing row uses a controlled remeasurement made with the same selector and content boundary before and after. The Phase 1 published headline of 15,226px and 302 interactives used a broader all-record composition and is retained below as historical context, not mixed into the controlled percentage.
- This ledger measures friction and target geometry. It is not a substitute for the independent keyboard, screen-reader, visual, console/network, or write-flow passes, and it makes no final accessibility conformance claim.

## Same-job path remeasurement

The activation boundary is the same one used in the baseline: reaching and completing the named operator task, not counting text entry as a click. “Page load” means a document reload; route/context changes are listed separately.

| Core job | Baseline observed path | Final observed path | Page/context cost | Fields hunted at once |
| --- | --- | --- | --- | --- |
| Week plan | About 7 activations from Overview through Run, Compare, and Freeze; four global weekly entrances; two publish fields | Minimum 8 activations from Today → Plan → Run review/confirm → Compare → Publish → Freeze review/confirm. The two extra confirmations are deliberate consequence safety | 0 document reloads; one stable Plan domain with six URL-backed active-only steps instead of four global destinations | Objective exposes 3 sliders plus named mode choices; Publish exposes its 2 fields only when active; hidden sibling panels contribute no fields |
| Broadcast day / pod | Day detail about rail → break → Enter; pod truth Break Library → covered day → pod, while Day and Overrides shared one long composition | 2 activations to enter Broadcast Day and open a programme/break through a compliant proxy; 3 to enter Pods, choose a day, and open a pod | 0 document reloads and 0 global-domain exits; Day, Pods, Library, and Decisions are local addressable contexts | No form field is required to inspect Day or Pod truth; the relevant inspector stays beside its record |
| Commercial delivery and onboarding | 5 activations across Campaigns → row → pacing → day evidence → onboarding, with at least 4 context shifts; 14 fields + 7 weekday toggles for existing-agency work and 24 controls for new-agency work appeared together | 5 activations reach the same onboarding task; 3 Next activations and 1 reviewed Create complete its four-step workflow | 0 document reloads and 0 exits from Commercial; the workflow is one isolated modal over retained record context | Existing-agency Identity shows at most 4 identity choices/fields; new-agency Identity 14; Commercial terms 5 fields plus 7 weekday choices; each Flight row 4 fields; Review has no editable field |

The week-plan activation count is intentionally not lower: broad plan writes and freezes now require an explicit review and confirmation. The friction win is retained context, active-only fields, and truthful consequence handling rather than one-click money-moving actions.

## Comparable before → after measurements

| Task state | Document height | Height change | Visible interactives | Interactive change | Target-size reading |
| --- | ---: | ---: | ---: | ---: | --- |
| Today | 2,628px → 1,459px | −44.5% | 43 → 32 | −25.6% | 35 baseline targets under 44px → no unresolved actionable target in the final probe; the top-bar slider root is 44px |
| Plan — Objective | 1,096px → 1,737px | +58.5% | 45 → 48 | +6.7% | 39 baseline targets under 44px → no unresolved actionable target in the final probe |
| Broadcast — Manual Decisions | 1,751px → 797px | −54.5% | 255 → 24 | −90.6% | 223 baseline targets under 44px → none in the final probe |
| Commercial — Campaigns | 7,488px → 2,733px | −63.5% | 355 → 114 | −67.9% | 222 baseline targets under 24px → none in the final probe; 44px inline actions and pagination select were verified after correction |
| Commercial — Pacing, controlled boundary | 6,816px → 1,993px | −70.8% | 97 → 42 | −56.7% | No target under 24px in the final full-page probe |
| Broadcast — Day, break selected | 1,751px → 1,236px | −29.4% | 253–255 → 170 | −32.8% to −33.3% | 81 baseline targets under 24px → none; narrow true-scale break visuals retain a separate 44px navigator/open proxy |

The after-state full-page probes also recorded 23 interactives in the first Today viewport, 31 in the first Objective viewport, 22 in Manual Decisions, 39 in Campaigns, 33 in Pacing, and 135 in the first Day viewport. These viewport counts explain immediate visual load but are not used in the percentages because the Phase 1 ledger did not publish the same field consistently.

### Pacing baseline reconciliation

Phase 1 recorded the original all-record Pacing surface at 15,226px with 302 interactives and 51 repeated upload links. The controlled pre-change boundary used for the row above measured 6,816px and 97 page-specific interactives; the after-state measured 1,711px and 42 page-specific interactives, or 50 when the global shell was included. The controlled pair is the defensible percentage. The larger published baseline still matters as evidence of the old worst-case composition, but comparing 15,226 directly with 1,711 would mix measurement boundaries.

## What changed in the task model

| Baseline friction | After-state treatment | Capability consequence |
| --- | --- | --- |
| Fifteen global entrances plus duplicated local navigation | Seven domains: Today, Plan, Broadcast, Commercial, Sources, Governance, History | The same capabilities are addressable inside a stable domain rather than presented as separate products |
| Four global weekly entrances and six simultaneously mounted Plan sections | One Plan workspace with six URL-backed, active-only steps | Objective, Run, Compare, Publish, Supply, and Board remain; Back/Forward restores the current step |
| Day timeline, inspectors, and the full override application in one document | Separate Day and Manual Decisions contexts | Direct timeline editing and manual pin/forbid/count/gold decisions remain, without simultaneous page-level competition |
| Pod, day table, and ranked library stacked together | Active-only Library, Day, and Pod contexts | Record drills remain addressable and return to their owning level |
| 52 campaign rows and all repeated actions mounted together | Stable progressive windows: 12 campaigns, 16 pacing campaigns, 18 commercial records | Totals and provenance remain visible; more records are deliberately revealed |
| One long onboarding form over a still-interactive Campaigns page | Four-step Identity → Commercial terms → Flights → Review workflow | The one-submit agency/client/campaign/flight transaction remains intact, with a named review boundary |
| Tiny timeline geometry as the only target | True-scale drawing plus `DayBreakNavigator` | Pointer direct manipulation remains; selection/opening does not require precision aiming |
| Browser navigation replaced the current address and leaked unrelated state | Scoped query ownership plus push history | Meaningful task state is shareable and Back/Forward-safe |

## Desktop-only support decision

Phone and tablet task friction is intentionally not optimized. The user explicitly chose a desktop console with a truthful gate instead of a compressed operational layout. At 390×844, the baseline Overview placed `main` at y=844 and produced a 5,182px document. The after-state replaces the application before session or data mounting with one localized “continue on desktop” main/H1. The fresh 390×844 probe measured an exact 390px document width, no operational shell or authentication UI, and no API `fetch`/XHR activity.

The 1,200px threshold and zoom exception are product heuristics, not universal device truth. Multi-browser checks at 200–400% desktop zoom, touch laptops, split-screen, display scaling, and remote desktops remain an explicit validation item in the [`assumption ledger`](./assumption-ledger.md).

## Latest visual-system correction

The user's later direction changes how the measured tasks look, not what they contain:

- Fable is the craft/completeness ceiling, not a decorative style source.
- The REDLINE reference applies to Hebrew typography and font fit, not copywriting.
- The final material system is warm cream, near-black ink/chrome, and restrained mineral sage; its anthroposophic quality is material warmth rather than ornament.
- Hebrew now uses the local Hebrew-only Noto Sans Hebrew variable face, paired with IBM Plex Sans for Latin/figures and IBM Plex Mono for identifiers.
- The product mark is now one transparent two-path off-axis frame-splice SVG shared by rail, loading, login, and gate; generated logo directions do not ship.
- The shell header is one 57px band, including Broadcast and Governance local navigation, rather than a tall primary row plus a second navigation tier.
- Route and addressable-tab changes preserve rail/header geometry and acknowledge only the workspace through a feature-detected View Transition, restrained fallback, or no animation under reduced-motion preference.
- Workspace and Card now own their respective outer gutter and inner inset. The final evidence harness measures logical edge contacts and names legitimate full-bleed rows/tables rather than treating every flush boundary as equivalent.
- Below 1,200px, the operator sees the desktop-required gate rather than a compressed workspace.

These changes leave the measured task controls unchanged by design. The final matrix recapture supplies the current rendered-height values above because font metrics, the compacted header, corrected insets, and wrapping did move the document boundary.

## Remaining friction and honest interpretation

- Campaigns is still a 2,867px document and Today is still 1,872px at the measured data volume. Progressive reveal reduces simultaneous density; it does not make large operational datasets intrinsically small.
- Objective is longer because recommendation evidence and decisions now arrive at their destination. Further shortening must preserve that provenance and consequence, not hide it again.
- The Day page still exposes 170 interactives across the full document because direct manipulation of a dense 24-hour schedule is real product capability. The redesign reduces collision and repairs activation targets; it does not pretend the timeline is a simple form.
- Pacing still reports missing-source days as unknown rather than zero. Lower interaction density does not repair missing business data.
- Target findings are route probes, not a complete automated accessibility result. Final release language must remain bounded by the independent QA evidence.
- The six rendered-height values and target verdicts now come from the final Noto/cream/header build. Interaction counts remain tied to the same named task-state probes and must be rerun if those task compositions change.

Implementation and architecture details are in [`phase-3-implementation.md`](./phase-3-implementation.md). Source-system convergence is measured separately in [`divergence-resolution.md`](./divergence-resolution.md).
