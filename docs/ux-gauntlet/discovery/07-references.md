# Reference investigation

What the seven named references actually do, in mechanics rather than adjectives, so a
later critic can run a real blind comparison instead of arguing from memory.

## Method, and what was and was not reachable

Captured on 2026-07-31 from this machine. Screenshots were taken with headless Chrome
150.0.7871.188 at a 1440 px viewport and with Playwright at 1440x900, and are stored under
`/Users/home/Code/questo/meridian/docs/ux-gauntlet/discovery/refs/` (34 files, 7.5 MB).
Page text was read either through the browser or by fetching the page and stripping markup.

Reachability, stated plainly:

- **Linear: fully reachable.** `https://linear.app/demo` returns HTTP 200 and loads a live,
  interactive workspace with no sign-in. Every Linear behaviour below with a "measured" tag
  was performed by me in that workspace, not read in a document.
- **Stripe Dashboard: behind a login.** `https://stripe.com/dashboard` redirects to a sign-in
  form (`refs/05-stripe-dashboard-login-wall.png`). I did not attempt to sign in and typed no
  password. Stripe evidence below comes from `docs.stripe.com`, which documents the Dashboard
  behaviour precisely.
- **Google Ads: behind a login.** Evidence comes from Google Ads Help articles, which name the
  exact status strings and column positions.
- **Premiere and Descript and Resolve: applications, not web surfaces.** Evidence comes from
  the vendors' own current help pages, which name modifier keys and increments exactly.
- **Frame.io: behind a login,** but `frame.io` embeds a full, legible capture of the real
  product in its hero (`refs/14-frameio-home.png`), which I read directly at pixel level.
  Field and status semantics come from `help.frame.io`.
- **Figma: editor behind a login.** Evidence comes from `help.figma.com`.
- **In-product AI: Linear's agent is visible in a live product capture** on `linear.app`
  (`refs/01-linear-home.png`) and documented at `linear.app/docs/linear-agent`. Cursor and the
  Figma agent are used as comparators from their own docs.

Three capture attempts failed and are not papered over: `linear.app/docs/keyboard-shortcuts-reference`
is a 404, `ads.google.com/home/` never finished loading in headless Chrome across three
attempts, and pressing `?` twice in the live Linear demo did not open the shortcuts panel, so
the documented `?` behaviour is cited from Linear's changelog and is **not** confirmed live.

## Capture index

| File | Source | What it shows |
| --- | --- | --- |
| `refs/01-linear-home.png` | linear.app | Real product shot including the agent dock with a Preview control |
| `refs/02-linear-features.png` | linear.app/features | Product framing |
| `refs/21-linear-docs-select-issues.png` | linear.app/docs/select-issues | Selection and bulk shortcuts |
| `refs/22-linear-docs-linear-agent.png` | linear.app/docs/linear-agent | Agent surface and scoping |
| `refs/30-linear-docs-display-options.png` | linear.app/docs/display-options | Grouping, ordering, display properties |
| `refs/31-linear-demo-issue-list.png` | linear.app/demo (live) | Issue list at 1440x900 |
| `refs/32-linear-demo-command-palette.png` | linear.app/demo (live) | Cmd+K palette, measured |
| `refs/33-linear-demo-selection-actionbar.png` | linear.app/demo (live) | `x` selection plus floating action bar |
| `refs/34-linear-demo-issue-detail.png` | linear.app/demo (live) | Record page with `1 / 31` position counter |
| `refs/35-linear-demo-cycle-progress.png` | linear.app/demo (live) | Cycle progress rail and burn-up |
| `refs/05-stripe-dashboard-login-wall.png` | stripe.com/dashboard | Proof the Dashboard is behind a login |
| `refs/06-stripe-payments-marketing.png` | stripe.com/payments | Product framing |
| `refs/07-stripe-docs-reports.png` | docs.stripe.com/stripe-reports | Report catalogue |
| `refs/08-stripe-docs-balance-report.png` | docs.stripe.com/reports/balance | Balance summary structure |
| `refs/23-stripe-revenue-recognition.png` | stripe.com/revenue-recognition | Product framing |
| `refs/24-stripe-docs-audit-numbers.png` | docs.stripe.com/revenue-recognition/reports/audit-numbers | Number to rows drill path |
| `refs/09-googleads-account-hierarchy.png` | support.google.com/.../1704396 | Account, campaign, ad group |
| `refs/09-googleads-pacing-insights.png` | support.google.com/.../13685469 | Budget pacing insight states |
| `refs/10-googleads-campaign-statuses.png` | support.google.com/.../1722131 | Status vocabulary plus remedies |
| `refs/11-googleads-limited-by-budget.png` | support.google.com/.../6385220 | Budget simulator remedy path |
| `refs/12-premiere-move-clips.png` | helpx.adobe.com | Nudge, ripple, insert, overwrite |
| `refs/13-premiere-timeline-nav.png` | helpx.adobe.com | Timeline anatomy and playhead field |
| `refs/19-descript-home.png` | descript.com | Product framing |
| `refs/28-descript-help-timeline.png` | help.descript.com | Timeline parts |
| `refs/29-descript-help-sequence-editor.png` | help.descript.com | Nudge increment table |
| `refs/20-resolve-edit.png` | blackmagicdesign.com | Product framing |
| `refs/14-frameio-home.png` | frame.io | Real product grid, cards, Fields toolbar |
| `refs/15-frameio-support-home.png` | support.frame.io | Legacy support index |
| `refs/25-frameio-help-metadata.png` | help.frame.io | 32 metadata fields, read-only fields |
| `refs/26-frameio-help-player.png` | help.frame.io | Status labels, guides, quality ladder |
| `refs/16-figma-home.png` | figma.com | Product framing |
| `refs/17-figma-help-select-layers.png` | help.figma.com | Selection and deep select mechanics |
| `refs/27-figma-help-agent.png` | help.figma.com | On-canvas agent, parallel prompts, undo |
| `refs/18-cursor-agent-docs.png` | cursor.com/docs/agent/overview | Checkpoints and restore |

## 1. Linear

### What is on the screen, measured

In the live demo at a 1600x823 viewport, the issue list rendered **17 full-width issue rows,
each exactly 44 CSS px tall and 1347 px wide** (measured by script in the page). A single row
carries seven text tokens plus two glyphs: identifier, title, two labels, an estimate, an
assignee avatar, a date, a priority glyph and a status glyph. Roughly nine facts in 44 px.

The left rail holds two ungrouped destinations in the demo (`Inbox` with a count badge and
`My issues`; the marketing capture at `refs/01-linear-home.png` shows four, adding `Reviews`
and `Pulse`), then the grouped sections `Workspace`, `Favorites` and `Your teams`. Filtering
between Active, Backlog and All issues is a segmented control at the top of the content, not
three more navigation entries. Group headers carry a status glyph, a name and a count, and the
docs state that the group header stays sticky while scrolling when sub-grouping is in use
(`linear.app/docs/display-options`).

### What is not on the screen

No toolbar of buttons above the list. No breadcrumb bar beyond one line. No per-row action
icons at rest: hovering a row reveals exactly one new affordance, a checkbox at that row's left
edge, and nothing else changes (measured, zoom capture of the hovered row).

### What the keyboard does

Measured in the demo:

- `Cmd K` opens a centred overlay with one input and results grouped by entity type. **Each
  result row prints its own shortcut on the right**: `C`, `V`, `Option C`, `N then P`. The list
  behind stays visible and is not dimmed away.
- Hovering a row and pressing `x` selects it. The checkbox fills, the row gets a tinted
  background and a solid left accent bar, and a floating bar fades in at the bottom centre
  reading `1 selected`, `Move to Backlog`, `Actions`, and a dismiss control. It is not modal:
  the list stays scrollable and further selection keeps working.
- Hovering a row and pressing `Return` opens the record full page.

Documented and consistent with the above (`linear.app/docs/select-issues`, `/search`, `/inbox`,
`/display-options`): arrow keys or `J`/`K` move the highlight; `Shift` plus arrows extends a
selection one row at a time; filter first then `Cmd A` selects the filtered set; `Esc` clears;
`Option` plus arrows reorders manually and `Option Shift` plus arrows sends to top or bottom;
`/` searches the workspace while `Cmd F` filters the current view as you type; `Shift V` opens
display options and `Cmd B` flips list and board; `G then I` reaches the Inbox from anywhere;
in the Inbox `U` toggles read, `H` snoozes, `Backspace` deletes one notification and
`Shift Backspace` deletes all read ones. Typing a type letter then space (`i`, `p`, `u`, `t`,
`l`, `f`, `d`) scopes the palette to that entity type.

### How a number connects to its detail

Opening a record keeps its place in the result set. The record header shows the identifier, the
title, a star, an overflow control, and on the right **`1 / 31` with a down and an up arrow**,
so the whole filtered set can be walked from inside a record without going back
(`refs/34-linear-demo-issue-detail.png`).

The cycle view is the pacing pattern (`refs/35-linear-demo-cycle-progress.png`). The right rail
prints a `Current` chip, the cycle's date range, then three progress rows that each carry a
colour swatch, a label, an absolute figure and a percentage on the same unit: scope 32,
started 22 at 69 percent, completed 0 at 0 percent. Under it a burn-up chart with a dotted
ideal line, and two shaded vertical bands at weekly spacing (INFERRED from their spacing and
position that these are non-working days). Under that, segmentation tabs for Assignees, Labels,
Priority, Projects and Teams, with each row reading as a share of a named total, for example
17 percent of 23.

### How state is shown and how empty is handled

On the record page the right rail is a `Properties` list where **an empty field is an action,
not a blank**: alongside the filled status, priority, assignee and cycle sits `Set estimate`,
and under `Project` sits `Add to project`. The cycle rail offers `Add document or link`.
Activity is an attributed, timestamped audit log that includes non-human actors and rule
events, for example an SLA being set and later breached.

### The device to steal

Selection produces a contextual action bar. Density comes from one row height plus an explicit
choice of which properties print. Every keyboard action is taught on the row that performs it.

## 2. Stripe Dashboard

The Dashboard itself is behind a login. What Stripe documents about it is unusually specific
and is the transferable part.

### How a figure carries its basis

Every financial report is configured by four declared things, and the docs spell out the
semantics of each (`docs.stripe.com/reports/options`):

- **Date range**, with inclusivity stated explicitly, from the first instant of the start date
  to the last instant of the end date.
- **Time zone**, chosen between the account's zone and UTC, and the docs state that the choice
  changes both which rows are filtered in and how times are rendered.
- **Currency**, which is the account's settlement currency, with a selector when there is more
  than one.
- **Account scope** for Connect platforms: the platform alone, all connected accounts, a chosen
  subset, or one account.

### Honest states

Stripe refuses to show a provisional number. Data is computed per whole day; the Dashboard
serves only complete days, and a partial day is reachable only through the Reporting API
(`docs.stripe.com/reports/balance`). Each report publishes an availability window, with a
caution that webhook notification can lag behind the report being ready. The failure mode is a
number that has not arrived yet, never a number that is quietly wrong.

I could not observe the Dashboard's zero-data empty states, because the Dashboard is behind a
login. Recording that rather than guessing.

### Drilling from a number to the rows behind it

This is the single most transferable Stripe mechanic
(`docs.stripe.com/revenue-recognition/reports/audit-numbers`, `refs/24-*.png`):

- Click an amount in the monthly summary and it expands into the list of customers that make
  it up, with each customer's share.
- Click a customer in that list and it expands into that customer's transactions for the month.
- The same click-the-amount path exists on the revenue waterfall and on the AR aging summary,
  where clicking resolves to customers or to invoices, and an invoice resolves to its detail.
- The same accounting view is reachable from the other direction, from a customer, an invoice,
  an invoice line item or a payment, through an overflow menu on the object itself.

The balance summary is built the same way: a starting and ending balance for the range, then a
breakdown by reporting category with gross, fee and net for each category. Download offers
either the summary exactly as rendered or the full itemised transactions with metadata, so the
screen and the export are declared to be the same numbers at two grains.

### The device to steal

Four basis facts are attached to the report, not to a tooltip. Any amount is a link to the rows
that produce it, and the drill has more than one level. Missing data is stated as not yet
available, with a named alternative route.

## 3. Google Ads

### Hierarchy

Three declared layers (`support.google.com/google-ads/answer/1704396`): account, campaign, ad
group. Budget and settings belong to the campaign; ads and keywords belong to the ad group. The
help page renders the containment as a diagram, and every management surface repeats that same
containment. There is no fourth layer invented for convenience.

### Status as a small closed vocabulary with remedies attached

The campaign Status column (`support.google.com/google-ads/answer/1722131`) carries exactly
seven values: `Eligible`, `Eligible (limited)`, `Not eligible`, `Paused`, `Removed`, `Pending`,
`Ended`. The help table pairs each with a description **and a "What to do" column**. The same
cell carries a second, independent state machine for the bid strategy: `Limited`,
`Misconfigured`, `Learning`. Two orthogonal states, one column, both actionable.

### Pacing against goal, and under-delivery before a human notices

The budget pacing insight (`support.google.com/google-ads/answer/13685469`) is a three-state
machine with a stated numeric trigger:

- `Limited by budget`: the campaign is already capping spend and has missed 5 percent or more
  of its potential traffic in the last week.
- `Budget remaining`: forecast spend leaves a significant part of the monthly budget unused.
- `On track`: forecast to use the budget fully; no action implied.

The forward-looking variant is the important one. `Projected to be limited by budget` comes
from a simulation of expected future traffic and fires on the same 5 percent threshold before
any shortfall has happened. A separate `Limited by budget soon` status is generated from
forecast seasonal events (`answer/13638324`) and the docs give a worked example: a 100 unit
budget currently spending 90 and returning 180 clicks, against a seasonal window where 150 would
return 450, so the simulator shows the recommended budget and the percentage gain.

The remedy is on the same row. A `Take action` column holds a `View recommendations` control
next to the status. In the campaign table, a chart icon on the row opens a budget simulator
popup offering the recommended daily budget or a typed one, applied in place
(`support.google.com/google-ads/answer/6385220`).

### Recommendations are reversible and expire

From `support.google.com/google-ads/answer/10169817`: applying a recommendation writes to the
account's Change history and can be undone there for 30 days, and the docs are explicit that a
partial application cannot be undone. Dismissals are reversible from a `Dismissed` filter on
the same page via an undismiss control, and a dismissal expires after 28 or 60 days depending
on the reason, so a suppressed problem comes back rather than disappearing.

### The device to steal

Under-delivery is a forecast state with a published threshold, not a chart the reader must
interpret. The remedy sits in the same row as the diagnosis. Every applied change is written to
a history that can undo it, and every dismissal has an expiry.

## 4. A professional editing timeline

Premiere is the primary reference for exact durations and dragging; Descript adds the
typed-duration and nudge-increment patterns.

### Exact durations

The Premiere timeline is five named parts (`helpx.adobe.com/premiere/desktop/.../navigation-controls-in-the-timeline.html`):
time ruler, work area bar, playhead, playhead position, zoom scroll bar. **The playhead
position is an editable numeric field**, not a readout: you can type a new time into it, or put
the pointer on it and drag left or right to scrub the number, and Cmd-clicking it flips the
display between timecode and a plain frame count. The ruler's own numbering changes granularity
with the zoom level, and the zoom scroll bar expands and contracts around the playhead so the
frame you care about stays put.

Descript makes duration a typed target: entering a target length in the Properties panel trims
the clip's right edge to match, and speed and duration are shown as one coupled control so
changing one visibly restates the other
(`help.descript.com/hc/en-us/articles/13823009241357`).

### Reordering by dragging

Premiere separates the drag results and names each one
(`helpx.adobe.com/premiere/desktop/.../different-ways-to-move-clips.html` and
`.../rearrange-clips-on-the-timeline.html`):

- Plain drag is an **overwrite**, the default, signalled by an overwrite icon during the drag.
- Holding Cmd while dropping converts it to an **insert**, and everything later in time shifts
  to make room.
- Cmd plus Option while dragging is a **rearrange edit**: a distinct Rearrange icon appears
  during the drag and, on release, only the destination track shifts. Other tracks are untouched.
- The **Ripple Edit** tool is the separate, explicit way to have neighbours close the gap.
- Comma inserts at the playhead and period overwrites at the playhead, so the same two results
  are available without a mouse.
- Arrow keys nudge the selection in small increments, with Cmd for larger ones.

Snapping is a toggle (`S`) and its feedback is exact: at the moment of alignment a vertical
line appears at the edge, marker or playhead being snapped to
(`helpx.adobe.com/premiere/desktop/.../snap-clips.html`).

Descript publishes its nudge increments as a table
(`help.descript.com/hc/en-us/articles/10256396876685`): arrow alone is one frame, Cmd plus
arrow is half a frame, Shift plus arrow is five frames, Option plus arrow is one second. And
attachment has its own signifier: dragging a layer's left edge toward a scene edge shows a red
dotted line, and releasing on it attaches.

### The device to steal

Every drag has a named result, a distinct cursor and a live signifier while it is in flight.
Nothing about ordering requires a dialog. The duration is a field you can type into, and the
ruler tells the truth about time at every zoom level.

## 5. Frame.io

Read directly from the real product capture in `refs/14-frameio-home.png` at pixel level, plus
`help.frame.io`.

### Per-asset technical status at a glance

- The asset grid is cards. Each card carries the **duration burned into the thumbnail corner**
  (for example `00:08`), the filename as the primary line, and the uploader plus the upload
  date as the secondary line.
- A toolbar above the grid holds `Appearance`, `Fields`, and `Sorted by Date Uploaded`. Above
  the first card is a count and a total size for the current folder.
- Hover or selection reveals a checkbox at the card's top left and an overflow at the bottom
  right; the selected card gets a coloured ring around the whole card, not a tick alone.
- The left rail keeps `Collections` whose entries are named for **states** (`Needs Review`,
  `Approved`) and for media kinds (`Videos`, `Images`, `Audio`), so a status is also a saved
  place you can navigate to.

`Fields` is the mechanism (`help.frame.io/en/articles/9092149`): 32 out-of-the-box metadata
fields exist per project, some of them read-only because they are facts about the file rather
than opinions about it (Audio Codec is the example given), and a dropdown toggles which of them
**print on the card itself**. Technical truth is a per-project display choice on the object, not
something buried behind a click.

`Status` is a labelled property with three values, `Needs Review`, `In Progress` and `Approved`,
edited from the Properties panel, and changing it notifies the project's users
(`help.frame.io/en/articles/9105311`). Custom fields extend the vocabulary through a manage-fields
dialog, so the closed vocabulary is closed but not fixed.

### Publishing where the product stops being able to tell the truth

`help.frame.io/en/articles/13321` publishes the proxy ladder exactly: the resolution rungs, the
codecs, the container, the profile levels, the bitrate ceilings, that framerate is preserved
from the source, that interlaced material is de-interlaced for web playback while the original
stays downloadable, and that files above 8 audio channels will not play back at all, with
channels 9 to 15 unsupported. The player's quality selector is bounded by both the source
resolution and the plan, and the frame guides for other aspect ratios come with an explicit
statement that the guides do not alter the file.

### The device to steal

The technical facts about a file are printed on the object itself, chosen per project, and the
read-only ones are visibly read-only. Where the system cannot verify or render something, it
says so in exact numbers instead of failing silently.

## 6. Figma

### Selection

From `help.figma.com/hc/en-us/articles/360040449873`:

- A click selects the **parent** by default. Double-click, or press Enter, to descend one level,
  repeatable until you reach the child you meant.
- Cmd-click is deep select: reach the top-level frame or any nested object directly.
- Right-click offers `Select layer`, which lists every layer under the cursor **in Layers-panel
  order**, so an ambiguous click becomes an explicit choice rather than a guess.
- Enter selects the child, Shift Enter the parent, Tab the next sibling, Shift Tab the previous.
- Shift-click adds to a selection and Shift-clicking the same object again removes it. A marquee
  selects; Cmd plus marquee selects nested objects; Shift plus marquee removes.
- In the Layers panel, Shift picks a range and Cmd picks individuals, matching the platform's
  file-list conventions rather than inventing new ones.
- `Option Cmd A` selects matching objects across frames, so one edit propagates to every copy.
- Hovering a layer row in the panel highlights that layer's position on the canvas with a box,
  and the highlight is a preference that can be turned off.
- A multi-selection is drawn as one bounding box and edits apply to the whole selection.

### Keyboard and the absence of modals

From `help.figma.com/hc/en-us/articles/360040328653`: arrow keys pan with nothing selected and
Shift increases the step, with the step scaled to the current zoom. A keyboard box-selection
tool (`Option Space`) puts a cursor on the canvas that arrows drive, Enter selects, Cmd plus
arrows draws a selection box, and **the viewport follows so the whole selection stays visible**.
The shortcuts panel (`Control Shift ?`) opens along the bottom, highlights shortcuts you have
already used, and the docs state you can keep working while it is open and watch it update. Even
the help is not a modal.

### Undo

Undo is `Cmd Z`, and beyond it version history is a browsable timeline back to the file's
creation (`help.figma.com/hc/en-us/articles/360038006754`). You can open a version and pan
around inside it **before** deciding to restore. Viewing history is available to viewers;
restoring requires edit access. Offline changes add explicit checkpoints to that history when
the connection returns.

### The device to steal

Direct manipulation with a named result and an explicit disambiguation path when a click is
ambiguous. Nothing that is done by moving something opens a dialog. History is previewable
before it is used, and the permission to read it is separate from the permission to apply it.

## 7. The best in-product AI agent

Primary reference: **Linear Agent**, because it is the only one of the three where the agent is
a first-class actor inside the same permission, assignment and audit system as the humans.
Comparators: Cursor for undo, Figma for on-canvas placement.

### Embedded in the surface it acts on

The agent opens on the record with `Cmd J` and is also invocable inline as `@Linear` in any
comment or description, so it acts where the work already is
(`linear.app/docs/linear-agent`). In the live product capture (`refs/01-linear-home.png`) the
panel is docked to the record, not to the app chrome, and the panel header prints the agent name
plus **the model in a chip**, with minimise, expand and close.

Agents are app users (`linear.app/docs/agents-in-linear`). Work is **delegated** by assigning it
to the agent while the human assignee remains the owner. Delegated issues still appear in the
human's My issues. Views can be filtered by `Delegate` and Insights can be segmented by it, so
the amount of work an agent is doing is a measurable property of the system rather than a
feeling.

### Preview before acting

Read directly from `refs/01-linear-home.png`: the panel shows a collapsible run trace labelled
with the elapsed time (`Worked for 7s`), then a plain-language list of what it changed per file,
then a change card reading `Changed 2 files +4 -4` with a **`Preview` control**, the draft PR
title, and the branch line naming the source and target. The activity feed on the issue mirrors
this as `Draft PR awaiting your review`. Nothing reaches the base branch without a human opening
the preview. `linear.app/docs/coding-sessions` states the same contract in words: Linear drafts
a PR and adds a diff to the issue, which you check before requesting review, then merge from
Linear.

Scope is bounded by identity: the docs state the agent can only reference or change content the
invoking user already has access to.

Its instructions are visible artefacts. Agent guidance exists at workspace and team level in a
markdown editor with history, and team guidance overrides workspace guidance. A successful
conversation can be saved as a reusable skill, invoked by slash command, shared to a team under
explicit permissions.

### How it is undone

- **Cursor** (`cursor.com/docs/agent/overview`) is the sharpest undo model: checkpoints are
  created automatically before significant changes; any checkpoint in the chat timeline can be
  clicked to **preview the files at that point**, then restored separately; a `Restore Checkpoint`
  control sits on previous requests; and the docs are explicit that checkpoints are local, are
  separate from git, and should only be used for undoing agent changes. Undo is an addressable
  object with a preview, not a single-step stack.
- **Figma** (`help.figma.com/hc/en-us/articles/37998629035799`) puts undo where the agent is:
  an `Undo` control inside the chat, with `Cmd Z` also working. The prompt box opens on the
  canvas against the current selection with `Cmd Enter`; each running prompt draws its own
  loading indicator **on the object it is working on**; several prompts run in parallel and each
  keeps its own thread; `@` references design-system components and variables so the agent works
  in the file's own vocabulary.

### The device to steal

The agent is an actor in the same permission and audit system as people, it shows a concrete
diff with a preview control before anything lands, and the reversal is an object you can point
at and inspect rather than a single Ctrl-Z.

## Devices that recur across all seven

These are the cross-cutting rules a critic can check on any Meridian surface:

1. Selection produces a contextual action bar, never a dialog (Linear, Figma, Frame.io).
2. A figure prints its basis beside it, or the product refuses to print the figure at all
   (Stripe's timezone, currency, range and complete-day rule; Linear's absolute plus percent on
   one unit; Google Ads' published 5 percent threshold).
3. Status is a small closed vocabulary, and every state names its remedy on the same row
   (Google Ads, Frame.io).
4. An empty field is an action, not a blank (`Set estimate`, `Add to project`,
   `Add document or link`).
5. Opening a record keeps its place in the list it came from (`1 / 31`).
6. Keyboard actions are taught in place, on the row that performs them (Linear's palette prints
   each row's shortcut; Figma's panel highlights the ones you have used).
7. Undo is an object you can point at and preview before using (Cursor checkpoints, Figma
   version history, Google Ads change history with a 30 day window).
8. Density comes from one row height and an explicit property choice, not from cramming
   (Linear 44 px plus display properties; Frame.io Fields toggles).
9. Every drag has a named result and a live signifier while in flight (Premiere's rearrange
   icon and snap line, Descript's red dotted attach line).
10. The agent is an actor in the same permission, assignment and audit system as the humans.

## The bar, per Meridian surface

Meridian's surfaces are the seventeen navigation entries defined at
`tv-break-dashboard/src/TVBreakDashboard.jsx:565-583`, plus the assistant dock mounted on every
page at `tv-break-dashboard/src/TVBreakDashboard.jsx:2524`, plus two capabilities that do not
exist in the repository at all. I verified those two absences rather than assuming them: a
recursive search of `kairos/`, `kairos_api/` and `tv-break-dashboard/src/` for `codec`,
`frame_rate`, `loudness` and `aspect ratio` returns no matches, so there is no per-ad media
verification anywhere; and there is no training surface in the dashboard.

| Meridian surface | Its bar | Three mechanics it must match or beat |
| --- | --- | --- |
| Overview | Linear My issues plus Google Ads budget pacing | 1. A three-state pacing verdict with a published numeric trigger, including a forward-looking projected state. 2. The remedy control on the same row as the diagnosis. 3. Zero clicks to the verdict, and every figure absolute plus percent on one unit. |
| Optimizer | Linear command palette | 1. `Cmd K` opens a grouped palette that prints each action's own shortcut. 2. Objective and guardrails are set without leaving the result view. 3. Every output figure states the basis it was computed on, Stripe style. |
| Schedule and the schedule editor | Premiere timeline plus Figma selection | 1. Named drag results with a distinct in-flight signifier and a snap line at the moment of alignment. 2. An editable numeric time field, typed or scrubbed, not a readout. 3. No dialog for any move, and undo always reachable. |
| Inventory | Linear issue list | 1. One row height, at least 17 rows and 9 facts per row in an 823 px viewport, the figure measured on Linear. 2. Hover reveals exactly one affordance. 3. Grouping, ordering and which properties print are user choices that can be saved as the default. |
| Break Library | Frame.io asset grid | 1. Technical facts printed on the object, chosen per view. 2. A closed status vocabulary that is also a navigable collection. 3. Selection ring plus checkbox plus contextual action bar, no dialog. |
| Campaigns | Google Ads | 1. Account, advertiser, campaign, flight containment with budget on exactly one layer. 2. Under-delivery surfaced as a forecast state before it happens. 3. The recommended action rendered on screen, applied in place, and written to a reversible history. |
| Forecasts | Stripe reports | 1. Range, timezone, currency and scope declared on the report. 2. Incomplete periods withheld rather than shown provisional, with the alternative route named. 3. Every amount opens the rows behind it, at more than one level. |
| Calendar | Linear cycle view | 1. Sunday-first Israeli week with Friday and Saturday shaded as the non-working band. 2. Progress as scope, started and completed, each absolute plus percent on one unit. 3. Segmentation tabs that re-slice the same total without navigating away. |
| Reports | Stripe balance summary | 1. Summary and itemised export declared to be the same numbers at two grains. 2. Category breakdown carrying gross, cost and net per row. 3. Clicking any amount resolves to its transactions. |
| Data | Frame.io plus Stripe availability | 1. Per-file technical status at a glance, with read-only facts visibly read-only. 2. A published statement of what the system cannot verify, in exact terms. 3. Staleness expressed as a named state with a remedy, never as a silent figure. |
| Advertisers | Linear record page plus Stripe drill | 1. `n / total` position counter so the whole filtered set is walkable from inside a record. 2. Empty properties rendered as actions. 3. Every money figure on the record opens its rows. |
| Agencies | Same as Advertisers | 1. One entity, one page, no duplicate under another name. 2. Rebate and commission terms shown with the basis they apply to. 3. Attributed, timestamped activity log including automated actors. |
| Pricing | Stripe basis discipline | 1. Every multiplier states whether it is live or wired-off, and against which base. 2. A price tester that resolves to the exact layers that produced the number. 3. No layer shown as active unless it multiplies a real figure. |
| Overrides | Google Ads recommendations | 1. Apply and dismiss are both first-class, both logged. 2. Applying is undoable from a history with a stated window. 3. A dismissal expires and the condition returns rather than vanishing. |
| Assistant dock (Kai) | Linear Agent, Cursor, Figma agent | 1. A concrete diff of what will change, with a preview control, before anything lands. 2. Undo as an addressable restore point that can be inspected before use. 3. Scoped to the caller's permissions and to the operator's own channel, with the model named on the surface. |
| Versions | Cursor checkpoints plus Figma version history | 1. A restore point can be opened and inspected before restoring. 2. Reading history and applying a restore are separately permissioned. 3. Every agent action and every applied override appears in the same timeline. |
| Settings | Linear preferences | 1. Israeli week fixed and visible, against the Monday-first weekday array still present at `TVBreakDashboard.jsx:585`. 2. Operator channel stated once and enforced everywhere downstream. 3. Nothing on this page can reach a training action. |
| Traffic operator pod (does not exist) | Premiere timeline plus Frame.io | 1. The pod as a real timeline whose ad durations sum exactly to the break length, visibly. 2. Per-ad verification of duration to the frame, format, aspect ratio and presence of audio, printed on the ad. 3. Reorder by dragging with a named result, and a failing ad impossible to miss. |
| Training console (does not exist) | Linear cycle rail plus Stripe availability | 1. Each held-out gate's verdict with its measured effect and the reason. 2. Coverage and drift as tri-state, never a confident guess. 3. Physically unreachable from any operator surface, and never confusable with a run. |

Two structural comparisons follow from the measurements above and belong in the blind
comparison rather than in a builder's summary. Linear reaches everything a person does through
two to four ungrouped destinations plus three grouped sections, and puts view switching inside
the content; Meridian currently offers seventeen flat navigation entries. Linear teaches its
keyboard on the row that performs each action; Meridian's palette, if it gets one, has to do the
same or it will not be used.
