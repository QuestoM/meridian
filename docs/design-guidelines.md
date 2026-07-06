# Kairos dashboard design guidelines

This is the real design language of the product, extracted from the token
system and the canonical component patterns already in the codebase, not
invented. Every surface should read as one product. When a screen looks like
patchwork it is because it drifted from these; the fix is always to move it back
onto them, never to add a new one-off style.

The product is a light, calm, data-dense operations console in Hebrew, right to
left. It should feel precise and trustworthy, never decorative and never
generated. No gradients-for-decoration, no glow, no shadow stacking, no emoji,
no playful accent shapes. Confidence comes from consistent rhythm and honest
hierarchy, not from ornament.

## Tokens are the only source of values

Colours, radii, type sizes, weights and spacing come from the CSS variables in
`:root` (styles.css). A raw hex, a raw px font-size, or a raw px radius in a
component is a drift and should be replaced with the token. The palette:

- Surfaces: `--bg` #f7f8fa (page), `--surface` #fff (panels), `--surface-muted`
  #fbfcfd (insets, table stripes).
- Text: `--ink` (primary), `--muted` (secondary), `--subtle` (captions, meta).
  Never put primary text below `--muted` contrast on a light surface.
- Lines: `--line` (default border), `--line-strong` (emphasis).
- Semantics: `--teal` (positive, the brand accent), `--blue` (informational),
  `--amber` (attention, needs review), `--red` (problem). Each has a soft
  companion (`--teal-soft`, `--amber-soft`) for fills. Use the semantic, never a
  raw colour, and never a semantic for decoration.
- Radii: `--radius-sm` 6px (chips, controls), `--radius-md` 8px (cards, panels),
  `--radius-lg` 10px (rare, large surfaces). One radius per element; do not mix.
- Type: `--text-sm` 12px is the floor for meaningful text; `--text-2xs` 10px is
  for a chip label only, never a sentence. Body is `--text-base` 13px. Weights
  come from the `--weight-*` scale.
- Spacing: `--space-*`. Padding and gaps come from the scale so rhythm is shared.

## The canonical components

Reuse these; do not reinvent them per screen.

- Panel: `.page-panel` is the card, `background var(--surface)`, `1px solid
  var(--line)`, `--radius-md`, its header is `.panel-head` (a title `h2` and an
  optional muted subtitle `span`). Every boxed region on a page is a panel.
- Stat tile: the number is the hero at `--text-2xl`+, the label is `--text-sm`
  muted above or below it, numbers are `dir="ltr"`. Never let a wrapping label
  push the value out of alignment; the value and its label are one aligned unit.
- Chip: a small rounded-`--radius-sm` token with `--text-2xs`/`--text-xs` label,
  a soft semantic fill for status (`--teal-soft`, `--amber-soft`) or a neutral
  `--surface-muted` for a code or key. A chip is a label, never a sentence.
- Button: the primary action is the dark `--primary` fill; secondary is a quiet
  outline (`1px solid var(--line)`, `--surface`); a tertiary is text-only. One
  primary per action group.
- Row and list item: a list of things (versions, proposals, uploads, breaks) is
  a stack of rows with even vertical rhythm; each row aligns its columns on a
  shared baseline. A row is not a mini-card with its own border unless the list
  genuinely needs separation, in which case every row in the list gets the same
  treatment. Never mix bordered and borderless rows in one list.
- Notice and callout: an inline message (unavailable, stale, empty, warning)
  uses a full box: `1px solid var(--line)`, a soft semantic fill, `--radius-md`,
  even padding on all four sides. A one-sided accent bar is allowed ONLY as a
  full box plus a thicker inline-start edge in the same semantic colour; a bar
  on one side with no box is forbidden, it reads as unfinished. Applied
  consistently, every notice of the same kind looks identical across screens.

## The rules that keep it coherent

These are the lines to hold. Most patchwork is one of these broken.

1. No lopsided borders. An element is either fully bounded or not bounded; a
   border or accent on a single side without a full box is not allowed. The
   accent-edge callout above is the one sanctioned exception and only as a box
   plus edge.
2. One radius, one border weight per element. Do not nest a 10px card inside a
   6px card inside an 8px panel; pick the token for the level and hold it.
3. Numbers are `dir="ltr"` inside the RTL layout, always, with their unit. A
   bare number that inherits RTL will misread.
4. No text below the floor. `--text-sm` for anything a person reads as words;
   `--text-2xs` only for a single-token chip label.
5. Even rhythm. Padding and gaps come from `--space-*`; a list has one gap
   value, a panel has one padding value. Cramped, uneven spacing is the clearest
   patchwork tell.
6. Alignment over decoration. Columns in a row share a baseline; a label and its
   value are one unit; a chip row wraps as a unit. Do not fix misalignment with
   a nudge, fix the structure.
7. Honest states are designed, not bolted on. Loading, empty, unavailable and
   error each use the notice or empty pattern, never a raw sentence floating in
   the layout.
8. One accent language. `--teal` is the brand positive and the only decorative
   accent; `--blue`/`--amber`/`--red` are reserved for their meanings. Do not
   colour a heading teal for flavour or an accent amber where nothing needs
   attention.
9. RTL is first-class. Use logical properties (`margin-inline`, `border-inline-
   start`, `padding-inline`) so mirroring is correct; never hard-code left/right
   for anything that should mirror.
10. Match the neighbourhood. A new element on a screen copies the spacing, radius
    and type of the panel it lives in. When in doubt, find the nearest canonical
    component and reuse its classes rather than writing new CSS.

## How to apply this to a drifted screen

Read the screen against the ten rules, list each concrete violation with its
selector, then move each one onto the canonical component or token. Do not
restyle what already conforms. The goal is that a person moving between the
Assistant, the Overview, the Schedule and the Settings pages never feels they
changed products.
