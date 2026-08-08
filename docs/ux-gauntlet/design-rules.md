# Design rules

Written 2026-08-07, after the owner asked why the interface reads as though it
was built patch on patch without a specification. He was right, and the
diagnosis is worse than it looks: **the rules already existed in the code and
were never written down**, so every new surface reinvented them.

This file is the specification. It is short on purpose.

## 1. No one-sided accent bar. Anywhere.

A rule drawn down one edge of a box to mark it as important or as being in a
state is banned. It reads as an unfinished frame, it inverts under
right-to-left, and it multiplies: one appears, the next surface copies it, and
soon there are twenty-six of them in seventeen files, each a slightly different
width and colour. That is measured, not hypothetical.

```css
/* banned */
border-inline-start: 3px solid var(--amber);
```

**Use a full border in the state colour, with the matching soft background.**
This pattern already existed in `shell/styles.css` and was simply never applied
anywhere else.

```css
/* correct */
border: 1px solid var(--amber);
background: var(--amber-soft);
color: var(--amber);
```

The one legitimate use of a one-sided declaration is a **structural divider**
between cells or panels, at `1px solid var(--line)`. That is a rule between two
things, not an accent on one thing, and it stays.

A `Npx solid transparent` inline-start is also allowed where it reserves space
for a selected-state marker that appears on selection. That is layout, not
decoration.

## 2. The state palette is four colours and each has a soft twin

| Meaning | Border and text | Background |
|---|---|---|
| Warning, unknown, needs attention | `--amber` | `--amber-soft` |
| Error, breach, over a limit | `--red` | `--red-soft` |
| Active, current, the operator's own | `--teal` | `--teal-soft` |
| Informational, secondary | `--blue` | `--blue-soft` |
| Neutral emphasis, no state | `--line` | `--surface-muted` |

Never invent a fifth colour and never use a state colour for decoration. A
coloured box means something is true about the data in it.

## 3. A row of facts needs a separator, not whitespace

Three facts separated only by spaces read as one broken sentence, and the effect
is worse in Hebrew, where the eye has no capital letters to find the boundaries.

Separate them with a thin vertical rule or a middle dot, and give the separator
the muted colour so it recedes. If the facts wrap, the separator must not be
left stranded at the end of a line.

## 4. Buttons in one group share one size and one baseline

A card that mixes a large filled button, a medium outlined button and a bare
text link on the same row has three visual weights competing, and the reader
cannot tell which act is primary.

- **One primary act per card**, filled.
- Everything else is secondary, outlined, at the same height as the primary.
- A disclosure that expands the card in place is a text button with a chevron,
  and it belongs on its own line, not in the row of acts.
- Buttons in a group are aligned on their baseline and share a height. They do
  not wrap mid-group; if the group does not fit, it stacks entirely.

## 5. Display text is never hard-wrapped in source

One string, one line in the source, and let the layout wrap it. A manual break
becomes a wrong break at the next width.

## 6. Direction: a column aligns to the reading edge, always

Right in Hebrew, left in English. That is `text-align: start`, the logical
value, and it is already what almost every stylesheet says.

What broke it was not the alignment but the direction underneath it. A `dir`
attribute, or a `direction` property, is an OVERRIDE: it fixes the internal
order of a run, which is usually what the author wanted, and it also re-anchors
the element's own alignment, which is never what they wanted. `text-align:
start` resolves against the element's own direction, so a cell carrying
`dir="ltr"` in a Hebrew table aligns left while its neighbours align right.

Measured on the week-compare table before the fix: the row header sat 8px from
the cell's right edge and the three numeric cells sat 86px, 85px and 55px from
it, a 78px spread across one row. After: every cell at 8px, spread 0.

**Isolation, not override.** `unicode-bidi: isolate` protects a run's internal
order without touching alignment. Never put direction on a block or a cell; put
an isolating inline element inside it.

```jsx
/* banned */ <td className="numeric" dir="ltr">{money}</td>
/* correct */ <td className="numeric"><Figure>{money}</Figure></td>
```

Use `Figure` for a quantity or the date it is measured on, `Code` for a string
read literally (a path, a URL, an engine key, a log line), and `Name` for a name
that arrives as data and may be Hebrew or Latin. For a value being joined into a
plain string where no element can go, use the `isolate` function.

`DirectionRoot` is the one element that SHOULD carry a direction, because it is
establishing one rather than escaping one. There are three: the app shell, a
screen rendered before a locale exists, and an overlay rendered through a portal
which lands outside the shell's subtree and inherits nothing. Everything below a
root inherits from it. A panel writing `dir={locale === 'he' ? 'rtl' : 'ltr'}`
is not a root, it is the disease, and it should simply be deleted.

## 7. A card owns one inset, and content bleeds only on purpose

The owner sent screenshots of the Break Library and asked why, inside a single
card, the header and the prose sit inset from the card's edge while the rows run
flush against the border. Measured in Hebrew before the fix:

| Card | Its title | Its content | Off by |
|---|---|---|---|
| Breaks in the day | 17px | first cell at 1px | 16px |
| Ranked breaks | 17px | first grid cell at 1px | 16px |
| Compliance ledger | 17px | its prose at 1px | 16px |
| Revenue vs retention | 17px | its prose at 11px | 6px |

None of those surfaces was the cause. **A card owned no inset at all.**
`.page-panel` was a border, a radius and a background with `padding: 0`, so every
child had to invent its own inset, and a child that invented none ran to the
border. On one page, four cards insetted their prose four different ways: one by
`padding-inline`, one by `margin-inline: 10px`, one by `margin-inline: 16px`, and
one not at all.

**The inset is `--card-inset`, and it is stated once.** Nothing else may state
it. A card does not apply it to itself, because the head's rule and a table's row
rules have to reach the card's border while their content sits at the inset.

```jsx
<Card>
  <CardHead title="Ranked breaks" tools={<Export />} />
  <CardBody>prose, controls, a form</CardBody>
  <CardBleed><SomeTable /></CardBleed>
</Card>
```

**`CardBody` is the default. `CardBleed` is the exception, and it must be typed
where a reader of the code can see it.** That is the right way round because on
every surface the owner flagged, the inset element read as correct and the flush
element read as broken. A card is now right by omission, and forgetting to add
padding cannot produce the defect again.

**A bleeding element still aligns its own content to `--card-inset`.** Its row
rules span the whole card, which is what makes a long table readable, and its
first column sits directly under the card's title. Bleeding the background
*without* aligning the content is not the opt-in; it is the original defect
wearing a class name.

`--card-inset` is **16px, and 16px is deliberately not on the spacing scale**.
The product's real card inset was written out in sixty-two declarations at 16px,
nearly all in the inline position, and that is what the eye that built these
surfaces settled on. Moving it to `--space-5` would have shifted sixty
declarations by 2px with nothing measured behind it. The defect was never the
value; it was that the value had no home.

## 8. The spacing scale is missing its two most-used steps

The scale is `4, 8, 12, 14, 18, 24`, and it is irregular: `14` and `18` sit where
a 4px grid would put `16` and `20`. **This is worth an owner decision, and it is
recorded here rather than acted on.**

The evidence is not that the scale is untidy. It is that the two values the
product uses most heavily have no token at all:

| Value | Times written in px | Has a token |
|---|---|---|
| 12px | 66 | yes, `--space-3` |
| 10px | **63** | **no** |
| 16px | **62** | **no** (now `--card-inset`) |
| 14px | 54 | yes, `--space-4` |
| 8px | 52 | yes, `--space-2` |

So the surfaces did not drift from the scale; the scale was written without
looking at what the surfaces use, and they routed around it. `16px` now has a
semantic home as the card inset. `10px` still has none, and is the control and
chip inset.

Nothing was regularised. **No existing token's value was changed**, because
moving `--space-4` from 14 to 16 and `--space-5` from 18 to 20 would shift 72
live declarations by 2px each on pages nobody measured. Whether the scale should
gain a `10px` and a `16px` step is the owner's call.

## 9. Which file to edit, by kind of change

The owner's actual ask: a change should happen in ONE place and not require
hunting. This is that map. Where a kind of change has no single home yet, this
table says so rather than implying coverage.

| To change | Edit | Enforced by |
|---|---|---|
| Direction, isolation, how a figure or name sits in a line | `tv-break-dashboard/src/shell/bidi.jsx`, and the `.bidi-figure` / `.bidi-code` / `.bidi-name` rules in `shell/styles.css` | `npm run test:direction` |
| A card, and the inset of anything inside it | `tv-break-dashboard/src/shell/card.css` for the shape and the rule, `shell/primitives.jsx` for `Card` / `CardHead` / `CardBody` / `CardBleed` | `npm run test:card` |
| Colour, spacing, radius, shadow, typography scale | `tv-break-dashboard/src/tokens.css`, the only place a token may be defined | **partly**: `test:card` fails a padding written in px where a token says the same thing. Colour, radius, shadow and type are not enforced |
| State colours and their soft twins | `tokens.css`, applied per rule 2 above | not enforced |
| A calendar day, a list of days, a window, a timestamp or a clock | `tv-break-dashboard/src/shell/dates.js` | `npm run test:dates` |
| Number, currency and percent formatting | `tv-break-dashboard/src/shell/format.jsx` | not enforced |
| Shared components: metric, page header, status badge, data table, card | `tv-break-dashboard/src/shell/primitives.jsx` | **partly**: `test:card` fails a card built by hand, but nothing checks that a metric or a status badge went through the primitive |
| Wording and bilingual copy | `tv-break-dashboard/src/shell/copy.js` and the per-surface copy modules | not enforced |

Four rows have a guard: direction, the card, dates, and the spacing half of the
tokens row. **The other four are convention held by review alone**, and the
accent-bar rule in section 1 is the proof that convention alone does not hold: it
had to be swept twice. If you are adding a rule here, add a test with it or say
plainly that you did not.

Two of the four guarded rows say **partly**, and the word is doing real work.
`test:card` fails a padding written in px where a token exists, and it fails a
box that hand-builds a card. It says nothing about a colour written as a hex, a
radius written as `8px`, or a metric tile built without the `Metric` primitive.
Those are still unguarded, and 54 hand-built cards are budgeted rather than
fixed. Do not read a guard's presence as coverage of its whole row.

### Dates, in one paragraph

`dd/mm/yyyy` in both locales, because an English-reading buyer in Tel Aviv must
not read `04/28` while the person beside them reads `28/04`. A list of days
collapses consecutive days into runs, and a run prints both ends in full with a
tight hyphen between them: `28/04/2025-03/05/2025`. Items in a list are
separated by a spaced middle dot, `28/04/2025 · 12/05/2025`, and the two
separators differ on shape AND on spacing so no reader can mistake the joiner
inside a range for the one between items. Tight binds, loose separates. A run's
shape never changes when it crosses a month or a year, because eliding the parts
the two ends share is what makes a boundary a special case, and there is no
reason to have one. Past six runs the line states how many days it stopped
naming instead of running off the card.

## 10. What to do when a rule is missing

Add it here first, then apply it. A pattern that exists in one file and nowhere
else is not a rule, it is an accident waiting to be copied.
