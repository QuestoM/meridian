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

## 7. Which file to edit, by kind of change

The owner's actual ask: a change should happen in ONE place and not require
hunting. This is that map. Where a kind of change has no single home yet, this
table says so rather than implying coverage.

| To change | Edit | Enforced by |
|---|---|---|
| Direction, isolation, how a figure or name sits in a line | `tv-break-dashboard/src/shell/bidi.jsx`, and the `.bidi-figure` / `.bidi-code` / `.bidi-name` rules in `shell/styles.css` | `npm run test:direction` |
| Colour, spacing, radius, shadow, typography scale | `tv-break-dashboard/src/tokens.css`, the only place a token may be defined | not enforced |
| State colours and their soft twins | `tokens.css`, applied per rule 2 above | not enforced |
| Number, currency, percent and date formatting | `tv-break-dashboard/src/shell/format.jsx` | not enforced |
| Shared components: metric, page header, status badge, data table | `tv-break-dashboard/src/shell/primitives.jsx` | not enforced |
| Wording and bilingual copy | `tv-break-dashboard/src/shell/copy.js` and the per-surface copy modules | not enforced |

Only the first row has a guard. **Everything below it is convention held by
review alone**, and the accent-bar rule in section 1 is the proof that
convention alone does not hold: it had to be swept twice. If you are adding a
rule here, add a test with it or say plainly that you did not.

## 8. What to do when a rule is missing

Add it here first, then apply it. A pattern that exists in one file and nowhere
else is not a rule, it is an accident waiting to be copied.
