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

## 6. What to do when a rule is missing

Add it here first, then apply it. A pattern that exists in one file and nowhere
else is not a rule, it is an accident waiting to be copied.
