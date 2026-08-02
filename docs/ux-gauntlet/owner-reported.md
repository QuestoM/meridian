# Defects the owner reported from the running product

Reported while wave one was in flight, measured by the lead, and left for the
piece that owns the surface rather than patched across ownership lines. Each one
names where it lives and what the fix is, so the owner does not have to
rediscover it.

## 1. The navigation carries two products at once

**Owner: shell, with P5 for the pages that moved.**

The sidebar still lists the flat page set from before the campaign, while the
grouped destinations wave one built are also live. The same capability is
therefore reachable twice under two names:

| Sidebar entry | The same thing inside a destination |
|---|---|
| לוח אירועים | Rules, tab "The calendar" (לוח האירועים) |
| תמחור | Rules, tab "The rate card" (כרטיס התעריפים) |

This is not a rendering bug. It is the old information architecture that was
never retired when the new one landed, and it is visible to anyone who opens
Settings: they see tabs that duplicate entries they just walked past in the rail.
The whole premise of the rebuild is nine destinations, so the rail is what has to
give, and every entry that now lives inside a destination is removed from it with
its route redirected rather than deleted, so a bookmark still lands somewhere.

Check the rest of the Rules tabs against the rail for the same collision before
fixing: the owner reported "some of them at least look like unnecessary
duplication", and two confirmed cases means the audit is worth doing in full.

## 2. The settings page is cramped against the frame

**Owner: shell.**

`.workspace` is a four-row grid with no inline padding. `.top-bar` supplies its
own `padding: 0 24px`, so the header sits correctly, but the rows below inherit
nothing and the panels sit a handful of pixels from the frame while the header
sits at twenty-four. The page reads as though the content escaped its column.

The fix belongs on the workspace rows rather than on each page, otherwise every
page re-solves it and they drift apart, which is how the header and the body came
to disagree in the first place. Use logical padding so it holds in Hebrew.

## 3. Header controls wrap and overflow their buttons

**Owner: shell.**

Measured on Overview at a wide viewport: "Run Optimization" wraps to two lines,
"Apply to weekly schedule" wraps to three and spills below its own border, and
"Nov 1", "Live API" and "Updated 03:37 AM" all wrap. The row is over-packed and
the labels are allowed to break.

Two parts to the fix and both are needed: the labels stop wrapping, and the row
stops trying to hold every control at every width. A button whose text has
overflowed its border is worse than a button that is not there, because it reads
as damage rather than as density.

## 4. Break markers on the schedule timeline are unreadable

**Owner: P3, handed to P10 in wave two.**

The dark pills on the day timeline print a single character per line. It is not
text wrapping: `.break-chip-clock` already sets `white-space: nowrap`. `BreakChip`
renders three stacked lines (clock, detail, meta) and the pill, whose width is
derived from the break's duration, is about twenty pixels wide at a full-day
zoom, so each line is clipped to its first character. `.editor-break` declares
`min-width: 64px` and the derived width defeats it.

Do not shrink the text to fit. Below a width that can hold the clock, the pill
shows no text at all and stays a clean marker; the full detail is already in
`title` and `aria-label`, so nothing is lost. This is the honest-empty-state rule
applied to layout: when there is no room to say something truthfully, say
nothing rather than say a fragment.

## 5. The job picker centred its card titles

**Owner: P1. Fixed.**

`.today-job-row` set `justify-items: start` but not `justify-content`, and the
button arrives from MUI centred with that style injected at runtime rather than
into the stylesheet. The item alignment applied inside the column track while the
track itself stayed centred, so each card read as a centred title over
start-aligned detail lines. Both properties are now set, both logical, so the
card flips whole in Hebrew.
