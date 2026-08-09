import React from 'react';
import { ButtonBase, IconButton, MenuItem, MenuList, Popper } from '@mui/material';
import { Building2, CalendarDays, ChevronLeft, ChevronRight, MonitorPlay, Users } from 'lucide-react';
import { navItems } from '../shell/nav';
import { Code, DirectionRoot, Figure, Name, documentDirection } from '../shell/bidi';
import { formatDay, formatSpan } from '../shell/dates';
import { pageText } from '../shell/surface-helpers';
export { insertMention, readMentionQuery } from './mention-trigger';
export { useMentions, useMentionRows } from './mention-state';
export { edgeKeys } from './mention-refs';
import './mention-picker.css';

// The @ picker: the composer's way of pointing at a thing that exists.
//
// It has its own file and its own stylesheet because assistant-console.css sits
// exactly at the 450-line law and AssistantPanel.jsx sat two lines under it, so
// neither could absorb a floating panel. The split was planned before the first
// edit rather than discovered by a failing count.
//
// THE ONE THING THIS FILE IS NOT ALLOWED TO BE is a gate. The structured mention
// path of the reference product this design was measured against was exercised
// zero times in 10,952 recorded turns, which says plainly that a mention system
// that is the only way to name a thing will not be used. So choosing a row
// inserts the STORE'S OWN NAME as plain text: the free-text path the assistant
// already has resolves it, typing the same name by hand still works, and an
// operator who never presses @ sees no change at all. What the picker buys is
// that the name inserted is exactly the one the store holds, spelled its way,
// and that a typed {type, id} rides beside it so two same-named things are two
// different references.
//
// THE INDEX IS SERVER-SIDE and this file never builds one. The saved plan holds
// every channel, because the retention model is measured against the
// competitive lineup, so a client-side index would put rival rows in the
// browser. Everything here fetches; nothing here filters.
//
// THE POPUP IS PORTALLED, which is the fourth root case bidi.jsx enumerates: a
// portalled element lands outside the shell's subtree and resolves its
// direction against the DOCUMENT. So it carries DirectionRoot, and that is the
// only place direction is stated in this piece.

// The glyph is navigational identity, not decoration: where a kind has a rail
// destination the row wears that rail item's own icon, read from nav.js rather
// than picked again here. The two kinds with no rail destination of their own
// name their lucide glyph on the server, in assistant_mentions_words.py.
const NAV_ICONS = Object.fromEntries(navItems);
const LOOSE_ICONS = { MonitorPlay, CalendarDays, Users, Building2 };

export function iconFor(key) {
  if (typeof key === 'string' && key.startsWith('nav:')) {
    return NAV_ICONS[key.slice(4)] || MonitorPlay;
  }
  return LOOSE_ICONS[key] || MonitorPlay;
}

// The dim second line. It arrives as PARTS and is rendered as parts, never
// concatenated and wrapped once: isolate's own documentation warns against
// wrapping a phrase, and a parent path here genuinely mixes a date with a name.
// A calendar day is read in shell/dates.js and nowhere else, which is why the
// server sends the ISO string and the formatting happens here.
function ParentPath({ parts, locale }) {
  if (!parts || !parts.length) return null;
  return (
    <span className="mention-parent">
      {parts.map((part, index) => (
        <React.Fragment key={`${part.kind}-${part.text || part.from}-${index}`}>
          {index ? <span className="mention-parent-dot" aria-hidden="true">·</span> : null}
          {part.kind === 'span' ? (
            <Figure>{part.to && part.to !== part.from
              ? formatSpan(part.from, part.to, locale)
              : formatDay(part.from)}</Figure>
          ) : null}
          {part.kind === 'name' ? <Name>{part.text}</Name> : null}
          {part.kind === 'code' ? <Code>{part.text}</Code> : null}
          {part.kind === 'figure' ? <Figure>{part.text}</Figure> : null}
        </React.Fragment>
      ))}
    </span>
  );
}

// THE BREADCRUMB. A drill with no header is a list that has silently stopped
// being the list you asked for, so the header says where you are and every step
// of it goes back. The root step returns to the flat search, which is the mode
// the operator started in and the one typing always returns to.
function Breadcrumb({ trail, locale, onUp, onRoot }) {
  if (!trail.length) return null;
  return (
    <div className="mention-crumbs">
      <ButtonBase className="mention-crumb" onMouseDown={(e) => e.preventDefault()} onClick={onRoot}>
        {pageText(locale, 'All', 'הכל')}
      </ButtonBase>
      {trail.map((step, index) => (
        <React.Fragment key={`${step.type}:${step.id}`}>
          <span className="mention-crumb-sep" aria-hidden="true">·</span>
          <ButtonBase
            className="mention-crumb"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => onUp(index)}
          >
            <Name>{step.label}</Name>
          </ButtonBase>
        </React.Fragment>
      ))}
    </div>
  );
}

// The descend affordance, on the LEADING edge of a container row. Which glyph
// that is comes from the document direction and is never hardcoded: in Hebrew
// the leading edge is the right, so the chevron points left, which is the same
// gesture into the row that a chevron pointing right makes in English.
function DescendControl({ locale, label, onDescend }) {
  const Chevron = documentDirection(locale) === 'rtl' ? ChevronLeft : ChevronRight;
  return (
    <IconButton
      size="small"
      className="mention-descend"
      aria-label={label}
      title={label}
      onMouseDown={(event) => event.preventDefault()}
      onClick={(event) => { event.stopPropagation(); onDescend(); }}
    >
      <Chevron size={14} />
    </IconButton>
  );
}

// Sit the panel on the composer's reading edge with a hair of clearance, and let
// it flip below when the dock is near the top of the viewport.
//
// NOT MEASURED IN A BROWSER. These three modifiers are reasoned from the dock's
// layout and from Popper's own defaults, exactly as R1 said of the same block:
// the offset, the flip and the overflow padding have not been checked against a
// rendered dock at its narrow width. That measurement is still owed.
const MODIFIERS = [
  { name: 'offset', options: { offset: [0, 6] } },
  { name: 'flip', enabled: true },
  { name: 'preventOverflow', options: { padding: 8 } },
];

export function MentionPicker({
  locale, anchorEl, open, rows, absent, omitted, loading, activeIndex, trail,
  onChoose, onHover, onDescend, onUpTo,
}) {
  if (!open || !anchorEl) return null;
  const descendLabel = pageText(locale, 'Show what is inside', 'הצג מה יש בפנים');
  // No-match and not-ours say the same sentence, deliberately. "None on your
  // channel" would confirm that a name the operator typed exists somewhere,
  // which is the one thing the boundary must never disclose. A drill that came
  // back empty says the server's own stated absence instead of showing nothing,
  // because an empty list in a picker reads as "zero of them" and the two are
  // different claims.
  const absentText = absent
    ? (locale === 'he' ? absent.reason_he || absent.reason : absent.reason)
    : pageText(locale, 'No matches', 'אין תוצאות');
  return (
    <Popper open placement="top-start" anchorEl={anchorEl} className="mention-popper" modifiers={MODIFIERS}>
      <DirectionRoot locale={locale} className="card mention-picker" id="kai-mention-picker" role="listbox">
        <Breadcrumb trail={trail || []} locale={locale} onUp={onUpTo} onRoot={() => onUpTo(-1)} />
        {!loading && !rows.length ? (
          <p className="mention-empty">{absentText}</p>
        ) : (
          <MenuList dense disablePadding autoFocus={false} className="mention-list">
            {rows.map((row, index) => {
              const Glyph = iconFor(row.icon);
              return (
                <MenuItem
                  key={`${row.type}:${row.id}`}
                  className="mention-row"
                  selected={index === activeIndex}
                  // The textarea keeps the focus throughout, so the arrow keys
                  // and Enter stay where the operator is typing.
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => onHover(index)}
                  onClick={() => onChoose(row)}
                >
                  {row.container
                    ? <DescendControl locale={locale} label={descendLabel} onDescend={() => onDescend(row)} />
                    : <span className="mention-descend-gap" aria-hidden="true" />}
                  <span className="mention-glyph" aria-hidden="true"><Glyph size={14} /></span>
                  <Name className="mention-label">{row.label}</Name>
                  <ParentPath parts={row.parent} locale={locale} />
                </MenuItem>
              );
            })}
          </MenuList>
        )}
        {omitted > 0 ? (
          <p className="mention-omitted">
            {pageText(locale, 'More match than fit here', 'יש יותר התאמות ממה שנכנס כאן')}
            {' '}<Figure>{String(omitted)}</Figure>
          </p>
        ) : null}
      </DirectionRoot>
    </Popper>
  );
}
