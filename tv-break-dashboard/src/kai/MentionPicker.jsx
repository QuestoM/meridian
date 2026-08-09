import React, { useEffect, useRef, useState } from 'react';
import { MenuItem, MenuList, Popper } from '@mui/material';
import { Building2, CalendarDays, MonitorPlay, Users } from 'lucide-react';
import { navItems } from '../shell/nav';
import { Code, DirectionRoot, Figure, Name } from '../shell/bidi';
import { formatDay, formatSpan } from '../shell/dates';
import { pageText } from '../shell/surface-helpers';
import { API_BASE } from '../shell/api';
// The trigger grammar lives in a plain module with no imports so a test can run
// it in node against the shipped code rather than describe it.
export { insertMention, readMentionQuery } from './mention-trigger';
import './mention-picker.css';

// The @ picker: the composer's way of pointing at a thing that exists.
//
// It has its own file and its own stylesheet because assistant-console.css sits
// exactly at the 450-line law and AssistantPanel.jsx sits two lines under it, so
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
// that the name inserted is exactly the one the store holds, spelled its way.
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
// destination the chip wears that rail item's own icon, read from nav.js rather
// than picked again here. The two kinds with no rail destination of their own
// name their lucide glyph on the server, in assistant_mentions_words.py.
const NAV_ICONS = Object.fromEntries(navItems);
const LOOSE_ICONS = { MonitorPlay, CalendarDays, Users, Building2 };

function iconFor(key) {
  if (typeof key === 'string' && key.startsWith('nav:')) {
    return NAV_ICONS[key.slice(4)] || MonitorPlay;
  }
  return LOOSE_ICONS[key] || MonitorPlay;
}

// One request per query, with the answer dropped when the query moved on. The
// staleness check is query equality rather than a request id, because two
// keystrokes that produce the same query should share an answer.
export function useMentionSearch(query, open) {
  const [state, setState] = useState({ rows: [], loading: false, query: null });
  const latest = useRef('');
  useEffect(() => {
    if (!open) {
      latest.current = '';
      setState({ rows: [], loading: false, query: null });
      return undefined;
    }
    const wanted = String(query || '');
    latest.current = wanted;
    let live = true;
    setState((prev) => ({ ...prev, loading: true }));
    const url = `${API_BASE}/api/assistant/mentions?q=${encodeURIComponent(wanted)}`;
    fetch(url)
      .then((response) => (response.ok ? response.json() : { rows: [] }))
      .then((body) => {
        if (!live || latest.current !== wanted) return;
        setState({ rows: Array.isArray(body.rows) ? body.rows : [], loading: false, query: wanted });
      })
      .catch(() => {
        if (!live || latest.current !== wanted) return;
        setState({ rows: [], loading: false, query: wanted });
      });
    return () => { live = false; };
  }, [query, open]);
  return state;
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

// Sit the panel on the composer's reading edge with a hair of clearance, and let
// it flip below when the dock is near the top of the viewport.
const MODIFIERS = [
  { name: 'offset', options: { offset: [0, 6] } },
  { name: 'flip', enabled: true },
  { name: 'preventOverflow', options: { padding: 8 } },
];

export function MentionPicker({ locale, anchorEl, open, rows, loading, activeIndex, onChoose, onHover }) {
  if (!open || !anchorEl) return null;
  const empty = !loading && !rows.length;
  return (
    <Popper open placement="top-start" anchorEl={anchorEl} className="mention-popper" modifiers={MODIFIERS}>
      <DirectionRoot locale={locale} className="card mention-picker" id="kai-mention-picker" role="listbox">
        {empty ? (
          // No-match and not-ours say the same sentence, deliberately. "None on
          // your channel" would confirm that a name the operator typed exists
          // somewhere, which is the one thing the boundary must never disclose.
          <p className="mention-empty">{pageText(locale, 'No matches', 'אין תוצאות')}</p>
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
                  <span className="mention-glyph" aria-hidden="true"><Glyph size={14} /></span>
                  <Name className="mention-label">{row.label}</Name>
                  <ParentPath parts={row.parent} locale={locale} />
                </MenuItem>
              );
            })}
          </MenuList>
        )}
      </DirectionRoot>
    </Popper>
  );
}
