import { useCallback, useEffect, useRef, useState } from 'react';
import { documentDirection } from '../shell/bidi';
import { API_BASE } from '../shell/api';
import { insertMention, readMentionQuery } from './mention-trigger';
import { addRef, edgeKeys, shiftRefs } from './mention-refs';

// The picker's state: what is open, what it is showing, and which key does what.
//
// It has its own module because AssistantComposer.jsx holds the textarea and the
// panel's own key handler, and neither of those is the place for a second mode
// with a navigation stack in it. What lives here is one hook and the two fetches
// behind it; what renders it is MentionPicker.jsx.
//
// TWO MODES, ONE POPUP.
//
// SEARCH is the default and the 90% case: a flat ranked list across kinds, which
// is what both reference products ship and they are right that it suffices for
// a code editor. DRILL is the part both of them verifiably declined -- one kills
// chaining with a trailing space, the other collapses the directory kind into
// the file kind -- and they could afford to because a flat fuzzy search
// substitutes for navigation WHEN EVERY LEAF HAS A UNIQUE TYPEABLE PATH. A spot
// has no name. So this product has to build what both declined.
//
// THE LEADING EDGE RESOLVES FROM THE DOCUMENT DIRECTION AND IS NEVER HARDCODED.
// In Hebrew the leading edge is the right, so the key that descends is
// ArrowLeft: it points INTO the row, which is the same gesture a file tree makes
// in English with ArrowRight. Getting this backwards would not be a cosmetic
// bug; it would make the drill unreachable in the language the product is
// written in.
//
// TYPING RETURNS TO SEARCH, deliberately and always. The trail is cleared by any
// change to the text, so the operator can never be several levels deep with a
// query on screen that no longer describes what is listed. Descending and
// ascending are the only things that move the trail, and neither touches the
// sentence being written.

export const LADDER_LIMIT = 4;

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) return null;
  return response.json();
}

// One request per state, with the answer dropped when the state moved on. The
// staleness check is key equality rather than a request id, because two
// keystrokes that produce the same key should share an answer.
export function useMentionRows(open, query, parent) {
  const [state, setState] = useState({ rows: [], absent: null, omitted: 0, loading: false });
  const latest = useRef('');
  const key = open ? `${parent ? `${parent.type}:${parent.id}:${parent.edge || ''}` : ''}|${query}` : '';
  useEffect(() => {
    if (!open) {
      latest.current = '';
      setState({ rows: [], absent: null, omitted: 0, loading: false });
      return undefined;
    }
    latest.current = key;
    let live = true;
    setState((prev) => ({ ...prev, loading: true }));
    const url = parent
      ? `${API_BASE}/api/assistant/mentions/children?type=${encodeURIComponent(parent.type)}&id=${encodeURIComponent(parent.id)}${parent.edge ? `&edge=${encodeURIComponent(parent.edge)}` : ''}`
      : `${API_BASE}/api/assistant/mentions?q=${encodeURIComponent(query)}`;
    fetchJson(url)
      .then((body) => {
        if (!live || latest.current !== key) return;
        const rows = body && Array.isArray(body.rows) ? body.rows : [];
        setState({
          rows,
          absent: body && body.absent ? body.absent : null,
          omitted: body && Number.isFinite(body.omitted) ? body.omitted : 0,
          loading: false,
        });
      })
      .catch(() => {
        if (!live || latest.current !== key) return;
        setState({ rows: [], absent: null, omitted: 0, loading: false });
      });
    return () => { live = false; };
  }, [open, key]);
  return state;
}

export function useMentions({ locale, composerRef, question, onQuestionChange, refs, onRefsChange }) {
  const [run, setRun] = useState(null);
  const [trail, setTrail] = useState([]);
  const [active, setActive] = useState(0);
  const open = run !== null;
  const parent = trail.length ? trail[trail.length - 1] : null;
  const { rows, absent, omitted, loading } = useMentionRows(open, run ? run.query : '', parent);
  // The ONE place a locale becomes a direction, and the shell's own function
  // does it. Nothing in this piece decides which way the page reads.
  const keys = edgeKeys(documentDirection(locale));

  useEffect(() => { setActive(0); }, [run && run.query, trail.length]);

  // Every text change carries the spans across it and returns to the flat list.
  const applyText = useCallback((next) => {
    onRefsChange(shiftRefs(question, next, refs));
    onQuestionChange(next);
  }, [question, refs, onQuestionChange, onRefsChange]);

  function sync(element) {
    setTrail([]);
    setRun(element ? readMentionQuery(element.value, element.selectionStart) : null);
  }

  // Accepting a row: the store's own name goes into the sentence as plain text,
  // and the typed {type, id} is bound to exactly those characters. The name is
  // what the free-text path already resolves, so an operator who deletes the
  // chip and types the same name by hand reaches the same object.
  function choose(row) {
    if (!run || !row) return;
    const next = insertMention(question, run, row.label);
    setRun(null);
    setTrail([]);
    onQuestionChange(next.text);
    onRefsChange(addRef(question, next.text, refs, {
      start: next.start,
      len: next.len,
      type: row.type,
      id: row.id,
      label: row.label,
      kindHe: row.kind_he || '',
      kindEn: row.kind_en || '',
      icon: row.icon || '',
    }));
    const element = composerRef && composerRef.current;
    if (element) {
      // After React has written the new value, put the caret past the name.
      window.requestAnimationFrame(() => {
        element.focus();
        element.setSelectionRange(next.caret, next.caret);
      });
    }
  }

  function descend(row) {
    if (!row || !row.container || trail.length >= LADDER_LIMIT) return;
    setTrail((prev) => [...prev, { type: row.type, id: row.id, label: row.label, edge: '' }]);
  }

  function ascend() {
    setTrail((prev) => prev.slice(0, -1));
  }

  // The breadcrumb's own move: back to the step that was clicked, and -1 back to
  // the flat list. One function for both, so the header cannot drift out of step
  // with the key that does the same thing.
  function upTo(index) {
    setTrail((prev) => prev.slice(0, Math.max(0, index + 1)));
  }

  function onKeyDown(event, fallback) {
    if (open && event.key === 'Escape') { event.preventDefault(); setRun(null); setTrail([]); return; }
    if (open) {
      const row = rows[active] || null;
      if (event.key === keys.descend && row && row.container) { event.preventDefault(); descend(row); return; }
      if (event.key === keys.ascend && trail.length) { event.preventDefault(); ascend(); return; }
      if (rows.length) {
        if (event.key === 'ArrowDown') { event.preventDefault(); setActive((i) => (i + 1) % rows.length); return; }
        if (event.key === 'ArrowUp') { event.preventDefault(); setActive((i) => (i - 1 + rows.length) % rows.length); return; }
        // Enter accepts at any depth, which is the whole point of the drill: a
        // container is a thing you can point at as well as a thing you can enter.
        if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); choose(row); return; }
        if (event.key === 'Tab') { event.preventDefault(); choose(row); return; }
      }
    }
    if (fallback) fallback(event);
  }

  return {
    open, rows, absent, omitted, loading, active, trail, keys,
    setActive, choose, descend, ascend, upTo, onKeyDown, sync, applyText,
  };
}
