import React, { useEffect, useMemo, useRef, useState } from 'react';
import { pageText } from '../../shell/format';
import { Code } from '../../shell/bidi';

// The command palette, taken from Linear's mechanic rather than its look.
//
// Three things make Linear's palette worth copying and all three are here: it
// opens over the surface without dimming it away, results are grouped by the
// kind of thing they act on, and every row prints its own shortcut on the right,
// so the palette teaches the keyboard instead of replacing it. Meridian had
// none of this: measured before this piece, the whole frontend contained zero
// occurrences of command palette, cmd k, hotkey or shortcut.
//
// A command is refused rather than hidden when it cannot run, and the row says
// why, so a planner never hunts for a control that was quietly removed.

export function shortcutText(shortcut, locale) {
  if (!shortcut) return '';
  return shortcut
    .map((key) => (key === 'mod' ? (isApple() ? 'Cmd' : 'Ctrl') : key))
    .join(locale === 'he' ? ' ואז ' : ' then ');
}

export function isApple() {
  if (typeof navigator === 'undefined') return false;
  return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
}

export function matches(command, query, locale) {
  const text = String(query || '').trim().toLowerCase();
  if (!text) return true;
  const haystack = [command.label(locale), command.group(locale), (command.keywords || []).join(' ')]
    .join(' ')
    .toLowerCase();
  return text.split(/\s+/).every((token) => haystack.includes(token));
}

export function CommandPalette({ open, commands, locale, onClose }) {
  const [query, setQuery] = useState('');
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  const visible = useMemo(
    () => commands.filter((command) => matches(command, query, locale)),
    [commands, query, locale],
  );

  useEffect(() => {
    if (!open) return;
    setQuery('');
    setCursor(0);
    const id = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(id);
  }, [open]);

  useEffect(() => {
    setCursor((current) => Math.min(current, Math.max(0, visible.length - 1)));
  }, [visible.length]);

  useEffect(() => {
    if (!open) return undefined;
    const node = listRef.current?.querySelector('[data-cursor="true"]');
    node?.scrollIntoView({ block: 'nearest' });
    return undefined;
  }, [cursor, open]);

  if (!open) return null;

  const groups = [];
  visible.forEach((command, index) => {
    const name = command.group(locale);
    const last = groups[groups.length - 1];
    if (last && last.name === name) last.items.push({ command, index });
    else groups.push({ name, items: [{ command, index }] });
  });

  function runAt(index) {
    const entry = visible[index];
    if (!entry || entry.disabled) return;
    onClose();
    entry.run();
  }

  function onKeyDown(event) {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setCursor((current) => Math.min(visible.length - 1, current + 1));
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setCursor((current) => Math.max(0, current - 1));
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      runAt(cursor);
    }
  }

  return (
    <div className="plan-palette-scrim" role="presentation" onMouseDown={onClose}>
      <div
        className="plan-palette"
        role="dialog"
        aria-modal="true"
        aria-label={pageText(locale, 'Command palette', 'לוח פקודות')}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <input
          ref={inputRef}
          className="plan-palette-input"
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={pageText(locale, 'Search commands', 'חיפוש פקודות')}
          aria-label={pageText(locale, 'Search commands', 'חיפוש פקודות')}
        />
        <div className="plan-palette-list" ref={listRef} role="listbox">
          {visible.length === 0 && (
            <p className="plan-palette-empty">
              {pageText(locale, 'No command matches that.', 'אין פקודה שתואמת לזה.')}
            </p>
          )}
          {groups.map((group) => (
            <div className="plan-palette-group" key={group.name}>
              <p className="plan-palette-group-name">{group.name}</p>
              {group.items.map(({ command, index }) => (
                <button
                  key={command.id}
                  type="button"
                  role="option"
                  aria-selected={index === cursor}
                  aria-disabled={Boolean(command.disabled)}
                  data-cursor={index === cursor ? 'true' : 'false'}
                  className={`plan-palette-row${index === cursor ? ' is-cursor' : ''}${command.disabled ? ' is-disabled' : ''}`}
                  onMouseEnter={() => setCursor(index)}
                  onClick={() => runAt(index)}
                >
                  <span className="plan-palette-label">
                    {command.label(locale)}
                    {command.disabled && command.disabledReason && (
                      <small className="plan-palette-reason">{command.disabledReason(locale)}</small>
                    )}
                  </span>
                  <kbd className="plan-palette-keys"><Code>{shortcutText(command.shortcut, locale)}</Code></kbd>
                </button>
              ))}
            </div>
          ))}
        </div>
        <p className="plan-palette-foot">
          {pageText(locale, 'Arrows move, Enter runs, Esc closes', 'חצים לניווט, Enter להפעלה, Esc לסגירה')}
        </p>
      </div>
    </div>
  );
}

export default CommandPalette;
