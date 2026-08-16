import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import { Button } from '../../studio/actions';
import { X } from 'lucide-react';
import { InputControl, Pressable } from '../../studio/dom-controls';
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
  const dialogRef = useRef(null);
  const previousFocusRef = useRef(null);
  const id = useId();
  const listId = `${id}-list`;
  const titleId = `${id}-title`;

  const visible = useMemo(
    () => commands.filter((command) => matches(command, query, locale)),
    [commands, query, locale],
  );

  useEffect(() => {
    if (!open) return;
    previousFocusRef.current = document.activeElement;
    setQuery('');
    setCursor(0);
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    const id = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => {
      window.clearTimeout(id);
      if (dialog?.open) dialog.close();
      previousFocusRef.current?.focus?.();
    };
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

  function trapDialogFocus(event) {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = Array.from(dialogRef.current?.querySelectorAll('input, button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])') || []);
    if (focusable.length === 0) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function closeFromBackdrop(event) {
    if (event.target !== dialogRef.current) return;
    const bounds = dialogRef.current.getBoundingClientRect();
    const outside = event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
    if (outside) onClose();
  }

  return (
      <dialog
        ref={dialogRef}
        className="card plan-palette"
        aria-labelledby={titleId}
        onCancel={(event) => {
          event.preventDefault();
          onClose();
        }}
        onKeyDown={trapDialogFocus}
        onClick={closeFromBackdrop}
      >
        <header className="plan-palette-head">
          <h2 id={titleId}>{pageText(locale, 'Command palette', 'לוח פקודות')}</h2>
          <Button
            type="button"
            variant="text"
            disableRipple
            className="plan-palette-close"
            aria-label={pageText(locale, 'Close command palette', 'סגירת לוח הפקודות')}
            onClick={onClose}
          >
            <X size={20} aria-hidden="true" />
          </Button>
        </header>
        <InputControl
          ref={inputRef}
          className="plan-palette-input"
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={pageText(locale, 'Search commands', 'חיפוש פקודות')}
          aria-label={pageText(locale, 'Search commands', 'חיפוש פקודות')}
          role="combobox"
          aria-autocomplete="list"
          aria-expanded="true"
          aria-controls={listId}
          aria-activedescendant={visible[cursor] ? `${id}-option-${cursor}` : undefined}
        />
        <div className="plan-palette-list" ref={listRef} id={listId} role="listbox" aria-label={pageText(locale, 'Available commands', 'פקודות זמינות')}>
          {visible.length === 0 && (
            <p className="plan-palette-empty">
              {pageText(locale, 'No command matches that.', 'אין פקודה שתואמת לזה.')}
            </p>
          )}
          {groups.map((group) => (
            <div className="plan-palette-group" key={group.name}>
              <p className="plan-palette-group-name">{group.name}</p>
              {group.items.map(({ command, index }) => (
                <Pressable
                  key={command.id}
                  id={`${id}-option-${index}`}
                  type="button"
                  role="option"
                  tabIndex={-1}
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
                </Pressable>
              ))}
            </div>
          ))}
        </div>
        <p className="plan-palette-foot">
          {pageText(locale, 'Arrows move, Enter runs, Esc closes', 'חצים לניווט, Enter להפעלה, Esc לסגירה')}
        </p>
      </dialog>
  );
}

export default CommandPalette;
