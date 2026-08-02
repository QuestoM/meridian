import { useCallback, useEffect, useState } from 'react';
import { applyEdit, clearEdit, emptyHistory, pushAction, redoAction, undoAction } from './day-board-model';

// Undo on the day board, as one stack read in two directions.
//
// Split out of DayBoard.jsx under the 450-line law, and it belongs in its own
// module anyway: every act on the board is a named action carrying the state it
// changed from, so reversing one never asks the server what the board used to
// look like. That is what makes undo a keystroke instead of a page.
//
// Cmd Z steps back and Cmd Shift Z steps forward, which is the reversal people
// already have in their fingers from every drawing tool. A new act after an undo
// drops the redone tail rather than branching, for the same reason.
export function useBoardHistory({ breaks, setEdits, setSelected }) {
  const [history, setHistory] = useState(emptyHistory);

  const applyHistory = useCallback((action, direction) => {
    if (!action || action.type !== 'edit') return;
    const item = breaks.find((row) => row.break_id === action.breakId);
    if (!item) return;
    const target = direction === 'undo' ? action.before : action.after;
    setEdits((current) => (target.edited === false && direction === 'undo'
      ? clearEdit(current, action.breakId)
      : applyEdit(current, item, target)));
    setSelected(action.breakId);
    // Undo puts the keyboard back on the break it just moved, unless the person
    // is typing, in which case stealing focus mid-word would be worse than not
    // following the selection at all.
    window.requestAnimationFrame(() => {
      const active = document.activeElement;
      if (active && active.tagName === 'INPUT') return;
      const chip = document.querySelector(`[data-break-id="${CSS.escape(action.breakId)}"]`);
      if (chip) chip.focus();
    });
  }, [breaks, setEdits, setSelected]);

  const undo = useCallback(() => {
    setHistory((current) => {
      const { history: next, action } = undoAction(current);
      if (action) applyHistory(action, 'undo');
      return next;
    });
  }, [applyHistory]);

  const redo = useCallback(() => {
    setHistory((current) => {
      const { history: next, action } = redoAction(current);
      if (action) applyHistory(action, 'redo');
      return next;
    });
  }, [applyHistory]);

  useEffect(() => {
    function onKey(event) {
      const meta = event.metaKey || event.ctrlKey;
      if (meta && (event.key === 'z' || event.key === 'Z')) {
        event.preventDefault();
        if (event.shiftKey) redo(); else undo();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [undo, redo]);

  const push = useCallback((action) => setHistory((current) => pushAction(current, action)), []);
  const reset = useCallback(() => setHistory(emptyHistory()), []);
  const forget = useCallback((action) => {
    setHistory((current) => ({ past: current.past.filter((entry) => entry !== action), future: current.future }));
  }, []);

  // Drop one break from every save this session still remembers.
  //
  // A saved placement can also be removed from the break itself, and that route
  // works on a break this tab never saved. When it happens to be a break this tab
  // did save, the remembered save must stop offering to reverse something that is
  // already reversed, while keeping the other breaks it saved reversible. A save
  // left holding nothing is dropped outright.
  const forgetRecord = useCallback((breakId) => {
    setHistory((current) => ({
      past: current.past
        .map((entry) => (entry.type === 'save'
          ? { ...entry, records: (entry.records || []).filter((record) => record.breakId !== breakId) }
          : entry))
        .filter((entry) => entry.type !== 'save' || entry.records.length > 0),
      future: current.future,
    }));
  }, []);
  const lastSave = [...history.past].reverse().find((action) => action.type === 'save') || null;

  return { history, push, reset, forget, forgetRecord, undo, redo, lastSave };
}
