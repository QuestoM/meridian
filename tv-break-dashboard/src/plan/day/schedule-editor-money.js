import { useCallback, useEffect, useRef, useState } from 'react';
// The extensions are explicit for the same reason schedule-editor-pin.js gives
// them: the tests execute this module in node rather than through the bundler,
// and node resolves a relative import literally. A module a test cannot run is a
// module a test can only assert a copy of, which is how the save scope below it
// once shipped wrong.
import { fetchDay, saveEffect, scoreDay, undoBreakPlacement } from './day-board-actions.js';
import { savePinPlacement } from './schedule-editor-pin.js';
import { inverseOf, settlementOf } from './day-board-settlement.js';
import { predictionFor } from './day-board-forecast.js';

// The money the editor's own save moves, and the exact inverse of that save.
//
// The measurement that forced this module. A critic drove the shipped editor on
// רשת 13 / 2024-11-01: one chip dragged 02:29:00 to 02:22:00, then שמור כנעיצה,
// 605 ms. The day fell from 1,062,669.88 to 1,037,270.00, which is 25,399.88 ILS
// and 2.39 per cent of the day, and the surface that spent it carried no
// currency figure at all before or after, no preview of what the save would do,
// and no control that took it back. JS-3's own Done clause is that the money it
// cost or earned is on screen and that it can be undone, and neither half was
// true on the surface a scheduler reaches from their own door.
//
// Nothing new is computed here. All three answers are the day board's own
// seams, called from the second timeline rather than reimplemented beside it:
// ``scoreDay`` for what the arrangement on screen is worth, ``saveEffect`` for
// what the save would really do measured before anything is written, and two
// reads of the day either side of the write for what it actually did. Driven
// over HTTP on that same chip: the cheap score reads 0.00 because a break is
// priced on its programme and not on the minute it starts at, ``saveEffect``
// reads -25,399.88 in 439.3 ms of engine time, and the written save lands on
// 1,037,270.00, which is that figure to the cent.

// The pending edits as the score endpoint's own move list, plus what could not
// be turned into one.
//
// An editor row is a display key and the plan addresses a break as
// <segment_id>~<ordinal>, so a row that resolves no segment carries no move.
// It is returned by name rather than dropped, because a figure that silently
// left an edit out would be a wrong figure. The same applies to a second day:
// a score is one day's answer, so edits on another day are named and excluded
// rather than folded into a total that does not cover them.
export function pendingMoves(lanes, edits, stateOf, targetFor) {
  const rows = [];
  const days = [];
  (lanes || []).forEach((lane) => (lane.items || []).forEach((item) => {
    const date = item.date || '';
    if (date && !days.includes(date)) days.push(date);
    if (!edits[item.id]) return;
    const { startSec, durationSec } = stateOf(item);
    rows.push({ item, date, target: targetFor(item, startSec, durationSec) });
  }));
  const edited = rows.find((row) => row.date);
  const day = (edited && edited.date) || days[0] || '';
  const moves = [];
  const unaddressable = [];
  const otherDays = [];
  rows.forEach((row) => {
    if (row.date && row.date !== day) {
      if (!otherDays.includes(row.date)) otherDays.push(row.date);
      return;
    }
    if (!row.target) {
      unaddressable.push(row.item.program_title || row.item.id);
      return;
    }
    moves.push({
      break_id: row.target.item.break_id,
      offset_seconds: Math.round(row.target.live.offsetSeconds),
      duration_seconds: Math.round(row.target.live.durationSeconds),
      is_gold: null,
    });
  });
  return { day, moves, unaddressable, otherDays };
}

// The three answers, held for exactly the arrangement they were measured on.
export function useEditorMoney({ pending, locale, notify, onGlobalRefresh }) {
  const [score, setScore] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [checking, setChecking] = useState(false);
  const [settlement, setSettlement] = useState(null);
  const [lastSave, setLastSave] = useState(null);
  const [pinned, setPinned] = useState({});
  const [busy, setBusy] = useState(false);
  // Bumped by every write, so the score re-reads a day the write changed even
  // though no edit on screen moved.
  const [version, setVersion] = useState(0);
  const { day, moves } = pending;
  // The move list is a fresh array on every render, so an effect that depended
  // on it would re-run forever. The signature is what actually changed, and the
  // list itself is read through a ref at the moment the call is made.
  const signature = JSON.stringify(moves);
  const movesRef = useRef(moves);
  movesRef.current = moves;

  // Score whatever is on screen now. The endpoint answers in well under a
  // millisecond of engine time, so this runs on every settled edit rather than
  // behind a button, exactly as it does on the day board.
  useEffect(() => {
    if (!day) return undefined;
    let alive = true;
    const handle = window.setTimeout(() => {
      scoreDay(day, movesRef.current)
        .then((payload) => { if (alive) setScore(payload); })
        .catch(() => { if (alive) setScore(null); });
    }, 40);
    return () => { alive = false; window.clearTimeout(handle); };
  }, [day, signature, version]);

  // A forecast belongs to one arrangement. The moment any edit changes it the
  // figure describes a board nobody is looking at, so it goes.
  useEffect(() => { setForecast(null); }, [day, signature, version]);

  const check = useCallback(async () => {
    if (!day || !movesRef.current.length) return;
    setChecking(true);
    try {
      setForecast(await saveEffect(day, movesRef.current));
    } catch (error) {
      setForecast(null);
      notify(
        `The check of what saving would do failed (${error.message}).`,
        `הבדיקה של מה תעשה השמירה נכשלה (${error.message}).`,
      );
    } finally {
      setChecking(false);
    }
  }, [day, notify]);

  // Hold the day the plan had, perform the write, read the day it became, and
  // report the difference beside the prediction that was on screen. Both sides
  // are the engine's own totals from one route; nothing here is modelled.
  const settle = useCallback(async (act, targetDay, predicted, run) => {
    setBusy(true);
    try {
      const before = await fetchDay(targetDay);
      const result = await run();
      const after = await fetchDay(targetDay);
      setSettlement(settlementOf({
        act,
        basis: (after && after.basis) || (before && before.basis),
        before: before && before.totals,
        after: after && after.totals,
        beforeBreaks: (before && before.breaks) || [],
        afterBreaks: (after && after.breaks) || [],
        predictedRevenue: predicted,
      }));
      setVersion((current) => current + 1);
      if (onGlobalRefresh) onGlobalRefresh();
      return result;
    } finally {
      setBusy(false);
    }
  }, [onGlobalRefresh]);

  // The save this surface performs, settled. The record it returns carries both
  // ids the inverse needs, which is what makes the undo below exact.
  const saveAndSettle = useCallback(async (rowId, target) => {
    const targetDay = (target.programme && target.programme.day) || day;
    const record = await settle(
      'save',
      targetDay,
      predictionFor(forecast, score),
      () => savePinPlacement(target),
    );
    setLastSave({ rowId, breakId: record.breakId, constraintId: record.constraintId, day: targetDay });
    setPinned((current) => ({ ...current, [rowId]: record.constraintId }));
    return record;
  }, [day, forecast, score, settle]);

  // The exact inverse of that save, addressed by the two ids the record carries
  // rather than by anything this browser tab happens to remember about it.
  const undoLastSave = useCallback(async () => {
    if (!lastSave) return;
    try {
      await settle(
        'undo',
        lastSave.day,
        inverseOf(settlement),
        () => undoBreakPlacement({ breakId: lastSave.breakId, constraintId: lastSave.constraintId }),
      );
      setPinned((current) => {
        const next = { ...current };
        delete next[lastSave.rowId];
        return next;
      });
      setLastSave(null);
      notify(
        'The saved placement was removed, and the plan places this break itself again.',
        'הנעיצה השמורה הוסרה, והתוכנית חוזרת למקם את הברייק בעצמה.',
      );
    } catch (error) {
      notify(`The undo failed (${error.message}).`, `הביטול נכשל (${error.message}).`);
    }
  }, [lastSave, settlement, settle, notify]);

  const isPinned = useCallback((rowId) => Boolean(pinned[rowId]), [pinned]);

  return {
    score,
    forecast,
    checking,
    settlement,
    busy,
    locale,
    day,
    unaddressable: pending.unaddressable,
    otherDays: pending.otherDays,
    moveCount: moves.length,
    canUndo: Boolean(lastSave) && !busy,
    check,
    saveAndSettle,
    undoLastSave,
    isPinned,
    dismiss: () => setSettlement(null),
  };
}
