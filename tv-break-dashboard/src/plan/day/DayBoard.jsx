import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { pageText } from '../../shell/format';
import { ScheduleTrackSurface, ProgrammeBand, useScheduleZoom } from './schedule-track-view';
import { timeWindow, spanStyle } from './schedule-track';
import { createDragHandlers } from './day-board-drag';
import DayBoardChip from './DayBoardChip';
import DayBoardReadout, { HourStrip } from './DayBoardReadout';
import DayBoardSettlement, { failureText } from './DayBoardSettlement';
import DayBoardToolbar from './DayBoardToolbar';
import { inverseOf, settlementOf } from './day-board-settlement';
import {
  DEFAULT_SNAP,
  MIN_DURATION_SECONDS,
  applyEdit,
  boardView,
  firstBreakInHour,
  liveBreak,
  maxDurationFor,
  movesFrom,
  nudgeTarget,
  offsetBounds,
  programmeIndex,
  snapTo,
  startSecondsOf,
} from './day-board-model';
import { useBoardHistory } from './day-board-history';
import { fitTheDay, fitTheProgramme } from './day-board-zoom';
import { predictionFor, useSaveForecast } from './day-board-forecast';
import * as writes from './day-board-writes';
import { fetchDay, scoreDay } from './day-board-actions';
import { LIVE_PLAN, breakCountText, planBasisLabel } from './plan-basis';
import './day-board.css';
import './day-readout.css';

// One broadcast day, as the physical thing a scheduler works on.
//
// The board holds the programmes as bands on a true time axis and every break as
// an object that can be selected, dragged, resized, marked gold, saved and
// undone. Nothing here opens a dialog: every act is either a direct
// manipulation or a keystroke, and every act is reversible from the same
// surface it happened on, and nothing on it is a dead end: a band opens its
// programme's own record and an hour bar selects the break the plan puts in it.
//
// The money answer is live because the engine has a scoring seam that costs
// microseconds rather than a re-optimization that costs a second. Measured
// end to end on this surface, a drop is answered in tens of milliseconds against
// a five hundred millisecond bar.
function DayBoard({ day, locale, notify, onGlobalRefresh, zoom, onOpenBreak, onOpenProgramme, onDayLoaded }) {
  const he = locale === 'he';
  const [board, setBoard] = useState(null);
  const [loadState, setLoadState] = useState('idle');
  const [loadError, setLoadError] = useState('');
  const [edits, setEdits] = useState({});
  const [selected, setSelected] = useState(null);
  const [score, setScore] = useState(null);
  const [saving, setSaving] = useState(false);
  // What the last save or undo actually did, held apart from the score so the
  // re-read that follows a write cannot re-base it away.
  const [settlement, setSettlement] = useState(null);
  // The gold act's own record. The second plan is free to leave out the very
  // break the act named, so the inverse is held by id and not read off a chip.
  const [lastGold, setLastGold] = useState(null);
  const [snapGrid, setSnapGrid] = useState(DEFAULT_SNAP);
  const [snapMark, setSnapMark] = useState(null);
  const trackRef = useRef(null);
  const localZoom = useScheduleZoom(6);
  const { pxPerMin, floor, setZoom, zoomBy, fitTo } = zoom || localZoom;

  // The day-loaded callback is held in a ref rather than in the loader's
  // dependency list. A parent that passes an inline arrow hands this component a
  // new function identity on every render, and a loader that depended on it
  // would be rebuilt each time, re-run its effect, set state and render again.
  // Measured before this ref existed: the board refetched the day every 110 ms
  // for as long as the page was open. A callback prop is never a load dependency.
  const onDayLoadedRef = useRef(onDayLoaded);
  onDayLoadedRef.current = onDayLoaded;

  const load = useCallback(async (targetDay) => {
    if (!targetDay) return;
    setLoadState('loading');
    setLoadError('');
    try {
      const payload = await fetchDay(targetDay);
      setBoard(payload);
      // The old score described the arrangement that was on screen a moment ago.
      // Measured: after an undo it kept 1,037,270 on screen for about 50 ms while
      // the day had gone back to 1,062,670. A figure that describes an
      // arrangement nobody is looking at is a wrong figure, so it is dropped and
      // the freshly loaded board's own totals stand in until the new score lands.
      setScore(null);
      setLoadState('ready');
      if (onDayLoadedRef.current) onDayLoadedRef.current(payload);
      // Returned as well as stored, because a write has to compare the day it
      // got back against the day it had, and reading that off React state would
      // race the render that sets it.
      return payload;
    } catch (error) {
      setBoard(null);
      setLoadError(error.message);
      setLoadState('error');
      return null;
    }
  }, []);

  const programmes = useMemo(() => programmeIndex(board?.programmes), [board]);
  const breaks = board?.breaks || [];
  const { history, push: pushHistory, reset: resetHistory, forget: forgetAction, forgetRecord, undo, redo, lastSave } =
    useBoardHistory({ breaks, setEdits, setSelected });
  // What saving would really cost, measured on request and dropped the moment any
  // edit changes it. See day-board-forecast.js for why the cheap score is not it.
  const { forecast, checking, check: checkSaveEffect } = useSaveForecast({ board, edits, notify });

  useEffect(() => {
    setEdits({});
    resetHistory();
    setScore(null);
    setSettlement(null);
    setLastGold(null);
    load(day);
  }, [day, load, resetHistory]);

  const axis = useMemo(() => {
    const values = [];
    (board?.programmes || []).forEach((programme) => {
      values.push(programme.start_seconds / 60, programme.end_seconds / 60);
    });
    return timeWindow(values);
  }, [board]);

  // Score whatever is on screen now. The endpoint answers in under a
  // millisecond of engine time, so this runs on every settled edit rather than
  // behind a button.
  useEffect(() => {
    if (!board) return undefined;
    let alive = true;
    const handle = window.setTimeout(() => {
      scoreDay(board.day, movesFrom(edits))
        .then((payload) => { if (alive) setScore(payload); })
        .catch(() => { if (alive) setScore(null); });
    }, 40);
    return () => { alive = false; window.clearTimeout(handle); };
  }, [board, edits]);

  const liveOf = useCallback((item) => liveBreak(item, edits), [edits]);

  function positionStyle(startSec, endSec) {
    return spanStyle(axis, pxPerMin, startSec / 60, endSec / 60);
  }

  function commit(item, next, actionName) {
    const before = liveOf(item);
    setEdits((current) => applyEdit(current, item, next));
    pushHistory({ type: 'edit', name: actionName, breakId: item.break_id, before, after: next });
  }

  // The pointer gestures live in their own module under the 450-line law. Refs
  // carry the live zoom and snap so a gesture in flight always reads the values
  // on screen rather than the ones captured when it began.
  const pxPerMinRef = useRef(pxPerMin);
  const snapGridRef = useRef(snapGrid);
  pxPerMinRef.current = pxPerMin;
  snapGridRef.current = snapGrid;
  const { onMovePointerDown: handleMovePointerDown, onResizePointerDown: handleResizePointerDown } = useMemo(
    () => createDragHandlers({
      trackRef,
      axis,
      pxPerMinRef,
      snapGridRef,
      programmes,
      liveOf,
      setEdits,
      setSelected,
      setSnapMark,
      pushHistory,
    }),
    [axis, programmes, liveOf, pushHistory],
  );

  // The whole keyboard is one pure function in the model, so the board holds only
  // the decision to commit what it returns and the undo stack sees one named act.
  function nudge(item, event) {
    const target = nudgeTarget(event, programmes.get(item.segment_id), liveOf(item), snapGrid);
    if (!target) return false;
    commit(item, target.next, target.name);
    return true;
  }

  // Every act that writes lives in day-board-writes.js. Each wrapper below hands
  // that module the state it cannot see and nothing else, and every one of them
  // settles, this one included: see that file for what a gold change costs.
  function toggleGold(item) {
    const live = liveOf(item);
    return writes.applyGold({ item, live, notify, settleAfter, rememberGold: setLastGold });
  }

  function handleKeyDown(event, item) {
    if (nudge(item, event)) {
      event.preventDefault();
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      onOpenBreak(item.break_id);
    } else if (event.key === 'g' || event.key === 'G') {
      event.preventDefault();
      toggleGold(item);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      setSelected(null);
      event.currentTarget.blur();
    }
  }

  // Every write settles the same way: hold the day it had, read the day it got,
  // and report the difference beside the prediction that was on screen. Measured
  // on רשת 13 / 2024-11-01, the two are not the same number, and before this the
  // board re-based itself on the fresh plan and printed a change of zero.
  async function settleAfter(act, predictedRevenue, run) {
    const priorTotals = board.totals;
    const priorBreaks = breaks;
    setSaving(true);
    try {
      await run();
      const fresh = await load(board.day);
      setSettlement(settlementOf({
        act,
        basis: (fresh && fresh.basis) || board.basis,
        before: priorTotals,
        after: fresh && fresh.totals,
        beforeBreaks: priorBreaks,
        afterBreaks: (fresh && fresh.breaks) || [],
        predictedRevenue,
      }));
      if (onGlobalRefresh) onGlobalRefresh();
      return true;
    } catch (error) {
      notify(...failureText(act, error.message));
      return false;
    } finally {
      setSaving(false);
    }
  }

  function saveAll() {
    if (!board) return undefined;
    return writes.saveEditedBreaks({
      breaks,
      edits,
      programmes,
      liveOf,
      predicted: predictionFor(forecast, score),
      settleAfter,
      setEdits,
      pushHistory,
      notify,
    });
  }

  // The inverse of a save, offered on the break itself rather than on the
  // session. The prediction it settles against is the inverse of what the save
  // turned out to cost, and after a reload there is no measured save behind it,
  // so the panel says there was no prediction rather than inventing a zero.
  function removeSavedPlacement(item) {
    return writes.removeSavedPlacement({ item, predicted: inverseOf(settlement), settleAfter, forgetRecord, notify });
  }

  // The same inverse for a record whose break the re-plan took away, addressed by
  // the record's own two ids. Without it the money a save of that shape spent has
  // no route back from this surface: see day-board-writes.js for the measurement.
  function removeUnboundPlacement(record) {
    return writes.removeUnboundPlacement({ record, predicted: inverseOf(settlement), settleAfter, forgetRecord, notify });
  }

  function undoLastSave() {
    return writes.undoSave({ lastSave, predicted: inverseOf(settlement), settleAfter, forgetAction, notify });
  }

  // The inverse of the gold act, offered on the same panel that printed its cost.
  function undoLastGold() {
    return writes.undoGold({ lastGold, settleAfter, forgetGold: () => setLastGold(null), notify });
  }

  const editCount = Object.keys(edits).length;
  const goldSettled = Boolean(settlement) && settlement.act === 'gold';
  const view = boardView(score, board);
  const selectedItem = breaks.find((item) => item.break_id === selected) || null;
  const activeHour = selectedItem
    ? Math.floor(startSecondsOf(selectedItem, programmes.get(selectedItem.segment_id), liveOf(selectedItem)) / 3600)
    : null;

  if (loadState === 'error') {
    return (
      <div className="day-board-empty">
        <h3>{pageText(locale, 'This day could not be opened', 'לא ניתן לפתוח את היום הזה')}</h3>
        <p dir="auto">{loadError}</p>
      </div>
    );
  }
  if (!board) {
    return (
      <div className="day-board-empty">
        <p>{pageText(locale, 'Opening the day', 'פותח את היום')}</p>
      </div>
    );
  }
  // The route answers a missing channel or a missing plan as a state rather than
  // a fault, so the board says what is missing instead of drawing an empty grid
  // that would read as a finished day with nothing in it.
  if (board.available === false) {
    return (
      <div className="day-board-empty">
        <h3>{pageText(locale, 'There is no day to open yet', 'אין עדיין יום לפתיחה')}</h3>
        <p dir="auto">{(he && board.reason_he) || board.reason}</p>
      </div>
    );
  }

  return (
    <div className="day-board" dir={he ? 'rtl' : 'ltr'}>
      <DayBoardToolbar
        board={board}
        locale={locale}
        snapGrid={snapGrid}
        onSnapGrid={setSnapGrid}
        pxPerMin={pxPerMin}
        onZoom={setZoom}
        onZoomStep={zoomBy}
        zoomFloor={floor}
        onFitDay={() => fitTheDay(axis, trackRef, fitTo)}
        onFitProgramme={() => selectedItem && fitTheProgramme(programmes.get(selectedItem.segment_id), trackRef, fitTo, selectedItem.break_id)}
        selectedItem={selectedItem}
        live={selectedItem ? liveOf(selectedItem) : null}
        programme={selectedItem ? programmes.get(selectedItem.segment_id) : null}
        busy={saving}
        onRemoveSaved={() => selectedItem && removeSavedPlacement(selectedItem)}
        onLength={(seconds) => {
          if (!selectedItem) return;
          const programme = programmes.get(selectedItem.segment_id);
          const live = liveOf(selectedItem);
          const durationSeconds = snapTo(seconds, 1, MIN_DURATION_SECONDS, maxDurationFor(programme, live.offsetSeconds));
          commit(selectedItem, { ...live, durationSeconds }, 'length');
        }}
        onStart={(clockSeconds) => {
          if (!selectedItem) return;
          const programme = programmes.get(selectedItem.segment_id);
          if (!programme) return;
          const live = liveOf(selectedItem);
          const bounds = offsetBounds(programme, live.durationSeconds);
          // A typed clock is an exact request, so it is clamped to the programme
          // and never snapped: a person who types 21:42:30 means 21:42:30.
          const offsetSeconds = snapTo(clockSeconds - programme.start_seconds, 1, bounds.min, bounds.max);
          commit(selectedItem, { ...live, offsetSeconds }, 'move');
        }}
        onGold={() => selectedItem && toggleGold(selectedItem)}
        onOpen={() => selectedItem && onOpenBreak(selectedItem.break_id)}
      />

      <ScheduleTrackSurface axis={axis} pxPerMin={pxPerMin} onZoom={setZoom} locale={locale} floor={floor}>
        {({ width, minWidth, ticks }) => (
          <div className="day-track-row" style={{ minWidth }}>
            <div className="day-track-lane" dir={he ? 'rtl' : 'ltr'}>
              <strong dir="auto">{board.operator_channel}</strong>
              <span dir="ltr">{board.day}</span>
              <span className="timeline-lane-count"><span dir="auto">{breakCountText(breaks.length, locale)}</span><small className="timeline-lane-basis" dir="auto">{planBasisLabel(LIVE_PLAN, locale)}</small></span>
            </div>
            <div className="day-track" style={{ width }} ref={trackRef}>
              {ticks.filter((tick) => tick.major).map((tick) => (
                <i className="day-track-tick" key={tick.minute} style={{ left: `${tick.left}px` }} />
              ))}
              {(board.programmes || []).map((programme) => (
                <ProgrammeBand
                  key={programme.segment_id}
                  title={programme.title}
                  classLabel={programme.genre}
                  windowText={`${Math.round(programme.duration_seconds / 60)}m`}
                  style={positionStyle(programme.start_seconds, programme.end_seconds)}
                  clickable={Boolean(onOpenProgramme)}
                  onOpen={() => onOpenProgramme(programme)}
                />
              ))}
              {snapMark !== null && (
                <i className="day-snap-line" style={{ left: spanStyle(axis, pxPerMin, snapMark / 60, snapMark / 60).left }} />
              )}
              {breaks.map((item) => {
                const programme = programmes.get(item.segment_id);
                const live = liveOf(item);
                const startSeconds = startSecondsOf(item, programme, live);
                // The width the chip is really drawn at, so it can decide which
                // of its own numbers it is able to print in full.
                const geometry = positionStyle(startSeconds, startSeconds + live.durationSeconds);
                return (
                  <DayBoardChip
                    key={item.break_id}
                    item={item}
                    live={live}
                    startSeconds={startSeconds}
                    selected={selected === item.break_id}
                    edited={live.edited}
                    saved={Boolean(item.saved_placement)}
                    locale={locale}
                    style={geometry}
                    widthPx={parseFloat(geometry.width)}
                    onSelect={setSelected}
                    onMovePointerDown={handleMovePointerDown}
                    onResizePointerDown={handleResizePointerDown}
                    onKeyDown={handleKeyDown}
                    onOpen={onOpenBreak}
                  />
                );
              })}
            </div>
          </div>
        )}
      </ScheduleTrackSurface>

      <HourStrip
        hours={view ? view.hours : board.hours}
        locale={locale}
        activeHour={activeHour}
        onOpenHour={(hour) => setSelected((current) => firstBreakInHour(breaks, programmes, liveOf, hour) || current)}
      />

      <DayBoardReadout
        score={view}
        locale={locale}
        editCount={editCount}
        canUndo={history.past.length > 0}
        saving={saving}
        unbound={board.unbound_placements}
        onRemoveUnbound={removeUnboundPlacement}
        forecast={forecast}
        checking={checking}
        onCheck={checkSaveEffect}
        onUndo={() => (editCount === 0 && lastSave ? undoLastSave() : undo())}
        onDiscard={() => { setEdits({}); resetHistory(); }}
        onSave={saveAll}
      />

      <DayBoardSettlement
        settlement={settlement}
        locale={locale}
        canUndo={(goldSettled ? Boolean(lastGold) : Boolean(lastSave)) && !saving}
        onUndo={goldSettled ? undoLastGold : undoLastSave}
        onDismiss={() => setSettlement(null)}
      />
    </div>
  );
}

export default DayBoard;
