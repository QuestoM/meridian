import React, { useMemo, useRef, useState } from 'react';
import ConstraintBuilder from '../../rules/ConstraintBuilder';
import ScheduleEditorBreak from './ScheduleEditorBreak';
import ScheduleEditorReadout from './ScheduleEditorReadout';
import ScheduleEditorScope, { LaneCount } from './ScheduleEditorScope';
import ScheduleEditorToolbar from './ScheduleEditorToolbar';
import ScheduleInspector from './ScheduleInspector';
import {
  ScheduleTrackSurface,
  ProgrammeBand,
  useScheduleZoom,
  useSegmentAnchors,
} from './schedule-track-view';
import {
  timeWindow,
  spanStyle,
  pixelToMinute,
} from './schedule-track';
import {
  secondsToClock,
  humanOffset,
  laneLabel,
  windowRange,
  programClassLabel,
} from './schedule-editor-format';
import { pinTarget } from './schedule-editor-pin';
import { pendingMoves, useEditorMoney } from './schedule-editor-money';
import { useEditorCoverage } from './schedule-editor-scope';
import { scopeSentence } from './day-board-actions';

// Local helpers kept self-contained so the editor can live in its own module
// without exporting internals from TVBreakDashboard.jsx. They mirror the time
// math used by TimelineView (timeToMinutes / minutesToTime).
function editorPageText(locale, en, he) {
  return locale === 'he' ? he : en;
}

function timeToSeconds(time) {
  const [hour, minute] = String(time || '00:00').split(':').map((part) => Number(part));
  const safeHour = Number.isFinite(hour) ? Math.max(0, Math.min(47, hour)) : 0;
  const safeMinute = Number.isFinite(minute) ? Math.max(0, Math.min(59, minute)) : 0;
  return (safeHour * 60 + safeMinute) * 60;
}

function normalizeRows(value) {
  if (Array.isArray(value)) return value;
  if (value && Array.isArray(value.rows)) return value.rows;
  return [];
}

// Snap a second value to the nearest grid multiple, clamped to a [min, max] range.
function snapSeconds(value, grid, min, max) {
  const snapped = Math.round(value / grid) * grid;
  return Math.max(min, Math.min(max, snapped));
}

// ScheduleEditor forks TimelineView: it keeps the true time-axis layout but each
// break becomes a draggable / resizable handle. Drag is constrained to the
// horizontal axis, snapped to a configurable grid, and the new offset from the
// programme start is computed by inverting the same percent math TimelineView uses.
function ScheduleEditor({ schedule, locale, notify, onRecompute, recomputeState, onGlobalRefresh, zoom }) {
  const breaks = useMemo(() => normalizeRows(schedule?.break_operations?.breaks), [schedule]);
  const programs = useMemo(() => normalizeRows(schedule?.break_operations?.programs), [schedule]);
  const he = locale === 'he';

  const [snapGrid, setSnapGrid] = useState(60);
  const [edits, setEdits] = useState({});
  const [savingPin, setSavingPin] = useState(null);
  const trackRefs = useRef({});

  // Shared zoom (passed from the page so the timeline and editor keep one
  // scale). A stand-alone fallback keeps the editor usable in isolation.
  const localZoom = useScheduleZoom();
  const { pxPerMin, setZoom, zoomBy } = zoom || localZoom;

  // Owned-channel segment anchors, resolved through the shared hook so a click
  // on a programme band opens the inspector for its addressable segment.
  const { resolve, loaded: anchorsLoaded } = useSegmentAnchors();
  const [inspect, setInspect] = useState(null); // {segmentId, channel, day} | null

  const openInspector = (program) => {
    if (!program) return;
    const hit = resolve(program.channel, program.date, program.start_time);
    if (hit) {
      setInspect(hit);
    } else {
      notify(
        'This programme is not on your owned channel, so it has no editable segment.',
        'התוכנית אינה בערוץ שבבעלותכם, ולכן אין לה מקטע לעריכה.',
      );
    }
  };

  // Build the per-lane model from the break list, attaching the matching
  // programme band so we can show a REAL programme name and start time.
  const lanes = useMemo(() => {
    const byProgram = new Map();
    programs.forEach((program) => {
      if (program && program.key) byProgram.set(program.key, program);
    });
    const grouped = new Map();
    breaks.forEach((breakItem, index) => {
      const program = byProgram.get(breakItem.program_key) || null;
      const laneKey = breakItem.lane || `${breakItem.channel || ''} / ${breakItem.day || ''}`;
      const id = breakItem.id || `break-${index}`;
      const programStartSec = program ? timeToSeconds(program.start_time) : timeToSeconds(breakItem.start_time);
      const programEndSec = program ? timeToSeconds(program.end_time) : timeToSeconds(breakItem.end_time);
      const entry = {
        ...breakItem,
        id,
        program,
        program_title: breakItem.program_title || (program && program.title) || breakItem.program_key || editorPageText(locale, 'Untitled programme', 'תוכנית ללא שם'),
        program_start_sec: programStartSec,
        program_end_sec: programEndSec,
        date: breakItem.date || (program && program.date) || '',
      };
      if (!grouped.has(laneKey)) {
        grouped.set(laneKey, {
          lane: laneKey,
          label: laneLabel(entry.channel, entry.date, laneKey, locale),
          program,
          items: [],
        });
      }
      grouped.get(laneKey).items.push(entry);
    });
    return Array.from(grouped.values());
  }, [breaks, programs, locale]);

  const allMinutes = useMemo(() => {
    const values = [];
    lanes.forEach((lane) => {
      lane.items.forEach((item) => {
        values.push(timeToSeconds(item.start_time) / 60, timeToSeconds(item.end_time) / 60);
        values.push(item.program_start_sec / 60, item.program_end_sec / 60);
      });
    });
    return values.filter((value) => Number.isFinite(value));
  }, [lanes]);

  // Shared time axis, so the editor and the read-only timeline resolve the same
  // hour window and the same pixel mapping at any zoom.
  const axis = useMemo(() => timeWindow(allMinutes), [allMinutes]);

  // positionStyle: the shared pixel mapping, bridged from seconds to minutes.
  function positionStyle(startSec, endSec) {
    return spanStyle(axis, pxPerMin, startSec / 60, endSec / 60);
  }

  // Invert the mapping: a client x within a track maps back to an absolute
  // second-of-day, which we snap and clamp to the programme window. The track's
  // own scroll offset is folded in so the drop lands where the pointer is.
  function pixelToStartSec(laneKey, clientX, durationSec, item) {
    const track = trackRefs.current[laneKey];
    if (!track) return timeToSeconds(item.start_time);
    const rect = track.getBoundingClientRect();
    const absoluteSec = pixelToMinute(axis, pxPerMin, clientX - rect.left) * 60;
    const minStart = item.program_start_sec;
    const maxStart = Math.max(minStart, item.program_end_sec - durationSec);
    return snapSeconds(absoluteSec, snapGrid, minStart, maxStart);
  }

  function currentState(item) {
    const edit = edits[item.id];
    const startSec = edit ? edit.start_sec : timeToSeconds(item.start_time);
    const durationSec = edit ? edit.duration_sec : Number(item.duration_sec || (timeToSeconds(item.end_time) - timeToSeconds(item.start_time)) || 120);
    return { startSec, durationSec };
  }

  function applyEdit(item, startSec, durationSec) {
    setEdits((current) => ({
      ...current,
      [item.id]: { start_sec: startSec, duration_sec: durationSec },
    }));
  }

  // Discard one dragged row: dropping edits[item.id] snaps the break back to
  // its saved position immediately. Nothing was persisted, so nothing else moves.
  function discardEdit(item) {
    setEdits((current) => {
      const next = { ...current };
      delete next[item.id];
      return next;
    });
  }

  function handleMovePointerDown(event, laneKey, item) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const { durationSec } = currentState(item);
    // Click-vs-drag discriminator: a pointer that barely moves is a CLICK that
    // opens the inspector for this break's programme; a real drag re-places the
    // break as before. The threshold keeps a small hand-tremor from stealing a
    // drag or a click.
    const downX = event.clientX;
    const downY = event.clientY;
    let moved = false;
    function onMove(moveEvent) {
      if (!moved && Math.hypot(moveEvent.clientX - downX, moveEvent.clientY - downY) > 5) {
        moved = true;
      }
      if (!moved) return;
      const startSec = pixelToStartSec(laneKey, moveEvent.clientX, durationSec, item);
      applyEdit(item, startSec, durationSec);
    }
    function onUp(upEvent) {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      try {
        upEvent.target.releasePointerCapture && upEvent.target.releasePointerCapture(event.pointerId);
      } catch (releaseError) {
        // Pointer capture may already be released; ignore.
      }
      if (!moved) openInspector(item.program);
    }
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }

  function handleResizePointerDown(event, laneKey, item) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const { startSec } = currentState(item);
    function onMove(moveEvent) {
      const track = trackRefs.current[laneKey];
      if (!track) return;
      const rect = track.getBoundingClientRect();
      const absoluteSec = pixelToMinute(axis, pxPerMin, moveEvent.clientX - rect.left) * 60;
      const maxEnd = item.program_end_sec;
      const rawDuration = Math.max(30, absoluteSec - startSec);
      const durationSec = snapSeconds(rawDuration, 30, 30, Math.max(30, maxEnd - startSec));
      applyEdit(item, startSec, durationSec);
    }
    function onUp() {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    }
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }

  // Keyboard accessibility: arrows nudge by one snap unit, Enter saves a pin.
  function handleKeyDown(event, laneKey, item) {
    const { startSec, durationSec } = currentState(item);
    const minStart = item.program_start_sec;
    const maxStart = Math.max(minStart, item.program_end_sec - durationSec);
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      applyEdit(item, snapSeconds(startSec - snapGrid, snapGrid, minStart, maxStart), durationSec);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      applyEdit(item, snapSeconds(startSec + snapGrid, snapGrid, minStart, maxStart), durationSec);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      const maxDuration = Math.max(30, item.program_end_sec - startSec);
      applyEdit(item, startSec, snapSeconds(durationSec + 30, 30, 30, maxDuration));
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      applyEdit(item, startSec, snapSeconds(durationSec - 30, 30, 30, Math.max(30, item.program_end_sec - startSec)));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      savePin(item);
    }
  }

  // The addressable break behind an editor row, or null when there is none.
  //
  // The editor's own row id is a display key built from the programme key, and
  // the plan addresses a break as <segment_id>~<ordinal>. The anchors this page
  // already loads for the programme inspector are what bridge the two, so a save
  // resolves the segment first and refuses when it cannot.
  function pinTargetFor(item, startSec, durationSec) {
    const anchor = item.program
      ? resolve(item.program.channel, item.program.date, item.program.start_time)
      : null;
    return anchor ? pinTarget(item, startSec, durationSec, anchor.segmentId) : null;
  }

  // What a save would bind, in words, on the row that carries the Save button.
  // No count is passed: this surface draws twelve programmes of a day and the
  // predicate binds every airing of a title inside an hour, so a count taken from
  // what is on screen could under-report. The sentence states the rule instead.
  function scopeFor(item, startSec, durationSec) {
    const target = pinTargetFor(item, startSec, durationSec);
    return target ? scopeSentence(target.programme, locale) : '';
  }

  // What this day is worth, what a save would do to it, and what one did. Driven
  // from the day board's own seams so the two timelines cannot diverge: see
  // schedule-editor-money.js for the 25,399.88 ILS this surface used to spend
  // without printing a figure.
  const pending = useMemo(
    () => pendingMoves(lanes, edits, currentState, pinTargetFor),
    [lanes, edits, resolve],
  );
  const money = useEditorMoney({ pending, locale, notify, onGlobalRefresh });

  // What this timeline draws against what the day and the plan actually hold.
  // See schedule-editor-scope.js for the measurement this exists to close.
  const coverage = useEditorCoverage({ breaksShown: breaks.length, programs, score: money.score, resolve, anchorsLoaded });

  async function savePin(item) {
    const { startSec, durationSec } = currentState(item);
    const offsetSeconds = Math.max(0, startSec - item.program_start_sec);
    const target = pinTargetFor(item, startSec, durationSec);
    if (!target) {
      notify(
        'This break has no addressable segment in the saved plan, so a placement cannot be recorded for it and nothing was saved.',
        'לברייק הזה אין מקטע מזוהה בתוכנית השמורה, ולכן לא ניתן לרשום לו נעיצה ודבר לא נשמר.',
      );
      return;
    }
    setSavingPin(item.id);
    try {
      await money.saveAndSettle(item.id, target);
      notify(
        `Break pinned at ${secondsToClock(startSec)}, ${humanOffset(offsetSeconds, 'en')} into ${item.program_title}. ${scopeSentence(target.programme, 'en')}.`,
        `הברייק נעוץ ב-${secondsToClock(startSec)}, ${humanOffset(offsetSeconds, 'he')} בתוך ${item.program_title}. ${scopeSentence(target.programme, 'he')}.`,
      );
      // A saved pin is a fingerprinted constraint change, so the freshness
      // banner and overview must re-read their verdict.
      onGlobalRefresh?.();
    } catch (error) {
      notify(
        `The pin was not saved (${error.message}).`,
        `הנעיצה לא נשמרה (${error.message}).`,
      );
    } finally {
      setSavingPin(null);
    }
  }

  if (!breaks.length) {
    return (
      <div className="schedule-editor-empty">
        <h3>{editorPageText(locale, 'No breaks to edit', 'אין ברייקים לעריכה')}</h3>
        <p>
          {editorPageText(
            locale,
            'Run the weekly plan or upload one to populate the editor with draggable breaks.',
            'הריצו את הלוח השבועי או העלו תוכנית כדי לאכלס את העורך בברייקים הניתנים לגרירה.',
          )}
        </p>
      </div>
    );
  }

  return (
    <div className="schedule-editor">
      <ScheduleEditorToolbar
        locale={locale}
        snapGrid={snapGrid}
        onSnapGrid={setSnapGrid}
        recomputeState={recomputeState}
        onRecompute={onRecompute}
        pxPerMin={pxPerMin}
        onZoom={setZoom}
        onZoomStep={zoomBy}
      />

      <ScheduleEditorScope coverage={coverage} locale={locale} />

      <ScheduleTrackSurface axis={axis} pxPerMin={pxPerMin} onZoom={setZoom} locale={locale}>
        {({ width, minWidth, ticks }) => lanes.map((lane) => (
          <div className="timeline-row" key={lane.lane} style={{ minWidth }}>
            <div className="timeline-lane" dir={he ? 'rtl' : 'ltr'}>
              <strong>{lane.label || lane.lane}</strong>
              <LaneCount shown={lane.items.length} planned={coverage.plannedByLane[lane.lane]} locale={locale} />
            </div>
            <div
              className="timeline-track"
              style={{ width }}
              ref={(node) => {
                trackRefs.current[lane.lane] = node;
              }}
            >
              {ticks.filter((tick) => tick.major).map((tick) => (
                <i key={`${lane.lane}-${tick.minute}`} style={{ left: `${tick.left}px` }} />
              ))}
              {lane.program && (
                <ProgrammeBand
                  title={lane.program.title}
                  classLabel={programClassLabel(lane.program.program_type, locale)}
                  windowText={windowRange(lane.program.start_time, lane.program.end_time)}
                  style={positionStyle(timeToSeconds(lane.program.start_time), timeToSeconds(lane.program.end_time))}
                  clickable
                  onOpen={() => openInspector(lane.program)}
                />
              )}
              {lane.items.map((item) => {
                const { startSec, durationSec } = currentState(item);
                const offsetSeconds = Math.max(0, startSec - item.program_start_sec);
                return (
                  <ScheduleEditorBreak
                    key={item.id}
                    item={item}
                    laneKey={lane.lane}
                    startSec={startSec}
                    durationSec={durationSec}
                    offsetSeconds={offsetSeconds}
                    edited={Boolean(edits[item.id])}
                    pinned={money.isPinned(item.id)}
                    locale={locale}
                    positionStyle={positionStyle}
                    onMovePointerDown={handleMovePointerDown}
                    onResizePointerDown={handleResizePointerDown}
                    onKeyDown={handleKeyDown}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </ScheduleTrackSurface>

      <ScheduleEditorReadout
        lanes={lanes}
        edits={edits}
        savingPin={savingPin}
        locale={locale}
        stateOf={currentState}
        pinnedFor={(item) => money.isPinned(item.id)}
        scopeFor={scopeFor}
        onSave={savePin}
        onDiscard={discardEdit}
        money={money}
      />

      {inspect && (
        <ScheduleInspector
          segmentId={inspect.segmentId}
          channel={inspect.channel}
          day={inspect.day}
          locale={locale}
          notify={notify}
          onClose={() => setInspect(null)}
          onGlobalRefresh={onGlobalRefresh}
        />
      )}
    </div>
  );
}

// ConstraintBuilder lives in src/ConstraintBuilder.jsx (imported above); re-exported here
// so TVBreakDashboard.jsx keeps its existing named import unchanged.

export default ScheduleEditor;
export { ConstraintBuilder };
