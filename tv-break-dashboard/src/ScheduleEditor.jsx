import React, { useMemo, useRef, useState } from 'react';
import { X } from 'lucide-react';
import ConstraintBuilder from './ConstraintBuilder';
import ScheduleEditorRow from './ScheduleEditorRow';
import ScheduleEditorBreak from './ScheduleEditorBreak';
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
  windowRange,
  programClassLabel,
} from './schedule-editor-format';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

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
  const [constraints, setConstraints] = useState([]);
  const [savingPin, setSavingPin] = useState(null);
  const trackRefs = useRef({});

  // Shared zoom (passed from the page so the timeline and editor keep one
  // scale). A stand-alone fallback keeps the editor usable in isolation.
  const localZoom = useScheduleZoom();
  const { pxPerMin, setZoom, zoomBy } = zoom || localZoom;

  // Owned-channel segment anchors, resolved through the shared hook so a click
  // on a programme band opens the inspector for its addressable segment.
  const { resolve } = useSegmentAnchors();
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
      if (!grouped.has(laneKey)) grouped.set(laneKey, { lane: laneKey, program, items: [] });
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
      savePin(item, 'date');
    }
  }

  function constraintIdFor(item) {
    return constraints.find((constraint) => constraint.break_id === item.id);
  }

  async function postConstraint(body) {
    const response = await fetch(`${API_BASE}/api/constraints`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (response.status === 404) {
      throw new Error('not-found');
    }
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  async function savePin(item, scopeType) {
    const { startSec, durationSec } = currentState(item);
    const offsetSeconds = Math.max(0, startSec - item.program_start_sec);
    const body = {
      scope_type: scopeType,
      scope_value: scopeType === 'programme' ? (item.program_key || item.program_title) : (item.date || ''),
      channel: item.channel || '',
      effect: 'FIX_OFFSET',
      offset_seconds: offsetSeconds,
      duration_seconds: durationSec,
      order_index: Number(item.break_num_in_program || 0),
    };
    setSavingPin(item.id);
    try {
      const saved = await postConstraint(body);
      const savedId = saved.constraint_id ?? saved.id;
      setConstraints((current) => [
        ...current.filter((constraint) => constraint.break_id !== item.id),
        { ...body, id: savedId || `pin-${item.id}`, break_id: item.id },
      ]);
      notify(
        `Break pinned at ${secondsToClock(startSec)}, ${humanOffset(offsetSeconds, 'en')} into ${item.program_title}.`,
        `הברייק נעוץ ב-${secondsToClock(startSec)}, ${humanOffset(offsetSeconds, 'he')} בתוך ${item.program_title}.`,
      );
      // A saved pin is a fingerprinted constraint change, so the freshness
      // banner and overview must re-read their verdict.
      onGlobalRefresh?.();
    } catch (error) {
      if (error.message === 'not-found') {
        notify(
          'The constraints API is not available yet. The pin was not saved.',
          'ממשק האילוצים עדיין לא זמין. הנעיצה לא נשמרה.',
        );
      } else {
        notify(
          `Pin failed (${error.message}).`,
          `הנעיצה נכשלה (${error.message}).`,
        );
      }
    } finally {
      setSavingPin(null);
    }
  }

  const [scopeChoice, setScopeChoice] = useState('date');

  if (!breaks.length) {
    return (
      <div className="schedule-editor-empty">
        <h3>{editorPageText(locale, 'No breaks to edit', 'אין ברייקים לעריכה')}</h3>
        <p>
          {editorPageText(
            locale,
            'Recompute the weekly schedule or upload a plan to populate the editor with draggable breaks.',
            'חשבו מחדש את הלוח השבועי או העלו תוכנית כדי לאכלס את העורך בברייקים הניתנים לגרירה.',
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
        scopeChoice={scopeChoice}
        onScopeChoice={setScopeChoice}
        recomputeState={recomputeState}
        onRecompute={onRecompute}
        pxPerMin={pxPerMin}
        onZoom={setZoom}
        onZoomStep={zoomBy}
      />

      <ScheduleTrackSurface axis={axis} pxPerMin={pxPerMin} onZoom={setZoom} locale={locale}>
        {({ width, minWidth, ticks }) => lanes.map((lane) => (
          <div className="timeline-row" key={lane.lane} style={{ minWidth }}>
            <div className="timeline-lane" dir={he ? 'rtl' : 'ltr'}>
              <strong>{lane.lane}</strong>
              <span>{lane.items.length} {editorPageText(locale, 'breaks', 'ברייקים')}</span>
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
                    pinned={Boolean(constraintIdFor(item))}
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

      <div className="schedule-editor-readout" dir={he ? 'rtl' : 'ltr'}>
        {Object.keys(edits).length === 0 ? (
          <p>{editorPageText(locale, 'Drag a break to set its offset, then save it as a pin.', 'גררו ברייק כדי לקבוע את ההיסט שלו, ואז שמרו אותו כנעיצה.')}</p>
        ) : (
          <ul className="schedule-editor-edit-list">
            {lanes.flatMap((lane) => lane.items.filter((item) => edits[item.id]).map((item) => {
              const { startSec, durationSec } = currentState(item);
              const offsetSeconds = Math.max(0, startSec - item.program_start_sec);
              const pinned = Boolean(constraintIdFor(item));
              return (
                <React.Fragment key={item.id}>
                  <ScheduleEditorRow
                    item={item}
                    startSec={startSec}
                    durationSec={durationSec}
                    offsetSeconds={offsetSeconds}
                    pinned={pinned}
                    saving={savingPin === item.id}
                    locale={locale}
                    onSave={() => savePin(item, scopeChoice)}
                  />
                  <li className="schedule-editor-discard-row" style={{ listStyle: 'none', display: 'flex', justifyContent: 'flex-end', margin: '2px 0 10px' }}>
                    <button
                      type="button"
                      onClick={() => discardEdit(item)}
                      disabled={savingPin === item.id}
                      aria-label={editorPageText(locale, `Discard the unsaved change to ${item.program_title}`, `ביטול השינוי שלא נשמר בתוכנית ${item.program_title}`)}
                      style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'transparent', border: 'none', cursor: 'pointer', color: 'inherit', opacity: 0.75, fontSize: 12, padding: '2px 4px' }}
                    >
                      <X size={12} aria-hidden="true" />
                      {editorPageText(locale, 'Discard change', 'ביטול השינוי')}
                    </button>
                  </li>
                </React.Fragment>
              );
            }))}
          </ul>
        )}
      </div>

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
