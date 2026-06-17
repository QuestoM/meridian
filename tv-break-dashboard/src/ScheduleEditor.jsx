import React, { useMemo, useRef, useState } from 'react';
import { Button, MenuItem, Select, TextField } from '@mui/material';
import { Lock, Save, Send, Trash2 } from 'lucide-react';

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

function secondsToTime(seconds) {
  const safe = Math.max(0, Math.min((47 * 60 + 59) * 60 + 59, Math.round(seconds)));
  const hour = Math.floor(safe / 3600) % 24;
  const minute = Math.floor((safe % 3600) / 60);
  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
}

function formatOffset(seconds) {
  const safe = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(safe / 60);
  const remainder = safe % 60;
  return `+${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
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
function ScheduleEditor({ schedule, locale, notify, onRecompute, recomputeState }) {
  const breaks = useMemo(() => normalizeRows(schedule?.break_operations?.breaks), [schedule]);
  const programs = useMemo(() => normalizeRows(schedule?.break_operations?.programs), [schedule]);
  const he = locale === 'he';

  const [snapGrid, setSnapGrid] = useState(60);
  const [edits, setEdits] = useState({});
  const [constraints, setConstraints] = useState([]);
  const [savingPin, setSavingPin] = useState(null);
  const trackRefs = useRef({});

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

  const allTimes = useMemo(() => {
    const values = [];
    lanes.forEach((lane) => {
      lane.items.forEach((item) => {
        values.push(timeToSeconds(item.start_time) / 60, timeToSeconds(item.end_time) / 60);
        values.push(item.program_start_sec / 60, item.program_end_sec / 60);
      });
    });
    return values.filter((value) => Number.isFinite(value));
  }, [lanes]);

  const startHour = allTimes.length ? Math.max(0, Math.floor((Math.min(...allTimes) - 30) / 60)) : 18;
  const endHour = allTimes.length ? Math.min(28, Math.max(startHour + 4, Math.ceil((Math.max(...allTimes) + 30) / 60))) : 24;
  const totalMinutes = Math.max(60, (endHour - startHour) * 60);
  const hours = Array.from({ length: endHour - startHour + 1 }, (_, index) => startHour + index);
  const minWidth = 200 + Math.max(680, totalMinutes * 3.8);

  // positionStyle: same mapping as TimelineView, but in seconds.
  function positionStyle(startSec, endSec) {
    const startMin = startSec / 60;
    const endMin = Math.max(startMin + 0.25, endSec / 60);
    const left = ((startMin - startHour * 60) / totalMinutes) * 100;
    const width = ((endMin - startMin) / totalMinutes) * 100;
    return {
      left: `${Math.max(0, Math.min(99, left))}%`,
      width: `${Math.max(0.8, Math.min(100 - Math.max(0, left), width))}%`,
    };
  }

  // Invert positionStyle: a pixel x within a track maps back to an absolute
  // second-of-day, which we snap and clamp to the programme window.
  function pixelToStartSec(laneKey, clientX, durationSec, item) {
    const track = trackRefs.current[laneKey];
    if (!track) return timeToSeconds(item.start_time);
    const rect = track.getBoundingClientRect();
    const ratio = rect.width ? (clientX - rect.left) / rect.width : 0;
    const absoluteMin = startHour * 60 + ratio * totalMinutes;
    const absoluteSec = absoluteMin * 60;
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

  function handleMovePointerDown(event, laneKey, item) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const { durationSec } = currentState(item);
    function onMove(moveEvent) {
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
      const ratio = rect.width ? (moveEvent.clientX - rect.left) / rect.width : 0;
      const absoluteSec = (startHour * 60 + ratio * totalMinutes) * 60;
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
      setConstraints((current) => [
        ...current.filter((constraint) => constraint.break_id !== item.id),
        { ...body, id: saved.id || `pin-${item.id}`, break_id: item.id },
      ]);
      notify(
        `Break pinned at ${formatOffset(offsetSeconds)} from ${item.program_title}.`,
        `הברייק נעוץ ב-${formatOffset(offsetSeconds)} מתחילת ${item.program_title}.`,
      );
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
            'חשב מחדש את הלוח השבועי או העלה תוכנית כדי לאכלס את העורך בברייקים הניתנים לגרירה.',
          )}
        </p>
      </div>
    );
  }

  return (
    <div className="schedule-editor">
      <div className="schedule-editor-toolbar" dir={he ? 'rtl' : 'ltr'}>
        <div className="schedule-editor-snap" role="group" aria-label={editorPageText(locale, 'Snap grid', 'רשת הצמדה')}>
          <span>{editorPageText(locale, 'Snap', 'הצמדה')}</span>
          <Button
            type="button"
            variant="outlined"
            className={snapGrid === 30 ? 'segmented active' : 'segmented'}
            aria-pressed={snapGrid === 30}
            onClick={() => setSnapGrid(30)}
          >
            30s
          </Button>
          <Button
            type="button"
            variant="outlined"
            className={snapGrid === 60 ? 'segmented active' : 'segmented'}
            aria-pressed={snapGrid === 60}
            onClick={() => setSnapGrid(60)}
          >
            60s
          </Button>
        </div>
        <div className="schedule-editor-scope">
          <span>{editorPageText(locale, 'Pin scope', 'היקף הנעיצה')}</span>
          <Select
            size="small"
            value={scopeChoice}
            onChange={(event) => setScopeChoice(event.target.value)}
            aria-label={editorPageText(locale, 'Pin scope', 'היקף הנעיצה')}
          >
            <MenuItem value="date">{editorPageText(locale, 'This date', 'תאריך זה')}</MenuItem>
            <MenuItem value="programme">{editorPageText(locale, 'Every airing of this programme', 'כל שידור של התוכנית')}</MenuItem>
          </Select>
        </div>
        <Button
          type="button"
          variant="outlined"
          className="run-button"
          disabled={recomputeState === 'running'}
          onClick={() => onRecompute && onRecompute()}
        >
          <Send size={14} />
          {recomputeState === 'running'
            ? editorPageText(locale, 'Recomputing...', 'מחשב מחדש...')
            : editorPageText(locale, 'Recompute weekly schedule', 'חשב מחדש את הלוח השבועי')}
        </Button>
      </div>

      <div className="timeline-scroll chart-ltr" dir="ltr">
        <div className="timeline-ruler" style={{ minWidth }}>
          <span />
          <div className="timeline-hours">
            {hours.map((hour) => (
              <span key={hour} style={{ left: `${((hour - startHour) / Math.max(1, endHour - startHour)) * 100}%` }}>
                {String(hour % 24).padStart(2, '0')}:00
              </span>
            ))}
          </div>
        </div>
        {lanes.map((lane) => (
          <div className="timeline-row" key={lane.lane} style={{ minWidth }}>
            <div className="timeline-lane" dir={he ? 'rtl' : 'ltr'}>
              <strong>{lane.lane}</strong>
              <span>{lane.items.length} {editorPageText(locale, 'breaks', 'ברייקים')}</span>
            </div>
            <div
              className="timeline-track"
              ref={(node) => {
                trackRefs.current[lane.lane] = node;
              }}
            >
              {hours.map((hour) => (
                <i key={`${lane.lane}-${hour}`} style={{ left: `${((hour - startHour) / Math.max(1, endHour - startHour)) * 100}%` }} />
              ))}
              {lane.program && (
                <div
                  className="timeline-program-band"
                  style={positionStyle(timeToSeconds(lane.program.start_time), timeToSeconds(lane.program.end_time))}
                  title={`${lane.program.title} / ${lane.program.start_time}-${lane.program.end_time}`}
                >
                  <span>{lane.program.title}</span>
                </div>
              )}
              {lane.items.map((item) => {
                const { startSec, durationSec } = currentState(item);
                const edited = Boolean(edits[item.id]);
                const pinned = Boolean(constraintIdFor(item));
                const offsetSeconds = Math.max(0, startSec - item.program_start_sec);
                const className = [
                  'editor-break',
                  pinned ? 'pinned' : '',
                  edited && !pinned ? 'unsaved' : '',
                  item.is_gold ? 'gold' : '',
                ].filter(Boolean).join(' ');
                return (
                  <div
                    key={item.id}
                    className={className}
                    role="button"
                    tabIndex={0}
                    style={positionStyle(startSec, startSec + durationSec)}
                    title={`${item.program_title} ${formatOffset(offsetSeconds)} / ${Math.round(durationSec)}s`}
                    onPointerDown={(event) => handleMovePointerDown(event, lane.lane, item)}
                    onKeyDown={(event) => handleKeyDown(event, lane.lane, item)}
                    aria-label={`${item.program_title} ${formatOffset(offsetSeconds)} ${Math.round(durationSec)} ${editorPageText(locale, 'seconds', 'שניות')}`}
                  >
                    {pinned && <Lock className="editor-break-lock" size={11} />}
                    <span>{secondsToTime(startSec)}</span>
                    <strong>{formatOffset(offsetSeconds)}</strong>
                    <em>{Math.round(durationSec)}s</em>
                    <i
                      className="editor-break-resize"
                      onPointerDown={(event) => handleResizePointerDown(event, lane.lane, item)}
                      aria-hidden="true"
                    />
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="schedule-editor-readout" dir={he ? 'rtl' : 'ltr'}>
        {Object.keys(edits).length === 0 ? (
          <p>{editorPageText(locale, 'Drag a break to set its offset, then save it as a pin.', 'גרור ברייק כדי לקבוע את ההיסט שלו, ואז שמור אותו כנעיצה.')}</p>
        ) : (
          <ul className="schedule-editor-edit-list">
            {lanes.flatMap((lane) => lane.items.filter((item) => edits[item.id]).map((item) => {
              const { startSec, durationSec } = currentState(item);
              const offsetSeconds = Math.max(0, startSec - item.program_start_sec);
              const pinned = Boolean(constraintIdFor(item));
              return (
                <li key={item.id} className={pinned ? 'is-pinned' : 'is-unsaved'}>
                  <div>
                    <strong>{item.program_title}</strong>
                    <span>{formatOffset(offsetSeconds)} {editorPageText(locale, 'from start', 'מההתחלה')} / {Math.round(durationSec)}s</span>
                  </div>
                  <Button
                    type="button"
                    variant="contained"
                    className="run-button compact"
                    disabled={savingPin === item.id}
                    onClick={() => savePin(item, scopeChoice)}
                  >
                    <Save size={13} />
                    {pinned
                      ? editorPageText(locale, 'Update pin', 'עדכון נעיצה')
                      : editorPageText(locale, 'Save as pin', 'שמור כנעיצה')}
                  </Button>
                </li>
              );
            }))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ConstraintBuilder is a standalone panel (mounted on the Settings page) that
// posts richer constraints to POST /api/constraints and lists / deletes existing
// ones via GET / DELETE. It degrades gracefully when the endpoint 404s.
function ConstraintBuilder({ locale, notify, onRecompute, recomputeState }) {
  const he = locale === 'he';
  const [options, setOptions] = useState({ programmes: [], channels: [], weekdays: [] });
  const [optionsLoaded, setOptionsLoaded] = useState(false);
  const [items, setItems] = useState([]);
  const [available, setAvailable] = useState(true);
  const [saving, setSaving] = useState(false);
  const [draft, setDraft] = useState({
    scope_type: 'date',
    scope_value: '',
    channel: '',
    effect: 'FIX_OFFSET',
    offset_mmss: '00:00',
    offset_min: '00:00',
    offset_max: '00:00',
    pin_count: 1,
    duration_min: 30,
    duration_max: 120,
    order_index: '',
  });

  React.useEffect(() => {
    let active = true;
    async function load() {
      try {
        const optionsResponse = await fetch(`${API_BASE}/api/constraints/options`);
        if (optionsResponse.ok && active) {
          setOptions(await optionsResponse.json());
        }
      } catch (error) {
        // Options are optional; the date input still works without them.
      } finally {
        if (active) setOptionsLoaded(true);
      }
      try {
        const listResponse = await fetch(`${API_BASE}/api/constraints`);
        if (listResponse.status === 404) {
          if (active) setAvailable(false);
          return;
        }
        if (listResponse.ok && active) {
          const payload = await listResponse.json();
          setItems(normalizeRows(payload));
        }
      } catch (error) {
        // Leave the list empty if the API is offline.
      }
    }
    load();
    return () => {
      active = false;
    };
  }, []);

  function updateDraft(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  function mmssToSeconds(value) {
    const [minutes, seconds] = String(value || '00:00').split(':').map((part) => Number(part));
    return (Number.isFinite(minutes) ? minutes : 0) * 60 + (Number.isFinite(seconds) ? seconds : 0);
  }

  function buildBody() {
    const body = {
      scope_type: draft.scope_type,
      scope_value: draft.scope_value,
      channel: draft.channel,
      effect: draft.effect,
      order_index: draft.order_index === '' ? null : Number(draft.order_index),
    };
    if (draft.effect === 'FIX_OFFSET') {
      body.offset_seconds = mmssToSeconds(draft.offset_mmss);
    } else if (draft.effect === 'OFFSET_WINDOW') {
      body.offset_seconds_min = mmssToSeconds(draft.offset_min);
      body.offset_seconds_max = mmssToSeconds(draft.offset_max);
    } else if (draft.effect === 'PIN_COUNT') {
      body.pin_count = Number(draft.pin_count);
    } else if (draft.effect === 'DURATION_RANGE') {
      body.duration_seconds_min = Number(draft.duration_min);
      body.duration_seconds_max = Number(draft.duration_max);
    } else if (draft.effect === 'GOLD') {
      body.is_gold = true;
    } else if (draft.effect === 'FORBID') {
      body.forbid = true;
    }
    return body;
  }

  async function saveConstraint() {
    setSaving(true);
    try {
      const response = await fetch(`${API_BASE}/api/constraints`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildBody()),
      });
      if (response.status === 404) {
        setAvailable(false);
        notify('The constraints API is not available yet.', 'ממשק האילוצים עדיין לא זמין.');
        return;
      }
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const saved = await response.json();
      setItems((current) => [...current, { ...buildBody(), id: saved.id || `constraint-${current.length + 1}` }]);
      notify('Constraint saved.', 'האילוץ נשמר.');
    } catch (error) {
      notify(`Saving the constraint failed (${error.message}).`, `שמירת האילוץ נכשלה (${error.message}).`);
    } finally {
      setSaving(false);
    }
  }

  async function deleteConstraint(id) {
    try {
      const response = await fetch(`${API_BASE}/api/constraints/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (response.status === 404) {
        setItems((current) => current.filter((item) => item.id !== id));
        return;
      }
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      setItems((current) => current.filter((item) => item.id !== id));
      notify('Constraint removed.', 'האילוץ הוסר.');
    } catch (error) {
      notify(`Removing the constraint failed (${error.message}).`, `הסרת האילוץ נכשלה (${error.message}).`);
    }
  }

  const effects = [
    { value: 'FIX_OFFSET', label: editorPageText(locale, 'Fix offset', 'היסט קבוע') },
    { value: 'OFFSET_WINDOW', label: editorPageText(locale, 'Offset window', 'חלון היסט') },
    { value: 'PIN_COUNT', label: editorPageText(locale, 'Pin count', 'מספר ברייקים קבוע') },
    { value: 'DURATION_RANGE', label: editorPageText(locale, 'Duration range', 'טווח אורך') },
    { value: 'GOLD', label: editorPageText(locale, 'Gold break', 'ברייק זהב') },
    { value: 'FORBID', label: editorPageText(locale, 'Forbid', 'איסור') },
  ];

  return (
    <section className="settings-panel wide constraint-builder">
      <div className="settings-panel-head">
        <div>
          <h2>{editorPageText(locale, 'Constraint builder', 'בונה אילוצים')}</h2>
          <p>{editorPageText(locale, 'Pin offsets, windows, counts, durations, gold and forbid rules', 'נעיצת היסטים, חלונות, ספירה, אורכים, זהב וכללי איסור')}</p>
        </div>
      </div>
      {!available && (
        <p className="constraint-builder-warning">
          {editorPageText(locale, 'The constraints API responded with 404. Saving is disabled until it is online.', 'ממשק האילוצים החזיר 404. השמירה מושבתת עד שיהיה זמין.')}
        </p>
      )}
      <div className="constraint-builder-form">
        <div className="constraint-field">
          <span className="adv-field-label">{editorPageText(locale, 'Scope', 'היקף')}</span>
          <Select size="small" value={draft.scope_type} onChange={(event) => updateDraft('scope_type', event.target.value)}>
            <MenuItem value="date">{editorPageText(locale, 'Date', 'תאריך')}</MenuItem>
            <MenuItem value="programme">{editorPageText(locale, 'Programme', 'תוכנית')}</MenuItem>
            <MenuItem value="weekday">{editorPageText(locale, 'Weekday', 'יום בשבוע')}</MenuItem>
            <MenuItem value="channel">{editorPageText(locale, 'Channel', 'ערוץ')}</MenuItem>
          </Select>
        </div>
        <div className="constraint-field">
          <span className="adv-field-label">{editorPageText(locale, 'Scope value', 'ערך ההיקף')}</span>
          {draft.scope_type === 'date' ? (
            <TextField type="date" size="small" value={draft.scope_value} onChange={(event) => updateDraft('scope_value', event.target.value)} InputLabelProps={{ shrink: true }} />
          ) : (
            <Select
              size="small"
              value={draft.scope_value}
              displayEmpty
              onChange={(event) => updateDraft('scope_value', event.target.value)}
            >
              <MenuItem value="">{editorPageText(locale, 'Select', 'בחר')}</MenuItem>
              {(draft.scope_type === 'programme' ? options.programmes : draft.scope_type === 'channel' ? options.channels : options.weekdays).map((option) => {
                const value = typeof option === 'string' ? option : option.key || option.value || option.name;
                const label = typeof option === 'string' ? option : option.label || option.name || value;
                return <MenuItem key={value} value={value}>{label}</MenuItem>;
              })}
            </Select>
          )}
        </div>
        <div className="constraint-field">
          <span className="adv-field-label">{editorPageText(locale, 'Channel', 'ערוץ')}</span>
          <Select size="small" value={draft.channel} displayEmpty onChange={(event) => updateDraft('channel', event.target.value)}>
            <MenuItem value="">{editorPageText(locale, 'Any', 'הכול')}</MenuItem>
            {(options.channels || []).map((channel) => {
              const value = typeof channel === 'string' ? channel : channel.key || channel.value || channel.name;
              return <MenuItem key={value} value={value}>{value}</MenuItem>;
            })}
          </Select>
        </div>
        <div className="constraint-field">
          <span className="adv-field-label">{editorPageText(locale, 'Effect', 'אפקט')}</span>
          <Select size="small" value={draft.effect} onChange={(event) => updateDraft('effect', event.target.value)}>
            {effects.map((effect) => <MenuItem key={effect.value} value={effect.value}>{effect.label}</MenuItem>)}
          </Select>
        </div>
        {draft.effect === 'FIX_OFFSET' && (
          <div className="constraint-field">
            <span className="adv-field-label">{editorPageText(locale, 'Offset (MM:SS)', 'היסט (דק:שנ)')}</span>
            <TextField size="small" value={draft.offset_mmss} onChange={(event) => updateDraft('offset_mmss', event.target.value)} inputProps={{ dir: 'ltr', placeholder: '02:30' }} />
          </div>
        )}
        {draft.effect === 'OFFSET_WINDOW' && (
          <>
            <div className="constraint-field">
              <span className="adv-field-label">{editorPageText(locale, 'Min offset (MM:SS)', 'היסט מינ (דק:שנ)')}</span>
              <TextField size="small" value={draft.offset_min} onChange={(event) => updateDraft('offset_min', event.target.value)} inputProps={{ dir: 'ltr' }} />
            </div>
            <div className="constraint-field">
              <span className="adv-field-label">{editorPageText(locale, 'Max offset (MM:SS)', 'היסט מקס (דק:שנ)')}</span>
              <TextField size="small" value={draft.offset_max} onChange={(event) => updateDraft('offset_max', event.target.value)} inputProps={{ dir: 'ltr' }} />
            </div>
          </>
        )}
        {draft.effect === 'PIN_COUNT' && (
          <div className="constraint-field">
            <span className="adv-field-label">{editorPageText(locale, 'Break count', 'מספר ברייקים')}</span>
            <TextField type="number" size="small" value={draft.pin_count} onChange={(event) => updateDraft('pin_count', event.target.value)} inputProps={{ min: 0, dir: 'ltr' }} />
          </div>
        )}
        {draft.effect === 'DURATION_RANGE' && (
          <>
            <div className="constraint-field">
              <span className="adv-field-label">{editorPageText(locale, 'Min duration (s)', 'אורך מינ (שנ)')}</span>
              <TextField type="number" size="small" value={draft.duration_min} onChange={(event) => updateDraft('duration_min', event.target.value)} inputProps={{ min: 0, dir: 'ltr' }} />
            </div>
            <div className="constraint-field">
              <span className="adv-field-label">{editorPageText(locale, 'Max duration (s)', 'אורך מקס (שנ)')}</span>
              <TextField type="number" size="small" value={draft.duration_max} onChange={(event) => updateDraft('duration_max', event.target.value)} inputProps={{ min: 0, dir: 'ltr' }} />
            </div>
          </>
        )}
        <div className="constraint-field">
          <span className="adv-field-label">{editorPageText(locale, 'Order index (optional)', 'אינדקס סדר (רשות)')}</span>
          <TextField type="number" size="small" value={draft.order_index} onChange={(event) => updateDraft('order_index', event.target.value)} inputProps={{ min: 0, dir: 'ltr' }} />
        </div>
      </div>
      <div className="constraint-builder-actions">
        <Button type="button" variant="contained" className="run-button" disabled={saving || !available} onClick={saveConstraint}>
          <Save size={14} />
          {editorPageText(locale, 'Save constraint', 'שמור אילוץ')}
        </Button>
        <Button type="button" variant="outlined" className="run-button" disabled={recomputeState === 'running'} onClick={() => onRecompute && onRecompute()}>
          <Send size={14} />
          {editorPageText(locale, 'Recompute weekly schedule', 'חשב מחדש את הלוח השבועי')}
        </Button>
      </div>
      <div className="constraint-list">
        <div className="panel-head">
          <h3>{editorPageText(locale, 'Existing constraints', 'אילוצים קיימים')}</h3>
          <span>{items.length}</span>
        </div>
        {items.length === 0 ? (
          <p className="constraint-list-empty">{editorPageText(locale, 'No constraints yet.', 'אין אילוצים עדיין.')}</p>
        ) : (
          <ul>
            {items.map((item) => (
              <li key={item.id}>
                <span className="constraint-chip">{item.effect}</span>
                <span className="constraint-scope">{item.scope_type}: {item.scope_value || editorPageText(locale, 'any', 'הכול')}</span>
                {item.channel && <span className="constraint-channel">{item.channel}</span>}
                <Button type="button" variant="text" className="constraint-delete" onClick={() => deleteConstraint(item.id)} aria-label={editorPageText(locale, 'Delete constraint', 'מחק אילוץ')}>
                  <Trash2 size={14} />
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

export default ScheduleEditor;
export { ConstraintBuilder };
