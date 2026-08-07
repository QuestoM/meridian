import React from 'react';
import { Button } from '@mui/material';
import { ArrowUpRight, CalendarDays, X } from 'lucide-react';
import { formatNumber, pageText } from '../../shell/format';
import { Figure } from '../../shell/bidi';
import { useScheduleZoom } from '../day/schedule-track-view';
import ScheduleEditor from '../day/ScheduleEditor';
import { SAVED_PLAN, withBasis } from '../day/plan-basis';
import GridAxisControl from './GridAxisControl';
import PlanningCanvas from './PlanningCanvas';
import TimelineView from './TimelineView';
import DaypartView from './DaypartView';
import { boardReason, exportScopeNote, scopeLine, weekdayLabel } from './plan-week-model';

// The week the plan produced, and the zoom control that steps into a day.
//
// This is the merge section 3.3 of the specification asks for. A timeline of
// breaks used to render on two destinations over the same rows, which is the
// Optimizer-versus-Schedule duplication discovery named as the first of the
// three heaviest structural facts. Here it renders once, and the day editor is
// the next zoom level rather than a different page, so the scheduler's drag and
// its 30 and 60 second snap arrive from inside the plan they belong to.
//
// Every view on this board is the operator's own channel. The route scopes the
// payload and the scope line prints what was kept, so a figure never reads as
// the market's by accident. The plan CSV is the one thing on this subject that
// is not scoped: it is the whole saved plan, every channel in the source, so it
// is not downloadable from here. The board names that file and points at the
// destination that owns it rather than serving three rivals' plans in one click.

const VIEWS = ['grid', 'strip', 'timeline', 'day'];

// Which broadcast day the day zoom is standing on, said out loud.
//
// The zoom has always drawn one day, the first the programme source carries,
// and never named it. Now the comparison can send a planner here with a day in
// mind, so the day is named, its two counts come from the payload that drew it,
// and a day the source does not carry prints the reason rather than the next
// day along.
function DayHeader({ board, focusDate, state, error, locale, onClear, weekday }) {
  if (state === 'loading') {
    return (
      <p className="plan-note plan-note-quiet" role="status">
        {pageText(locale, `Opening broadcast day ${focusDate}`, `פותח את יום השידור ${focusDate}`)}
      </p>
    );
  }
  if (state === 'error') {
    return (
      <p className="plan-note plan-note-red" role="alert">
        {pageText(locale, `That broadcast day could not be opened: ${error}`, `לא ניתן היה לפתוח את יום השידור הזה: ${error}`)}
      </p>
    );
  }
  const reason = board && board.available === false ? boardReason(board, locale) : null;
  const date = board?.date || focusDate || null;
  return (
    <div className={`plan-board-day${reason ? ' is-empty' : ''}`}>
      <div className="plan-board-day-name">
        <CalendarDays size={15} aria-hidden="true" />
        <strong className="numeric"><Figure>{date || '-'}</Figure></strong>
        {weekday ? <span>{weekdayLabel(weekday, locale)}</span> : null}
      </div>
      {reason ? (
        <p className="plan-board-day-reason" role="status">{reason}</p>
      ) : (
        <p className="plan-board-day-counts">
          {pageText(
            locale,
            withBasis(
              `${formatNumber(board?.programmes, locale)} programmes and ${formatNumber(board?.breaks, locale)} breaks on your channel`,
              SAVED_PLAN,
              'en',
            ),
            withBasis(
              `${formatNumber(board?.programmes, locale)} תוכניות ו-${formatNumber(board?.breaks, locale)} ברייקים בערוץ שלכם`,
              SAVED_PLAN,
              'he',
            ),
          )}
        </p>
      )}
      {focusDate && onClear ? (
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={onClear}>
          <X size={14} />
          {pageText(locale, 'Back to the week', 'חזרה לשבוע')}
        </Button>
      ) : null}
    </div>
  );
}

function viewLabel(view, locale) {
  if (view === 'grid') return pageText(locale, 'Grid', 'רשת');
  if (view === 'strip') return pageText(locale, 'Broadcast strips', 'רצועות שידור');
  if (view === 'timeline') return pageText(locale, 'Timeline', 'ציר זמן');
  return pageText(locale, 'One day, editable', 'יום אחד, לעריכה');
}

export function BoardPanel({
  schedule,
  copy,
  locale,
  notify,
  planEvents,
  dayEvents,
  onGlobalRefresh,
  onRun,
  runState,
  selectedProgramKey,
  onSelectProgram,
  view,
  onViewChange,
  gridAxis,
  onGridAxisChange,
  focusDate,
  dayPayload,
  dayState,
  dayError,
  onClearFocus,
}) {
  const zoom = useScheduleZoom();
  const rows = Array.isArray(schedule?.break_schedule) ? schedule.break_schedule : [];
  const total = Number(schedule?.break_schedule_total_rows);
  const scope = scopeLine(schedule?.scope?.plan, locale);
  const fileNote = exportScopeNote(schedule?.scope?.plan, locale);
  // The route reports whether it could scope the payload at all. When it could
  // not, the rows it returns are the whole market, and a disclosure over them
  // does not make them the operator's: rival programme titles and rival revenue
  // would still be on the screen. Measured on this tree at 2026-08-01, with the
  // operator channel cleared from settings, that is exactly what rendered. So
  // the board declines to draw and names the input that is missing.
  const unscoped = schedule?.scope?.plan?.scoped === false;
  // The day zoom reads the day payload when a day was asked for and the week
  // payload's own embedded board when one was not. It never mixes the two: a
  // focused day that has not arrived yet draws nothing rather than the day the
  // week payload happens to carry.
  const daySchedule = focusDate ? dayPayload : schedule;
  const dayBoard = daySchedule?.board || null;
  const dayWeekday = daySchedule?.break_operations?.programs?.[0]?.day || null;
  const dayDrawable = Boolean(daySchedule) && dayBoard?.available !== false
    && (!focusDate || dayState === 'ready');

  return (
    <section className="plan-section" aria-labelledby="plan-board-title">
      <div className="plan-section-head">
        <div>
          <h2 id="plan-board-title">{pageText(locale, 'The week the plan produced', 'השבוע שהתוכנית ייצרה')}</h2>
          <p>
            {scope || pageText(locale, 'Every programme on your channel, and the breaks the plan placed in it.', 'כל תוכנית בערוץ שלכם, והברייקים שהתוכנית שיבצה בה.')}
          </p>
          {fileNote && <p className="plan-basis-note">{fileNote}</p>}
        </div>
        {fileNote && (
          <Button
            className="secondary-button compact"
            type="button"
            variant="outlined"
            onClick={() => { window.location.hash = 'Reports'; }}
          >
            <ArrowUpRight size={14} />
            {pageText(locale, 'Open the plan file on Sources', 'מעבר לקובץ התוכנית במסך המקורות')}
          </Button>
        )}
      </div>

      {unscoped && (
        <div className="plan-note plan-note-amber" role="status">
          <p>{pageText(locale, 'No operator channel is configured, so the rows behind this board are every channel in the source rather than yours, and drawing them would put another broadcaster on your screen. The path to a week board: set the operator channel in Settings.', 'לא הוגדר ערוץ מפעיל, ולכן השורות שמאחורי הלוח הזה הן של כל הערוצים במקור הנתונים ולא שלכם, וציור שלהן היה מציג כאן שידורים של גוף אחר. הדרך ללוח שבוע: הגדירו ערוץ מפעיל במסך ההגדרות.')}</p>
          <Button
            className="secondary-button compact"
            type="button"
            variant="outlined"
            onClick={() => { window.location.hash = 'Settings'; }}
          >
            {pageText(locale, 'Open settings', 'פתיחת ההגדרות')}
          </Button>
        </div>
      )}

      {!unscoped && (
        <>
          <div className="plan-board-toolbar">
            <div className="plan-board-views" role="group" aria-label={pageText(locale, 'Zoom', 'זום')}>
              {VIEWS.map((item) => (
                <Button
                  key={item}
                  className={view === item ? 'segmented active' : 'segmented'}
                  type="button"
                  variant="outlined"
                  aria-pressed={view === item}
                  onClick={() => onViewChange(item)}
                >
                  {viewLabel(item, locale)}
                </Button>
              ))}
            </div>
            {view === 'grid' && <GridAxisControl value={gridAxis} onChange={onGridAxisChange} locale={locale} />}
            {Number.isFinite(total) && total > rows.length && (
              <span className="plan-board-count">
                {pageText(
                  locale,
                  `showing the first ${formatNumber(rows.length, locale)} of ${formatNumber(total, locale)} plan rows`,
                  `מוצגות ${formatNumber(rows.length, locale)} השורות הראשונות מתוך ${formatNumber(total, locale)} שורות תוכנית`,
                )}
              </span>
            )}
          </div>

          {view === 'grid' && (
            <PlanningCanvas
              rows={schedule?.rows || []}
              copy={copy}
              locale={locale}
              axis={gridAxis}
              dayEvents={dayEvents}
              selectedProgramKey={selectedProgramKey}
              onSelectProgram={onSelectProgram}
            />
          )}
          {view === 'strip' && (
            <DaypartView
              rows={schedule?.rows || []}
              locale={locale}
              selectedProgramKey={selectedProgramKey}
              onSelectProgram={onSelectProgram}
            />
          )}
          {view === 'timeline' && (
            <TimelineView
              timeline={schedule?.break_operations}
              rows={schedule?.rows || []}
              locale={locale}
              notify={notify}
              zoom={zoom}
              onGlobalRefresh={onGlobalRefresh}
              selectedProgramKey={selectedProgramKey}
              onSelectProgram={onSelectProgram}
            />
          )}
          {view === 'day' && (
            <>
              <DayHeader
                board={dayBoard}
                focusDate={focusDate || null}
                state={focusDate ? dayState : 'ready'}
                error={dayError}
                locale={locale}
                onClear={onClearFocus}
                weekday={dayWeekday}
              />
              {dayDrawable && (
                <ScheduleEditor
                  schedule={daySchedule}
                  locale={locale}
                  notify={notify}
                  onRecompute={onRun}
                  recomputeState={runState}
                  onGlobalRefresh={onGlobalRefresh}
                  zoom={zoom}
                />
              )}
            </>
          )}
          {planEvents && planEvents.length > 0 && view === 'grid' && (
            <p className="plan-basis-note">
              {pageText(
                locale,
                'Calendar events are marked on the day columns for reading only. No retention or revenue figure changes because of one.',
                'אירועי לוח מסומנים על עמודות הימים לקריאה בלבד. אף ערך שימור או הכנסה אינו משתנה בגללם.',
              )}
            </p>
          )}
        </>
      )}
    </section>
  );
}

export default BoardPanel;
