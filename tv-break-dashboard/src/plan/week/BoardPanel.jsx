import React from 'react';
import { Button } from '../../studio/actions';
import { ArrowUpRight, BarChart3, X } from 'lucide-react';
import { Numeric, formatCurrency, formatMinutes, formatNumber, formatPercent, pageText } from '../../shell/format';
import { programTypeLabel } from '../../shell/labels';
import { flattenScheduleRows } from '../../shell/plan-model';
import { useScheduleZoom } from '../day/schedule-track-view';
import GridAxisControl from './GridAxisControl';
import PlanningCanvas from './PlanningCanvas';
import TimelineView from './TimelineView';
import DaypartView from './DaypartView';
import PlanBoardWorkbench from './PlanBoardWorkbench';
import { exportScopeNote, scopeLine } from './plan-week-model';

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

const VIEWS = ['day', 'grid', 'strip', 'timeline'];

function viewLabel(view, locale) {
  if (view === 'grid') return pageText(locale, 'Week grid', 'רשת שבועית');
  if (view === 'strip') return pageText(locale, 'Daypart analysis', 'ניתוח רצועות');
  if (view === 'timeline') return pageText(locale, 'Source timeline', 'ציר זמן מקור');
  return pageText(locale, 'Day workbench', 'שולחן עבודה יומי');
}

function PlannerContext({ program, locale, onClear }) {
  if (!program) {
    return (
      <aside className="planner-context is-empty">
        <span className="plan-inspector-eyebrow">{pageText(locale, 'Programme inspector', 'בודק תוכנית')}</span>
        <h3>{pageText(locale, 'No programme selected', 'לא נבחרה תוכנית')}</h3>
        <p>{pageText(locale, 'Select any programme band to read the full title and its measured plan facts without leaving the week.', 'בחרו רצועת תוכנית כדי לקרוא את הכותרת המלאה ואת עובדות התוכנית שנמדדו, בלי לעזוב את השבוע.')}</p>
      </aside>
    );
  }
  return (
    <aside className="planner-context" aria-live="polite">
      <div className="planner-context-head">
        <span className="plan-inspector-eyebrow">{pageText(locale, 'Focused programme', 'תוכנית במיקוד')}</span>
        <Button type="button" variant="text" aria-label={pageText(locale, 'Clear programme focus', 'ניקוי מיקוד התוכנית')} onClick={onClear}><X size={16} /></Button>
      </div>
      <h3><bdi>{program.title}</bdi></h3>
      <p><bdi>{program.channel}</bdi> · {programTypeLabel(program.program_type, locale)}</p>
      <dl>
        <div><dt>{pageText(locale, 'Transmission', 'שידור')}</dt><dd><Numeric>{program.date || program.day} · {program.time}</Numeric></dd></div>
        <div><dt>{pageText(locale, 'Duration', 'משך')}</dt><dd><Numeric>{formatMinutes(Number(program.duration_minutes || 0) * 60, locale)}</Numeric></dd></div>
        <div><dt>{pageText(locale, 'Planned breaks', 'ברייקים מתוכננים')}</dt><dd><Numeric>{formatNumber(program.break_markers, locale)}</Numeric></dd></div>
        <div><dt>{pageText(locale, 'Expected revenue', 'הכנסה צפויה')}</dt><dd><Numeric>{formatCurrency(program.revenue, locale)}</Numeric></dd></div>
        <div><dt>{pageText(locale, 'Expected retention', 'שימור צפוי')}</dt><dd><Numeric>{formatPercent(program.retention, locale)}</Numeric></dd></div>
      </dl>
    </aside>
  );
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
  runDisabled,
  runDisabledReason,
  selectedProgramKey,
  onSelectProgram,
  view,
  onViewChange,
  gridAxis,
  onGridAxisChange,
  focusDate,
  onFocusDateChange,
  versions,
  live,
  freshness,
  canEdit,
  canEditReason,
  versionName,
  versionNote,
  publishState,
  publishError,
  selectedVersion,
  diff,
  onVersionName,
  onVersionNote,
  onPublish,
  onVersionDiff,
  onRestore,
  onOpenHistory,
}) {
  const zoom = useScheduleZoom();
  const rows = Array.isArray(schedule?.break_schedule) ? schedule.break_schedule : [];
  const selectedProgram = flattenScheduleRows(schedule?.rows || []).find((program) => program.key === selectedProgramKey) || null;
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
  return (
    <section className="card plan-section" aria-labelledby="plan-board-title">
      <div className="plan-section-head">
        <div>
          <h2 id="plan-board-title">{pageText(locale, 'Plan board', 'לוח התכנון')}</h2>
          <p>
            {scope || pageText(locale, 'Operate one day against the weekly plan of record, with analytical views kept as references.', 'עבודה על יום אחד מול תוכנית הייחוס השבועית, כשהתצוגות האנליטיות נשארות לעיון.')}
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
            {view !== 'day' && (
              <span className="plan-analysis-label">
                <BarChart3 size={14} aria-hidden="true" />
                {pageText(locale, 'Read-only analysis', 'ניתוח לקריאה בלבד')}
              </span>
            )}
            {view === 'grid' && <GridAxisControl value={gridAxis} onChange={onGridAxisChange} locale={locale} />}
            {view !== 'day' && Number.isFinite(total) && total > rows.length && (
              <span className="plan-board-count">
                {pageText(
                  locale,
                  `showing the first ${formatNumber(rows.length, locale)} of ${formatNumber(total, locale)} plan rows`,
                  `מוצגות ${formatNumber(rows.length, locale)} השורות הראשונות מתוך ${formatNumber(total, locale)} שורות תוכנית`,
                )}
              </span>
            )}
          </div>

          {view !== 'day' && (
            <div className="plan-board-stage">
              <div className="plan-board-instrument">
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
              </div>
              <PlannerContext program={selectedProgram} locale={locale} onClear={() => onSelectProgram?.(null)} />
            </div>
          )}
          {view === 'day' && (
            <PlanBoardWorkbench
              schedule={schedule}
              locale={locale}
              notify={notify}
              onGlobalRefresh={onGlobalRefresh}
              focusDate={focusDate}
              onFocusDateChange={onFocusDateChange}
              versions={versions}
              live={live}
              freshness={freshness}
              canEdit={canEdit}
              canEditReason={canEditReason}
              versionName={versionName}
              versionNote={versionNote}
              publishState={publishState}
              publishError={publishError}
              selectedVersion={selectedVersion}
              diff={diff}
              runState={runState}
              runDisabled={runDisabled}
              runDisabledReason={runDisabledReason}
              onVersionName={onVersionName}
              onVersionNote={onVersionNote}
              onPublish={onPublish}
              onVersionDiff={onVersionDiff}
              onRestore={onRestore}
              onRun={onRun}
              onOpenHistory={onOpenHistory}
            />
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
