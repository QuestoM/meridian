import React, { useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Download, Printer } from 'lucide-react';
import { finiteNumber, formatCurrency, formatMinutes, formatNumber, formatPercent, pageText } from '../../shell/format';
import { breakLengthLabel, breakPositionLabel, dayLabel, programTypeLabel } from '../../shell/labels';
import { normalizeRows } from '../../shell/plan-model';
import { gridAxisFromLocation } from '../../shell/nav';
import { downloadJson, downloadScheduleCsv } from '../../shell/downloads';
import { DataTable, PageHeader } from '../../shell/primitives';
import { planEventWeekdayMap } from '../../rules/CalendarEventsModel';
import { useScheduleZoom } from '../day/schedule-track-view';
import ScheduleEditor from '../day/ScheduleEditor';
import GoldBreakManager from '../break/GoldBreakManager';
import GridAxisControl from './GridAxisControl';
import PlanningCanvas from './PlanningCanvas';
import TimelineView from './TimelineView';
import DaypartView from './DaypartView';

export function SchedulePage({ schedule, overview, copy, locale, notify, onRecompute, recomputeState, refreshKey, onGlobalRefresh, planEvents }) {
  const rows = normalizeRows(schedule.break_schedule);
  // Display-only badge data: which planner weekdays sit inside an active
  // calendar event, derived from the server-computed plan overlap dates.
  const dayEvents = useMemo(() => planEventWeekdayMap(planEvents), [planEvents]);
  // The API slices break_schedule to its first 200 rows; when it also reports
  // the full count, the table header says so instead of posing as complete.
  const totalRows = finiteNumber(schedule.total_rows);
  const [scheduleMode, setScheduleMode] = useState('grid');
  const [scheduleAxis, setScheduleAxis] = useState(gridAxisFromLocation);
  const [selectedProgramKey, setSelectedProgramKey] = useState(null);
  // Zoom is shared across the timeline and editor so switching modes keeps one
  // scale (the video-editor style time scale, held in state per page visit).
  const zoom = useScheduleZoom();
  function handleSelectProgram(program) {
    setSelectedProgramKey(program.key);
  }
  return (
    <section className="page-workspace schedule-printable">
      <PageHeader
        locale={locale}
        titleEn="Schedule control"
        titleHe="בקרת לוח שידורים"
        bodyEn="Review the weekly break plan by programme type, day, length, expected revenue, and retention guardrail."
        bodyHe="בדיקת תוכנית הברייקים השמורה לפי סוג תוכנית, יום, אורך, הכנסת תוכנית ושימור צפייה."
        action={
          <div className="schedule-actions no-print">
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => downloadScheduleCsv(locale, notify, overview?.schedule_freshness)}
            >
              <Download size={14} />
              {pageText(locale, 'Download CSV', 'הורדת CSV')}
            </Button>
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => window.print()}
            >
              <Printer size={14} />
              {pageText(locale, 'Print', 'הדפסה')}
            </Button>
          </div>
        }
      />
      <section className="planner-surface compact-surface no-print">
        <div className="surface-toolbar">
          <div className="toolbar-left">
            <Button
              className={scheduleMode === 'grid' ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              aria-pressed={scheduleMode === 'grid'}
              onClick={() => setScheduleMode('grid')}
            >
              {copy.toolbar[0]}
            </Button>
            <Button
              className={scheduleMode === 'daypart' ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              aria-pressed={scheduleMode === 'daypart'}
              onClick={() => setScheduleMode('daypart')}
            >
              {copy.toolbar[2]}
            </Button>
            <Button
              className={scheduleMode === 'timeline' ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              aria-pressed={scheduleMode === 'timeline'}
              onClick={() => setScheduleMode('timeline')}
            >
              {copy.toolbar[1]}
            </Button>
            <Button
              className={scheduleMode === 'editor' ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              aria-pressed={scheduleMode === 'editor'}
              onClick={() => setScheduleMode('editor')}
            >
              {pageText(locale, 'Editor', 'עורך')}
            </Button>
          </div>
          <div className="toolbar-right">
            {scheduleMode === 'grid' && (
              <GridAxisControl value={scheduleAxis} onChange={setScheduleAxis} locale={locale} />
            )}
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => downloadJson('kairos-weekly-plan-first-200.json', { schedule: rows, grid: schedule.rows || [], axis: scheduleAxis })}
            >
              <Download size={14} />
              {copy.exportOptions[1]}
            </Button>
          </div>
        </div>
        {scheduleMode === 'grid' ? (
          <PlanningCanvas
            rows={schedule.rows || []}
            copy={copy}
            locale={locale}
            axis={scheduleAxis}
            dayEvents={dayEvents}
            selectedProgramKey={selectedProgramKey}
            onSelectProgram={handleSelectProgram}
          />
        ) : scheduleMode === 'timeline' ? (
          <TimelineView
            timeline={schedule.break_operations}
            rows={schedule.rows || []}
            locale={locale}
            notify={notify}
            zoom={zoom}
            onGlobalRefresh={onGlobalRefresh}
            selectedProgramKey={selectedProgramKey}
            onSelectProgram={handleSelectProgram}
          />
        ) : scheduleMode === 'editor' ? (
          <ScheduleEditor
            schedule={schedule}
            locale={locale}
            notify={notify}
            onRecompute={onRecompute}
            recomputeState={recomputeState}
            onGlobalRefresh={onGlobalRefresh}
            zoom={zoom}
          />
        ) : (
          <DaypartView
            rows={schedule.rows || []}
            locale={locale}
            selectedProgramKey={selectedProgramKey}
            onSelectProgram={handleSelectProgram}
          />
        )}
      </section>
      <section className="page-panel schedule-print-region">
        <div className="panel-head">
          <h2>{pageText(locale, 'Break plan rows', 'שורות תוכנית ברייקים')}</h2>
          <span>
            {totalRows !== null && totalRows > rows.length
              ? pageText(locale, `first ${formatNumber(rows.length, locale)} of ${formatNumber(totalRows, locale)} rows`, `${formatNumber(rows.length, locale)} הראשונות מתוך ${formatNumber(totalRows, locale)} שורות`)
              : `${rows.length} ${pageText(locale, 'rows', 'שורות')}`}
          </span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No scheduled breaks were found.', 'לא נמצאו ברייקים מתוכננים.')}
          rows={rows}
          columns={[
            { key: 'day', label: pageText(locale, 'Day', 'יום'), render: (row) => dayLabel(row.day, locale) },
            { key: 'program_type', label: pageText(locale, 'Programme type', 'סוג תוכנית'), render: (row) => programTypeLabel(row.program_type, locale) },
            { key: 'position', label: pageText(locale, 'Position', 'מיקום'), render: (row) => breakPositionLabel(row.position, locale) },
            { key: 'break_type', label: pageText(locale, 'Break type', 'סוג ברייק'), render: (row) => breakLengthLabel(row.break_type, locale) },
            { key: 'num_breaks', label: pageText(locale, 'Breaks', 'ברייקים'), render: (row) => formatNumber(row.num_breaks, locale) },
            { key: 'total_break_time', label: pageText(locale, 'Ad minutes', 'דקות פרסום'), render: (row) => formatMinutes(row.total_break_time, locale) },
            { key: 'predicted_revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.predicted_revenue, locale) },
            { key: 'predicted_retention', label: pageText(locale, 'Retention', 'שימור'), render: (row) => formatPercent(Number(row.predicted_retention || 0) * 100, locale) },
          ]}
        />
      </section>
      <GoldBreakManager locale={locale} refreshKey={refreshKey} />
    </section>
  );
}

export default SchedulePage;
