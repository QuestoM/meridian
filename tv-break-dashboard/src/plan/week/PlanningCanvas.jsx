import React from 'react';
import { Button, Tooltip } from '@mui/material';
import { Numeric, formatCurrency, formatMinutes, formatNumber, formatPercent, pageText } from '../../shell/format';
import { dayLabel, daypartLabel, gridAxisLabel, programTypeLabel } from '../../shell/labels';
import {
  daypartForTime,
  daypartKeys,
  flattenScheduleRows,
  hourFromTime,
  programKey,
} from '../../shell/plan-model';
import { SUNDAY_FIRST, isWeekend } from './plan-week-model';

export function buildPlannerColumns(rows, axis, locale) {
  if (axis === 'daypart') {
    return daypartKeys.map((daypart) => ({ key: daypart, label: daypartLabel(daypart, locale) }));
  }
  if (axis === 'hour') {
    const hours = Array.from(new Set(flattenScheduleRows(rows).map((program) => hourFromTime(program.time)))).sort((a, b) => a - b);
    return (hours.length ? hours : [20]).map((hour) => ({
      key: `hour-${hour}`,
      hour,
      label: `${String(hour).padStart(2, '0')}:00`,
    }));
  }
  if (axis === 'type') {
    const types = Array.from(new Set(flattenScheduleRows(rows).map((program) => program.program_type || 'Other'))).sort();
    return (types.length ? types : ['Other']).map((programType) => ({
      key: `type-${programType}`,
      programType,
      label: programTypeLabel(programType, locale),
    }));
  }
  // The Israeli week: Sunday to Saturday, weekend Friday and Saturday. The
  // shell's own weekday array is Monday-first and frozen, so this surface reads
  // its order from this tree instead.
  return SUNDAY_FIRST.map((day) => ({ key: day, label: dayLabel(day, locale), weekend: isWeekend(day) }));
}

export function programsForPlannerColumn(programs, column, axis) {
  if (axis === 'daypart') {
    return programs.filter((program) => daypartForTime(program.time) === column.key);
  }
  if (axis === 'hour') {
    return programs.filter((program) => hourFromTime(program.time) === column.hour);
  }
  if (axis === 'type') {
    return programs.filter((program) => (program.program_type || 'Other') === column.programType);
  }
  return programs.filter((program) => program.day === column.key);
}

export function PlanningCanvas({ rows, copy, locale, axis = 'day', dayEvents = null, showPrograms = true, showBreaks = true, selectedProgramKey, onSelectProgram }) {
  const columns = buildPlannerColumns(rows, axis, locale);
  const cellMinWidth = axis === 'hour' ? 112 : 136;
  const gridTemplateColumns = `142px repeat(${columns.length}, minmax(${cellMinWidth}px, 1fr))`;
  const minWidth = 142 + columns.length * cellMinWidth;
  return (
    <div className="planning-canvas">
      <div className="canvas-header" style={{ gridTemplateColumns, minWidth }}>
        <span>{copy.channelProgram} / {gridAxisLabel(axis, locale)}</span>
        {columns.map((column) => {
          // Display-only calendar-event badge on day columns: name only, no
          // number on any surface changes because of an event.
          const eventNames = axis === 'day' && dayEvents ? dayEvents[column.key] : null;
          return (
            <span key={column.key} className={column.weekend ? 'canvas-weekend' : undefined}>
              {column.label}
              {Array.isArray(eventNames) && eventNames.length > 0 && (
                <Tooltip title={pageText(locale, 'An active calendar event covers this plan day. Display only; no retention or revenue number changes.', 'אירוע פעיל מלוח האירועים חל ביום התוכנית הזה. תצוגה בלבד; אף מספר שימור או הכנסה אינו משתנה.')} arrow>
                  <em className="canvas-day-event">{eventNames.join(', ')}</em>
                </Tooltip>
              )}
            </span>
          );
        })}
      </div>
      {rows.map((row, rowIndex) => {
        const programs = Array.isArray(row.programs) ? row.programs : [];
        const channelName = String(row.channel || 'Channel');
        return (
        <div className="channel-row" key={channelName || `channel-${rowIndex}`} style={{ gridTemplateColumns, minWidth }}>
          <div className="channel-name">
            <span>{channelName}</span>
            <small>{programTypeLabel(programs[0]?.program_type, locale) || pageText(locale, 'Mixed', 'מעורב')}</small>
          </div>
          {columns.map((column) => {
            const cellPrograms = programsForPlannerColumn(programs, column, axis);
            const program = cellPrograms.find((item) => item.selected) || cellPrograms[0];
            const programWithChannel = program
              ? { channel: channelName, ...program, key: programKey(channelName, program) }
              : null;
            const totalRevenue = cellPrograms.reduce((sum, item) => sum + Number(item.revenue || 0), 0);
            const averageRetention = cellPrograms.length
              ? cellPrograms.reduce((sum, item) => sum + Number(item.retention || 0), 0) / cellPrograms.length
              : 0;
            const markerCount = cellPrograms.reduce((sum, item) => sum + Number(item.break_markers || 0), 0);
            const timeRange = cellPrograms.length
              ? `${cellPrograms[0].time} - ${cellPrograms[cellPrograms.length - 1].time}`
              : '';
            const selectedInCell = selectedProgramKey
              ? cellPrograms.some((item) => programKey(channelName, item) === selectedProgramKey)
              : cellPrograms.some((item) => item.selected);
            return (
              <ProgramCell
                key={`${channelName}-${column.key}`}
                program={programWithChannel}
                locale={locale}
                selected={selectedInCell}
                programCount={cellPrograms.length}
                totalRevenue={totalRevenue}
                averageRetention={averageRetention}
                markerCount={markerCount}
                timeRange={timeRange}
                showPrograms={showPrograms}
                showBreaks={showBreaks}
                onSelect={onSelectProgram}
              />
            );
          })}
        </div>
        );
      })}
    </div>
  );
}

export function ProgramCell({
  program,
  locale,
  selected = false,
  programCount = 1,
  totalRevenue,
  averageRetention,
  markerCount,
  timeRange,
  showPrograms = true,
  showBreaks = true,
  onSelect,
}) {
  if (!program) return <div className="program-cell empty" />;
  // Marker dots mirror the planned break count; zero breaks shows zero dots
  // (the fixed-height strip keeps the cell layout stable) instead of a
  // fabricated minimum of one.
  const markers = Array.from({
    length: Math.max(0, Math.min(10, Number(markerCount ?? program.break_markers ?? 0) || 0)),
  });
  const revenue = totalRevenue ?? program.revenue;
  const retention = averageRetention ?? program.retention;
  // Time ranges and clock values are isolated LTR runs (Numeric): a bare
  // "20:00 - 22:30" inside an RTL cell renders with the end time first.
  const meta = programCount > 1
    ? <>{formatNumber(programCount, locale)} {pageText(locale, 'programs', 'תוכניות')} / <Numeric>{timeRange}</Numeric></>
    : <><Numeric>{program.time}</Numeric> / {formatMinutes(Number(program.duration_minutes || 0) * 60, locale)}</>;
  return (
    <Tooltip title={`${program.title} / ${program.channel} / ${dayLabel(program.day, locale)} ${program.time}`} arrow placement="bottom">
    <Button
      className={selected ? 'program-cell selected' : 'program-cell'}
      type="button"
      variant="text"
      disableRipple
      aria-pressed={selected}
      onClick={() => onSelect?.(program)}
    >
      {showPrograms ? (
        <span className="program-title">{program.title}</span>
      ) : (
        <span className="program-title muted-title">{programTypeLabel(program.program_type, locale) || pageText(locale, 'Program hidden', 'תוכנית מוסתרת')}</span>
      )}
      <span className="program-meta">{meta}</span>
      {showBreaks && (
        <span className="break-markers">
          {markers.map((_, index) => (
            <i key={index} className={index % 3 === 0 ? 'marker revenue' : 'marker'} />
          ))}
        </span>
      )}
      <span className="cell-metrics">
        <span><Numeric>{formatCurrency(revenue, locale)}</Numeric></span>
        <span><Numeric>{formatPercent(retention, locale)}</Numeric></span>
      </span>
    </Button>
    </Tooltip>
  );
}

export default PlanningCanvas;
