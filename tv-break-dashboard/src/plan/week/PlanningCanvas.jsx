import React from 'react';
import { Tooltip } from '@mui/material';
import { Button } from '../../studio/actions';
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

const DAY_MINUTES = 24 * 60;

function minuteOfDay(time) {
  const [hour, minute] = String(time || '00:00').split(':').map(Number);
  return (Number.isFinite(hour) ? hour : 0) * 60 + (Number.isFinite(minute) ? minute : 0);
}

// Programme sources can contain a promo marker and a programme covering the
// same clock. They are both real rows, so the board keeps both and packs them
// into parallel lanes instead of painting one over the other.
export function packPlannerPrograms(items) {
  const laneEnds = [];
  const packed = [...items]
    .sort((a, b) => minuteOfDay(a.time) - minuteOfDay(b.time) || Number(b.duration_minutes || 0) - Number(a.duration_minutes || 0))
    .map((program) => {
      const start = minuteOfDay(program.time);
      const duration = Math.max(1, Number(program.duration_minutes || 0));
      let lane = laneEnds.findIndex((end) => end <= start);
      if (lane < 0) lane = laneEnds.length;
      laneEnds[lane] = start + duration;
      return { ...program, _plannerStart: start, _plannerDuration: duration, _plannerLane: lane };
    });
  return { items: packed, lanes: Math.max(1, laneEnds.length) };
}

function DayTimelineProgram({ program, channel, locale, selected, showPrograms, showBreaks, onSelect }) {
  const key = programKey(channel, program);
  const top = (program._plannerStart / DAY_MINUTES) * 100;
  const height = Math.max(0.22, (program._plannerDuration / DAY_MINUTES) * 100);
  const laneWidth = 100 / program._plannerLanes;
  const laneStart = program._plannerLane * laneWidth;
  const markerCount = Math.max(0, Number(program.break_markers || 0));
  const fullProgram = { channel, ...program, key };
  const label = `${program.title} / ${program.time} / ${formatMinutes(program._plannerDuration * 60, locale)} / ${formatCurrency(program.revenue, locale)}`;
  return (
    <Button
      type="button"
      variant="text"
      className={`planner-program${selected ? ' is-selected' : ''}${program._plannerDuration < 18 ? ' is-compact' : ''}`}
      style={{
        '--planner-top': `${top}%`,
        '--planner-height': `${height}%`,
        '--planner-inline': `${laneStart}%`,
        '--planner-width': `${laneWidth}%`,
      }}
      title={label}
      aria-label={label}
      aria-pressed={selected}
      onClick={() => onSelect?.(fullProgram)}
    >
      <span className="planner-program-title">
        {showPrograms ? program.title : programTypeLabel(program.program_type, locale) || pageText(locale, 'Programme', 'תוכנית')}
      </span>
      <span className="planner-program-clock"><Numeric>{program.time}</Numeric></span>
      {showBreaks && markerCount > 0 ? (
        <span className="planner-program-breaks" aria-hidden="true">
          {Array.from({ length: Math.min(6, markerCount) }, (_, index) => <i key={index} />)}
        </span>
      ) : null}
    </Button>
  );
}

function DayTimelineColumn({ programs, channel, locale, selectedProgramKey, showPrograms, showBreaks, onSelectProgram }) {
  const packed = packPlannerPrograms(programs);
  return (
    <div className="planner-day-track" role="group">
      {Array.from({ length: 25 }, (_, hour) => (
        <i className="planner-hour-line" key={hour} style={{ '--planner-hour': hour }} aria-hidden="true" />
      ))}
      {packed.items.map((program, index) => (
        <DayTimelineProgram
          key={`${programKey(channel, program)}-${index}`}
          program={{ ...program, _plannerLanes: packed.lanes }}
          channel={channel}
          locale={locale}
          selected={programKey(channel, program) === selectedProgramKey}
          showPrograms={showPrograms}
          showBreaks={showBreaks}
          onSelect={onSelectProgram}
        />
      ))}
    </div>
  );
}

function DayTimelineCanvas({ rows, columns, locale, dayEvents, showPrograms, showBreaks, selectedProgramKey, onSelectProgram }) {
  return (
    <div className="planner-day-scroll">
      {rows.map((row, rowIndex) => {
        const programs = Array.isArray(row.programs) ? row.programs : [];
        const channel = String(row.channel || pageText(locale, 'Channel', 'ערוץ'));
        const columnWidths = columns.map((column) => Math.max(
          154,
          packPlannerPrograms(programsForPlannerColumn(programs, column, 'day')).lanes * 44,
        ));
        const minWidth = 84 + columnWidths.reduce((sum, width) => sum + width, 0);
        const gridTemplateColumns = `84px ${columnWidths.map((width) => `minmax(${width}px, 1fr)`).join(' ')}`;
        return (
          <section className="planner-channel" key={`${channel}-${rowIndex}`} style={{ minWidth }}>
            <header className="planner-channel-head">
              <strong><bdi>{channel}</bdi></strong>
              <span>{formatNumber(programs.length, locale)} {pageText(locale, 'programme rows', 'שורות תוכנית')}</span>
            </header>
            <div className="planner-day-grid" style={{ gridTemplateColumns }}>
              <div className="planner-time-head">{pageText(locale, 'Clock', 'שעה')}</div>
              {columns.map((column) => {
                const eventNames = dayEvents ? dayEvents[column.key] : null;
                return (
                  <div className={`planner-day-head${column.weekend ? ' is-weekend' : ''}`} key={column.key}>
                    <strong>{column.label}</strong>
                    {Array.isArray(eventNames) && eventNames.length > 0 ? <span>{eventNames.join(', ')}</span> : null}
                  </div>
                );
              })}
              <div className="planner-time-axis" aria-hidden="true">
                {Array.from({ length: 25 }, (_, hour) => (
                  <span key={hour} style={{ '--planner-hour': hour }}><Numeric>{`${String(hour % 24).padStart(2, '0')}:00`}</Numeric></span>
                ))}
              </div>
              {columns.map((column) => (
                <DayTimelineColumn
                  key={`${channel}-${column.key}`}
                  programs={programsForPlannerColumn(programs, column, 'day')}
                  channel={channel}
                  locale={locale}
                  selectedProgramKey={selectedProgramKey}
                  showPrograms={showPrograms}
                  showBreaks={showBreaks}
                  onSelectProgram={onSelectProgram}
                />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}

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
  if (axis === 'day') {
    return (
      <div className="planning-canvas planning-canvas-timeline">
        <div className="planner-instrument-legend">
          <span>{pageText(locale, '24-hour transmission grid', 'רשת שידור של 24 שעות')}</span>
          <span>{pageText(locale, 'Overlapping source rows are packed into parallel lanes. Focus a programme for its full detail.', 'שורות מקור חופפות נארזות לנתיבים מקבילים. מיקוד בתוכנית פותח את מלוא הפרטים.')}</span>
        </div>
        <DayTimelineCanvas
          rows={rows}
          columns={columns}
          locale={locale}
          dayEvents={dayEvents}
          showPrograms={showPrograms}
          showBreaks={showBreaks}
          selectedProgramKey={selectedProgramKey}
          onSelectProgram={onSelectProgram}
        />
      </div>
    );
  }
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
