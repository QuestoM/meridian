import React, { useState } from 'react';
import { Button, Tooltip } from '@mui/material';
import { Numeric, formatCurrency, formatMinutes, formatNumber, pageText } from '../../shell/format';
import { programTypeLabel } from '../../shell/labels';
import { flattenScheduleRows, normalizeRows } from '../../shell/plan-model';
import {
  ProgrammeBand,
  ScheduleTrackSurface,
  ZoomControl,
  useScheduleZoom,
  useSegmentAnchors,
} from '../day/schedule-track-view';
import { spanStyle, timeWindow } from '../day/schedule-track';
import BreakChip from '../break/BreakChip';
import ScheduleInspector from '../day/ScheduleInspector';

export function timeToMinutes(time) {
  const [hour, minute] = String(time || '00:00').split(':').map((part) => Number(part));
  const safeHour = Number.isFinite(hour) ? Math.max(0, Math.min(47, hour)) : 0;
  const safeMinute = Number.isFinite(minute) ? Math.max(0, Math.min(59, minute)) : 0;
  return safeHour * 60 + safeMinute;
}

export function minutesToTime(minutes) {
  const safe = Math.max(0, Math.min(47 * 60 + 59, Math.round(minutes)));
  const hour = Math.floor(safe / 60) % 24;
  const minute = safe % 60;
  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
}

// When break-operations arrives empty, the timeline still lays out the real
// programme bands from the schedule rows, but it does NOT invent break chips:
// synthesized evenly-spaced break start times would read as real placements.
// The track stays honestly empty of breaks, and the break-derived summary
// figures render as unknown rather than confident zeros.
export function buildTimelineFallback(rows) {
  const programs = flattenScheduleRows(rows).slice(0, 24).map((program, index) => {
    const duration = Number(program.duration_minutes || 30);
    const start = timeToMinutes(program.time);
    return {
      id: `fallback-program-${index}`,
      key: program.key,
      lane: `${program.channel} / ${program.day}`,
      channel: program.channel,
      title: program.title,
      program_type: program.program_type || 'Other',
      day: program.day,
      start_time: minutesToTime(start),
      end_time: minutesToTime(start + duration),
      duration_minutes: duration,
      revenue: Number(program.revenue || 0),
      retention: Number(program.retention || 0),
      break_markers: Number(program.break_markers || 0),
    };
  });
  return {
    programs,
    breaks: [],
    summary: {
      programs: programs.length,
      breaks: 0,
      ad_seconds: null,
      revenue: null,
    },
  };
}

export function normalizedTimeline(timeline, rows) {
  const fallback = buildTimelineFallback(rows);
  const programs = normalizeRows(timeline?.programs).length ? normalizeRows(timeline.programs) : fallback.programs;
  const breaks = normalizeRows(timeline?.breaks).length ? normalizeRows(timeline.breaks) : fallback.breaks;
  const summary = timeline?.summary || fallback.summary;
  return { programs, breaks, summary };
}

export function TimelineView({ timeline, rows, locale, notify, zoom, onGlobalRefresh, selectedProgramKey, onSelectProgram }) {
  const { programs, breaks, summary } = normalizedTimeline(timeline, rows);
  const lanes = Array.from(new Set([...programs.map((item) => item.lane), ...breaks.map((item) => item.lane)].filter(Boolean)));
  const allMinutes = [
    ...programs.flatMap((item) => [timeToMinutes(item.start_time), timeToMinutes(item.end_time)]),
    ...breaks.flatMap((item) => [timeToMinutes(item.start_time), timeToMinutes(item.end_time)]),
  ].filter((value) => Number.isFinite(value));
  // Shared time axis and zoom, so the timeline and the editor line up on the
  // same hour window and pixel mapping at whatever scale is set.
  const axis = timeWindow(allMinutes.length ? allMinutes : [20 * 60, 23 * 60]);
  const localZoom = useScheduleZoom();
  const { pxPerMin, setZoom, zoomBy } = zoom || localZoom;
  const positionStyle = (startTime, endTime) => spanStyle(axis, pxPerMin, timeToMinutes(startTime), timeToMinutes(endTime));

  // Owned-channel segment anchors, resolved through the shared hook so a click
  // on a programme band opens the same inspector the editor uses. A programme
  // that is not on the owned channel has no editable segment; we say so plainly.
  const { resolve } = useSegmentAnchors();
  const [inspect, setInspect] = useState(null);
  const openInspector = (program) => {
    if (!program) return;
    const hit = resolve(program.channel, program.date, program.start_time);
    if (hit) {
      setInspect(hit);
    } else if (notify) {
      notify(
        'This programme is not on your owned channel, so it has no editable segment.',
        'התוכנית אינה בערוץ שבבעלותכם, ולכן אין לה מקטע לעריכה.',
      );
    }
  };

  return (
    <div className="timeline-view">
      <div className="timeline-topbar no-print">
        <div className="timeline-summary" dir={locale === 'he' ? 'rtl' : 'ltr'}>
          <div>
            <strong>{formatNumber(summary.programs, locale)}</strong>
            <span>{pageText(locale, 'programs on timeline', 'תוכניות בציר')}</span>
          </div>
          <div>
            <strong>{formatNumber(summary.breaks, locale)}</strong>
            <span>{pageText(locale, 'planned breaks', 'ברייקים מתוכננים')}</span>
          </div>
          <div>
            <strong><Numeric>{formatMinutes(summary.ad_seconds, locale)}</Numeric></strong>
            <span>{pageText(locale, 'commercial time', 'זמן פרסום')}</span>
          </div>
          <div>
            <strong><Numeric>{formatCurrency(summary.revenue, locale)}</Numeric></strong>
            <span>{pageText(locale, 'modelled revenue', 'הכנסה מחושבת')}</span>
          </div>
        </div>
        <ZoomControl pxPerMin={pxPerMin} onZoom={setZoom} onStep={zoomBy} locale={locale} />
      </div>

      {breaks.length === 0 && (
        <p className="data-basis-note">
          {programs.length === 0
            ? pageText(locale, 'No timeline data yet. Run a recompute to build the weekly plan.', 'אין עדיין נתוני ציר זמן. הריצו חישוב מחדש כדי לבנות את התוכנית השבועית.')
            : pageText(locale, 'No planned break data arrived for this week, so the timeline shows programme bands without break chips.', 'לא התקבלו נתוני ברייקים מתוכננים לשבוע הזה, ולכן ציר הזמן מציג רצועות תוכניות בלי סימוני ברייקים.')}
        </p>
      )}

      <ScheduleTrackSurface axis={axis} pxPerMin={pxPerMin} onZoom={setZoom} locale={locale}>
        {({ width, minWidth, ticks }) => lanes.map((lane) => {
          const lanePrograms = programs.filter((item) => item.lane === lane);
          const laneBreaks = breaks.filter((item) => item.lane === lane);
          const laneRevenue = laneBreaks.reduce((sum, item) => sum + Number(item.revenue_calculated || 0), 0);
          return (
            <div className="timeline-row" key={lane} style={{ minWidth }}>
              <div className="timeline-lane" dir={locale === 'he' ? 'rtl' : 'ltr'}>
                <strong>{lane}</strong>
                <span>{laneBreaks.length} {pageText(locale, 'breaks', 'ברייקים')} / <Numeric>{formatCurrency(laneRevenue, locale)}</Numeric></span>
              </div>
              <div className="timeline-track" style={{ width }}>
                {ticks.filter((tick) => tick.major).map((tick) => (
                  <i key={`${lane}-${tick.minute}`} style={{ left: `${tick.left}px` }} />
                ))}
                {lanePrograms.map((program) => (
                  <ProgrammeBand
                    key={program.key || `${program.title}-${program.start_time}`}
                    title={program.title}
                    classLabel={programTypeLabel(program.program_type, locale)}
                    windowText={`${program.start_time} - ${program.end_time}`}
                    style={positionStyle(program.start_time, program.end_time)}
                    clickable
                    onOpen={() => openInspector(program)}
                  />
                ))}
                {laneBreaks.map((breakItem) => {
                  const selected = selectedProgramKey === breakItem.program_key;
                  const selectedProgram = {
                    key: breakItem.program_key,
                    title: breakItem.program_title,
                    channel: breakItem.channel,
                    day: breakItem.day,
                    time: breakItem.start_time,
                    duration_minutes: Math.round(Number(breakItem.duration_sec || 0) / 60),
                    revenue: breakItem.revenue_calculated,
                    retention: breakItem.retention,
                    break_markers: breakItem.breaks_in_program,
                    program_type: breakItem.program_type,
                    selected_break: breakItem,
                  };
                  const className = [
                    'break-chip',
                    'break-chip-readonly',
                    'timeline-break',
                    selected ? 'selected' : '',
                    breakItem.status === 'at_risk' ? 'risk' : '',
                    breakItem.is_gold ? 'gold' : '',
                  ].filter(Boolean).join(' ');
                  // Anchor the chip at its start time and let the shared chip
                  // width govern legibility. A break is a fixed 120s span, so
                  // scaling the width to that duration collapses it to a few
                  // pixels at low zoom; keeping only the left keeps every chip
                  // as readable as the editor's.
                  const { left } = positionStyle(breakItem.start_time, breakItem.end_time);
                  return (
                    <Tooltip title={`${breakItem.program_title} / ${breakItem.start_time}-${breakItem.end_time}`} arrow placement="bottom" key={breakItem.id}>
                    <Button
                      className={className}
                      type="button"
                      variant="contained"
                      disableRipple
                      style={{ left }}
                      aria-pressed={selected}
                      onClick={() => onSelectProgram(selectedProgram)}
                    >
                      <BreakChip
                        clock={breakItem.start_time}
                        detail={`${breakItem.break_num_in_program}/${breakItem.breaks_in_program}`}
                        gold={Boolean(breakItem.is_gold)}
                        goldLabel={pageText(locale, 'gold', 'זהב')}
                      />
                    </Button>
                    </Tooltip>
                  );
                })}
              </div>
            </div>
          );
        })}
      </ScheduleTrackSurface>

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

export default TimelineView;
