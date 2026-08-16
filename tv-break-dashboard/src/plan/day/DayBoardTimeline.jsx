import React from 'react';
import { Button } from '../../studio/actions';
import { pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import { ScheduleTrackSurface, ProgrammeBand } from './schedule-track-view';
import { spanStyle } from './schedule-track';
import DayBoardChip from './DayBoardChip';
import { startSecondsOf } from './day-board-model';
import { LIVE_PLAN, breakCountText, planBasisLabel } from './plan-basis';

export default function DayBoardTimeline({
  board,
  breaks,
  programmes,
  liveOf,
  selected,
  locale,
  axis,
  pxPerMin,
  floor,
  onZoom,
  trackRef,
  snapMark,
  liveDay,
  transmissionCursorSeconds,
  transmissionCursorClock,
  onOpenProgramme,
  onOpenBreak,
  onMovePointerDown,
  onResizePointerDown,
}) {
  const positionStyle = (startSec, endSec) => spanStyle(axis, pxPerMin, startSec / 60, endSec / 60);
  const programmeBands = (board.programmes || []).map((programme) => {
    const geometry = positionStyle(programme.start_seconds, programme.end_seconds);
    return { programme, geometry, widthPx: Number.parseFloat(geometry.width) || 0 };
  });
  const shortProgrammes = programmeBands.filter(({ widthPx }) => widthPx < 44);
  return (
    <>
      <ScheduleTrackSurface axis={axis} pxPerMin={pxPerMin} onZoom={onZoom} locale={locale} floor={floor}>
        {({ width, minWidth, ticks }) => (
          <div className="day-track-row" style={{ minWidth }}>
            <div className="day-track-lane">
              <strong><Name>{board.operator_channel}</Name></strong><Figure>{board.day}</Figure>
              <span className="timeline-lane-count"><span>{breakCountText(breaks.length, locale)}</span><small className="timeline-lane-basis">{planBasisLabel(LIVE_PLAN, locale)}</small></span>
            </div>
            <div className="day-track" style={{ width }} ref={trackRef}>
              {ticks.filter((tick) => tick.major).map((tick) => (
                <i className="day-track-tick" key={tick.minute} style={{ left: `${tick.left}px` }} />
              ))}
              {transmissionCursorSeconds !== null && (
                <div
                  className={`day-transmission-cursor${liveDay ? ' is-live' : ' is-selection'}`}
                  style={{ left: spanStyle(axis, pxPerMin, transmissionCursorSeconds / 60, transmissionCursorSeconds / 60).left }}
                  role="status"
                  aria-label={liveDay
                    ? pageText(locale, `Live transmission time ${transmissionCursorClock}`, `זמן שידור חי ${transmissionCursorClock}`)
                    : pageText(locale, `Selected break at ${transmissionCursorClock}`, `ברייק נבחר בשעה ${transmissionCursorClock}`)}
                >
                  <span>{liveDay ? pageText(locale, 'LIVE', 'חי') : pageText(locale, 'SELECTED', 'נבחר')}</span>
                  <b><Figure>{transmissionCursorClock}</Figure></b>
                </div>
              )}
              {programmeBands.map(({ programme, geometry, widthPx }) => (
                <ProgrammeBand
                  key={programme.segment_id}
                  title={programme.title}
                  classLabel={programme.genre}
                  windowText={`${Math.round(programme.duration_seconds / 60)}m`}
                  style={geometry}
                  clickable={Boolean(onOpenProgramme) && widthPx >= 44}
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
                    onMovePointerDown={onMovePointerDown}
                    onResizePointerDown={onResizePointerDown}
                    onOpen={onOpenBreak}
                  />
                );
              })}
            </div>
          </div>
        )}
      </ScheduleTrackSurface>
      {onOpenProgramme && shortProgrammes.length > 0 ? (
        <nav className="day-short-programmes" aria-label={pageText(locale, 'Short programmes in this day', 'תוכניות קצרות ביום הזה')}>
          <strong>{pageText(locale, 'Short programmes', 'תוכניות קצרות')}</strong>
          <div>
            {shortProgrammes.map(({ programme }) => (
              <Button key={programme.segment_id} type="button" variant="outlined" onClick={() => onOpenProgramme(programme)}>
                <Name>{programme.title}</Name>
                <Figure>{`${Math.round(programme.duration_seconds / 60)}m`}</Figure>
              </Button>
            ))}
          </div>
        </nav>
      ) : null}
    </>
  );
}
