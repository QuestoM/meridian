import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { Minus, Plus } from 'lucide-react';
import {
  LANE_GUTTER,
  MIN_PX_PER_MIN,
  MAX_PX_PER_MIN,
  DEFAULT_PX_PER_MIN,
  clampZoom,
  buildTicks,
  trackWidth,
  pixelToMinute,
} from './schedule-track';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

// Shared zoom state for the schedule track. Held in component state per page
// visit so switching between timeline and editor keeps the same scale; nothing
// is persisted to storage. A step multiplies or divides the current factor so
// zoom feels even across the wide band.
export function useScheduleZoom(initial = DEFAULT_PX_PER_MIN) {
  const [pxPerMin, setPxPerMin] = useState(() => clampZoom(initial));
  const zoomBy = useCallback((factor) => {
    setPxPerMin((current) => clampZoom(current * factor));
  }, []);
  const setZoom = useCallback((value) => {
    setPxPerMin(clampZoom(value));
  }, []);
  return { pxPerMin, setZoom, zoomBy };
}

// Owned-channel segment anchors, fetched once and keyed by channel|date|
// start_clock, so a click on a programme resolves to its addressable segment.
// resolve returns the segment record or null; the caller decides how to notify
// when a programme is not on the owned channel.
export function useSegmentAnchors() {
  const [segMap, setSegMap] = useState(() => new Map());
  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/schedule/segments`)
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (!active || !payload) return;
        const map = new Map();
        (payload.segments || []).forEach((seg) => {
          const a = seg.anchor || {};
          const key = `${seg.channel || ''}|${a.date || seg.day || ''}|${a.start_clock || ''}`;
          map.set(key, { segmentId: seg.segment_id, channel: seg.channel, day: seg.day });
        });
        setSegMap(map);
      })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  const resolve = useCallback((channel, date, startClock) => {
    const key = `${channel || ''}|${date || ''}|${startClock || ''}`;
    const hit = segMap.get(key);
    return hit && hit.segmentId ? hit : null;
  }, [segMap]);

  return { segMap, resolve };
}

// The video-editor zoom control: a compact slider flanked by minus and plus
// buttons. It scales the minutes-to-pixels factor shared by both views. The
// readout shows the current scale relative to the base so the operator has a
// concrete sense of how far in they are.
export function ZoomControl({ pxPerMin, onZoom, onStep, locale }) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const relative = pxPerMin / DEFAULT_PX_PER_MIN;
  const relativeText = relative >= 1 ? `${relative.toFixed(1)}x` : `${relative.toFixed(2)}x`;
  return (
    <div className="track-zoom" role="group" aria-label={label('Zoom the time scale', 'שינוי קנה המידה של הזמן')}>
      <span className="track-zoom-label">{label('Zoom', 'זום')}</span>
      <Button
        type="button"
        variant="outlined"
        className="track-zoom-btn"
        aria-label={label('Zoom out', 'הקטנה')}
        onClick={() => onStep(1 / 1.4)}
      >
        <Minus size={14} />
      </Button>
      <input
        className="track-zoom-slider"
        type="range"
        min={MIN_PX_PER_MIN}
        max={MAX_PX_PER_MIN}
        step={0.1}
        value={pxPerMin}
        onChange={(event) => onZoom(Number(event.target.value))}
        aria-label={label('Time scale', 'קנה מידה של הזמן')}
      />
      <Button
        type="button"
        variant="outlined"
        className="track-zoom-btn"
        aria-label={label('Zoom in', 'הגדלה')}
        onClick={() => onStep(1.4)}
      >
        <Plus size={14} />
      </Button>
      <span className="track-zoom-readout" dir="ltr">{relativeText}</span>
    </div>
  );
}

// The scrolling track surface shared by both views: a sticky hour ruler with
// adaptive fine ticks over a horizontal scroll container that holds the lane
// rows. Ctrl or cmd plus wheel zooms centered on the cursor by keeping the time
// under the pointer fixed while the scale changes. The ruler and every row are
// sized to the same track width so they stay aligned at any zoom.
export function ScheduleTrackSurface({ axis, pxPerMin, onZoom, children, locale }) {
  const scrollRef = useRef(null);
  const width = trackWidth(axis, pxPerMin);
  const minWidth = LANE_GUTTER + width;
  const ticks = buildTicks(axis, pxPerMin);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return undefined;
    function onWheel(event) {
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      const rect = node.getBoundingClientRect();
      // Cursor position within the track area, past the fixed lane gutter.
      const cursorInTrack = event.clientX - rect.left - LANE_GUTTER + node.scrollLeft;
      const minuteUnderCursor = pixelToMinute(axis, pxPerMin, Math.max(0, cursorInTrack));
      const next = clampZoom(pxPerMin * (event.deltaY < 0 ? 1.12 : 1 / 1.12));
      if (next === pxPerMin) return;
      onZoom(next);
      // Keep the time under the cursor put: recompute where that minute lands at
      // the new scale and shift scrollLeft to match, after the paint.
      requestAnimationFrame(() => {
        const startMin = axis.startHour * 60;
        const targetLeft = (minuteUnderCursor - startMin) * next;
        node.scrollLeft = targetLeft - (event.clientX - rect.left - LANE_GUTTER);
      });
    }
    node.addEventListener('wheel', onWheel, { passive: false });
    return () => node.removeEventListener('wheel', onWheel);
  }, [axis, pxPerMin, onZoom]);

  return (
    <div className="timeline-scroll chart-ltr" dir="ltr" ref={scrollRef}>
      <div className="timeline-ruler" style={{ minWidth }}>
        <span className="timeline-ruler-gutter" />
        <div className="timeline-hours" style={{ width }}>
          {ticks.map((tick) => (
            <span
              key={tick.minute}
              className={tick.major ? 'timeline-tick major' : 'timeline-tick'}
              style={{ left: `${tick.left}px` }}
            >
              <i aria-hidden="true" />
              {tick.major && <b>{tick.label}</b>}
            </span>
          ))}
        </div>
      </div>
      {children({ width, minWidth, ticks })}
    </div>
  );
}

// One legible programme band, the shared visual language for both views: title,
// class and clock window in the newer editor hierarchy. The editor passes drag
// affordances as children; the timeline leaves it read-only. When clickable it
// becomes a real button that opens the inspector.
export function ProgrammeBand({
  title,
  classLabel,
  windowText,
  style,
  clickable,
  onOpen,
  children,
}) {
  const common = {
    className: `timeline-program-band${clickable ? ' timeline-program-clickable' : ''}`,
    style,
    title: `${title || ''}${windowText ? ` / ${windowText}` : ''}`,
  };
  const body = (
    <>
      <span className="timeline-program-title">{title}</span>
      <span className="timeline-program-meta">
        {classLabel && <span className="timeline-program-class">{classLabel}</span>}
        {windowText && <span className="timeline-program-window" dir="ltr">{windowText}</span>}
      </span>
      {children}
    </>
  );
  if (clickable) {
    return (
      <div
        {...common}
        role="button"
        tabIndex={0}
        onClick={onOpen}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onOpen();
          }
        }}
      >
        {body}
      </div>
    );
  }
  return <div {...common}>{body}</div>;
}
