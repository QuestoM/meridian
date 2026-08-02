// Shared time-axis math for the schedule track surface, used by both the
// read-only timeline and the drag-and-drop editor so the two views share one
// coordinate system. The single source of truth is pixels-per-minute: the track
// width is totalMinutes multiplied by that factor, and every band or chip is
// placed at (minuteOfDay minus startMinute) times the factor. Because the hour
// ruler uses the exact same mapping, ruler and tracks stay aligned at any zoom.

// Zoom is expressed as pixels-per-minute so the video-editor slider scales the
// horizontal time scale directly. The stops span a wide, legible band from a
// compressed overview to a second-precise close-up.
export const MIN_PX_PER_MIN = 1.9;
export const MAX_PX_PER_MIN = 30;
export const DEFAULT_PX_PER_MIN = 3.8;

// The lane label gutter width in pixels, shared so the ruler and every row line
// up under the same first column.
export const LANE_GUTTER = 172;

// Clamp a pixels-per-minute value to the supported zoom band.
//
// The floor is a parameter rather than a constant so a surface that has measured
// a span too wide for the band can open it. A day is 26 hours and the widest
// supported scale draws it 2,964 px wide against 978 px of visible track, so a
// fit control that could not go below the floor could not fit the day. Every
// caller that passes nothing gets exactly the band it had.
export function clampZoom(value, floor = MIN_PX_PER_MIN) {
  if (!Number.isFinite(value)) return DEFAULT_PX_PER_MIN;
  return Math.max(floor, Math.min(MAX_PX_PER_MIN, value));
}

// The scale at which a span of minutes exactly fills the width on screen. The
// inverse of trackWidth below, and the honest form of a fit control: it is
// computed from the span and the width actually measured, never from a step.
export function fitZoom(totalMinutes, availableWidth) {
  const minutes = Math.max(1, Number(totalMinutes) || 0);
  const width = Math.max(1, Number(availableWidth) || 0);
  return width / minutes;
}

// Derive the visible hour window from a set of minute-of-day values, padded by
// half an hour on each side and floored to whole hours, with a sane fallback
// window when there is nothing to show yet.
export function timeWindow(minuteValues, fallbackStartHour = 18, fallbackEndHour = 24) {
  const finite = minuteValues.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    const startHour = fallbackStartHour;
    const endHour = Math.max(startHour + 4, fallbackEndHour);
    return buildAxis(startHour, endHour);
  }
  const startHour = Math.max(0, Math.floor((Math.min(...finite) - 30) / 60));
  const endHour = Math.min(28, Math.max(startHour + 4, Math.ceil((Math.max(...finite) + 30) / 60)));
  return buildAxis(startHour, endHour);
}

function buildAxis(startHour, endHour) {
  const totalMinutes = Math.max(60, (endHour - startHour) * 60);
  const hours = Array.from({ length: endHour - startHour + 1 }, (_, index) => startHour + index);
  return { startHour, endHour, totalMinutes, hours };
}

// The tick step in minutes for the current zoom. Coarser when compressed (hour
// lines only), finer as the operator zooms in so the exact start and end of a
// break become readable: 30 minute, then 15, then 5, then 1 minute ticks.
// Below the scale's own floor an hour line stops having room for its own clock,
// so the ruler steps to three hours rather than printing labels into each other.
// The threshold is the 46 px the labeller below already calls readable.
export function tickStep(pxPerMin) {
  if (pxPerMin >= 18) return 1;
  if (pxPerMin >= 9) return 5;
  if (pxPerMin >= 5) return 15;
  if (pxPerMin >= 3) return 30;
  if (pxPerMin * 60 >= 46) return 60;
  return 180;
}

// Whether a given minute tick should carry a clock label. Labels are shown on
// every tick once they are far enough apart to read, otherwise only on the hour
// (and half hour when there is room) so the ruler never turns into a smear.
export function tickIsLabelled(minuteOfDay, pxPerMin, step) {
  const spacingPx = step * pxPerMin;
  if (spacingPx >= 46) return true;
  if (minuteOfDay % 60 === 0) return true;
  if (spacingPx >= 24 && minuteOfDay % 30 === 0) return true;
  return false;
}

// Build the tick list for the ruler: every step-minute mark inside the window,
// each carrying its absolute minute, a left pixel offset, its clock label and
// whether it is a labelled (major) tick.
export function buildTicks(axis, pxPerMin) {
  const step = tickStep(pxPerMin);
  const startMin = axis.startHour * 60;
  const endMin = axis.endHour * 60;
  const ticks = [];
  for (let minute = startMin; minute <= endMin; minute += step) {
    ticks.push({
      minute,
      left: (minute - startMin) * pxPerMin,
      major: tickIsLabelled(minute, pxPerMin, step) || minute % 60 === 0,
      label: minuteClock(minute),
    });
  }
  return ticks;
}

// The pixel width of the full track for a window at a given zoom.
export function trackWidth(axis, pxPerMin) {
  return Math.max(680, axis.totalMinutes * pxPerMin);
}

// Place a [startMinute, endMinute] span as absolute left and width in pixels,
// clamped so a zero-length span still shows a hair-width sliver.
export function spanStyle(axis, pxPerMin, startMinute, endMinute) {
  const startMin = axis.startHour * 60;
  const safeStart = Math.max(startMin, startMinute);
  const safeEnd = Math.max(safeStart + 0.25, endMinute);
  return {
    left: `${(safeStart - startMin) * pxPerMin}px`,
    width: `${Math.max(2, (safeEnd - safeStart) * pxPerMin)}px`,
  };
}

// Invert the axis mapping: a client x within a track (already offset by the
// track's left edge) maps back to an absolute minute of day.
export function pixelToMinute(axis, pxPerMin, offsetX) {
  return axis.startHour * 60 + offsetX / Math.max(0.0001, pxPerMin);
}

// A whole-minute value rendered as an HH:MM clock, wrapping past midnight.
export function minuteClock(minute) {
  const safe = Math.max(0, Math.round(minute));
  const hour = Math.floor(safe / 60) % 24;
  const min = safe % 60;
  return `${pad(hour)}:${pad(min)}`;
}

function pad(value) {
  return String(value).padStart(2, '0');
}
