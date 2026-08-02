import { LANE_GUTTER, fitZoom } from './schedule-track';

// The two framings a day board is ever asked for, each one press away.
//
// A drawing tool publishes exactly this pair, fit the whole drawing and fit the
// selection, because a scale you can only step through is a scale you get lost
// in. Measured on רשת 13 / 2024-11-01 before these existed: reading a break's own
// numbers meant the maximum scale, 30 px per minute, where the day track is
// 46,800 px wide, about 34 screen widths, and the only routes back were five
// presses of the minus button or a drag of the slider.
//
// Both are computed from the span and the width actually on screen, so neither
// is a fixed step and neither can claim a fit it did not achieve.

// The pixels the track itself has, which is the scroll box less the lane column.
export function availableWidth(trackRef) {
  const box = scrollBox(trackRef);
  return box ? Math.max(120, box.clientWidth - LANE_GUTTER) : 0;
}

// The whole broadcast day, edge to edge. The day is 26 hours and the scale's own
// floor cannot draw it inside 978 px, so the fit lowers the floor to exactly the
// scale it measured and no further.
export function fitTheDay(axis, trackRef, fitTo) {
  const width = availableWidth(trackRef);
  if (!axis || !width) return null;
  const scale = fitZoom(axis.totalMinutes, width);
  fitTo(scale);
  const box = scrollBox(trackRef);
  if (box) window.requestAnimationFrame(() => { box.scrollLeft = 0; });
  return scale;
}

// The programme the selection sits in, which is also the window a break may be
// moved inside, so this is the frame the next act happens in.
export function fitTheProgramme(programme, trackRef, fitTo, breakId) {
  const width = availableWidth(trackRef);
  if (!programme || !width) return null;
  const scale = fitZoom(programme.duration_seconds / 60, width);
  fitTo(scale);
  window.requestAnimationFrame(() => reveal(breakId));
  return scale;
}

// Bring one break back under the eye after the scale changed. The id carries a
// pipe, a tilde and Hebrew, so it is matched against the dataset rather than
// built into a selector.
export function reveal(breakId) {
  if (!breakId || typeof document === 'undefined') return;
  const chip = [...document.querySelectorAll('.day-chip')].find((node) => node.dataset.breakId === breakId);
  if (chip) chip.scrollIntoView({ inline: 'center', block: 'nearest' });
}

function scrollBox(trackRef) {
  const track = trackRef && trackRef.current;
  return track ? track.closest('.timeline-scroll') : null;
}
