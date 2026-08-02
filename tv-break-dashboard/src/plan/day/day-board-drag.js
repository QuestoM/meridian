// The two direct manipulations on the day track: moving a break and changing
// its length. Split out of DayBoard.jsx under the 450-line law.
//
// Both are built as one factory so the two gestures share exactly one coordinate
// inversion, one snap and one history entry shape. A drag that ends where it
// began records nothing, because an accidental round trip is not a decision.
//
// The signifier while the gesture is in flight is a snap line at the edge being
// snapped to, which is the device a professional editing timeline uses: the
// exact place the object will land is drawn before the pointer is released.

import { MIN_DURATION_SECONDS, applyEdit, maxDurationFor, offsetBounds, snapTo } from './day-board-model';
import { pixelToMinute } from './schedule-track';

export function createDragHandlers({
  trackRef, axis, pxPerMinRef, snapGridRef, programmes, liveOf,
  setEdits, setSelected, setSnapMark, pushHistory,
}) {
  function pixelToOffset(clientX, programme, durationSeconds, fallback) {
    const node = trackRef.current;
    if (!node || !programme) return fallback;
    const rect = node.getBoundingClientRect();
    const absolute = pixelToMinute(axis, pxPerMinRef.current, clientX - rect.left + node.scrollLeft) * 60;
    const bounds = offsetBounds(programme, durationSeconds);
    return snapTo(absolute - programme.start_seconds, snapGridRef.current, bounds.min, bounds.max);
  }

  function begin(event, item, focusTarget) {
    event.preventDefault();
    event.stopPropagation();
    // Selecting with the pointer hands the keyboard the same break, so the
    // arrows, the length field, gold and Enter all act on what was just
    // touched. Without this a drag leaves the keyboard pointing at nothing.
    if (focusTarget) focusTarget.focus();
    setSelected(item.break_id);
    return { programme: programmes.get(item.segment_id), start: liveOf(item) };
  }

  function finish(item, start, latest, name) {
    setSnapMark(null);
    const moved = Math.abs(latest.offsetSeconds - start.offsetSeconds) > 0.001
      || Math.abs(latest.durationSeconds - start.durationSeconds) > 0.001;
    if (!moved) return;
    pushHistory({ type: 'edit', name, breakId: item.break_id, before: start, after: latest });
  }

  function onMovePointerDown(event, item) {
    const { programme, start } = begin(event, item, event.currentTarget);
    let latest = start;
    function move(moveEvent) {
      const offsetSeconds = pixelToOffset(moveEvent.clientX, programme, start.durationSeconds, start.offsetSeconds);
      latest = { ...start, offsetSeconds };
      setSnapMark(programme ? programme.start_seconds + offsetSeconds : null);
      setEdits((current) => applyEdit(current, item, latest));
    }
    function up() {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      finish(item, start, latest, 'move');
    }
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  function onResizePointerDown(event, item) {
    const chip = event.currentTarget.closest('.day-chip');
    const { programme, start } = begin(event, item, chip);
    let latest = start;
    function move(moveEvent) {
      const node = trackRef.current;
      if (!node || !programme) return;
      const rect = node.getBoundingClientRect();
      const absolute = pixelToMinute(axis, pxPerMinRef.current, moveEvent.clientX - rect.left + node.scrollLeft) * 60;
      const chipStart = programme.start_seconds + start.offsetSeconds;
      const raw = Math.max(MIN_DURATION_SECONDS, absolute - chipStart);
      const durationSeconds = snapTo(
        raw, snapGridRef.current, MIN_DURATION_SECONDS, maxDurationFor(programme, start.offsetSeconds),
      );
      latest = { ...start, durationSeconds };
      setSnapMark(chipStart + durationSeconds);
      setEdits((current) => applyEdit(current, item, latest));
    }
    function up() {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      finish(item, start, latest, 'length');
    }
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  return { onMovePointerDown, onResizePointerDown };
}
