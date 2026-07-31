import React from 'react';
import { Lock } from 'lucide-react';
import { secondsToClock, humanOffset } from './schedule-editor-format';
import BreakChip from '../break/BreakChip';

// One draggable break chip on the editor timeline. This is a presentation shell:
// the drag, resize and keyboard handlers all live in the editor and are passed in
// unchanged, and positionStyle is the editor's own second to percent mapping. The
// chip shares its visual language with the read-only timeline chip through the
// break-chip base class and the BreakChip body, showing the exact clock second,
// the human offset into the programme and the break length, with the clock kept
// left to right so a time is never mirrored in a right to left layout.
function ScheduleEditorBreak({
  item,
  laneKey,
  startSec,
  durationSec,
  offsetSeconds,
  edited,
  pinned,
  locale,
  positionStyle,
  onMovePointerDown,
  onResizePointerDown,
  onKeyDown,
}) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const className = [
    'break-chip',
    'editor-break',
    pinned ? 'pinned' : '',
    edited && !pinned ? 'unsaved' : '',
    item.is_gold ? 'gold' : '',
  ].filter(Boolean).join(' ');
  const clock = secondsToClock(startSec);
  const offsetText = humanOffset(offsetSeconds, locale);
  const intoTitle = `${clock}, ${offsetText} ${label('into', 'בתוך')} ${item.program_title}`;
  const seconds = Math.round(durationSec);

  return (
    // The native title stays: it only echoes the chip's own clock/offset/length
    // identity (plus the programme name behind it), not an explanation, and a
    // hover-managed MUI tooltip would fight the pointer-capture drag handlers.
    <div
      className={className}
      role="button"
      tabIndex={0}
      style={positionStyle(startSec, startSec + durationSec)}
      title={`${intoTitle} / ${seconds}s`}
      onPointerDown={(event) => onMovePointerDown(event, laneKey, item)}
      onKeyDown={(event) => onKeyDown(event, laneKey, item)}
      aria-label={`${intoTitle} ${seconds} ${label('seconds', 'שניות')}`}
    >
      {pinned && <Lock className="editor-break-lock" size={12} />}
      <BreakChip clock={clock} detail={offsetText} meta={`${seconds}s`} />
      <i
        className="editor-break-resize"
        onPointerDown={(event) => onResizePointerDown(event, laneKey, item)}
        aria-hidden="true"
      />
    </div>
  );
}

export default ScheduleEditorBreak;
