import React from 'react';
import { Lock } from 'lucide-react';
import { secondsToClock, humanOffset } from './schedule-editor-format';

// One draggable break chip on the editor timeline. This is a presentation shell:
// the drag, resize and keyboard handlers all live in the editor and are passed in
// unchanged, and positionStyle is the editor's own second to percent mapping. The
// chip's job is legibility, showing the exact clock second, the human offset into
// the programme and the break length, with the clock kept left to right so a time
// is never mirrored in a right to left layout.
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
      <span className="editor-break-clock" dir="ltr">{clock}</span>
      <strong className="editor-break-offset">{offsetText}</strong>
      <em className="editor-break-length" dir="ltr">{seconds}s</em>
      <i
        className="editor-break-resize"
        onPointerDown={(event) => onResizePointerDown(event, laneKey, item)}
        aria-hidden="true"
      />
    </div>
  );
}

export default ScheduleEditorBreak;
