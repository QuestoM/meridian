import React from 'react';
import { Button } from '@mui/material';
import { Save } from 'lucide-react';
import {
  secondsToClock,
  humanOffset,
  windowRange,
  programClassLabel,
  breakPositionLabel,
} from './schedule-editor-format';

// One legible row in the editor readout. It restates a break the operator has
// moved: the programme it sits inside (prominent), the exact clock second, the
// human offset, the class, the programme window and the break length. Clock and
// second values stay left to right inside the right to left card so a time is
// never mirrored. Every field is passed in from the editor's real lane model;
// this component computes no new values beyond formatting.
function ScheduleEditorRow({
  item,
  startSec,
  durationSec,
  offsetSeconds,
  pinned,
  saving,
  locale,
  onSave,
}) {
  const he = locale === 'he';
  const label = (en, heText) => (he ? heText : en);
  const classLabel = programClassLabel(item.program_type, locale);
  const range = windowRange(item.program && item.program.start_time, item.program && item.program.end_time);
  const intoLabel = `${humanOffset(offsetSeconds, locale)} ${label('into', 'בתוך')} ${item.program_title}`;
  const positionLabel = breakPositionLabel(item.position, locale);
  const sequence = item.break_num_in_program && item.breaks_in_program
    ? `${item.break_num_in_program} ${label('of', 'מתוך')} ${item.breaks_in_program}`
    : '';

  return (
    <li className={pinned ? 'is-pinned' : 'is-unsaved'}>
      <div className="editor-row-detail">
        <strong className="editor-row-title">{item.program_title}</strong>
        <div className="editor-row-meta">
          {classLabel && <span className="editor-row-class">{classLabel}</span>}
          {range && (
            <span className="editor-row-window" dir="ltr">{range}</span>
          )}
          {positionLabel && <span className="editor-row-slot">{positionLabel}</span>}
          {sequence && <span className="editor-row-slot" dir="ltr">{sequence}</span>}
        </div>
        <div className="editor-row-position">
          <span className="editor-row-clock" dir="ltr">{secondsToClock(startSec)}</span>
          <span className="editor-row-into">{intoLabel}</span>
        </div>
        <span className="editor-row-length">
          {label('Length', 'אורך')} <span dir="ltr">{Math.round(durationSec)}s</span>
        </span>
      </div>
      <Button
        type="button"
        variant="contained"
        className="run-button compact"
        disabled={saving}
        onClick={onSave}
      >
        <Save size={13} />
        {pinned ? label('Update pin', 'עדכון נעיצה') : label('Save as pin', 'שמור כנעיצה')}
      </Button>
    </li>
  );
}

export default ScheduleEditorRow;
