import React, { useId } from 'react';
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import { Button, IconButton } from '../../studio/actions';
import { ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react';
import { pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import { clockOf } from './day-board-model';
import './day-break-navigator.css';

// A timeline is allowed to draw a two-minute break at its true width. It is not
// allowed to make that twelve-pixel rectangle the only way to reach the break.
// This compact navigator is the non-overlapping, 44px activation surface for
// every object on the track. The track remains a precise direct-manipulation
// canvas for pointer users; this control owns selection and keyboard editing.
export default function DayBreakNavigator({ breaks, programmes, liveOf, selected, onSelect, onOpen, onKeyDown, locale }) {
  const hintId = useId();
  const selectLabelId = useId();
  const selectInputId = useId();
  const selectLabel = pageText(locale, 'Selected break', 'הברייק הנבחר');
  const rows = (breaks || []).map((item) => {
    const programme = programmes.get(item.segment_id);
    const live = liveOf(item);
    const start = (programme?.start_seconds || 0) + live.offsetSeconds;
    return {
      item,
      clock: clockOf(start),
      seconds: Math.round(live.durationSeconds),
      programme: item.programme || programme?.title || '',
    };
  });
  const index = rows.findIndex((row) => row.item.break_id === selected);
  const active = index >= 0 ? rows[index] : null;
  const previous = index > 0 ? rows[index - 1] : null;
  const next = index >= 0 && index < rows.length - 1 ? rows[index + 1] : null;

  function choose(id) {
    if (id) onSelect(id);
  }

  return (
    <div className="day-break-navigator" aria-label={pageText(locale, 'Break selection and keyboard editing', 'בחירת ברייק ועריכה במקלדת')}>
      <FormControl className="day-break-select" size="small">
        <InputLabel id={selectLabelId} htmlFor={selectInputId}>{selectLabel}</InputLabel>
        <Select
          labelId={selectLabelId}
          value={selected || ''}
          label={selectLabel}
          inputProps={{ id: selectInputId, 'aria-label': selectLabel }}
          onChange={(event) => choose(event.target.value)}
        >
          <MenuItem value="" disabled>{pageText(locale, `Choose from ${rows.length} breaks`, `בחירה מתוך ${rows.length} ברייקים`)}</MenuItem>
          {rows.map((row, rowIndex) => (
            <MenuItem value={row.item.break_id} key={row.item.break_id}>
              {`${rowIndex + 1}. ${row.clock} · ${row.seconds}s · ${row.programme}`}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <div className="day-break-walk" role="group" aria-label={pageText(locale, 'Walk breaks in timeline order', 'מעבר בין ברייקים לפי סדר ציר הזמן')}>
        <IconButton
          type="button"
          disabled={!previous}
          onClick={() => choose(previous?.item.break_id)}
          aria-label={pageText(locale, 'Previous break', 'הברייק הקודם')}
        >
          <ChevronRight size={20} aria-hidden="true" />
        </IconButton>
        <span aria-live="polite"><Figure>{index >= 0 ? index + 1 : 0}</Figure> / <Figure>{rows.length}</Figure></span>
        <IconButton
          type="button"
          disabled={!next}
          onClick={() => choose(next?.item.break_id)}
          aria-label={pageText(locale, 'Next break', 'הברייק הבא')}
        >
          <ChevronLeft size={20} aria-hidden="true" />
        </IconButton>
      </div>
      <Button
        type="button"
        className="day-break-open-proxy"
        disabled={!active}
        aria-describedby={hintId}
        onClick={() => active && onOpen(active.item.break_id)}
        onKeyDown={(event) => active && onKeyDown(event, active.item)}
      >
        <ExternalLink size={18} aria-hidden="true" />
        {active ? (
          <span>
            {pageText(locale, 'Open', 'פתיחה')} <Name>{active.programme}</Name> · <Figure>{active.clock}</Figure>
          </span>
        ) : pageText(locale, 'Open break detail', 'פתיחת פרטי הברייק')}
      </Button>
      <p id={hintId} className="day-break-key-hint">
        {pageText(
          locale,
          'On Open: arrow keys move; Up and Down resize; Shift moves five steps; Alt moves one second; G toggles gold.',
          'בכפתור פתיחה: החצים מזיזים; מעלה ומטה משנים אורך; Shift מזיז חמש יחידות; Alt מזיז שנייה; G מחליף מצב זהב.',
        )}
      </p>
    </div>
  );
}
