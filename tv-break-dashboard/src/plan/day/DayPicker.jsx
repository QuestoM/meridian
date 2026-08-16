import React, { useState } from 'react';
import { Popover } from '@mui/material';
import { Button } from '../../studio/actions';
import { CalendarDays } from 'lucide-react';
import { pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import { formatDay, formatMonthTitle, parseDay } from '../../shell/dates';
import {
  WEEKDAY_NAMES_EN,
  WEEKDAY_NAMES_HE,
  isWeekend,
  weekdayIndex,
  weeksOf,
} from './day-board-model';

// The Israeli week, presented Sunday first while the data stays ISO-keyed.
//
// The days come from the plan itself, so this picker can never offer a day the
// operator has no plan for. Friday and Saturday are the weekend and are marked
// as such; nothing else about them is different, because the plan treats them as
// ordinary broadcast days and pretending otherwise would be a fabrication.
const DAY_NAMES_HE = WEEKDAY_NAMES_HE;
const DAY_NAMES_EN = WEEKDAY_NAMES_EN;
const DAY_SHORT_HE = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ש'];
const DAY_SHORT_EN = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];

function DayPicker({ days, value, onChange, locale, channel, windowed = false }) {
  const he = locale === 'he';
  const weeks = weeksOf(days);
  const [monthAnchor, setMonthAnchor] = useState(null);
  if (!days || !days.length) return null;
  const selectedWeek = weeks.find((week) => week.days.includes(value)) || weeks[0];
  const visibleDays = windowed ? days.slice(0, 7) : selectedWeek.days;
  const selectedParts = parseDay(value || days[0]);
  const monthTitle = selectedParts ? formatMonthTitle(selectedParts.year, selectedParts.month, locale) : pageText(locale, 'All dates', 'כל התאריכים');

  function DayButton({ iso }) {
    const index = weekdayIndex(iso);
    const active = iso === value;
    const names = he ? DAY_NAMES_HE : DAY_NAMES_EN;
    const shorts = he ? DAY_SHORT_HE : DAY_SHORT_EN;
    return (
      <Button
        key={iso}
        type="button"
        variant="text"
        className={`day-pill${active ? ' is-on' : ''}${isWeekend(iso) ? ' is-weekend' : ''}`}
        aria-pressed={active}
        aria-label={`${names[index] || ''} ${formatDay(iso)}`}
        onClick={() => {
          onChange(iso);
          setMonthAnchor(null);
        }}
      >
        <b>{shorts[index] || ''}</b>
        <Figure>{iso.slice(8)}</Figure>
      </Button>
    );
  }

  return (
    <div className="card day-picker">
      <div className="day-picker-head">
        <span className="day-picker-label">{pageText(locale, 'Broadcast day', 'יום שידור')}</span>
        {channel && <Name className="day-picker-channel">{channel}</Name>}
        <Figure className="day-picker-current">{formatDay(value || days[0])}</Figure>
      </div>
      <div className="day-picker-week" aria-label={windowed ? pageText(locale, 'Plan window', 'חלון התוכנית') : pageText(locale, 'Selected week', 'השבוע הנבחר')}>
        {visibleDays.map((iso) => <DayButton key={iso} iso={iso} />)}
      </div>
      <Button
        className="day-picker-month"
        type="button"
        variant="outlined"
        aria-haspopup="dialog"
        aria-expanded={Boolean(monthAnchor)}
        onClick={(event) => setMonthAnchor(event.currentTarget)}
      >
        <CalendarDays size={16} aria-hidden="true" />
        {monthTitle}
      </Button>
      <Popover
        open={Boolean(monthAnchor)}
        anchorEl={monthAnchor}
        onClose={() => setMonthAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'end' }}
        transformOrigin={{ vertical: 'top', horizontal: 'end' }}
      >
        <div className="day-picker-popover">
          <strong>{pageText(locale, 'Available broadcast days', 'ימי שידור זמינים')}</strong>
          <span>{formatDay(value || days[0])} · {days.length}</span>
          <div className="day-picker-weeks">
            {weeks.map((week) => (
              <div className="day-picker-week" key={week.days[0]}>
                {week.days.map((iso) => <DayButton key={iso} iso={iso} />)}
              </div>
            ))}
          </div>
        </div>
      </Popover>
    </div>
  );
}

export default DayPicker;
