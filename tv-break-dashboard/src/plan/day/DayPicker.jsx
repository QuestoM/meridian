import React from 'react';
import { pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import { formatDay } from '../../shell/dates';
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

function DayPicker({ days, value, onChange, locale, channel }) {
  const he = locale === 'he';
  const weeks = weeksOf(days);
  if (!days || !days.length) return null;
  return (
    <div className="day-picker">
      <div className="day-picker-head">
        <span className="day-picker-label">{pageText(locale, 'Broadcast day', 'יום שידור')}</span>
        {channel && <Name className="day-picker-channel">{channel}</Name>}
        <Figure className="day-picker-count">{days.length}</Figure>
      </div>
      <div className="day-picker-weeks">
        {weeks.map((week) => (
          <div className="day-picker-week" key={week.days[0]}>
            {week.days.map((iso) => {
              const index = weekdayIndex(iso);
              const active = iso === value;
              const names = he ? DAY_NAMES_HE : DAY_NAMES_EN;
              const shorts = he ? DAY_SHORT_HE : DAY_SHORT_EN;
              return (
                <button
                  key={iso}
                  type="button"
                  className={`day-pill${active ? ' is-on' : ''}${isWeekend(iso) ? ' is-weekend' : ''}`}
                  aria-pressed={active}
                  aria-label={`${names[index] || ''} ${formatDay(iso)}`}
                  onClick={() => onChange(iso)}
                >
                  <b>{shorts[index] || ''}</b>
                  <Figure>{iso.slice(8)}</Figure>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export default DayPicker;
