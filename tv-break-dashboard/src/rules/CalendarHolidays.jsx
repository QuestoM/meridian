import React, { useMemo, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Button } from '../studio/actions';
import { ChevronDown } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { Name } from '../shell/bidi';
import { Pressable } from '../studio/dom-controls';

// The bundled-holidays panel of the Calendar page: the read-only reference list
// grouped into per-year accordions, collapsed by default so 50+ holiday rows
// never dominate the page. Each group header carries the year, its row count
// and the import-as-events button; rows render only when a year is opened.

function HolidayYearGroup({ year, rows, locale, busy, canEdit, open, onToggle, onImport }) {
  return (
    <div className="cal-holiday-year">
      <div className="cal-holiday-year-head">
        <Pressable
          type="button"
          className="cal-holiday-year-toggle"
          aria-expanded={open}
          onClick={onToggle}
        >
          <ChevronDown size={14} className={`cal-row-caret${open ? ' open' : ''}`} aria-hidden="true" />
          <span className="bidi-figure figure-nowrap">{year}</span>
          <span className="cal-count-note">{pageText(locale, `${rows.length} rows`, `${rows.length} שורות`)}</span>
        </Pressable>
        {canEdit && (
          <Tooltip title={pageText(locale, 'Creates one event per holiday of this year, with intensity 1 until you judge it, so you can attach intensity or deactivate single rows. Holidays already in the list are skipped.', 'יוצר אירוע לכל חג בשנה הזו, עם עוצמה 1 עד שתקבעו אותה, כך שתוכלו לצרף עוצמה או להשבית שורות בודדות. חגים שכבר ברשימה מדולגים.')} arrow>
            <span>
              <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onImport}>
                {pageText(locale, `Import ${year} as events`, `ייבוא חגי ${year} כאירועים`)}
              </Button>
            </span>
          </Tooltip>
        )}
      </div>
      {open && rows.map((holiday) => (
        <div className="cal-holiday-row" key={`${holiday.date}-${holiday.name}`}>
          <span className="bidi-figure figure-nowrap">{String(holiday.date || '').slice(0, 10)}</span>
          <Name className="cal-holiday-name">{holiday.name}</Name>
          <span className="cal-chip">{holiday.kind === 'national' ? pageText(locale, 'National', 'לאומי') : pageText(locale, 'Religious', 'דתי')}</span>
          {holiday.is_school_holiday && <span className="cal-chip">{pageText(locale, 'School holiday', 'חופשת לימודים')}</span>}
        </div>
      ))}
    </div>
  );
}

function CalendarHolidays({ holidays, holidaysNote, locale, busy, canEdit, onImportYear }) {
  const [openYears, setOpenYears] = useState(() => new Set());

  const holidayYears = useMemo(() => {
    const groups = {};
    for (const holiday of holidays || []) {
      const year = String(holiday.date || '').slice(0, 4);
      if (!/^\d{4}$/.test(year)) continue;
      if (!groups[year]) groups[year] = [];
      groups[year].push(holiday);
    }
    return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b));
  }, [holidays]);

  function toggleYear(year) {
    setOpenYears((current) => {
      const next = new Set(current);
      if (next.has(year)) {
        next.delete(year);
      } else {
        next.add(year);
      }
      return next;
    });
  }

  return (
    <section className="page-panel cal-panel">
      <div className="panel-head">
        <h2>{pageText(locale, 'Bundled holidays (read only)', 'חגים מובנים (לקריאה בלבד)')}</h2>
        <span>{(holidays || []).length} {pageText(locale, 'rows', 'שורות')}</span>
      </div>
      <div className="cal-panel-body">
        {holidaysNote && <p className="cal-panel-note cal-verify-note">{holidaysNote}</p>}
        {holidayYears.length === 0 ? (
          <p className="cal-empty">{pageText(locale, 'The backend did not report a bundled holiday list.', 'השרת לא דיווח על רשימת חגים מובנית.')}</p>
        ) : (
          holidayYears.map(([year, rows]) => (
            <HolidayYearGroup
              key={year}
              year={year}
              rows={rows}
              locale={locale}
              busy={busy}
              canEdit={canEdit}
              open={openYears.has(year)}
              onToggle={() => toggleYear(year)}
              onImport={() => onImportYear(year, rows)}
            />
          ))
        )}
      </div>
    </section>
  );
}

export default CalendarHolidays;
