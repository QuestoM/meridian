import React, { useEffect, useMemo, useState } from 'react';
import { Button, Tooltip } from '@mui/material';
import { Plus } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { useAssistantEntity } from '../shell/assistant-page-context';
import { eventTypeChipClass, eventTypeLabel, formatEventDate } from './CalendarEventsModel';
import { EventEditor } from './CalendarEventsList';
import { WEEKDAY_HEADERS, activeEventsOnDay, addMonths, isoDay, localIsoDate, monthMatrix, monthTitle, packWeekLanes } from './calendar-events-lib';
import './calendar-month-grid.css';

// The month-grid calendar view: seven Sunday-first columns (the week flows
// right to left in Hebrew, Sunday on the right), month navigation, active
// events as continuous colored bars across their days, a click-to-open day
// panel, and the same inline event editor the list view uses. Persistence
// stays in CalendarEvents.jsx; this module only presents and edits.

// Bars per week row; must match the grid-template-rows in calendar-month-grid.css.
const LANE_CAP = 4;

function barTooltip(event, locale) {
  const dates = event.end_date
    ? `${formatEventDate(event.start_date, locale)} - ${formatEventDate(event.end_date, locale)}`
    : pageText(locale, `${formatEventDate(event.start_date, locale)}, no end date`, `${formatEventDate(event.start_date, locale)}, ללא תאריך סיום`);
  return `${event.name} · ${eventTypeLabel(event.type, locale)} · ${pageText(locale, `intensity ${event.intensity}/5`, `עוצמה ${event.intensity}/5`)} · ${dates}`;
}

function barClass(segment) {
  const classes = ['cal-mg-bar', String(segment.event.type || 'other')];
  if (segment.continuesBefore) classes.push('cont-before');
  if (segment.continuesAfter) classes.push('cont-after');
  return classes.join(' ');
}

function DayEventRow({ event, locale, busy, canEdit, onEdit }) {
  return (
    <div className="cal-mg-dayevent">
      <span className="cal-event-name" dir="auto">{event.name}</span>
      <span className={eventTypeChipClass(event.type)}>{eventTypeLabel(event.type, locale)}</span>
      <span className="cal-mg-dayevent-facts">
        <span className="ltr-run">{formatEventDate(event.start_date, locale)}</span>
        {event.end_date
          ? <span className="ltr-run">{`- ${formatEventDate(event.end_date, locale)}`}</span>
          : <span>{pageText(locale, 'no end date, treated as ongoing', 'ללא תאריך סיום, נחשב מתמשך')}</span>}
        <span>{pageText(locale, `intensity ${event.intensity}/5`, `עוצמה ${event.intensity}/5`)}</span>
      </span>
      {canEdit && (
        <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onEdit}>
          {pageText(locale, 'Edit', 'עריכה')}
        </Button>
      )}
    </div>
  );
}

function CalendarMonthGrid({ events, locale, busy, canEdit, onSave, focus }) {
  const today = localIsoDate();
  const [cursor, setCursor] = useState(() => ({ year: Number(today.slice(0, 4)), month: Number(today.slice(5, 7)) }));
  const [selectedDay, setSelectedDay] = useState(null);
  // editor: { event } edits an existing event; { event: null, draftDay } creates
  // one prefilled on the selected day. Same EventEditor as the list view.
  const [editor, setEditor] = useState(null);

  // The upcoming strip jumps here: land on the event's month with its day open.
  useEffect(() => {
    if (!focus || !focus.date) return;
    const iso = isoDay(focus.date);
    setCursor({ year: Number(iso.slice(0, 4)), month: Number(iso.slice(5, 7)) });
    setSelectedDay(iso);
    setEditor(null);
  }, [focus]);

  const focusedEvent = editor?.event || null;
  useAssistantEntity('event', focusedEvent ? focusedEvent.event_id : '', focusedEvent ? focusedEvent.name : '');

  const weeks = useMemo(() => monthMatrix(cursor.year, cursor.month), [cursor]);
  const weekLanes = useMemo(() => weeks.map((week) => packWeekLanes(week, events, LANE_CAP)), [weeks, events]);
  const dayEvents = useMemo(() => (selectedDay ? activeEventsOnDay(events, selectedDay) : []), [events, selectedDay]);
  const activeCount = useMemo(() => (events || []).filter((event) => event && event.active !== false).length, [events]);
  const headers = WEEKDAY_HEADERS[locale === 'he' ? 'he' : 'en'];

  function pickDay(day) {
    setSelectedDay((current) => (current === day ? null : day));
    setEditor(null);
  }

  function openEvent(event, day) {
    setSelectedDay(day);
    setEditor(canEdit ? { event } : null);
  }

  async function handleSave(form) {
    const ok = await onSave(form, editor?.event || null);
    if (ok) setEditor(null);
  }

  return (
    <section className="page-panel cal-panel">
      <div className="panel-head">
        <h2>{pageText(locale, 'Monthly calendar', 'לוח חודשי')}</h2>
        <span>{activeCount} {pageText(locale, 'active events', 'אירועים פעילים')}</span>
      </div>
      <div className="cal-panel-body">
        <div className="cal-mg-nav">
          <h3 className="cal-mg-title">{monthTitle(cursor.year, cursor.month, locale)}</h3>
          <div className="cal-mg-nav-buttons">
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setCursor((c) => addMonths(c.year, c.month, -1))}>
              {pageText(locale, 'Previous', 'הקודם')}
            </Button>
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => { setCursor({ year: Number(today.slice(0, 4)), month: Number(today.slice(5, 7)) }); setSelectedDay(today); setEditor(null); }}>
              {pageText(locale, 'Today', 'היום')}
            </Button>
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setCursor((c) => addMonths(c.year, c.month, 1))}>
              {pageText(locale, 'Next', 'הבא')}
            </Button>
          </div>
        </div>

        <div className="cal-mg-weekdays">
          {headers.map((label, index) => (
            <span key={label} className={index >= 5 ? 'cal-mg-weekday weekend' : 'cal-mg-weekday'}>{label}</span>
          ))}
        </div>

        <div className="cal-mg-weeks">
        {weeks.map((week, weekIndex) => {
          const { lanes, overflow } = weekLanes[weekIndex];
          return (
            <div className="cal-mg-week" key={week[0]}>
              {week.map((day, dayIndex) => {
                const classes = ['cal-mg-day'];
                if (Number(day.slice(5, 7)) !== cursor.month) classes.push('outside');
                if (dayIndex >= 5) classes.push('weekend');
                if (dayIndex === 6) classes.push('last');
                if (day === today) classes.push('today');
                if (day === selectedDay) classes.push('selected');
                return (
                  <button
                    type="button"
                    key={day}
                    className={classes.join(' ')}
                    style={{ gridColumn: dayIndex + 1 }}
                    aria-pressed={day === selectedDay}
                    aria-label={formatEventDate(day, locale)}
                    onClick={() => pickDay(day)}
                  >
                    <span className="cal-mg-daynum ltr-run">{Number(day.slice(8, 10))}</span>
                  </button>
                );
              })}
              {lanes.map((lane, laneIndex) => lane.map((segment) => (
                <Tooltip title={barTooltip(segment.event, locale)} arrow key={`${segment.event.event_id || segment.event.name}-${laneIndex}`}>
                  <button
                    type="button"
                    className={barClass(segment)}
                    style={{ gridColumn: `${segment.startCol} / span ${segment.endCol - segment.startCol + 1}`, gridRow: laneIndex + 2 }}
                    onClick={() => openEvent(segment.event, week[segment.startCol - 1])}
                  >
                    <span className="cal-mg-bar-label" dir="auto">{segment.event.name}</span>
                  </button>
                </Tooltip>
              )))}
              {Object.entries(overflow).map(([col, count]) => (
                <button
                  type="button"
                  key={col}
                  className="cal-mg-more"
                  style={{ gridColumn: Number(col), gridRow: LANE_CAP + 2 }}
                  onClick={() => { setSelectedDay(week[Number(col) - 1]); setEditor(null); }}
                >
                  {pageText(locale, `+${count} more`, `ועוד ${count}`)}
                </button>
              ))}
            </div>
          );
        })}
        </div>

        {selectedDay && (
          <div className="cal-mg-daypanel">
            <div className="cal-mg-daypanel-head">
              <h3>{formatEventDate(selectedDay, locale)}</h3>
              {canEdit && !editor && (
                <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={() => setEditor({ event: null, draftDay: selectedDay })}>
                  <Plus size={14} />
                  {pageText(locale, 'New event on this day', 'הוספת אירוע ביום הזה')}
                </Button>
              )}
            </div>
            {dayEvents.length === 0 && !editor && (
              <p className="cal-empty">{pageText(locale, 'No active events on this day.', 'אין אירועים פעילים ביום הזה.')}</p>
            )}
            {dayEvents.map((event) => (
              editor?.event?.event_id === event.event_id ? (
                <EventEditor key={event.event_id} initial={event} locale={locale} busy={busy} onSave={handleSave} onCancel={() => setEditor(null)} />
              ) : (
                <DayEventRow key={event.event_id} event={event} locale={locale} busy={busy} canEdit={canEdit} onEdit={() => setEditor({ event })} />
              )
            ))}
            {editor && !editor.event && (
              <EventEditor initial={{ start_date: editor.draftDay, end_date: editor.draftDay }} locale={locale} busy={busy} onSave={handleSave} onCancel={() => setEditor(null)} />
            )}
          </div>
        )}
      </div>
    </section>
  );
}

export default CalendarMonthGrid;
