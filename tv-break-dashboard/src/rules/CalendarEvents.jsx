import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Info, RefreshCcw } from 'lucide-react';
import { API_BASE, pageText } from '../shell/surface-helpers';
import { readEventsLayer } from './pricing-layers-lib';
import { ModelContextPanel, OverlapPanel, eventTypeChipClass, eventTypeLabel, formatEventDate } from './CalendarEventsModel';
import CalendarEventsList from './CalendarEventsList';
import CalendarHolidays from './CalendarHolidays';
import CalendarMonthGrid from './CalendarMonthGrid';
import CalendarPricingBanner from './CalendarPricingBanner';
import { eventRange, isoDay, localIsoDate, readStoredCalendarView, storeCalendarView, upcomingEvents } from './calendar-events-lib';
import './calendar-events.css';

// The Calendar page container: loads /api/events, owns every write (create,
// update, deactivate, holiday import) and the page-level banners, and delegates
// presentation to the month grid (primary view), CalendarEventsList (searchable,
// paged operator events) and CalendarHolidays (per-year read-only accordions).
// The events GET may carry can_edit per the permissions contract; absent means
// editable, exactly as before, so an older backend changes nothing.

// Only the CRUD contract fields travel back to the API; server-computed overlap
// fields never round-trip into a write. An empty end_date string is the API's
// open-ended marker (null on PUT would mean "leave unchanged").
function eventBody(event, patch = {}) {
  return {
    name: event.name,
    type: event.type,
    start_date: event.start_date,
    end_date: event.end_date || '',
    intensity: Number(event.intensity) || 1,
    price_multiplier: Number(event.price_multiplier) || 1,
    notes: event.notes || '',
    active: event.active !== false,
    ...patch,
  };
}

function CalendarEvents({ locale, notify, refreshKey, onGlobalRefresh, onOpenRateCard }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState('loading');
  const [busy, setBusy] = useState(false);
  // The event a just-finished save produced, so the list can surface it (clear
  // hiding filters, widen the page window, expand the row) instead of letting a
  // new event land invisibly at the bottom of 60+ rows.
  const [highlightId, setHighlightId] = useState(null);
  // Grid (primary) or list; the choice persists across sessions.
  const [view, setView] = useState(readStoredCalendarView);
  // A one-shot jump target for the month grid, set by the upcoming strip.
  const [gridFocus, setGridFocus] = useState(null);

  function switchView(next) {
    setView(next);
    storeCalendarView(next);
  }

  const load = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/events`);
      if (response.status === 404) {
        setStatus('missing');
        return;
      }
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      setData(await response.json());
      setStatus('ok');
    } catch {
      setStatus('offline');
    }
  }, []);

  // Tri-state events-pricing-layer activation: true / false / null when the server
  // does not report the layer, so the honesty note says exactly that, never "off".
  const [eventsPricing, setEventsPricing] = useState(null);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const response = await fetch(`${API_BASE}/api/pricing`);
        if (!response.ok) throw new Error(String(response.status));
        const card = await response.json();
        if (!alive) return;
        const { supported, enabled } = readEventsLayer(card);
        setEventsPricing(supported ? enabled : null);
      } catch {
        if (alive) setEventsPricing(null);
      }
    })();
    return () => { alive = false; };
  }, [refreshKey]);

  const events = useMemo(() => (Array.isArray(data?.events) ? data.events : []), [data]);
  const holidays = useMemo(() => (Array.isArray(data?.holidays) ? data.holidays : []), [data]);
  // Permissions contract: company users edit, channel users read. An absent
  // field means an older backend without the affiliation model, so editable.
  const canEdit = data?.can_edit !== false;
  const todayIso = localIsoDate();
  const upcoming = useMemo(() => upcomingEvents(events, todayIso), [events, todayIso]);
  // Verify-before-use label: prefer a backend-sent note; the fallback mirrors
  // the bundled table's own header (a static checked-in list, not a calendar
  // service), so the caution is never silently dropped.
  const backendNote = data?.holidays_note ?? data?.holidays_verify_note ?? data?.holidays_label;
  const holidaysNote = typeof backendNote === 'string' && backendNote.trim()
    ? backendNote
    : pageText(locale, 'A static checked-in reference list, not a live calendar service. Verify dates against the official calendar before operational use; observed dates can shift.', 'רשימת ייחוס קבועה השמורה בקוד, לא שירות לוח שנה חי. אמתו את התאריכים מול הלוח הרשמי לפני שימוש תפעולי; תאריכי קיום עשויים לזוז.');

  async function persist(method, path, body) {
    const response = await fetch(`${API_BASE}${path}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `${response.status} ${response.statusText}`);
    }
    return response.json().catch(() => ({}));
  }

  // Save one event (create when existing is null). Returns true on success so
  // the list can close its editor.
  async function saveEvent(form, existing) {
    if (!form.name.trim() || !form.start_date) {
      notify('An event needs a name and a start date.', 'אירוע צריך שם ותאריך התחלה.');
      return false;
    }
    if (form.end_date && form.end_date < form.start_date) {
      notify('The end date cannot be before the start date.', 'תאריך הסיום לא יכול להקדים את תאריך ההתחלה.');
      return false;
    }
    const mult = Number(form.price_multiplier);
    if (!Number.isFinite(mult) || mult < 0.1 || mult > 5) {
      notify('The price multiplier must be between 0.1 and 5.', 'מכפיל התמחור חייב להיות בין 0.1 ל-5.');
      return false;
    }
    setBusy(true);
    try {
      const body = eventBody({ ...form, active: existing ? existing.active !== false : true });
      let saved;
      if (existing?.event_id) {
        saved = await persist('PUT', `/api/events/${encodeURIComponent(existing.event_id)}`, body);
      } else {
        saved = await persist('POST', '/api/events', body);
      }
      notify('Event saved. Every change keeps a version snapshot and can be restored from the Restore changes page.', 'האירוע נשמר. כל שינוי שומר תמונת גרסה וניתן לשחזור מעמוד שחזור שינויים.');
      setHighlightId(saved?.event_id || existing?.event_id || null);
      await load();
      onGlobalRefresh?.();
      return true;
    } catch (error) {
      notify(`Saving the event failed (${error.message}).`, `שמירת האירוע נכשלה (${error.message}).`);
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function setEventActive(event, active) {
    setBusy(true);
    try {
      await persist('PUT', `/api/events/${encodeURIComponent(event.event_id)}`, eventBody(event, { active }));
      notify(
        active ? 'Event reactivated. It can be restored from the Restore changes page.' : 'Event deactivated. It can be restored from the Restore changes page.',
        active ? 'האירוע הופעל מחדש. ניתן לשחזר מעמוד שחזור שינויים.' : 'האירוע הושבת. ניתן לשחזר מעמוד שחזור שינויים.',
      );
      await load();
      onGlobalRefresh?.();
    } catch (error) {
      notify(`Updating the event failed (${error.message}).`, `עדכון האירוע נכשל (${error.message}).`);
    } finally {
      setBusy(false);
    }
  }

  async function importYear(year, rows) {
    const existing = new Set(events.map((event) => `${event.name}|${String(event.start_date || '').slice(0, 10)}`));
    const fresh = rows.filter((row) => !existing.has(`${row.name}|${String(row.date || '').slice(0, 10)}`));
    if (!fresh.length) {
      notify('Every holiday of that year is already in the events list.', 'כל חגי השנה הזו כבר נמצאים ברשימת האירועים.');
      return;
    }
    setBusy(true);
    let created = 0;
    try {
      for (const row of fresh) {
        await persist('POST', '/api/events', {
          name: row.name,
          type: 'holiday',
          start_date: row.date,
          end_date: row.date,
          intensity: 1,
          notes: pageText(locale, 'Imported from the bundled holiday list', 'יובא מרשימת החגים המובנית'),
          active: true,
        });
        created += 1;
      }
      notify(`Imported ${created} holidays as events. They can be restored from the Restore changes page.`, `יובאו ${created} חגים כאירועים. ניתן לשחזר מעמוד שחזור שינויים.`);
    } catch (error) {
      notify(`Holiday import stopped after ${created} events (${error.message}).`, `ייבוא החגים נעצר אחרי ${created} אירועים (${error.message}).`);
    } finally {
      setBusy(false);
      await load();
      onGlobalRefresh?.();
    }
  }

  if (status === 'loading') {
    return (
      <section className="page-workspace">
        <div className="page-header"><h1>{pageText(locale, 'Events calendar', 'לוח אירועים')}</h1></div>
        <p>{pageText(locale, 'Loading the events calendar...', 'טוען את לוח האירועים...')}</p>
      </section>
    );
  }

  if (status !== 'ok') {
    return (
      <section className="page-workspace">
        <div className="page-header"><h1>{pageText(locale, 'Events calendar', 'לוח אירועים')}</h1></div>
        <div className="cal-banner">
          <Info size={16} aria-hidden="true" />
          <p>{status === 'missing'
            ? pageText(locale, 'This server version does not carry the events service yet, so no calendar is shown rather than an invented one.', 'גרסת השרת הזו עדיין אינה כוללת את שירות האירועים, ולכן לא מוצג לוח במקום להמציא נתון.')
            : pageText(locale, 'The events service is unreachable right now. No calendar is shown rather than a stale or invented one.', 'שירות האירועים אינו זמין כרגע. לא מוצג לוח במקום נתון ישן או מומצא.')}</p>
        </div>
      </section>
    );
  }

  // The upcoming strip jumps the grid to the event: an ongoing event lands on
  // today (where its bar is visible), a future one on its start day.
  function jumpToEvent(event) {
    const { start, end } = eventRange(event);
    switchView('grid');
    setGridFocus({ date: start <= todayIso && end >= todayIso ? todayIso : isoDay(event.start_date), nonce: Date.now() });
  }

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Events calendar', 'לוח אירועים')}</h1>
          <p>{pageText(locale, 'Manage holidays, wars and special events next to an honest picture of what the model actually measures today. Events do not change retention numbers until an effect is measured on richer history.', 'ניהול חגים, מלחמות ואירועים מיוחדים לצד תמונה כנה של מה שהמודל באמת מודד היום. אירועים אינם משנים מספרי שימור עד שנמדדת השפעה על היסטוריה עשירה יותר.')}</p>
          {!canEdit && <p className="cal-readonly-note">{pageText(locale, 'Event editing is available to the company team only.', 'עריכת אירועים זמינה לצוות החברה בלבד.')}</p>}
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={load}>
          <RefreshCcw size={14} />
          {pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      {upcoming.length > 0 && (
        <div className="cal-upcoming" role="list" aria-label={pageText(locale, 'Upcoming events', 'אירועים קרובים')}>
          <span className="cal-upcoming-label">{pageText(locale, 'Coming up:', 'הקרובים:')}</span>
          {upcoming.map((event) => (
            <button type="button" role="listitem" className="cal-upcoming-item" key={event.event_id || event.name} onClick={() => jumpToEvent(event)}>
              <span className={eventTypeChipClass(event.type)}>{eventTypeLabel(event.type, locale)}</span>
              <span dir="auto">{event.name}</span>
              <span className="cal-upcoming-dates">
                {eventRange(event).start <= todayIso
                  ? pageText(locale, 'ongoing', 'מתמשך')
                  : <span className="ltr-run">{formatEventDate(event.start_date, locale)}</span>}
              </span>
            </button>
          ))}
        </div>
      )}

      <div className="cal-view-bar" role="group" aria-label={pageText(locale, 'Calendar view', 'תצוגת הלוח')}>
        <button type="button" className={view === 'grid' ? 'segmented active' : 'segmented'} aria-pressed={view === 'grid'} onClick={() => switchView('grid')}>
          {pageText(locale, 'Calendar', 'לוח')}
        </button>
        <button type="button" className={view === 'list' ? 'segmented active' : 'segmented'} aria-pressed={view === 'list'} onClick={() => switchView('list')}>
          {pageText(locale, 'List', 'רשימה')}
        </button>
      </div>

      {view === 'grid' ? (
        <>
          <CalendarMonthGrid
            events={events}
            locale={locale}
            busy={busy}
            canEdit={canEdit}
            onSave={saveEvent}
            focus={gridFocus}
          />
          <CalendarHolidays
            holidays={holidays}
            holidaysNote={holidaysNote}
            locale={locale}
            busy={busy}
            canEdit={canEdit}
            onImportYear={importYear}
          />
        </>
      ) : (
        <div className="cal-grid">
          <CalendarEventsList
            events={events}
            locale={locale}
            busy={busy}
            canEdit={canEdit}
            highlightId={highlightId}
            onSave={saveEvent}
            onSetActive={setEventActive}
          />
          <CalendarHolidays
            holidays={holidays}
            holidaysNote={holidaysNote}
            locale={locale}
            busy={busy}
            canEdit={canEdit}
            onImportYear={importYear}
          />
        </div>
      )}

      <CalendarPricingBanner locale={locale} eventsPricing={eventsPricing} onOpenRateCard={onOpenRateCard} />

      <ModelContextPanel context={data?.model_context} locale={locale} />
      <OverlapPanel events={events} locale={locale} />
    </section>
  );
}

export default CalendarEvents;
