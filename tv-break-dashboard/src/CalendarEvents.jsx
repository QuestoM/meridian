import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, TextField, Tooltip } from '@mui/material';
import { Info, Plus, RefreshCcw } from 'lucide-react';
import DateField from './DateField';
import { API_BASE, pageText } from './surface-helpers';
import { readEventsLayer } from './pricing-layers-lib';
import {
  EVENT_TYPES,
  ModelContextPanel,
  OverlapPanel,
  eventTypeChipClass,
  eventTypeLabel,
  formatEventDate,
} from './CalendarEventsModel';
import './calendar-events.css';

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

function EventEditor({ initial, locale, busy, onSave, onCancel }) {
  const [form, setForm] = useState({
    name: initial?.name || '',
    type: initial?.type || 'special',
    start_date: initial?.start_date || '',
    end_date: initial?.end_date || '',
    intensity: initial?.intensity || 3,
    price_multiplier: initial?.price_multiplier != null && Number.isFinite(Number(initial.price_multiplier)) ? Number(initial.price_multiplier) : 1,
    notes: initial?.notes || '',
  });
  const set = (key) => (value) => setForm((current) => ({ ...current, [key]: value }));
  return (
    <div className="cal-editor">
      <TextField
        label={pageText(locale, 'Event name', 'שם האירוע')}
        size="small"
        value={form.name}
        onChange={(event) => set('name')(event.target.value)}
      />
      <label className="cal-editor-field">
        {pageText(locale, 'Type', 'סוג')}
        <select value={form.type} onChange={(event) => set('type')(event.target.value)}>
          {Object.keys(EVENT_TYPES).map((key) => (
            <option key={key} value={key}>{eventTypeLabel(key, locale)}</option>
          ))}
        </select>
      </label>
      <DateField
        label={pageText(locale, 'Start date', 'תאריך התחלה')}
        value={form.start_date}
        onChange={set('start_date')}
      />
      <DateField
        label={pageText(locale, 'End date', 'תאריך סיום')}
        value={form.end_date}
        onChange={set('end_date')}
        helperText={pageText(locale, 'Leave empty for an open-ended event', 'השאירו ריק לאירוע ללא תאריך סיום')}
      />
      <label className="cal-editor-field">
        <Tooltip title={pageText(locale, 'Operator judgement on a 1 to 5 scale, stored with the event. It is not measured from data and does not change any retention or revenue number.', 'שיקול דעת של המפעיל בסולם 1 עד 5, נשמר עם האירוע. הערך אינו נמדד מנתונים ואינו משנה אף מספר שימור או הכנסה.')} arrow>
          <span className="cal-label-hint">{pageText(locale, 'Intensity (operator judgement)', 'עוצמה (שיקול דעת מפעיל)')}</span>
        </Tooltip>
        <select value={form.intensity} onChange={(event) => set('intensity')(Number(event.target.value))}>
          {[1, 2, 3, 4, 5].map((level) => <option key={level} value={level}>{level}</option>)}
        </select>
      </label>
      <label className="cal-editor-field">
        <Tooltip title={pageText(locale, 'An operator assertion, not a measurement; it affects the forecast only while the events layer is activated on the Pricing page.', 'הצהרת מפעיל, לא מדידה; משפיע על התחזית רק כאשר שכבת האירועים מופעלת בעמוד התמחור.')} arrow placement="bottom">
          <span className="cal-label-hint">{pageText(locale, 'Price multiplier', 'מכפיל תמחור')}</span>
        </Tooltip>
        <input type="number" min="0.1" max="5" step="0.05" dir="ltr" value={form.price_multiplier} onChange={(event) => set('price_multiplier')(event.target.value)} />
      </label>
      <TextField
        label={pageText(locale, 'Notes', 'הערות')}
        size="small"
        value={form.notes}
        onChange={(event) => set('notes')(event.target.value)}
      />
      <div className="cal-editor-actions">
        <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={() => onSave(form)}>
          {pageText(locale, 'Save', 'שמירה')}
        </Button>
        <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onCancel}>
          {pageText(locale, 'Cancel', 'ביטול')}
        </Button>
      </div>
    </div>
  );
}

function EventRow({ event, locale, busy, confirming, onEdit, onConfirmDeactivate, onCancelConfirm, onDeactivate, onReactivate }) {
  const openEnded = !event.end_date;
  return (
    <div className={event.active === false ? 'cal-event-row inactive' : 'cal-event-row'}>
      <div className="cal-event-main">
        <span className="cal-event-name">{event.name}</span>
        <span className={eventTypeChipClass(event.type)}>{eventTypeLabel(event.type, locale)}</span>
        {event.active === false && <span className="cal-chip off">{pageText(locale, 'Deactivated', 'מושבת')}</span>}
      </div>
      <div className="cal-event-facts">
        <span className="ltr-run">{formatEventDate(event.start_date, locale)}</span>
        {openEnded ? (
          <span className={event.type === 'war' ? 'cal-open-ended war' : 'cal-open-ended'}>
            {pageText(locale, 'no end date, treated as ongoing until you set one', 'ללא תאריך סיום, נחשב מתמשך עד שתקבעו תאריך')}
          </span>
        ) : (
          <span className="ltr-run">{`- ${formatEventDate(event.end_date, locale)}`}</span>
        )}
        <span>{pageText(locale, `intensity ${event.intensity}/5`, `עוצמה ${event.intensity}/5`)}</span>
        {Number.isFinite(Number(event.price_multiplier)) && Number(event.price_multiplier) !== 1 && (
          <span className="cal-chip" title={pageText(locale, 'Affects the forecast only while the events layer is activated on the Pricing page', 'משפיע על התחזית רק כאשר שכבת האירועים מופעלת בעמוד התמחור')}>
            {pageText(locale, `price multiplier x${Number(event.price_multiplier)}`, `מכפיל תמחור x${Number(event.price_multiplier)}`)}
          </span>
        )}
      </div>
      {event.notes && <p className="cal-event-notes">{event.notes}</p>}
      <div className="cal-event-actions">
        {!confirming && (
          <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onEdit}>
            {pageText(locale, 'Edit', 'עריכה')}
          </Button>
        )}
        {!confirming && event.active !== false && (
          <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onConfirmDeactivate}>
            {pageText(locale, 'Deactivation', 'השבתה')}
          </Button>
        )}
        {!confirming && event.active === false && (
          <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onReactivate}>
            {pageText(locale, 'Reactivation', 'הפעלה מחדש')}
          </Button>
        )}
        {confirming && (
          <span className="cal-confirm" role="alertdialog">
            <span>{pageText(locale, 'Deactivation removes the event from every surface but keeps it in the list. It can be restored from the Restore changes page.', 'ההשבתה מסירה את האירוע מכל התצוגות אך שומרת אותו ברשימה. ניתן לשחזר מעמוד שחזור שינויים.')}</span>
            <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onDeactivate}>
              {pageText(locale, 'Confirm deactivation', 'אישור השבתה')}
            </Button>
            <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onCancelConfirm}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
          </span>
        )}
      </div>
    </div>
  );
}

function CalendarEvents({ locale, notify, refreshKey, onGlobalRefresh }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState('loading');
  const [editor, setEditor] = useState(null);
  const [confirmId, setConfirmId] = useState(null);
  const [busy, setBusy] = useState(false);

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
  // Verify-before-use label: prefer a backend-sent note; the fallback mirrors
  // the bundled table's own header (a static checked-in list, not a calendar
  // service), so the caution is never silently dropped.
  const backendNote = data?.holidays_note ?? data?.holidays_verify_note ?? data?.holidays_label;
  const holidaysNote = typeof backendNote === 'string' && backendNote.trim()
    ? backendNote
    : pageText(locale, 'A static checked-in reference list, not a live calendar service. Verify dates against the official calendar before operational use; observed dates can shift.', 'רשימת ייחוס קבועה השמורה בקוד, לא שירות לוח שנה חי. אמתו את התאריכים מול הלוח הרשמי לפני שימוש תפעולי; תאריכי קיום עשויים לזוז.');
  const holidayYears = useMemo(() => {
    const groups = {};
    for (const holiday of holidays) {
      const year = String(holiday.date || '').slice(0, 4);
      if (!/^\d{4}$/.test(year)) continue;
      if (!groups[year]) groups[year] = [];
      groups[year].push(holiday);
    }
    return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b));
  }, [holidays]);

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

  async function saveEvent(form) {
    if (!form.name.trim() || !form.start_date) {
      notify('An event needs a name and a start date.', 'אירוע צריך שם ותאריך התחלה.');
      return;
    }
    if (form.end_date && form.end_date < form.start_date) {
      notify('The end date cannot be before the start date.', 'תאריך הסיום לא יכול להקדים את תאריך ההתחלה.');
      return;
    }
    const mult = Number(form.price_multiplier);
    if (!Number.isFinite(mult) || mult < 0.1 || mult > 5) {
      notify('The price multiplier must be between 0.1 and 5.', 'מכפיל התמחור חייב להיות בין 0.1 ל-5.');
      return;
    }
    setBusy(true);
    try {
      const body = eventBody({ ...form, active: editor?.event ? editor.event.active !== false : true });
      if (editor?.event?.event_id) {
        await persist('PUT', `/api/events/${encodeURIComponent(editor.event.event_id)}`, body);
      } else {
        await persist('POST', '/api/events', body);
      }
      notify('Event saved. Every change keeps a version snapshot and can be restored from the Restore changes page.', 'האירוע נשמר. כל שינוי שומר תמונת גרסה וניתן לשחזור מעמוד שחזור שינויים.');
      setEditor(null);
      await load();
      onGlobalRefresh?.();
    } catch (error) {
      notify(`Saving the event failed (${error.message}).`, `שמירת האירוע נכשלה (${error.message}).`);
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
      setConfirmId(null);
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

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Events calendar', 'לוח אירועים')}</h1>
          <p>{pageText(locale, 'Manage holidays, wars and special events next to an honest picture of what the model actually measures today. Events do not change retention numbers until an effect is measured on richer history.', 'ניהול חגים, מלחמות ואירועים מיוחדים לצד תמונה כנה של מה שהמודל באמת מודד היום. אירועים אינם משנים מספרי שימור עד שנמדדת השפעה על היסטוריה עשירה יותר.')}</p>
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={load}>
          <RefreshCcw size={14} />
          {pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      <div className="cal-grid">
        <section className="page-panel cal-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Operator events', 'אירועי מפעיל')}</h2>
            <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy || Boolean(editor)} onClick={() => setEditor({ event: null })}>
              <Plus size={14} />
              {pageText(locale, 'New event', 'הוספת אירוע')}
            </Button>
          </div>
          <p className="cal-panel-note">{pageText(locale, 'Every save keeps a version snapshot, restorable from the Restore changes page.', 'כל שמירה שומרת תמונת גרסה, הניתנת לשחזור מעמוד שחזור שינויים.')}</p>
          {editor && !editor.event && (
            <EventEditor initial={null} locale={locale} busy={busy} onSave={saveEvent} onCancel={() => setEditor(null)} />
          )}
          {events.length === 0 && !editor && (
            <p className="cal-empty">{pageText(locale, 'No events yet. Add an event or import holidays from the bundled list.', 'אין עדיין אירועים. הוסיפו אירוע או ייבאו חגים מהרשימה המובנית.')}</p>
          )}
          {events.map((event) => (
            editor?.event?.event_id === event.event_id ? (
              <EventEditor key={event.event_id} initial={event} locale={locale} busy={busy} onSave={saveEvent} onCancel={() => setEditor(null)} />
            ) : (
              <EventRow
                key={event.event_id}
                event={event}
                locale={locale}
                busy={busy}
                confirming={confirmId === event.event_id}
                onEdit={() => { setConfirmId(null); setEditor({ event }); }}
                onConfirmDeactivate={() => setConfirmId(event.event_id)}
                onCancelConfirm={() => setConfirmId(null)}
                onDeactivate={() => setEventActive(event, false)}
                onReactivate={() => setEventActive(event, true)}
              />
            )
          ))}
        </section>

        <section className="page-panel cal-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Bundled holidays (read only)', 'חגים מובנים (לקריאה בלבד)')}</h2>
            <span>{holidays.length} {pageText(locale, 'rows', 'שורות')}</span>
          </div>
          {holidaysNote && <p className="cal-panel-note cal-verify-note">{holidaysNote}</p>}
          {holidayYears.length === 0 ? (
            <p className="cal-empty">{pageText(locale, 'The backend did not report a bundled holiday list.', 'השרת לא דיווח על רשימת חגים מובנית.')}</p>
          ) : (
            holidayYears.map(([year, rows]) => (
              <div className="cal-holiday-year" key={year}>
                <div className="cal-holiday-year-head">
                  <span className="ltr-run">{year}</span>
                  <Tooltip title={pageText(locale, 'Creates one event per holiday of this year, with intensity 1 until you judge it, so you can attach intensity or deactivate single rows. Holidays already in the list are skipped.', 'יוצר אירוע לכל חג בשנה הזו, עם עוצמה 1 עד שתקבעו אותה, כך שתוכלו לצרף עוצמה או להשבית שורות בודדות. חגים שכבר ברשימה מדולגים.')} arrow>
                    <span>
                      <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={() => importYear(year, rows)}>
                        {pageText(locale, `Import ${year} as events`, `ייבוא חגי ${year} כאירועים`)}
                      </Button>
                    </span>
                  </Tooltip>
                </div>
                {rows.map((holiday) => (
                  <div className="cal-holiday-row" key={`${holiday.date}-${holiday.name}`}>
                    <span className="ltr-run">{String(holiday.date || '').slice(0, 10)}</span>
                    <span className="cal-holiday-name">{holiday.name}</span>
                    <span className="cal-chip">{holiday.kind === 'national' ? pageText(locale, 'National', 'לאומי') : pageText(locale, 'Religious', 'דתי')}</span>
                    {holiday.is_school_holiday && <span className="cal-chip">{pageText(locale, 'School holiday', 'חופשת לימודים')}</span>}
                  </div>
                ))}
              </div>
            ))
          )}
        </section>
      </div>

      <div className="cal-banner">
        <Info size={16} aria-hidden="true" />
        <p>{eventsPricing === null
          ? pageText(locale, 'Each event also carries a price multiplier hook for the events layer on the Pricing page. This server does not report that layer, so its activation state is unknown here.', 'לכל אירוע קיים גם מכפיל תמחור המחובר לשכבת האירועים בעמוד התמחור. השרת הזה אינו מדווח על השכבה, ולכן מצב ההפעלה שלה אינו ידוע כאן.')
          : eventsPricing
            ? pageText(locale, 'Each event carries a price multiplier wired to the events layer on the Pricing page. The layer is currently activated, so multipliers other than 1.0 change expected revenue in the forecast on event days.', 'לכל אירוע קיים מכפיל תמחור המחובר לשכבת האירועים בעמוד התמחור. השכבה מופעלת כעת, ולכן מכפילים שונים מ-1.0 משנים את ההכנסה הצפויה בתחזית בימי אירועים.')
            : pageText(locale, 'Each event carries a price multiplier wired to the events layer on the Pricing page. The layer is currently off, so no multiplier changes any forecast number until it is activated there.', 'לכל אירוע קיים מכפיל תמחור המחובר לשכבת האירועים בעמוד התמחור. השכבה כבויה כעת, ולכן אף מכפיל אינו משנה מספר בתחזית עד הפעלתה שם.')}</p>
      </div>

      <ModelContextPanel context={data?.model_context} locale={locale} />
      <OverlapPanel events={events} locale={locale} />
    </section>
  );
}

export default CalendarEvents;
