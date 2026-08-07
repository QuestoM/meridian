import React, { useEffect, useMemo, useState } from 'react';
import { Button, TextField, Tooltip } from '@mui/material';
import { ChevronDown, Plus, Search } from 'lucide-react';
import DateField from '../shell/DateField';
import { pageText } from '../shell/surface-helpers';
import { Name } from '../shell/bidi';
import { useAssistantEntity } from '../shell/assistant-page-context';
import { EVENT_TYPES, eventTypeChipClass, eventTypeLabel, formatEventDate } from './CalendarEventsModel';

// The operator-events panel of the Calendar page: search, type filter chips,
// active-and-upcoming-first ordering, compact expandable rows, an inline editor
// per row, and pagination past PAGE_SIZE rows so 60+ seeded events never turn
// the page into one long dump. Persistence stays in CalendarEvents.jsx; this
// module only presents and edits.

const PAGE_SIZE = 15;

// Filter chip order per the page vocabulary: war, holiday, sport, special, other.
const TYPE_FILTER_ORDER = ['war', 'holiday', 'sport', 'special', 'other'];

// Calendar fact, not a model claim: an event counts as active-and-upcoming when
// it is not deactivated and its window has not fully passed (open-ended events
// are ongoing by the page's own stated convention).
export function isActiveOrUpcoming(event, todayIso) {
  if (!event || event.active === false) {
    return false;
  }
  const start = String(event.start_date || '').slice(0, 10);
  const end = String(event.end_date || '').slice(0, 10);
  if (!end) {
    return true;
  }
  return end >= todayIso || start >= todayIso;
}

// Active-and-upcoming first, then start date descending, then name (stable).
export function sortEventsForList(events, todayIso) {
  return [...(events || [])].sort((a, b) => {
    const liveA = isActiveOrUpcoming(a, todayIso) ? 0 : 1;
    const liveB = isActiveOrUpcoming(b, todayIso) ? 0 : 1;
    if (liveA !== liveB) {
      return liveA - liveB;
    }
    const startA = String(a.start_date || '').slice(0, 10);
    const startB = String(b.start_date || '').slice(0, 10);
    if (startA !== startB) {
      return startB.localeCompare(startA);
    }
    return String(a.name || '').localeCompare(String(b.name || ''), 'he');
  });
}

// Case-insensitive search over name and notes only (the two free-text fields).
export function eventMatchesSearch(event, term) {
  if (!term) {
    return true;
  }
  return `${String(event.name || '')} ${String(event.notes || '')}`.toLowerCase().includes(term);
}

// Shared with the month-grid view, which opens the same editor for the same
// event fields; persistence stays with the container's onSave either way.
export function EventEditor({ initial, locale, busy, onSave, onCancel }) {
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
        <Tooltip title={pageText(locale, 'An operator assertion, not a measurement; it affects the forecast only while the events layer is activated on the rate card.', 'הצהרת מפעיל, לא מדידה; משפיע על התחזית רק כאשר שכבת האירועים מופעלת בכרטיס התעריפים.')} arrow placement="bottom">
          <span className="cal-label-hint">{pageText(locale, 'Price multiplier', 'מכפיל תמחור')}</span>
        </Tooltip>
        <input type="number" min="0.1" max="5" step="0.05" value={form.price_multiplier} onChange={(event) => set('price_multiplier')(event.target.value)} />
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

// One compact event row: the always-visible header line carries name, type and
// dates; details and actions live in the expansion so the list stays scannable.
function EventRow({ event, locale, busy, canEdit, expanded, highlighted, confirming, onToggle, onEdit, onConfirmDeactivate, onCancelConfirm, onDeactivate, onReactivate }) {
  const openEnded = !event.end_date;
  const classes = ['cal-event-row'];
  if (event.active === false) {
    classes.push('inactive');
  }
  if (highlighted) {
    classes.push('cal-row-highlight');
  }
  return (
    <div className={classes.join(' ')}>
      <button type="button" className="cal-event-head" aria-expanded={expanded} onClick={onToggle}>
        <ChevronDown size={14} className={`cal-row-caret${expanded ? ' open' : ''}`} aria-hidden="true" />
        <Name className="cal-event-name">{event.name}</Name>
        <span className={eventTypeChipClass(event.type)}>{eventTypeLabel(event.type, locale)}</span>
        {event.active === false && <span className="cal-chip off">{pageText(locale, 'Deactivated', 'מושבת')}</span>}
        <span className="cal-event-dates">
          <span className="bidi-figure figure-nowrap">{formatEventDate(event.start_date, locale)}</span>
          {openEnded ? (
            <span className={event.type === 'war' ? 'cal-open-ended war' : 'cal-open-ended'}>
              {pageText(locale, 'no end date, treated as ongoing until you set one', 'ללא תאריך סיום, נחשב מתמשך עד שתקבעו תאריך')}
            </span>
          ) : (
            <span className="bidi-figure figure-nowrap">{`- ${formatEventDate(event.end_date, locale)}`}</span>
          )}
        </span>
      </button>
      {expanded && (
        <div className="cal-event-details">
          <div className="cal-event-facts">
            <span>{pageText(locale, `intensity ${event.intensity}/5`, `עוצמה ${event.intensity}/5`)}</span>
            {Number.isFinite(Number(event.price_multiplier)) && Number(event.price_multiplier) !== 1 && (
              <span className="cal-chip" title={pageText(locale, 'Affects the forecast only while the events layer is activated on the rate card', 'משפיע על התחזית רק כאשר שכבת האירועים מופעלת בכרטיס התעריפים')}>
                {pageText(locale, `price multiplier x${Number(event.price_multiplier)}`, `מכפיל תמחור x${Number(event.price_multiplier)}`)}
              </span>
            )}
          </div>
          {event.notes && <p className="cal-event-notes"><Name>{event.notes}</Name></p>}
          {canEdit && (
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
          )}
        </div>
      )}
    </div>
  );
}

function CalendarEventsList({ events, locale, busy, canEdit, highlightId, onSave, onSetActive }) {
  const [editor, setEditor] = useState(null);
  const [confirmId, setConfirmId] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  const todayIso = new Date().toISOString().slice(0, 10);
  const focusedEvent = useMemo(() => (events || []).find((event) => event.event_id === expandedId) || null, [events, expandedId]);
  useAssistantEntity('event', focusedEvent ? focusedEvent.event_id : '', focusedEvent ? focusedEvent.name : '');
  const sorted = useMemo(() => sortEventsForList(events, todayIso), [events, todayIso]);
  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();
    return sorted.filter((event) => {
      if (typeFilter !== 'all' && String(event.type || 'other') !== typeFilter) {
        return false;
      }
      return eventMatchesSearch(event, term);
    });
  }, [sorted, search, typeFilter]);

  // Reset the page window when the query changes; a filter is a fresh view.
  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
  }, [search, typeFilter]);

  // A just-saved event must be visible: clear a query that hides it, widen the
  // page window to reach it, and expand it so the operator sees what landed.
  useEffect(() => {
    if (!highlightId) {
      return;
    }
    const index = filtered.findIndex((event) => event.event_id === highlightId);
    if (index < 0) {
      if (search || typeFilter !== 'all') {
        setSearch('');
        setTypeFilter('all');
      }
      return;
    }
    if (index >= visibleCount) {
      setVisibleCount(index + 1);
    }
    setExpandedId(highlightId);
  }, [highlightId, filtered, search, typeFilter, visibleCount]);

  const visible = filtered.slice(0, visibleCount);
  const typeCounts = useMemo(() => {
    const counts = {};
    sorted.forEach((event) => {
      const key = String(event.type || 'other');
      counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
  }, [sorted]);

  async function handleSave(form) {
    const ok = await onSave(form, editor?.event || null);
    if (ok) {
      setEditor(null);
    }
  }

  return (
    <section className="page-panel cal-panel">
      <div className="panel-head">
        <h2>{pageText(locale, 'Operator events', 'אירועי מפעיל')}</h2>
        {canEdit && (
          <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy || Boolean(editor)} onClick={() => setEditor({ event: null })}>
            <Plus size={14} />
            {pageText(locale, 'New event', 'הוספת אירוע')}
          </Button>
        )}
      </div>
      <div className="cal-panel-body">
      <p className="cal-panel-note">{pageText(locale, 'Active and upcoming events first, newest start date on top. Every save keeps a version snapshot, restorable from the Restore changes page.', 'אירועים פעילים וקרובים תחילה, תאריך ההתחלה החדש ביותר למעלה. כל שמירה שומרת תמונת גרסה, הניתנת לשחזור מעמוד שחזור שינויים.')}</p>

      <div className="cal-toolbar">
        <div className="cal-search">
          <Search size={14} aria-hidden="true" />
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={pageText(locale, 'Search by name or notes', 'חיפוש לפי שם או הערות')}
            aria-label={pageText(locale, 'Search events', 'חיפוש אירועים')}
          />
        </div>
        <div className="cal-filter-chips" role="group" aria-label={pageText(locale, 'Filter by type', 'סינון לפי סוג')}>
          <button type="button" className={`adv-chip${typeFilter === 'all' ? ' active' : ''}`} aria-pressed={typeFilter === 'all'} onClick={() => setTypeFilter('all')}>
            {pageText(locale, 'All', 'הכול')}
          </button>
          {TYPE_FILTER_ORDER.filter((key) => typeCounts[key]).map((key) => (
            <button
              key={key}
              type="button"
              className={`adv-chip${typeFilter === key ? ' active' : ''}`}
              aria-pressed={typeFilter === key}
              onClick={() => setTypeFilter(key)}
            >
              {`${eventTypeLabel(key, locale)} (${typeCounts[key]})`}
            </button>
          ))}
        </div>
      </div>

      {editor && !editor.event && (
        <EventEditor initial={null} locale={locale} busy={busy} onSave={handleSave} onCancel={() => setEditor(null)} />
      )}

      {sorted.length === 0 && !editor && (
        <p className="cal-empty">{pageText(locale, 'No events yet. Add an event or import holidays from the bundled list.', 'אין עדיין אירועים. הוסיפו אירוע או ייבאו חגים מהרשימה המובנית.')}</p>
      )}
      {sorted.length > 0 && filtered.length === 0 && (
        <p className="cal-empty">{pageText(locale, 'No events match your search or filter.', 'אין אירועים שתואמים את החיפוש או הסינון.')}</p>
      )}

      {visible.map((event) => (
        editor?.event?.event_id === event.event_id ? (
          <EventEditor key={event.event_id} initial={event} locale={locale} busy={busy} onSave={handleSave} onCancel={() => setEditor(null)} />
        ) : (
          <EventRow
            key={event.event_id}
            event={event}
            locale={locale}
            busy={busy}
            canEdit={canEdit}
            expanded={expandedId === event.event_id}
            highlighted={highlightId === event.event_id}
            confirming={confirmId === event.event_id}
            onToggle={() => setExpandedId((current) => (current === event.event_id ? null : event.event_id))}
            onEdit={() => { setConfirmId(null); setEditor({ event }); }}
            onConfirmDeactivate={() => setConfirmId(event.event_id)}
            onCancelConfirm={() => setConfirmId(null)}
            onDeactivate={async () => { await onSetActive(event, false); setConfirmId(null); }}
            onReactivate={() => onSetActive(event, true)}
          />
        )
      ))}

      {filtered.length > visibleCount && (
        <div className="cal-show-more">
          <span className="cal-count-note">{pageText(locale, `Showing ${visible.length} of ${filtered.length}`, `מוצגים ${visible.length} מתוך ${filtered.length}`)}</span>
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}>
            {pageText(locale, 'Show more', 'הצגת עוד')}
          </Button>
        </div>
      )}
      </div>
    </section>
  );
}

export default CalendarEventsList;
