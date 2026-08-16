import React, { useEffect, useMemo, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Button } from '../studio/actions';
import { ChevronDown, Plus, Search } from 'lucide-react';
import DateField from '../shell/DateField';
import { pageText } from '../shell/surface-helpers';
import { Name } from '../shell/bidi';
import { formatDay, formatDayRange } from '../shell/dates';
import { useAssistantEntity } from '../shell/assistant-page-context';
import { InputControl, Pressable, SelectControl } from '../studio/dom-controls';
import ConsequenceDialog from '../safety/ConsequenceDialog';
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
  const fieldId = React.useId();
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
      <div className="cal-editor-field">
        <label htmlFor={`${fieldId}-name`}>{pageText(locale, 'Event name', 'שם האירוע')}</label>
        <InputControl id={`${fieldId}-name`} value={form.name} onChange={(event) => set('name')(event.target.value)} />
      </div>
      <div className="cal-editor-field">
        <label htmlFor={`${fieldId}-type`}>{pageText(locale, 'Type', 'סוג')}</label>
        <SelectControl id={`${fieldId}-type`} value={form.type} onChange={(event) => set('type')(event.target.value)}>
          {Object.keys(EVENT_TYPES).map((key) => (
            <option key={key} value={key}>{eventTypeLabel(key, locale)}</option>
          ))}
        </SelectControl>
      </div>
      <div className="cal-editor-field">
        <label htmlFor={`${fieldId}-start`}>{pageText(locale, 'Start date', 'תאריך התחלה')}</label>
        <DateField id={`${fieldId}-start`} value={form.start_date} onChange={set('start_date')} fullWidth />
      </div>
      <div className="cal-editor-field">
        <label htmlFor={`${fieldId}-end`}>{pageText(locale, 'End date', 'תאריך סיום')}</label>
        <DateField
          id={`${fieldId}-end`}
          value={form.end_date}
          onChange={set('end_date')}
          helperText={pageText(locale, 'Leave empty for an open-ended event', 'השאירו ריק לאירוע ללא תאריך סיום')}
          fullWidth
        />
      </div>
      <div className="cal-editor-field">
        <Tooltip describeChild title={pageText(locale, 'Operator judgement on a 1 to 5 scale, stored with the event. It is not measured from data and does not change any retention or revenue number.', 'שיקול דעת של המפעיל בסולם 1 עד 5, נשמר עם האירוע. הערך אינו נמדד מנתונים ואינו משנה אף מספר שימור או הכנסה.')} arrow>
          <label className="cal-label-hint" htmlFor={`${fieldId}-intensity`}>{pageText(locale, 'Intensity (operator judgement)', 'עוצמה (שיקול דעת מפעיל)')}</label>
        </Tooltip>
        <SelectControl id={`${fieldId}-intensity`} value={form.intensity} onChange={(event) => set('intensity')(Number(event.target.value))}>
          {[1, 2, 3, 4, 5].map((level) => <option key={level} value={level}>{level}</option>)}
        </SelectControl>
      </div>
      <div className="cal-editor-field">
        <Tooltip describeChild title={pageText(locale, 'An operator assertion, not a measurement; it affects the forecast only while the events layer is activated on the rate card.', 'הצהרת מפעיל, לא מדידה; משפיע על התחזית רק כאשר שכבת האירועים מופעלת בכרטיס התעריפים.')} arrow placement="bottom">
          <label className="cal-label-hint" htmlFor={`${fieldId}-multiplier`}>{pageText(locale, 'Price multiplier', 'מכפיל תמחור')}</label>
        </Tooltip>
        <InputControl id={`${fieldId}-multiplier`} type="number" min="0.1" max="5" step="0.05" value={form.price_multiplier} onChange={(event) => set('price_multiplier')(event.target.value)} />
      </div>
      <div className="cal-editor-field">
        <label htmlFor={`${fieldId}-notes`}>{pageText(locale, 'Notes', 'הערות')}</label>
        <InputControl id={`${fieldId}-notes`} value={form.notes} onChange={(event) => set('notes')(event.target.value)} />
      </div>
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
function EventRow({ event, locale, busy, canEdit, expanded, highlighted, onToggle, onEdit, onConfirmDeactivate, onReactivate }) {
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
      <Pressable type="button" className="cal-event-head" aria-expanded={expanded} onClick={onToggle}>
        <ChevronDown size={14} className={`cal-row-caret${expanded ? ' open' : ''}`} aria-hidden="true" />
        <Name className="cal-event-name">{event.name}</Name>
        <span className={eventTypeChipClass(event.type)}>{eventTypeLabel(event.type, locale)}</span>
        {event.active === false && <span className="cal-chip off">{pageText(locale, 'Deactivated', 'מושבת')}</span>}
        <span className="cal-event-dates">
          {openEnded ? (
            <>
              <span className="bidi-figure figure-nowrap">{formatDay(event.start_date)}</span>
              <span className={event.type === 'war' ? 'cal-open-ended war' : 'cal-open-ended'}>
                {pageText(locale, 'no end date, treated as ongoing until you set one', 'ללא תאריך סיום, נחשב מתמשך עד שתקבעו תאריך')}
              </span>
            </>
          ) : (
            <span className="bidi-figure figure-nowrap">{formatDayRange(event.start_date, event.end_date)}</span>
          )}
        </span>
      </Pressable>
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
            <Button className="secondary-button compact" type="button" variant="outlined" disabled={busy} onClick={onEdit}>
              {pageText(locale, 'Edit', 'עריכה')}
            </Button>
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              disabled={busy}
              onClick={event.active === false ? onReactivate : onConfirmDeactivate}
            >
              {event.active === false
                ? pageText(locale, 'Reactivate', 'הפעלה מחדש')
                : pageText(locale, 'Deactivate', 'השבתה')}
            </Button>
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
  const reviewEvent = useMemo(() => (events || []).find((event) => event.event_id === confirmId) || null, [events, confirmId]);
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

  async function confirmDeactivate() {
    if (!reviewEvent) return;
    await onSetActive(reviewEvent, false);
    setConfirmId(null);
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
          <InputControl
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={pageText(locale, 'Search by name or notes', 'חיפוש לפי שם או הערות')}
            aria-label={pageText(locale, 'Search events', 'חיפוש אירועים')}
          />
        </div>
        <div className="cal-filter-chips" role="group" aria-label={pageText(locale, 'Filter by type', 'סינון לפי סוג')}>
          <Pressable type="button" className={`adv-chip${typeFilter === 'all' ? ' active' : ''}`} aria-pressed={typeFilter === 'all'} onClick={() => setTypeFilter('all')}>
            {pageText(locale, 'All', 'הכול')}
          </Pressable>
          {TYPE_FILTER_ORDER.filter((key) => typeCounts[key]).map((key) => (
            <Pressable
              key={key}
              type="button"
              className={`adv-chip${typeFilter === key ? ' active' : ''}`}
              aria-pressed={typeFilter === key}
              onClick={() => setTypeFilter(key)}
            >
              {`${eventTypeLabel(key, locale)} (${typeCounts[key]})`}
            </Pressable>
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
            onToggle={() => setExpandedId((current) => (current === event.event_id ? null : event.event_id))}
            onEdit={() => { setConfirmId(null); setEditor({ event }); }}
            onConfirmDeactivate={() => setConfirmId(event.event_id)}
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

      <ConsequenceDialog
        open={Boolean(reviewEvent)}
        locale={locale}
        title={pageText(locale, 'Deactivate this event?', 'להשבית את האירוע?')}
        description={pageText(locale, 'Deactivation keeps the record but removes it from every active-event calculation.', 'ההשבתה שומרת את הרשומה אך מוציאה אותה מכל חישוב של אירועים פעילים.')}
        object={reviewEvent ? (
          <span className="consequence-review__object">
            <Name>{reviewEvent.name}</Name>
            {' · '}
            <span className="bidi-figure figure-nowrap">
              {reviewEvent.end_date ? formatDayRange(reviewEvent.start_date, reviewEvent.end_date) : formatDay(reviewEvent.start_date)}
            </span>
            {' · ID '}<bdi>{String(reviewEvent.event_id)}</bdi>
          </span>
        ) : ''}
        scope={pageText(locale, 'This event record only. It remains stored and visible in this list as deactivated; every other event remains unchanged.', 'רשומת האירוע הזו בלבד. היא תישאר שמורה וגלויה ברשימה כמושבתת; כל שאר האירועים נשארים ללא שינוי.')}
        consequence={pageText(locale, 'It stops appearing as active or upcoming and stops contributing its dates and multiplier to event-aware forecasts and pricing while the events layer is enabled.', 'האירוע יפסיק להופיע כפעיל או קרוב, והתאריכים והמכפיל שלו יפסיקו להשתתף בתחזיות ובתמחור מבוססי אירועים כאשר שכבת האירועים פעילה.')}
        recovery={pageText(locale, 'Reactivate it from this list at any time, or restore the pre-change event snapshot from Restore changes.', 'ניתן להפעיל אותו מחדש מהרשימה בכל עת, או לשחזר את תמונת המצב מלפני השינוי דרך שחזור שינויים.')}
        confirmLabel={pageText(locale, 'Deactivate event', 'השבתת האירוע')}
        workingLabel={pageText(locale, 'Deactivating event', 'משבית את האירוע')}
        busy={busy}
        onCancel={() => setConfirmId(null)}
        onConfirm={confirmDeactivate}
      />
    </section>
  );
}

export default CalendarEventsList;
