import React, { useState } from 'react';
import { Tooltip } from '@mui/material';
import { Button } from '../studio/actions';
import { ChevronDown } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { Figure, Name } from '../shell/bidi';
import { InputControl, Pressable } from '../studio/dom-controls';
import { LAYER_TEXT, eventDatesLabel, readEventsLayer } from './pricing-layers-lib';

// The expandable per-event list under the count line. When the server payload
// carries the event list, each active non-1.0 event shows its name, dates and
// multiplier; when the server sends only a count, the expanded body says so
// plainly instead of inventing rows.
function EventsList({ events, locale }) {
  if (events === null) {
    return (
      <p className="pricing-base-note">{pageText(locale, 'The server reported only a count, not the events themselves, so the list cannot be shown.', 'השרת דיווח על ספירה בלבד, ללא פירוט האירועים עצמם, ולכן לא ניתן להציג את הרשימה.')}</p>
    );
  }
  if (events.length === 0) {
    return (
      <p className="pricing-base-note">{pageText(locale, 'The server sent an event list, but none of the entries carries a usable non-1.0 multiplier.', 'השרת שלח רשימת אירועים, אך אף רשומה אינה נושאת מכפיל תקין השונה מ-1.0.')}</p>
    );
  }
  return (
    <ul className="pricing-events-list">
      {events.map((entry, index) => (
        <li key={`${entry.name || 'event'}-${entry.start || ''}-${index}`} className="pricing-break-row">
          <span>
            <Name>{entry.name || pageText(locale, 'Unnamed event', 'אירוע ללא שם')}</Name>
            {' '}
            <Figure className="src">{eventDatesLabel(entry, locale)}</Figure>
          </span>
          <Figure className="mult">x {entry.multiplier.toFixed(2)}</Figure>
        </li>
      ))}
    </ul>
  );
}

// The calendar-events pricing layer card on the Pricing page. Activation is
// owner-gated behind an inline confirm (same pattern as the rate-card reset)
// because turning it on changes forecast revenue on event days. On a server
// that predates the events layer the card degrades honestly: a clear chip and
// a disabled toggle with an explanation, never a fabricated off state.
function PricingEventsLayer({ state, locale, stagedEnabled, onToggle }) {
  const [confirmOn, setConfirmOn] = useState(false);
  const [listOpen, setListOpen] = useState(false);
  const { supported, enabled, count, events } = readEventsLayer(state);
  // The chip states what is in force on the saved card; the switch states what
  // the draft asks for. Bound to the saved value alone, the switch snapped back
  // the instant it was clicked while the effect panel below priced the change.
  const shownEnabled = stagedEnabled ?? enabled;

  const chip = supported ? (enabled ? 'live' : 'off') : 'empty';
  const chipText = supported
    ? (enabled ? pageText(locale, 'Live', 'פעיל') : pageText(locale, 'Wired off', 'מחווט-כבוי'))
    : pageText(locale, 'Not on this server', 'לא קיים בגרסת שרת זו');

  function requestToggle(nextEnabled) {
    if (nextEnabled) {
      setConfirmOn(true);
    } else {
      setConfirmOn(false);
      onToggle(false);
    }
  }

  const toggle = (
    <label className="pricing-toggle">
      <InputControl
        type="checkbox"
        checked={supported && shownEnabled}
        disabled={!supported}
        onChange={(event) => requestToggle(event.target.checked)}
      />
      {pageText(locale, 'On', 'הפעלה')}
    </label>
  );

  return (
    <div className="card pricing-layer-card">
      <div className="pricing-layer-head">
        <div>
          <span className="pricing-layer-title">{pageText(locale, LAYER_TEXT.events.en, LAYER_TEXT.events.he)}</span>
          <p className="pricing-layer-desc">{pageText(locale, 'A price multiplier for days inside active calendar events. An operator assertion, not a measurement. Off until activated here.', LAYER_TEXT.events.descHe)}</p>
        </div>
        <div className="pricing-layer-heading">
          <span className={`pricing-chip ${chip}`}>{chipText}</span>
          {supported ? toggle : (
            <Tooltip title={pageText(locale, 'This server version does not carry the events pricing layer yet, so the toggle is disabled instead of showing an invented state.', 'גרסת השרת הזו עדיין אינה כוללת את שכבת האירועים בתמחור, ולכן ההפעלה מושבתת במקום להציג מצב מומצא.')} arrow>
              <span>{toggle}</span>
            </Tooltip>
          )}
        </div>
      </div>
      {supported && count !== null && (
        <div className="pricing-events-summary">
          <p className="pricing-base-note compact">{pageText(locale, `${count} active events carry a price multiplier other than 1.0.`, `${count} אירועים פעילים נושאים מכפיל תמחור שונה מ-1.0.`)}</p>
          {count > 0 && (
            <Pressable
              type="button"
              className="secondary-button compact pricing-events-disclosure"
              aria-expanded={listOpen}
              onClick={() => setListOpen((value) => !value)}
            >
              <ChevronDown size={16} className={listOpen ? 'pricing-events-caret open' : 'pricing-events-caret'} />
              {listOpen ? pageText(locale, 'Hide the events', 'הסתרת האירועים') : pageText(locale, 'Show which events', 'אילו אירועים')}
            </Pressable>
          )}
        </div>
      )}
      {supported && count !== null && count > 0 && listOpen && (
        <EventsList events={events} locale={locale} />
      )}
      {supported && count === null && (
        <p className="pricing-base-note">{pageText(locale, 'The server did not report how many active events carry a non-1.0 multiplier, so no count is shown.', 'השרת לא דיווח כמה אירועים פעילים נושאים מכפיל שונה מ-1.0, ולכן לא מוצגת ספירה.')}</p>
      )}
      {confirmOn && (
        <span className="pricing-events-confirm" role="alertdialog">
          <span>{pageText(locale, 'Turning the events layer on changes forecast revenue on event days. The change applies from the next optimizer run and forecast.', 'הפעלת שכבת האירועים משנה את ההכנסה הצפויה בתחזית בימי אירועים. השינוי חל מריצת האופטימייזר והתחזית הבאות.')}</span>
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => { setConfirmOn(false); onToggle(true); }}>
            {pageText(locale, 'Confirm activation', 'אישור הפעלה')}
          </Button>
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmOn(false)}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </Button>
        </span>
      )}
    </div>
  );
}

export default PricingEventsLayer;
