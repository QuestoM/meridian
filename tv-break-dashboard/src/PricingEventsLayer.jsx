import React, { useState } from 'react';
import { Button, Tooltip } from '@mui/material';
import { pageText } from './advertisers-helpers';
import { LAYER_TEXT, readEventsLayer } from './pricing-layers-lib';

// The calendar-events pricing layer card on the Pricing page. Activation is
// owner-gated behind an inline confirm (same pattern as the rate-card reset)
// because turning it on changes forecast revenue on event days. On a server
// that predates the events layer the card degrades honestly: a clear chip and
// a disabled toggle with an explanation, never a fabricated off state.
function PricingEventsLayer({ state, locale, onToggle }) {
  const [confirmOn, setConfirmOn] = useState(false);
  const { supported, enabled, count } = readEventsLayer(state);

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
      <input
        type="checkbox"
        checked={supported && enabled}
        disabled={!supported}
        onChange={(event) => requestToggle(event.target.checked)}
      />
      {pageText(locale, 'On', 'הפעלה')}
    </label>
  );

  return (
    <div className="pricing-layer-card">
      <div className="pricing-layer-head">
        <div>
          <span className="pricing-layer-title">{pageText(locale, LAYER_TEXT.events.en, LAYER_TEXT.events.he)}</span>
          <p className="pricing-layer-desc">{pageText(locale, 'A price multiplier for days inside active calendar events. An operator assertion, not a measurement. Off until activated here.', LAYER_TEXT.events.descHe)}</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className={`pricing-chip ${chip}`}>{chipText}</span>
          {supported ? toggle : (
            <Tooltip title={pageText(locale, 'This server version does not carry the events pricing layer yet, so the toggle is disabled instead of showing an invented state.', 'גרסת השרת הזו עדיין אינה כוללת את שכבת האירועים בתמחור, ולכן ההפעלה מושבתת במקום להציג מצב מומצא.')} arrow>
              <span>{toggle}</span>
            </Tooltip>
          )}
        </div>
      </div>
      {supported && count !== null && (
        <p className="pricing-base-note">{pageText(locale, `${count} active events carry a price multiplier other than 1.0.`, `${count} אירועים פעילים נושאים מכפיל תמחור שונה מ-1.0.`)}</p>
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
