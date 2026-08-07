import React, { useState } from 'react';
import { amount, isolate, localized, pick, vocabularyLabel, vocabularyMeaning } from './pacing-helpers';

// The make-good ledger: what was measured, what was offered against it, and who
// acted. Every row is a record with a state, and the states it may move to next
// are the only controls it shows, so the machine is legible from the screen rather
// than from documentation.
//
// The offer is entered inline on the row. It is not a dialog because it is not a
// destructive act and because a form that covers the figure it is about makes the
// person copy a number out of their own memory.

// A control is named by the act it performs, not by the state it lands in. The
// server publishes the states as nouns because a payload describes a record; a
// button is pressed by a person, so it takes the verb.
const ACTS = {
  offered: { en: 'Record an offer', he: 'רשמו הצעה' },
  settled: { en: 'Settle it', he: 'סגרו את הפיצוי' },
  declined: { en: 'Mark it declined', he: 'סמנו שנדחה' },
  withdrawn: { en: 'Withdraw it', he: 'בטלו את הפיצוי' },
};

function actWord(state, locale) {
  const act = ACTS[state];
  if (!act) return state;
  return pick(locale, act.en, act.he);
}

function Figures({ record, locale, vocabulary }) {
  const shortfall = record.shortfall;
  return (
    <div className="makegood-figures">
      <strong>
        {pick(
          locale,
          `${amount(shortfall.deficit_value, shortfall.unit, locale)} short`,
          `חסרים ${isolate(amount(shortfall.deficit_value, shortfall.unit, locale))}`,
        )}
      </strong>
      <small>
        {pick(
          locale,
          `${amount(shortfall.counted_value, shortfall.unit, locale)} counted of ${amount(shortfall.goal_value, shortfall.unit, locale)}`,
          `${isolate(amount(shortfall.counted_value, shortfall.unit, locale))} נספרו מתוך ${isolate(amount(shortfall.goal_value, shortfall.unit, locale))}`,
        )}
      </small>
      <span className={`makegood-kind ${shortfall.deficit_kind}`}
            title={vocabularyMeaning(vocabulary.deficit_kinds, shortfall.deficit_kind, locale)}>
        {vocabularyLabel(vocabulary.deficit_kinds, shortfall.deficit_kind, locale)}
      </span>
    </div>
  );
}

function OfferForm({ record, locale, busy, onSubmit, onCancel }) {
  const [value, setValue] = useState(String(record.shortfall.deficit_value ?? ''));
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [note, setNote] = useState('');
  return (
    <form
      className="makegood-offer"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit({
          state: 'offered',
          offer_value: Number(value),
          offer_window_start: start,
          offer_window_end: end,
          note,
        });
      }}
    >
      <label>
        <span>{pick(locale, 'Offer, in the shortfall unit', 'ההצעה, ביחידת החוסר')}</span>
        <input type="number" step="0.01" min="0" dir="ltr" value={value} onChange={(e) => setValue(e.target.value)} required />
      </label>
      <label>
        <span>{pick(locale, 'Window opens', 'החלון נפתח')}</span>
        <input type="date" dir="ltr" value={start} onChange={(e) => setStart(e.target.value)} />
      </label>
      <label>
        <span>{pick(locale, 'Window closes', 'החלון נסגר')}</span>
        <input type="date" dir="ltr" value={end} onChange={(e) => setEnd(e.target.value)} />
      </label>
      <label className="makegood-note">
        <span>{pick(locale, 'What was agreed', 'מה סוכם')}</span>
        <input type="text" maxLength={500} value={note} onChange={(e) => setNote(e.target.value)} />
      </label>
      <div className="makegood-offer-actions">
        <button type="submit" disabled={busy}>{pick(locale, 'Record the offer', 'רשמו את ההצעה')}</button>
        <button type="button" className="ghost" onClick={onCancel}>{pick(locale, 'Cancel', 'ביטול')}</button>
      </div>
    </form>
  );
}

function Offer({ record, locale }) {
  if (record.offer.value === null || record.offer.value === undefined) return null;
  const window = record.offer.window_start && record.offer.window_end
    ? `${record.offer.window_start} - ${record.offer.window_end}`
    : '';
  return (
    <p className="makegood-offer-record">
      {pick(
        locale,
        `Offered ${amount(record.offer.value, record.shortfall.unit, locale)}`,
        `הוצעו ${isolate(amount(record.offer.value, record.shortfall.unit, locale))}`,
      )}
      {window ? <span dir="ltr"> {window}</span> : null}
      {record.offer.offered_by ? ` ${pick(locale, 'by', 'על ידי')} ${record.offer.offered_by}` : ''}
      {record.offer.note ? ` ${record.offer.note}` : ''}
    </p>
  );
}

export default function MakeGoodLedger({ payload, locale, canEdit, editRefusal, busyId, onMove, onOpenCampaign }) {
  const [offering, setOffering] = useState('');
  const vocabulary = payload.vocabulary || {};
  const rows = payload.make_goods || [];

  return (
    <section className="makegood-ledger" aria-label={pick(locale, 'Make-good ledger', 'ספר פיצויי השידור')}>
      <p className="pacing-basis">
        {localized(payload.sign_off, 'reason', locale)}
        {' '}
        {localized(payload.sign_off, 'path_forward', locale)}
        {' '}
        {localized(payload.sign_off, 'offer_reserves_nothing', locale)}
      </p>

      {rows.length === 0 ? (
        <p className="pacing-empty">
          {pick(
            locale,
            'No make-good has been raised. A campaign with a measured shortfall carries the control that raises one.',
            'לא נפתח פיצוי שידור. קמפיין עם חוסר נמדד נושא את הפקד שפותח אחד.',
          )}
        </p>
      ) : null}

      {rows.map((record) => (
        <article className={`makegood-row ${record.state}`} key={record.make_good_id}>
          <div className="makegood-row-head">
            <span className={`makegood-state ${record.state}`}>
              {vocabularyLabel(vocabulary.states, record.state, locale)}
            </span>
            <button type="button" className="makegood-campaign" onClick={() => onOpenCampaign(record.campaign_id)}>
              {record.campaign_name || record.campaign_id}
            </button>
            <small className="makegood-flight" dir="ltr">
              {record.flight.starts_on} - {record.flight.ends_on}
            </small>
            {record.is_demo ? <span className="pacing-demo">{pick(locale, 'Demo', 'הדגמה')}</span> : null}
          </div>

          <Figures record={record} locale={locale} vocabulary={vocabulary} />
          <Offer record={record} locale={locale} />

          <small className="makegood-trail">
            {pick(
              locale,
              `Raised ${record.raised_at}${record.raised_by ? ` by ${record.raised_by}` : ''}. Counted as of ${record.shortfall.counted_as_of}.`,
              `נפתח ${isolate(record.raised_at)}${record.raised_by ? ` על ידי ${record.raised_by}` : ''}. נספר נכון ל${isolate(record.shortfall.counted_as_of)}.`,
            )}
          </small>

          {canEdit ? (
            <div className="makegood-actions">
              {record.next_states.map((state) => (
                <button
                  key={state}
                  type="button"
                  disabled={busyId === record.make_good_id}
                  onClick={() => (state === 'offered'
                    ? setOffering(record.make_good_id)
                    : onMove(record.make_good_id, { state }))}
                >
                  {actWord(state, locale)}
                </button>
              ))}
              {record.next_states.length === 0 ? (
                <span className="makegood-closed">{pick(locale, 'Closed', 'סגור')}</span>
              ) : null}
            </div>
          ) : (
            <span className="pacing-remedy-note">{editRefusal}</span>
          )}

          {offering === record.make_good_id ? (
            <OfferForm
              record={record}
              locale={locale}
              busy={busyId === record.make_good_id}
              onCancel={() => setOffering('')}
              onSubmit={(payloadOut) => {
                setOffering('');
                onMove(record.make_good_id, payloadOut);
              }}
            />
          ) : null}
        </article>
      ))}
    </section>
  );
}
