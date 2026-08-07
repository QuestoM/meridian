import React, { useState } from 'react';
import { amount, instant, isolate, localized, pair, pick, vocabularyLabel, vocabularyMeaning } from './pacing-helpers';

// The decision ledger: what was measured, what was decided about it, and who
// acted. Every row is a record with a state, and the states it may move to next
// are the only controls it shows, so the machine is legible from the screen rather
// than from documentation.
//
// Two kinds of record share it, because the job has two endings. A make-good is
// compensating delivery raised against a measured shortfall. An acceptance is the
// recorded decision that the risk stands. Reading them in one timeline is the
// point: a person answering for a campaign has to see both.
//
// The offer is entered inline on the row. It is not a dialog because it is not a
// destructive act and because a form that covers the figure it is about makes the
// person copy a number out of their own memory.

const ACCEPTANCE = 'acceptance';

// A control is named by the act it performs, not by the state it lands in. The
// server publishes the states as nouns because a payload describes a record; a
// button is pressed by a person, so it takes the verb. The same state ends the
// two kinds differently, so the verb is chosen by kind as well.
const ACTS = {
  offered: { en: 'Record an offer', he: 'רשמו הצעה' },
  settled: { en: 'Settle it', he: 'סגרו את הפיצוי' },
  declined: { en: 'Mark it declined', he: 'סמנו שנדחה' },
  withdrawn: { en: 'Withdraw it', he: 'בטלו את הפיצוי' },
};
const ACCEPTANCE_ACTS = {
  withdrawn: { en: 'Revoke the decision', he: 'בטלו את ההחלטה' },
};

function actWord(state, kind, locale) {
  const table = kind === ACCEPTANCE ? { ...ACTS, ...ACCEPTANCE_ACTS } : ACTS;
  const act = table[state];
  if (!act) return state;
  return pick(locale, act.en, act.he);
}

function headlineFigure(record, locale) {
  const shortfall = record.shortfall;
  const value = amount(shortfall.deficit_value, shortfall.unit, locale);
  if (record.kind === ACCEPTANCE) {
    return pick(locale, `${value} behind when the risk was taken on`, `פיגור של ${isolate(value)} בעת קבלת הסיכון`);
  }
  return pick(locale, `${value} short`, `חסרים ${isolate(value)}`);
}

function Figures({ record, locale, vocabulary }) {
  const shortfall = record.shortfall;
  return (
    <div className="makegood-figures">
      <strong>{headlineFigure(record, locale)}</strong>
      <small>
        {pick(
          locale,
          `counted ${pair(shortfall.counted_value, shortfall.goal_value, shortfall.unit, locale)}`,
          `נספרו ${pair(shortfall.counted_value, shortfall.goal_value, shortfall.unit, locale)}`,
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

// The separating space lives outside the window, never inside it. Inside, the
// bidi reorder puts it on the far edge of the run and the window's last digits
// sit against the offer's first, which was measured rendering an offer of 0.6
// against a window ending 2025-05-10 as the string 2025-05-100.6.
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
      {window ? ' ' : ''}
      {window ? <span dir="ltr">{window}</span> : null}
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
              `Raised ${instant(record.raised_at)}${record.raised_by ? ` by ${record.raised_by}` : ''}. Counted as of ${instant(record.shortfall.counted_as_of)}.`,
              `נפתח ${isolate(instant(record.raised_at))}${record.raised_by ? ` על ידי ${record.raised_by}` : ''}. נספר נכון ל־${isolate(instant(record.shortfall.counted_as_of))}.`,
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
