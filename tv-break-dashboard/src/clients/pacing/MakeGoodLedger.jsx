import React, { useState } from 'react';
import { Figure, Name } from '../../shell/bidi';
import { amount, instant, isolate, localized, pair, pick, unitWord, vocabularyLabel, vocabularyMeaning } from './pacing-helpers';

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

// What a record's own timeline calls its opening. A make-good is raised against a
// shortfall and an acceptance is recorded about one, and a single verb for both
// would tell a Hebrew reader that a decision was opened.
function openedWord(kind, locale) {
  if (kind === ACCEPTANCE) return pick(locale, 'Recorded', 'נרשמה');
  return pick(locale, 'Raised', 'נפתח');
}

function headlineFigure(record, locale) {
  const shortfall = record.shortfall;
  const value = amount(shortfall.deficit_value, shortfall.unit, locale);
  if (record.kind === ACCEPTANCE) {
    return pick(locale, `${value} behind when the risk was taken on`, `פיגור של ${value} בעת קבלת הסיכון`);
  }
  return pick(locale, `${value} short`, `חסרים ${value}`);
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
      {/* The unit is named on the field rather than described. "In the shortfall
          unit" makes a person look up the row to learn what they are typing. The
          word comes from the same helper that puts a unit after a printed figure,
          because the ledger read carries no unit vocabulary and reaching for one
          that is not there put the store's key on a label. */}
      <label>
        <span>
          {pick(
            locale,
            `Offer, in ${unitWord(record.shortfall.unit, locale)}`,
            `ההצעה, ב${unitWord(record.shortfall.unit, locale)}`,
          )}
        </span>
        <input type="number" step="0.01" min="0" className="bidi-figure" value={value} onChange={(e) => setValue(e.target.value)} required />
      </label>
      <label>
        <span>{pick(locale, 'Window opens', 'החלון נפתח')}</span>
        <input type="date" className="bidi-figure" value={start} onChange={(e) => setStart(e.target.value)} />
      </label>
      <label>
        <span>{pick(locale, 'Window closes', 'החלון נסגר')}</span>
        <input type="date" className="bidi-figure" value={end} onChange={(e) => setEnd(e.target.value)} />
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

// Closing a record without a delivery is the one act here that cannot be undone,
// so it is the one act that asks before it fires. The reference is Stripe, which
// requires a reason on every refund, requires a note when that reason is the open
// one, and confirms a cancellation with a button that names the act rather than
// with the word Yes. Before this, Withdraw it and Revoke the decision fired on a
// single click with an optional free-text note and no reason at all, and a
// withdrawn record was unauditable.
//
// The reasons are the ledger's own published list, so this form offers exactly
// what the store will accept and never a fifth option it would refuse.
function CloseForm({ record, state, locale, busy, vocabulary, onSubmit, onCancel }) {
  const offered = ((vocabulary.close_reasons || {})[state]) || [];
  const openReason = vocabulary.reason_needing_a_note || 'other';
  const [reason, setReason] = useState('');
  const [note, setNote] = useState('');
  const needsNote = reason === openReason;
  return (
    <form
      className="makegood-close-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit({ state, reason, note });
      }}
    >
      <p className="makegood-close-ask">
        {pick(
          locale,
          'This closes the record and nothing reopens it. The ledger keeps the row and records who closed it and why.',
          'הפעולה סוגרת את הרשומה ואין דרך לפתוח אותה מחדש. ספר ההחלטות שומר את השורה ורושם מי סגר אותה ומדוע.',
        )}
      </p>
      <label>
        <span>{pick(locale, 'Why', 'מדוע')}</span>
        <select value={reason} required onChange={(e) => setReason(e.target.value)}>
          <option value="">{pick(locale, 'Choose a reason', 'בחרו סיבה')}</option>
          {offered.map((entry) => (
            <option key={entry.value} value={entry.value}>{pick(locale, entry.label_en, entry.label_he)}</option>
          ))}
        </select>
      </label>
      <label className="makegood-note">
        <span>{needsNote ? pick(locale, 'What happened', 'מה קרה') : pick(locale, 'What happened, if it helps', 'מה קרה, אם זה עוזר')}</span>
        <input type="text" maxLength={500} required={needsNote} value={note} onChange={(e) => setNote(e.target.value)} />
      </label>
      <div className="makegood-offer-actions">
        <button type="submit" disabled={busy}>
          {pick(locale, `Yes, ${actWord(state, record.kind, 'en').toLowerCase()}`, `כן, ${actWord(state, record.kind, 'he')}`)}
        </button>
        <button type="button" className="ghost" onClick={onCancel}>{pick(locale, 'Keep it', 'השאירו אותה')}</button>
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
        `הוצעו ${amount(record.offer.value, record.shortfall.unit, locale)}`,
      )}
      {window ? ' ' : ''}
      {window ? <Figure>{window}</Figure> : null}
      {record.offer.offered_by ? ` ${pick(locale, 'by', 'על ידי')} ${record.offer.offered_by}` : ''}
      {record.offer.note ? ` ${record.offer.note}` : ''}
    </p>
  );
}

// Why a record was closed, on the record. A reason the store now requires and
// nothing rendered would be a field written for a database rather than for the
// person who has to answer for the decision next quarter.
function Closure({ record, locale, vocabulary }) {
  if (!record.close_reason) return null;
  const reasons = (vocabulary.close_reasons || {})[record.state] || [];
  return (
    <p className="makegood-closure">
      {pick(locale, 'Closed as: ', 'נסגרה בתור: ')}
      {vocabularyLabel(reasons, record.close_reason, locale)}
      {record.closed_by ? ` ${pick(locale, 'by', 'על ידי')} ${record.closed_by}` : ''}
    </p>
  );
}

export default function MakeGoodLedger({ payload, locale, canEdit, editRefusal, busyId, onMove, onOpenCampaign }) {
  const [offering, setOffering] = useState('');
  // Which record is being closed and into which state, so the confirmation is
  // about one act on one row and never about the last button pressed.
  const [closing, setClosing] = useState({ id: '', state: '' });
  const vocabulary = payload.vocabulary || {};
  const asks = vocabulary.reason_required || [];
  // Both endings, in one timeline. Reading only make_goods was measured hiding
  // every recorded acceptance: the tab badge counted one record and the list said
  // no make-good had been raised, which is the exact pair of screens the second
  // ending exists to make impossible. make_goods is the fallback for a payload
  // written before this read carried both.
  const rows = payload.decisions || payload.make_goods || [];
  const accepted = payload.accepted_count || 0;

  return (
    <section className="makegood-ledger" aria-label={pick(locale, 'Decision ledger', 'ספר ההחלטות')}>
      <p className="pacing-basis">
        {localized(payload.sign_off, 'reason', locale)}
        {' '}
        {localized(payload.sign_off, 'path_forward', locale)}
        {' '}
        {localized(payload.sign_off, 'offer_reserves_nothing', locale)}
      </p>
      {accepted ? (
        <p className="pacing-basis">{localized(payload, 'acceptance_means', locale)}</p>
      ) : null}

      {rows.length === 0 ? (
        <p className="pacing-empty">
          {/* The old sentence promised a control this data does not carry.
              Measured on the shipped board, 0 of 56 rows reach a raise, because a
              make-good is owed only once the whole log is counted. Saying the
              rule tells a reader why they will not find the control, instead of
              sending them to look for it. */}
          {pick(
            locale,
            'Nothing has been decided yet. A make-good is offered on a campaign only once every remaining broadcast day carries a source and the flight still falls short, so a flight with unsourced days ahead of it offers the booking instead. Every campaign the board asks a decision about carries the control that records the risk as taken on.',
            'עדיין לא הוכרעה אף החלטה. פיצוי שידור מוצע בקמפיין רק כשלכל יום שידור שנותר יש מקור והטיסה עדיין אינה מגיעה ליעד, ולכן טיסה שלפניה ימים בלי מקור מציעה במקומו את ההזמנה. כל קמפיין שהלוח מבקש עליו החלטה נושא את הפקד שרושם את קבלת הסיכון.',
          )}
        </p>
      ) : null}

      {rows.map((record) => (
        <article className={`makegood-row ${record.state}`} key={record.make_good_id}>
          <div className="makegood-row-head">
            <span className={`makegood-state ${record.state}`}>
              {vocabularyLabel(vocabulary.states, record.state, locale)}
            </span>
            {/* Which of the two endings this row is. The state words already
                differ, and a person scanning a mixed timeline should not have to
                read a state to learn which kind of record they are looking at. */}
            <span className={`makegood-kindmark ${record.kind}`}
                  title={vocabularyMeaning(vocabulary.kinds, record.kind, locale)}>
              {vocabularyLabel(vocabulary.kinds, record.kind, locale)}
            </span>
            <button type="button" className="makegood-campaign"
                    onClick={() => onOpenCampaign(record.campaign_id)}>
              <Name>{record.campaign_name || record.campaign_id}</Name>
            </button>
            <small className="makegood-flight">
              <Figure>{record.flight.starts_on} - {record.flight.ends_on}</Figure>
            </small>
            {record.is_demo ? <span className="pacing-demo">{pick(locale, 'Demo', 'הדגמה')}</span> : null}
          </div>

          <Figures record={record} locale={locale} vocabulary={vocabulary} />
          <Offer record={record} locale={locale} />
          <Closure record={record} locale={locale} vocabulary={vocabulary} />

          <small className="makegood-trail">
            {pick(
              locale,
              `${openedWord(record.kind, locale)} ${instant(record.raised_at)}${record.raised_by ? ` by ${record.raised_by}` : ''}. Counted as of ${instant(record.shortfall.counted_as_of)}.`,
              `${openedWord(record.kind, locale)} ${isolate(instant(record.raised_at))}${record.raised_by ? ` על ידי ${record.raised_by}` : ''}. נספר נכון ל־${isolate(instant(record.shortfall.counted_as_of))}.`,
            )}
          </small>

          {canEdit ? (
            <div className="makegood-actions">
              {record.next_states.map((state) => (
                <button
                  key={state}
                  type="button"
                  disabled={busyId === record.make_good_id}
                  onClick={() => {
                    if (state === 'offered') return setOffering(record.make_good_id);
                    if (asks.indexOf(state) >= 0) return setClosing({ id: record.make_good_id, state });
                    return onMove(record.make_good_id, { state });
                  }}
                >
                  {actWord(state, record.kind, locale)}
                </button>
              ))}
              {record.next_states.length === 0 ? (
                <span className="makegood-closed">{pick(locale, 'Closed', 'סגור')}</span>
              ) : null}
            </div>
          ) : (
            <span className="pacing-remedy-note">{editRefusal}</span>
          )}

          {closing.id === record.make_good_id ? (
            <CloseForm
              record={record}
              state={closing.state}
              locale={locale}
              vocabulary={vocabulary}
              busy={busyId === record.make_good_id}
              onCancel={() => setClosing({ id: '', state: '' })}
              onSubmit={async (payloadOut) => {
                const landed = await onMove(record.make_good_id, payloadOut);
                if (landed) setClosing({ id: '', state: '' });
              }}
            />
          ) : null}

          {offering === record.make_good_id ? (
            <OfferForm
              record={record}
              locale={locale}
              busy={busyId === record.make_good_id}
              onCancel={() => setOffering('')}
              onSubmit={async (payloadOut) => {
                // The form closes when the move landed, and only then. It used to
                // close first, so a refused offer took the value, the window and
                // the note down with it and the reader typed all three again.
                const landed = await onMove(record.make_good_id, payloadOut);
                if (landed) setOffering('');
              }}
            />
          ) : null}
        </article>
      ))}
    </section>
  );
}
