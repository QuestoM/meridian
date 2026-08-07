import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Check, Plus, Trash2, X } from 'lucide-react';
import { pageText } from '../shell/format';
import { loadOnboardingOptions, onboardClient } from './clients-api';
import { localized, refusalText, vocabularyLabel } from './clients-money-helpers';

// JS-5 in one form. The agency, the client under it, the campaign, its flights
// and its terms are one submit, because the measured failure today is that two
// of the three entities have no creation path and the third is linked through a
// different endpoint on a different screen.
//
// Nothing here can create a duplicate. An agency the product already holds is
// reused and reported as reused, a client already linked is left alone, and a
// campaign whose name and client match an existing one is refused with the id
// that already holds it. The result panel says which of the three happened.

const EMPTY_FLIGHT = { starts_on: '', ends_on: '', goal_kind: 'spots', goal_value: '' };

// An empty scope reads as ANY (agency rule) or nothing (campaign term).
const NO_WEEKDAY_AGENCY = [
  'No weekday is selected. Submitting like this is refused: an agency condition with no weekday scope covers every day.',
  'לא נבחר יום בשבוע. שליחה כך תסורב: תנאי סוכנות ללא היקף ימים חל על כל יום.',
];
const NO_WEEKDAY_TERM = [
  'No weekday is selected. Submitting like this is refused: the discount percent would have no day it covers.',
  'לא נבחר יום בשבוע. שליחה כך תסורב: אחוז ההנחה יהיה ללא יום שהוא חל עליו.',
];
const NO_DISCOUNT_TO_SCOPE = [
  'No weekday is selected. Nothing is refused, because there is no discount percent to give a day to.',
  'לא נבחר יום בשבוע. דבר אינו נדחה, כיוון שאין אחוז הנחה שצריך לתת לו יום.',
];

// The percent is half of the rule the endpoint enforces: check_weekday_scope
// returns without refusing anything when it is zero or blank. So a flow carrying
// no discount is told nothing is refused, because nothing is, and the refusal
// sentence is kept for the case that really raises it.
function weekdayCoverage(selected, options, locale, asAgencyRule, percent) {
  const amount = Number(percent);
  const discounting = Number.isFinite(amount) && amount !== 0;
  if (!selected.length) {
    if (!discounting) {
      return pageText(locale, ...NO_DISCOUNT_TO_SCOPE);
    }
    return pageText(locale, ...(asAgencyRule ? NO_WEEKDAY_AGENCY : NO_WEEKDAY_TERM));
  }
  const names = options.filter((day) => selected.includes(day.key)).map((day) => (locale === 'he' ? day.he : day.en));
  return pageText(locale, `Covers ${names.join(', ')}.`, `חל על ${names.join(', ')}.`);
}

// The word for the object a refusal names. Two of the refusals this flow can
// raise name a record that already exists and tell the reader to open it, and
// both of them arrived as a sentence with no way to the thing they named. A kind
// this surface cannot open has no word here and grows no control, so a refusal
// never carries a button that goes nowhere.
function openWord(kind, locale) {
  if (kind === 'campaign') {
    return pageText(locale, 'Open that campaign', 'פתחו את הקמפיין הזה');
  }
  if (kind === 'agency') {
    return pageText(locale, 'Open that agency record', 'פתחו את כרטיס הסוכנות');
  }
  return '';
}

// One refusal, with the way to the object it names when the endpoint sent one.
// It is its own component because a refusal is the one state of this flow that
// cannot be reached by rendering the form, and a state that cannot be rendered
// cannot be measured.
export function RefusalNotice({ error, opens, locale, onOpen }) {
  if (!error) {
    return null;
  }
  const word = opens && onOpen ? openWord(opens.kind, locale) : '';
  return (
    <p className="clients-error" role="alert">
      <span>{error}</span>
      {word ? (
        <button type="button" className="clients-retry" onClick={() => onOpen(opens)}>
          {word}
        </button>
      ) : null}
    </p>
  );
}

function Field({ label, value, onChange, type = 'text', ltr = false, list, required = false, hint }) {
  return (
    <label className="clients-field">
      <span>{label}</span>
      <input
        type={type}
        value={value}
        list={list}
        required={required}
        dir={ltr ? 'ltr' : 'auto'}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}

export default function OnboardClientFlow({ locale, prefill, onClose, onDone, onOpenRefused }) {
  const he = locale === 'he';
  const [options, setOptions] = useState(null);
  const [agencyMode, setAgencyMode] = useState('existing');
  const [agencyId, setAgencyId] = useState('');
  const [agency, setAgency] = useState({
    name: '',
    agency_type: '',
    contact_name: '',
    contact_role: '',
    contact_phone: '',
    contact_email: '',
    vat_id: '',
    rebate_percent: '',
    commission_percent: '',
    credit_limit_ils: '',
    payment_terms_days: '60',
  });
  const [advertiser, setAdvertiser] = useState((prefill && prefill.advertiser) || '');
  const [campaign, setCampaign] = useState({ name: '', starts_on: '', ends_on: '', rebate_percent: '' });
  const [discount, setDiscount] = useState({ percent: '', weekdays: ['6'], asAgencyRule: false });
  const [flights, setFlights] = useState([{ ...EMPTY_FLIGHT }]);
  const [state, setState] = useState({ status: 'idle', error: '', opens: null, result: null });
  // A choice the operator already made outranks the read. The options land after
  // the panel is on screen, so applying the default on arrival overwrote it:
  // measured, "a new agency" chosen at t+0 reverted to the first agency on the
  // list when the read landed, and a submit after that revert booked the
  // campaign under an agency nobody picked, with that agency's rebate. The two
  // halves are tracked apart, because the agency id sitting behind a hidden
  // select is not a choice anybody made and must still get its default.
  const chosen = useRef({ mode: false, agency: false });

  useEffect(() => {
    let active = true;
    loadOnboardingOptions()
      .then((payload) => {
        if (!active) return;
        setOptions(payload);
        const wanted = prefill && prefill.agencyId
          ? payload.agencies.find((entry) => entry.agency_id === prefill.agencyId)
          : null;
        if (!chosen.current.agency) {
          setAgencyId(wanted ? wanted.agency_id : (payload.agencies.length ? payload.agencies[0].agency_id : ''));
        }
        if (!chosen.current.mode) {
          setAgencyMode(payload.agencies.length ? 'existing' : 'new');
        }
      })
      .catch(() => {
        if (!active) return;
        setState({ status: 'idle', error: pageText(locale, 'The form could not load its choices.', 'הטופס לא הצליח לטעון את האפשרויות.'), opens: null, result: null });
      });
    return () => { active = false; };
  }, [locale, prefill]);

  // Every write to the agency block goes through one of these three, so making
  // the choice is what records it and there is no flag to remember to set.
  function chooseAgencyMode(mode) {
    chosen.current.mode = true;
    setAgencyMode(mode);
  }

  function chooseAgency(id) {
    chosen.current.agency = true;
    setAgencyId(id);
  }

  function editAgency(patch) {
    chosen.current.mode = true;
    setAgency((current) => ({ ...current, ...patch }));
  }

  const weekdays = useMemo(() => (options && options.weekdays) || [], [options]);
  const goalKinds = useMemo(() => (options && options.goal_kinds) || ['spots'], [options]);
  const goalWords = useMemo(() => (options && options.goal_kind_vocabulary) || [], [options]);

  function toggleWeekday(key) {
    setDiscount((current) => ({
      ...current,
      weekdays: current.weekdays.includes(key)
        ? current.weekdays.filter((entry) => entry !== key)
        : [...current.weekdays, key],
    }));
  }

  async function submit(event) {
    event.preventDefault();
    setState({ status: 'saving', error: '', opens: null, result: null });
    const payload = {
      agency: agencyMode === 'existing'
        ? { agency_id: agencyId }
        : {
          agency_id: options ? options.next_agency_id : '',
          name: agency.name,
          agency_type: agency.agency_type,
          contact_name: agency.contact_name,
          contact_role: agency.contact_role,
          contact_phone: agency.contact_phone,
          contact_email: agency.contact_email,
          vat_id: agency.vat_id,
          rebate_percent: Number(agency.rebate_percent || 0),
          commission_percent: Number(agency.commission_percent || 0),
          credit_limit_ils: Number(agency.credit_limit_ils || 0),
          payment_terms_days: Number(agency.payment_terms_days || 60),
        },
      advertiser,
      campaign_name: campaign.name,
      campaign_starts_on: campaign.starts_on,
      campaign_ends_on: campaign.ends_on,
      rebate_percent: campaign.rebate_percent === '' ? null : Number(campaign.rebate_percent),
      surcharge_discount_percent: discount.percent === '' ? null : Number(discount.percent),
      surcharge_weekdays: discount.weekdays.join(','),
      apply_surcharge_as_agency_rule: discount.asAgencyRule,
      flights: flights
        .filter((flight) => flight.starts_on && flight.ends_on && flight.goal_value !== '')
        .map((flight) => ({ ...flight, goal_value: Number(flight.goal_value) })),
    };
    try {
      const result = await onboardClient(payload);
      setState({ status: 'done', error: '', opens: null, result });
    } catch (error) {
      // The record the refusal names, as the endpoint addressed it. Two of these
      // refusals tell the reader to open something that already exists, so the
      // address travels with the sentence and the notice grows the way to it.
      setState({ status: 'idle', error: refusalText(error, locale), opens: error.opens || null, result: null });
    }
  }

  if (state.status === 'done') {
    const result = state.result;
    return (
      <aside className="clients-record clients-onboard" dir={he ? 'rtl' : 'ltr'} role="dialog">
        <header className="clients-record-head">
          <h3>{pageText(locale, 'The client is on file', 'הלקוח נקלט')}</h3>
          <button type="button" className="clients-icon-button" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
            <X size={15} aria-hidden="true" />
          </button>
        </header>
        <ul className="clients-result">
          <li>
            <Check size={13} aria-hidden="true" />
            <strong>{result.agency.agency_id}</strong>
            <span>{result.agency.outcome === 'created'
              ? pageText(locale, 'agency created', 'סוכנות נוצרה')
              : pageText(locale, 'agency already existed, reused', 'הסוכנות כבר קיימת, נעשה בה שימוש')}</span>
          </li>
          <li>
            <Check size={13} aria-hidden="true" />
            <strong>{result.advertiser.advertiser}</strong>
            <span>{result.advertiser.outcome === 'linked'
              ? pageText(locale, 'client linked to the agency', 'הלקוח שויך לסוכנות')
              : pageText(locale, 'client was already linked', 'הלקוח כבר היה משויך')}</span>
          </li>
          <li>
            <Check size={13} aria-hidden="true" />
            <strong>{pageText(locale, 'Client record', 'כרטיס הלקוח')}</strong>
            <span>{(result.advertiser.identity || {}).outcome === 'registered'
              ? pageText(locale, 'named as a client of this operator, so a rule can be written against its name', 'נרשם כלקוח של המפעיל, ולכן אפשר לכתוב כלל על שמו')
              : pageText(locale, 'already a named client, so nothing was written', 'כבר לקוח בשם, ולכן לא נכתב דבר')}</span>
          </li>
          <li>
            <Check size={13} aria-hidden="true" />
            <strong>{result.campaign.campaign_id}</strong>
            <span>{pageText(locale, `campaign created with ${result.flights.length} flights`, `קמפיין נוצר עם ⁦${result.flights.length}⁩ טיסות שידור`)}</span>
          </li>
          <li>
            <Check size={13} aria-hidden="true" />
            <strong>{pageText(locale, 'Terms', 'תנאים')}</strong>
            <span>{result.discount.outcome === 'priced_as_agency_condition'
              ? localized(result.discount, 'covers', locale)
              : localized(result.discount, 'note', locale)}</span>
          </li>
        </ul>
        <button type="button" className="clients-primary" onClick={() => onDone(result.campaign.advertiser)}>
          {pageText(locale, 'Open the client record', 'פתחו את כרטיס הלקוח')}
        </button>
      </aside>
    );
  }

  return (
    <aside className="clients-record clients-onboard" dir={he ? 'rtl' : 'ltr'} role="dialog">
      <header className="clients-record-head">
        <h3>{pageText(locale, 'Onboard a client', 'קליטת לקוח')}</h3>
        <button type="button" className="clients-icon-button" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
          <X size={15} aria-hidden="true" />
        </button>
      </header>
      <p className="clients-basis-note">
        {pageText(
          locale,
          'One insertion order, one submit. The agency, the client and the campaign are created and linked together.',
          'הזמנה אחת, שליחה אחת. הסוכנות, הלקוח והקמפיין נוצרים ומקושרים יחד.',
        )}
      </p>

      <form onSubmit={submit} className="clients-form">
        <fieldset>
          <legend>{pageText(locale, 'Agency', 'סוכנות')}</legend>
          <div className="clients-radio-row">
            <label>
              <input type="radio" checked={agencyMode === 'existing'} onChange={() => chooseAgencyMode('existing')} />
              {pageText(locale, 'An agency we already work with', 'סוכנות שאנחנו כבר עובדים איתה')}
            </label>
            <label>
              <input type="radio" checked={agencyMode === 'new'} onChange={() => chooseAgencyMode('new')} />
              {pageText(locale, 'A new agency', 'סוכנות חדשה')}
            </label>
          </div>
          {agencyMode === 'existing' ? (
            <label className="clients-field">
              <span>{pageText(locale, 'Agency', 'סוכנות')}</span>
              <select value={agencyId} onChange={(event) => chooseAgency(event.target.value)}>
                {(options ? options.agencies : []).map((entry) => (
                  <option key={entry.agency_id} value={entry.agency_id}>
                    {`${entry.name} (${entry.agency_id}), ${entry.rebate_percent}%`}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <>
            <div className="clients-field-grid">
              <Field label={pageText(locale, 'Agency name', 'שם הסוכנות')} value={agency.name} onChange={(value) => editAgency({ name: value })} required hint={pageText(locale, 'exactly as the daily file spells it', 'בדיוק כפי שהקובץ היומי מאיית')} />
              <label className="clients-field">
                <span>{pageText(locale, 'Agency type', 'סוג הסוכנות')}</span>
                <select value={agency.agency_type} onChange={(event) => editAgency({ agency_type: event.target.value })}>
                  <option value="">{pageText(locale, 'Not stated', 'לא צוין')}</option>
                  {(options ? options.agency_types : []).map((type) => <option key={type} value={type}>{type}</option>)}
                </select>
              </label>
              <Field label={pageText(locale, 'Rebate percent', 'אחוז רבייט')} value={agency.rebate_percent} onChange={(value) => editAgency({ rebate_percent: value })} type="number" ltr />
              <Field label={pageText(locale, 'Commission percent', 'אחוז עמלה')} value={agency.commission_percent} onChange={(value) => editAgency({ commission_percent: value })} type="number" ltr />
              <Field label={pageText(locale, 'Payment terms in days', 'תנאי תשלום בימים')} value={agency.payment_terms_days} onChange={(value) => editAgency({ payment_terms_days: value })} type="number" ltr />
              <Field label={pageText(locale, 'Credit limit in shekels', 'מסגרת אשראי בשקלים')} value={agency.credit_limit_ils} onChange={(value) => editAgency({ credit_limit_ils: value })} type="number" ltr />
              <Field label={pageText(locale, 'Contact name', 'שם איש קשר')} value={agency.contact_name} onChange={(value) => editAgency({ contact_name: value })} />
              <Field label={pageText(locale, 'Contact role', 'תפקיד איש הקשר')} value={agency.contact_role} onChange={(value) => editAgency({ contact_role: value })} />
              <Field label={pageText(locale, 'Contact phone', 'טלפון איש קשר')} value={agency.contact_phone} onChange={(value) => editAgency({ contact_phone: value })} ltr />
              <Field label={pageText(locale, 'Contact email', 'דוא״ל איש קשר')} value={agency.contact_email} onChange={(value) => editAgency({ contact_email: value })} ltr />
              <Field label={pageText(locale, 'VAT id', 'ח״פ / עוסק')} value={agency.vat_id} onChange={(value) => editAgency({ vat_id: value })} ltr />
            </div>
            <p className="clients-basis-note">
              {pageText(
                locale,
                'A second contact, the address and the aliases are on the agency card once the agency exists.',
                'איש קשר שני, הכתובת והכינויים נמצאים בכרטיס הסוכנות ברגע שהסוכנות קיימת.',
              )}
            </p>
            </>
          )}
        </fieldset>

        <fieldset>
          <legend>{pageText(locale, 'Client', 'לקוח')}</legend>
          <Field
            label={pageText(locale, 'Client name', 'שם הלקוח')}
            value={advertiser}
            onChange={setAdvertiser}
            list="clients-known-advertisers"
            required
            hint={pageText(locale, 'pick one we have seen on air, or type a new one', 'בחרו לקוח שראינו בשידור, או הקלידו חדש')}
          />
          <datalist id="clients-known-advertisers">
            {(options ? options.advertisers : []).map((entry) => (
              <option key={entry.advertiser} value={entry.advertiser} />
            ))}
          </datalist>
        </fieldset>

        <fieldset>
          <legend>{pageText(locale, 'Campaign', 'קמפיין')}</legend>
          <div className="clients-field-grid">
            <Field label={pageText(locale, 'Campaign name', 'שם הקמפיין')} value={campaign.name} onChange={(value) => setCampaign({ ...campaign, name: value })} required />
            <Field label={pageText(locale, 'Starts on', 'מתחיל ב')} value={campaign.starts_on} onChange={(value) => setCampaign({ ...campaign, starts_on: value })} type="date" ltr required />
            <Field label={pageText(locale, 'Ends on', 'מסתיים ב')} value={campaign.ends_on} onChange={(value) => setCampaign({ ...campaign, ends_on: value })} type="date" ltr required />
            <Field label={pageText(locale, 'Campaign rebate percent', 'אחוז רבייט לקמפיין')} value={campaign.rebate_percent} onChange={(value) => setCampaign({ ...campaign, rebate_percent: value })} type="number" ltr />
          </div>
        </fieldset>

        <fieldset>
          <legend>{pageText(locale, 'Weekday surcharge discount', 'הנחה על תוספת יום בשבוע')}</legend>
          <div className="clients-field-grid">
            <Field label={pageText(locale, 'Discount percent off the surcharge', 'אחוז הנחה מהתוספת')} value={discount.percent} onChange={(value) => setDiscount({ ...discount, percent: value })} type="number" ltr />
          </div>
          <div className="clients-weekdays">
            {weekdays.map((day) => (
              <button
                key={day.key}
                type="button"
                className={discount.weekdays.includes(day.key) ? 'active' : ''}
                onClick={() => toggleWeekday(day.key)}
              >
                {locale === 'he' ? day.he : day.en}
              </button>
            ))}
          </div>
          <p className="clients-basis-note" role="status">
            {weekdayCoverage(discount.weekdays, weekdays, locale, discount.asAgencyRule, discount.percent)}
          </p>
          <label className="clients-checkbox">
            <input type="checkbox" checked={discount.asAgencyRule} onChange={(event) => setDiscount({ ...discount, asAgencyRule: event.target.checked })} />
            {pageText(locale, 'Apply it as an agency rule so it prices spots', 'החילו ככלל סוכנות כדי שיתמחר תשדירים')}
          </label>
          <p className="clients-basis-note">
            {pageText(
              locale,
              'Left off, the discount is stored as the agreed term and prices nothing. Applied, it becomes an agency condition and covers every campaign bought through that agency.',
              'ללא סימון, ההנחה נשמרת כתנאי מוסכם ואינה מתמחרת דבר. עם סימון, היא הופכת לתנאי סוכנות וחלה על כל קמפיין שנקנה דרך אותה סוכנות.',
            )}
          </p>
        </fieldset>

        <fieldset>
          <legend>{pageText(locale, 'Flights', 'טיסות שידור')}</legend>
          {flights.map((flight, index) => (
            <div className="clients-flight-row" key={`flight-${index}`}>
              <Field label={pageText(locale, 'From', 'מתאריך')} value={flight.starts_on} onChange={(value) => setFlights(flights.map((entry, position) => (position === index ? { ...entry, starts_on: value } : entry)))} type="date" ltr />
              <Field label={pageText(locale, 'To', 'עד תאריך')} value={flight.ends_on} onChange={(value) => setFlights(flights.map((entry, position) => (position === index ? { ...entry, ends_on: value } : entry)))} type="date" ltr />
              <label className="clients-field">
                <span>{pageText(locale, 'Goal unit', 'יחידת יעד')}</span>
                <select value={flight.goal_kind} onChange={(event) => setFlights(flights.map((entry, position) => (position === index ? { ...entry, goal_kind: event.target.value } : entry)))}>
                  {goalKinds.map((kind) => (
                    <option key={kind} value={kind}>{vocabularyLabel(goalWords, kind, locale)}</option>
                  ))}
                </select>
              </label>
              <Field label={pageText(locale, 'Goal', 'יעד')} value={flight.goal_value} onChange={(value) => setFlights(flights.map((entry, position) => (position === index ? { ...entry, goal_value: value } : entry)))} type="number" ltr />
              <button type="button" className="clients-icon-button" onClick={() => setFlights(flights.filter((entry, position) => position !== index))} aria-label={pageText(locale, 'Remove flight', 'הסירו טיסת שידור')}>
                <Trash2 size={14} aria-hidden="true" />
              </button>
            </div>
          ))}
          <button type="button" className="clients-secondary" onClick={() => setFlights([...flights, { ...EMPTY_FLIGHT }])}>
            <Plus size={13} aria-hidden="true" />
            {pageText(locale, 'Add a flight', 'הוסיפו טיסת שידור')}
          </button>
          <p className="clients-basis-note">
            {pageText(
              locale,
              'A goal is what was booked. No delivery feed exists, so what a flight delivered stays unknown rather than guessed.',
              'יעד הוא מה שהוזמן. אין הזנת אספקה, ולכן מה שסופק נשאר לא ידוע ולא מנוחש.',
            )}
          </p>
        </fieldset>

        <RefusalNotice error={state.error} opens={state.opens} locale={locale} onOpen={onOpenRefused} />
        <div className="clients-form-actions">
          <button type="submit" className="clients-primary" disabled={state.status === 'saving'}>
            {state.status === 'saving'
              ? pageText(locale, 'Saving', 'שומר')
              : pageText(locale, 'Create and link all three', 'צרו וקשרו את שלושתם')}
          </button>
          <button type="button" className="clients-secondary" onClick={onClose}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </button>
        </div>
      </form>
    </aside>
  );
}
