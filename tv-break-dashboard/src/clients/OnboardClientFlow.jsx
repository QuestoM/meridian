import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { Dialog } from '../studio/modal';
import { Check, Plus, Trash2 } from 'lucide-react';
import { pageText } from '../shell/format';
import { loadOnboardingOptions, onboardClient } from './clients-api';
import { goalLabel, localized, refusalText, vocabularyLabel } from './clients-money-helpers';
import { weekdayCoverage } from './weekday-scope-helpers';
import { isolate } from '../shell/bidi';
import { formatSpan } from '../shell/dates';
import { InputControl, SelectControl } from '../studio/dom-controls';
const EMPTY_FLIGHT = { starts_on: '', ends_on: '', goal_kind: 'spots', goal_value: '' };
const STEPS = [{ en: 'Identity', he: 'זהות' }, { en: 'Commercial terms', he: 'תנאים מסחריים' },
  { en: 'Flights', he: 'טיסות שידור' }, { en: 'Review', he: 'בדיקה' }];
const EMPTY_AGENCY = { name: '', agency_type: '', contact_name: '', contact_role: '', contact_phone: '',
  contact_email: '', vat_id: '', rebate_percent: '', commission_percent: '', credit_limit_ils: '', payment_terms_days: '60' };
// Hebrew counts its nouns: one flight is a SINGULAR noun with a trailing אחת,
// none is a word rather than a zero, and only two or more take the plural. The
// count read "נוצר עם 1 טיסות שידור" on a screen whose whole job is to be
// trusted, which is exactly where a grammar slip is most expensive.
function flightsCreatedLine(count, locale) {
  if (locale !== 'he') {
    if (count === 0) return 'campaign created with no flights yet';
    return `campaign created with ${count} flight${count === 1 ? '' : 's'}`;
  }
  if (count === 0) return 'קמפיין נוצר, בלי טיסות שידור בשלב זה';
  if (count === 1) return 'קמפיין נוצר עם טיסת שידור אחת';
  return `קמפיין נוצר עם ${isolate(count)} טיסות שידור`;
}

function openWord(kind, locale) {
  if (kind === 'campaign') return pageText(locale, 'Open that campaign', 'פתחו את הקמפיין הזה');
  if (kind === 'agency') return pageText(locale, 'Open that agency record', 'פתחו את כרטיס הסוכנות');
  return '';
}
export function RefusalNotice({ error, opens, locale, onOpen }) {
  if (!error) return null;
  const word = opens && onOpen ? openWord(opens.kind, locale) : '';
  return (
    <p className="clients-error" role="alert">
      <span>{error}</span>
      {word ? (
        <Button type="button" className="clients-retry" onClick={() => onOpen(opens)}>
          {word}
        </Button>
      ) : null}
    </p>
  );
}
function Field({ label, value, onChange, type = 'text', ltr = false, list, required = false, hint, min, minLength }) {
  return (
    <label className="clients-field">
      <span>{label}</span>
      <InputControl type={type} value={value} list={list} required={required} dir={ltr ? 'ltr' : 'auto'}
                    min={min || undefined} minLength={minLength || undefined}
                    onChange={(event) => onChange(event.target.value)} />
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}
export default function OnboardClientFlow({ locale, prefill, onClose, onDone, onOpenRefused }) {
  const [step, setStep] = useState(0);
  const [options, setOptions] = useState(null);
  const [agencyMode, setAgencyMode] = useState('existing');
  const [agencyId, setAgencyId] = useState('');
  const [agency, setAgency] = useState(EMPTY_AGENCY);
  const [advertiser, setAdvertiser] = useState((prefill && prefill.advertiser) || '');
  const [campaign, setCampaign] = useState({ name: '', starts_on: '', ends_on: '', rebate_percent: '' });
  const [discount, setDiscount] = useState({ percent: '', weekdays: ['6'], asAgencyRule: false });
  const [flights, setFlights] = useState([{ ...EMPTY_FLIGHT }]);
  const [state, setState] = useState({ status: 'idle', error: '', opens: null, result: null });
  const formRef = useRef(null);
  const stepTitleRef = useRef(null);
  // A choice already made outranks the later options read.
  const chosen = useRef({ mode: false, agency: false });
  useEffect(() => { if (step > 0 && stepTitleRef.current) stepTitleRef.current.focus(); }, [step]);

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

  function chooseAgencyMode(mode) { chosen.current.mode = true; setAgencyMode(mode); }

  function chooseAgency(id) { chosen.current.agency = true; setAgencyId(id); }

  function editAgency(patch) { chosen.current.mode = true; setAgency((current) => ({ ...current, ...patch })); }
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

  function nextStep() {
    if (!formRef.current || formRef.current.reportValidity()) setStep((current) => Math.min(STEPS.length - 1, current + 1));
  }
  const readyFlights = flights.filter((flight) => flight.starts_on && flight.ends_on && flight.goal_value !== '');
  // The agency by NAME wherever a person reads it; the id stays secondary.
  function agencyWord() {
    if (agencyMode !== 'existing') return agency.name;
    const record = ((options && options.agencies) || []).find((entry) => entry.agency_id === agencyId);
    return record ? `${record.name} (${record.agency_id})` : agencyId;
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
      setState({ status: 'idle', error: refusalText(error, locale), opens: error.opens || null, result: null });
    }
  }
  if (state.status === 'done') {
    const result = state.result;
    const outcomeRows = [
      [agencyWord(), result.agency.outcome === 'created'
        ? pageText(locale, 'agency created', 'סוכנות נוצרה') : pageText(locale, 'agency already existed, reused', 'הסוכנות כבר קיימת, נעשה בה שימוש')],
      [result.advertiser.advertiser, result.advertiser.outcome === 'linked'
        ? pageText(locale, 'client linked to the agency', 'הלקוח שויך לסוכנות') : pageText(locale, 'client was already linked', 'הלקוח כבר היה משויך')],
      [pageText(locale, 'Client record', 'כרטיס הלקוח'), (result.advertiser.identity || {}).outcome === 'registered'
        ? pageText(locale, 'named as a client of this operator, so a rule can be written against its name', 'נרשם כלקוח של המפעיל, ולכן אפשר לכתוב כלל על שמו')
        : pageText(locale, 'already a named client, so nothing was written', 'כבר לקוח בשם, ולכן לא נכתב דבר')],
      [result.campaign.campaign_id, flightsCreatedLine(result.flights.length, locale)],
      [pageText(locale, 'Terms', 'תנאים'), result.discount.outcome === 'priced_as_agency_condition'
        ? localized(result.discount, 'covers', locale) : localized(result.discount, 'note', locale)],
    ];
    return (
      <Dialog open onClose={onClose} className="clients-onboard"
              title={pageText(locale, 'The client is on file', 'הלקוח נקלט')}
              closeLabel={pageText(locale, 'Close onboarding result', 'סגירת תוצאת הקליטה')}>
        <ul className="clients-result">
          {outcomeRows.map(([title, detail]) => (
            <li key={title}><Check size={13} aria-hidden="true" /><strong>{title}</strong><span>{detail}</span></li>
          ))}
        </ul>
        <Button type="button" className="clients-primary" onClick={() => onDone(result.campaign.advertiser)}>
          {pageText(locale, 'Open the client record', 'פתחו את כרטיס הלקוח')}
        </Button>
      </Dialog>
    );
  }
  return (
    <Dialog open onClose={onClose} size="wide" className="clients-onboard" dismissOnBackdrop={false}
            title={pageText(locale, 'Onboard a client and book a campaign', 'קליטת לקוח והזמנת קמפיין')}
            description={pageText(locale,
              'One insertion order, one submit. The agency, the client and the campaign are created and linked together.',
              'הזמנה אחת, שליחה אחת. הסוכנות, הלקוח והקמפיין נוצרים ומקושרים יחד.')}
            closeLabel={pageText(locale, 'Cancel onboarding', 'ביטול הקליטה')}>
      <ol className="clients-workflow-steps" id="clients-onboard-steps" aria-label={pageText(locale, 'Onboarding progress', 'התקדמות הקליטה')}>
        {STEPS.map((entry, index) => (
          <li key={entry.en} className={index === step ? 'active' : index < step ? 'complete' : ''}>
            <span className="clients-workflow-step" aria-current={index === step ? 'step' : undefined}>
              <span>{index + 1}</span>
              {pageText(locale, entry.en, entry.he)}
            </span>
          </li>
        ))}
      </ol>
      <form ref={formRef} onSubmit={submit} className="clients-form" aria-busy={!options || state.status === 'saving'}>
        <div className="clients-workflow-panel" id="onboard-step-identity" hidden={step !== 0}>
        <h4 ref={step === 0 ? stepTitleRef : null} tabIndex={-1}>{pageText(locale, 'Identity', 'זהות')}</h4>
        <fieldset disabled={step !== 0}>
          <legend>{pageText(locale, 'Agency', 'סוכנות')}</legend>
          <div className="clients-radio-row">
            <label>
              <InputControl type="radio" checked={agencyMode === 'existing'} onChange={() => chooseAgencyMode('existing')} />
              {pageText(locale, 'An agency we already work with', 'סוכנות שאנחנו כבר עובדים איתה')}
            </label>
            <label>
              <InputControl type="radio" checked={agencyMode === 'new'} onChange={() => chooseAgencyMode('new')} />
              {pageText(locale, 'A new agency', 'סוכנות חדשה')}
            </label>
          </div>
          {agencyMode === 'existing' ? (
            <label className="clients-field">
              <span>{pageText(locale, 'Agency', 'סוכנות')}</span>
              <SelectControl value={agencyId} onChange={(event) => chooseAgency(event.target.value)}>
                {(options ? options.agencies : []).map((entry) => (
                  <option key={entry.agency_id} value={entry.agency_id}>
                    {`${entry.name} (${entry.agency_id}), ${entry.rebate_percent}%`}
                  </option>
                ))}
              </SelectControl>
            </label>
          ) : (
            <>
            <div className="clients-field-grid">
              <Field label={pageText(locale, 'Agency name', 'שם הסוכנות')} value={agency.name} onChange={(value) => editAgency({ name: value })} required hint={pageText(locale, 'exactly as the daily file spells it', 'בדיוק כפי שהקובץ היומי מאיית')} />
              <label className="clients-field">
                <span>{pageText(locale, 'Agency type', 'סוג הסוכנות')}</span>
                <SelectControl value={agency.agency_type} onChange={(event) => editAgency({ agency_type: event.target.value })}>
                  <option value="">{pageText(locale, 'Not stated', 'לא צוין')}</option>
                  {(options ? options.agency_types : []).map((type) => <option key={type} value={type}>{type}</option>)}
                </SelectControl>
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

        <fieldset disabled={step !== 0}>
          <legend>{pageText(locale, 'Client', 'לקוח')}</legend>
          <Field
            label={pageText(locale, 'Client name', 'שם הלקוח')}
            value={advertiser}
            onChange={setAdvertiser}
            list="clients-known-advertisers"
            required
            minLength={2}
            hint={pageText(locale, 'pick one we have seen on air, or type a new one', 'בחרו לקוח שראינו בשידור, או הקלידו חדש')}
          />
          <datalist id="clients-known-advertisers">
            {(options ? options.advertisers : []).map((entry) => (
              <option key={entry.advertiser} value={entry.advertiser} />
            ))}
          </datalist>
        </fieldset>
        </div>

        <div className="clients-workflow-panel" id="onboard-step-commercial" hidden={step !== 1}>
        <h4 ref={step === 1 ? stepTitleRef : null} tabIndex={-1}>{pageText(locale, 'Commercial terms', 'תנאים מסחריים')}</h4>
        <fieldset disabled={step !== 1}>
          <legend>{pageText(locale, 'Campaign', 'קמפיין')}</legend>
          <div className="clients-field-grid">
            <Field label={pageText(locale, 'Campaign name', 'שם הקמפיין')} value={campaign.name} onChange={(value) => setCampaign({ ...campaign, name: value })} required minLength={2} />
            <Field label={pageText(locale, 'Starts on', 'מתחיל ב')} value={campaign.starts_on} onChange={(value) => setCampaign({ ...campaign, starts_on: value })} type="date" ltr required />
            <Field label={pageText(locale, 'Ends on', 'מסתיים ב')} value={campaign.ends_on} onChange={(value) => setCampaign({ ...campaign, ends_on: value })} type="date" ltr required min={campaign.starts_on} />
            <Field label={pageText(locale, 'Campaign rebate percent', 'אחוז רבייט לקמפיין')} value={campaign.rebate_percent} onChange={(value) => setCampaign({ ...campaign, rebate_percent: value })} type="number" ltr />
          </div>
        </fieldset>

        <fieldset disabled={step !== 1}>
          <legend>{pageText(locale, 'Weekday surcharge discount', 'הנחה על תוספת יום בשבוע')}</legend>
          <div className="clients-field-grid">
            <Field label={pageText(locale, 'Discount percent off the surcharge', 'אחוז הנחה מהתוספת')} value={discount.percent} onChange={(value) => setDiscount({ ...discount, percent: value })} type="number" ltr />
          </div>
          <div className="clients-weekdays">
            {weekdays.map((day) => (
              <Button
                key={day.key}
                type="button"
                className={discount.weekdays.includes(day.key) ? 'active' : ''}
                onClick={() => toggleWeekday(day.key)}
              >
                {locale === 'he' ? day.he : day.en}
              </Button>
            ))}
          </div>
          <p className="clients-basis-note" role="status">
            {weekdayCoverage(discount.weekdays, weekdays, locale, {
              asAgencyRule: discount.asAgencyRule,
              percent: discount.percent,
            })}
          </p>
          <label className="clients-checkbox">
            <InputControl type="checkbox" checked={discount.asAgencyRule} onChange={(event) => setDiscount({ ...discount, asAgencyRule: event.target.checked })} />
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
        </div>

        <div className="clients-workflow-panel" id="onboard-step-flights" hidden={step !== 2}>
        <h4 ref={step === 2 ? stepTitleRef : null} tabIndex={-1}>{pageText(locale, 'Flights', 'טיסות שידור')}</h4>
        <fieldset disabled={step !== 2}>
          <legend>{pageText(locale, 'Flights', 'טיסות שידור')}</legend>
          {flights.map((flight, index) => (
            <div className="clients-flight-row" key={`flight-${index}`}>
              <Field label={pageText(locale, 'From', 'מתאריך')} value={flight.starts_on} onChange={(value) => setFlights(flights.map((entry, position) => (position === index ? { ...entry, starts_on: value } : entry)))} type="date" ltr />
              <Field label={pageText(locale, 'To', 'עד תאריך')} value={flight.ends_on} onChange={(value) => setFlights(flights.map((entry, position) => (position === index ? { ...entry, ends_on: value } : entry)))} type="date" ltr min={flight.starts_on} />
              <label className="clients-field">
                <span>{pageText(locale, 'Goal unit', 'יחידת יעד')}</span>
                <SelectControl value={flight.goal_kind} onChange={(event) => setFlights(flights.map((entry, position) => (position === index ? { ...entry, goal_kind: event.target.value } : entry)))}>
                  {goalKinds.map((kind) => (
                    <option key={kind} value={kind}>{vocabularyLabel(goalWords, kind, locale)}</option>
                  ))}
                </SelectControl>
              </label>
              <Field label={pageText(locale, 'Goal', 'יעד')} value={flight.goal_value} onChange={(value) => setFlights(flights.map((entry, position) => (position === index ? { ...entry, goal_value: value } : entry)))} type="number" ltr />
              <Button type="button" className="clients-icon-button" onClick={() => setFlights(flights.filter((entry, position) => position !== index))} aria-label={pageText(locale, 'Remove flight', 'הסירו טיסת שידור')}>
                <Trash2 size={14} aria-hidden="true" />
              </Button>
            </div>
          ))}
          <Button type="button" className="clients-secondary" onClick={() => setFlights([...flights, { ...EMPTY_FLIGHT }])}>
            <Plus size={13} aria-hidden="true" />
            {pageText(locale, 'Add a flight', 'הוסיפו טיסת שידור')}
          </Button>
          <p className="clients-basis-note">
            {pageText(
              locale,
              'A goal is what was booked. No delivery feed exists, so what a flight delivered stays unknown rather than guessed.',
              'יעד הוא מה שהוזמן. אין הזנת אספקה, ולכן מה שסופק נשאר לא ידוע ולא מנוחש.',
            )}
          </p>
        </fieldset>
        </div>

        <div className="clients-workflow-panel clients-workflow-review" id="onboard-step-review" hidden={step !== 3}>
          <h4 ref={step === 3 ? stepTitleRef : null} tabIndex={-1}>{pageText(locale, 'Review before creating', 'בדיקה לפני יצירה')}</h4>
          <dl>
            <div><dt>{pageText(locale, 'Agency', 'סוכנות')}</dt><dd>{agencyWord()}</dd></div>
            <div><dt>{pageText(locale, 'Client', 'לקוח')}</dt><dd>{advertiser}</dd></div>
            <div><dt>{pageText(locale, 'Campaign', 'קמפיין')}</dt><dd>{campaign.name}</dd></div>
            <div><dt>{pageText(locale, 'Window', 'חלון')}</dt><dd>{formatSpan(campaign.starts_on, campaign.ends_on, locale)}</dd></div>
            {campaign.rebate_percent !== '' ? (
              <div><dt>{pageText(locale, 'Campaign rebate', 'רבייט לקמפיין')}</dt><dd>{`${campaign.rebate_percent}%`}</dd></div>
            ) : null}
            {discount.percent !== '' ? (
              <div><dt>{pageText(locale, 'Surcharge discount', 'הנחה מהתוספת')}</dt><dd>{`${discount.percent}%`}</dd></div>
            ) : null}
            {readyFlights.length ? readyFlights.map((flight, index) => (
              <div key={`review-flight-${index}`}>
                <dt>{pageText(locale, `Flight ${index + 1}`, `טיסה ${index + 1}`)}</dt>
                <dd>{`${formatSpan(flight.starts_on, flight.ends_on, locale)} · ${goalLabel(flight, locale, goalWords)}`}</dd>
              </div>
            )) : (
              // Silence here would be the worst answer: a campaign with no
              // flight books nothing, and the review must say so rather than
              // simply omitting the row, which reads as "nothing to report".
              <div>
                <dt>{pageText(locale, 'Flights', 'טיסות שידור')}</dt>
                <dd>{pageText(locale, 'none yet: the campaign is created with nothing booked against it',
                              'אין עדיין: הקמפיין ייווצר בלי שהוזמן דבר מולו')}</dd>
              </div>
            )}
          </dl>
          <p className="clients-basis-note">
            {pageText(locale, 'Nothing is written until you choose Create and link below.', 'דבר אינו נכתב עד לבחירה ב״יצירה וקישור״ למטה.')}
          </p>
        </div>

        <RefusalNotice error={state.error} opens={state.opens} locale={locale} onOpen={onOpenRefused} />
        <div className="clients-form-actions">
          {step < STEPS.length - 1 ? (
            <a href="#clients-onboard-steps" role="button" className="clients-primary" onClick={(event) => { event.preventDefault(); nextStep(); }}
               onKeyDown={(event) => { if (event.key === ' ') { event.preventDefault(); nextStep(); } }}>
              {pageText(locale, 'Continue', 'המשך')}
            </a>
          ) : (
            <Button type="submit" className="clients-primary" disabled={state.status === 'saving'}>
              {state.status === 'saving'
                ? pageText(locale, 'Saving', 'שומר')
                : pageText(locale, 'Create and link all three', 'צרו וקשרו את שלושתם')}
            </Button>
          )}
          {step > 0 ? (
            <a href="#clients-onboard-steps" role="button" className="clients-secondary" onClick={(event) => { event.preventDefault(); setStep((current) => current - 1); }}
               onKeyDown={(event) => { if (event.key === ' ') { event.preventDefault(); setStep((current) => current - 1); } }}>
              {pageText(locale, 'Back', 'חזרה')}
            </a>
          ) : null}
          <Button type="button" className="clients-secondary" onClick={onClose}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
