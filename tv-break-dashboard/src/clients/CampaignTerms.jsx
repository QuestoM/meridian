import React, { useState } from 'react';
import { pageText } from '../shell/format';
import { changedFields, localized, refusalText } from './clients-money-helpers';
import { createCampaign, updateCampaign } from './clients-api';

// The campaign's own two editable halves: the window it runs in and the terms
// that were agreed for it. One form serves booking a second campaign for a
// client already on file and amending one that is already booked, because the
// fields are the same fields and a second form would drift from this one.
//
// An amend sends only what changed. A field nobody touched is not in the
// request, so it cannot be disturbed and the endpoint's duplicate refusal stays
// off a name that was not edited.

const PERCENTS = ['rebate_percent', 'surcharge_discount_percent'];

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

function draftOf(campaign) {
  return {
    name: (campaign && campaign.name) || '',
    agency_id: (campaign && campaign.agency_id) || '',
    starts_on: (campaign && campaign.starts_on) || '',
    ends_on: (campaign && campaign.ends_on) || '',
    rebate_percent: campaign && campaign.rebate_percent !== null && campaign.rebate_percent !== undefined
      ? String(campaign.rebate_percent)
      : '',
    surcharge_discount_percent:
      campaign && campaign.surcharge_discount_percent !== null && campaign.surcharge_discount_percent !== undefined
        ? String(campaign.surcharge_discount_percent)
        : '',
    surcharge_weekdays: (campaign && campaign.surcharge_weekdays) || '',
    notes: (campaign && campaign.notes) || '',
  };
}

// The store keeps a weekday scope as sorted ISO tokens, so the draft is held in
// exactly that form and a re-selection of the same days is not a false change.
function toggleWeekday(scope, key) {
  const held = String(scope || '').split(',').filter(Boolean);
  const next = held.includes(key) ? held.filter((entry) => entry !== key) : [...held, key];
  return next.sort().join(',');
}

// This screen never writes an agency condition, only the campaign's own term,
// so an empty scope here always means the discount percent would be stored
// with no day it covers, never the onboarding flow's ANY-widening. The line
// says that consequence plainly, the same rule the endpoint itself enforces,
// so a chip change and a refused submit never disagree about what happens.
//
// The percent is half of that rule and the line has to carry it. check_weekday_scope
// returns without refusing anything when the percent is zero or blank, so an
// operator amending only the notes of a campaign that has no discount was being
// promised a refusal that never came. No percent means no scope to state.
function weekdayCoverage(selected, options, locale, percent) {
  const amount = Number(percent);
  const discounting = Number.isFinite(amount) && amount !== 0;
  if (!selected.length) {
    if (!discounting) {
      return pageText(
        locale,
        'No weekday is selected. Nothing is refused, because there is no discount percent to give a day to.',
        'לא נבחר יום בשבוע. דבר אינו נדחה, כיוון שאין אחוז הנחה שצריך לתת לו יום.',
      );
    }
    return pageText(
      locale,
      'No weekday is selected. Submitting like this is refused: the discount percent would have no day it covers.',
      'לא נבחר יום בשבוע. שליחה כך תסורב: אחוז ההנחה יהיה ללא יום שהוא חל עליו.',
    );
  }
  const names = options.filter((day) => selected.includes(day.key)).map((day) => (locale === 'he' ? day.he : day.en));
  return pageText(locale, `Covers ${names.join(', ')}.`, `חל על ${names.join(', ')}.`);
}

export default function CampaignTerms({
  mode,
  campaign,
  options,
  terms,
  locale,
  onCancel,
  onSaved,
}) {
  const creating = mode === 'create';
  const [advertiser, setAdvertiser] = useState((campaign && campaign.advertiser) || '');
  const [draft, setDraft] = useState(() => draftOf(creating ? null : campaign));
  const [state, setState] = useState({ status: 'idle', error: '' });
  const weekdays = (options && options.weekdays) || [];
  const agencies = (options && options.agencies) || [];

  function set(key, value) {
    setDraft((current) => ({ ...current, [key]: value }));
  }

  function clearedPercent() {
    if (creating) {
      return '';
    }
    return PERCENTS.find((key) => draftOf(campaign)[key] !== '' && String(draft[key]).trim() === '') || '';
  }

  // A percent crosses the wire as a number and a blank one as null, which the
  // endpoint reads as "this term was not agreed" rather than as zero percent.
  function payloadFrom(fields) {
    const payload = {};
    Object.entries(fields).forEach(([key, value]) => {
      if (PERCENTS.includes(key)) {
        payload[key] = String(value).trim() === '' ? null : Number(value);
      } else {
        payload[key] = value;
      }
    });
    return payload;
  }

  async function submit(event) {
    event.preventDefault();
    const cleared = clearedPercent();
    if (cleared) {
      setState({
        status: 'idle',
        error: pageText(
          locale,
          'A percent already on file cannot be emptied here. Enter 0 to record that no discount was agreed.',
          'אחוז שכבר רשום אינו ניתן לריקון כאן. הזינו 0 כדי לרשום שלא סוכמה הנחה.',
        ),
      });
      return;
    }
    setState({ status: 'saving', error: '' });
    try {
      if (creating) {
        const record = await createCampaign({ ...payloadFrom(draft), advertiser });
        onSaved(record, 'created');
        return;
      }
      const changes = changedFields(campaign, draft);
      if (!Object.keys(changes).length) {
        setState({
          status: 'idle',
          error: pageText(locale, 'Nothing on this campaign changed.', 'דבר בקמפיין הזה לא השתנה.'),
        });
        return;
      }
      const record = await updateCampaign(campaign.campaign_id, payloadFrom(changes));
      onSaved(record, 'updated');
    } catch (error) {
      setState({ status: 'idle', error: refusalText(error, locale) });
    }
  }

  return (
    <form className="clients-form clients-inline-form" onSubmit={submit}>
      <fieldset>
        <legend>
          {creating
            ? pageText(locale, 'Book a campaign for a client on file', 'הזמינו קמפיין ללקוח שכבר רשום')
            : pageText(locale, 'Window and terms', 'חלון ותנאים')}
        </legend>
        {creating ? (
          <>
            <Field
              label={pageText(locale, 'Client', 'לקוח')}
              value={advertiser}
              onChange={setAdvertiser}
              list="clients-campaign-advertisers"
              required
              hint={pageText(locale, 'a client already on file', 'לקוח שכבר רשום')}
            />
            <datalist id="clients-campaign-advertisers">
              {((options && options.advertisers) || []).map((entry) => (
                <option key={entry.advertiser} value={entry.advertiser} />
              ))}
            </datalist>
          </>
        ) : (
          <p className="clients-basis-note">
            {pageText(
              locale,
              `This campaign belongs to ${campaign.advertiser} and stays there. End it and book another to move the work to a different client.`,
              `הקמפיין הזה שייך ל${campaign.advertiser} ונשאר שם. סיימו אותו והזמינו אחר כדי להעביר את העבודה ללקוח אחר.`,
            )}
          </p>
        )}
        <div className="clients-field-grid">
          <Field
            label={pageText(locale, 'Campaign name', 'שם הקמפיין')}
            value={draft.name}
            onChange={(value) => set('name', value)}
            required
          />
          <label className="clients-field">
            <span>{pageText(locale, 'Agency', 'סוכנות')}</span>
            <select value={draft.agency_id} onChange={(event) => set('agency_id', event.target.value)}>
              <option value="">{pageText(locale, 'No agency', 'ללא סוכנות')}</option>
              {agencies.map((entry) => (
                <option key={entry.agency_id} value={entry.agency_id}>
                  {`${entry.name} (${entry.agency_id})`}
                </option>
              ))}
            </select>
          </label>
          <Field
            label={pageText(locale, 'Starts on', 'מתחיל ב')}
            value={draft.starts_on}
            onChange={(value) => set('starts_on', value)}
            type="date"
            ltr
            required
          />
          <Field
            label={pageText(locale, 'Ends on', 'מסתיים ב')}
            value={draft.ends_on}
            onChange={(value) => set('ends_on', value)}
            type="date"
            ltr
            required
          />
          <Field
            label={pageText(locale, 'Rebate percent', 'אחוז רבייט')}
            value={draft.rebate_percent}
            onChange={(value) => set('rebate_percent', value)}
            type="number"
            ltr
          />
          <Field
            label={pageText(locale, 'Discount off the weekday surcharge', 'הנחה מתוספת יום בשבוע')}
            value={draft.surcharge_discount_percent}
            onChange={(value) => set('surcharge_discount_percent', value)}
            type="number"
            ltr
          />
          <Field
            label={pageText(locale, 'Notes', 'הערות')}
            value={draft.notes}
            onChange={(value) => set('notes', value)}
          />
        </div>
        <div className="clients-weekdays">
          {weekdays.map((day) => (
            <button
              key={day.key}
              type="button"
              className={String(draft.surcharge_weekdays || '').split(',').includes(day.key) ? 'active' : ''}
              onClick={() => set('surcharge_weekdays', toggleWeekday(draft.surcharge_weekdays, day.key))}
            >
              {locale === 'he' ? day.he : day.en}
            </button>
          ))}
        </div>
        <p className="clients-basis-note" role="status">
          {weekdayCoverage(
            String(draft.surcharge_weekdays || '').split(',').filter(Boolean),
            weekdays,
            locale,
            draft.surcharge_discount_percent,
          )}
        </p>
        <p className="clients-basis-note">{localized(terms, 'reason', locale)}</p>
      </fieldset>

      {state.error ? <p className="clients-error" role="alert">{state.error}</p> : null}
      <div className="clients-form-actions">
        <button type="submit" className="clients-primary" disabled={state.status === 'saving'}>
          {state.status === 'saving'
            ? pageText(locale, 'Saving', 'שומר')
            : creating
              ? pageText(locale, 'Book the campaign', 'הזמינו את הקמפיין')
              : pageText(locale, 'Save the window and terms', 'שמרו את החלון והתנאים')}
        </button>
        <button type="button" className="clients-secondary" onClick={onCancel}>
          {pageText(locale, 'Cancel', 'ביטול')}
        </button>
      </div>
    </form>
  );
}
