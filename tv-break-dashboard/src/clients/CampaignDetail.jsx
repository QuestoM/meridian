import React, { useState } from 'react';
import { Button } from '../studio/actions';
import { Pencil } from 'lucide-react';
import { pageText } from '../shell/format';
import { localized } from './clients-money-helpers';
import CampaignFlights from './CampaignFlights';
import CampaignTerms from './CampaignTerms';
import { DeliveryProgress } from './DeliveryState';
import DemoBadge from './DemoBadge';

// What sits under one campaign row: the terms that were agreed, how far the
// counted delivery has got against them, and the flights that carry the booked
// goals. The terms and the flights are amendable here, which is the whole point
// of the row opening at all.
//
// An empty term is a control rather than a blank, the same device the client
// record uses, so a campaign with no rebate on file offers to set one instead of
// showing a dash that leads nowhere.
//
// The progress against the rating goal and the budget is this campaign's own
// ledger and never the board's. Each percent carries the endpoint's own state
// word, so a figure counted over part of the flight reads as a floor, and a goal
// this product cannot measure in its own currency reads as unmeasurable rather
// than as a zero.

function weekdayNames(scope, weekdays, locale) {
  const tokens = String(scope || '').split(',').filter(Boolean);
  if (!tokens.length) {
    return '';
  }
  const ordered = (weekdays || []).filter((day) => tokens.includes(day.key));
  if (!ordered.length) {
    return tokens.join(', ');
  }
  return ordered.map((day) => (locale === 'he' ? day.he : day.en)).join(', ');
}

function Term({ label, value, locale, canEdit, onEdit }) {
  return (
    <div className={value ? 'clients-property' : 'clients-property empty'}>
      <dt>{label}</dt>
      <dd>
        {value ? <span>{value}</span> : null}
        {!value && canEdit ? (
          <Button type="button" className="clients-inline-action" onClick={onEdit}>
            {pageText(locale, 'Set it', 'הגדירו')}
          </Button>
        ) : null}
        {!value && !canEdit ? (
          <span className="clients-unset">{pageText(locale, 'Not agreed', 'לא סוכם')}</span>
        ) : null}
      </dd>
    </div>
  );
}

export default function CampaignDetail({
  campaign,
  board,
  options,
  optionsError,
  locale,
  canEdit = false,
  notify,
  onChanged,
}) {
  const [editing, setEditing] = useState(false);
  const weekdays = (options && options.weekdays) || [];
  const days = weekdayNames(campaign.surcharge_weekdays, weekdays, locale);

  if (editing) {
    if (!options) {
      return (
        <p className="clients-reason">
          {optionsError || pageText(locale, 'Loading the choices this form offers', 'טוען את האפשרויות שהטופס מציע')}
        </p>
      );
    }
    return (
      <CampaignTerms
        mode="edit"
        campaign={campaign}
        options={options}
        terms={board.terms}
        locale={locale}
        onCancel={() => setEditing(false)}
        onSaved={() => {
          setEditing(false);
          notify('Campaign saved.', 'הקמפיין נשמר.');
          onChanged();
        }}
      />
    );
  }

  return (
    <>
      {campaign.is_demo ? (
        <p className="clients-basis-note">
          <DemoBadge demo={campaign.demo} locale={locale} />
          {' '}
          {localized(campaign.demo, 'meaning', locale)}
        </p>
      ) : null}
      <div className="clients-detail-terms">
        <dl className="clients-properties">
          <Term
            label={pageText(locale, 'Rebate', 'רבייט')}
            value={campaign.rebate_percent === null ? '' : `${campaign.rebate_percent}%`}
            locale={locale}
            canEdit={canEdit}
            onEdit={() => setEditing(true)}
          />
          <Term
            label={pageText(locale, 'Discount off the weekday surcharge', 'הנחה מתוספת יום בשבוע')}
            value={campaign.surcharge_discount_percent === null
              ? ''
              : `${campaign.surcharge_discount_percent}%${days ? `, ${days}` : ''}`}
            locale={locale}
            canEdit={canEdit}
            onEdit={() => setEditing(true)}
          />
          <Term
            label={pageText(locale, 'Notes', 'הערות')}
            value={campaign.notes}
            locale={locale}
            canEdit={canEdit}
            onEdit={() => setEditing(true)}
          />
          <Term
            label={pageText(locale, 'Booked on', 'נרשם ב')}
            value={campaign.created_at ? `${campaign.created_at.slice(0, 10)} ${campaign.created_by}`.trim() : ''}
            locale={locale}
            canEdit={false}
            onEdit={() => {}}
          />
        </dl>
        {canEdit ? (
          <Button type="button" className="clients-secondary" onClick={() => setEditing(true)}>
            <Pencil size={13} aria-hidden="true" />
            {pageText(locale, 'Edit the window and terms', 'ערכו את החלון והתנאים')}
          </Button>
        ) : null}
      </div>
      <p className="clients-basis-note">{localized(board.terms, 'reason', locale)}</p>

      <DeliveryProgress delivery={campaign.delivery} locale={locale} />

      <CampaignFlights
        campaign={campaign}
        locale={locale}
        goalKinds={board.goal_kinds || []}
        goalWords={board.goal_kind_vocabulary || []}
        delivery={campaign.delivery}
        airStates={(board.delivery && board.delivery.air_state_vocabulary) || []}
        canEdit={canEdit}
        notify={notify}
        onChanged={onChanged}
      />
    </>
  );
}
