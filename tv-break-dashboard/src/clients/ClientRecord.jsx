import React from 'react';
import { ChevronLeft, ChevronRight, Plus, X } from 'lucide-react';
import { pageText } from '../shell/format';
import ClientRuleCard from './ClientRuleCard';
import { exactMoney, goToView, goalLabel, hasLedgerRow, localized, positionOf, sourceLabel, step, vocabularyLabel, windowLabel } from './clients-money-helpers';

// One client, opened without losing the set it came from. The counter and the
// two arrows are Linear's device: a record page that knows it is the nth of a
// filtered set and can walk that set from the inside.
//
// Every figure on the record opens the rows behind it, and every empty property
// is a control rather than a blank, which is the other half of the same idea:
// nothing on the record is a dead end and nothing on it is decoration. A control
// that cannot be completed here is not offered as a control at all: it states
// what is missing and where it comes from, in the place the control would stand.
//
// The client's identity and its price live in the rule card below the money,
// because both of them are edited here rather than on another screen. What that
// section can and cannot write, and why, is stated in the section itself.
//
// The money figures are the one place where that has a limit worth stating. A
// client with no priced spot has no rows to open, so its figures are dashes and
// the control is not a control: it names the day being read and the path that
// would price this client, and the reader stays on the record that said so.

function Property({ label, value, action, onAction, locale }) {
  if (value) {
    return (
      <div className="clients-property">
        <dt>{label}</dt>
        <dd>{value}</dd>
      </div>
    );
  }
  return (
    <div className="clients-property empty">
      <dt>{label}</dt>
      <dd>
        {onAction ? (
          <button type="button" className="clients-inline-action" onClick={onAction}>
            <Plus size={12} aria-hidden="true" />
            {action || pageText(locale, 'Set', 'הגדירו')}
          </button>
        ) : (
          <span className="clients-reason">{action || pageText(locale, 'Nothing yet', 'אין עדיין')}</span>
        )}
      </dd>
    </div>
  );
}

// The three figures, rendered the same whether or not they open anything, so a
// client without money is missing its rows and not its record.
function Figures({ client, locale }) {
  return (
    <>
      <span>
        <small>{pageText(locale, 'Gross', 'ברוטו')}</small>
        <strong className="numeric" dir="ltr">{client.gross === null ? '-' : exactMoney(client.gross, locale)}</strong>
      </span>
      <span>
        <small>{pageText(locale, 'Net after rebates', 'נטו אחרי רבייט')}</small>
        <strong className="numeric" dir="ltr">{client.net === null ? '-' : exactMoney(client.net, locale)}</strong>
      </span>
      <span>
        <small>{pageText(locale, 'Spots', 'תשדירים')}</small>
        <strong className="numeric" dir="ltr">{client.spots === null ? '-' : client.spots}</strong>
      </span>
    </>
  );
}

function Flights({ campaign, locale, goalWords }) {
  if (!campaign.flights.length) {
    return (
      <p className="clients-reason">
        {pageText(locale, 'No flight booked on this campaign yet.', 'לא הוזמנה טיסת שידור בקמפיין הזה עדיין.')}
      </p>
    );
  }
  return (
    <ul className="clients-flights">
      {campaign.flights.map((flight) => (
        <li key={flight.flight_id}>
          <span className="clients-flight-id">{flight.flight_id}</span>
          <span className="numeric" dir="ltr">{windowLabel(flight.starts_on, flight.ends_on, locale)}</span>
          <span className="clients-goal">
            <small>{pageText(locale, 'booked', 'הוזמן')}</small>
            <span className="numeric" dir="ltr">{goalLabel(flight, locale, goalWords)}</span>
          </span>
          <span className="clients-unknown">{pageText(locale, 'delivered: unknown', 'סופק: לא ידוע')}</span>
        </li>
      ))}
    </ul>
  );
}

export default function ClientRecord({
  client,
  rows,
  locale,
  basis = null,
  delivery,
  statuses = [],
  goalWords = [],
  canEdit = true,
  ruleRows = null,
  ruleBusy = false,
  editRefusal = '',
  ledgerCampaigns = [],
  onClose,
  onStep,
  onOpenMoney,
  onCreateRule,
  onAddSpelling,
  onOpenRuleCard,
  onBookCampaign,
  onOpenAgency,
  onOpenCampaignMoney,
}) {
  const he = locale === 'he';
  const found = positionOf(rows, client.advertiser);
  const opensRows = hasLedgerRow(client);
  // The agency record is one tab away with all of its terms on it, so the line
  // that names the agency is the way to it. It stays a plain line when there is
  // no id to open, because a control that opens nothing is worse than a label.
  const agencyOpens = Boolean(onOpenAgency && client.agency_id);
  const agencyLine = pageText(locale, `Buys through ${client.agency_name}`, `קונה דרך ${client.agency_name}`);

  return (
    <aside className="clients-record" dir={he ? 'rtl' : 'ltr'} role="dialog" aria-label={client.shown_name || client.advertiser}>
      <header className="clients-record-head">
        <div>
          <h3>{client.shown_name || client.advertiser}</h3>
          <p className="clients-record-sub">
            {client.agency_name && agencyOpens ? (
              <button type="button" className="clients-link" onClick={() => onOpenAgency(client.agency_id)}>
                {agencyLine}
              </button>
            ) : null}
            {client.agency_name && !agencyOpens ? <span>{agencyLine}</span> : null}
            {client.agency_name ? null : (
              <span className="clients-flag">{pageText(locale, 'no agency link', 'ללא שיוך לסוכנות')}</span>
            )}
            <span className="clients-source">{sourceLabel(client.source, locale)}</span>
          </p>
        </div>
        <div className="clients-record-actions">
          {found ? (
            <span className="clients-position">
              <button type="button" onClick={() => onStep(step(rows, client.advertiser, -1))} aria-label={pageText(locale, 'Previous client', 'הלקוח הקודם')}>
                <ChevronRight size={14} aria-hidden="true" />
              </button>
              <span className="numeric" dir="ltr">{`${found.position} / ${found.total}`}</span>
              <button type="button" onClick={() => onStep(step(rows, client.advertiser, 1))} aria-label={pageText(locale, 'Next client', 'הלקוח הבא')}>
                <ChevronLeft size={14} aria-hidden="true" />
              </button>
            </span>
          ) : null}
          <button type="button" className="clients-icon-button" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
            <X size={15} aria-hidden="true" />
          </button>
        </div>
      </header>

      <section className="clients-record-money">
        {opensRows ? (
          <button type="button" className="clients-money-open" onClick={() => onOpenMoney(client.advertiser)}>
            <Figures client={client} locale={locale} />
            <em>{pageText(locale, 'Open every row behind these figures', 'פתחו כל שורה שמאחורי הסכומים')}</em>
          </button>
        ) : (
          <div className="clients-money-open empty">
            <Figures client={client} locale={locale} />
            <em>{pageText(locale, 'No priced spot for this client, so there is no row to open', 'אין תשדיר מתומחר ללקוח הזה, ולכן אין שורה לפתוח')}</em>
          </div>
        )}
        {localized(client, 'money_reason', locale) ? <p className="clients-reason">{localized(client, 'money_reason', locale)}</p> : null}
        {!opensRows && basis && basis.day ? (
          <p className="clients-basis-path">
            {pageText(
              locale,
              `The day being read is ${basis.day}, from ${basis.file}. A daily file carrying this client prices their spots and fills these figures.`,
              `היום הנקרא הוא ⁦${basis.day}⁩, מתוך ⁦${basis.file}⁩. קובץ יומי שנושא את הלקוח הזה מתמחר את התשדירים שלו וממלא את הסכומים.`,
            )}
            <button type="button" className="clients-inline-action" onClick={() => goToView('Data')}>
              {pageText(locale, 'Open Data', 'פתחו את מסך הנתונים')}
            </button>
          </p>
        ) : null}
        {client.dropped_by_frequency ? (
          <p className="clients-reason">
            {pageText(
              locale,
              `${client.dropped_by_frequency} of this client's spots were removed by a rule, so their money is not above.`,
              `⁦${client.dropped_by_frequency}⁩ מתשדירי הלקוח הוסרו על ידי כלל, ולכן הכסף שלהם אינו למעלה.`,
            )}
          </p>
        ) : null}
      </section>

      <ClientRuleCard
        client={client}
        rows={ruleRows}
        locale={locale}
        canEdit={canEdit}
        busy={ruleBusy}
        refusal={editRefusal}
        onCreateRule={onCreateRule}
        onAddSpelling={onAddSpelling}
        onOpenRuleCard={onOpenRuleCard}
      />

      <dl className="clients-properties">
        <Property
          label={pageText(locale, 'Campaigns booked', 'קמפיינים שהוזמנו')}
          value={client.campaign_count ? String(client.campaign_count) : ''}
          action={canEdit
            ? pageText(locale, 'Book the first campaign', 'הזמינו קמפיין ראשון')
            : pageText(locale, 'Nothing is booked yet', 'לא הוזמן דבר עדיין')}
          onAction={canEdit ? onBookCampaign : null}
          locale={locale}
        />
      </dl>

      <section className="clients-record-campaigns">
        <h4>{pageText(locale, 'Booked campaigns', 'קמפיינים שהוזמנו')}</h4>
        {client.campaigns.length ? (
          client.campaigns.map((campaign) => (
            <article key={campaign.campaign_id} className="clients-campaign">
              <header>
                <strong>{campaign.name}</strong>
                <span className="clients-campaign-id">{campaign.campaign_id}</span>
                <span className="numeric" dir="ltr">{windowLabel(campaign.starts_on, campaign.ends_on, locale)}</span>
                <span className={`clients-state ${campaign.status}`}>{vocabularyLabel(statuses, campaign.status, locale)}</span>
              </header>
              <Flights campaign={campaign} locale={locale} goalWords={goalWords} />
            </article>
          ))
        ) : (
          <p className="clients-reason">
            {pageText(locale, 'Nothing is booked for this client yet.', 'לא הוזמן דבר עבור הלקוח הזה עדיין.')}
          </p>
        )}
        {delivery && !delivery.available ? (
          <p className="clients-basis-note">{localized(delivery, 'reason', locale)}</p>
        ) : null}
        {delivery && !delivery.available ? (
          <p className="clients-basis-path">{localized(delivery, 'path_forward', locale)}</p>
        ) : null}
      </section>

      {client.observed_campaigns && client.observed_campaigns.length ? (
        <section className="clients-record-observed">
          <h4>{pageText(locale, 'Campaigns seen on air', 'קמפיינים שנצפו בשידור')}</h4>
          <p className="clients-basis-note">
            {pageText(
              locale,
              'These are the campaign names the daily file carries for this client. They are what aired, not what was booked here.',
              'אלה שמות הקמפיינים שהקובץ היומי נושא עבור הלקוח. זה מה ששודר, לא מה שהוזמן כאן.',
            )}
          </p>
          <ul className="clients-observed-list">
            {client.observed_campaigns.map((name) => (
              <li key={name}>
                {onOpenCampaignMoney && ledgerCampaigns.includes(String(name)) ? (
                  <button type="button" className="clients-link" onClick={() => onOpenCampaignMoney(String(name))}>
                    {name}
                  </button>
                ) : name}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </aside>
  );
}
