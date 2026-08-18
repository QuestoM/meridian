import React from 'react';
import { Button } from '../studio/actions';
import { Figure } from '../shell/bidi';
import { ChevronLeft, ChevronRight, Plus, X } from 'lucide-react';
import { pageText } from '../shell/format';
import ClientRuleCard from './ClientRuleCard';
import { exactMoney, goToView, goalLabel, hasLedgerRow, localized, positionOf, sourceLabel, step, vocabularyLabel, windowLabel } from './clients-money-helpers';
import { DeliveryCell } from './DeliveryState';
import { DeliveryBasis, DeliveryLedgerNote } from './DeliveryBasisNotes';
import DemoBadge from './DemoBadge';
import { isolate } from '../shell/bidi';
import { formatDay } from '../shell/dates';

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
          <Button type="button" className="clients-inline-action" onClick={onAction}>
            <Plus size={12} aria-hidden="true" />
            {action || pageText(locale, 'Set', 'הגדירו')}
          </Button>
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
        <strong className="numeric"><Figure>{client.gross === null ? '-' : exactMoney(client.gross, locale)}</Figure></strong>
      </span>
      <span>
        <small>{pageText(locale, 'Net after rebates', 'נטו אחרי רבייט')}</small>
        <strong className="numeric"><Figure>{client.net === null ? '-' : exactMoney(client.net, locale)}</Figure></strong>
      </span>
      <span>
        <small>{pageText(locale, 'Spots', 'תשדירים')}</small>
        <strong className="numeric"><Figure>{client.spots === null ? '-' : client.spots}</Figure></strong>
      </span>
    </>
  );
}

// The flights of one booked campaign, each with what it committed to and what
// the delivery ledger has counted against it. The ledger is the campaign board's
// own, handed down by the workspace, so this record and that board can never
// print two different answers to the same question.
function Flights({ campaign, delivery, airStates, locale, goalWords }) {
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
          <DemoBadge demo={flight.demo} locale={locale} />
          <Figure className="numeric">{windowLabel(flight.starts_on, flight.ends_on, locale)}</Figure>
          <span className="clients-goal">
            <small>{pageText(locale, 'booked', 'הוזמן')}</small>
            <Figure className="numeric">{goalLabel(flight, locale, goalWords)}</Figure>
          </span>
          <span className="clients-goal">
            <small>{pageText(locale, 'delivered', 'סופק')}</small>
            {/* Answered in the unit on the line above it. "Booked 100 GRP"
                beside "delivered 3 spots" is two units and one invited
                comparison that cannot be made. */}
            <DeliveryCell
              delivery={delivery}
              window={{ starts_on: flight.starts_on, ends_on: flight.ends_on }}
              vocabulary={airStates}
              goal={{ kind: flight.goal_kind, unit: vocabularyLabel(goalWords, flight.goal_kind, locale) }}
              locale={locale}
            />
          </span>
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
  // The payload-level delivery block: one read's instant, its floor rule and
  // its vocabulary. Absent, every row states its own basis inline instead.
  ledger = null,
  deliveryByCampaign = {},
  airStates = [],
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
  const demoCampaignCount = (client.campaigns || []).filter((campaign) => campaign.is_demo).length;
  // A counted rating point may not reach a reader without the ledger's own
  // sentence about what its rating column is, so the caveat rides on whether a
  // flight below is actually booked in rating points.
  const showsRating = (client.campaigns || []).some(
    (campaign) => (campaign.flights || []).some((flight) => flight.goal_kind === 'grp'),
  );
  // The agency record is one tab away with all of its terms on it, so the line
  // that names the agency is the way to it. It stays a plain line when there is
  // no id to open, because a control that opens nothing is worse than a label.
  const agencyOpens = Boolean(onOpenAgency && client.agency_id);
  const agencyLine = pageText(locale, `Buys through ${client.agency_name}`, `קונה דרך ${client.agency_name}`);

  return (
    <aside className="card card-dense card-body clients-record" role="complementary" aria-label={client.shown_name || client.advertiser}>
      <header className="clients-record-head">
        <div>
          <h3>{client.shown_name || client.advertiser}</h3>
          <p className="clients-record-sub">
            {client.agency_name && agencyOpens ? (
              <Button type="button" className="clients-link" onClick={() => onOpenAgency(client.agency_id)}>
                {agencyLine}
              </Button>
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
              <Button type="button" onClick={() => onStep(step(rows, client.advertiser, -1))} aria-label={pageText(locale, 'Previous client', 'הלקוח הקודם')}>
                <ChevronRight size={14} aria-hidden="true" />
              </Button>
              {/* "20 / 42" is one run and must not break: wrapped across two
                  lines in a narrow drawer header it reads as two numbers
                  stacked, and in RTL the halves swap besides. */}
              <Figure className="numeric figure-nowrap">{`${found.position} / ${found.total}`}</Figure>
              <Button type="button" onClick={() => onStep(step(rows, client.advertiser, 1))} aria-label={pageText(locale, 'Next client', 'הלקוח הבא')}>
                <ChevronLeft size={14} aria-hidden="true" />
              </Button>
            </span>
          ) : null}
          <Button type="button" className="clients-icon-button" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
            <X size={15} aria-hidden="true" />
          </Button>
        </div>
      </header>

      <section className="clients-record-money">
        {opensRows ? (
          <Button type="button" className="clients-money-open" onClick={() => onOpenMoney(client.advertiser)}>
            <Figures client={client} locale={locale} />
            <em>{pageText(locale, 'Open every row behind these figures', 'פתחו כל שורה שמאחורי הסכומים')}</em>
          </Button>
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
              `The day being read is ${formatDay(basis.day)}, from ${basis.file}. A daily file carrying this client prices their spots and fills these figures.`,
              `היום הנקרא הוא ${isolate(basis.day)}, מתוך ${isolate(basis.file)}. קובץ יומי שנושא את הלקוח הזה מתמחר את התשדירים שלו וממלא את הסכומים.`,
            )}
            <Button type="button" className="clients-inline-action" onClick={() => goToView('Data')}>
              {pageText(locale, 'Open Data', 'פתחו את מסך הנתונים')}
            </Button>
          </p>
        ) : null}
        {client.dropped_by_frequency ? (
          <p className="clients-reason">
            {pageText(
              locale,
              `${client.dropped_by_frequency} of this client's spots were removed by a rule, so their money is not above.`,
              `${isolate(client.dropped_by_frequency)} מתשדירי הלקוח הוסרו על ידי כלל, ולכן הכסף שלהם אינו למעלה.`,
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

      {/* The count and the list say the same thing, so they are one thing. This
          was a property row reading "קמפיינים שהוזמנו: 2" immediately above a
          section heading reading "קמפיינים שהוזמנו" - two labels, identical
          words, one screen. The heading carries the count now, and the action
          lives with the empty state, which is where somebody who has nothing
          booked is actually looking. */}
      <section className="clients-record-campaigns">
        <h4>
          {pageText(locale, 'Booked campaigns', 'קמפיינים שהוזמנו')}
          {client.campaign_count ? (
            <span className="clients-record-count">
              <Figure className="numeric">{client.campaign_count}</Figure>
              {/* Parenthesised, the way the workspace header states the same
                  fact about the same kind of count. Two adjacent numbers with
                  nothing between them - the total, then how many are demo -
                  read as one number: on this client both are 2 and the heading
                  said "2 2 of them demo data". */}
              {demoCampaignCount > 0 ? (
                <span className="clients-record-demo">
                  {pageText(
                    locale,
                    `(${demoCampaignCount} of them demo data)`,
                    `(${isolate(demoCampaignCount)} מהם נתוני הדגמה)`,
                  )}
                </span>
              ) : null}
            </span>
          ) : null}
        </h4>
        {/* The ledger's own sentences, once for this drawer. Every campaign
            below was counted on the same read, at the same instant, out of the
            same tri-state, so stating that under each of them was the same two
            paragraphs repeated per row. What stays on the row is what differs
            per row: its unsourced days, its file and the rule that capped it. */}
        {client.campaigns.length ? (
          <DeliveryLedgerNote ledger={ledger} locale={locale} ratingBasis={showsRating} />
        ) : null}
        {client.campaigns.length ? (
          client.campaigns.map((campaign) => (
            <article key={campaign.campaign_id} className="clients-campaign">
              <header>
                <strong>{campaign.name}</strong>
                <DemoBadge demo={campaign.demo} locale={locale} />
                <span className="clients-campaign-id">{campaign.campaign_id}</span>
                <Figure className="numeric">{windowLabel(campaign.starts_on, campaign.ends_on, locale)}</Figure>
                <span className={`clients-state ${campaign.status}`}>{vocabularyLabel(statuses, campaign.status, locale)}</span>
              </header>
              <Flights
                campaign={campaign}
                delivery={deliveryByCampaign[campaign.campaign_id] || null}
                airStates={airStates}
                locale={locale}
                goalWords={goalWords}
              />
              <DeliveryBasis
                delivery={deliveryByCampaign[campaign.campaign_id] || null}
                locale={locale}
                ledgerNote={!ledger || !ledger.available}
              />
            </article>
          ))
        ) : (
          <p className="clients-reason">
            {pageText(locale, 'Nothing is booked for this client yet.', 'לא הוזמן דבר עבור הלקוח הזה עדיין.')}
            {canEdit ? (
              <Button type="button" className="clients-link" onClick={onBookCampaign}>
                {pageText(locale, 'Book the first campaign', 'הזמינו קמפיין ראשון')}
              </Button>
            ) : null}
          </p>
        )}
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
                  <Button type="button" className="clients-link" onClick={() => onOpenCampaignMoney(String(name))}>
                    {name}
                  </Button>
                ) : name}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </aside>
  );
}
