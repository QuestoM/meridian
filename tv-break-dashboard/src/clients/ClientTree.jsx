import React, { useMemo, useState } from 'react';
import { Button } from '../studio/actions';
import { Figure } from '../shell/bidi';
import { Building2, ChevronDown, ChevronUp, Plus, Search } from 'lucide-react';
import { pageText } from '../shell/format';
import { basisLine, exactMoney, filterAgencies, filterClients, localized } from './clients-money-helpers';
import { isolate } from '../shell/bidi';
import { InputControl } from '../studio/dom-controls';

const AGENCY_WINDOW = 18;

// The tree's own read carries no demo tally (it is a join of agencies, clients
// and the priced ledger, none of which know about the campaign seed), but every
// campaign record inside it carries its own is_demo flag from the campaigns
// store. Counting it here is arithmetic on data already on the payload, not a
// figure invented on this screen.
function demoTally(agencies, unlinked, booked) {
  let demo = 0;
  let total = 0;
  const walk = (client) => {
    (client.campaigns || []).forEach((campaign) => {
      total += 1;
      if (campaign.is_demo) {
        demo += 1;
      }
    });
  };
  (agencies || []).forEach((agency) => (agency.clients || []).forEach(walk));
  (unlinked || []).forEach(walk);
  (booked || []).forEach(walk);
  return { demo, total };
}

// The commercial spine as one containment: agency, then the clients that buy
// through it, then what each of them has booked and delivered. Google Ads keeps
// account, campaign and ad group in one containment and invents no fourth layer
// for convenience; this is the same discipline over the four objects this
// product actually has.
//
// Money on every row is the delivered ledger and nothing else, so an agency row
// is exactly the sum of its client rows. An empty property is a control, not a
// blank: an agency with no rebate offers to set one, a client with no campaign
// offers to book one.

// Hebrew and English both have a singular. Every count on this surface can be
// one, and "1 clients" is a line nobody wrote on purpose, so the word is chosen
// from the same number that is printed beside it.
function plural(count, singular, many) {
  return count === 1 ? singular : many;
}

function clientsWord(count, locale) {
  return plural(
    count,
    pageText(locale, 'client', 'לקוח'),
    pageText(locale, 'clients', 'לקוחות'),
  );
}

function Terms({ agency, locale }) {
  const terms = agency.terms;
  const items = [
    {
      label: pageText(locale, 'Rebate', 'רבייט'),
      value: terms.rebate_percent === null ? null : `${terms.rebate_percent}%`,
      basis: pageText(locale, 'off gross, reporting only', 'מהברוטו, לדיווח בלבד'),
    },
    {
      label: pageText(locale, 'Commission', 'עמלה'),
      value: terms.commission_percent === null ? null : `${terms.commission_percent}%`,
      basis: pageText(locale, 'agency terms, not applied by the engine', 'תנאי סוכנות, לא מיושם על ידי המנוע'),
    },
    {
      label: pageText(locale, 'Payment terms', 'תנאי תשלום'),
      value: terms.payment_terms_days ? `${terms.payment_terms_days}` : null,
      basis: pageText(locale, 'days', 'ימים'),
    },
    {
      label: pageText(locale, 'Credit limit', 'מסגרת אשראי'),
      value: terms.credit_limit_ils ? exactMoney(terms.credit_limit_ils, locale) : null,
      basis: pageText(locale, 'as agreed', 'כמוסכם'),
    },
    {
      label: pageText(locale, 'VAT id', 'ח״פ / עוסק'),
      value: terms.vat_id || null,
      basis: pageText(locale, 'registration', 'רישום'),
    },
  ];
  return (
    <dl className="clients-terms">
      {items.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>
            {item.value ? (
              <>
                <Figure className="numeric">{item.value}</Figure>
                <small>{item.basis}</small>
              </>
            ) : (
              <span className="clients-unset">{pageText(locale, 'Not set', 'לא הוגדר')}</span>
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function ClientRow({ client, locale, onOpen }) {
  const demoCampaigns = (client.campaigns || []).filter((campaign) => campaign.is_demo).length;
  return (
    <tr>
      <td>
        <Button type="button" className="clients-link" onClick={() => onOpen(client.advertiser)}>
          {client.shown_name || client.advertiser}
        </Button>
        {client.resolved ? null : (
          <span className="clients-flag">{pageText(locale, 'not seen on air yet', 'טרם נצפה בשידור')}</span>
        )}
      </td>
      <td className="numeric"><Figure>{client.gross === null ? '-' : exactMoney(client.gross, locale)}</Figure></td>
      <td className="numeric"><Figure>{client.net === null ? '-' : exactMoney(client.net, locale)}</Figure></td>
      <td className="numeric"><Figure>{client.spots === null ? '-' : client.spots}</Figure></td>
      <td className="numeric">
        <Figure>{client.campaign_count}</Figure>
        {demoCampaigns > 0 ? (
          <span className="clients-flag">
            {pageText(
              locale,
              demoCampaigns === client.campaign_count ? 'all demo' : `${demoCampaigns} demo`,
              demoCampaigns === client.campaign_count ? 'כולם הדגמה' : `${isolate(demoCampaigns)} הדגמה`,
            )}
          </span>
        ) : null}
      </td>
      <td>
        {localized(client, 'money_reason', locale) ? <small className="clients-reason">{localized(client, 'money_reason', locale)}</small> : null}
        {client.campaign_count === 0 ? (
          <Button type="button" className="clients-inline-action" onClick={() => onOpen(client.advertiser)}>
            <Plus size={12} aria-hidden="true" />
            {pageText(locale, 'Book a campaign', 'הזמינו קמפיין')}
          </Button>
        ) : null}
      </td>
    </tr>
  );
}

// A group of clients that no agency claims. Two of them exist, for two different
// reasons, and each one states its reason in the header, because a row of
// dashes under an unexplained heading reads as missing data rather than as the
// state it is. Both are the same rows the counter above counts.
function FlatGroup({ title, note, clients, locale, onOpen }) {
  if (!clients.length) {
    return null;
  }
  return (
    <article className="card card-dense clients-agency clients-unlinked">
      <div className="card-body clients-agency-head static">
        <span className="clients-agency-name">
          <strong>{title}</strong>
          <small>{note}</small>
        </span>
        <span className="clients-agency-facts">
          <span className="numeric"><Figure>{clients.length}</Figure></span>
          <small>{clientsWord(clients.length, locale)}</small>
        </span>
      </div>
      <div className="card-body clients-agency-body">
        <table className="clients-table">
          <tbody>
            {clients.map((client) => (
              <ClientRow key={client.advertiser} client={client} locale={locale} onOpen={onOpen} />
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

export default function ClientTree({ tree, locale, onOpenClient }) {
  const [query, setQuery] = useState('');
  const [openAgency, setOpenAgency] = useState('');
  const [visibleCount, setVisibleCount] = useState(AGENCY_WINDOW);
  const he = locale === 'he';

  const agencies = useMemo(
    () => filterAgencies((tree && tree.agencies) || [], query),
    [tree, query],
  );
  const unlinked = useMemo(
    () => filterClients((tree && tree.unlinked) || [], query),
    [tree, query],
  );
  const booked = useMemo(
    () => filterClients((tree && tree.clients_booked_without_spots) || [], query),
    [tree, query],
  );
  const shownAgencies = agencies.slice(0, visibleCount);
  // Every hook this component owns is called above the loading return below,
  // including this one, and each reads the tree defensively because the first
  // render happens before the read resolves. A hook placed after that return
  // runs on the render that has the tree and not on the render that does not,
  // and React refuses to reconcile a component whose hook count grew between
  // two renders: it unmounts the whole application, not this panel. The demo
  // tally is derived from the payload rather than fetched, so it costs nothing
  // to compute it early and return zero while the tree is still null.
  const { demo: demoCampaignCount } = useMemo(
    () => demoTally(
      tree && tree.agencies,
      tree && tree.unlinked,
      tree && tree.clients_booked_without_spots,
    ),
    [tree],
  );

  if (!tree) {
    return <div className="clients-loading">{pageText(locale, 'Loading clients', 'טוען לקוחות')}</div>;
  }

  const counts = tree.counts || {};
  const agencyCount = counts.agencies || 0;
  const clientCount = counts.clients || 0;
  const campaignCount = counts.campaigns || 0;
  const agencyWord = plural(
    agencyCount,
    pageText(locale, 'agency', 'סוכנות'),
    pageText(locale, 'agencies', 'סוכנויות'),
  );
  const campaignWord = plural(
    campaignCount,
    pageText(locale, 'campaign', 'קמפיין'),
    pageText(locale, 'campaigns', 'קמפיינים'),
  );
  const countsLine = demoCampaignCount > 0
    ? pageText(
      locale,
      `${agencyCount} ${agencyWord}, ${clientCount} ${clientsWord(clientCount, locale)}, ${campaignCount} ${campaignWord} (${demoCampaignCount} demo seed data)`,
      `${isolate(agencyCount)} ${agencyWord}, ${isolate(clientCount)} ${clientsWord(clientCount, locale)}, ${isolate(campaignCount)} ${campaignWord} (${isolate(demoCampaignCount)} נתוני זרע הדגמה)`,
    )
    : pageText(
      locale,
      `${agencyCount} ${agencyWord}, ${clientCount} ${clientsWord(clientCount, locale)}, ${campaignCount} ${campaignWord}`,
      `${isolate(agencyCount)} ${agencyWord}, ${isolate(clientCount)} ${clientsWord(clientCount, locale)}, ${isolate(campaignCount)} ${campaignWord}`,
    );
  return (
    <section className="clients-tree" id="commercial-client-tree">
      <div className="clients-toolbar">
        <label className="clients-search">
          <Search size={14} aria-hidden="true" />
          <InputControl
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={pageText(locale, 'Find an agency or a client', 'חפשו סוכנות או לקוח')}
            aria-label={pageText(locale, 'Find an agency or a client', 'חפשו סוכנות או לקוח')}
          />
        </label>
        <span className="clients-counts">{countsLine}</span>
      </div>

      <p className="clients-basis">{basisLine(tree.basis, locale)}</p>

      {shownAgencies.map((agency) => {
        const expanded = openAgency === agency.agency_id;
        return (
          <article key={agency.agency_id} className={`card card-dense clients-agency${expanded ? ' open' : ''}`}>
            <Button
              type="button"
              className="card-body clients-agency-head"
              onClick={() => setOpenAgency(expanded ? '' : agency.agency_id)}
              aria-expanded={expanded}
            >
              <span className="clients-agency-icon"><Building2 size={16} strokeWidth={1.8} /></span>
              <span className="clients-agency-name">
                <strong>{agency.name}</strong>
                <small>{agency.agency_id}</small>
                {agency.status === 'suspended' ? (
                  <span className="clients-flag">{pageText(locale, 'suspended', 'מושהית')}</span>
                ) : null}
              </span>
              <span className="clients-agency-facts">
                <span className="numeric"><Figure>{agency.client_count}</Figure></span>
                <small>{clientsWord(agency.client_count, locale)}</small>
                <span className="numeric"><Figure>{exactMoney(agency.gross, locale)}</Figure></span>
                <small>{pageText(locale, 'gross', 'ברוטו')}</small>
                <span className="numeric"><Figure>{exactMoney(agency.net, locale)}</Figure></span>
                <small>{pageText(locale, 'net', 'נטו')}</small>
              </span>
              {expanded ? <ChevronUp size={15} aria-hidden="true" /> : <ChevronDown size={15} aria-hidden="true" />}
            </Button>
            {expanded ? (
              <div className="card-body clients-agency-body">
                <Terms agency={agency} locale={locale} />
                {localized(agency, 'money_reason', locale) ? (
                  <p className="clients-reason">{localized(agency, 'money_reason', locale)}</p>
                ) : null}
                <table className="clients-table">
                  <thead>
                    <tr>
                      <th scope="col">{pageText(locale, 'Client', 'לקוח')}</th>
                      <th scope="col" className="numeric-col">{pageText(locale, 'Gross', 'ברוטו')}</th>
                      <th scope="col" className="numeric-col">{pageText(locale, 'Net', 'נטו')}</th>
                      <th scope="col" className="numeric-col">{pageText(locale, 'Spots', 'תשדירים')}</th>
                      <th scope="col" className="numeric-col">{pageText(locale, 'Campaigns', 'קמפיינים')}</th>
                      <th scope="col">{pageText(locale, 'State', 'מצב')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {agency.clients.map((client) => (
                      <ClientRow key={client.advertiser} client={client} locale={locale} onOpen={onOpenClient} />
                    ))}
                  </tbody>
                </table>
                {agency.clients.length === 0 ? (
                  <p className="clients-reason">
                    {pageText(locale, 'No client buys through this agency yet.', 'אף לקוח אינו קונה דרך הסוכנות הזו עדיין.')}
                  </p>
                ) : null}
              </div>
            ) : null}
          </article>
        );
      })}

      {shownAgencies.length < agencies.length ? (
        <div className="clients-window-more" role="status">
          <span>{pageText(locale, `Showing ${shownAgencies.length} of ${agencies.length} agencies`, `מוצגות ${isolate(shownAgencies.length)} מתוך ${isolate(agencies.length)} סוכנויות`)}</span>
          <a href="#commercial-client-tree" role="button" className="clients-secondary"
             onClick={(event) => { event.preventDefault(); setVisibleCount((count) => count + AGENCY_WINDOW); }}
             onKeyDown={(event) => { if (event.key === ' ') { event.preventDefault(); setVisibleCount((count) => count + AGENCY_WINDOW); } }}>
            {pageText(locale, 'Show the next agencies', 'הציגו את הסוכנויות הבאות')}
          </a>
        </div>
      ) : null}

      <FlatGroup
        title={pageText(locale, 'Clients with no agency', 'לקוחות ללא סוכנות')}
        note={pageText(locale, 'they delivered, and no agency claims them', 'הם סיפקו, ואף סוכנות אינה משויכת אליהם')}
        clients={unlinked}
        locale={locale}
        onOpen={onOpenClient}
      />

      <FlatGroup
        title={pageText(locale, 'Booked, nothing priced in the day being read', 'הוזמנו, ללא תמחור ביום הנקרא')}
        note={pageText(locale, 'they have a campaign on file and no priced spot in the day being read', 'יש להם קמפיין רשום ואין תשדיר מתומחר ביום הנקרא')}
        clients={booked}
        locale={locale}
        onOpen={onOpenClient}
      />

      {query && !agencies.length && !unlinked.length && !booked.length ? (
        <p className="clients-reason">
          {pageText(
            locale,
            `Nothing here matches ${query}. The search reads agency names and ids, and client names and spellings.`,
            `שום דבר כאן אינו תואם ל־${isolate(query)}. החיפוש קורא שמות ומזהים של סוכנויות, ושמות וכתיבים של לקוחות.`,
          )}
        </p>
      ) : null}
    </section>
  );
}
