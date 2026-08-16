import React from 'react';
import { Button } from '../studio/actions';
import { UserPlus } from 'lucide-react';
import { pageText } from '../shell/format';

// The chrome around the five views: the destination's title and sentence, the
// one control that starts an order, the view strip, and the banner that names a
// read which failed.
//
// Split out of ClientsWorkspace.jsx to keep that file inside the project's
// file-size law. It holds no state and performs no read: everything it renders
// is a prop, so the workspace keeps every decision and this module keeps the
// markup. Nothing moved changed.

// The five views this destination holds, in the order the strip shows them. The
// list lives here because the strip is the only thing that renders it, and the
// workspace imports it to name a read that failed by the view it feeds.
export const VIEW_LABELS = [
  { key: 'clients', en: 'Clients', he: 'לקוחות' },
  { key: 'money', en: 'Money', he: 'כסף' },
  { key: 'campaigns', en: 'Campaigns', he: 'קמפיינים' },
  { key: 'pacing', en: 'Delivery pace', he: 'קצב אספקה' },
  { key: 'advertisers', en: 'Pricing rules', he: 'כללי תמחור' },
  { key: 'agencies', en: 'Agency records', he: 'כרטיסי סוכנות' },
];

const VIEW_HEADERS = {
  clients: {
    en: 'Clients',
    he: 'לקוחות',
    descriptionEn: 'Agencies, clients, campaigns, flight windows and the delivery state of each campaign.',
    descriptionHe: 'סוכנויות, לקוחות, קמפיינים, חלונות שידור ומצב האספקה של כל קמפיין.',
  },
  money: {
    en: 'Commercial ledger',
    he: 'ספר מסחרי',
    descriptionEn: 'Gross and net value from the priced traffic ledger, grouped by the commercial object that owns it.',
    descriptionHe: 'ערכי ברוטו ונטו מספר התשדירים המתומחרים, לפי סוכנות, לקוח או קמפיין.',
  },
  campaigns: {
    en: 'Campaigns',
    he: 'קמפיינים',
    descriptionEn: 'Booked flight windows, delivery commitments, counted delivery and current campaign state.',
    descriptionHe: 'חלונות שידור, התחייבויות, אספקה שנספרה והמצב הנוכחי של כל קמפיין.',
  },
  pacing: {
    en: 'Delivery pace',
    he: 'קצב אספקה',
    descriptionEn: 'Published pacing decisions, commitment ratios and make-good work from the current delivery read.',
    descriptionHe: 'החלטות קצב, יחסי התחייבות ופיצויי שידור מתוך קריאת האספקה הנוכחית.',
  },
  advertisers: {
    en: 'Pricing rules',
    he: 'כללי תמחור',
    descriptionEn: 'Advertiser identities, aliases and the pricing rule applied when the traffic ledger is read.',
    descriptionHe: 'זהויות מפרסמים, כתיבים חלופיים וכלל התמחור שמופעל בקריאת ספר התשדירים.',
  },
  agencies: {
    en: 'Agency records',
    he: 'כרטיסי סוכנות',
    descriptionEn: 'Agency terms, contacts and the clients linked to each commercial account.',
    descriptionHe: 'תנאי סוכנות, אנשי קשר והלקוחות המקושרים לכל חשבון מסחרי.',
  },
};

export function ClientsHeader({ locale, gate, onOnboard, active = 'clients' }) {
  const header = VIEW_HEADERS[active] || VIEW_HEADERS.clients;
  const showOnboard = active === 'clients';
  return (
    <div className="page-header">
      <div>
        <h1>{pageText(locale, header.en, header.he)}</h1>
        <p>{pageText(locale, header.descriptionEn, header.descriptionHe)}</p>
      </div>
      {showOnboard && gate.canEdit ? (
        <Button type="button" className="clients-primary" onClick={onOnboard}>
          <UserPlus size={14} aria-hidden="true" />
          {pageText(locale, 'Onboard a client', 'קליטת לקוח')}
        </Button>
      ) : showOnboard ? (
        <p className="clients-refusal">{gate.reason}</p>
      ) : null}
    </div>
  );
}

export function ClientsViewStrip({ locale, active, onSelect }) {
  function moveTab(event) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const tabs = Array.from(event.currentTarget.querySelectorAll('[role="tab"]'));
    const current = tabs.indexOf(document.activeElement);
    const rtl = locale === 'he';
    let next = current < 0 ? 0 : current;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (event.key === 'ArrowRight') next = (next + (rtl ? -1 : 1) + tabs.length) % tabs.length;
    if (event.key === 'ArrowLeft') next = (next + (rtl ? 1 : -1) + tabs.length) % tabs.length;
    const key = tabs[next]?.dataset.view;
    if (key) {
      tabs[next].focus();
      onSelect(key);
    }
  }

  return (
    <nav
      className="clients-views"
      role="tablist"
      aria-label={pageText(locale, 'Commercial workspace views', 'תצוגות סביבת העבודה המסחרית')}
      onKeyDown={moveTab}
    >
      {VIEW_LABELS.map((entry) => (
        <Button
          key={entry.key}
          type="button"
          role="tab"
          id={`commercial-tab-${entry.key}`}
          data-view={entry.key}
          aria-controls={`commercial-panel-${entry.key}`}
          aria-selected={entry.key === active}
          tabIndex={entry.key === active ? 0 : -1}
          className={entry.key === active ? 'active' : ''}
          onClick={() => onSelect(entry.key)}
        >
          {pageText(locale, entry.en, entry.he)}
        </Button>
      ))}
    </nav>
  );
}

// A read that failed is a failure, not an empty result, and it is named by the
// view it feeds so the reader knows which part of the screen is not answering.
export function ClientsLoadFailure({ locale, failed, onRetry }) {
  if (!failed || failed.length === 0) {
    return null;
  }
  return (
    <div className="clients-error" role="alert">
      <p>
        {pageText(
          locale,
          `These sections failed to load: ${failed.map((s) => s.en).join(', ')}. What is missing is a failure, not an empty result.`,
          `הקטעים הבאים לא נטענו: ${failed.map((s) => s.he).join(', ')}. מה שחסר הוא כשל, לא תוצאה ריקה.`,
        )}
      </p>
      <Button type="button" className="clients-retry" onClick={onRetry}>
        {pageText(locale, 'Try again', 'נסה שוב')}
      </Button>
    </div>
  );
}
