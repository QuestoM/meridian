import React from 'react';
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
  { key: 'advertisers', en: 'Pricing rules', he: 'כללי תמחור' },
  { key: 'agencies', en: 'Agency records', he: 'כרטיסי סוכנות' },
];

export function ClientsHeader({ locale, gate, onOnboard }) {
  return (
    <div className="page-header">
      <div>
        <h1>{pageText(locale, 'Clients', 'לקוחות')}</h1>
        <p>
          {pageText(
            locale,
            'Agencies, the clients that buy through them, the campaigns booked under each client, and what every one of them delivered.',
            'סוכנויות, הלקוחות שקונים דרכן, הקמפיינים שהוזמנו תחת כל לקוח, ומה כל אחד מהם סיפק.',
          )}
        </p>
      </div>
      {gate.canEdit ? (
        <button type="button" className="clients-primary" onClick={onOnboard}>
          <UserPlus size={14} aria-hidden="true" />
          {pageText(locale, 'Onboard a client', 'קליטת לקוח')}
        </button>
      ) : (
        <p className="clients-refusal">{gate.reason}</p>
      )}
    </div>
  );
}

export function ClientsViewStrip({ locale, active, onSelect }) {
  return (
    <nav className="clients-views" role="tablist" aria-label={pageText(locale, 'Clients views', 'תצוגות לקוחות')}>
      {VIEW_LABELS.map((entry) => (
        <button
          key={entry.key}
          type="button"
          role="tab"
          aria-selected={entry.key === active}
          className={entry.key === active ? 'active' : ''}
          onClick={() => onSelect(entry.key)}
        >
          {pageText(locale, entry.en, entry.he)}
        </button>
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
      <button type="button" className="clients-retry" onClick={onRetry}>
        {pageText(locale, 'Try again', 'נסה שוב')}
      </button>
    </div>
  );
}
