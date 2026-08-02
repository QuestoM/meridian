import React, { useMemo, useState } from 'react';
import { BellOff, RotateCcw, Trash2, X } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import './activity-feed.css';

// A persistent activity feed for the operator's own actions and alerts. Every
// notify() call lands here as a dated entry, not only a 2.6 second toast, so
// nothing scrolls away unseen. Entries can be dismissed and dismissed entries
// can be restored, which is the "can I dismiss or bring back a notification"
// capability the owner asked for. Entries are real events (a decision saved, a
// plan run finished, a download): never fabricated, so the feed is an honest
// record of what happened in this session.

function timeLabel(ts, locale) {
  if (!ts) return '';
  try {
    return new Date(ts).toLocaleTimeString(locale === 'he' ? 'he-IL' : 'en-US', { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

export default function ActivityFeed({ notifications, locale, onDismiss, onRestore, onClearAll, onRestoreAll, onClose }) {
  const he = locale === 'he';
  const [showDismissed, setShowDismissed] = useState(false);

  const active = useMemo(
    () => notifications.filter((n) => !n.dismissed).slice().reverse(),
    [notifications],
  );
  const dismissed = useMemo(
    () => notifications.filter((n) => n.dismissed).slice().reverse(),
    [notifications],
  );

  return (
    <aside className="activity-feed" role="dialog" aria-label={pageText(locale, 'Activity', 'פעילות')} dir={he ? 'rtl' : 'ltr'}>
      <div className="af-head">
        <div>
          <span className="af-kicker">{pageText(locale, 'Activity', 'פעילות')}</span>
          <h3>{pageText(locale, 'Notifications', 'התראות')}</h3>
        </div>
        <button type="button" className="af-close" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
          <X size={18} />
        </button>
      </div>

      <div className="af-actions">
        <button type="button" className="af-action" onClick={onClearAll} disabled={active.length === 0}>
          <BellOff size={13} />
          {pageText(locale, 'Dismiss all', 'סימון הכל כנצפה')}
        </button>
        <button type="button" className="af-action" onClick={() => setShowDismissed((v) => !v)} disabled={dismissed.length === 0}>
          <RotateCcw size={13} />
          {showDismissed
            ? pageText(locale, 'Hide dismissed', 'הסתרת שנצפו')
            : `${pageText(locale, 'Show dismissed', 'הצגת שנצפו')} (${dismissed.length})`}
        </button>
      </div>

      <div className="af-body">
        {active.length === 0 && !showDismissed && (
          <p className="af-empty">{pageText(locale, 'No new activity. Actions you take appear here.', 'אין פעילות חדשה. פעולות שתבצעו יופיעו כאן.')}</p>
        )}

        {active.map((n) => (
          <div className="af-item" key={n.id}>
            <div className="af-item-main">
              <span className="af-item-time" dir="ltr">{timeLabel(n.ts, locale)}</span>
              <span className="af-item-text">{pageText(locale, n.en, n.he)}</span>
            </div>
            <button type="button" className="af-item-btn" onClick={() => onDismiss(n.id)} aria-label={pageText(locale, 'Dismiss', 'סימון כנצפה')}>
              <X size={14} />
            </button>
          </div>
        ))}

        {showDismissed && dismissed.length > 0 && (
          <>
            <div className="af-divider">
              <span>{pageText(locale, 'Dismissed', 'נצפו')}</span>
              <button type="button" className="af-restore-all" onClick={onRestoreAll}>
                {pageText(locale, 'Restore all', 'שחזור הכל')}
              </button>
            </div>
            {dismissed.map((n) => (
              <div className="af-item dismissed" key={n.id}>
                <div className="af-item-main">
                  <span className="af-item-time" dir="ltr">{timeLabel(n.ts, locale)}</span>
                  <span className="af-item-text">{pageText(locale, n.en, n.he)}</span>
                </div>
                <button type="button" className="af-item-btn restore" onClick={() => onRestore(n.id)} aria-label={pageText(locale, 'Restore', 'שחזור')}>
                  <RotateCcw size={14} />
                </button>
              </div>
            ))}
          </>
        )}
      </div>

      <div className="af-foot">
        <a className="af-foot-link" href="#Versions" onClick={onClose}>
          {pageText(locale, 'Open the full history, including who changed what and how to put it back', 'לצפייה בהיסטוריה המלאה, כולל מי שינה מה ואיך להחזיר')}
        </a>
      </div>
    </aside>
  );
}
