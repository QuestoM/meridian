import React, { useEffect, useState } from 'react';
import { RefreshCcw } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import AssistantConversationsRail from './AssistantConversationsRail';
import AssistantConversationsChanges from './AssistantConversationsChanges';

// The assistant side rail: the conversations section on top, then the pending
// actions and applied-changes tabs. On a backend without conversations (the
// index answers 404) the conversations section and the changes tab are hidden
// and the rail renders exactly the pre-conversations pending view, so nothing
// pretends to work against endpoints that do not exist.

export default function AssistantConversationsSidebar({ locale, conv, notify, disabled, pendingCount, proposalsState, proposalsError, visibleBatches, renderProposalCard, refreshing, onRefresh, onShowRestore }) {
  const [tab, setTab] = useState('pending');
  const [changesNonce, setChangesNonce] = useState(0);
  const tabsShown = conv.supported === true;
  const railShown = conv.supported === true || conv.listState === 'error';

  useEffect(() => {
    if (!tabsShown && tab !== 'pending') setTab('pending');
  }, [tabsShown, tab]);

  const pendingBadge = pendingCount > 0 ? <span className="asst-badge" dir="ltr">{pendingCount}</span> : null;

  return (
    <aside className="page-panel asst-rail">
      {railShown ? <AssistantConversationsRail locale={locale} conv={conv} disabled={disabled} /> : null}

      <div className="asst-rail-tabs">
        {tabsShown ? (
          <>
            <button type="button" className={`asst-tab${tab === 'pending' ? ' active' : ''}`} onClick={() => setTab('pending')}>
              {pageText(locale, 'Pending actions', 'פעולות ממתינות')}
              {pendingBadge}
            </button>
            <button type="button" className={`asst-tab${tab === 'changes' ? ' active' : ''}`} onClick={() => setTab('changes')}>
              {pageText(locale, 'Applied changes', 'שינויים שבוצעו')}
            </button>
          </>
        ) : (
          <span className="asst-rail-title">
            {pageText(locale, 'Pending actions', 'פעולות ממתינות')}
            {pendingBadge}
          </span>
        )}
        <button
          type="button"
          className="asst-refresh"
          onClick={() => { if (tab === 'changes' && tabsShown) setChangesNonce((nonce) => nonce + 1); else onRefresh(); }}
          disabled={refreshing}
          aria-label={pageText(locale, 'Refresh', 'רענון')}
        >
          <RefreshCcw size={13} className={refreshing ? 'asst-spin' : ''} />
        </button>
      </div>

      <div className="asst-rail-body">
        {tab === 'changes' && tabsShown ? (
          <AssistantConversationsChanges locale={locale} conversationId={conv.activeId} notify={notify} onShowRestore={onShowRestore} reloadNonce={changesNonce} />
        ) : proposalsState === 'loading' ? (
          <div className="asst-loading">{pageText(locale, 'Loading pending actions', 'טוען פעולות ממתינות')}</div>
        ) : proposalsState === 'error' ? (
          <div className="asst-error-note">{pageText(locale, 'Pending actions could not be loaded (', 'לא ניתן לטעון את הפעולות הממתינות (')}<bdi dir="auto">{proposalsError}</bdi>{').'}</div>
        ) : visibleBatches.length === 0 ? (
          <div className="asst-empty">{pageText(locale, 'No pending actions. When you ask the assistant for a change, its proposals appear here for approval.', 'אין פעולות ממתינות. כשתבקשו מהעוזר שינוי, ההצעות שלו יופיעו כאן לאישור.')}</div>
        ) : (
          visibleBatches.map((batch) => renderProposalCard(batch))
        )}
      </div>
    </aside>
  );
}
