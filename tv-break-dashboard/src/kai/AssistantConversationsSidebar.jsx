import React, { useEffect, useRef, useState } from 'react';
import { RefreshCcw } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { Figure, Name } from '../shell/bidi';
import AssistantConversationsChanges from './AssistantConversationsChanges';
import { Pressable } from '../studio/dom-controls';

// The assistant action surface: pending actions and applied actions only.
// Conversation navigation belongs to the conversation surface and opens there
// on demand, so this panel never sacrifices action-review space to history.

export default function AssistantConversationsSidebar({ locale, conv, notify, disabled, pendingCount, proposalsState, proposalsError, visibleBatches, renderProposalCard, refreshing, onRefresh, onShowRestore, hidden = false, dockPanel = false }) {
  const [tab, setTab] = useState(() => {
    if (typeof window === 'undefined') return 'pending';
    return new URLSearchParams(window.location.search).get('assistantRail') === 'changes' ? 'changes' : 'pending';
  });
  const [changesNonce, setChangesNonce] = useState(0);
  const tabsRef = useRef([]);
  const tabsShown = conv.supported === true;

  useEffect(() => {
    if (conv.supported === false && tab !== 'pending') setTab('pending');
  }, [conv.supported, tab]);

  useEffect(() => {
    function syncFromAddress() {
      const requested = new URLSearchParams(window.location.search).get('assistantRail');
      setTab(requested === 'changes' && conv.supported !== false ? 'changes' : 'pending');
    }
    window.addEventListener('popstate', syncFromAddress);
    return () => window.removeEventListener('popstate', syncFromAddress);
  }, [conv.supported]);

  function pickTab(next) {
    setTab(next);
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    if (next === 'changes') params.set('assistantRail', next);
    else params.delete('assistantRail');
    const query = params.toString();
    window.history.pushState({ workspace: 'assistant', section: next }, '', `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`);
  }

  function onTabKeyDown(event, index) {
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = 1;
    else if (event.key === 'ArrowRight') next = (index + (locale === 'he' ? -1 : 1) + 2) % 2;
    else if (event.key === 'ArrowLeft') next = (index + (locale === 'he' ? 1 : -1) + 2) % 2;
    else return;
    event.preventDefault();
    pickTab(next === 0 ? 'pending' : 'changes');
    tabsRef.current[next]?.focus();
  }

  const pendingBadge = pendingCount > 0 ? <span className="asst-badge"><Figure>{pendingCount}</Figure></span> : null;

  return (
    <aside
      className="page-panel asst-rail"
      hidden={hidden}
      id={dockPanel ? 'assistant-dock-panel-actions' : undefined}
      role={dockPanel ? 'tabpanel' : undefined}
      aria-labelledby={dockPanel ? 'assistant-dock-tab-actions' : undefined}
    >
      <div className="asst-rail-tabs">
        {tabsShown ? (
          <div className="asst-rail-tablist" role="tablist" aria-label={pageText(locale, 'Assistant activity', 'פעילות העוזר')}>
            <Pressable ref={(node) => { tabsRef.current[0] = node; }} type="button" role="tab" id="assistant-tab-pending" aria-selected={tab === 'pending'} aria-controls="assistant-rail-panel" tabIndex={tab === 'pending' ? 0 : -1} className={`asst-tab${tab === 'pending' ? ' active' : ''}`} onClick={() => pickTab('pending')} onKeyDown={(event) => onTabKeyDown(event, 0)}>
              {pageText(locale, 'Pending actions', 'פעולות ממתינות')}
              {pendingBadge}
            </Pressable>
            <Pressable ref={(node) => { tabsRef.current[1] = node; }} type="button" role="tab" id="assistant-tab-changes" aria-selected={tab === 'changes'} aria-controls="assistant-rail-panel" tabIndex={tab === 'changes' ? 0 : -1} className={`asst-tab${tab === 'changes' ? ' active' : ''}`} onClick={() => pickTab('changes')} onKeyDown={(event) => onTabKeyDown(event, 1)}>
              {pageText(locale, 'Applied actions', 'פעולות שבוצעו')}
            </Pressable>
          </div>
        ) : (
          <span className="asst-rail-title">
            {pageText(locale, 'Pending actions', 'פעולות ממתינות')}
            {pendingBadge}
          </span>
        )}
        <Pressable
          type="button"
          className="asst-refresh"
          onClick={() => { if (tab === 'changes' && tabsShown) setChangesNonce((nonce) => nonce + 1); else onRefresh(); }}
          disabled={refreshing}
          aria-label={pageText(locale, 'Refresh', 'רענון')}
        >
          <RefreshCcw size={13} className={refreshing ? 'asst-spin' : ''} />
        </Pressable>
      </div>

      <div className="asst-rail-body" id="assistant-rail-panel" role={tabsShown ? 'tabpanel' : undefined} aria-labelledby={tabsShown ? `assistant-tab-${tab}` : undefined} tabIndex={tabsShown ? 0 : undefined}>
        {tab === 'changes' && tabsShown ? (
          <AssistantConversationsChanges locale={locale} conversationId={conv.activeId} notify={notify} onShowRestore={onShowRestore} reloadNonce={changesNonce} />
        ) : proposalsState === 'loading' ? (
          <div className="asst-loading">{pageText(locale, 'Loading pending actions', 'טוען פעולות ממתינות')}</div>
        ) : proposalsState === 'error' ? (
          <div className="asst-error-note">{pageText(locale, 'Pending actions could not be loaded (', 'לא ניתן לטעון את הפעולות הממתינות (')}<Name>{proposalsError}</Name>{').'}</div>
        ) : visibleBatches.length === 0 ? (
          <div className="asst-empty">{pageText(locale, 'No pending actions. When you ask the assistant for a change, its proposals appear here for approval.', 'אין פעולות ממתינות. כשתבקשו מהעוזר שינוי, ההצעות שלו יופיעו כאן לאישור.')}</div>
        ) : (
          visibleBatches.map((batch) => renderProposalCard(batch))
        )}
      </div>
    </aside>
  );
}
