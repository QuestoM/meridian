import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { pageText } from '../shell/surface-helpers';
import { Figure, Code, Name } from '../shell/bidi';
import { requestJson } from './assistant-stream';
import { keepPrefixWarm } from './kai-keep-warm';
import { useConversations } from './AssistantConversationsApi';
import { showRestoreVersion, useAssistantBatches, useAssistantThread } from './assistant-panel-state';
import { useAssistantPage } from '../shell/assistant-page-context';
import { useAsk } from './assistant-panel-ask';
import AssistantConversationsSidebar from './AssistantConversationsSidebar';
import AssistantConversationsRail from './AssistantConversationsRail';
import AssistantConversationToolbar from './AssistantConversationToolbar';
import AssistantProposalCard from './AssistantProposalCard';
import AssistantRunTrace, { useElapsed } from './AssistantRunTrace';
import AssistantUpload from './AssistantUpload';
import { Pressable } from '../studio/dom-controls';
import { AssistantExchange, ModelText, RichText } from './AssistantThread';
import { AssistantComposer, AssistantEmptyThread } from './AssistantComposer';
import { FOCUS_EVENT, FOCUS_PENDING } from './kai-shortcuts';
import { isolate } from '../shell/bidi';
import './assistant-console.css';
import './studio-ledger-kai.css';
import { formatDay } from '../shell/dates';

// The assistant console, named Mabat: a chat column grounded in the saved data
// plus a rail for pending proposals and the conversation history. Answers come
// only from the server; asks stream live and fall back quietly to the plain ask
// endpoint when the stream is unavailable. Proposed actions apply only after an
// explicit confirm, with an automatic restore point first, and that restore
// point is the undo control on the card that created it.
//
// Three things every long-running answer owes the person, all of them measured
// rather than decorative: a run trace naming what is being read right now, a
// clock in seconds since the question was sent, and a stop control. Discovery
// measured this panel sitting on "preparing an answer" for 499 s with no reply,
// no error and nothing to press.
//
// The grounding context warms only when the composer receives focus or input.
// Opening the dock for reading stays a provider-free operation.


function startedLabel(iso) {
  return formatDay(String(iso || '').slice(0, 10));
}

export default function AssistantPanel({ locale, notify, dock = false }) {
  const [status, setStatus] = useState(null);
  const [statusState, setStatusState] = useState('loading');
  const [clearing, setClearing] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [dockView, setDockView] = useState('conversation');
  const [historyOpen, setHistoryOpen] = useState(false);
  const dockTabsRef = useRef([]);
  const historyRef = useRef(null);
  const conv = useConversations(notify);
  const {
    batchMap, mergeBatches, refreshRail, refreshing, proposalsState, proposalsError,
    applyBusyId, applyResults, applyItems, rejectItems, pendingCount, visibleBatches,
  } = useAssistantBatches(notify);
  const { thread, threadLoading, actingUser, appendEntry, resetLocal, markAdopted } = useAssistantThread(conv);
  const pageState = useAssistantPage();
  const threadRef = useRef(null);
  const composerRef = useRef(null);
  const markUnavailable = useCallback((reason) => {
    setStatus((prev) => ({ ...(prev || {}), available: false, reason: reason || '' }));
  }, []);
  // The question, its typed references, and sending them: one idea, in one
  // module, because AssistantPanel.jsx stood two lines under the size law and a
  // reference has to travel beside the prose it belongs to.
  const {
    question, setQuestion, refs, setRefs, asking, live, ask, stopAsk, onComposerKeyDown,
  } = useAsk({
    locale, conv, pageState, mergeBatches, refreshRail, appendEntry, markAdopted,
    onUnavailable: markUnavailable,
  });
  const elapsed = useElapsed(asking, live ? live.startedAt : null);

  useEffect(() => {
    let active = true;
    requestJson('/api/assistant/status')
      .then((body) => {
        if (!active) return;
        setStatus(body);
        setStatusState('ready');
      })
      .catch(() => {
        if (!active) return;
        setStatusState('error');
    });
    refreshRail();
    // Provider-backed prefix warming is intentionally tied to composer
    // activity below. Merely mounting or testing the dock performs status and
    // saved-state reads only; it never starts a model/provider request.
    return () => {
      active = false;
    };
  }, [refreshRail]);

  // Cmd J from any surface opens the dock and lands the cursor here. When the
  // dock was closed the panel did not exist to hear the event, so the shortcut
  // leaves a flag this mount consumes.
  useEffect(() => {
    function onFocusRequest() {
      if (composerRef.current) composerRef.current.focus();
    }
    window.addEventListener(FOCUS_EVENT, onFocusRequest);
    if (window[FOCUS_PENDING]) {
      window[FOCUS_PENDING] = false;
      onFocusRequest();
    }
    return () => window.removeEventListener(FOCUS_EVENT, onFocusRequest);
  }, []);

  useEffect(() => {
    const node = threadRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [thread, asking, live, threadLoading]);

  // With conversations available the header clear removes the active
  // conversation through the same endpoint the rail uses; the hook then
  // selects the newest remaining conversation and the thread reloads. On an
  // older backend the legacy whole-thread delete stays.
  const clearConversation = useCallback(async () => {
    setClearing(true);
    try {
      if (conv.supported && conv.activeId) {
        await conv.remove(conv.activeId);
      } else {
        await requestJson('/api/assistant/thread', { method: 'DELETE' });
        resetLocal();
      }
      setConfirmClear(false);
    } catch (error) {
      if (notify) notify(`Clearing the conversation failed (${isolate(error.message)}).`, `מחיקת השיחה נכשלה (${isolate(error.message)}).`);
    } finally {
      setClearing(false);
    }
  }, [notify, conv, resetLocal]);

  // A suggestion and an ask-again both put PROSE back in the box and no typed
  // references with it. That is deliberate rather than missing: the labels are
  // still in the sentence and the free-text routes still resolve them, and a
  // reference restored without the operator making it would be a binding they
  // never chose. The old references go with the old text.
  function pickSuggestion(text) {
    setRefs([]);
    setQuestion(text);
    if (composerRef.current) composerRef.current.focus();
  }

  function showConversation() {
    setDockView('conversation');
    setHistoryOpen(false);
  }

  async function startConversation() {
    setConfirmClear(false);
    const createdId = await conv.create();
    if (!createdId) return;
    setQuestion('');
    setRefs([]);
    setHistoryOpen(false);
    window.requestAnimationFrame(() => composerRef.current?.focus({ preventScroll: true }));
  }

  function toggleHistory() {
    setConfirmClear(false);
    setHistoryOpen((open) => {
      if (!open) conv.refreshList();
      return !open;
    });
  }

  function returnFromHistory(id) {
    if (id !== conv.activeId) {
      setQuestion('');
      setRefs([]);
    }
    setHistoryOpen(false);
    window.requestAnimationFrame(() => threadRef.current?.focus({ preventScroll: true }));
  }

  useEffect(() => {
    if (!historyOpen) return undefined;
    const frame = window.requestAnimationFrame(() => historyRef.current?.focus({ preventScroll: true }));
    return () => window.cancelAnimationFrame(frame);
  }, [historyOpen]);

  function onDockTabKeyDown(event, index) {
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = 1;
    else if (event.key === 'ArrowRight') next = (index + (locale === 'he' ? -1 : 1) + 2) % 2;
    else if (event.key === 'ArrowLeft') next = (index + (locale === 'he' ? 1 : -1) + 2) % 2;
    else return;
    event.preventDefault();
    const view = next === 0 ? 'conversation' : 'actions';
    if (view === 'conversation') showConversation();
    else setDockView(view);
    dockTabsRef.current[next]?.focus();
  }

  // Auto-grow: the composer rests at one row (matching the send button) and
  // grows with the content up to a cap, so the default state never towers over
  // the button and a long question still stays fully visible.
  useEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [question]);

  const unavailable = statusState === 'ready' && status && status.available === false;
  // Provider diagnostics can contain implementation names and environment
  // variables. Operators get the service state and recovery owner, never raw
  // infrastructure prose from the status endpoint.
  const reasonLabel = unavailable
    ? pageText(locale, 'The assistant service is not connected in this environment.', 'שירות העוזר אינו מחובר בסביבה הזו.')
    : '';
  const statusText = statusState === 'loading' ? pageText(locale, 'Checking availability', 'בודק זמינות')
    : statusState === 'error' ? pageText(locale, 'No connection to the Kairos server', 'אין חיבור לשרת Kairos')
    : unavailable ? pageText(locale, 'Not available', 'לא זמין')
    : pageText(locale, 'Connected', 'מחובר');
  const dotClass = statusState === 'loading' ? 'loading' : statusState === 'error' ? 'error' : unavailable ? 'off' : 'on';

  // The quiet location chip: where the assistant thinks the operator is, so
  // the grounding sent with each ask is transparent on screen.
  const page = pageState && pageState.page && pageState.page.label ? pageState.page : null;
  const entityLabel = pageState && pageState.entity && pageState.entity.label ? pageState.entity.label : '';
  // The page name follows the header language, but the open record's name is
  // data and is Hebrew in this market, so the English chip mixes scripts. Each
  // name carries its own isolate rather than being interpolated into the
  // sentence, or the record name takes the comma and welds to the page name.
  const locationText = page ? (
    <>
      {pageText(locale, 'You are on the ', 'אתם בעמוד ')}
      <Name>{page.label}</Name>
      {pageText(locale, ' page', '')}
      {entityLabel ? <>{', '}<Name>{entityLabel}</Name></> : null}
    </>
  ) : null;

  // Conversation statistics from data already on screen: exchange count from
  // the loaded entries, the start date of the first entry, and applied changes
  // counted from this conversation's loaded batches. Nothing is invented, so
  // the applied figure only renders when at least one applied item is loaded.
  const startedAt = thread.length > 0 && thread[0].at ? startedLabel(thread[0].at) : '';
  const appliedCount = useMemo(() => {
    const seen = new Set();
    let total = 0;
    for (const entry of thread) {
      if (!entry.batchId || seen.has(entry.batchId)) continue;
      seen.add(entry.batchId);
      const batch = batchMap[entry.batchId];
      if (batch) total += batch.items.filter((item) => item.status === 'applied').length;
    }
    return total;
  }, [thread, batchMap]);

  function renderProposalCard(batch) {
    return <AssistantProposalCard key={batch.batch_id} batch={batch} locale={locale} busy={applyBusyId === batch.batch_id} applyResult={applyResults[batch.batch_id] || null} onApply={(ids) => applyItems(batch.batch_id, ids)} onReject={(ids) => rejectItems(batch.batch_id, ids)} onShowRestore={showRestoreVersion} notify={notify} onUndone={refreshRail} />;
  }

  // The action plane's own honest state, from the endpoint that owns it. It
  // says whether Mabat may propose a change at all, and when it may not, why.
  const actionPlane = status && status.action_plane && typeof status.action_plane === 'object' ? status.action_plane : null;

  const statusCluster = (
    <div className="asst-status" role="status" aria-live="polite" aria-atomic="true">
      <span className={`asst-dot ${dotClass}`} aria-hidden="true" />
      <span>{statusText}</span>
      {status && status.model ? <Code>{status.model}</Code> : null}
      {actionPlane && actionPlane.enabled === true ? <span className="asst-chip on">{pageText(locale, 'Can propose changes', 'יכול להציע שינויים')}</span> : null}
      {actionPlane && actionPlane.enabled === false ? <span className="asst-chip" title={actionPlane.reason || ''}>{pageText(locale, 'Answers only', 'מענה בלבד')}</span> : null}
    </div>
  );

  return (
    <section className={dock ? 'asst-workspace asst-in-dock' : 'page-workspace asst-workspace'}>
      {dock ? (
        <div className="asst-dock-meta">
          {statusCluster}
          {locationText ? <span className="asst-location">{locationText}</span> : null}
          {!threadLoading && thread.length > 0 ? (
            <p className="asst-stats">
              <span>{thread.length === 1 ? pageText(locale, 'one question in this conversation', 'שאלה אחת בשיחה') : pageText(locale, `${thread.length} questions in this conversation`, `${thread.length} שאלות בשיחה`)}</span>
              {startedAt ? <span>{pageText(locale, 'started ', 'התחילה ב-')}<Figure>{startedAt}</Figure></span> : null}
              {appliedCount > 0 ? <span>{appliedCount === 1 ? pageText(locale, 'one change applied', 'הוחל שינוי אחד') : pageText(locale, `${appliedCount} changes applied`, `הוחלו ${appliedCount} שינויים`)}</span> : null}
            </p>
          ) : null}
        </div>
      ) : (
        <div className="page-header">
          <div>
            <h1>{pageText(locale, 'Mabat, operations assistant', 'מבט, עוזר תפעולי')}</h1>
            <p>{pageText(locale, 'Mabat answers from saved data only. Proposed actions apply only after your approval, with an automatic restore point first.', 'מבט עונה רק מן הנתונים השמורים. פעולה מוצעת תוחל רק לאחר אישור שלכם, ולאחר יצירת נקודת שחזור.')}</p>
          </div>
          {statusCluster}
        </div>
      )}

      {dock ? (
        <div className="asst-dock-view-tabs" role="tablist" aria-label={pageText(locale, 'Assistant work areas', 'אזורי העבודה של העוזר')}>
          <Pressable
            ref={(node) => { dockTabsRef.current[0] = node; }}
            type="button"
            role="tab"
            id="assistant-dock-tab-conversation"
            aria-controls="assistant-dock-panel-conversation"
            aria-selected={dockView === 'conversation'}
            tabIndex={dockView === 'conversation' ? 0 : -1}
            className={dockView === 'conversation' ? 'active' : ''}
            onClick={showConversation}
            onKeyDown={(event) => onDockTabKeyDown(event, 0)}
          >
            {pageText(locale, 'Conversation', 'שיחה')}
          </Pressable>
          <Pressable
            ref={(node) => { dockTabsRef.current[1] = node; }}
            type="button"
            role="tab"
            id="assistant-dock-tab-actions"
            aria-controls="assistant-dock-panel-actions"
            aria-selected={dockView === 'actions'}
            tabIndex={dockView === 'actions' ? 0 : -1}
            className={dockView === 'actions' ? 'active' : ''}
            onClick={() => setDockView('actions')}
            onKeyDown={(event) => onDockTabKeyDown(event, 1)}
          >
            {pageText(locale, 'Actions', 'פעולות')}
          </Pressable>
        </div>
      ) : null}

      <div className="asst-layout">
        <section
          className="page-panel asst-chat"
          hidden={dock && dockView !== 'conversation'}
          id={dock ? 'assistant-dock-panel-conversation' : undefined}
          role={dock ? 'tabpanel' : undefined}
          aria-labelledby={dock ? 'assistant-dock-tab-conversation' : undefined}
        >
          <div className="panel-head">
            <div>
              <h2>{pageText(locale, 'Conversation', 'שיחה')}</h2>
              <span>{pageText(locale, 'Saved to your account and shown here when you return.', 'נשמרת לחשבון שלכם ומוצגת כאן בכל חזרה.')}</span>
              {actingUser ? <span className="asst-user">{pageText(locale, 'Acting user', 'מבצע')}: <b><Name>{actingUser === 'auth-disabled' ? pageText(locale, 'No sign-in', 'ללא כניסה') : actingUser}</Name></b></span> : null}
            </div>
            <AssistantConversationToolbar
              locale={locale} supported={conv.supported} listState={conv.listState} busy={conv.busy} asking={asking}
              historyOpen={historyOpen} canClear={thread.length > 0 && !threadLoading && !historyOpen}
              confirmClear={confirmClear} clearing={clearing} onNew={startConversation}
              onToggleHistory={toggleHistory} onRequestClear={() => setConfirmClear(true)}
              onClear={clearConversation} onCancelClear={() => setConfirmClear(false)}
            />
          </div>

          <div
            className="asst-conversation-history"
            id="assistant-conversation-history"
            ref={historyRef}
            tabIndex={-1}
            role="region"
            aria-label={pageText(locale, 'Conversation history', 'היסטוריית שיחות')}
            hidden={!historyOpen}
          >
            <AssistantConversationsRail locale={locale} conv={conv} disabled={asking} onSelect={returnFromHistory} />
          </div>
          {!historyOpen ? <>
          <div className="asst-thread" ref={threadRef} tabIndex={-1} aria-live="polite" aria-relevant="additions text" aria-busy={threadLoading || asking || undefined}>
            {threadLoading && thread.length === 0 ? (
              <div className="asst-loading">{pageText(locale, 'Loading your conversation', 'טוען את השיחה שלכם')}</div>
            ) : null}
            {!threadLoading && thread.length === 0 && !asking ? (
              <AssistantEmptyThread locale={locale} showSuggestions={!unavailable && statusState !== 'loading'} onPick={pickSuggestion} />
            ) : null}

            {thread.map((entry) => (
              <AssistantExchange key={entry.id} entry={entry} locale={locale} proposalCard={entry.batchId && batchMap[entry.batchId] ? renderProposalCard(batchMap[entry.batchId]) : null} onAskAgain={() => pickSuggestion(entry.question)} />
            ))}

            {live ? (
              <article className="asst-exchange">
                <RichText className="asst-q" text={live.question} />
                <AssistantRunTrace locale={locale} live={live} elapsed={elapsed} onStop={stopAsk} />
                {live.text ? <ModelText className="card asst-a" text={live.text} /> : null}
              </article>
            ) : null}
          </div>

          {unavailable ? (
            <div className="asst-unavailable" role="status">
              <strong>{pageText(locale, 'Mabat is not available.', 'מבט אינו זמין.')}</strong>
              {reasonLabel ? <span>{reasonLabel}</span> : null}
              <span>{pageText(locale, 'Ask a system administrator to connect the service, then try again.', 'בקשו ממנהל המערכת לחבר את השירות, ואז נסו שוב.')}</span>
            </div>
          ) : null}

          <AssistantUpload
            locale={locale}
            notify={notify}
            disabled={asking || unavailable}
            onSuggest={(text) => { setQuestion(text); if (composerRef.current) composerRef.current.focus(); }}
          />

          <AssistantComposer
            locale={locale}
            composerRef={composerRef}
            question={question}
            onQuestionChange={setQuestion}
            refs={refs}
            onRefsChange={setRefs}
            onKeyDown={onComposerKeyDown}
            unavailable={unavailable}
            asking={asking}
            onSend={ask}
            onStop={stopAsk}
            onActivity={keepPrefixWarm}
          />
          </> : null}
        </section>

        <AssistantConversationsSidebar
          locale={locale}
          conv={conv}
          notify={notify}
          disabled={asking}
          pendingCount={pendingCount}
          proposalsState={proposalsState}
          proposalsError={proposalsError}
          visibleBatches={visibleBatches}
          renderProposalCard={renderProposalCard}
          refreshing={refreshing}
          onRefresh={refreshRail}
          onShowRestore={showRestoreVersion}
          hidden={dock && dockView !== 'actions'}
          dockPanel={dock}
        />
      </div>
    </section>
  );
}
