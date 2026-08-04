import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { Trash2 } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { postJson, requestJson, streamAsk, warmContext } from './assistant-stream';
import { useConversations } from './AssistantConversationsApi';
import { asArray, normalizeBatch, useAssistantBatches, useAssistantThread } from './assistant-panel-state';
import { buildPageContext, useAssistantPage } from '../shell/assistant-page-context';
import AssistantConversationsSidebar from './AssistantConversationsSidebar';
import AssistantProposalCard from './AssistantProposalCard';
import AssistantRunTrace, { useElapsed } from './AssistantRunTrace';
import AssistantUpload from './AssistantUpload';
import { AssistantExchange, ModelText, RichText } from './AssistantThread';
import { AssistantComposer, AssistantEmptyThread } from './AssistantComposer';
import { FOCUS_EVENT, FOCUS_PENDING } from './kai-shortcuts';
import { unrecordedProposalClaim } from './kai-claimed-action';
import { applyStage, noteStageLimits } from './kai-live-turn';
import { isolate } from './kai-bidi';
import './assistant-console.css';

// The assistant console, named Kai: a chat column grounded in the saved data
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
// The grounding context is warmed the moment the panel mounts, which is the
// moment the person starts typing, so the ask does not pay for it afterwards.


function startedLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleDateString(locale === 'he' ? 'he-IL' : 'en-US', { day: '2-digit', month: '2-digit', year: 'numeric' });
}

export default function AssistantPanel({ locale, notify, dock = false }) {
  const [status, setStatus] = useState(null);
  const [statusState, setStatusState] = useState('loading');
  const [question, setQuestion] = useState('');
  const [clearing, setClearing] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [asking, setAsking] = useState(false);
  const [live, setLive] = useState(null);
  const conv = useConversations(notify);
  const {
    batchMap, mergeBatches, refreshRail, refreshing, proposalsState, proposalsError,
    applyBusyId, applyResults, applyItems, rejectItems, pendingCount, visibleBatches,
  } = useAssistantBatches(notify);
  const { thread, threadLoading, actingUser, appendEntry, resetLocal, markAdopted } = useAssistantThread(conv);
  const pageState = useAssistantPage();
  const threadRef = useRef(null);
  const composerRef = useRef(null);
  const abortRef = useRef(null);
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
    // Build the grounding context while the person is still typing. Measured on
    // the server: 11.13 s cold, 0.034 s warm, so this is the difference between
    // paying for it before the first token and not paying for it at all.
    const controller = new AbortController();
    warmContext(controller.signal);
    return () => {
      active = false;
      controller.abort();
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

  const finishAsk = useCallback((trimmed, body, measured) => {
    if (body.available === false) {
      setStatus((prev) => ({ ...(prev || {}), available: false, reason: body.reason || body.error }));
      appendEntry({ question: trimmed, error: String(body.reason || body.error || '') });
      return;
    }
    // Adopt the conversation id the ask landed in: a new conversation minted
    // by the server appears in the list, and the id rides on the next ask.
    const returnedConv = body.conversation_id ? String(body.conversation_id) : null;
    if (returnedConv) {
      if (returnedConv !== conv.activeId) {
        markAdopted(returnedConv);
        conv.adopt(returnedConv);
      }
      if (conv.supported) conv.refreshList();
    }
    const batch = body.proposals ? normalizeBatch(body.proposals) : null;
    if (batch) mergeBatches([batch], false);
    appendEntry({
      question: trimmed,
      answer: body.answer ? String(body.answer) : null,
      error: body.error ? String(body.error) : !body.answer && !batch ? pageText(locale, 'The server returned no answer.', 'השרת לא החזיר תשובה.') : null,
      // Said a proposal is pending when this payload recorded none.
      unrecordedClaim: unrecordedProposalClaim(body, batch),
      disclosure: typeof body.context_disclosure === 'string' ? body.context_disclosure : '',
      sources: asArray(body.grounding && body.grounding.sources),
      toolTrace: asArray(body.tool_trace),
      truncated: Boolean(body.truncated),
      // All measured in this browser: the limit stages the server sent while
      // the answer was still streaming, and the wall clock from the send. The
      // last is the number a stopwatch beside the screen would read.
      stoppedAtDeadline: Boolean(measured && measured.stoppedAtDeadline),
      stoppedAtCeiling: Boolean(measured && measured.stoppedAtCeiling),
      elapsedSeconds: measured && Number.isFinite(measured.elapsedSeconds) ? measured.elapsedSeconds : null,
      batchId: batch ? batch.batch_id : null,
      at: (body.grounding && body.grounding.generated_at) || new Date().toISOString(),
    });
    if (batch) refreshRail();
  }, [locale, appendEntry, mergeBatches, refreshRail, conv, markAdopted]);

  const stopAsk = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
  }, []);

  const ask = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed || asking) return;
    setAsking(true);
    // Clear at send time, not on completion: the composer stays typeable while
    // the answer streams, so a late clear would wipe whatever has already been
    // typed for the next question.
    setQuestion('');
    const startedAt = Date.now();
    setLive({ question: trimmed, text: '', stage: null, steps: [], startedAt });
    const controller = new AbortController();
    abortRef.current = controller;
    const conversationId = conv.supported && conv.activeId ? conv.activeId : null;
    // Advisory grounding only, per the frozen contract: where the operator is
    // right now, plus the focused entity when a record is open. Absent context
    // sends nothing and the ask behaves exactly as before.
    const pageContext = buildPageContext(pageState);
    // Stage frames the finished exchange keeps: the scope Kai grounded on, and
    // whether the run stopped at one of its own limits, the time limit or the
    // turn budget. None is in the ask body, whose key set is the frozen
    // contract, so each is captured as it streams.
    const measured = { stoppedAtDeadline: false, stoppedAtCeiling: false, elapsedSeconds: null };
    try {
      let body;
      try {
        body = await streamAsk(trimmed, {
          conversationId,
          pageContext,
          signal: controller.signal,
          onStage: (stage) => {
            noteStageLimits(stage, measured);
            setLive((prev) => applyStage(prev, stage));
          },
          onStep: (step) => setLive((prev) => (prev ? { ...prev, steps: [...prev.steps, step] } : prev)),
          onDelta: (text) => setLive((prev) => (prev ? { ...prev, text: prev.text + text } : prev)),
        });
      } catch (streamError) {
        // A stop is the person's own decision, so it ends here rather than
        // silently starting the same ask again on the plain endpoint.
        if (controller.signal.aborted) throw streamError;
        // The stream endpoint is unavailable or broke mid-flight; the plain
        // ask returns the same answer without live updates, so retry there.
        setLive({ question: trimmed, text: '', stage: null, steps: [], startedAt });
        body = await postJson('/api/assistant/ask', {
          question: trimmed,
          ...(conversationId ? { conversation_id: conversationId } : {}),
          ...(pageContext ? { page_context: pageContext } : {}),
        }, { signal: controller.signal });
      }
      measured.elapsedSeconds = (Date.now() - startedAt) / 1000;
      finishAsk(trimmed, body, measured);
    } catch (error) {
      const stopped = controller.signal.aborted;
      appendEntry({
        question: trimmed,
        error: stopped
          ? pageText(locale, 'You stopped this question. Nothing was changed.', 'עצרתם את השאלה הזו. שום דבר לא שונה.')
          : error.message,
      });
      // Put the question back for an easy retry, but never overwrite text
      // typed while the ask was in flight.
      setQuestion((current) => (current ? current : trimmed));
    } finally {
      abortRef.current = null;
      setLive(null);
      setAsking(false);
    }
  }, [question, asking, finishAsk, appendEntry, locale, conv.supported, conv.activeId, pageState]);

  function onComposerKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      ask();
    }
  }

  function pickSuggestion(text) {
    setQuestion(text);
    if (composerRef.current) composerRef.current.focus();
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
  const reasonLabel = status && status.reason === 'API key not configured'
    ? pageText(locale, 'The API key is not configured on the server.', 'מפתח ה-API אינו מוגדר בשרת.')
    : String((status && status.reason) || '');
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
      <bdi dir="auto">{page.label}</bdi>
      {pageText(locale, ' page', '')}
      {entityLabel ? <>{', '}<bdi dir="auto">{entityLabel}</bdi></> : null}
    </>
  ) : null;

  // Conversation statistics from data already on screen: exchange count from
  // the loaded entries, the start date of the first entry, and applied changes
  // counted from this conversation's loaded batches. Nothing is invented, so
  // the applied figure only renders when at least one applied item is loaded.
  const startedAt = thread.length > 0 && thread[0].at ? startedLabel(thread[0].at, locale) : '';
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
    return <AssistantProposalCard key={batch.batch_id} batch={batch} locale={locale} busy={applyBusyId === batch.batch_id} applyResult={applyResults[batch.batch_id] || null} onApply={(ids) => applyItems(batch.batch_id, ids)} onReject={(ids) => rejectItems(batch.batch_id, ids)} onShowRestore={() => { window.location.hash = 'Versions'; }} notify={notify} onUndone={refreshRail} />;
  }

  // The action plane's own honest state, from the endpoint that owns it. It
  // says whether Kai may propose a change at all, and when it may not, why.
  const actionPlane = status && status.action_plane && typeof status.action_plane === 'object' ? status.action_plane : null;

  const statusCluster = (
    <div className="asst-status" role="status">
      <span className={`asst-dot ${dotClass}`} aria-hidden="true" />
      <span>{statusText}</span>
      {status && status.model ? <code dir="ltr">{status.model}</code> : null}
      {actionPlane && actionPlane.enabled === true ? <span className="asst-chip on">{pageText(locale, 'Can propose changes', 'יכול להציע שינויים')}</span> : null}
      {actionPlane && actionPlane.enabled === false ? <span className="asst-chip" title={actionPlane.reason || ''}>{pageText(locale, 'Answers only', 'מענה בלבד')}</span> : null}
    </div>
  );

  return (
    <section className={dock ? 'asst-workspace asst-in-dock' : 'page-workspace asst-workspace'}>
      {dock ? (
        <div className="asst-dock-meta">
          {statusCluster}
          {locationText ? <span className="asst-location" dir="auto">{locationText}</span> : null}
          {!threadLoading && thread.length > 0 ? (
            <p className="asst-stats">
              <span>{thread.length === 1 ? pageText(locale, 'one question in this conversation', 'שאלה אחת בשיחה') : pageText(locale, `${thread.length} questions in this conversation`, `${thread.length} שאלות בשיחה`)}</span>
              {startedAt ? <span dir="auto">{pageText(locale, 'started ', 'התחילה ב-')}<bdi dir="ltr">{startedAt}</bdi></span> : null}
              {appliedCount > 0 ? <span>{appliedCount === 1 ? pageText(locale, 'one change applied', 'הוחל שינוי אחד') : pageText(locale, `${appliedCount} changes applied`, `הוחלו ${appliedCount} שינויים`)}</span> : null}
            </p>
          ) : null}
        </div>
      ) : (
        <div className="page-header">
          <div>
            <h1>{pageText(locale, 'Kai, the AI assistant', 'קאי, עוזר ה-AI')}</h1>
            <p>{pageText(locale, 'Kai answers from the saved data only, and any action it proposes applies only after your approval, with an automatic restore point first.', 'קאי עונה מהנתונים השמורים בלבד, וכל פעולה שהוא מציע מוחלת רק לאחר אישור שלכם, עם נקודת שחזור אוטומטית לפני כן.')}</p>
          </div>
          {statusCluster}
        </div>
      )}

      <div className="asst-layout">
        <section className="page-panel asst-chat">
          <div className="panel-head">
            <div>
              <h2>{pageText(locale, 'Conversation', 'שיחה')}</h2>
              <span>{pageText(locale, 'Saved to your account and shown here when you return.', 'נשמרת לחשבון שלכם ומוצגת כאן בכל חזרה.')}</span>
              {actingUser ? <span className="asst-user" dir="auto">{pageText(locale, 'Acting user', 'מבצע')}: <b dir="ltr">{actingUser}</b></span> : null}
            </div>
            {thread.length > 0 && !threadLoading ? (
              confirmClear ? (
                <span className="asst-clear-confirm">
                  <span>{pageText(locale, 'Delete the whole conversation?', 'למחוק את כל השיחה?')}</span>
                  <Button variant="contained" size="small" color="error" disabled={clearing} onClick={clearConversation}>
                    {clearing ? pageText(locale, 'Deleting', 'מוחק') : pageText(locale, 'Delete', 'מחק')}
                  </Button>
                  <Button variant="text" size="small" disabled={clearing} onClick={() => setConfirmClear(false)}>
                    {pageText(locale, 'Cancel', 'ביטול')}
                  </Button>
                </span>
              ) : (
                <button type="button" className="asst-clear-btn" onClick={() => setConfirmClear(true)}>
                  <Trash2 size={13} />
                  {pageText(locale, 'Clear', 'מחיקה')}
                </button>
              )
            ) : null}
          </div>

          <div className="asst-thread" ref={threadRef}>
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
                {live.text ? <ModelText className="asst-a" text={live.text} /> : null}
              </article>
            ) : null}
          </div>

          {unavailable ? (
            <div className="asst-unavailable" role="status">
              <strong>{pageText(locale, 'Kai is not available.', 'קאי אינו זמין.')}</strong>
              {reasonLabel ? <span>{reasonLabel}</span> : null}
              <span>{pageText(locale, 'To enable it, set the ANTHROPIC_API_KEY or KAIROS_ASSISTANT_API_KEY environment variable and restart the server.', 'להפעלה, הגדירו את משתנה הסביבה ANTHROPIC_API_KEY או KAIROS_ASSISTANT_API_KEY והפעילו מחדש את השרת.')}</span>
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
            onKeyDown={onComposerKeyDown}
            unavailable={unavailable}
            asking={asking}
            onSend={ask}
            onStop={stopAsk}
          />
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
          onShowRestore={() => { window.location.hash = 'Versions'; }}
        />
      </div>
    </section>
  );
}
