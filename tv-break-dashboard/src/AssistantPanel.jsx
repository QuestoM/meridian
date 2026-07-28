import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { Bot, Send, Sparkles, Trash2 } from 'lucide-react';
import { pageText } from './surface-helpers';
import { postJson, requestJson, streamAsk } from './assistant-stream';
import { useConversations } from './AssistantConversationsApi';
import { asArray, normalizeBatch, useAssistantBatches, useAssistantThread } from './assistant-panel-state';
import { buildPageContext, useAssistantPage } from './assistant-page-context';
import AssistantConversationsSidebar from './AssistantConversationsSidebar';
import AssistantProposalCard from './AssistantProposalCard';
import AssistantUpload from './AssistantUpload';
import { AssistantExchange, StreamProgress, RichText } from './AssistantThread';
import './assistant-console.css';

// The assistant console, named Kai: a chat column grounded in the saved data
// plus a rail for pending proposals and the conversation history. Answers
// come only from the server; asks stream live (step names and answer text as
// they are produced) and fall back quietly to the plain ask endpoint when the
// stream is unavailable. Proposed actions apply only after an explicit
// operator confirm, with an automatic restore point first. Every surface has
// honest loading, error and empty states, and nothing is polled in a loop:
// rail data refreshes on mount, after actions, and via the manual refresh.
// In dock mode (the persistent side panel) the page chrome is replaced by a
// compact status row, the current-location chip, and real conversation stats.

const SUGGESTIONS = [
  ['What is the weekly net and why', 'מה הנטו השבועי ולמה'],
  ['Suggest a way to raise the net without hurting retention', 'הצע דרך להעלות את הנטו בלי לפגוע בשימור'],
  ['Create a constraint that blocks a break in the first 15 minutes of the evening news', 'צור אילוץ שאין ברייק ב-15 הדקות הראשונות של מהדורת הערב'],
  ['Raise the revenue weight to 65 and recompute', 'העלה את משקל ההכנסות ל-65 והרץ חישוב מחדש'],
  ['Get me to a higher net without dropping retention below 0.75', 'הבא אותי לנטו גבוה יותר בלי לרדת מתחת ל-0.75 שימור'],
  ['Suggest settings that raise the weekly net, and show me the effect before I approve', 'הצע הגדרות שמגדילות את הנטו השבועי, ותראה לי את ההשפעה לפני שאאשר'],
];

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
    return () => {
      active = false;
    };
  }, [refreshRail]);

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
      if (notify) notify(`Clearing the conversation failed (${error.message}).`, `מחיקת השיחה נכשלה (${error.message}).`);
    } finally {
      setClearing(false);
    }
  }, [notify, conv, resetLocal]);

  const finishAsk = useCallback((trimmed, body) => {
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
      disclosure: typeof body.context_disclosure === 'string' ? body.context_disclosure : '',
      sources: asArray(body.grounding && body.grounding.sources),
      toolTrace: asArray(body.tool_trace),
      truncated: Boolean(body.truncated),
      batchId: batch ? batch.batch_id : null,
      at: (body.grounding && body.grounding.generated_at) || new Date().toISOString(),
    });
    if (batch) refreshRail();
  }, [locale, appendEntry, mergeBatches, refreshRail, conv, markAdopted]);

  const ask = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed || asking) return;
    setAsking(true);
    // Clear at send time, not on completion: the composer stays typeable while
    // the answer streams, so a late clear would wipe whatever the operator has
    // already started typing for the next question.
    setQuestion('');
    setLive({ question: trimmed, text: '', step: null });
    const conversationId = conv.supported && conv.activeId ? conv.activeId : null;
    // Advisory grounding only, per the frozen contract: where the operator is
    // right now, plus the focused entity when a record is open. Absent context
    // sends nothing and the ask behaves exactly as before.
    const pageContext = buildPageContext(pageState);
    try {
      let body;
      try {
        body = await streamAsk(trimmed, {
          conversationId,
          pageContext,
          onStep: (step) => setLive((prev) => (prev ? { ...prev, step } : prev)),
          onDelta: (text) => setLive((prev) => (prev ? { ...prev, text: prev.text + text } : prev)),
        });
      } catch {
        // The stream endpoint is unavailable or broke mid-flight; the plain
        // ask returns the same answer without live updates, so retry there.
        setLive({ question: trimmed, text: '', step: null });
        body = await postJson('/api/assistant/ask', {
          question: trimmed,
          ...(conversationId ? { conversation_id: conversationId } : {}),
          ...(pageContext ? { page_context: pageContext } : {}),
        });
      }
      finishAsk(trimmed, body);
    } catch (error) {
      appendEntry({ question: trimmed, error: error.message });
      // Put the failed question back for an easy retry, but never overwrite
      // text the operator typed while the ask was in flight.
      setQuestion((current) => (current ? current : trimmed));
    } finally {
      setLive(null);
      setAsking(false);
    }
  }, [question, asking, finishAsk, appendEntry, conv.supported, conv.activeId, pageState]);

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
  const locationText = page
    ? (entityLabel
      ? pageText(locale, `You are on the ${page.label} page, ${entityLabel}`, `אתם בעמוד ${page.label}, ${entityLabel}`)
      : pageText(locale, `You are on the ${page.label} page`, `אתם בעמוד ${page.label}`))
    : null;

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
    return <AssistantProposalCard key={batch.batch_id} batch={batch} locale={locale} busy={applyBusyId === batch.batch_id} applyResult={applyResults[batch.batch_id] || null} onApply={(ids) => applyItems(batch.batch_id, ids)} onReject={(ids) => rejectItems(batch.batch_id, ids)} onShowRestore={() => { window.location.hash = 'Versions'; }} />;
  }

  const statusCluster = (
    <div className="asst-status" role="status">
      <span className={`asst-dot ${dotClass}`} aria-hidden="true" />
      <span>{statusText}</span>
      {status && status.model ? <code dir="ltr">{status.model}</code> : null}
      {status && status.actions_enabled === true ? <span className="asst-chip on">{pageText(locale, 'Actions enabled', 'פעולות מופעלות')}</span> : null}
      {status && status.actions_enabled === false ? <span className="asst-chip">{pageText(locale, 'Answers only', 'מענה בלבד')}</span> : null}
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
              {startedAt ? <span>{pageText(locale, `started ${startedAt}`, `התחילה ב-${startedAt}`)}</span> : null}
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
              <div className="asst-thread-empty">
                <Bot size={18} />
                <p>{pageText(locale, 'No questions asked yet. Kai answers from the saved data only, and the conversation is saved and will appear here next time.', 'עוד לא נשאלו שאלות. קאי עונה מהנתונים השמורים בלבד, והשיחה נשמרת ותופיע כאן בפעם הבאה.')}</p>
                {!unavailable && statusState !== 'loading' ? (
                  <div className="asst-suggestions">
                    <span className="asst-suggestions-label"><Sparkles size={12} />{pageText(locale, 'You can start with one of these', 'אפשר להתחיל מאחת מאלה')}</span>
                    {SUGGESTIONS.map((pair) => (
                      <button type="button" className="asst-suggestion" key={pair[1]} onClick={() => pickSuggestion(pageText(locale, pair[0], pair[1]))}>
                        {pageText(locale, pair[0], pair[1])}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}

            {thread.map((entry) => (
              <AssistantExchange key={entry.id} entry={entry} locale={locale} proposalCard={entry.batchId && batchMap[entry.batchId] ? renderProposalCard(batchMap[entry.batchId]) : null} />
            ))}

            {live ? (
              <article className="asst-exchange">
                <RichText className="asst-q" text={live.question} />
                {live.text ? <RichText className="asst-a" text={live.text} /> : null}
              </article>
            ) : null}

            {asking ? (
              <div className="asst-thinking">
                <span className="asst-thinking-dots" aria-hidden="true"><span /><span /><span /></span>
                {pageText(locale, 'Kai is preparing an answer from the saved data', 'קאי מכין תשובה מהנתונים השמורים')}
              </div>
            ) : null}
            {asking && live && live.step ? <StreamProgress locale={locale} step={live.step} /> : null}
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

          <div className="asst-composer">
            <textarea
              ref={composerRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={onComposerKeyDown}
              rows={1}
              maxLength={2000}
              dir={question ? 'auto' : (locale === 'he' ? 'rtl' : 'ltr')}
              placeholder={unavailable ? pageText(locale, 'Kai is not available right now', 'קאי אינו זמין כרגע') : pageText(locale, 'Ask about the plan or request a change, in Hebrew or English', 'שאלו על התוכנית או בקשו שינוי, בעברית או באנגלית')}
              disabled={unavailable}
              aria-label={pageText(locale, 'Question for Kai', 'שאלה לקאי')}
            />
            <Button variant="contained" size="small" className="asst-send-btn" onClick={ask} disabled={asking || unavailable || !question.trim()} endIcon={<Send size={14} style={locale === 'he' ? { transform: 'scaleX(-1)' } : undefined} />}>
              {asking ? pageText(locale, 'Sending', 'שולח') : pageText(locale, 'Send', 'שליחה')}
            </Button>
          </div>
          <p className="asst-hint">{pageText(locale, 'Enter sends, Shift+Enter adds a line.', 'מקש Enter שולח, Shift+Enter יורד שורה.')}</p>
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
