import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { Bot, RefreshCcw, Send, Sparkles } from 'lucide-react';
import { pageText } from './surface-helpers';
import { postJson, requestJson, streamAsk } from './assistant-stream';
import AssistantProposalCard from './AssistantProposalCard';
import AssistantHistory from './AssistantHistory';
import AssistantUpload from './AssistantUpload';
import AssistantThread, { AssistantExchange, StreamProgress } from './AssistantThread';
import './assistant-console.css';

// The assistant console: a chat column grounded in the saved data plus a side
// rail for pending proposals, the audit history and the versions timeline. Answers
// come only from the server; asks stream live (step names and answer text as
// they are produced) and fall back quietly to the plain ask endpoint when the
// stream is unavailable. Proposed actions apply only after an explicit
// operator confirm, with an automatic restore point first. Every surface has
// honest loading, error and empty states, and nothing is polled in a loop:
// rail data refreshes on mount, after actions, and via the manual refresh.

const HISTORY_CAP = 20;

const SUGGESTIONS = [
  ['What is the weekly net and why', 'מה הנטו השבועי ולמה'],
  ['Suggest a way to raise the net without hurting retention', 'הצע דרך להעלות את הנטו בלי לפגוע בשימור'],
  ['Create a constraint that blocks a break in the first 15 minutes of the evening news', 'צור אילוץ שאין הפסקה ב-15 הדקות הראשונות של מהדורת הערב'],
  ['Raise the revenue weight to 65 and recompute', 'העלה את משקל ההכנסות ל-65 והרץ חישוב מחדש'],
  ['Get me to a higher net without dropping retention below 0.75', 'הבא אותי לנטו גבוה יותר בלי לרדת מתחת ל-0.75 שימור'],
  ['Suggest settings that raise the weekly net, and show me the effect before I approve', 'הצע הגדרות שמגדילות את הנטו השבועי, ותראה לי את ההשפעה לפני שאאשר'],
];

function asArray(value, ...keys) {
  if (Array.isArray(value)) return value;
  for (const key of keys) {
    if (value && Array.isArray(value[key])) return value[key];
  }
  return [];
}

// Normalizes a batch from either the ask response or GET /proposals into one
// shape. Items without an id stay visible but cannot be selected or applied.
function normalizeBatch(raw) {
  if (!raw || typeof raw !== 'object' || !raw.batch_id) return null;
  const items = asArray(raw.items).map((item, index) => ({
    id: item && item.id != null ? String(item.id) : null,
    key: item && item.id != null ? String(item.id) : `row-${index}`,
    kind: item && item.kind ? String(item.kind) : '',
    summary: item && item.summary ? String(item.summary) : '',
    payload: item && item.payload && typeof item.payload === 'object' ? item.payload : null,
    reason: item && item.reason ? String(item.reason) : '',
    status: item && item.status ? String(item.status) : 'pending',
    error: item && item.error ? String(item.error) : '',
    effect: item && item.effect && typeof item.effect === 'object' ? item.effect : null,
    diff: item && Array.isArray(item.diff) ? item.diff : null,
  }));
  return { batch_id: String(raw.batch_id), created_at: raw.created_at || null, items };
}

export default function AssistantPanel({ locale, notify }) {
  const he = locale === 'he';
  const [status, setStatus] = useState(null);
  const [statusState, setStatusState] = useState('loading');
  const [question, setQuestion] = useState('');
  const [thread, setThread] = useState([]);
  const [asking, setAsking] = useState(false);
  const [railTab, setRailTab] = useState('proposals');
  const [batchMap, setBatchMap] = useState({});
  const [batchOrder, setBatchOrder] = useState([]);
  const [proposalsState, setProposalsState] = useState('loading');
  const [proposalsError, setProposalsError] = useState('');
  const [audit, setAudit] = useState({ state: 'loading', entries: [], error: '' });
  const [applyBusyId, setApplyBusyId] = useState(null);
  const [applyResults, setApplyResults] = useState({});
  const [refreshing, setRefreshing] = useState(false);
  const [live, setLive] = useState(null);
  const idRef = useRef(0);
  const threadRef = useRef(null);
  const composerRef = useRef(null);

  const mergeBatches = useCallback((incoming, fromServer) => {
    const clean = incoming.filter(Boolean);
    if (!clean.length) return;
    setBatchMap((prev) => {
      const next = { ...prev };
      for (const batch of clean) {
        const existing = prev[batch.batch_id];
        if (!existing) {
          next[batch.batch_id] = batch;
        } else {
          const errorByKey = new Map(existing.items.map((item) => [item.key, item.error]));
          const effectByKey = new Map(existing.items.map((item) => [item.key, item.effect]));
          const diffByKey = new Map(existing.items.map((item) => [item.key, item.diff]));
          next[batch.batch_id] = { ...existing, ...batch, items: batch.items.map((item) => ({ ...item, error: item.error || errorByKey.get(item.key) || '', effect: item.effect || effectByKey.get(item.key) || null, diff: item.diff || diffByKey.get(item.key) || null })) };
        }
      }
      return next;
    });
    setBatchOrder((prev) => {
      const ids = clean.map((batch) => batch.batch_id);
      if (fromServer) return [...ids, ...prev.filter((id) => !ids.includes(id))];
      return [...ids.filter((id) => !prev.includes(id)), ...prev];
    });
  }, []);

  const refreshRail = useCallback(async () => {
    setRefreshing(true);
    const [proposalsResult, auditResult] = await Promise.allSettled([
      requestJson('/api/assistant/proposals'),
      requestJson('/api/assistant/audit?limit=50'),
    ]);
    if (proposalsResult.status === 'fulfilled') {
      mergeBatches(asArray(proposalsResult.value, 'batches', 'proposals', 'items').map(normalizeBatch), true);
      setProposalsState('ready');
      setProposalsError('');
    } else {
      setProposalsState('error');
      setProposalsError(proposalsResult.reason && proposalsResult.reason.message ? proposalsResult.reason.message : 'unknown');
    }
    if (auditResult.status === 'fulfilled') {
      setAudit({ state: 'ready', entries: asArray(auditResult.value, 'entries', 'audit', 'items'), error: '' });
    } else {
      setAudit((prev) => ({ ...prev, state: 'error', error: auditResult.reason && auditResult.reason.message ? auditResult.reason.message : 'unknown' }));
    }
    setRefreshing(false);
  }, [mergeBatches]);

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
  }, [thread, asking, live]);

  const appendEntry = useCallback((entry) => {
    idRef.current += 1;
    const row = { id: `ask-${idRef.current}`, at: new Date().toISOString(), answer: null, error: null, disclosure: '', sources: [], toolTrace: [], truncated: false, batchId: null, ...entry };
    setThread((prev) => [...prev, row].slice(-HISTORY_CAP));
  }, []);

  const finishAsk = useCallback((trimmed, body) => {
    if (body.available === false) {
      setStatus((prev) => ({ ...(prev || {}), available: false, reason: body.reason || body.error }));
      appendEntry({ question: trimmed, error: String(body.reason || body.error || '') });
      return;
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
  }, [locale, appendEntry, mergeBatches, refreshRail]);

  const ask = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed || asking) return;
    setAsking(true);
    setLive({ question: trimmed, text: '', step: null });
    try {
      let body;
      try {
        body = await streamAsk(trimmed, {
          onStep: (step) => setLive((prev) => (prev ? { ...prev, step } : prev)),
          onDelta: (text) => setLive((prev) => (prev ? { ...prev, text: prev.text + text } : prev)),
        });
      } catch {
        // The stream endpoint is unavailable or broke mid-flight; the plain
        // ask returns the same answer without live updates, so retry there.
        setLive({ question: trimmed, text: '', step: null });
        body = await postJson('/api/assistant/ask', { question: trimmed });
      }
      finishAsk(trimmed, body);
      setQuestion('');
    } catch (error) {
      appendEntry({ question: trimmed, error: error.message });
    } finally {
      setLive(null);
      setAsking(false);
    }
  }, [question, asking, finishAsk, appendEntry]);

  const applyItems = useCallback(async (batchId, itemIds) => {
    if (!itemIds.length || applyBusyId) return;
    setApplyBusyId(batchId);
    try {
      const body = await postJson(`/api/assistant/proposals/${encodeURIComponent(batchId)}/apply`, { item_ids: itemIds });
      const results = asArray(body, 'results', 'items');
      const restoreId = body.restore_id || null;
      setApplyResults((prev) => ({ ...prev, [batchId]: { results, restoreId } }));
      setBatchMap((prev) => {
        const batch = prev[batchId];
        if (!batch) return prev;
        const byId = new Map(results.filter((row) => row && row.id != null).map((row) => [String(row.id), row]));
        return { ...prev, [batchId]: { ...batch, items: batch.items.map((item) => (item.id && byId.has(item.id) ? { ...item, status: String(byId.get(item.id).status || item.status), error: byId.get(item.id).error ? String(byId.get(item.id).error) : item.error } : item)) } };
      });
      const applied = results.filter((row) => row && row.status === 'applied').length;
      const failed = results.filter((row) => row && row.status === 'failed').length;
      if (failed) notify(`Applied ${applied} of ${itemIds.length} actions, ${failed} failed. Details are on the action card.`, `הוחלו ${applied} מתוך ${itemIds.length} פעולות, ${failed === 1 ? 'אחת נכשלה' : `${failed} נכשלו`}. הפרטים מופיעים בכרטיס הפעולות.`);
      else if (applied === 1) notify('Applied one action and created a restore point.', 'הוחלה פעולה אחת ונוצרה נקודת שחזור.');
      else notify(`Applied ${applied} actions and created a restore point.`, `הוחלו ${applied} פעולות ונוצרה נקודת שחזור.`);
      refreshRail();
    } catch (error) {
      notify(`Applying the actions failed (${error.message}).`, `החלת הפעולות נכשלה (${error.message}).`);
    } finally {
      setApplyBusyId(null);
    }
  }, [applyBusyId, notify, refreshRail]);

  const rejectItems = useCallback(async (batchId, itemIds) => {
    if (!itemIds.length || applyBusyId) return;
    setApplyBusyId(batchId);
    try {
      await postJson(`/api/assistant/proposals/${encodeURIComponent(batchId)}/reject`, { item_ids: itemIds });
      setBatchMap((prev) => {
        const batch = prev[batchId];
        if (!batch) return prev;
        return { ...prev, [batchId]: { ...batch, items: batch.items.map((item) => (item.id && itemIds.includes(item.id) ? { ...item, status: 'rejected' } : item)) } };
      });
      notify('The selected actions were rejected.', 'הפעולות שנבחרו נדחו.');
      refreshRail();
    } catch (error) {
      notify(`Rejecting the actions failed (${error.message}).`, `דחיית הפעולות נכשלה (${error.message}).`);
    } finally {
      setApplyBusyId(null);
    }
  }, [applyBusyId, notify, refreshRail]);

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

  const pendingCount = useMemo(() => Object.values(batchMap).reduce((sum, batch) => sum + batch.items.filter((item) => item.status === 'pending').length, 0), [batchMap]);
  const visibleBatches = useMemo(() => batchOrder.map((id) => batchMap[id]).filter((batch) => batch && (batch.items.some((item) => item.status === 'pending') || applyResults[batch.batch_id])), [batchOrder, batchMap, applyResults]);

  const unavailable = statusState === 'ready' && status && status.available === false;
  const reasonLabel = status && status.reason === 'API key not configured'
    ? pageText(locale, 'The API key is not configured on the server.', 'מפתח ה-API אינו מוגדר בשרת.')
    : String((status && status.reason) || '');
  const statusText = statusState === 'loading' ? pageText(locale, 'Checking availability', 'בודק זמינות')
    : statusState === 'error' ? pageText(locale, 'No connection to the Kairos server', 'אין חיבור לשרת Kairos')
    : unavailable ? pageText(locale, 'Not available', 'לא זמין')
    : pageText(locale, 'Connected', 'מחובר');
  const dotClass = statusState === 'loading' ? 'loading' : statusState === 'error' ? 'error' : unavailable ? 'off' : 'on';

  const TABS = [
    ['proposals', pageText(locale, 'Pending actions', 'פעולות ממתינות')],
    ['history', pageText(locale, 'History', 'היסטוריה')],
  ];

  function renderProposalCard(batch) {
    return <AssistantProposalCard key={batch.batch_id} batch={batch} locale={locale} busy={applyBusyId === batch.batch_id} applyResult={applyResults[batch.batch_id] || null} onApply={(ids) => applyItems(batch.batch_id, ids)} onReject={(ids) => rejectItems(batch.batch_id, ids)} onShowRestore={() => { window.location.hash = 'Versions'; }} />;
  }

  return (
    <section className="page-workspace asst-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'AI assistant', 'עוזר AI')}</h1>
          <p>{pageText(locale, 'The assistant answers from the saved data only, and any action it proposes applies only after your approval, with an automatic restore point first.', 'העוזר עונה מהנתונים השמורים בלבד, וכל פעולה שהוא מציע מוחלת רק לאחר אישור שלכם, עם נקודת שחזור אוטומטית לפני כן.')}</p>
        </div>
        <div className="asst-status" role="status">
          <span className={`asst-dot ${dotClass}`} aria-hidden="true" />
          <span>{statusText}</span>
          {status && status.model ? <code dir="ltr">{status.model}</code> : null}
          {status && status.actions_enabled === true ? <span className="asst-chip on">{pageText(locale, 'Actions enabled', 'פעולות מופעלות')}</span> : null}
          {status && status.actions_enabled === false ? <span className="asst-chip">{pageText(locale, 'Answers only', 'מענה בלבד')}</span> : null}
        </div>
      </div>

      <div className="asst-layout">
        <section className="page-panel asst-chat">
          <div className="panel-head">
            <h2>{pageText(locale, 'Conversation', 'שיחה')}</h2>
            <span>{pageText(locale, 'Previous conversations are saved to your account and appear above the thread', 'שיחות קודמות נשמרות לחשבון שלכם ומופיעות מעל השיחה')}</span>
          </div>

          <AssistantThread locale={locale} />

          <div className="asst-thread" ref={threadRef}>
            {thread.length === 0 && !asking ? (
              <div className="asst-thread-empty">
                <Bot size={18} />
                <p>{pageText(locale, 'No questions asked yet in this session.', 'עוד לא נשאלו שאלות בהפעלה הנוכחית.')}</p>
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
                <p className="asst-q" dir="auto">{live.question}</p>
                {live.text ? <div className="asst-a" dir="auto">{live.text}</div> : null}
              </article>
            ) : null}

            {asking ? (
              <div className="asst-thinking">
                <span className="asst-thinking-dots" aria-hidden="true"><span /><span /><span /></span>
                {pageText(locale, 'Working on an answer from the saved data', 'מכין תשובה מהנתונים השמורים')}
              </div>
            ) : null}
            {asking && live && live.step ? <StreamProgress locale={locale} step={live.step} /> : null}
          </div>

          {unavailable ? (
            <div className="asst-unavailable" role="status">
              <strong>{pageText(locale, 'The assistant is not available.', 'העוזר אינו זמין.')}</strong>
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
              rows={2}
              maxLength={2000}
              dir={he ? 'rtl' : 'ltr'}
              placeholder={unavailable ? pageText(locale, 'The assistant is not available right now', 'העוזר אינו זמין כרגע') : pageText(locale, 'Ask about the plan or request a change, in Hebrew or English', 'שאלו על התוכנית או בקשו שינוי, בעברית או באנגלית')}
              disabled={asking || unavailable}
              aria-label={pageText(locale, 'Question for the assistant', 'שאלה לעוזר')}
            />
            <Button variant="contained" size="small" onClick={ask} disabled={asking || unavailable || !question.trim()} startIcon={<Send size={14} />}>
              {pageText(locale, 'Ask', 'שאלו')}
            </Button>
          </div>
          <p className="asst-hint">{pageText(locale, 'Enter sends, Shift+Enter adds a line.', 'מקש Enter שולח, Shift+Enter יורד שורה.')}</p>
        </section>

        <aside className="page-panel asst-rail">
          <div className="asst-rail-tabs" role="tablist">
            {TABS.map(([key, label]) => (
              <button type="button" role="tab" aria-selected={railTab === key} className={`asst-tab${railTab === key ? ' active' : ''}`} key={key} onClick={() => setRailTab(key)}>
                {label}
                {key === 'proposals' && pendingCount > 0 ? <span className="asst-badge" dir="ltr">{pendingCount}</span> : null}
              </button>
            ))}
            <button type="button" className="asst-refresh" onClick={refreshRail} disabled={refreshing} aria-label={pageText(locale, 'Refresh', 'רענון')}>
              <RefreshCcw size={13} className={refreshing ? 'asst-spin' : ''} />
            </button>
          </div>
          <div className="asst-rail-body">
            {railTab === 'proposals' ? (
              proposalsState === 'loading' ? (
                <div className="asst-loading">{pageText(locale, 'Loading pending actions', 'טוען פעולות ממתינות')}</div>
              ) : proposalsState === 'error' ? (
                <div className="asst-error-note">{pageText(locale, `Pending actions could not be loaded (${proposalsError}).`, `לא ניתן לטעון את הפעולות הממתינות (${proposalsError}).`)}</div>
              ) : visibleBatches.length === 0 ? (
                <div className="asst-empty">{pageText(locale, 'No pending actions. When you ask the assistant for a change, its proposals appear here for approval.', 'אין פעולות ממתינות. כשתבקשו מהעוזר שינוי, ההצעות שלו יופיעו כאן לאישור.')}</div>
              ) : (
                visibleBatches.map((batch) => renderProposalCard(batch))
              )
            ) : (
              <AssistantHistory locale={locale} audit={audit} />
            )}
          </div>
        </aside>
      </div>
    </section>
  );
}
