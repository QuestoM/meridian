import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { postJson, requestJson } from './assistant-stream';
import { unrecordedProposalClaim } from './kai-claimed-action';
import { isolate } from './kai-bidi';
import { writeAddress } from '../history/history-labels.js';
import { pointAddress } from '../history/history-address.js';

// State hooks for the assistant console, split out of AssistantPanel.jsx so
// the panel stays a readable render component. useAssistantBatches owns the
// proposal-batch lifecycle (rail refresh, apply, reject); useAssistantThread
// owns the saved-conversation thread (load, append, legacy clear). Behavior is
// byte-identical to the logic that previously lived inline in the panel.

export function asArray(value, ...keys) {
  if (Array.isArray(value)) return value;
  for (const key of keys) {
    if (value && Array.isArray(value[key])) return value[key];
  }
  return [];
}

// A tool call written into the text channel, which the server now cuts before
// an answer is ever returned or stored. Conversations saved before it did still
// carry one, and a stored line is replayed on screen exactly as stored, so the
// same cut is applied on load: the reader sees the prose that was written and
// never a call, which reaches nothing and means nothing to them. The rule is
// the one in kairos_api/assistant_protocol_text.py, kept identical on purpose.
const TOOL_PROTOCOL = /<\s*\/?\s*(?:[A-Za-z][\w.-]*:)?(?:invoke|function_calls|parameter|antml)\b/i;
const PROTOCOL_LEAD_IN = /(?:^|\n)[ \t]*<?[A-Za-z_:]*calls?[ \t]*$/;

export function storedAnswer(text) {
  const match = TOOL_PROTOCOL.exec(text);
  if (!match) return text;
  const kept = text.slice(0, match.index).replace(/\s+$/, '').replace(PROTOCOL_LEAD_IN, '').trim();
  return kept || null;
}

// Normalizes a batch from either the ask response or GET /proposals into one
// shape. Items without an id stay visible but cannot be selected or applied.
// restore_points ride along so the undo control survives a reload: the server
// stores every restore point an apply created ON the batch, which is what makes
// the reversal an object you can come back to rather than one tab's memory.
export function normalizeBatch(raw) {
  if (!raw || typeof raw !== 'object' || !raw.batch_id) return null;
  const items = asArray(raw.items).map((item, index) => ({
    id: item && item.id != null ? String(item.id) : null,
    key: item && item.id != null ? String(item.id) : `row-${index}`,
    kind: item && item.kind ? String(item.kind) : '',
    summary: item && item.summary ? String(item.summary) : '',
    // The terms the summary was built from travel with it. A whitelist that
    // keeps the English record and drops them prints English prose on a Hebrew
    // screen, which is the one thing this surface may not do.
    summary_terms: item && item.summary_terms && typeof item.summary_terms === 'object' ? item.summary_terms : null,
    payload: item && item.payload && typeof item.payload === 'object' ? item.payload : null,
    reason: item && item.reason ? String(item.reason) : '',
    status: item && item.status ? String(item.status) : 'pending',
    error: item && item.error ? String(item.error) : '',
    effect: item && item.effect && typeof item.effect === 'object' ? item.effect : null,
    // The basis travels with the money it qualifies. A whitelist that keeps the
    // figures and drops their channel and day prints a number with no basis,
    // which is the one thing this surface may not do.
    effect_basis: item && item.effect_basis && typeof item.effect_basis === 'object' ? item.effect_basis : null,
    diff: item && Array.isArray(item.diff) ? item.diff : null,
    permission: item && item.permission && typeof item.permission === 'object' ? item.permission : null,
  }));
  const restorePoints = asArray(raw.restore_points)
    .filter((point) => point && point.restore_id)
    .map((point) => ({
      restoreId: String(point.restore_id),
      appliedAt: point.applied_at ? String(point.applied_at) : null,
      appliedBy: point.applied_by ? String(point.applied_by) : '',
      // The restore id and the History page's id are two different things: the
      // restore id is what /api/assistant/restore applies against, the version
      // id is the row the History page can open and select. Both ride on the
      // point so the "see it in the history" control has something to address.
      versionId: point.version_id ? String(point.version_id) : null,
    }));
  return { batch_id: String(raw.batch_id), created_at: raw.created_at || null, items, restorePoints };
}

// Opens the History page addressed at one row rather than at its own newest
// row: the version id, when there is one, is written as ?entry=version:<id>
// before the hash changes, so the page the hash lands on reads it on mount.
// Without a version id (a restore point older than the timeline column, or a
// caller with none to give) this falls back to the plain jump it always did.
//
// A hash assignment to the value it already holds fires no hashchange event.
// The dock's own control lands on this page (that is where an apply just
// happened), so the second and every later click leaves the hash unmoved,
// the shell never remounts the destination, and HistoryPage, which reads the
// address only in its mount-time initialisers, never sees the address just
// written above: the url names one entry while the screen keeps showing
// whichever row it already had selected. When that is the state, the hash is
// bounced off a value the router does not recognise first, one tick apart so
// the shell's own listener actually processes the transition away before the
// transition back, so the assignment back to 'Versions' is a real transition
// its listener remounts the destination for.
//
// The bounce cannot be two plain `window.location.hash = ...` assignments: a
// hash assignment is a navigation, and every navigation it performs pushes a
// browser-history entry, even when it lands back on the same view. Measured:
// four chip clicks made while already on the History page pushed eight
// entries between them, and one Back press from the correctly addressed
// History page then landed on Overview at the just-abandoned address, a url
// that names a History entry on a page that cannot open it. `location.replace`
// performs the same navigation, and fires the same hashchange, without adding
// a history entry, so the bounce is written with it instead. The base has to
// be passed explicitly: a bare `location.replace('#')` drops the ?entry
// search this function just wrote.
export function showRestoreVersion(versionId) {
  if (versionId) writeAddress(pointAddress(versionId));
  const onVersions = decodeURIComponent(window.location.hash.replace(/^#/, '')) === 'Versions';
  if (versionId && onVersions) {
    const base = window.location.pathname + window.location.search;
    window.location.replace(`${base}#`);
    window.setTimeout(() => { window.location.replace(`${base}#Versions`); }, 0);
    return;
  }
  window.location.hash = 'Versions';
}

export function useAssistantBatches(notify) {
  const [batchMap, setBatchMap] = useState({});
  const [batchOrder, setBatchOrder] = useState([]);
  const [proposalsState, setProposalsState] = useState('loading');
  const [proposalsError, setProposalsError] = useState('');
  const [applyBusyId, setApplyBusyId] = useState(null);
  const [applyResults, setApplyResults] = useState({});
  const [refreshing, setRefreshing] = useState(false);

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
          const basisByKey = new Map(existing.items.map((item) => [item.key, item.effect_basis]));
          const diffByKey = new Map(existing.items.map((item) => [item.key, item.diff]));
          const permissionByKey = new Map(existing.items.map((item) => [item.key, item.permission]));
          const restorePoints = batch.restorePoints && batch.restorePoints.length ? batch.restorePoints : existing.restorePoints;
          next[batch.batch_id] = { ...existing, ...batch, restorePoints, items: batch.items.map((item) => ({ ...item, error: item.error || errorByKey.get(item.key) || '', effect: item.effect || effectByKey.get(item.key) || null, effect_basis: item.effect_basis || basisByKey.get(item.key) || null, diff: item.diff || diffByKey.get(item.key) || null, permission: item.permission || permissionByKey.get(item.key) || null })) };
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
    try {
      const value = await requestJson('/api/assistant/proposals');
      mergeBatches(asArray(value, 'batches', 'proposals', 'items').map(normalizeBatch), true);
      setProposalsState('ready');
      setProposalsError('');
    } catch (err) {
      setProposalsState('error');
      setProposalsError(err && err.message ? err.message : 'unknown');
    }
    setRefreshing(false);
  }, [mergeBatches]);

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
        // The restore point lands on the batch immediately, so the undo control
        // is the same object before and after a reload rather than two. The
        // version id rides beside it from the same response, so "see it in the
        // history" addresses the right row before the next refresh even lands.
        const restorePoints = restoreId
          ? [...(batch.restorePoints || []), { restoreId: String(restoreId), appliedAt: new Date().toISOString(), appliedBy: '', versionId: body.version_id ? String(body.version_id) : null }]
          : batch.restorePoints;
        return { ...prev, [batchId]: { ...batch, restorePoints, items: batch.items.map((item) => (item.id && byId.has(item.id) ? { ...item, status: String(byId.get(item.id).status || item.status), error: byId.get(item.id).error ? String(byId.get(item.id).error) : item.error } : item)) } };
      });
      const appliedCount = results.filter((row) => row && row.status === 'applied').length;
      const failedCount = results.filter((row) => row && row.status === 'failed').length;
      if (failedCount === 1) notify(`Applied ${appliedCount} of ${itemIds.length} actions, one failed. Details are on the action card.`, `הוחלו ${appliedCount} מתוך ${itemIds.length} פעולות, אחת נכשלה. הפרטים מופיעים בכרטיס הפעולות.`);
      else if (failedCount) notify(`Applied ${appliedCount} of ${itemIds.length} actions, ${failedCount} failed. Details are on the action card.`, `הוחלו ${appliedCount} מתוך ${itemIds.length} פעולות, ${failedCount} נכשלו. הפרטים מופיעים בכרטיס הפעולות.`);
      else if (appliedCount === 1) notify('Applied one action and created a restore point.', 'הוחלה פעולה אחת ונוצרה נקודת שחזור.');
      else notify(`Applied ${appliedCount} actions and created a restore point.`, `הוחלו ${appliedCount} פעולות ונוצרה נקודת שחזור.`);
      refreshRail();
    } catch (error) {
      notify(`Applying the actions failed (${isolate(error.message)}).`, `החלת הפעולות נכשלה (${isolate(error.message)}).`);
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
      notify(`Rejecting the actions failed (${isolate(error.message)}).`, `דחיית הפעולות נכשלה (${isolate(error.message)}).`);
    } finally {
      setApplyBusyId(null);
    }
  }, [applyBusyId, notify, refreshRail]);

  const pendingCount = useMemo(() => Object.values(batchMap).reduce((sum, batch) => sum + batch.items.filter((item) => item.status === 'pending').length, 0), [batchMap]);
  // The rail shows what still needs a decision, plus anything that produced a
  // restore point: an applied change whose undo is gone from the screen is an
  // undo that only existed for one browser tab.
  const visibleBatches = useMemo(() => batchOrder.map((id) => batchMap[id]).filter((batch) => batch && (batch.items.some((item) => item.status === 'pending') || applyResults[batch.batch_id] || (batch.restorePoints || []).length)), [batchOrder, batchMap, applyResults]);

  return { batchMap, mergeBatches, refreshRail, refreshing, proposalsState, proposalsError, applyBusyId, applyResults, applyItems, rejectItems, pendingCount, visibleBatches };
}

export function useAssistantThread(conv) {
  const [thread, setThread] = useState([]);
  const [threadLoading, setThreadLoading] = useState(true);
  const [actingUser, setActingUser] = useState('');
  const idRef = useRef(0);
  // Conversation ids the server minted during an ask: the thread on screen is
  // already current, so the load effect skips exactly one refetch for them.
  const adoptedRef = useRef(null);

  // Load the saved conversation so returning to the assistant shows the past
  // exchanges instead of an empty chat. Each stored entry (question, answer,
  // time, batch id) becomes a thread row; keeping the stored batch_id is what
  // lets a proposal card reattach to its exchange after a reload. With no
  // active id the server returns the newest conversation and its id is
  // adopted quietly; picking a conversation in the rail reloads here.
  useEffect(() => {
    if (conv.activeId && adoptedRef.current === conv.activeId) {
      adoptedRef.current = null;
      return undefined;
    }
    let active = true;
    setThreadLoading(true);
    const path = conv.activeId ? `/api/assistant/thread?conversation_id=${encodeURIComponent(conv.activeId)}` : '/api/assistant/thread';
    requestJson(path)
      .then((body) => {
        if (!active) return;
        const entries = Array.isArray(body.entries) ? body.entries : [];
        const rows = entries.map((entry, index) => {
          const shown = entry && entry.answer ? storedAnswer(String(entry.answer)) : null;
          const batchId = entry && entry.batch_id ? String(entry.batch_id) : null;
          return {
            id: `saved-${index}`,
            at: entry && entry.at ? entry.at : null,
            question: entry && entry.question ? String(entry.question) : '',
            answer: shown,
            // A stored line that was nothing but a call leaves the exchange with
            // no answer at all, which the view says in words rather than showing
            // a question with silence under it.
            answerWithheld: Boolean(entry && entry.answer && !shown),
            // The saved half of the same rule. A batch id is stored on the entry
            // exactly when the ask created a batch (assistant.py:318-322), so an
            // entry without one recorded nothing, and a stored answer saying a
            // proposal is pending would otherwise print alone on every reload.
            unrecordedClaim: unrecordedProposalClaim({ answer: shown }, batchId),
            error: null,
            disclosure: '',
            sources: [],
            toolTrace: [],
            truncated: false,
            batchId,
          };
        });
        idRef.current = rows.length;
        setThread(rows);
        if (typeof body.user === 'string' && body.user) setActingUser(body.user);
        if (body.conversation_id && String(body.conversation_id) !== conv.activeId) {
          adoptedRef.current = String(body.conversation_id);
          conv.adopt(String(body.conversation_id));
        }
      })
      .catch(() => {
        // Honest empty chat if the thread cannot be read; the requested
        // conversation may have been deleted, so resync the list.
        if (!active) return;
        setThread([]);
        if (conv.activeId) conv.refreshList();
      })
      .finally(() => { if (active) setThreadLoading(false); });
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conv.activeId, conv.loadNonce]);

  // The on-screen thread keeps the newest 20 exchanges, matching the previous
  // inline HISTORY_CAP; the server stores the full conversation regardless.
  const appendEntry = useCallback((entry) => {
    idRef.current += 1;
    const row = { id: `ask-${idRef.current}`, at: new Date().toISOString(), answer: null, error: null, disclosure: '', sources: [], toolTrace: [], truncated: false, batchId: null, ...entry };
    setThread((prev) => [...prev, row].slice(-20));
  }, []);

  // Local wipe for the legacy whole-thread delete path (a backend without
  // conversations); the conversation-aware clear reloads through the hook.
  const resetLocal = useCallback(() => {
    setThread([]);
    idRef.current = 0;
  }, []);

  const markAdopted = useCallback((id) => { adoptedRef.current = id; }, []);

  return { thread, threadLoading, actingUser, appendEntry, resetLocal, markAdopted };
}
