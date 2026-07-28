import { useCallback, useEffect, useState } from 'react';
import { API_BASE } from './surface-helpers';

// Transport and shared state for the assistant conversations feature: the
// conversation index CRUD, the per-conversation changes and restore calls, a
// conversation-aware streaming ask, and the useConversations hook the panel
// wires in. Every request carries credentials like the rest of the console.
// A 404 on the index marks the backend as not supporting conversations yet,
// and the UI hides the whole section instead of pretending it works.

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, { credentials: 'include', ...options });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && (body.detail || body.error);
    const error = new Error(detail ? String(detail) : `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body || {};
}

function jsonOptions(method, payload) {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) };
}

export function fetchConversationChanges(conversationId) {
  return apiRequest(`/api/assistant/conversations/${encodeURIComponent(conversationId)}/changes`);
}

export function restoreConversation(conversationId) {
  return apiRequest(`/api/assistant/conversations/${encodeURIComponent(conversationId)}/restore`, { method: 'POST' });
}

// The same SSE contract as streamAsk in assistant-stream.js, plus the active
// conversation_id in the request body so the ask lands in the conversation the
// operator is looking at. Kept separate because the shared helper posts a
// fixed body; the parsing rules (CRLF-safe frame splitting, terminal frame
// required) are identical, and a null conversationId omits the field so the
// call also works against a backend without conversations.
export async function streamAskConversation(question, conversationId, { onStep, onDelta } = {}) {
  const payload = conversationId ? { question, conversation_id: conversationId } : { question };
  const response = await fetch(`${API_BASE}/api/assistant/ask/stream`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
  });
  const contentType = response.headers.get('content-type') || '';
  if (!response.ok || !response.body || !contentType.includes('text/event-stream')) {
    throw new Error(`streaming not available (HTTP ${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let held = '';
  let terminal = null;

  const handleFrame = (rawFrame) => {
    let event = 'message';
    const dataLines = [];
    for (const line of rawFrame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
    }
    if (!dataLines.length) return;
    let parsed;
    try {
      parsed = JSON.parse(dataLines.join('\n'));
    } catch {
      if (event === 'final' || event === 'error') throw new Error('the stream returned an unreadable terminal frame');
      return;
    }
    if (event === 'step' && typeof onStep === 'function' && parsed && typeof parsed === 'object') onStep(parsed);
    else if (event === 'delta' && typeof onDelta === 'function' && parsed && typeof parsed.text === 'string') onDelta(parsed.text);
    else if (event === 'final') terminal = parsed && typeof parsed === 'object' ? parsed : {};
    else if (event === 'error') terminal = { error: String((parsed && parsed.error) || 'stream error') };
  };

  const drain = (text) => {
    let chunk = held + text;
    held = '';
    if (chunk.endsWith('\r')) {
      held = '\r';
      chunk = chunk.slice(0, -1);
    }
    buffer += chunk.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    let sep = buffer.indexOf('\n\n');
    while (sep !== -1 && !terminal) {
      const rawFrame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      handleFrame(rawFrame);
      sep = buffer.indexOf('\n\n');
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (value) drain(decoder.decode(value, { stream: true }));
      if (terminal) break;
      if (done) {
        drain(decoder.decode());
        const tail = buffer.trim();
        if (tail && !terminal) handleFrame(tail);
        break;
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }

  if (!terminal) throw new Error('the stream ended before a final frame');
  return terminal;
}

// Conversation-index state for the assistant panel. supported is null while
// unknown, true once the index loads, and false when the backend answers 404
// (an older server without conversations); the UI degrades by hiding the
// section. select() bumps loadNonce so re-picking the visible conversation
// still reloads it; adopt() changes the id quietly for ids the server minted
// during an ask, where the thread on screen is already current.
export function useConversations(notify) {
  const [supported, setSupported] = useState(null);
  const [listState, setListState] = useState('loading');
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [loadNonce, setLoadNonce] = useState(0);
  const [busy, setBusy] = useState(false);

  const refreshList = useCallback(async () => {
    try {
      const body = await apiRequest('/api/assistant/conversations');
      const rows = Array.isArray(body.conversations) ? body.conversations.filter((row) => row && row.id != null) : [];
      setConversations(rows);
      setSupported(true);
      setListState('ready');
      return rows;
    } catch (error) {
      if (error && error.status === 404) {
        setSupported(false);
        setListState('unsupported');
      } else {
        setListState('error');
      }
      return null;
    }
  }, []);

  useEffect(() => { refreshList(); }, [refreshList]);

  const select = useCallback((id) => {
    setActiveId(id);
    setLoadNonce((nonce) => nonce + 1);
  }, []);

  const adopt = useCallback((id) => { setActiveId(id); }, []);

  const create = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    try {
      const body = await apiRequest('/api/assistant/conversations', jsonOptions('POST', {}));
      if (body && body.id != null) {
        setActiveId(String(body.id));
        setLoadNonce((nonce) => nonce + 1);
      }
      await refreshList();
    } catch (error) {
      if (notify) notify(`Creating a conversation failed (${error.message}).`, `יצירת שיחה נכשלה (${error.message}).`);
    } finally {
      setBusy(false);
    }
  }, [busy, notify, refreshList]);

  const rename = useCallback(async (id, title) => {
    if (busy) return;
    setBusy(true);
    try {
      await apiRequest(`/api/assistant/conversations/${encodeURIComponent(id)}`, jsonOptions('PATCH', { title }));
      await refreshList();
    } catch (error) {
      if (notify) notify(`Renaming the conversation failed (${error.message}).`, `שינוי שם השיחה נכשל (${error.message}).`);
    } finally {
      setBusy(false);
    }
  }, [busy, notify, refreshList]);

  const remove = useCallback(async (id) => {
    if (busy) return;
    setBusy(true);
    try {
      const body = await apiRequest(`/api/assistant/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
      const removed = Number(body && body.entries_removed) || 0;
      if (notify) notify(removed === 1 ? 'The conversation was deleted along with its one saved question.' : `The conversation was deleted along with its ${removed} saved questions.`, removed === 1 ? 'השיחה נמחקה יחד עם השאלה האחת שנשמרה בה.' : `השיחה נמחקה יחד עם ${removed} השאלות שנשמרו בה.`);
      const rows = await refreshList();
      if (activeId === id) {
        setActiveId(rows && rows.length ? String(rows[0].id) : null);
        setLoadNonce((nonce) => nonce + 1);
      }
    } catch (error) {
      if (notify) notify(`Deleting the conversation failed (${error.message}).`, `מחיקת השיחה נכשלה (${error.message}).`);
    } finally {
      setBusy(false);
    }
  }, [busy, notify, refreshList, activeId]);

  return { supported, listState, conversations, activeId, loadNonce, busy, refreshList, select, adopt, create, rename, remove };
}
