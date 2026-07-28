import { API_BASE } from './surface-helpers';

// Transport for the assistant console. requestJson and postJson are the plain
// JSON helpers shared by the panel and the previous-conversations block.
// streamAsk consumes the server-sent-event stream of an ask (step frames as
// tools run, delta frames of answer text, one terminal frame) and resolves
// with the exact body the non-streaming ask endpoint would have returned.
// Transport or protocol failures throw so the caller can fall back to the
// plain ask endpoint; a server-sent error frame resolves as an error body,
// exactly like the non-streaming path, so the ask is never run twice.

export async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: options.credentials || 'include',
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && (body.detail || body.error);
    throw new Error(detail ? String(detail) : `HTTP ${response.status}`);
  }
  return body || {};
}

export function postJson(path, payload) {
  return requestJson(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

// One raw SSE frame (the text between blank lines) to { event, data }. Data
// lines are joined with a newline per the SSE spec; comment and id lines are
// ignored. Returns null for frames with no data (keep-alive comments).
function parseFrame(frame) {
  let event = 'message';
  const dataLines = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  if (!dataLines.length) return null;
  return { event, data: dataLines.join('\n') };
}

// Optional fields ride in the body per the frozen contract: conversation_id
// scopes the ask to the active conversation, and page_context is the advisory
// current-location grounding ({ view, label, entity } or absent). Both are
// omitted when null, so the request degrades to exactly today's behavior.
export async function streamAsk(question, { conversationId = null, pageContext = null, onStep, onDelta } = {}) {
  const payload = { question };
  if (conversationId) payload.conversation_id = conversationId;
  if (pageContext) payload.page_context = pageContext;
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
    const frame = parseFrame(rawFrame);
    if (!frame) return;
    let parsed;
    try {
      parsed = JSON.parse(frame.data);
    } catch {
      if (frame.event === 'final' || frame.event === 'error') throw new Error('the stream returned an unreadable terminal frame');
      return;
    }
    if (frame.event === 'step' && typeof onStep === 'function' && parsed && typeof parsed === 'object') onStep(parsed);
    else if (frame.event === 'delta' && typeof onDelta === 'function' && parsed && typeof parsed.text === 'string') onDelta(parsed.text);
    else if (frame.event === 'final') terminal = parsed && typeof parsed === 'object' ? parsed : {};
    else if (frame.event === 'error') terminal = { error: String((parsed && parsed.error) || 'stream error') };
  };

  // Robust across chunk boundaries: a trailing carriage return is held back
  // until the next chunk so a split CRLF never fabricates a frame boundary;
  // everything else is normalized to bare newlines before frame extraction.
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
