import { API_BASE } from '../../shell/api';

// Plan, the week: the comparison arriving one broadcast day at a time.
//
// The comparison is fourteen real optimizations, two legs over the plan's own
// seven days, and measured on the reference data that is 11 to 13 seconds cold.
// Nobody should watch a spinner for that, and a spinner is also a lie about
// progress, so the server emits a finished day the moment both its legs are
// decided and this consumes them: the first comparable day is on screen in about
// one second and the week fills in beside it.
//
// The last frame carries the identical body the plain route returns, so the
// panel's finished state is the same object either way. Any transport failure
// throws and the caller falls back to the plain route, which is slower to first
// figure but returns the same week.

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

export async function streamCompare(body, { onWindow, onDay, signal } = {}) {
  const response = await fetch(`${API_BASE}/api/scenario-compare/stream`, {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
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

  const handleFrame = (raw) => {
    const frame = parseFrame(raw);
    if (!frame) return;
    let parsed;
    try {
      parsed = JSON.parse(frame.data);
    } catch {
      if (frame.event === 'final' || frame.event === 'error') throw new Error('the stream returned an unreadable terminal frame');
      return;
    }
    if (frame.event === 'window' && typeof onWindow === 'function') onWindow(parsed);
    else if (frame.event === 'day' && typeof onDay === 'function') onDay(parsed);
    else if (frame.event === 'final') terminal = parsed && typeof parsed === 'object' ? parsed : {};
    else if (frame.event === 'error') terminal = { available: false, reason: String((parsed && parsed.reason) || 'the comparison failed') };
  };

  // Robust across chunk boundaries: a trailing carriage return is held back
  // until the next chunk so a split CRLF never fabricates a frame boundary.
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
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      handleFrame(raw);
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
        break;
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // The stream is already closed; nothing to release.
    }
  }
  if (!terminal) throw new Error('the comparison stream ended before it finished');
  return terminal;
}

export default streamCompare;
