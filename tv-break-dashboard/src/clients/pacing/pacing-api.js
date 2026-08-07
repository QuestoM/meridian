// Clients, pacing: the four reads and three writes this destination makes.
//
// A failed read is a failure and never an empty result, so every function here
// throws with the server's own bilingual refusal attached rather than returning a
// shape the caller could mistake for data.

import { API_BASE } from '../../shell/surface-helpers';

export class PacingError extends Error {
  constructor(status, detail) {
    super(`pacing request failed with ${status}`);
    this.status = status;
    this.detail = detail || null;
  }
}

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;
  }
  if (!response.ok) {
    throw new PacingError(response.status, body && body.detail ? body.detail : null);
  }
  return body;
}

export function loadBoard() {
  return request('/api/pacing');
}

export function loadLedger() {
  return request('/api/make-goods');
}

// The broadcast days behind one row, read when a reader opens them rather than
// on every board load. They were 144 KB of a 366 KB payload and they grow as
// campaigns times flight days.
export function loadDays(campaignId) {
  return request(`/api/pacing/${encodeURIComponent(campaignId)}/days`);
}

// The second ending. Taking a risk on records a decision and changes no figure,
// so the request carries a note and nothing that is a number.
export function acceptRisk(campaignId, note) {
  return request(`/api/pacing/${encodeURIComponent(campaignId)}/accept`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note: note || '' }),
  });
}

export function raiseMakeGood(campaignId, note) {
  return request('/api/make-goods', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ campaign_id: campaignId, note: note || '' }),
  });
}

export function moveMakeGood(makeGoodId, payload) {
  return request(`/api/make-goods/${encodeURIComponent(makeGoodId)}/state`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

// The refusal a control shows, in the reader's own language, falling back to the
// other one rather than to a generic sentence that says nothing.
export function refusalText(error, locale) {
  const detail = error && error.detail ? error.detail : null;
  if (!detail) return '';
  const key = locale === 'he' ? 'message_he' : 'message_en';
  return String(detail[key] || detail.message_en || detail.message_he || '');
}

export function refusalOpens(error) {
  const detail = error && error.detail ? error.detail : null;
  return detail && detail.opens ? detail.opens : null;
}
