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

// The four strings a reason block carries, and the two day lists a goal line
// repeats from the flight it belongs to.
const PROSE = ['reason_en', 'reason_he', 'path_forward_en', 'path_forward_he'];
const DAY_LISTS = [['unsourced_remaining_days', 'forward'], ['unsourced_elapsed_days', 'pace']];
const BLANK = { reason_en: '', reason_he: '', path_forward_en: '', path_forward_he: '' };

function fill(block, source) {
  if (!block || PROSE.some((key) => key in block)) return;
  PROSE.forEach((key) => { block[key] = source[key] || ''; });
}

// The board sends each reason paragraph once and each row sends the key that
// selects it. Measured on the shipped data, the same prose was 87,976 bytes of
// 189,850 bytes of rows and the same list of unsourced dates was written three
// times per row. This puts the full shape back before any component sees it, so
// the wire got smaller and nothing that reads a row changed at all.
export function expandBoard(payload) {
  if (!payload || !payload.wire || !payload.wire.collapsed) return payload;
  const reasons = payload.reasons || {};
  const forwards = payload.forward_reasons || {};
  const rule = payload.reference_rule || {};
  (payload.rows || []).forEach((row) => {
    fill(row.headline, reasons[(row.headline || {}).code || ''] || BLANK);
    ['rating', 'money'].forEach((key) => {
      const line = row[key];
      if (!line) return;
      fill(line.pace, reasons[(line.pace || {}).code || ''] || BLANK);
      fill(line.forward, forwards[(line.forward || {}).state || ''] || BLANK);
      if (line.reference && !('rule_en' in line.reference)) {
        line.reference.rule_en = rule.rule_en || '';
        line.reference.rule_he = rule.rule_he || '';
      }
      DAY_LISTS.forEach(([name, holder]) => {
        const block = line[holder];
        if (block && name in block && block[name] === null) {
          block[name] = ((row.flight || {})[name] || []).slice();
        }
      });
    });
  });
  return payload;
}

export function loadBoard() {
  return request('/api/pacing').then(expandBoard);
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
  // Not every refusal that reaches this surface was worded by this piece. The
  // auth middleware answers with detail as a plain string, and reading only the
  // bilingual shape returned an empty string for it, so a refused write said
  // nothing at all. A sentence in one language beats no sentence.
  if (typeof detail === 'string') return detail;
  const key = locale === 'he' ? 'message_he' : 'message_en';
  return String(detail[key] || detail.message_en || detail.message_he || '');
}

export function refusalOpens(error) {
  const detail = error && error.detail ? error.detail : null;
  return detail && detail.opens ? detail.opens : null;
}
