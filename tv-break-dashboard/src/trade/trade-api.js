// The trade-agreement surface's own reads and writes, in one place.
//
// Same contract as clients-api.js and for the same reason: on a surface where a
// refusal is part of the work, a failed read and an empty result are different
// facts and the screen has to say which one happened. Every call either returns
// the payload or throws an Error carrying BOTH languages, because a Hebrew
// review flow that refuses in English has refused to somebody who cannot act on
// it.

import { API_BASE } from '../shell/api';

const TRADE = '/api/trade';

async function readJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw refusal(payload, response);
  }
  return response.json();
}

// The server's refusals arrive as FastAPI `detail`. The trade routes send a
// plain string (the store's own sentence, which is authored in Hebrew), and the
// older campaign routes send an {message_en, message_he} pair. Both are carried
// through unchanged; nothing here invents a translation for a sentence the
// server wrote in one language.
function refusal(payload, response) {
  const detail = payload ? payload.detail : null;
  let en = `${response.status} ${response.statusText}`;
  let he = en;
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    en = String(detail.message_en || detail.message || en);
    he = String(detail.message_he || en);
  } else if (detail) {
    en = String(detail);
    he = en;
  }
  const error = new Error(en);
  error.messageEn = en;
  error.messageHe = he;
  error.status = response.status;
  return error;
}

async function sendJson(path, method, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw refusal(payload, response);
  return payload;
}

export function refusalText(error, locale) {
  if (!error) return '';
  if (locale === 'he') return error.messageHe || error.messageEn || String(error.message || '');
  return error.messageEn || String(error.message || '');
}

// ------------------------------------------------------------------ agreements

export async function loadAgreements() {
  return readJson(`${TRADE}/agreements`);
}

export async function loadAgreement(agreementId) {
  return readJson(`${TRADE}/agreements/${encodeURIComponent(agreementId)}`);
}

export async function createAgreement(payload) {
  return sendJson(`${TRADE}/agreements`, 'POST', payload);
}

export async function changeStatus(agreementId, target, note = '') {
  return sendJson(
    `${TRADE}/agreements/${encodeURIComponent(agreementId)}/status`,
    'POST',
    { target, note },
  );
}

// ------------------------------------------------------------------- documents

// The upload is multipart, so it does not go through sendJson: a
// Content-Type set by hand would strip the boundary the server needs.
export async function uploadDocument(agreementId, file) {
  const body = new FormData();
  body.append('file', file, file.name);
  const response = await fetch(
    `${API_BASE}${TRADE}/agreements/${encodeURIComponent(agreementId)}/documents`,
    { method: 'POST', body },
  );
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw refusal(payload, response);
  return payload;
}

export function documentFileUrl(agreementId, documentId) {
  const agreement = encodeURIComponent(agreementId);
  const document = encodeURIComponent(documentId);
  return `${API_BASE}${TRADE}/agreements/${agreement}/documents/${document}/file`;
}

// The PDF is fetched as bytes rather than pointed at directly, because an
// <iframe> whose src is an API route inherits none of the app's credentials
// handling and a 401 renders as the browser's own error page inside the review.
// A blob URL either exists or the fetch failed, and a failed fetch is a state
// this surface can name.
export async function fetchDocumentBlob(agreementId, documentId) {
  const response = await fetch(documentFileUrl(agreementId, documentId));
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw refusal(payload, response);
  }
  return response.blob();
}

export async function startExtraction(agreementId, documentId) {
  const agreement = encodeURIComponent(agreementId);
  const document = encodeURIComponent(documentId);
  return sendJson(`${TRADE}/agreements/${agreement}/documents/${document}/extract`, 'POST');
}

export async function loadJob(jobId) {
  return readJson(`/api/jobs/${encodeURIComponent(jobId)}`);
}

export async function loadProposal(agreementId, documentId) {
  const agreement = encodeURIComponent(agreementId);
  const document = encodeURIComponent(documentId);
  return readJson(`${TRADE}/agreements/${agreement}/documents/${document}/proposal`);
}

// --------------------------------------------------------------- review actions

function reviewPath(agreementId, documentId, tail) {
  const agreement = encodeURIComponent(agreementId);
  const document = encodeURIComponent(documentId);
  return `${TRADE}/agreements/${agreement}/documents/${document}/${tail}`;
}

export async function markClausesSeen(agreementId, documentId, clauseIds) {
  return sendJson(reviewPath(agreementId, documentId, 'seen'), 'POST', { clause_ids: clauseIds });
}

export async function decideInstance(agreementId, documentId, instanceId, body) {
  const tail = `instances/${encodeURIComponent(instanceId)}/decide`;
  return sendJson(reviewPath(agreementId, documentId, tail), 'POST', body);
}

export async function promoteInstance(agreementId, documentId, instanceId) {
  const tail = `instances/${encodeURIComponent(instanceId)}/promote`;
  return sendJson(reviewPath(agreementId, documentId, tail), 'POST', {});
}

export async function addInstance(agreementId, documentId, body) {
  return sendJson(reviewPath(agreementId, documentId, 'instances'), 'POST', body);
}

export async function acknowledgeClause(agreementId, documentId, clauseId, note) {
  const tail = `clauses/${encodeURIComponent(clauseId)}/acknowledge`;
  return sendJson(reviewPath(agreementId, documentId, tail), 'POST', { note });
}

export async function resolveConflict(agreementId, documentId, conflictId, body) {
  const tail = `conflicts/${encodeURIComponent(conflictId)}/resolve`;
  return sendJson(reviewPath(agreementId, documentId, tail), 'POST', body);
}

// -------------------------------------------------------------------- approval

export async function approveAgreement(agreementId, note = '') {
  return sendJson(`${TRADE}/agreements/${encodeURIComponent(agreementId)}/approve`, 'POST', { note });
}

// ----------------------------------------------------- obligations, simulation

export async function loadObligations(agreementId) {
  return readJson(`${TRADE}/agreements/${encodeURIComponent(agreementId)}/obligations`);
}

export async function simulateAgreement(agreementId, body = {}) {
  return sendJson(`${TRADE}/agreements/${encodeURIComponent(agreementId)}/simulate`, 'POST', body);
}

export async function loadAttribution(ruleId) {
  return readJson(`${TRADE}/attribution/${ruleId.split('/').map(encodeURIComponent).join('/')}`);
}
