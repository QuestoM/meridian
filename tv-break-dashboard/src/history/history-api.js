// The reads and writes History makes, in one place.
//
// Every call carries the session cookie and returns a tagged result rather than
// throwing, so a surface renders a real state (ready, refused, unreachable)
// instead of an empty page with no explanation. The refusal detail the server
// sends is passed through verbatim, because the words a person reads before a
// click and the words the server sends must be the same words.

import { API_BASE } from '../shell/api';

async function call(path, options = {}) {
  try {
    const response = await fetch(`${API_BASE}${path}`, { credentials: 'include', ...options });
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    if (!response.ok) {
      const detail = body && body.detail ? String(body.detail) : `${response.status}`;
      return { ok: false, status: response.status, data: null, error: detail };
    }
    return { ok: true, status: response.status, data: body, error: null };
  } catch (error) {
    return { ok: false, status: 0, data: null, error: error.message };
  }
}

function query(params) {
  const search = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== null && value !== undefined && String(value) !== '') search.set(key, String(value));
  });
  const text = search.toString();
  return text ? `?${text}` : '';
}

export function fetchTimeline(params) {
  return call(`/api/history${query(params)}`);
}

export function fetchRun(runId) {
  return call(`/api/history/runs/${encodeURIComponent(runId)}`);
}

export function fetchSince(day) {
  return call(`/api/history/since${query({ day })}`);
}

export function fetchVersionDiff(versionId) {
  return call(`/api/versions/${encodeURIComponent(versionId)}/diff`);
}

const JSON_HEADERS = { 'Content-Type': 'application/json' };

export function restoreVersion(versionId, files) {
  return call(`/api/versions/${encodeURIComponent(versionId)}/restore`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ files }),
  });
}

export function renameVersion(versionId, label) {
  return call(`/api/versions/${encodeURIComponent(versionId)}`, {
    method: 'PATCH',
    headers: JSON_HEADERS,
    body: JSON.stringify({ label }),
  });
}

export function saveRestorePoint(label) {
  return call('/api/versions/snapshot', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ label }),
  });
}
