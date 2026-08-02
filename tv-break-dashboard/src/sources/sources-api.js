import { API_BASE } from '../shell/api';

// Every call the Sources destination makes, in one place, each returning a
// plain result rather than throwing, so a surface renders an honest offline
// state instead of an empty one that reads like an empty file.
//
// Two of them are not this destination's own endpoints: the compliance verdict
// and the forecast are what the frozen report downloaders build their CSVs
// from, and the preview of those reports reads exactly the same payload, so a
// row on screen and a row in the file cannot come from two derivations.

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function fetchUploadStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/uploads/status`, { credentials: 'include' });
    if (!response.ok) {
      return { online: false, status: null };
    }
    return { online: true, status: await readJson(response) };
  } catch {
    return { online: false, status: null };
  }
}

export async function fetchReports() {
  try {
    const response = await fetch(`${API_BASE}/api/reports`, { credentials: 'include' });
    if (!response.ok) return { online: false, reports: null };
    return { online: true, reports: await readJson(response) };
  } catch {
    return { online: false, reports: null };
  }
}

export async function fetchReportPreview(reportId, limit = 20) {
  try {
    const response = await fetch(`${API_BASE}/api/reports/${reportId}/preview?limit=${limit}`, { credentials: 'include' });
    if (!response.ok) return { online: false, preview: null };
    return { online: true, preview: await readJson(response) };
  } catch {
    return { online: false, preview: null };
  }
}

export async function fetchCompliance() {
  try {
    const response = await fetch(`${API_BASE}/api/compliance`, { credentials: 'include' });
    if (!response.ok) return { online: false, body: null };
    return { online: true, body: await readJson(response) };
  } catch {
    return { online: false, body: null };
  }
}

export async function fetchForecasts() {
  try {
    const response = await fetch(`${API_BASE}/api/forecasts`, { credentials: 'include' });
    if (!response.ok) return { online: false, body: null };
    return { online: true, body: await readJson(response) };
  } catch {
    return { online: false, body: null };
  }
}

export async function fetchPreview(kind, limit = 20) {
  try {
    const response = await fetch(`${API_BASE}/api/uploads/${kind}/preview?limit=${limit}`, { credentials: 'include' });
    if (!response.ok) return { online: false, preview: null };
    return { online: true, preview: await readJson(response) };
  } catch {
    return { online: false, preview: null };
  }
}

// The door: the same gate the upload runs, with nothing written. A refusal
// here is the refusal the upload would give, so a person never commits a file
// to find out.
export async function checkFile(kind, file) {
  const body = new FormData();
  body.append('file', file);
  try {
    const response = await fetch(`${API_BASE}/api/uploads/${kind}/check`, {
      method: 'POST',
      body,
      credentials: 'include',
    });
    const payload = await readJson(response);
    if (!response.ok) {
      return {
        ok: false,
        accepted: false,
        detail: (payload && payload.detail) || `${response.status} ${response.statusText}`,
        // The refusal's own Hebrew, when the server wrote that sentence itself.
        detail_he: (payload && payload.detail_he) || '',
        errors: (payload && payload.errors) || [],
        findings: (payload && payload.findings) || [],
      };
    }
    return { ok: true, ...(payload || {}) };
  } catch (error) {
    return { ok: false, accepted: false, detail: String(error.message || error), errors: [], findings: [] };
  }
}

export async function uploadFile(kind, file) {
  const body = new FormData();
  body.append('file', file);
  try {
    const response = await fetch(`${API_BASE}/api/uploads/${kind}`, {
      method: 'POST',
      body,
      credentials: 'include',
    });
    const payload = await readJson(response);
    if (!response.ok) {
      return {
        ok: false,
        detail: (payload && payload.detail) || `${response.status} ${response.statusText}`,
        detail_he: (payload && payload.detail_he) || '',
        errors: (payload && payload.errors) || [],
        findings: (payload && payload.findings) || [],
      };
    }
    return { ok: true, ...(payload || {}) };
  } catch (error) {
    return { ok: false, detail: String(error.message || error), errors: [], findings: [] };
  }
}
