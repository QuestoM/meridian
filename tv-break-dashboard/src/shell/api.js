// Empty default = same origin. In Vite dev the /api proxy reaches the Kairos
// API; in production the API serves the built SPA.

export const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

export async function fetchJson(path, fallback) {
  try {
    const response = await fetch(`${API_BASE}${path}`);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return { data: await response.json(), online: true, error: null };
  } catch (error) {
    return { data: fallback, online: false, error: error.message };
  }
}

// Backend segment overrides accept the kinds pin | force | forbid | gold. The
// recommendation payload speaks a slightly richer intent vocabulary, so map its
// proposed_kind onto the override kind the store expects: a "lower_count" intent
// resolves to a forced break count (force), everything else passes through.
export function mapProposedKind(proposedKind) {
  const value = String(proposedKind || '').trim();
  if (value === 'lower_count') return 'force';
  if (value === 'gold' || value === 'pin' || value === 'forbid' || value === 'force') return value;
  return '';
}

// Posts a break decision. Returns { ok, status, error, decision }. A 404 means an
// older backend without the decision route, which is treated as ok so the annotation
// only decision log keeps working; a real error surfaces its status honestly.
export async function postBreakDecision(payload) {
  try {
    const response = await fetch(`${API_BASE}/api/break-decisions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (response.status === 404) return { ok: true, status: 404, error: null, decision: null };
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (body && body.detail) detail = String(body.detail);
      } catch {
        // Non-JSON error body: keep the status line as the honest message.
      }
      return { ok: false, status: response.status, error: detail, decision: null };
    }
    let decision = null;
    try {
      decision = await response.json();
    } catch {
      decision = null;
    }
    return { ok: true, status: response.status, error: null, decision };
  } catch (error) {
    // Network unreachable: the UI keeps its local decision state offline.
    return { ok: true, status: 0, error: error.message, decision: null, offline: true };
  }
}
