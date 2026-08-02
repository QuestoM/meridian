// Every read and write the console makes, with three honest failure states.
//
// The surface is walled on affiliation, so a refusal is a first-class outcome
// and not an error: a channel account that somehow reaches this code reads
// `refused`, an unreachable server reads `unreachable`, and neither ever
// resolves to a payload with invented content. Nothing here retries silently.

export const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

export const SECTIONS = ['gates', 'coverage', 'drift', 'candidates', 'training', 'versions', 'provenance'];

// Each section names the one route it reads, so a reader can check any figure
// on the screen against one endpoint rather than guessing which produced it.
// The map covers the reads that own a section; a read about one named thing
// belongs to the section that lists that thing and is published beneath as its
// own reader. Between the two, every GET the surface publishes is reachable by
// a person and not only by the server, which a test asserts from the route
// table rather than from a list typed by hand.
export const SECTION_ROUTE = {
  gates: '/api/model/gates',
  coverage: '/api/model/coverage',
  drift: '/api/model/drift',
  candidates: '/api/model/candidates',
  training: '/api/model/training',
  versions: '/api/model/versions',
  provenance: '/api/model/provenance',
};

async function request(path, options) {
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      credentials: 'include',
      ...options,
    });
    if (response.status === 403) {
      let detail = '';
      try {
        detail = String((await response.json()).detail || '');
      } catch {
        detail = '';
      }
      return { status: 'refused', payload: null, detail };
    }
    if (!response.ok) {
      let detail = `${response.status}`;
      try {
        detail = String((await response.json()).detail || detail);
      } catch {
        // A non-JSON error body keeps the status line as the honest message.
      }
      return { status: 'error', payload: null, detail };
    }
    return { status: 'ok', payload: await response.json(), detail: '' };
  } catch {
    return { status: 'unreachable', payload: null, detail: '' };
  }
}

export function read(path) {
  return request(path, { method: 'GET' });
}

export function write(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export const readConsole = () => read('/api/model/console');
export const readSection = (section) => read(SECTION_ROUTE[section] || SECTION_ROUTE.gates);
// One candidate in full, with the verdict already recorded about it. The list
// route cannot answer this: a verdict is about a named candidate, so the read
// that carries it is keyed by that name, and the card for that candidate is
// where it belongs.
export const readCandidate = (id) => read(`/api/model/candidates/${encodeURIComponent(id)}`);
export const recordVersion = () => write('/api/model/versions');
export const recordDecision = (payload) => write('/api/model/decisions', payload);
export const measureCandidate = (id) => write(`/api/model/candidates/${encodeURIComponent(id)}/measure`);
export const startTraining = (artifact, flags) => write('/api/model/training', { artifact, flags });
