import { API_BASE } from '../../shell/api';

// Plan, the week: the calls this destination makes on its own behalf.
//
// The frozen shell router hands each of the four entrances a different prop set:
// only Schedule receives a run handler, only Settings receives a save handler,
// and none receives both. A destination whose behaviour changed with the door
// somebody came through would not be one destination, so it fetches what it
// needs itself and tells the shell to refresh afterwards when the shell gave it
// a way to.
//
// Every function returns a tagged result rather than throwing, because a failure
// on this surface has to be rendered as an honest state with its reason and not
// swallowed into an empty panel.

async function call(path, options = {}) {
  try {
    const response = await fetch(`${API_BASE}${path}`, options);
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    if (!response.ok) {
      const detail = body && body.detail ? String(body.detail) : `${response.status} ${response.statusText}`;
      return { ok: false, status: response.status, error: detail, data: body };
    }
    return { ok: true, status: response.status, error: null, data: body };
  } catch (error) {
    return { ok: false, status: 0, error: error.message, data: null, offline: true };
  }
}

const jsonPost = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export function readSettings() {
  return call('/api/settings');
}

// The week, and optionally one broadcast day of it. With a date the payload is
// the same shape with its embedded break board scoped to that day, so the board
// a comparison row opens is the day that row is about and never a nearby one.
export function readSchedule(date) {
  const wanted = String(date || '').trim();
  return call(wanted ? `/api/schedule?date=${encodeURIComponent(wanted)}` : '/api/schedule');
}

export function readInventory() {
  return call('/api/inventory');
}

export function saveSettings(next) {
  return call('/api/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(next) });
}

export function readParameters() {
  return call('/api/parameters');
}

// The goal and the progress against it, for the plan's own week. The window,
// the projection, the target and the three-state verdict all come back from one
// call, so the surface never computes a threshold of its own.
export function readPlanProgress() {
  return call('/api/plan-progress');
}

// What a second of airtime is worth, computed once by the piece that owns the
// rate card and read here rather than computed a second time.
export function readYieldPerSecond() {
  return call('/api/yield-per-second');
}

// Whether the saved plan is still in step with the inputs it was built from.
// The verdict rides on the overview payload but is computed per request rather
// than served from that payload's cache, so a read taken the moment a run
// finishes reports the run rather than the state before it.
export async function readPlanFreshness() {
  const result = await call('/api/overview');
  if (!result.ok) return result;
  const verdict = result.data && result.data.schedule_freshness;
  if (!verdict || typeof verdict !== 'object') {
    return { ok: false, status: result.status, error: 'the server returned no plan freshness verdict', data: null };
  }
  return { ok: true, status: result.status, error: null, data: verdict };
}

export function readPlanVersions() {
  return call('/api/plan-versions');
}

export function publishPlanVersion(name, note) {
  return call('/api/plan-versions', jsonPost({ name, note: note || '' }));
}

export function readPlanVersionDiff(versionId, against) {
  const query = against ? `?against=${encodeURIComponent(against)}` : '';
  return call(`/api/plan-versions/${encodeURIComponent(versionId)}/diff${query}`);
}

export function restorePlanVersion(versionId) {
  return call(`/api/plan-versions/${encodeURIComponent(versionId)}/restore`, jsonPost({}));
}

export function compareScenarios(body) {
  return call('/api/scenario-compare', jsonPost(body));
}

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

// Run the weekly plan as a background job and report real progress. The
// synchronous route is the fallback for a backend without the job API, exactly
// as the shell's own handler does, so this path cannot be the slower one.
export async function runWeeklyPlan({ onProgress, scope = null } = {}) {
  const started = await call('/api/jobs/recompute', jsonPost(scope ? { scope } : {}));
  if (started.status === 404) {
    return call('/api/recompute-schedule', { method: 'POST' });
  }
  if (!started.ok) return started;
  const jobId = started.data?.job_id;
  if (!jobId) return { ok: false, status: 0, error: 'the run started without a job id', data: null };
  for (let attempt = 0; attempt < 400; attempt += 1) {
    await sleep(1500);
    const record = await call(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (!record.ok) return record;
    const job = record.data || {};
    if (job.progress && Number.isFinite(job.progress.done) && Number.isFinite(job.progress.total)) {
      onProgress?.(job.progress);
    }
    if (job.status === 'done') return { ok: true, status: 200, error: null, data: job.result || {} };
    if (job.status === 'failed') {
      return { ok: false, status: 0, error: String(job.error || 'the run failed'), data: null };
    }
  }
  return { ok: false, status: 0, error: 'the run is still going after ten minutes', data: null };
}
