// Pure helpers and the scoped day-recompute job client for the override
// console. Talks to the
// async jobs API and reports an honest terminal status; errors come from
// the server record, never invented here.

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

// Returns {status: 'done'|'failed'|'missing'|'timeout', error?: string}.
// 'missing' means the jobs API is absent (older backend), so callers should
// point the operator at the full recompute instead.
export async function runDayRecomputeJob(apiBase, scope) {
  const startResponse = await fetch(`${apiBase}/api/jobs/recompute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scope }),
  });
  if (startResponse.status === 404) return { status: 'missing' };
  if (!startResponse.ok) return { status: 'failed', error: `${startResponse.status} ${startResponse.statusText}` };
  const { job_id: jobId } = await startResponse.json();
  for (let attempt = 0; attempt < 120; attempt += 1) {
    await sleep(1000);
    const statusResponse = await fetch(`${apiBase}/api/jobs/${jobId}`);
    if (!statusResponse.ok) return { status: 'failed', error: `${statusResponse.status} ${statusResponse.statusText}` };
    const record = await statusResponse.json();
    if (record.status === 'done') return { status: 'done' };
    if (record.status === 'failed') return { status: 'failed', error: record.error || '' };
  }
  return { status: 'timeout' };
}

export const asList = (json, key) => (Array.isArray(json) ? json : (json && json[key]) || []);

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export function fmtNum(value, locale) {
  if (!isNum(value)) return '-';
  return value.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { maximumFractionDigits: 2 });
}

export function anchorText(o) {
  const parts = [o.anchor_date, o.anchor_start, o.anchor_title].filter(Boolean);
  return parts.length ? parts.join(' - ') : '';
}

// An override reads stale when the backend says so, or when its anchor no
// longer matches the live segment carrying the same id (a re-ingest drifted).
export function isStale(o, segById) {
  if (o.status === 'stale') return true;
  const seg = segById.get(o.target_id);
  if (!seg || !seg.anchor) return false;
  const a = seg.anchor;
  const program = a.program || a.title;
  const drift = (o.anchor_date && a.date && o.anchor_date !== a.date)
    || (o.anchor_start && a.start_clock && o.anchor_start !== a.start_clock)
    || (o.anchor_title && program && o.anchor_title !== program);
  return Boolean(drift);
}
