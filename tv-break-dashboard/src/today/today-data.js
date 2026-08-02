// Today's own read, and why it does not wait for the rest of the app.
//
// The shell loads eleven endpoints with Promise.all, so every page it feeds
// renders when the slowest one lands. Measured on the running instance:
// /api/schedule is 2.43 s and 516 KB, and it is the long pole. Today has a
// five-second bar with zero clicks in it, so it fetches its own answer, which
// is one round trip, and renders the moment that lands.
//
// The request is started when this module is evaluated rather than in an
// effect, so it overlaps the bundle's own startup and the session probe that
// gates the first render. It only primes when the location actually resolves
// to Today, so a person landing elsewhere pays nothing.
//
// Credentials are set here explicitly. The shell installs a fetch wrapper that
// adds them, but it installs it in an effect, which is after this line runs.

import { NO_CHANNEL } from './today-scope';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

export const TODAY_PATH = '/api/today';
export const TARGET_PATH = '/api/plan-target';
export const DAY_PATH = '/api/today/day';

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, { credentials: 'include', ...(options || {}) });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = String(body.detail);
    } catch {
      // A non-JSON error body leaves the status line as the honest message.
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

export function fetchToday() {
  return request(TODAY_PATH);
}

// The second level of the drill, fetched when a day is opened. Kept off the
// first paint on purpose: seven day rows are what the five-second answer needs,
// and the ninety-odd rows behind one of them are what the next click needs.
export function fetchTodayDay(date) {
  return request(`${DAY_PATH}/${encodeURIComponent(date)}`);
}

export function saveTarget(body) {
  return request(TARGET_PATH, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function clearTarget() {
  return request(TARGET_PATH, { method: 'DELETE' });
}

let primed = null;

function landsOnToday() {
  if (typeof window === 'undefined') return false;
  const hash = decodeURIComponent(String(window.location.hash || '').replace(/^#/, ''));
  return hash === '' || hash === 'Overview' || hash === 'Assistant';
}

export function primeToday() {
  if (!primed) primed = fetchToday();
  return primed;
}

// The primed promise is consumed exactly once, by the first mount. Every later
// read is a fresh request, so a refresh really refreshes.
export function takePrimedToday() {
  const promise = primed;
  primed = null;
  return promise;
}

if (landsOnToday()) {
  primeToday().catch(() => {
    // A failed prime is not an error yet: the mount retries and reports it
    // there, where there is a surface to report it on.
  });
}

// The fallback shape, derived from the payload the shell already has. It
// carries no target and no per-day rows, because those come from the endpoint
// this fallback exists to survive the absence of. Every figure in it is read
// from the overview body, never computed here.
// The same refusal the endpoint carries, so the degraded path cannot serve the
// market total as the operator's just because the endpoint that refuses it is
// unreachable. It now lives in `today-scope.js`, beside the test that decides
// when it applies, because the blocks below the three answers refuse on the
// same cause and two copies of a refusal drift apart. Re-exported so this
// module's own surface is unchanged.
export { NO_CHANNEL };

export function todayFromOverview(overview) {
  const body = overview && typeof overview === 'object' ? overview : {};
  const summary = body.summary && typeof body.summary === 'object' ? body.summary : {};
  const week = summary.week && typeof summary.week === 'object' ? summary.week : null;
  const freshness = body.schedule_freshness && typeof body.schedule_freshness === 'object' ? body.schedule_freshness : {};
  const items = Array.isArray(body.recommendations) ? body.recommendations : [];
  const scoped = Boolean(summary.scope_channel);
  // The shell hands every page an offline-shaped payload until its own eleven
  // reads land. Its source counts are null, which is what tells this fallback
  // apart from a real answer that happens to be empty, and while that is what is
  // in hand the surface says it is still reading rather than rendering the
  // shape's zeros as findings.
  const answered = body.source_counts !== null && body.source_counts !== undefined;
  return {
    degraded: true,
    answered,
    channel: summary.scope_channel || null,
    window: {
      available: Boolean(week && week.date_from && week.date_to),
      date_from: week ? week.date_from : null,
      date_to: week ? week.date_to : null,
      n_dates: week ? week.n_dates : 0,
      basis: week ? week.basis : '',
      is_calendar_week: Boolean(week && week.basis === 'reference_date'),
    },
    money: {
      metric: 'projected_revenue',
      currency: 'ILS',
      amount_ils: scoped ? (week ? week.projected_revenue : summary.projected_revenue) : null,
      available: scoped && Boolean(week ? week.projected_revenue !== null : summary.projected_revenue !== null),
      unavailable: scoped ? null : NO_CHANNEL,
      scope: {
        channel: summary.scope_channel || null,
        date_from: week ? week.date_from : null,
        date_to: week ? week.date_to : null,
        n_dates: week ? week.n_dates : 0,
        source: 'saved_plan',
        basis: week ? week.basis : '',
      },
      days: [],
      days_total_ils: null,
      residual_ils: null,
      reconciled: false,
      total_breaks: scoped ? (week ? week.total_breaks : summary.total_breaks) : null,
      total_ad_seconds: scoped ? (week ? week.total_ad_seconds : summary.total_ad_seconds) : null,
      average_retention: scoped ? (week ? week.average_retention : summary.average_retention) : null,
    },
    target: { state: 'unavailable', can_edit: false },
    verdict: { state: 'unavailable', reason: 'no_target_store', variance_ils: null, variance_percent: null, threshold_en: '', threshold_he: '' },
    health: healthFromOverview(body, freshness, scoped),
    decisions: scoped
      ? { count: items.length, ranked_by: 'projected_revenue', items, scope: decisionScope(summary, scoped), unavailable: null }
      : { count: 0, ranked_by: 'projected_revenue', items: [], scope: decisionScope(summary, scoped), unavailable: NO_CHANNEL },
    plan_run_at: freshness.computed_at || null,
    model_trained_at: null,
  };
}

// The span the ranked list was drawn from, read off the same summary basis
// fields the endpoint reads. It is the whole saved plan for the operator's
// channel, which is wider than the money window above it, so the degraded path
// states it too rather than leaving five dated rows unexplained.
function decisionScope(summary, scoped) {
  return {
    channel: scoped ? summary.scope_channel || null : null,
    date_from: scoped ? summary.date_from || null : null,
    date_to: scoped ? summary.date_to || null : null,
    n_dates: scoped ? Number(summary.n_dates || 0) : 0,
    inclusive: true,
    source: 'saved_plan',
    grain: 'whole_saved_plan',
  };
}

// The same three checks the endpoint computes, from the same fields, for the
// degraded path only. The model-side groups are folded into one notice here
// too, so no engine word reaches the surface by the back door.
const MODEL_SIDE = new Set(['coefficients', 'the impact model', 'impact_model', 'the audience model', 'audience_model']);

function healthFromOverview(body, freshness, scoped) {
  const status = String(freshness.status || 'unknown').toLowerCase();
  const changed = Array.isArray(freshness.changed) ? freshness.changed : [];
  const yours = changed.filter((key) => !MODEL_SIDE.has(String(key))).map((key) => ({ key: String(key), label_en: String(key), label_he: String(key) }));
  const modelChanged = changed.some((key) => MODEL_SIDE.has(String(key)));
  const counts = body.source_counts && typeof body.source_counts === 'object' ? body.source_counts : null;
  const checks = [];
  if (!scoped) checks.push({ id: 'operator_channel_unset', status: 'attention', opens: 'settings', ...NO_CHANNEL });
  if (status === 'stale' && yours.length) checks.push({ id: 'plan_out_of_date', status: 'attention', opens: 'plan', changed: yours, plan_run_at: freshness.computed_at });
  if (status === 'stale' && modelChanged) checks.push({ id: 'newer_model_version', status: 'notice', opens: 'plan', model_trained_at: null, plan_run_at: freshness.computed_at });
  if (status === 'fresh') checks.push({ id: 'plan_current', status: 'ok', opens: 'plan', plan_run_at: freshness.computed_at });
  if (status === 'unknown') checks.push({ id: 'plan_currency_unknown', status: 'unknown', opens: 'plan', plan_run_at: freshness.computed_at });
  const compliance = body.compliance && typeof body.compliance === 'object' ? body.compliance : {};
  const all = Array.isArray(compliance.checks) ? compliance.checks : [];
  const breached = all.filter((check) => String(check.status || '').toLowerCase() !== 'compliant');
  checks.push({
    id: 'licence',
    status: breached.length ? 'attention' : all.length ? 'ok' : 'unknown',
    opens: 'licence',
    checks_total: all.length,
    checks_breached: breached.length,
    breached_labels_en: breached.map((check) => String(check.label_en || '')),
    breached_labels_he: breached.map((check) => String(check.label_he || '')),
    profile: compliance.profile || null,
    effective_date: compliance.effective_date || null,
  });
  const missing = counts ? ['programmes', 'spots', 'planned_break_rows'].filter((key) => !Number(counts[key] || 0)) : [];
  checks.push({
    id: 'inputs',
    status: !counts ? 'unknown' : missing.length ? 'attention' : 'ok',
    opens: 'sources',
    programmes: counts ? Number(counts.programmes || 0) : null,
    spots: counts ? Number(counts.spots || 0) : null,
    planned_break_rows: counts ? Number(counts.planned_break_rows || 0) : null,
    missing,
    newest_input_at: body.data_freshness || null,
  });
  const attention = checks.filter((check) => check.status === 'attention');
  return { state: attention.length ? 'attention' : 'clear', attention_count: attention.length, checks };
}
