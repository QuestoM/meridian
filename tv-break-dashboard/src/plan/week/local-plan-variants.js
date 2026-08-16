const STORAGE_PREFIX = 'kairos.plan.local-day-drafts.v1';
const MAX_VARIANTS_PER_PLAN = 20;

function hashText(value) {
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    first = Math.imul(first ^ code, 0x01000193);
    second = Math.imul(second ^ code, 0x85ebca6b);
  }
  return `day-${(first >>> 0).toString(16).padStart(8, '0')}-${(second >>> 0).toString(16).padStart(8, '0')}`;
}

function boardFingerprint(board) {
  if (!board?.available) return '';
  const payload = {
    channel: board.operator_channel || board.basis?.channel || '',
    day: board.day || board.basis?.day || '',
    basis: {
      revenue_weight: board.basis?.revenue_weight,
      risk_lambda: board.basis?.risk_lambda,
      objective_mode: board.basis?.objective_mode,
      segments: board.basis?.segments,
    },
    programmes: (board.programmes || []).map((row) => [
      row.segment_id, row.start_seconds, row.duration_seconds, row.breaks,
    ]),
    breaks: (board.breaks || []).map((row) => [
      row.break_id, row.segment_id, row.offset_seconds, row.duration_seconds, Boolean(row.is_gold),
    ]),
  };
  return hashText(JSON.stringify(payload));
}

function cleanScope(scope) {
  return {
    channel: String(scope?.channel || ''),
    day: String(scope?.day || ''),
    planIdentity: String(scope?.planIdentity || ''),
    computedAt: String(scope?.computedAt || ''),
    fingerprint: String(scope?.fingerprint || ''),
  };
}

function encoded(value) {
  return encodeURIComponent(String(value || ''));
}

function dayPrefix(scope) {
  return `${STORAGE_PREFIX}:${encoded(scope.channel)}:${encoded(scope.day)}:`;
}

function storageKey(scope) {
  return `${dayPrefix(scope)}${encoded(scope.planIdentity)}:${encoded(scope.computedAt)}:${encoded(scope.fingerprint)}`;
}

function emptyBucket(scope) {
  return { schema: 1, scope: cleanScope(scope), baseline: null, variants: [] };
}

function readKey(key, scope) {
  const raw = window.localStorage.getItem(key);
  if (!raw) return emptyBucket(scope);
  const parsed = JSON.parse(raw);
  return {
    schema: 1,
    scope: cleanScope(parsed?.scope || scope),
    baseline: parsed?.baseline || null,
    variants: Array.isArray(parsed?.variants) ? parsed.variants : [],
  };
}

function writeBucket(scope, bucket) {
  window.localStorage.setItem(storageKey(scope), JSON.stringify(bucket));
}

function arrangementFor(board, edits = {}) {
  return (board?.breaks || []).map((item) => {
    const edit = edits[item.break_id] || {};
    return {
      break_id: String(item.break_id),
      segment_id: String(item.segment_id),
      offset_seconds: Number(edit.offset_seconds ?? item.offset_seconds),
      duration_seconds: Number(edit.duration_seconds ?? item.duration_seconds),
      is_gold: Boolean(item.is_gold),
    };
  });
}

export function createVariantScope(board, live, freshness) {
  const channel = String(board?.operator_channel || board?.basis?.channel || '').trim();
  const day = String(board?.day || board?.basis?.day || '').trim();
  const planIdentity = live?.sha256 ? `weekly-sha256:${live.sha256}` : '';
  const computedAt = String(live?.computed_at || freshness?.computed_at || '').trim();
  const fingerprint = boardFingerprint(board);
  return {
    channel,
    day,
    planIdentity,
    computedAt,
    fingerprint,
    verifiable: Boolean(channel && day && planIdentity && computedAt && fingerprint),
  };
}

export function sameVariantScope(first, second) {
  return ['channel', 'day', 'planIdentity', 'computedAt', 'fingerprint']
    .every((field) => String(first?.[field] || '') === String(second?.[field] || ''));
}

export function variantIsCurrent(variant, scope) {
  return Boolean(scope?.verifiable && sameVariantScope(variant?.scope, scope));
}

export function loadDayDrafts(scope) {
  if (typeof window === 'undefined' || !scope?.channel || !scope?.day) {
    return { baselines: [], variants: [], error: null };
  }
  try {
    const prefix = dayPrefix(scope);
    const buckets = [];
    for (let index = 0; index < window.localStorage.length; index += 1) {
      const key = window.localStorage.key(index);
      if (key?.startsWith(prefix)) buckets.push(readKey(key, scope));
    }
    const baselines = buckets.map((bucket) => bucket.baseline).filter(Boolean)
      .sort((a, b) => String(b.capturedAt).localeCompare(String(a.capturedAt)));
    const variants = buckets.flatMap((bucket) => bucket.variants).filter(Boolean)
      .sort((a, b) => String(b.savedAt).localeCompare(String(a.savedAt)));
    return { baselines, variants, error: null };
  } catch (error) {
    return { baselines: [], variants: [], error: error.message };
  }
}

export function ensureLocalBaseline(scope, board) {
  if (!scope?.verifiable || typeof window === 'undefined') {
    return { ok: false, error: 'This plan has no verifiable file identity and run timestamp.' };
  }
  try {
    const key = storageKey(scope);
    const bucket = readKey(key, scope);
    if (bucket.baseline) return { ok: true, baseline: bucket.baseline, created: false };
    const baseline = {
      id: `baseline:${scope.fingerprint}`,
      kind: 'optimizer-baseline',
      immutable: true,
      capturedAt: new Date().toISOString(),
      scope: cleanScope(scope),
      totals: { ...(board?.totals || {}) },
      committedTotals: board?.basis?.committed ? { ...board.basis.committed } : null,
      arrangement: arrangementFor(board),
    };
    writeBucket(scope, { ...bucket, scope: cleanScope(scope), baseline });
    return { ok: true, baseline, created: true };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

export function storeLocalVariant(scope, board, draft) {
  if (!scope?.verifiable || typeof window === 'undefined') {
    return { ok: false, error: 'The exact plan identity cannot be verified.' };
  }
  try {
    const key = storageKey(scope);
    const bucket = readKey(key, scope);
    if (!bucket.baseline) return { ok: false, error: 'The immutable browser baseline is missing.' };
    const id = globalThis.crypto?.randomUUID?.() || `draft-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const variant = {
      id,
      kind: 'browser-local-manual-variant',
      name: String(draft.name || '').trim().slice(0, 80),
      savedAt: new Date().toISOString(),
      scope: cleanScope(scope),
      edits: JSON.parse(JSON.stringify(draft.edits || {})),
      editCount: Object.keys(draft.edits || {}).length,
      arrangement: arrangementFor(board, draft.edits),
      optimizerTotals: { ...(draft.optimizerTotals || board?.totals || {}) },
      totals: { ...(draft.totals || {}) },
      delta: { ...(draft.delta || {}) },
      compliance: { ...(draft.compliance || {}) },
    };
    const variants = [variant, ...bucket.variants].slice(0, MAX_VARIANTS_PER_PLAN);
    writeBucket(scope, { ...bucket, scope: cleanScope(scope), variants });
    return { ok: true, variant };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

export function removeLocalVariant(variant) {
  if (!variant?.id || typeof window === 'undefined') return { ok: false, error: 'Draft identity is missing.' };
  try {
    const scope = cleanScope(variant.scope);
    const key = storageKey(scope);
    const bucket = readKey(key, scope);
    writeBucket(scope, { ...bucket, variants: bucket.variants.filter((item) => item.id !== variant.id) });
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

function editValue(edit) {
  if (!edit) return '';
  return `${Number(edit.offset_seconds)}|${Number(edit.duration_seconds)}`;
}

export function changedEditsBetween(first = {}, second = {}) {
  const ids = new Set([...Object.keys(first), ...Object.keys(second)]);
  return Array.from(ids).filter((id) => editValue(first[id]) !== editValue(second[id])).length;
}

export function sameEdits(first = {}, second = {}) {
  return changedEditsBetween(first, second) === 0;
}
