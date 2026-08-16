import { serializeNode } from './constraint-predicate';

export const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

export function t(locale, en, he) {
  return locale === 'he' ? he : en;
}

export function rowSentence(item, sentences, locale) {
  const said = sentences.get(String((item || {}).restriction_id || '').trim());
  if (said) return locale === 'he' ? said.he : said.en;
  return String((item || {}).notes || '');
}

export async function failure(res) {
  const body = await res.json().catch(() => null);
  const raw = body && body.detail;
  const words = raw && typeof raw === 'object' ? raw : null;
  const error = new Error(words ? String(words.en || words.he || '') : (raw ? String(raw) : `${res.status} ${res.statusText}`));
  error.words = words;
  return error;
}

export function normalizeRows(value) {
  if (Array.isArray(value)) return value;
  if (value && Array.isArray(value.constraints)) return value.constraints;
  if (value && Array.isArray(value.rows)) return value.rows;
  return [];
}

function mmssToSeconds(value) {
  const [minutes, seconds] = String(value || '00:00').split(':').map((part) => Number(part));
  return (Number.isFinite(minutes) ? minutes : 0) * 60 + (Number.isFinite(seconds) ? seconds : 0);
}

export function buildBody(draft, where) {
  const body = {
    scope_type: 'always',
    scope_value: '',
    channel: '',
    effect: draft.effect,
    order_index: draft.order_index === '' ? null : Number(draft.order_index),
    notes: draft.notes || '',
  };
  if (draft.effect === 'FIX_OFFSET') {
    body.offset_seconds = mmssToSeconds(draft.offset_mmss);
  } else if (draft.effect === 'OFFSET_WINDOW') {
    body.offset_min_seconds = mmssToSeconds(draft.offset_min);
    body.offset_max_seconds = mmssToSeconds(draft.offset_max);
  } else if (draft.effect === 'PIN_COUNT') {
    body.count = Number(draft.pin_count);
  } else if (draft.effect === 'DURATION_RANGE') {
    body.duration_min_seconds = Number(draft.duration_min);
    body.duration_max_seconds = Number(draft.duration_max);
  }
  const serializedWhere = serializeNode(where);
  if (serializedWhere.conditions && serializedWhere.conditions.length > 0) body.where = serializedWhere;
  return body;
}

export function predicateComplete(node) {
  if (!node || typeof node !== 'object') return false;
  if (Array.isArray(node.conditions)) {
    return node.conditions.length > 0 && node.conditions.every(predicateComplete);
  }
  const value = node.value;
  if (Array.isArray(value)) return value.length > 0 && value.every((item) => String(item).trim());
  if (value && typeof value === 'object') {
    return value.min !== '' && value.min !== null && value.min !== undefined
      && value.max !== '' && value.max !== null && value.max !== undefined;
  }
  return String(value ?? '').trim().length > 0;
}
