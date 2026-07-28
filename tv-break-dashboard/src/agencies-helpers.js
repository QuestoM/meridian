// Pure helpers for the Agencies page. Framework-free so they are trivially
// testable and reusable. The record shape mirrors the /api/agencies contract:
// agency_id, name, display_name, aliases, agency_type, contact_* (primary),
// contact2_* (secondary), address_city, address_street, vat_id,
// payment_terms_days, rebate_percent, commission_percent, credit_limit_ils,
// status, onboarded_at, notes, data_source.

import { pageText } from './advertisers-helpers';

export { pageText };

// Fields the drawer edits. agency_id, onboarded_at and data_source are
// read-only provenance and never leave the client as edits.
export const AGENCY_TEXT_FIELDS = [
  'name',
  'display_name',
  'aliases',
  'agency_type',
  'contact_name',
  'contact_role',
  'contact_phone',
  'contact_email',
  'contact2_name',
  'contact2_role',
  'contact2_phone',
  'contact2_email',
  'address_city',
  'address_street',
  'vat_id',
  'notes',
];

export const AGENCY_NUMBER_FIELDS = [
  'payment_terms_days',
  'rebate_percent',
  'commission_percent',
  'credit_limit_ils',
];

// Accept either a bare array or an { agencies: [...] } envelope, so the page
// keeps working whichever shape the backend lands with.
export function normalizeAgencies(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (payload && Array.isArray(payload.agencies)) {
    return payload.agencies;
  }
  return [];
}

// Same tolerance for the linked-advertisers envelope: { links: [...] },
// { advertisers: [...] } or a bare array of { advertiser, source } records.
export function normalizeLinks(payload) {
  const raw = Array.isArray(payload)
    ? payload
    : (payload && (payload.links || payload.advertisers)) || [];
  return (Array.isArray(raw) ? raw : []).map((entry) => {
    if (typeof entry === 'string') {
      return { advertiser: entry, source: 'observed' };
    }
    return {
      advertiser: String(entry.advertiser ?? entry.advertiser_id ?? ''),
      source: entry.source === 'manual' ? 'manual' : 'observed',
    };
  }).filter((entry) => entry.advertiser);
}

export function normalizeAgencyConditions(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (payload && Array.isArray(payload.conditions)) {
    return payload.conditions;
  }
  return [];
}

// A record is synthetic seed data only when the data itself says so.
export function isSynthetic(row) {
  return String(row?.data_source || '').trim().toLowerCase() === 'synthetic';
}

// Status vocabulary: known statuses get bilingual labels and a tone; unknown
// statuses render verbatim (engine data is never dropped or reinterpreted).
const STATUS_META = {
  active: { en: 'Active', he: 'פעילה', tone: 'teal' },
  inactive: { en: 'Inactive', he: 'לא פעילה', tone: 'muted' },
  suspended: { en: 'Suspended', he: 'מושהית', tone: 'amber' },
  pending: { en: 'Onboarding', he: 'בהקמה', tone: 'blue' },
};

export function statusMeta(status, locale) {
  const key = String(status || '').trim().toLowerCase();
  const meta = STATUS_META[key];
  if (meta) {
    return { key, label: pageText(locale, meta.en, meta.he), tone: meta.tone };
  }
  if (!key) {
    return { key: '', label: pageText(locale, 'Unknown', 'לא ידוע'), tone: 'muted' };
  }
  return { key, label: key, tone: 'muted' };
}

// Distinct status keys present in the data, in a stable known-first order.
export function statusKeys(agencies) {
  const known = Object.keys(STATUS_META);
  const seen = new Set(
    (agencies || []).map((row) => String(row.status || '').trim().toLowerCase()).filter(Boolean),
  );
  return [...known.filter((key) => seen.has(key)), ...[...seen].filter((key) => !known.includes(key)).sort()];
}

// Case-insensitive search over id, names, aliases, type and contact names,
// combined with the status chip filter.
export function filterAgencies(agencies, { search, status }) {
  const term = String(search || '').trim().toLowerCase();
  return (agencies || []).filter((row) => {
    if (status && status !== 'all' && String(row.status || '').trim().toLowerCase() !== status) {
      return false;
    }
    if (!term) {
      return true;
    }
    const haystack = [
      row.agency_id, row.name, row.display_name, row.aliases,
      row.contact_name, row.contact2_name, row.notes,
    ].map((value) => String(value ?? '').toLowerCase()).join(' ');
    return haystack.includes(term);
  });
}

function numberOrNull(value) {
  const text = value === null || value === undefined ? '' : String(value).trim();
  if (text === '') {
    return null;
  }
  const num = Number(text);
  return Number.isFinite(num) ? num : null;
}

// True when a drawer draft differs from the loaded record.
export function isAgencyDirty(original, draft) {
  if (!original || !draft) {
    return false;
  }
  const textChanged = AGENCY_TEXT_FIELDS.some(
    (field) => String(original[field] ?? '') !== String(draft[field] ?? ''),
  );
  const numberChanged = AGENCY_NUMBER_FIELDS.some(
    (field) => numberOrNull(original[field]) !== numberOrNull(draft[field]),
  );
  const statusChanged = String(original.status ?? '') !== String(draft.status ?? '');
  return textChanged || numberChanged || statusChanged;
}

// Build the PUT body: every editable field, numbers coerced, blanks preserved
// as null so the backend can distinguish "cleared" from a value.
export function toAgencyPayload(draft) {
  const body = {};
  AGENCY_TEXT_FIELDS.forEach((field) => {
    body[field] = String(draft[field] ?? '');
  });
  AGENCY_NUMBER_FIELDS.forEach((field) => {
    body[field] = numberOrNull(draft[field]);
  });
  body.status = String(draft.status ?? '');
  return body;
}

export function linkSourceLabel(source, locale) {
  if (source === 'manual') {
    return pageText(locale, 'Manual link', 'קישור ידני');
  }
  return pageText(locale, 'Observed in data', 'נצפה בנתונים');
}

// The best human name for a card title; the raw id stays visible separately.
export function agencyTitle(row) {
  return String(row.display_name || row.name || row.agency_id || '');
}

// Normalize the /api/agencies/summary payload. available is true only when the
// backend says so AND the money figures are real numbers; anything else renders
// the honest empty state, never a fabricated zero.
export function normalizeAgencySummary(payload) {
  const source = payload && typeof payload === 'object' ? payload : {};
  const gross = numberOrNull(source.gross_revenue);
  const net = numberOrNull(source.net_revenue);
  const rebate = numberOrNull(source.rebate_total);
  const spots = numberOrNull(source.spot_count);
  return {
    available: source.available === true && gross !== null && net !== null,
    gross,
    net,
    rebate,
    spots,
    basis: typeof source.basis === 'string' && source.basis.trim() ? source.basis : null,
  };
}
