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

function linkName(entry) {
  if (typeof entry === 'string') {
    return entry.trim();
  }
  if (entry && typeof entry === 'object') {
    return String(entry.advertiser ?? entry.advertiser_id ?? '').trim();
  }
  return '';
}

// GET /api/agencies/{id}/advertisers answers three name lists, not a links
// array: observed (read from the daily spot file and from the stored links),
// manual (linked by hand) and effective, which is what this agency actually
// holds after the backend's manual-wins rule removes any name another agency
// claims by hand. The list a person is owed is therefore the effective one, and
// a name is manual only when the manual list carries it.
//
// Reading only { links } or { advertisers } was the measured defect: the real
// payload has neither key, so every agency rendered as empty while its names
// sat in the response. Both older envelopes and a bare array of
// { advertiser, source } records still parse, so nothing that worked stops.
export function normalizeLinks(payload) {
  const envelope = payload && typeof payload === 'object' && !Array.isArray(payload) ? payload : {};
  const manualNames = new Set((Array.isArray(envelope.manual) ? envelope.manual : []).map(linkName).filter(Boolean));
  let raw;
  if (Array.isArray(envelope.effective)) {
    raw = envelope.effective;
  } else if (Array.isArray(envelope.observed) || manualNames.size > 0) {
    raw = [...(envelope.observed || []), ...(envelope.manual || [])];
  } else if (Array.isArray(payload)) {
    raw = payload;
  } else {
    raw = envelope.links || envelope.advertisers || [];
  }
  const seen = new Set();
  const rows = [];
  (Array.isArray(raw) ? raw : []).forEach((entry) => {
    const advertiser = linkName(entry);
    if (!advertiser || seen.has(advertiser)) {
      return;
    }
    seen.add(advertiser);
    const manual = manualNames.has(advertiser)
      || (entry && typeof entry === 'object' && entry.source === 'manual');
    rows.push({ advertiser, source: manual ? 'manual' : 'observed' });
  });
  return rows;
}

// The daily spot file the observed names were read from, so the section can
// state its own basis. null when the payload names none, which is a real state:
// no daily file is loaded.
export function linksSourceFile(payload) {
  const value = payload && typeof payload === 'object' ? payload.observed_source_file : null;
  const text = String(value ?? '').trim();
  return text || null;
}

// Hebrew and English both have a singular, and three of the nine agencies on
// this data hold exactly one advertiser, so the word follows the number.
export function linksWord(count, locale) {
  if (count === 1) {
    return pageText(locale, 'advertiser', 'מפרסם');
  }
  return pageText(locale, 'advertisers', 'מפרסמים');
}

// The basis under a list that has names: which file they were observed in. The
// file name is a Latin run inside a Hebrew line, so it is isolated the way every
// other embedded run on this destination is, or the sentence's own full stop
// renders as part of the file name.
export function linkBasisNote(sourceFile, locale) {
  return pageText(locale, `Observed in the daily spot file ${sourceFile}.`, `נצפו בקובץ הספוטים היומי ⁦${sourceFile}⁩.`);
}

// The honest empty state. It names the file that was read and found nothing, or
// says no file is loaded and where to load one, rather than implying that an
// agency with advertisers has none.
export function linkEmptyNote(sourceFile, locale) {
  if (sourceFile) {
    return pageText(locale, `No advertiser in the daily spot file ${sourceFile} books through this agency, and none is linked by hand.`, `אף מפרסם בקובץ הספוטים היומי ⁦${sourceFile}⁩ אינו מזמין דרך סוכנות זו, ואין קישור ידני.`);
  }
  return pageText(locale, 'No daily spot file is loaded, so observed links cannot be read. Load one on the Data page.', 'לא טעון קובץ ספוטים יומי, ולכן לא ניתן לקרוא קישורים נצפים. אפשר לטעון אותו בעמוד הנתונים.');
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
