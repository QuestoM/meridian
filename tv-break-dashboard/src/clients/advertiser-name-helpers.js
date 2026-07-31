// Display-name layer for advertiser records. Framework-free, like the other
// advertiser helper modules, so every rule here is trivially testable.
//
// The backend record carries { advertiser_id, display_name, name_source } where
// name_source is one of:
//   operator  - the operator stored a display name (display_name is non-empty)
//   observed  - the raw id itself is a real advertiser name seen in the daily data
//   unnamed   - only a raw token exists (for example the ADV_01..45 seed keys);
//               we prettify it for display but never invent a company name, and
//               the record is flagged so the operator can fill a real name.

import { pageText } from './advertisers-helpers';

// The rules-store seed key shape (ADV_01..ADV_45). These tokens are inert on
// real spots, so their display fallback is an honest generic label, not a name.
const SEED_ID_PATTERN = /^ADV[_-]?0*(\d+)$/i;

// The operator-stored display name, trimmed; empty string when none is stored.
export function operatorName(row) {
  return String(row?.display_name ?? '').trim();
}

// True when the record has no real name at all: nothing stored by the operator
// and the raw id is not an observed daily-data advertiser name.
export function isUnnamed(row) {
  return !operatorName(row) && String(row?.name_source ?? '') !== 'observed';
}

// Honest prettification of a raw token for display. A seed key becomes the
// generic bilingual label ("Advertiser 7" / "מפרסם 7"); any other latin token
// gets underscores and dashes turned into spaces and each word title-cased;
// Hebrew text passes through unchanged. The raw id stays visible elsewhere.
export function prettifyRawId(rawId, locale) {
  const raw = String(rawId ?? '').trim();
  const seed = SEED_ID_PATTERN.exec(raw);
  if (seed) {
    return pageText(locale, `Advertiser ${Number(seed[1])}`, `מפרסם ${Number(seed[1])}`);
  }
  const pretty = raw
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .map((word) => (/^[a-z]/i.test(word) ? word[0].toUpperCase() + word.slice(1).toLowerCase() : word))
    .join(' ');
  return pretty || raw;
}

// The name shown prominently for a record: the operator's stored name first,
// then the raw id itself when it is a real observed name, then the honest
// prettified token. Never empty for a row with an advertiser_id.
export function displayNameOf(row, locale) {
  const stored = operatorName(row);
  if (stored) {
    return stored;
  }
  const raw = String(row?.advertiser_id ?? '');
  if (String(row?.name_source ?? '') === 'observed') {
    return raw;
  }
  return prettifyRawId(raw, locale);
}

// Whether the quiet secondary raw-id line should render: only when it adds
// information the prominent name does not already carry.
export function showsRawIdLine(row, locale) {
  return displayNameOf(row, locale) !== String(row?.advertiser_id ?? '');
}

// Search text for a record's name layer in both locales, so a search matches
// what the operator sees regardless of language, plus the stored name itself.
export function nameSearchHaystack(row) {
  return [operatorName(row), displayNameOf(row, 'en'), displayNameOf(row, 'he')].join(' ');
}

// Split rows into named (operator or observed) and unnamed groups, preserving
// the incoming order within each group.
export function partitionByName(rows) {
  const named = [];
  const unnamed = [];
  (rows || []).forEach((row) => (isUnnamed(row) ? unnamed : named).push(row));
  return { named, unnamed };
}

// Sort for the by-name view: named records alphabetically by their display
// name (locale-aware collation), then every unnamed raw-token record grouped
// last, ordered by raw id with numeric-aware compare so ADV_2 precedes ADV_10.
export function sortByDisplayName(rows, locale) {
  const { named, unnamed } = partitionByName(rows);
  const lang = locale === 'he' ? 'he' : 'en';
  named.sort((a, b) => displayNameOf(a, locale).localeCompare(displayNameOf(b, locale), lang, { numeric: true }));
  unnamed.sort((a, b) => String(a.advertiser_id ?? '').localeCompare(String(b.advertiser_id ?? ''), 'en', { numeric: true }));
  return [...named, ...unnamed];
}
