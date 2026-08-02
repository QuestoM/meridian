// Display-name layer for advertiser records. Framework-free, like the other
// advertiser helper modules, so every rule here is trivially testable.
//
// The backend record carries { advertiser_id, name, display_name, name_source }.
// `name` is the advertiser the row is bound to and is what makes the row price
// anybody; `display_name` is the operator's own label for it. `name_source` is
// one of:
//   operator  - the operator stored a display name or bound the row by name
//   observed  - the raw id itself is a real advertiser name seen in the daily data
//   unnamed   - only a raw token exists (for example the ADV_01..45 seed keys),
//               which is a pricing row bound to nobody. It is shown as its own
//               id and flagged, and no company name is ever invented for it.

// The rules-store seed key shape (ADV_01..ADV_45). These tokens are inert on
// real spots, so they are shown as themselves rather than dressed as a name.
const SEED_ID_PATTERN = /^ADV[_-]?0*(\d+)$/i;

// The operator-stored display name, trimmed; empty string when none is stored.
export function operatorName(row) {
  return String(row?.display_name ?? '').trim();
}

// The advertiser this row is bound to, trimmed. Filling this cell is what makes
// the row price that advertiser, so a row that carries one is a named record
// even when nobody typed a separate display name for it.
export function boundName(row) {
  return String(row?.name ?? '').trim();
}

// The real name the identity join found for the advertiser this row prices:
// the daily ledger's own name space, joined by advertiser_id, attached by
// mergeRowWithIdentity as `bound_advertiser`. This is what turns "ADV_01" into
// the Hebrew name an operator actually recognises; the rules store's own
// `name` cell is empty on every seeded row, so this is the source that matters
// in practice.
export function identityName(row) {
  return String(row?.bound_advertiser ?? '').trim();
}

// True when the record has no real name at all: nothing bound, nothing stored by
// the operator, no identity join, and the raw id is not an observed daily-data
// advertiser name.
export function isUnnamed(row) {
  return (
    !operatorName(row)
    && !boundName(row)
    && !identityName(row)
    && String(row?.name_source ?? '') !== 'observed'
  );
}

// Honest prettification of a raw token for display. A seed key is shown as
// itself: calling ADV_01 "Advertiser 1" reads as a nameless advertiser when the
// row is a pricing rule bound to nobody, and the card says which it is beside
// the id. Any other latin token gets underscores and dashes turned into spaces
// and each word title-cased; Hebrew text passes through unchanged.
export function prettifyRawId(rawId, locale) {
  const raw = String(rawId ?? '').trim();
  if (SEED_ID_PATTERN.test(raw)) {
    return raw;
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
// then the identity join's real observed name, then the rules store's own
// bound name, then the raw id itself when it is a real observed name, then the
// honest prettified token. Never empty for a row with an advertiser_id.
export function displayNameOf(row, locale) {
  const stored = operatorName(row) || identityName(row) || boundName(row);
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
  return [
    operatorName(row),
    identityName(row),
    boundName(row),
    String(row?.aliases ?? ''),
    displayNameOf(row, 'en'),
    displayNameOf(row, 'he'),
  ].join(' ');
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
