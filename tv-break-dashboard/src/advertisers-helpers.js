// Pure helpers for the Advertisers management page.
// Kept framework-free so they are trivially testable and reusable.

export const POSITION_PRESETS = ['ANY', '1', '2', '3', 'last'];
export const GENRE_PRESETS = ['ANY', 'News', 'PrimeShow1', 'PrimeShow2', 'Other'];

export const EMPTY_ADVERTISER = {
  advertiser_id: '',
  default_premium: 1,
  allow_positions: 'ANY',
  allow_genres: 'ANY',
  prime_time_only: false,
  notes: '',
};

export function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}

export function normalizeRows(value) {
  return Array.isArray(value) ? value : [];
}

// Parse a stored constraint string ("ANY" or "1,2,last") into a token array.
// Whitespace is trimmed, empties dropped. "ANY" (case-insensitive) collapses to ["ANY"].
export function parseTokens(value) {
  if (value === null || value === undefined) {
    return ['ANY'];
  }
  const raw = String(value)
    .split(',')
    .map((token) => token.trim())
    .filter((token) => token.length > 0);
  if (raw.length === 0) {
    return ['ANY'];
  }
  if (raw.some((token) => token.toUpperCase() === 'ANY')) {
    return ['ANY'];
  }
  return raw;
}

// Serialize a token array back to the stored form.
// No specific tokens (or ANY selected) -> "ANY"; otherwise comma-joined specifics.
export function serializeTokens(tokens) {
  const specifics = (tokens || []).filter((token) => token && token.toUpperCase() !== 'ANY');
  if (specifics.length === 0) {
    return 'ANY';
  }
  return specifics.join(',');
}

// Toggle a token within a selection following the ANY/specific exclusivity rules.
export function toggleToken(tokens, token) {
  const current = parseTokens(serializeTokens(tokens));
  if (token.toUpperCase() === 'ANY') {
    return ['ANY'];
  }
  const withoutAny = current.filter((value) => value.toUpperCase() !== 'ANY');
  if (withoutAny.includes(token)) {
    const next = withoutAny.filter((value) => value !== token);
    return next.length === 0 ? ['ANY'] : next;
  }
  return [...withoutAny, token];
}

// Build the full ordered chip list for a field: presets first (ANY first),
// then any unknown tokens stored on the row so engine data is never dropped.
export function chipOptions(presets, selectedTokens) {
  const options = [...presets];
  (selectedTokens || []).forEach((token) => {
    if (token.toUpperCase() !== 'ANY' && !options.includes(token)) {
      options.push(token);
    }
  });
  return options;
}

export function isAnySelected(tokens) {
  const parsed = parseTokens(serializeTokens(tokens));
  return parsed.length === 1 && parsed[0].toUpperCase() === 'ANY';
}

export function isRestricted(value) {
  return serializeTokens(parseTokens(value)) !== 'ANY';
}

// Live multiplier hint. Returns { text, tone } where tone is teal | amber | muted.
export function premiumHint(value, locale) {
  const premium = Number(value);
  if (!Number.isFinite(premium)) {
    return { text: '', tone: 'muted' };
  }
  const deltaPct = Math.round((premium - 1) * 100);
  if (deltaPct === 0) {
    return { text: pageText(locale, 'rate card', 'מחיר מחירון'), tone: 'muted' };
  }
  const sign = deltaPct > 0 ? '+' : '−';
  return {
    text: `${sign}${Math.abs(deltaPct)}%`,
    tone: deltaPct > 0 ? 'teal' : 'amber',
  };
}

// Suggest the next advertiser id by scanning existing ADV_## ids.
export function suggestNextId(advertisers) {
  let max = 0;
  let sawPattern = false;
  (advertisers || []).forEach((row) => {
    const match = /^ADV_(\d+)$/i.exec(String(row.advertiser_id || ''));
    if (match) {
      sawPattern = true;
      const num = Number(match[1]);
      if (num > max) {
        max = num;
      }
    }
  });
  if (!sawPattern) {
    return 'ADV_01';
  }
  const next = max + 1;
  return `ADV_${String(next).padStart(2, '0')}`;
}

// Stable comparison of two advertiser rows on the editable fields only.
const EDITABLE_FIELDS = ['default_premium', 'allow_positions', 'allow_genres', 'prime_time_only', 'notes'];

export function isDirty(original, draft) {
  if (!original || !draft) {
    return false;
  }
  return EDITABLE_FIELDS.some((field) => {
    if (field === 'default_premium') {
      return Number(original[field] ?? 0) !== Number(draft[field] ?? 0);
    }
    if (field === 'prime_time_only') {
      return Boolean(original[field]) !== Boolean(draft[field]);
    }
    // Normalize constraint fields so "1,2" === "1, 2" and "" === "ANY".
    if (field === 'allow_positions' || field === 'allow_genres') {
      return serializeTokens(parseTokens(original[field])) !== serializeTokens(parseTokens(draft[field]));
    }
    return String(original[field] ?? '') !== String(draft[field] ?? '');
  });
}

// Build the PUT payload for a draft row (editable fields only).
export function toPayload(draft) {
  return {
    default_premium: Number(draft.default_premium ?? 0),
    allow_positions: serializeTokens(parseTokens(draft.allow_positions)),
    allow_genres: serializeTokens(parseTokens(draft.allow_genres)),
    prime_time_only: Boolean(draft.prime_time_only),
    notes: draft.notes ?? '',
  };
}

// Apply search + active filter to the advertiser list.
export function filterAdvertisers(advertisers, { search, filter }) {
  const term = (search || '').trim().toLowerCase();
  return (advertisers || []).filter((row) => {
    if (term) {
      const haystack = `${row.advertiser_id || ''} ${row.notes || ''}`.toLowerCase();
      if (!haystack.includes(term)) {
        return false;
      }
    }
    if (filter === 'premium') {
      return Number(row.default_premium ?? 1) !== 1;
    }
    if (filter === 'prime') {
      return Boolean(row.prime_time_only);
    }
    if (filter === 'restricted') {
      return isRestricted(row.allow_positions) || isRestricted(row.allow_genres);
    }
    return true;
  });
}

export function sortAdvertisers(advertisers, sortKey) {
  const rows = [...(advertisers || [])];
  if (sortKey === 'premium-desc') {
    return rows.sort((a, b) => Number(b.default_premium ?? 1) - Number(a.default_premium ?? 1));
  }
  if (sortKey === 'premium-asc') {
    return rows.sort((a, b) => Number(a.default_premium ?? 1) - Number(b.default_premium ?? 1));
  }
  return rows.sort((a, b) => String(a.advertiser_id || '').localeCompare(String(b.advertiser_id || '')));
}

export function computeSummary(advertisers) {
  const rows = advertisers || [];
  return {
    total: rows.length,
    custom: rows.filter((row) => Number(row.default_premium ?? 1) !== 1).length,
    prime: rows.filter((row) => Boolean(row.prime_time_only)).length,
    restricted: rows.filter(
      (row) => isRestricted(row.allow_positions) || isRestricted(row.allow_genres),
    ).length,
  };
}
