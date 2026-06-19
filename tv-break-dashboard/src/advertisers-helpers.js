// Pure helpers for the Advertisers management page.
// Kept framework-free so they are trivially testable and reusable.

// Fallback presets used only until the backend /api/advertisers/options list
// loads. "gold" is the premium gold break (Hebrew: ברייק זהב). Real vocabularies
// (genres, programmes, dayparts) come from the options endpoint at runtime.
export const POSITION_PRESETS = ['ANY', 'first', 'middle', 'last', 'gold'];
export const GENRE_PRESETS = ['ANY'];
export const DAYPART_PRESETS = ['ANY', 'morning', 'noon', 'evening', 'prime', 'night'];

export const CONDITION_EFFECTS = ['premium', 'require', 'forbid', 'pressure'];

// How a premium rule's value is read by the engine (mirrors advertiser_rules.py).
export const PREMIUM_MODES = ['multiplier', 'percent', 'cpp_absolute', 'cpp_add', 'cpp_discount'];

// Normalize a stored/incoming mode to one of PREMIUM_MODES; default 'multiplier'
// so a legacy rule (no mode) reads exactly as before.
export function normalizeMode(value) {
  const text = String(value || '').trim().toLowerCase();
  return PREMIUM_MODES.includes(text) ? text : 'multiplier';
}

export const EMPTY_ADVERTISER = {
  advertiser_id: '',
  default_premium: 1,
  allow_positions: 'ANY',
  allow_genres: 'ANY',
  prime_time_only: false,
  // Per-advertiser delivery-pacing strength defaults. Empty string means "use the
  // channel-wide default" (the backend reads blank as None). urgency_k is how hard
  // this advertiser's behind-pace campaigns lean toward their inventory; ahead_k is
  // how hard over-delivered campaigns are leaned away. A per-campaign value in
  // campaign_flights.csv still overrides these.
  urgency_k: '',
  ahead_k: '',
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
const EDITABLE_FIELDS = ['default_premium', 'allow_positions', 'allow_genres', 'prime_time_only', 'urgency_k', 'ahead_k', 'notes'];

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
// Read an optional pacing-strength field into the payload shape the API expects.
// A blank or invalid value sends the matching clear flag so a PUT removes the
// override (falls back to the channel-wide default); a non-negative number sends
// the value. ``field`` is 'urgency_k' or 'ahead_k'. On create the clear flag is
// simply ignored by the backend, so the same shape is safe for POST.
function pacingField(draft, field) {
  const raw = draft[field];
  const text = raw === null || raw === undefined ? '' : String(raw).trim();
  const value = text === '' ? NaN : Number(text);
  if (text === '' || !Number.isFinite(value) || value < 0) {
    return { [`clear_${field}`]: true };
  }
  return { [field]: value };
}

export function toPayload(draft) {
  return {
    default_premium: Number(draft.default_premium ?? 0),
    allow_positions: serializeTokens(parseTokens(draft.allow_positions)),
    allow_genres: serializeTokens(parseTokens(draft.allow_genres)),
    prime_time_only: Boolean(draft.prime_time_only),
    ...pacingField(draft, 'urgency_k'),
    ...pacingField(draft, 'ahead_k'),
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

// ---------------------------------------------------------------------------
// Scoped conditions (per-advertiser rules) helpers.
//
// A condition is one scoped rule on an advertiser:
//   { rule_id, scope_positions, scope_genres, scope_dayparts, effect, value, notes }
// scope_* are comma-joined tokens or "ANY"; effect in {premium, require, forbid};
// value is a float multiplier used only when effect === "premium".
// ---------------------------------------------------------------------------

// Normalize the conditions array delivered with an advertiser row.
export function normalizeConditions(value) {
  return Array.isArray(value) ? value : [];
}

// Normalize the overlaps/conflicts findings delivered with an advertiser row.
export function normalizeOverlaps(value) {
  return Array.isArray(value) ? value : [];
}

// Build the editable client-side shape of a condition from the backend record.
// Unknown scope tokens are preserved (we never invent or drop daypart tokens).
export function parseCondition(condition) {
  const source = condition || {};
  const effect = CONDITION_EFFECTS.includes(source.effect) ? source.effect : 'premium';
  return {
    rule_id: source.rule_id ?? '',
    scope_positions: serializeTokens(parseTokens(source.scope_positions)),
    scope_genres: serializeTokens(parseTokens(source.scope_genres)),
    scope_dayparts: serializeTokens(parseTokens(source.scope_dayparts)),
    scope_programmes: serializeTokens(parseTokens(source.scope_programmes)),
    effect,
    mode: normalizeMode(source.mode),
    // Keep value sane: premium and pressure use it, but we always carry a number.
    value: Number.isFinite(Number(source.value)) ? Number(source.value) : 1,
    notes: source.notes ?? '',
  };
}

// Build the POST/PUT body for a condition draft. effect=premium carries value;
// require/forbid send value 1.0 (ignored by the engine) so the body stays uniform.
export function toConditionPayload(draft) {
  const source = draft || {};
  const effect = CONDITION_EFFECTS.includes(source.effect) ? source.effect : 'premium';
  // premium uses value+mode; pressure uses value (a percent); require/forbid send
  // value 1 (ignored by the engine) so the body stays uniform.
  const usesValue = effect === 'premium' || effect === 'pressure';
  return {
    scope_positions: serializeTokens(parseTokens(source.scope_positions)),
    scope_genres: serializeTokens(parseTokens(source.scope_genres)),
    scope_dayparts: serializeTokens(parseTokens(source.scope_dayparts)),
    scope_programmes: serializeTokens(parseTokens(source.scope_programmes)),
    effect,
    mode: effect === 'premium' ? normalizeMode(source.mode) : 'multiplier',
    value: usesValue ? Number(source.value ?? 1) : 1,
    notes: source.notes ?? '',
  };
}

const CONDITION_FIELDS = [
  'scope_positions', 'scope_genres', 'scope_dayparts', 'scope_programmes',
  'effect', 'mode', 'value', 'notes',
];

// True when a condition draft differs from its original (scope-normalized).
export function isConditionDirty(original, draft) {
  if (!original || !draft) {
    return false;
  }
  const valueEffect = (effect) => effect === 'premium' || effect === 'pressure';
  return CONDITION_FIELDS.some((field) => {
    if (field === 'value') {
      // Value matters for premium and pressure; for require/forbid it is inert.
      if (!valueEffect(draft.effect) && !valueEffect(original.effect)) {
        return false;
      }
      return Number(original.value ?? 1) !== Number(draft.value ?? 1);
    }
    if (field === 'mode') {
      // Mode only matters for a premium rule.
      if (draft.effect !== 'premium' && original.effect !== 'premium') {
        return false;
      }
      return normalizeMode(original.mode) !== normalizeMode(draft.mode);
    }
    if (field === 'effect') {
      return String(original.effect ?? '') !== String(draft.effect ?? '');
    }
    if (field === 'notes') {
      return String(original.notes ?? '') !== String(draft.notes ?? '');
    }
    // scope_* fields: compare normalized token form.
    return serializeTokens(parseTokens(original[field])) !== serializeTokens(parseTokens(draft[field]));
  });
}

// A blank condition draft for the "Add rule" affordance. Defaults to a +15%
// percent premium, the friendliest coefficient mode for a new rule.
export function emptyCondition() {
  return {
    rule_id: '',
    scope_positions: 'ANY',
    scope_genres: 'ANY',
    scope_dayparts: 'ANY',
    scope_programmes: 'ANY',
    effect: 'premium',
    mode: 'percent',
    value: 15,
    notes: '',
  };
}

// Live hint for the coefficient (premium) field, mode-aware. Returns
// { text, tone } where tone is teal | amber | muted.
export function coefficientHint(value, mode, locale) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) {
    return { text: '', tone: 'muted' };
  }
  const normalized = normalizeMode(mode);
  if (normalized === 'percent') {
    if (amount === 0) {
      return { text: pageText(locale, 'rate card', 'מחיר מחירון'), tone: 'muted' };
    }
    const sign = amount > 0 ? '+' : '−';
    return { text: `${sign}${Math.abs(amount)}%`, tone: amount > 0 ? 'teal' : 'amber' };
  }
  if (normalized === 'cpp_absolute') {
    return { text: pageText(locale, `CPP set to ${amount}`, `נקודה = ${amount}`), tone: 'teal' };
  }
  if (normalized === 'cpp_add') {
    return { text: pageText(locale, `CPP +${amount}`, `נקודה +${amount}`), tone: 'teal' };
  }
  if (normalized === 'cpp_discount') {
    return { text: pageText(locale, `CPP −${amount}`, `נקודה −${amount}`), tone: 'amber' };
  }
  // multiplier
  return premiumHint(amount, locale);
}

// Live hint for the pressure (placement preference) field. Pressure steers
// placement without ever appearing in revenue, so it is always informational.
export function pressureHint(value, locale) {
  const amount = Number(value);
  if (!Number.isFinite(amount) || amount === 0) {
    return { text: pageText(locale, 'no steer', 'ללא הטיה'), tone: 'muted' };
  }
  const sign = amount > 0 ? '+' : '−';
  return {
    text: pageText(locale, `${sign}${Math.abs(amount)}% placement only`, `${sign}${Math.abs(amount)}% שיבוץ בלבד`),
    tone: 'muted',
  };
}

// Collect every daypart token already present across an advertiser's conditions,
// so the chip selector can offer real tokens (ANY + observed) without inventing
// a fixed taxonomy. Unknown tokens are preserved and de-duplicated, ANY excluded.
export function collectDaypartTokens(conditions) {
  const seen = [];
  normalizeConditions(conditions).forEach((condition) => {
    parseTokens(condition && condition.scope_dayparts).forEach((token) => {
      if (token.toUpperCase() !== 'ANY' && !seen.includes(token)) {
        seen.push(token);
      }
    });
  });
  return seen;
}

// Map an overlap finding kind to a severity tone used for styling and ordering.
// conflict = strong warning; stacked_premium = informational; overlap = mild.
export function overlapTone(kind) {
  if (kind === 'conflict') {
    return 'conflict';
  }
  if (kind === 'stacked_premium') {
    return 'stacked';
  }
  return 'overlap';
}

// Pull the human-readable message straight from a backend finding. We do NOT
// recompute semantics on the client: prefer message, then detail, then a plain
// fallback that just names the involved rule ids.
export function overlapMessage(finding) {
  const source = finding || {};
  if (source.message) {
    return String(source.message);
  }
  if (source.detail) {
    return String(source.detail);
  }
  const ids = [source.rule_id_a, source.rule_id_b].filter(Boolean);
  if (ids.length > 0) {
    return ids.join(' / ');
  }
  return '';
}

// Count badge text for the collapsed scoped-rules header. Surfaces conflicts
// first (the thing an operator most needs to see), then rule count.
export function scopedRulesBadge(conditions, overlaps, locale) {
  const ruleCount = normalizeConditions(conditions).length;
  const conflictCount = normalizeOverlaps(overlaps).filter((finding) => finding && finding.kind === 'conflict').length;
  const parts = [];
  if (ruleCount > 0) {
    parts.push(pageText(locale, `${ruleCount} scoped ${ruleCount === 1 ? 'rule' : 'rules'}`, `${ruleCount} ${ruleCount === 1 ? 'כלל ממוקד' : 'כללים ממוקדים'}`));
  }
  if (conflictCount > 0) {
    parts.push(pageText(locale, `${conflictCount} ${conflictCount === 1 ? 'conflict' : 'conflicts'}`, `${conflictCount} ${conflictCount === 1 ? 'התנגשות' : 'התנגשויות'}`));
  }
  return parts;
}
