import { isolate } from '../shell/bidi';
// Pure helpers for the Advertisers management page.
// Kept framework-free so they are trivially testable and reusable.

// Fallback presets used only until the backend /api/advertisers/options list
// loads. The trade's positions inside a break are 1, 2, 3, 4, 5 and L, where L
// is LAST and is its own position rather than the fifth ordinal. "gold" is the
// premium gold break (Hebrew: ברייק זהב), a break tag the rule engine accepts in
// this dimension. Real vocabularies (genres, programmes, dayparts) come from the
// options endpoint at runtime.
export const POSITION_PRESETS = ['ANY', '1', '2', '3', '4', '5', 'L', 'gold'];
export const GENRE_PRESETS = ['ANY'];

export const CONDITION_EFFECTS = ['premium', 'require', 'forbid', 'pressure'];

// How a premium rule's value is read by the engine (mirrors advertiser_rules.py).
// premium_discount is a percent 0..100 taken off the premium surcharge only:
// final_premium = 1 + (final_premium_before_rule - 1) * (1 - value/100). It can
// never push a premium below 1.0 or above its pre-discount value.
export const PREMIUM_MODES = ['multiplier', 'percent', 'cpp_absolute', 'cpp_add', 'cpp_discount', 'premium_discount'];

// Weekday scope chips in Hebrew week order (Sunday first), mapped to ISO weekday
// numbers per the conditions contract: Monday=1 .. Sunday=7, so Hebrew שבת is 6.
export const WEEKDAY_OPTIONS = [
  { value: '7', he: 'א', en: 'Sun' },
  { value: '1', he: 'ב', en: 'Mon' },
  { value: '2', he: 'ג', en: 'Tue' },
  { value: '3', he: 'ד', en: 'Wed' },
  { value: '4', he: 'ה', en: 'Thu' },
  { value: '5', he: 'ו', en: 'Fri' },
  { value: '6', he: 'שבת', en: 'Sat' },
];

// Normalize a stored weekday scope to a stable form: "ANY", or known ISO tokens
// sorted ascending with unknown tokens preserved after them (engine data is
// never dropped). A missing column reads as ANY, like the other scopes.
export function normalizeWeekdayScope(value) {
  const tokens = parseTokens(value);
  if (isAnySelected(tokens)) {
    return 'ANY';
  }
  const known = tokens.filter((token) => /^[1-7]$/.test(token)).sort();
  const unknown = tokens.filter((token) => !/^[1-7]$/.test(token));
  return serializeTokens([...known, ...unknown]);
}

// Normalize a stored/incoming mode to one of PREMIUM_MODES; default 'multiplier'
// so a legacy rule (no mode) reads exactly as before.
export function normalizeMode(value) {
  const text = String(value || '').trim().toLowerCase();
  return PREMIUM_MODES.includes(text) ? text : 'multiplier';
}

export const EMPTY_ADVERTISER = {
  advertiser_id: '',
  display_name: '',
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

export function isAnySelected(tokens) {
  const parsed = parseTokens(serializeTokens(tokens));
  return parsed.length === 1 && parsed[0].toUpperCase() === 'ANY';
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
const EDITABLE_FIELDS = ['display_name', 'default_premium', 'allow_positions', 'allow_genres', 'prime_time_only', 'urgency_k', 'ahead_k', 'notes'];

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
    display_name: String(draft.display_name ?? '').trim(),
    default_premium: Number(draft.default_premium ?? 0),
    allow_positions: serializeTokens(parseTokens(draft.allow_positions)),
    allow_genres: serializeTokens(parseTokens(draft.allow_genres)),
    prime_time_only: Boolean(draft.prime_time_only),
    ...pacingField(draft, 'urgency_k'),
    ...pacingField(draft, 'ahead_k'),
    notes: draft.notes ?? '',
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
    scope_weekdays: normalizeWeekdayScope(source.scope_weekdays),
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
    scope_weekdays: normalizeWeekdayScope(source.scope_weekdays),
    effect,
    mode: effect === 'premium' ? normalizeMode(source.mode) : 'multiplier',
    value: usesValue ? Number(source.value ?? 1) : 1,
    notes: source.notes ?? '',
  };
}

const CONDITION_FIELDS = [
  'scope_positions', 'scope_genres', 'scope_dayparts', 'scope_programmes', 'scope_weekdays',
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
    // Weekdays compare order-blind (sorted) so re-toggling the same days is clean.
    if (field === 'scope_weekdays') {
      return normalizeWeekdayScope(original[field]) !== normalizeWeekdayScope(draft[field]);
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
    scope_weekdays: 'ANY',
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
  if (normalized === 'premium_discount') {
    if (amount < 0 || amount > 100) {
      return { text: pageText(locale, 'must be 0-100', 'הערך חייב להיות בין 0 ל-100'), tone: 'amber' };
    }
    if (amount === 0) {
      return { text: pageText(locale, 'no discount', 'ללא הנחה'), tone: 'muted' };
    }
    // isolate keeps the signed percent one run, so the sign is not shuffled
    // to the wrong side of the digits in RTL.
    return { text: pageText(locale, `surcharge −${amount}%`, `${isolate(`−${amount}%`)} מהתוספת בלבד`), tone: 'amber' };
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
  // The Hebrew string isolates the signed percent as one run, so the sign is
  // not bidi-shuffled to the wrong side of the digits in RTL.
  return {
    text: pageText(locale, `${sign}${Math.abs(amount)}% placement only`, `${isolate(`${sign}${Math.abs(amount)}%`)} שיבוץ בלבד`),
    tone: 'muted',
  };
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

// Pull the human-readable message straight from a backend finding. The client
// never derives the semantics a second time: prefer message, then detail, then
// a plain fallback that just names the involved rule ids.
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
