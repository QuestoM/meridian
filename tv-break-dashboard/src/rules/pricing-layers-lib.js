import { pageText } from '../shell/surface-helpers';

// Shared vocabulary and readers for the Pricing page modules (PricingManager,
// PricingSlotTester, PricingEventsLayer). Split out so each module stays small.

// Maps a layer's display name to the YAML override key the engine reads.
export const LAYER_TO_YAML = {
  program: 'program_type',
  day: 'day_of_week',
  show: 'show',
  position: 'position_in_break',
  ad_type: 'ad_type',
};

export const DAY_NAMES = {
  1: ['Mon', 'שני'], 2: ['Tue', 'שלישי'], 3: ['Wed', 'רביעי'], 4: ['Thu', 'חמישי'],
  5: ['Fri', 'שישי'], 6: ['Sat', 'שבת'], 7: ['Sun', 'ראשון'],
};

// The Israeli week, Sunday first, written as the ISO weekday keys the rate card
// is stored under. The store is ISO and stays ISO, where 1 is Monday and 7 is
// Sunday; a plain object iterates integer-like keys in ascending numeric order,
// which is what read the operator's own week back to them Monday first. Only the
// reading order changes here. The keys, the values and the saved payload do not.
export const DAY_ORDER = ['7', '1', '2', '3', '4', '5', '6'];

// Position-in-break keys are internal engine keys; these are the human labels.
export const POSITION_NAMES = {
  1: ['First', 'ראשון'], 2: ['Second', 'שני'], 3: ['Third', 'שלישי'],
  default_middle: ['Middle default', 'אמצע (ברירת מחדל)'], last: ['Last', 'אחרון'],
};

// Bilingual titles + Hebrew descriptions for the stable engine layers; the API text is English-only.
export const LAYER_TEXT = {
  program: { en: 'Programme type', he: 'סוג תוכנית', descHe: 'פרמיית מחלקת תוכנית (חדשות, תוכניות פריים, אחר). חלה תמיד.' },
  day: { en: 'Day of week', he: 'יום בשבוע', descHe: 'פרמיית יום בשבוע. חלה תמיד.' },
  show: { en: 'Specific show', he: 'תוכנית ספציפית', descHe: 'פרמיה לתוכנית ספציפית (למשל האח הגדול). נערמת על מחלקת התוכנית.' },
  position: { en: 'Position in break', he: 'מיקום בברייק', descHe: 'פרמיית מיקום בברייק (ראשון, שני, אחרון). כבויה עד להפעלה.' },
  ad_type: { en: 'Ad type', he: 'סוג פרסומת', descHe: 'פרמיית סוג פרסומת (פרסומת, חסות, פרומו). כבויה עד להפעלה.' },
  events: { en: 'Calendar events', he: 'אירועי לוח שנה', descHe: 'מכפיל תמחור לימים שבתוך אירועים פעילים מלוח האירועים. הצהרת מפעיל, לא מדידה. כבויה עד להפעלה.' },
  event: { en: 'Calendar event', he: 'אירוע לוח שנה' },
  // Override target layers the price-slot tester can name beyond the stable set.
  prime: { en: 'Prime time', he: 'פריים טיים' },
  final: { en: 'Final price', he: 'המחיר הסופי' },
};

export const layerLabel = (name, locale) =>
  (LAYER_TEXT[name] ? pageText(locale, LAYER_TEXT[name].en, LAYER_TEXT[name].he) : name.charAt(0).toUpperCase() + name.slice(1));

// Humanizes a multiplier's provenance ("rate_card" / "override:<rule_id>" / the
// events calendar): the operator reads plain words, never the raw tag.
export function sourceLabel(source, locale) {
  if (source === 'rate_card') return pageText(locale, 'rate card', 'כרטיס תעריפים');
  if (typeof source === 'string' && source.startsWith('override:')) {
    return pageText(locale, `override ${source.slice(9)}`, `עקיפה ${source.slice(9)}`);
  }
  if (typeof source === 'string' && (source === 'operator_event' || source === 'events' || source.startsWith('event'))) {
    return pageText(locale, 'events calendar, operator assertion', 'לוח אירועים, הצהרת מפעיל');
  }
  return pageText(locale, 'rate card', 'כרטיס תעריפים');
}

// Turns a layer's raw key into a human, bilingual label so no snake_case key
// reaches the operator. Day keys are ISO weekdays; position keys are named engine
// slots; program, show and ad-type keys are already human and pass through.
export function keyLabel(layerName, key, locale) {
  if (layerName === 'day' && DAY_NAMES[key]) {
    return pageText(locale, DAY_NAMES[key][0], DAY_NAMES[key][1]);
  }
  if (layerName === 'position' && POSITION_NAMES[key]) {
    return pageText(locale, POSITION_NAMES[key][0], POSITION_NAMES[key][1]);
  }
  return String(key);
}

// One premium layer's keys and multipliers, in the order a person reads them.
// Only the day layer has a reading order of its own; every other layer keeps the
// order the server sent. A day key the order does not name is kept and appended
// rather than dropped, so a rate card carrying something new still shows it.
export function layerEntries(layer) {
  const entries = Object.entries((layer && layer.values) || {});
  if (!layer || layer.name !== 'day') return entries;
  const named = new Map(entries.map(([key, value]) => [String(key), value]));
  const ordered = DAY_ORDER.filter((key) => named.has(key)).map((key) => [key, named.get(key)]);
  const rest = entries.filter(([key]) => !DAY_ORDER.includes(String(key)));
  return [...ordered, ...rest];
}

// Normalizes one per-event entry from the pricing payload's events list. Only
// entries with a usable non-1.0 multiplier are kept; names and dates pass
// through as the server sent them, or null when absent (rendered as absent).
function normalizeEventEntry(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const multiplierRaw = raw.price_multiplier ?? raw.multiplier;
  const multiplier = Number(multiplierRaw);
  if (!Number.isFinite(multiplier) || multiplier === 1) return null;
  const name = raw.name ?? raw.title ?? raw.event_name ?? null;
  return {
    name: name === null || name === undefined ? null : String(name),
    start: raw.start_date ?? raw.start ?? null,
    end: raw.end_date ?? raw.end ?? null,
    multiplier,
  };
}

// Reads the events-layer state from the GET /api/pricing payload under the
// contract's plausible shapes (a "events" entry in layers, a dedicated object,
// or only the activation flag). supported=false means this server predates the
// events layer; the UI then shows exactly that instead of faking an off state.
// count is null when the server did not report it, never invented. events is
// the per-event list (name, dates, multiplier) when the server carries one,
// else null so the UI states plainly that the server sent only a count.
export function readEventsLayer(state) {
  if (!state || typeof state !== 'object') return { supported: false, enabled: false, count: null, events: null };
  const fromLayers = Array.isArray(state.layers)
    ? state.layers.find((layer) => layer && (layer.name === 'events' || layer.name === 'event'))
    : null;
  const direct = [state.events_layer, state.events].find((value) => value && typeof value === 'object' && !Array.isArray(value)) || null;
  const activation = state.activation && typeof state.activation === 'object' ? state.activation : {};
  const source = fromLayers || direct;
  if (!source && !('events' in activation)) return { supported: false, enabled: false, count: null, events: null };
  const enabled = Boolean(source ? (source.enabled ?? source.active ?? activation.events) : activation.events);
  const countCandidates = source ? [source.active_event_count, source.active_events, source.event_count, source.count] : [];
  const countRaw = countCandidates.find((value) => value !== undefined && value !== null && !Array.isArray(value));
  const count = Number.isFinite(Number(countRaw)) && countRaw !== undefined && countRaw !== null ? Number(countRaw) : null;
  const listRaw = source ? [source.events, source.active_events, source.active_event_list, source.event_list, source.list].find(Array.isArray) : undefined;
  const events = Array.isArray(listRaw) ? listRaw.map(normalizeEventEntry).filter(Boolean) : null;
  return { supported: true, enabled, count, events };
}

// One human date-span line for an event: "start - end", open-ended when the
// server sent no end date. Dates render as sent (ISO), direction-isolated.
export function eventDatesLabel(entry, locale) {
  const start = entry && entry.start ? String(entry.start) : '';
  const end = entry && entry.end ? String(entry.end) : '';
  if (start && end) return `${start} - ${end}`;
  if (start) return pageText(locale, `from ${start}, open-ended`, `מ-${start}, ללא תאריך סיום`);
  return pageText(locale, 'no dates reported', 'לא דווחו תאריכים');
}
