import { pageText } from '../shell/surface-helpers';
import { formatSpan } from '../shell/dates';

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

// The trade's positions inside a break are 1, 2, 3, 4, 5 and L, where L is LAST
// and is its own position rather than the fifth ordinal. These are the human
// labels for those engine keys; "last" is the rate card's legacy key for L and
// still reads, so a card saved before the rename still labels correctly.
export const POSITION_NAMES = {
  1: ['First', 'ראשון'], 2: ['Second', 'שני'], 3: ['Third', 'שלישי'],
  4: ['Fourth', 'רביעי'], 5: ['Fifth', 'חמישי'],
  L: ['Last (L)', 'אחרון (L)'],
  default_middle: ['Middle default', 'אמצע (ברירת מחדל)'], last: ['Last (L)', 'אחרון (L)'],
};

// Bilingual titles and descriptions for the stable engine layers, both halves in
// one entry. The description used to be a half from here and a half from the
// API: `descHe` for a Hebrew reader, the API's own English-only `description`
// for everybody else, so a layer this table did not name printed English prose
// under a Hebrew heading, and the two languages could say different things about
// the same layer. Each `descEn` is the API's own sentence, kept byte-for-byte so
// no English screen moved, and `layerDescription` refuses to take one half from
// here and the other from the wire.
export const LAYER_TEXT = {
  program: { en: 'Programme type', he: 'סוג תוכנית', descHe: 'פרמיית מחלקת תוכנית (חדשות, תוכניות פריים, אחר). חלה תמיד.', descEn: 'Program-class premium (News, prime shows, other). Always applied.' },
  day: { en: 'Day of week', he: 'יום בשבוע', descHe: 'פרמיית יום בשבוע. חלה תמיד.', descEn: 'Day-of-week premium. Always applied.' },
  show: { en: 'Specific show', he: 'תוכנית ספציפית', descHe: 'פרמיה לתוכנית ספציפית (למשל האח הגדול). נערמת על מחלקת התוכנית.', descEn: 'Per-show premium (for example Big Brother). Stacks on the program class.' },
  position: { en: 'Position in break', he: 'מיקום בברייק', descHe: 'פרמיית מיקום בברייק (1 עד 5 ו-L לאחרון). כבויה עד להפעלה.', descEn: 'Position-in-break premium (1 to 5 and L for last). Off until activated.' },
  ad_type: { en: 'Ad type', he: 'סוג פרסומת', descHe: 'פרמיית סוג פרסומת (פרסומת, חסות, פרומו). כבויה עד להפעלה.', descEn: 'Ad-type premium (commercial, sponsorship, promo). Off until activated.' },
  events: { en: 'Calendar events', he: 'אירועי לוח שנה', descHe: 'מכפיל תמחור לימים שבתוך אירועים פעילים מלוח האירועים. הצהרת מפעיל, לא מדידה. כבויה עד להפעלה.', descEn: 'A price multiplier for days inside an active event on the events calendar. An operator assertion, not a measurement. Off until activated.' },
  event: { en: 'Calendar event', he: 'אירוע לוח שנה' },
  // Override target layers the price-slot tester can name beyond the stable set.
  prime: { en: 'Prime time', he: 'פריים טיים' },
  final: { en: 'Final price', he: 'המחיר הסופי' },
};

export const layerLabel = (name, locale) =>
  (LAYER_TEXT[name] ? pageText(locale, LAYER_TEXT[name].en, LAYER_TEXT[name].he) : name.charAt(0).toUpperCase() + name.slice(1));

// One layer's description, both halves from one entry or neither. A layer this
// table does not name falls back to the server's own words for both readers,
// which is the producer's sentence unaltered rather than one language for one
// reader and another language for the other.
export function layerDescription(layer, locale) {
  const entry = LAYER_TEXT[(layer || {}).name] || {};
  if (entry.descEn && entry.descHe) return pageText(locale, entry.descEn, entry.descHe);
  return String((layer || {}).description || '');
}

// What each priced category IS, in the reader's own language.
//
// The rate card's keys are config, and config is written once in one language:
// `program_type` is keyed News / PrimeShow1 / PrimeShow2 / Other and `ad_type`
// is keyed פרסומת / חסות / פרומו, both straight out of
// config/optimization_weights.yaml. Printed raw, each language got the half that
// is not its own: the English card listed three Hebrew words down its ad-type
// column, the Hebrew card listed four Latin ones down its programme column, and
// the zero-multiplier warning put a Hebrew category inside an English sentence.
// One entry per category, both halves in it, and the rows and the warning read
// the same one, so a category cannot be named two ways on one screen.
//
// The `show` layer is deliberately absent: its keys are programme titles, which
// are proper nouns like a channel name and are not translated in either
// direction. They pass through, which is what `keyLabel` does with any key no
// table names.
export const CATEGORY_TEXT = {
  program: {
    News: ['News', 'חדשות'],
    PrimeShow1: ['First prime show', 'תוכנית פריים ראשונה'],
    PrimeShow2: ['Second prime show', 'תוכנית פריים שנייה'],
    Other: ['Other', 'אחר'],
  },
  ad_type: {
    'פרסומת': ['Commercial', 'פרסומת'],
    'חסות': ['Sponsorship', 'חסות'],
    'פרומו': ['Promo', 'פרומו'],
  },
};

// A comma-joined run of categories, every one of them read through the table
// above, so a sentence that names a set of them stays in one language.
export function categoryList(layerName, categories, locale) {
  return (Array.isArray(categories) ? categories : [])
    .map((key) => keyLabel(layerName, key, locale))
    .join(', ');
}

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

// Turns a layer's raw key into a human, bilingual label so neither a snake_case
// key nor a config word in the other language reaches the operator. Day keys are
// ISO weekdays, position keys are named engine slots, and programme-class and
// ad-type keys are config words in whichever language the file was written in.
// A key no table names passes through, which is what a programme title needs.
export function keyLabel(layerName, key, locale) {
  if (layerName === 'day' && DAY_NAMES[key]) {
    return pageText(locale, DAY_NAMES[key][0], DAY_NAMES[key][1]);
  }
  if (layerName === 'position' && POSITION_NAMES[key]) {
    return pageText(locale, POSITION_NAMES[key][0], POSITION_NAMES[key][1]);
  }
  const named = (CATEGORY_TEXT[layerName] || {})[key];
  if (named) return pageText(locale, named[0], named[1]);
  return String(key);
}

// One premium layer's keys and multipliers, in the order a person reads them.
// Only the day layer has a reading order of its own; every other layer keeps the
// order the server sent. A day key the order does not name is kept and appended
// rather than dropped, so a rate card carrying something new still shows it.
//
// A layer that ships a vocabulary (the position layer) is read in that order and
// includes the keys nobody has priced yet, with a null value. Position 4 and
// position 5 exist in the trade and are unset on the shipped card, so they are
// shown as unset and settable rather than hidden or faked as a premium of 1.
export function layerEntries(layer) {
  const entries = Object.entries((layer && layer.values) || {});
  if (layer && Array.isArray(layer.vocabulary) && layer.vocabulary.length > 0) {
    const named = new Map(entries.map(([key, value]) => [String(key), value]));
    const ordered = layer.vocabulary.map((entry) => [String(entry.key), named.has(String(entry.key)) ? named.get(String(entry.key)) : null]);
    const seen = new Set(layer.vocabulary.map((entry) => String(entry.key)));
    const rest = entries.filter(([key]) => !seen.has(String(key)));
    return [...ordered, ...rest];
  }
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

// One date-span line for an event, open-ended when the server sent no end date.
// The shape and the open-ended wording both come from shell/dates.js, so an
// event window and a campaign flight read the same way.
export function eventDatesLabel(entry, locale) {
  return formatSpan(entry && entry.start, entry && entry.end, locale);
}
