// The words this workspace reads back in place of the engine's own keys.
//
// Split out of `rules-lib.js` under the file-size law, and it is a real seam
// rather than a filing convenience: everything here is a lookup from a stored
// value to a sentence a person reads, with no fetch and no state, so a surface
// that only needs a label does not pull the whole API client in with it.
// `rules-lib.js` re-exports every name below, so no importer changed.

import { WALLS } from '../session.js';

// The store's effect keys, and the words a person reads instead of them. The
// engine writes `fix_offset`; nobody on this surface has to read that. One table
// serves the builder's own picker and every list that reads a saved row back, so
// the two cannot say different words about the same value.
export const EFFECT_LIST = [
  { value: 'FIX_OFFSET', label_en: 'Fix offset', label_he: 'היסט קבוע' },
  { value: 'OFFSET_WINDOW', label_en: 'Offset window', label_he: 'חלון היסט' },
  { value: 'PIN_COUNT', label_en: 'Pin count', label_he: 'מספר ברייקים קבוע' },
  { value: 'DURATION_RANGE', label_en: 'Duration range', label_he: 'טווח אורך' },
  { value: 'GOLD', label_en: 'Gold break', label_he: 'ברייק זהב' },
  { value: 'FORBID', label_en: 'Forbid', label_he: 'איסור' },
];

// A stored effect is lowercase and a locally created one is uppercase, so the
// lookup is case-insensitive and server-loaded rows read exactly like fresh
// ones. An effect this table does not know is spaced out and capitalized rather
// than guessed at, which is honest and still never renders a raw key.
export function effectLabel(effect, locale) {
  const key = String(effect || '').trim().toUpperCase();
  const match = EFFECT_LIST.find((entry) => entry.value === key);
  if (match) return locale === 'he' ? match.label_he : match.label_en;
  return key.replace(/_/g, ' ').toLowerCase().replace(/^./, (letter) => letter.toUpperCase());
}

// A refusal in the reader's own language. The server sends its detail in Hebrew,
// because that is the language the product's operators work in and the wall has
// exactly one string per rule. On an English page that string is a paragraph the
// reader has to translate, so the English sentence for each wall this workspace
// throws is written here and keyed off the frozen wall's own detail, which means
// the two cannot drift: change the Hebrew and the key stops matching, and the
// fallback below returns the server's own words rather than a stale translation.
const REFUSALS_EN = new Map([
  [WALLS.guardrails.detail, 'Only an administrator changes the regulatory limits.'],
  [WALLS.audienceActivation.detail, 'Only company staff throw the audience model switch.'],
  [WALLS.events.detail, 'Only company staff edit events.'],
  [WALLS.eventPricing.detail, 'Only company staff turn event pricing on.'],
  [WALLS.companySurface.detail, 'This view is for company staff.'],
  [WALLS.readOnlyRole.detail, 'A viewing account has no edit permission.'],
  ['בחירת ערוץ המפעיל שמורה למנהל המערכת', 'Only an administrator changes the operator channel.'],
]);

export function refusalSentence(reason, locale) {
  const text = String(reason || '').trim();
  if (!text || locale === 'he') return text;
  return REFUSALS_EN.get(text) || text;
}

// The four regulatory limits, named the way the person accountable for them
// says them. The store keys are engine words and a compliance owner never has
// to read one: the limit list, the change log and the attestation all read
// through this table.
const LIMIT_WORDS = {
  max_ad_minutes_per_hour: ['Ad minutes per broadcast hour', 'דקות פרסום לשעת שידור'],
  max_breaks_per_hour: ['Breaks per hour', 'ברייקים בשעה'],
  min_break_spacing_minutes: ['Minimum spacing between breaks', 'מרווח מינימלי בין ברייקים'],
  protected_program_max_ad_minutes_per_hour: ['Ad minutes per hour in protected content', 'דקות פרסום לשעה בתוכן מוגן'],
};

export function limitLabel(key, locale) {
  const words = LIMIT_WORDS[key];
  if (!words) return String(key || '');
  return locale === 'he' ? words[1] : words[0];
}

// The compliance payload states each check's unit in English, because it is an
// engine payload. A unit is a word a person reads, so it is read back in the
// page's own language. An unknown unit is returned as it came rather than
// guessed at, so a new check shows an untranslated unit and not a wrong one.
const UNIT_WORDS = {
  'minutes/hour': 'דקות לשעה',
  'breaks/hour': 'ברייקים לשעה',
  'minutes/day': 'דקות ליום',
  'breaks/day': 'ברייקים ליום',
  minutes: 'דקות',
  breaks: 'ברייקים',
  '%': '%',
};

export function unitLabel(unit, locale) {
  const text = String(unit || '');
  if (locale !== 'he') return text;
  return UNIT_WORDS[text] || text;
}
