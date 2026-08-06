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

// How many programmes matched beyond the ones on screen, and the act that
// reaches them. The same rule the night picker answers to: a list that shows
// fewer things than matched has to say so and has to name the way to the rest,
// which for a type-ahead is the typing itself. Measured on the reference EPG,
// an empty query matches 106 programmes on the operator's channel.
export function moreProgrammesSentence(locale, hidden) {
  const rest = Number(hidden) || 0;
  if (locale === 'he') {
    if (rest === 1) return 'עוד תוכנית אחת תואמת. המשיכו להקליד כדי להגיע אליה.';
    return `עוד ${rest} תוכניות תואמות. המשיכו להקליד כדי לצמצם את הרשימה.`;
  }
  if (rest === 1) return '1 more programme matches. Keep typing to reach it.';
  return `${rest} more programmes match. Keep typing to narrow the list.`;
}

// How many recorded licence changes have not yet reached their effective date.
// A compliance owner reads this beside the change log, so it has to hold up
// under the same scrutiny: a scheduled change is the normal case right after a
// revision is filed, not the rare one, so the sentence is written out for both
// counts rather than templated with a count that does not agree with its verb.
export function scheduledChangesSentence(locale, count) {
  const total = Number(count) || 0;
  if (locale === 'he') {
    if (total === 1) return 'שינוי אחד תועד לתאריך עתידי ואינו בתוקף עדיין.';
    return `${total} שינויים תועדו לתאריך עתידי ואינם בתוקף עדיין.`;
  }
  if (total === 1) return '1 change is recorded for a future date and is not in force yet.';
  return `${total} changes are recorded for a future date and are not in force yet.`;
}

// A rule the plan does not breach compiles to no row, so there is nothing to
// store and the save stays shut. Said plainly, and said about the night that was
// chosen rather than about the whole window, because the two are different facts
// and only one of them is true when a single night is named.
export function nothingToSaveSentence(locale, night) {
  if (!night) {
    if (locale === 'he') return 'שום דבר בחלון התוכנית אינו מפר את הכלל הזה, ולכן אין מה לשמור בשלב זה.';
    return 'Nothing in the plan window breaks this rule, so there is nothing to save yet.';
  }
  if (locale === 'he') return `אף שידור ביום ${night} אינו מפר את הכלל הזה במצב התוכנית הנוכחי, ולכן אין מה לשמור.`;
  return `No airing on ${night} breaks this rule as the plan stands, so there is nothing to save.`;
}

// And the way out of that, when there is one. The same sentence priced again
// with the night dropped says whether the run as a whole breaches it, so the
// dead end becomes the one rule that can be saved instead of a shut button.
export function widerScopeSentence(locale, breaching, matched) {
  const rules = Number(breaching) || 0;
  const all = Number(matched) || 0;
  if (locale === 'he') return `על פני כל הריצה, ${rules} מתוך ${all} שידורי התוכנית כן מפרים אותו.`;
  return `Across the whole run, ${rules} of the programme's ${all} airings do break it.`;
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
