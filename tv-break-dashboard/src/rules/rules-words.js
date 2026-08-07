// The words this workspace reads back in place of the engine's own keys.
//
// Split out of `rules-lib.js` under the file-size law, and it is a real seam
// rather than a filing convenience: everything here is a lookup from a stored
// value to a sentence a person reads, with no fetch and no state, so a surface
// that only needs a label does not pull the whole API client in with it.
// `rules-lib.js` re-exports every name below, so no importer changed.

import { WALLS } from '../session.js';
import { isolate } from './rules-bidi.js';

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

// The refusal a walled payload carries, in the reader's own language, from the
// one call on the server that produced both halves together.
//
// `can_edit_reason` is one string and it is Hebrew, because the 403 detail and
// the reason a control renders before the click are the same string by
// contract. Measured on the shipped licence page in English: that string was
// printed verbatim above four English fields. The store now stamps
// `can_edit_reason_en` and `can_edit_reason_he` beside it from one entry, so the
// pair is the answer whenever it is there. The table above stays as the answer
// for an endpoint that has not been upgraded and for a gate the session decided
// on its own, and the server's own words stay the answer of last resort, so a
// wall nobody has translated is never printed as a blank.
export function refusalWords(payload, reason, locale) {
  const body = payload && typeof payload === 'object' ? payload : {};
  const pair = locale === 'he' ? body.can_edit_reason_he : body.can_edit_reason_en;
  const own = String(pair || '').trim();
  if (own && body.can_edit === false) return own;
  return refusalSentence(reason, locale);
}

// The reader's own half of a refusal a route raised. The rules routes send an
// object under `detail` carrying both halves and the code that produced them;
// every other route on this product sends one string, which is kept as the
// answer for both readers rather than guessed at.
export function detailWords(error, locale) {
  const words = error && typeof error === 'object' ? error.words : null;
  if (words && typeof words === 'object') {
    const own = String((locale === 'he' ? words.he : words.en) || '').trim();
    if (own) return own;
  }
  return String((error && error.message) || '').trim();
}

// What this surface will not send, said before it is sent. The route that
// answers a rejected licence change is frozen and forwards the English half of
// the server's refusal alone, so the one case a person can actually reach by
// typing is refused here instead, in the reader's own language, off the same
// bounds the server validates against and the same limit names the list above
// prints. Returns nothing at all when the value is one the licence can hold.
export function limitBoundsRefusal(locale, key, value, bounds) {
  const rule = (bounds || {})[key];
  if (!rule) return '';
  const number = Number(value);
  if (String(value).trim() === '' || !Number.isFinite(number)) {
    return locale === 'he'
      ? `${limitLabel(key, 'he')}: הערך חייב להיות מספר.`
      : `${limitLabel(key, 'en')}: the value has to be a number.`;
  }
  if (number >= Number(rule.min) && number <= Number(rule.max)) return '';
  return locale === 'he'
    ? `${limitLabel(key, 'he')}: הערך חייב להיות בין ${isolate(rule.min)} ל-${isolate(rule.max)}.`
    : `${limitLabel(key, 'en')}: the value has to be between ${rule.min} and ${rule.max}.`;
}

// Why a figure panel has no figure, in the reader's own language.
//
// The two panels on this surface that state an empty basis, the rate-card
// effect and the worth of a second, read `reason` off endpoints outside this
// piece's row (`/api/pricing/effect` and `/api/yield-per-second`), and those
// author it in English alone. Measured: a Hebrew reader with no saved plan got
// `No saved weekly schedule with segment ids on disk.` under a Hebrew heading.
// The sentence cannot be re-authored where it is produced, so the Hebrew half is
// authored here keyed off the producer's own English, exactly as the wall
// refusals above are: change the English and the key stops matching, and the
// fallback returns the producer's own words rather than a stale translation.
const BASIS_REASONS_HE = new Map([
  ['No saved weekly plan with segment ids is on disk to price.', 'אין בדיסק תוכנית שבועית שמורה עם מזהי מקטעים לתמחר.'],
  ['No saved weekly schedule with segment ids on disk.', 'אין בדיסק לוח שבועי שמור עם מזהי מקטעים.'],
  ['No saved weekly schedule on disk.', 'אין בדיסק לוח שבועי שמור.'],
  ['The saved plan carries no rows for the declared operator channel.', 'בתוכנית השמורה אין שורות לערוץ המפעיל שהוגדר.'],
  ['The saved plan carries no rows for the configured operator channel.', 'בתוכנית השמורה אין שורות לערוץ המפעיל שהוגדר.'],
  ['Plan segment rebuild failed; see the server log.', 'בניית מקטעי התוכנית מחדש נכשלה. ראו את יומן השרת.'],
  ['The optimization engine is unavailable.', 'מנוע האופטימיזציה אינו זמין.'],
  ['Saved plan no longer joins the EPG rebuild; recompute the schedule.', 'התוכנית השמורה כבר אינה מתחברת לבניית לוח השידורים מחדש. חשבו את הלוח מחדש.'],
  ['Saved schedule has no ad-seconds to monetize.', 'ללוח השמור אין שניות פרסום להפיק מהן הכנסה.'],
  ['Band computation failed; see the server log.', 'חישוב הרצועה נכשל. ראו את יומן השרת.'],
]);

export function basisReason(reason, locale) {
  const text = String(reason || '').trim();
  if (!text || locale !== 'he') return text;
  return BASIS_REASONS_HE.get(text) || text;
}

// Money on this surface is never compacted. The shell formatter switches to
// compact notation above 100,000, which would render a before of 1,067,846 and
// an after of 1,030,969 as the same two characters, and the whole point of this
// surface is that the difference between them is the decision. Full grouped
// digits, always, with the sign kept so a cost reads as a cost. It lives beside
// the words because it is one too: a stored value read back as display text.
export function money(value, locale) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '--';
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    maximumFractionDigits: 0,
    minimumFractionDigits: 0,
  }).format(number);
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

// The population line Today's ledger card and the licence page both carry, one
// sentence naming the channel every figure on the card is drawn from. The
// channel name is Hebrew even on the English page, because channel names are
// not translated, so an English sentence that leads with it hands the
// bidirectional algorithm a Hebrew character to resolve the whole paragraph's
// direction from: the line paints right to left inside a left to right card
// and the icon lands on the wrong edge. Leading with an English word fixes
// the paragraph's own direction, but the channel name is still a foreign
// script sitting inside that paragraph next to a colon and a digit run of
// its own (":  2391"), and an unisolated Hebrew run pulls the neutral colon
// and the following number backwards with it, so the measured result was
// still wrong: "is 2391 :13 רשת breaks judged". The channel therefore gets
// the same first-strong isolate this product already wraps every other
// foreign-script figure in (rules-bidi.js), which fences its own script off
// from the neutrals on either side of it and leaves the rest of the
// sentence, English or Hebrew, to run in the direction it already had.
export function complianceScopeSentence(locale, scope) {
  const channel = isolate(scope.scope_channel);
  const rowsOut = scope.rows_out;
  const excluded = scope.competitor_rows_excluded;
  if (locale === 'he') {
    return `${channel}, הערוץ שבבעלות המפעיל: ${rowsOut} ברייקים נשפטו, ${excluded} בערוצים אחרים לא נכללו.`;
  }
  return `This operator's own channel is ${channel}: ${rowsOut} breaks judged, ${excluded} on other channels left out.`;
}

// The calendar's price-multiplier banner. It used to name "the Pricing page",
// a destination that was renamed once pricing moved into this same workspace
// as the rate card tab, so the sentence now names the tab it actually opens.
export function calendarPricingBannerSentence(locale, eventsPricing) {
  if (eventsPricing === null) {
    return locale === 'he'
      ? 'לכל אירוע קיים גם מכפיל תמחור המחובר לשכבת האירועים בכרטיס התעריפים. השרת הזה אינו מדווח על השכבה, ולכן מצב ההפעלה שלה אינו ידוע כאן.'
      : 'Each event also carries a price multiplier hook for the events layer on the rate card. This server does not report that layer, so its activation state is unknown here.';
  }
  if (eventsPricing) {
    return locale === 'he'
      ? 'לכל אירוע קיים מכפיל תמחור המחובר לשכבת האירועים בכרטיס התעריפים. השכבה מופעלת כעת, ולכן מכפילים שונים מ-1.0 משנים את ההכנסה הצפויה בתחזית בימי אירועים.'
      : 'Each event carries a price multiplier wired to the events layer on the rate card. The layer is currently activated, so multipliers other than 1.0 change expected revenue in the forecast on event days.';
  }
  return locale === 'he'
    ? 'לכל אירוע קיים מכפיל תמחור המחובר לשכבת האירועים בכרטיס התעריפים. השכבה כבויה כעת, ולכן אף מכפיל אינו משנה מספר בתחזית עד הפעלתה שם.'
    : 'Each event carries a price multiplier wired to the events layer on the rate card. The layer is currently off, so no multiplier changes any forecast number until it is activated there.';
}

// The label on the banner's own control, named after the tab it opens rather
// than a page. Matches RulesWorkspace's own SECTIONS label for that tab.
export function rateCardTabLinkLabel(locale) {
  return locale === 'he' ? 'לפתיחת כרטיס התעריפים' : 'Open the rate card tab';
}
