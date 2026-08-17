// What an extracted term WILL DO, and how a reviewer is held to it.
//
// Split from trade-vocabulary.js at the project's file-size boundary, along the
// seam the surface already has: its sibling holds the vocabulary of the agreement
// as an OBJECT — its level, its lifecycle, its counterparty, its window, how its
// document was read. This file holds the vocabulary of the agreement as an
// EFFECT — the mechanism each term drives, the blockers that refuse approval, the
// verdicts a reviewer reaches, the alarm on a commitment, and the money lines of a
// simulation.
//
// trade-vocabulary.js re-exports everything here, so a caller resolves one
// vocabulary and does not have to know which half a word lives in.
//
// TWO RULES DECIDE WHAT GOES IN HERE, and they are the same two as the sibling's.
// The backend's own sentence wins: `sentence_he`, `mechanism_he`, `alarm_reason`,
// `reason_he`, `headline_he` and `basis_he` are printed as they arrive, and what
// this file adds is the frame around them. And a tone is never the whole message:
// every entry carries copy, so a mechanism is legible without colour.

import { pageText } from '../shell/format';

// ----------------------------------------------------------------- mechanisms

// What a term will DO once the agreement is approved. The engine decides the
// mechanism and sends its Hebrew label; this table adds the tone, the English
// label and the one-line explanation of the mechanism itself.
//
// The tone answers one question: will this act on its own? A mechanism that
// refuses or reprices carries an acting tone, `records` is deliberately quiet,
// and `inert` is the loudest thing on the screen because a term that looks
// binding and is not is the single most expensive misreading of this surface.
const MECHANISMS = {
  blocks: {
    en: 'Blocks placement', tone: 'warning', acts: true,
    heNote: 'הכלל יסרב שיבוץ שמפר אותו.',
    enNote: 'The rule refuses a placement that breaches it.',
  },
  warns: {
    en: 'Warns', tone: 'warning', acts: true,
    heNote: 'הכלל יתריע על הפרה ולא יחסום אותה.',
    enNote: 'The rule raises an alert on a breach rather than refusing it.',
  },
  prices: {
    en: 'Changes price', tone: 'positive', acts: true,
    heNote: 'הכלל ישנה את המחיר שמחושב לתשדיר.',
    enNote: 'The rule changes the price computed for a spot.',
  },
  steers: {
    en: 'Steers placement', tone: 'positive', acts: true,
    heNote: 'הכלל יטה את השיבוץ בלי לחסום אותו.',
    enNote: 'The rule biases placement without refusing anything.',
  },
  measures: {
    en: 'Measured continuously', tone: 'info', acts: true,
    heNote: 'העמידה בסעיף נמדדת ומוצגת; הפעולה נשארת אנושית.',
    enNote: 'Standing against the clause is measured and shown; acting on it stays human.',
  },
  settles: {
    en: 'Enters settlement', tone: 'positive', acts: true,
    heNote: 'הסעיף נכנס לחישוב ההתחשבנות של התקופה.',
    enNote: 'The clause enters the period settlement arithmetic.',
  },
  records: {
    en: 'Recorded only', tone: 'neutral', acts: false,
    heNote: 'הסעיף נרשם ומוצג; הוא אינו משנה תמחור או שיבוץ.',
    enNote: 'The clause is recorded and shown; it changes no price and no placement.',
  },
  inert: {
    en: 'Will not act automatically', tone: 'danger', acts: false,
    heNote: 'הסעיף נשמר במלואו אך לא יופעל אוטומטית. הסיבות מפורטות למטה.',
    enNote: 'The clause is stored in full but nothing will act on it. The reasons are listed below.',
  },
};

export const MECHANISM_ORDER = [
  'blocks', 'warns', 'prices', 'steers', 'settles', 'measures', 'records', 'inert',
];

export function mechanismTone(mechanism) {
  const entry = MECHANISMS[String(mechanism || '')];
  return entry ? entry.tone : 'neutral';
}

export function mechanismActs(mechanism) {
  const entry = MECHANISMS[String(mechanism || '')];
  return Boolean(entry && entry.acts);
}

// The label. In Hebrew the engine's own `mechanism_he` is preferred when the
// caller has it, because that string is the compiler's word for what it did.
export function mechanismLabel(mechanism, locale, serverHebrew) {
  if (locale === 'he') return String(serverHebrew || '') || mechanismName(mechanism, 'he');
  const entry = MECHANISMS[String(mechanism || '')];
  return entry ? entry.en : String(mechanism || '');
}

const MECHANISM_HE = {
  blocks: 'חוסם שיבוץ',
  warns: 'מתריע',
  prices: 'משנה מחיר',
  steers: 'מטה שיבוץ',
  measures: 'נמדד ברציפות',
  settles: 'נכנס להתחשבנות',
  records: 'נרשם בלבד',
  inert: 'לא יפעל אוטומטית',
};

export function mechanismName(mechanism, locale) {
  const key = String(mechanism || '');
  if (locale === 'he') return MECHANISM_HE[key] || key;
  const entry = MECHANISMS[key];
  return entry ? entry.en : key;
}

export function mechanismNote(mechanism, locale) {
  const entry = MECHANISMS[String(mechanism || '')];
  if (!entry) return '';
  return locale === 'he' ? entry.heNote : entry.enNote;
}

// ---------------------------------------------------------------- the gate

// Why approval is refused. The gate sends a kind and a count; the sentence that
// names what the reviewer must actually do lives here, because it is an
// instruction to a person rather than a fact about the document.
// Hebrew takes the singular for one; "1 סעיפים" reads as a machine talking.
const BLOCKERS = {
  clauses_unseen: {
    he: (n) => (n === 1
      ? 'סעיף אחד עדיין לא נקרא. כל סעיף במסמך צריך לעבור לפני העיניים.'
      : `${n} סעיפים עדיין לא נקראו. כל סעיף במסמך צריך לעבור לפני העיניים.`),
    en: (n) => (n === 1
      ? "One clause has not been read yet. Every clause in the document has to pass before a reviewer's eyes."
      : `${n} clauses have not been read yet. Every clause in the document has to pass before a reviewer's eyes.`),
  },
  instances_undecided: {
    he: (n) => (n === 1
      ? 'מונח אחד ממתין להחלטה: אישור, תיקון או דחייה.'
      : `${n} מונחים ממתינים להחלטה: אישור, תיקון או דחייה.`),
    en: (n) => (n === 1
      ? 'One term is waiting for a decision: confirm, edit or reject.'
      : `${n} terms are waiting for a decision: confirm, edit or reject.`),
  },
  unmapped_unacknowledged: {
    he: (n) => (n === 1
      ? 'סעיף אחד לא מופה לשום מונח ולא אושר ידנית. סעיף שהמערכת לא הבינה נשאר חסום עד שאדם יאמר מה הוא.'
      : `${n} סעיפים לא מופו לשום מונח ולא אושרו ידנית. סעיף שהמערכת לא הבינה נשאר חסום עד שאדם יאמר מה הוא.`),
    en: (n) => (n === 1
      ? 'One clause mapped to no term and was not acknowledged. A clause the system did not understand stays blocking until a person says what it is.'
      : `${n} clauses mapped to no term and were not acknowledged. A clause the system did not understand stays blocking until a person says what it is.`),
  },
  conflicts_open: {
    he: (n) => (n === 1
      ? 'סתירה אחת פתוחה בין סעיפים. יש להכריע איזו גרסה קובעת.'
      : `${n} סתירות פתוחות בין סעיפים. יש להכריע איזו גרסה קובעת.`),
    en: (n) => (n === 1
      ? 'One open conflict between clauses. A reviewer has to decide which version governs.'
      : `${n} open conflicts between clauses. A reviewer has to decide which version governs.`),
  },
  no_documents: {
    he: () => 'אין מסמך מצורף. אין מה לסקור.',
    en: () => 'No document is attached. There is nothing to review.',
  },
  no_extraction: {
    he: () => 'המסמך צורף אך טרם חולץ. יש להריץ חילוץ לפני הסקירה.',
    en: () => 'The document is attached but not extracted yet. Run extraction before reviewing.',
  },
  status: {
    he: () => 'ההסכם אינו במצב סקירה, ולכן אינו ניתן לאישור.',
    en: () => 'The agreement is not in review, so it cannot be approved.',
  },
};

export function blockerSentence(blocker, locale) {
  const kind = String(blocker && blocker.kind ? blocker.kind : '');
  const entry = BLOCKERS[kind];
  const count = blocker && Number.isFinite(Number(blocker.count)) ? Number(blocker.count) : 0;
  if (!entry) {
    return pageText(
      locale,
      `${kind || 'An unnamed blocker'} is blocking approval (${count}). The gate names it but this screen has no sentence written for it yet.`,
      `${kind || 'חסם ללא שם'} חוסם את האישור (${count}). השער מדווח עליו, אך למסך הזה אין עדיין משפט מוכן עבורו.`,
    );
  }
  return locale === 'he' ? entry.he(count) : entry.en(count);
}

// How a clause was disposed of by the extraction.
const DISPOSITIONS = {
  mapped: { he: 'מופה למונח', en: 'Mapped to a term', tone: 'positive' },
  irrelevant: { he: 'לא מסחרי', en: 'Not commercial', tone: 'neutral' },
  unmapped: { he: 'לא מופה', en: 'Not mapped', tone: 'warning' },
};

export function dispositionLabel(disposition, locale) {
  const entry = DISPOSITIONS[String(disposition || '')];
  if (!entry) return String(disposition || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function dispositionTone(disposition) {
  const entry = DISPOSITIONS[String(disposition || '')];
  return entry ? entry.tone : 'neutral';
}

// The reviewer's own verdict on a proposed term.
const REVIEW_STATES = {
  proposed: { he: 'ממתין להחלטה', en: 'Awaiting a decision', tone: 'warning' },
  confirmed: { he: 'אושר', en: 'Confirmed', tone: 'positive' },
  edited: { he: 'אושר עם תיקון', en: 'Confirmed with an edit', tone: 'positive' },
  rejected: { he: 'נדחה', en: 'Rejected', tone: 'danger' },
  reviewer_added: { he: 'נוסף בסקירה', en: 'Added by the reviewer', tone: 'info' },
};

export function reviewStateLabel(state, locale) {
  const entry = REVIEW_STATES[String(state || 'proposed')];
  if (!entry) return String(state || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function reviewStateTone(state) {
  const entry = REVIEW_STATES[String(state || 'proposed')];
  return entry ? entry.tone : 'neutral';
}

const CONFIDENCE = {
  high: { he: 'ביטחון גבוה', en: 'High confidence', tone: 'positive' },
  medium: { he: 'ביטחון בינוני', en: 'Medium confidence', tone: 'warning' },
  low: { he: 'ביטחון נמוך', en: 'Low confidence', tone: 'danger' },
};

export function confidenceLabel(value, locale) {
  const entry = CONFIDENCE[String(value || '')];
  if (!entry) return String(value || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function confidenceTone(value) {
  const entry = CONFIDENCE[String(value || '')];
  return entry ? entry.tone : 'neutral';
}

// ---------------------------------------------------------------- obligations

// A commitment's standing. `unknown` is a first-class answer here and it is not
// a synonym for compliant: an obligation the engine cannot measure says so, and
// the reason travels with it from the engine.
// The engine's own ladder, from kairos/trade/obligations.py: on_track, watch,
// at_risk, breached, unknown. Nothing is added to it here — a sixth alarm on
// this screen would be a state the engine cannot produce.
const ALARMS = {
  on_track: { he: 'בקצב', en: 'On track', tone: 'positive' },
  watch: { he: 'במעקב', en: 'Watch', tone: 'warning' },
  at_risk: { he: 'בסיכון', en: 'At risk', tone: 'warning' },
  breached: { he: 'בהפרה', en: 'Breached', tone: 'danger' },
  unknown: { he: 'לא ניתן למדידה', en: 'Not measurable', tone: 'info' },
};

// Worst first. A director scanning six agreements needs the breach, not the
// alphabet.
export const ALARM_ORDER = ['breached', 'at_risk', 'watch', 'unknown', 'on_track'];

export function alarmLabel(alarm, locale) {
  const entry = ALARMS[String(alarm || 'unknown')];
  if (!entry) return String(alarm || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function alarmTone(alarm) {
  const entry = ALARMS[String(alarm || 'unknown')];
  return entry ? entry.tone : 'neutral';
}

// ---------------------------------------------------------------- simulation

// The money lines a simulation can produce, in the order a commercial director
// reads them: what aired, what the deal gives away, what is left. A line absent
// from the payload is a line this agreement has no term for, and it is simply
// not rendered rather than shown as zero.
export const MONEY_LINES = [
  { key: 'gross_aired', he: 'ברוטו שהתבצע בפועל', en: 'Gross actually aired', kind: 'money' },
  { key: 'scheduled_ahead', he: 'משובץ קדימה', en: 'Scheduled ahead', kind: 'money' },
  { key: 'discount_ladder', he: 'הנחת מדרגות היקף', en: 'Volume ladder discount', kind: 'block' },
  { key: 'agency_commission', he: 'עמלת סוכנות', en: 'Agency commission', kind: 'block' },
  { key: 'net_after_simulated_terms', he: 'נטו לאחר תנאי ההסכם', en: 'Net after the agreement terms', kind: 'money', lead: true },
];

export function moneyLineLabel(key, locale) {
  const entry = MONEY_LINES.find((line) => line.key === key);
  if (!entry) return String(key || '');
  return locale === 'he' ? entry.he : entry.en;
}

// The named fields inside a ladder or commission block, so a nested figure is
// labelled rather than printed as a key.
const BLOCK_FIELDS = {
  spend_basis: { he: 'ההיקף שעליו חושב', en: 'Spend it was computed on', kind: 'money' },
  tier_reached_percent: { he: 'המדרגה שהושגה', en: 'Tier reached', kind: 'percent' },
  discount_value: { he: 'שווי ההנחה', en: 'Discount value', kind: 'money' },
  distance_to_next: { he: 'מרחק למדרגה הבאה', en: 'Distance to the next tier', kind: 'money' },
  percent: { he: 'שיעור', en: 'Rate', kind: 'percent' },
  commission_value: { he: 'שווי העמלה', en: 'Commission value', kind: 'money' },
  amount_base: { he: 'הבסיס לחישוב', en: 'Base it was computed on', kind: 'money' },
};

export function blockFieldLabel(key, locale) {
  const entry = BLOCK_FIELDS[String(key || '')];
  if (!entry) return String(key || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function blockFieldKind(key) {
  const entry = BLOCK_FIELDS[String(key || '')];
  return entry ? entry.kind : 'plain';
}

// HOW THE ENGINE RESOLVED THE SCOPE, in the reader's language.
//
// `scope.resolved` is a short English phrase the obligations engine writes for its
// own logs — "campaigns of X", "no advertiser links on file for agency Y". Printed
// as it arrives on a Hebrew screen it reads as untranslated interface copy, which
// is the same defect class as showing a reader an internal key. The engine's set
// of phrases is closed and small, so it is recognised here; a phrase this table
// does not know falls through verbatim rather than being guessed at, because a
// wrong translation of how a figure was scoped is worse than an English one.
const SCOPE_RESOLUTIONS = [
  {
    match: /^campaigns of (.+)$/,
    he: (name) => `הקמפיינים של ${name}`,
    en: (name) => `the campaigns of ${name}`,
  },
  {
    match: /^no advertiser links on file for agency (.+)$/,
    he: (name) => `אין מפרסמים מקושרים בקבצים לסוכנות ${name}`,
    en: (name) => `no advertisers are linked on file to the agency ${name}`,
  },
  {
    match: /^all campaigns on file$/,
    he: () => 'כל הקמפיינים שבקבצים',
    en: () => 'every campaign on file',
  },
  {
    match: /^no campaigns on file$/,
    he: () => 'אין קמפיינים בקבצים',
    en: () => 'no campaign is on file',
  },
  {
    match: /^scoped to named campaigns$/,
    he: () => 'הקמפיינים שההסכם נוקב בשמם',
    en: () => 'the campaigns the agreement names',
  },
];

export function scopeResolution(resolved, locale) {
  const text = String(resolved || '').trim();
  if (!text) return '';
  for (const entry of SCOPE_RESOLUTIONS) {
    const hit = entry.match.exec(text);
    if (hit) return locale === 'he' ? entry.he(hit[1]) : entry.en(hit[1]);
  }
  return text;
}

export const BLOCK_FIELD_ORDER = Object.keys(BLOCK_FIELDS);
