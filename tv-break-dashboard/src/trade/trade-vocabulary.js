// The agreement surface's own words: statuses, levels, mechanisms, blockers,
// alarms and the money lines of a simulation.
//
// trade-terms.js is the twin of the term registry and term-language.js turns one
// term's parameters into a sentence. This file is the layer around both: the
// vocabulary an agreement REVIEW needs, which belongs to the review process
// rather than to any single clause.
//
// TWO RULES DECIDE WHAT GOES IN HERE.
//
// The backend's sentence wins. `sentence_he`, `mechanism_he`, `alarm_reason`,
// `reason_he`, `headline_he` and `basis_he` are all authored by the engine from
// its own compiler verdict, and they are printed as they arrive. What this file
// holds is the frame around them: the label on a badge, the name of a blocker
// kind, the heading of a money line. Nothing here re-derives an effect.
//
// A tone is never the whole message. Every entry carries copy, and the surface
// pairs it with an icon, so a mechanism is legible without colour.

import { pageText } from '../shell/format';

// ------------------------------------------------------------------ lifecycle

// The three the store accepts (kairos_api/trade_store.py LEVELS). An amendment
// or an appendix is not a fourth level: it is an agreement at one of these three
// levels carrying a parent, which is how precedence between them is decided.
const LEVELS = {
  agency_framework: { he: 'הסכם מסגרת עם סוכנות', en: 'Agency framework' },
  advertiser: { he: 'הסכם מפרסם', en: 'Advertiser agreement' },
  campaign: { he: 'הסכם קמפיין', en: 'Campaign agreement' },
};

export function levelLabel(level, locale) {
  const entry = LEVELS[String(level || '')];
  if (!entry) return String(level || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function levelOptions(locale) {
  return Object.entries(LEVELS).map(([value, entry]) => ({
    value,
    label: locale === 'he' ? entry.he : entry.en,
  }));
}

// The lifecycle, with the tone that says whether the agreement is acting on the
// business. Only an approved agreement can bind a rule, and the vocabulary says
// so rather than leaving four shades of grey to be guessed at.
const STATUSES = {
  draft: { he: 'טיוטה', en: 'Draft', tone: 'neutral' },
  in_review: { he: 'בסקירה', en: 'In review', tone: 'info' },
  approved: { he: 'מאושר', en: 'Approved', tone: 'positive' },
  superseded: { he: 'הוחלף', en: 'Superseded', tone: 'neutral' },
  expired: { he: 'פג תוקף', en: 'Expired', tone: 'warning' },
  withdrawn: { he: 'בוטל', en: 'Withdrawn', tone: 'danger' },
};

export function statusLabel(status, locale) {
  const entry = STATUSES[String(status || '')];
  if (!entry) return String(status || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function statusTone(status) {
  const entry = STATUSES[String(status || '')];
  return entry ? entry.tone : 'neutral';
}

const COUNTERPARTY_KINDS = {
  agency: { he: 'סוכנות', en: 'Agency' },
  advertiser: { he: 'מפרסם', en: 'Advertiser' },
};

export function counterpartyKind(kind, locale) {
  const entry = COUNTERPARTY_KINDS[String(kind || '')];
  if (!entry) return String(kind || '');
  return locale === 'he' ? entry.he : entry.en;
}

// The counterparty block arrives in two shapes across the corpus and the API:
// {kind, name} from the store's own create, and {counterparty_type, agency} from
// an extraction. Both are read here so no surface has to know which it got.
export function counterpartyName(counterparty) {
  if (!counterparty || typeof counterparty !== 'object') return '';
  return String(counterparty.name || counterparty.agency || counterparty.advertiser || '');
}

export function counterpartyKindOf(counterparty) {
  if (!counterparty || typeof counterparty !== 'object') return '';
  return String(counterparty.kind || counterparty.counterparty_type || '');
}

// --------------------------------------------------------------- the window

// EVERY AGREEMENT HAS AN END DATE, and the store enforces it: an obligation with
// no closing date has no measurement window, so its pace, its projection and its
// alarm are all undefined. An agreement the parties meant to run until somebody
// cancels is therefore stored against a sentinel far-future date with an
// `open_ended` flag, and this reader turns that back into the sentence the
// parties actually agreed rather than printing a literal 2099 deadline.
//
// The window arrives under two key pairs — `starts_on`/`ends_on` from the store
// and `from`/`to` from an extraction — so both are read here and nowhere else.
export const OPEN_ENDED_UNTIL = '2099-12-31';

export function windowOf(window) {
  if (!window || typeof window !== 'object') return { from: '', to: '', openEnded: false };
  const from = String(window.starts_on || window.from || '');
  const to = String(window.ends_on || window.to || '');
  const openEnded = Boolean(window.open_ended) || to === OPEN_ENDED_UNTIL;
  return { from, to: openEnded ? '' : to, openEnded };
}

export function openEndedLabel(locale) {
  return pageText(
    locale,
    'Open-ended: it runs until one side cancels',
    'ללא מועד סיום: בתוקף עד שאחד הצדדים יבטל',
  );
}

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
const BLOCKERS = {
  clauses_unseen: {
    he: (n) => `${n} סעיפים עדיין לא נקראו. כל סעיף במסמך צריך לעבור לפני העיניים.`,
    en: (n) => `${n} clauses have not been read yet. Every clause in the document has to pass before a reviewer's eyes.`,
  },
  instances_undecided: {
    he: (n) => `${n} מונחים ממתינים להחלטה: אישור, תיקון או דחייה.`,
    en: (n) => `${n} terms are waiting for a decision: confirm, edit or reject.`,
  },
  unmapped_unacknowledged: {
    he: (n) => `${n} סעיפים לא מופו לשום מונח ולא אושרו ידנית. סעיף שהמערכת לא הבינה נשאר חסום עד שאדם יאמר מה הוא.`,
    en: (n) => `${n} clauses mapped to no term and were not acknowledged. A clause the system did not understand stays blocking until a person says what it is.`,
  },
  conflicts_open: {
    he: (n) => `${n} סתירות פתוחות בין סעיפים. יש להכריע איזו גרסה קובעת.`,
    en: (n) => `${n} open conflicts between clauses. A reviewer has to decide which version governs.`,
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

export const BLOCK_FIELD_ORDER = Object.keys(BLOCK_FIELDS);
