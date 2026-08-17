// Plain language for an extracted term. The shared half.
//
// WHY THIS EXISTS. The pipeline hands the review surface a term id and a
// parameter object: {"rows":[{"daypart":"פריים (19:30—23:00)","cpp":1380}],
// "audience":"בתי אב יהודיים","base_length_seconds":30}. A commercial director
// signing an annual framework cannot audit that, and a reviewer who is shown it
// as JSON is not reviewing the agreement — they are proofreading a data
// structure. So every term family gets a describer that turns its parameters
// into the sentence the clause actually says, plus the separate sentence about
// what approving it will DO to planning and pricing.
//
// THREE RULES HOLD THIS HONEST.
//
// Verbatim stays verbatim. The document's own daypart names, audience phrasing
// and position words are extracted in the document's vocabulary on purpose, and
// they are printed in it. Nothing here canonicalises "רצועת פריים" into a
// product daypart: that mapping is a decision a person makes at review, and a
// renderer that quietly performs it would hide the only place it is visible.
//
// A missing value is named, never smoothed. A term whose schema requires a rate
// and whose extraction has none renders the gap as a gap. `MISSING` below is
// that sentinel, and the review card reads it as an incompleteness rather than
// as a zero.
//
// The effect sentence comes from the taxonomy's own status, not from this file's
// opinion. BINDS says a rule will move real behaviour; REPRESENTABLE says it is
// held in full and wired to nothing; the difference is the whole product claim
// and it is asserted in exactly one place.

import { formatDay, formatSpan } from '../shell/dates';
import { formatNumber, pageText } from '../shell/format';
import { statusCopy, termName } from './trade-terms';

// A value the document did not supply. Distinct from zero and from empty.
export const MISSING = Symbol('missing');

export function isMissing(value) {
  return value === MISSING || value === null || value === undefined || value === '';
}

// Contract money is written out in full. formatCurrency switches to compact
// notation above 100,000, which is right for a tile that conveys scale and
// wrong for a commitment: a director reconciling a 15,500,000 ILS annual
// undertaking against the signed page reads the amount, and "₪15.5M" is not the
// amount. Nothing is rounded away here.
export function contractMoney(amount, locale = 'he') {
  const number = Number(amount);
  if (!Number.isFinite(number)) return null;
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    maximumFractionDigits: 2,
    minimumFractionDigits: 0,
  }).format(number);
}

const BASIS_COPY = {
  gross: { he: 'ברוטו', en: 'gross' },
  net_of_commission: { he: 'נטו לאחר עמלה', en: 'net of commission' },
  net_of_discount: { he: 'נטו לאחר הנחות', en: 'net of discounts' },
  ratecard: { he: 'לפי מחירון', en: 'at rate card' },
  unstated: { he: 'הבסיס לא נקבע במסמך', en: 'basis not stated in the document' },
};

export function basisLabel(basis, locale) {
  const entry = BASIS_COPY[String(basis || 'unstated')];
  if (!entry) return String(basis || '');
  return locale === 'he' ? entry.he : entry.en;
}

// A money block from the schema: {amount, currency, basis}. The basis travels
// with the figure because a 9,000,000 commitment at rate card and the same
// number net of commission are different obligations.
export function moneyPhrase(block, locale) {
  if (!block || typeof block !== 'object') return MISSING;
  const amount = contractMoney(block.amount, locale);
  if (!amount) return MISSING;
  const currency = String(block.currency || 'ILS');
  const figure = currency === 'ILS' ? amount : `${amount} ${currency}`;
  return `${figure} (${basisLabel(block.basis, locale)})`;
}

export function percentPhrase(value, locale) {
  const number = Number(value);
  if (!Number.isFinite(number)) return MISSING;
  return `${formatNumber(number, locale)}%`;
}

export function secondsPhrase(value, locale) {
  const number = Number(value);
  if (!Number.isFinite(number)) return MISSING;
  return pageText(locale, `${formatNumber(number, locale)} sec`, `${formatNumber(number, locale)} שנ׳`);
}

const PERIOD_COPY = {
  campaign: { he: 'לכל קמפיין', en: 'per campaign' },
  month: { he: 'לחודש', en: 'per month' },
  quarter: { he: 'לרבעון', en: 'per quarter' },
  year: { he: 'לשנה', en: 'per year' },
  custom: { he: 'לחלון שנקבע בסעיף', en: 'over the window the clause sets' },
};

export function periodLabel(value, locale) {
  const entry = PERIOD_COPY[String(value || '')];
  if (!entry) return value ? String(value) : MISSING;
  return locale === 'he' ? entry.he : entry.en;
}

export function listPhrase(values, locale) {
  const items = (Array.isArray(values) ? values : []).map((v) => String(v).trim()).filter(Boolean);
  if (items.length === 0) return MISSING;
  if (items.length === 1) return items[0];
  const last = items[items.length - 1];
  const head = items.slice(0, -1).join(', ');
  return pageText(locale, `${head} and ${last}`, `${head} ו${last}`);
}

// The instance envelope's own scope: who and what the term applies to. Rendered
// as one line per dimension the document actually restricted, because an empty
// dimension means "not restricted" and printing it as "all" would be an
// assertion the clause never made.
const SCOPE_LABELS = {
  advertisers: { he: 'מפרסמים', en: 'Advertisers' },
  brands: { he: 'מותגים', en: 'Brands' },
  campaigns: { he: 'קמפיינים', en: 'Campaigns' },
  channels: { he: 'ערוצים', en: 'Channels' },
  programmes: { he: 'תוכניות', en: 'Programmes' },
  genres: { he: 'ז׳אנרים', en: 'Genres' },
  dayparts: { he: 'רצועות', en: 'Dayparts' },
  weekdays: { he: 'ימים', en: 'Weekdays' },
  positions: { he: 'מיקומים בברייק', en: 'Positions in break' },
  lengths_seconds: { he: 'אורכים בשניות', en: 'Lengths in seconds' },
};

export function scopeLines(scope, locale) {
  if (!scope || typeof scope !== 'object') return [];
  return Object.entries(SCOPE_LABELS)
    .filter(([key]) => Array.isArray(scope[key]) && scope[key].length > 0)
    .map(([key, label]) => ({
      key,
      label: locale === 'he' ? label.he : label.en,
      value: scope[key].map((v) => String(v)).join(', '),
    }));
}

// A term's own effective window, when it differs from the agreement's. Every
// calendar day goes through shell/dates.js so the reader sees dd/mm/yyyy in
// both locales and no ISO string reaches the page.
export function windowPhrase(window, locale) {
  if (!window || typeof window !== 'object') return null;
  const from = window.from || window.starts_on;
  const to = window.to || window.ends_on;
  if (from && to) return formatSpan(from, to, locale);
  if (from) return pageText(locale, `from ${formatDay(from)}`, `מיום ${formatDay(from)}`);
  if (to) return pageText(locale, `until ${formatDay(to)}`, `עד ליום ${formatDay(to)}`);
  return null;
}

// What approving this term will do, from the taxonomy status. One sentence, and
// it is the same sentence everywhere the term appears.
export function effectSentence(termId, locale) {
  const status = statusCopy(termId, locale);
  return status ? status.note : null;
}

// The fallback describer, for a term the dispatch table does not cover. It
// prints the parameter names and their values rather than a JSON blob, and it
// says plainly that no sentence was written for this term — a reviewer being
// shown raw fields should know that is what is happening.
export function fallbackDescription(instance, locale) {
  const params = instance.params && typeof instance.params === 'object' ? instance.params : {};
  const rows = Object.entries(params).map(([key, value]) => ({
    label: key,
    value: typeof value === 'object' && value !== null
      ? JSON.stringify(value)
      : String(value),
    raw: true,
  }));
  return {
    headline: pageText(
      locale,
      `${termName(instance.term_id, locale)}: the clause is stored in full; no plain-language sentence is written for this term yet, so its extracted fields are shown as they are.`,
      `${termName(instance.term_id, locale)}: הסעיף נשמר במלואו. לא נכתב עדיין משפט בשפה פשוטה למונח הזה, ולכן השדות שחולצו מוצגים כפי שהם.`,
    ),
    rows,
    verbatim: true,
  };
}

// Free-text fields the document itself wrote, carried through untouched. They
// are the reviewer's best evidence and paraphrasing them would destroy it.
export function verbatimRows(params, keys, labels, locale) {
  return keys
    .filter((key) => !isMissing(params[key]))
    .map((key) => ({
      label: locale === 'he' ? labels[key].he : labels[key].en,
      value: String(params[key]),
      quote: true,
    }));
}
