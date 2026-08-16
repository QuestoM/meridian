import { API_BASE } from '../shell/api';
import { formatDayWithWeekday } from '../shell/dates';

// The restriction kinds, in the order the composer offers them. Each carries the
// sentence frame a representative reads and the parameter it needs, so the
// composer renders a sentence rather than a form. The server compiles them; this
// table only decides what a person can say.
export const KINDS = [
  {
    id: 'clean_tail',
    en: 'Keep the end clean',
    he: 'לשמור על הסוף נקי',
    param: 'protected_minutes',
    defaults: { protected_minutes: 8 },
  },
  {
    id: 'clean_open',
    en: 'Keep the opening clean',
    he: 'לשמור על הפתיחה נקייה',
    param: 'protected_minutes',
    defaults: { protected_minutes: 5 },
  },
  {
    id: 'no_breaks',
    en: 'No breaks at all',
    he: 'בלי ברייקים בכלל',
    param: null,
    defaults: {},
  },
  {
    id: 'exact_breaks',
    en: 'A set number of breaks',
    he: 'מספר ברייקים קבוע',
    param: 'count',
    defaults: { count: 1 },
  },
  {
    id: 'fixed_slot',
    en: 'A break at a fixed minute',
    he: 'ברייק בדקה קבועה',
    param: 'offset_seconds',
    defaults: { offset_seconds: 1320 },
  },
  {
    id: 'gold',
    en: 'Mark as gold breaks',
    he: 'לסמן כברייקי זהב',
    param: null,
    defaults: {},
  },
];

export function kindMeta(id) {
  return KINDS.find((kind) => kind.id === id) || KINDS[0];
}

async function readJson(path, init) {
  const response = await fetch(`${API_BASE}${path}`, init);
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    // The rules routes send `detail` as an object carrying both languages, so
    // both halves ride on the error and a renderer picks its reader's own.
    // Every other route sends one string, which stays what both readers get.
    const raw = body && body.detail;
    const words = raw && typeof raw === 'object' ? raw : null;
    const detail = words ? String(words.en || words.he || '') : (raw ? String(raw) : `${response.status} ${response.statusText}`);
    const error = new Error(detail);
    error.status = response.status;
    error.words = words;
    throw error;
  }
  return body;
}

const inFlightReads = new Map();

function readOnce(path) {
  // Deduplicate concurrent identical reads only. This covers React's
  // development remount without retaining stale governance state after a
  // save, removal or explicit refresh.
  if (!inFlightReads.has(path)) {
    const request = readJson(path).finally(() => inFlightReads.delete(path));
    inFlightReads.set(path, request);
  }
  return inFlightReads.get(path);
}

export function fetchTitles(query) {
  return readOnce(`/api/constraints/restrictions/titles?q=${encodeURIComponent(query || '')}`);
}

export function fetchAirings(title) {
  return readOnce(`/api/constraints/restrictions/airings?title=${encodeURIComponent(title || '')}`);
}

export function fetchRestrictions() {
  return readOnce('/api/constraints/restrictions');
}

const jsonPost = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export function previewRestriction(draft, signal) {
  return readJson('/api/constraints/restrictions/preview', { ...jsonPost(draft), signal });
}

export function saveRestriction(draft) {
  return readJson('/api/constraints/restrictions', jsonPost(draft));
}

export function deleteRestriction(restrictionId) {
  return readJson(`/api/constraints/restrictions/${encodeURIComponent(restrictionId)}`, { method: 'DELETE' });
}

export function fetchGuardrails() {
  return readOnce('/api/rules/guardrails');
}

export function fetchCompliance() {
  return readOnce('/api/compliance');
}

// What the Today ledger card draws, given its own fetch and the prop its
// parent passed. Pure and framework-free so a mutant here shows up in a plain
// node run rather than only in a browser: own data wins whenever it has
// answered; the prop is a fallback used only once the own fetch has failed;
// and a fallback that carries no scope key is a market-wide figure, never
// printed as the operator's own.
export function complianceViewState(own, ownFailed, fallback) {
  const data = own || (ownFailed ? fallback : null);
  if (!data) return { kind: 'loading' };
  const scope = data.scope || null;
  if (!scope) return { kind: 'basis_missing' };
  if (!scope.scoped) return { kind: 'no_channel', reasonEn: scope.reason_en, reasonHe: scope.reason_he };
  return { kind: 'scoped', data, scope };
}

export function fetchAttestation(since) {
  const suffix = since ? `?since=${encodeURIComponent(since)}` : '';
  return readOnce(`/api/rules/attestation${suffix}`);
}

export function recordGuardrailChange(values, effectiveDate, reason) {
  return readJson('/api/rules/guardrails', jsonPost({ values, effective_date: effectiveDate, reason }));
}

export function fetchActivation() {
  return readJson('/api/rules/model-activation');
}

export function setActivation(active) {
  return readJson('/api/rules/model-activation', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ active }),
  });
}

export function fetchOperatorChannel() {
  return readJson('/api/rules/operator-channel');
}

export function setOperatorChannel(channel) {
  return readJson('/api/rules/operator-channel', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ operator_channel: channel }),
  });
}

export function fetchPricingEffect(overrides, reset) {
  return readJson('/api/pricing/effect', jsonPost({ overrides: overrides || {}, reset: !!reset }));
}

// A predicate the server accepts, built from the two things a composer collects.
// Anything richer is the existing AND/OR builder's job; this covers the sentence.
export function buildWhere({ title, day }) {
  const conditions = [];
  if (title) conditions.push({ field: 'programme', operator: 'is', value: title });
  if (day) conditions.push({ field: 'date', operator: 'is', value: day });
  if (!conditions.length) return null;
  return { combinator: 'and', conditions };
}

// A date read back to a person, with the weekday it falls on. The stored value
// stays the ISO date; only the reading is localized, and the reading itself is
// shell/dates.js's to decide.
export function dayLabel(iso, locale) {
  return formatDayWithWeekday(iso, locale);
}

export function clock(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

export function minutes(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return null;
  return Math.round(value / 60);
}

// A per-second rate keeps its decimals. The worth of a second is 142.7044 and
// rounding it to the shekel would erase four digits of a figure whose whole job
// is to be compared against another one four digits away.
export function rate(value, locale) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '--';
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    maximumFractionDigits: 4,
    minimumFractionDigits: 4,
  }).format(number);
}

// A programme length, never rounded to a zero. A ten second filler rendered as
// "0 min" reads as a bug and hides exactly the case a window restriction bites
// hardest on, so anything under a minute is reported in seconds.
export function lengthLabel(seconds, locale) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return '--';
  if (value < 60) return locale === 'he' ? `${Math.round(value)} שנ` : `${Math.round(value)} sec`;
  return locale === 'he' ? `${Math.round(value / 60)} דק` : `${Math.round(value / 60)} min`;
}

// A before and an after, read as one value that moved. The isolation each side
// needs is not a wording question and not an attribute on the element, so it
// lives in its own module with the measurement that forced it.
export { isolate, valuePair } from './rules-bidi';

export function pairLabel(locale, before, after) {
  return locale === 'he' ? `מ-${before} ל-${after}` : `from ${before} to ${after}`;
}

// A count with its own noun, written out rather than templated. Hebrew does not
// survive "1 שידורים" and neither does English survive "1 airings", and a
// surplus of exactly one airing is the commonest case this surface has to say.
function airingWord(count, locale) {
  if (locale === 'he') return count === 1 ? 'שידור אחד' : `${count} שידורים`;
  return count === 1 ? '1 airing' : `${count} airings`;
}

function breakWord(count, locale) {
  if (locale === 'he') return count === 1 ? 'ברייק אחד' : `${count} ברייקים`;
  return count === 1 ? '1 break' : `${count} breaks`;
}

function nightWord(count, locale) {
  if (locale === 'he') return count === 1 ? 'לילה אחד' : `${count} לילות`;
  return count === 1 ? '1 night' : `${count} nights`;
}

// The head line above the night picker. It states both counts because they are
// two different facts and the second one is the one a person can act on: a
// restriction scoped to a date names a night, so two airings on one night are
// one choice. Measured on the reference plan, 43 airings of one programme fall
// on 19 nights, and a head line that said only 43 promised a precision the
// picker below it could not offer.
export function nightsHeadSentence(airings, nights, locale) {
  const total = Number(airings) || 0;
  const count = Number(nights) || 0;
  if (locale === 'he') {
    return `${airingWord(total, 'he')} ב-${nightWord(count, 'he')} בחלון התוכנית. בחרו לילה, או השאירו לכל השידורים.`;
  }
  return `${airingWord(total, 'en')} on ${nightWord(count, 'en')} in the plan window. Pick a night, or leave it for all of them.`;
}

// What one night chip says under its date. A night with more than one airing
// says how many, because that is the fact a single length would hide; a night
// with one says how long that programme runs, which is what the chip has always
// said and what a window restriction is judged against.
export function nightDetail(night, locale) {
  const airings = Number(night?.airings) || 0;
  if (airings > 1) return airingWord(airings, locale);
  return lengthLabel(night?.duration_seconds, locale);
}

// The whole fact about a night, for a reader who cannot see the chip's layout.
// The planned break count is stated only when the plan holds one, so an unknown
// is silent rather than read out as nought.
export function nightAriaLabel(night, locale) {
  const parts = [dayLabel(night?.day, locale), nightDetail(night, locale)];
  if (night?.planned_breaks !== null && night?.planned_breaks !== undefined) {
    parts.push(breakWord(Number(night.planned_breaks), locale));
  }
  return parts.filter(Boolean).join(', ');
}

// How many store rows one restriction wrote, said rather than templated. A
// restriction that binds a single airing is the ordinary case on this list, and
// "1 כללים" is not Hebrew any more than "1 rules" is English. The verb moves
// with the count in both languages, so the whole sentence is written out twice.
export function rulesWrittenSentence(count, locale) {
  const total = Number(count) || 0;
  if (locale === 'he') {
    return total === 1 ? 'כלל אחד נכתב לתוכנית' : `${total} כללים נכתבו לתוכנית`;
  }
  return total === 1 ? '1 rule written to the plan' : `${total} rules written to the plan`;
}

// The rows that bind the plan and carry no author, counted the same way.
export function unauthoredSentence(count, locale) {
  const total = Number(count) || 0;
  if (locale === 'he') {
    if (total === 1) return 'כלל אחד מחייב את התוכנית ואין לו מחבר. הוא נכתב לפני שלהגבלות היה תיעוד של מי ביקש.';
    return `${total} כללים מחייבים את התוכנית ואין להם מחבר. הם נכתבו לפני שלהגבלות היה תיעוד של מי ביקש.`;
  }
  if (total === 1) return '1 rule binds the plan and carries no author. It was written before restrictions had a record of who asked.';
  return `${total} rules bind the plan and carry no author. They were written before restrictions had a record of who asked.`;
}

// What a window rule reaches beyond the airings its own sentence named, said
// before the save. A rule is derived from the airings that breach it, and the
// predicate it compiles to can still hold an airing that was already keeping the
// window clean, because two airings of one programme can start inside one clock
// hour. That surplus used to arrive folded into one total; here it is its own
// sentence, with its own money, so a representative removing more than they
// asked for reads that rather than deducing it.
export function collateralSentence(locale, collateral, moneyText) {
  const he = locale === 'he';
  const bound = Number(collateral?.bound || 0);
  const changed = Number(collateral?.changed || 0);
  const breaks = Number(collateral?.breaks_removed || 0);
  const airings = airingWord(changed, locale);
  const head = he
    ? (bound === 1 ? 'הכלל מחייב גם שידור אחד שכבר שומר על החלון נקי.' : `הכלל מחייב גם ${bound} שידורים שכבר שומרים על החלון נקי.`)
    : (bound === 1 ? 'This rule also binds 1 airing that already keeps the window clean.' : `This rule also binds ${bound} airings that already keep the window clean.`);
  if (changed === 0) {
    const quiet = he
      ? 'הוא משאיר את מספר הברייקים בכולם, ולכן הם אינם עולים דבר.'
      : 'It leaves the break count on every one of them, so they cost nothing.';
    return `${head} ${quiet}`;
  }
  if (!moneyText) {
    const unpriced = he
      ? `הוא משנה את מספר הברייקים ב-${changed} מהם, והעלות שלהם אינה ידועה ולא אפס.`
      : `It changes the break count on ${changed} of them, and that cost is unknown rather than nought.`;
    return `${head} ${unpriced}`;
  }
  if (breaks > 0) {
    const cut = he
      ? `הוא מוריד ${breakWord(breaks, locale)} מתוך ${airings}, בשווי ${moneyText}.`
      : `It takes ${breakWord(breaks, locale)} off ${airings}, worth ${moneyText}.`;
    return `${head} ${cut}`;
  }
  if (breaks < 0) {
    const added = he
      ? `הוא מוסיף ${breakWord(-breaks, locale)} על פני ${airings}, בשווי ${moneyText}.`
      : `It adds ${breakWord(-breaks, locale)} across ${airings}, worth ${moneyText}.`;
    return `${head} ${added}`;
  }
  const moved = he
    ? `מספר הברייקים משתנה ב-${changed} מהם, בשווי ${moneyText}.`
    : `The break count changes on ${changed} of them, worth ${moneyText}.`;
  return `${head} ${moved}`;
}

// The engine keys read back as words a person says, in their own module
// under the file-size law. Re-exported here so every importer of this file
// keeps reaching them by the same name.
export {
  EFFECT_LIST,
  basisReason,
  calendarPricingBannerSentence,
  complianceScopeSentence,
  detailWords,
  effectLabel,
  limitBoundsRefusal,
  limitLabel,
  money,
  moreProgrammesSentence,
  nothingToSaveSentence,
  rateCardTabLinkLabel,
  refusalSentence,
  refusalWords,
  scheduledChangesSentence,
  unitLabel,
  widerScopeSentence,
} from './rules-words';

// The section a mounted Rules workspace should follow a later `?rules=`
// change to. A change the workspace's own tab clicks did not cause (a legacy
// route elsewhere in the shell rewriting the query, or the browser's own
// back and forward buttons) never remounts the workspace, so a caller has to
// compare the query on every render rather than only in useState's
// initializer. queryValue already comes pre-validated (empty when the query
// is missing or names no section this workspace has), so an empty value
// changes nothing and any other value is followed.
export function nextRulesSection(current, queryValue) {
  return queryValue || current;
}

// Deep-merge one rate-card patch onto a draft, the same shape the server
// deep-merges onto the saved overrides, so what is priced is what is saved.
export function mergeOverrides(base, patch) {
  const out = { ...(base || {}) };
  Object.entries(patch || {}).forEach(([key, value]) => {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      out[key] = mergeOverrides(out[key] || {}, value);
    } else {
      out[key] = value;
    }
  });
  return out;
}

// The staged value at one path, or undefined when the draft says nothing about
// it. A draft-bound control shows this when it exists and the saved figure
// otherwise, which is what makes discarding a draft put the saved figure back in
// the box instead of leaving the discarded one there.
export function draftValueAt(base, path) {
  return (path || []).reduce(
    (node, key) => (node && typeof node === 'object' ? node[key] : undefined),
    base,
  );
}

// Taking one leaf back out of a draft, which is what typing the saved figure
// back into a draft box means. mergeOverrides can only add, so without this a
// revert left the earlier edit in the draft while the box showed the saved
// value, and the effect panel priced a figure nobody could see. Empty parents
// are pruned and an emptied draft comes back as null, so the panel closes when
// there is nothing left to price.
export function dropOverride(base, path) {
  if (!base || typeof base !== 'object') return null;
  if (!Array.isArray(path) || path.length === 0) return base;
  const [head, ...rest] = path;
  if (!Object.prototype.hasOwnProperty.call(base, head)) return base;
  const out = { ...base };
  if (rest.length === 0) {
    delete out[head];
  } else {
    const child = dropOverride(out[head], rest);
    if (child && Object.keys(child).length > 0) out[head] = child;
    else delete out[head];
  }
  return Object.keys(out).length > 0 ? out : null;
}
