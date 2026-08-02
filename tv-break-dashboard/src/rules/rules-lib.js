import { API_BASE } from '../shell/api';

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
    const detail = body && body.detail ? String(body.detail) : `${response.status} ${response.statusText}`;
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return body;
}

export function fetchTitles(query) {
  return readJson(`/api/constraints/restrictions/titles?q=${encodeURIComponent(query || '')}`);
}

export function fetchAirings(title) {
  return readJson(`/api/constraints/restrictions/airings?title=${encodeURIComponent(title || '')}`);
}

export function fetchRestrictions() {
  return readJson('/api/constraints/restrictions');
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
  return readJson('/api/rules/guardrails');
}

export function fetchAttestation(since) {
  const suffix = since ? `?since=${encodeURIComponent(since)}` : '';
  return readJson(`/api/rules/attestation${suffix}`);
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

// The Israeli week, Sunday first, for a date read back to a person. The stored
// value stays the ISO date; only the reading is localized.
const WEEKDAY_HE = ['ראשון', 'שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת'];
const WEEKDAY_EN = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

export function dayLabel(iso, locale) {
  const text = String(iso || '').slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) return text;
  const parsed = new Date(`${text}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return text;
  const names = locale === 'he' ? WEEKDAY_HE : WEEKDAY_EN;
  return `${names[parsed.getDay()]}, ${text}`;
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

// Money on this surface is never compacted. The shell formatter switches to
// compact notation above 100,000, which would render a before of 1,067,846 and
// an after of 1,030,969 as the same two characters, and the whole point of this
// surface is that the difference between them is the decision. Full grouped
// digits, always, with the sign kept so a cost reads as a cost.
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
