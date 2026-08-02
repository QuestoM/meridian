import { word } from '../../vocabulary.js';

// Plan, the week: the shape of the destination, in one place.
//
// The four navigation entries the shell still carries (Optimizer, Schedule,
// Inventory, Forecasts) are four entrances to this one destination, each landing
// on the step it was named for. That is the merge section 3.5 of the
// specification calls for, performed inside the tree this piece owns: the shell
// router and its navigation list are frozen, so the entries stay, and what they
// open stops being four separate places.

// The Israeli week, presentation order. Data stays ISO-keyed everywhere; only
// the reading order is fixed here. The shell's own weekday array is Monday-first
// and frozen, so a surface in this tree orders with SUNDAY_FIRST instead.
export const SUNDAY_FIRST = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

// Friday and Saturday, and only those two.
export const WEEKEND_DAYS = ['Fri', 'Sat'];

export function isWeekend(day) {
  return WEEKEND_DAYS.includes(String(day || '').slice(0, 3));
}

export function weekOrder(day) {
  const index = SUNDAY_FIRST.indexOf(String(day || '').slice(0, 3));
  return index === -1 ? SUNDAY_FIRST.length : index;
}

// The day names, presentation only. The payload stays ISO-keyed and carries the
// three-letter key; a reader gets their own language.
const WEEKDAY_LABELS = {
  Sun: { en: 'Sunday', he: 'ראשון' },
  Mon: { en: 'Monday', he: 'שני' },
  Tue: { en: 'Tuesday', he: 'שלישי' },
  Wed: { en: 'Wednesday', he: 'רביעי' },
  Thu: { en: 'Thursday', he: 'חמישי' },
  Fri: { en: 'Friday', he: 'שישי' },
  Sat: { en: 'Saturday', he: 'שבת' },
};

export function weekdayLabel(day, locale) {
  const entry = WEEKDAY_LABELS[String(day || '').slice(0, 3)];
  if (!entry) return '';
  return locale === 'he' ? entry.he : entry.en;
}

export function bySundayFirst(a, b) {
  return weekOrder(a?.day) - weekOrder(b?.day);
}

// The six sections, in the order the planner's job runs. The first four are
// JS-2's own sequence, so the rail reads as the job rather than as a menu; the
// last two are the reference views the same person needs beside it.
export const SECTIONS = [
  { id: 'objective', step: 1, key: 'o', en: 'Objective', he: 'מטרה' },
  { id: 'run', step: 2, key: 'r', en: 'Run', he: 'הרצה' },
  { id: 'compare', step: 3, key: 'c', en: 'Compare', he: 'השוואה' },
  { id: 'publish', step: 4, key: 'p', en: 'Publish', he: 'הפצה' },
  { id: 'supply', step: null, key: 's', en: 'Supply', he: 'היצע' },
  { id: 'board', step: null, key: 'b', en: 'Week board', he: 'לוח השבוע' },
];

export const SECTION_IDS = SECTIONS.map((section) => section.id);

export function sectionLabel(id, locale) {
  const section = SECTIONS.find((item) => item.id === id);
  if (!section) return '';
  return locale === 'he' ? section.he : section.en;
}

// Which section each surviving navigation entry lands on.
export const ENTRANCES = {
  Optimizer: 'objective',
  Schedule: 'board',
  Inventory: 'supply',
  Forecasts: 'compare',
};

export function sectionForEntrance(entrance) {
  return ENTRANCES[entrance] || 'objective';
}

// The four objective templates. Same four values the settings surface has always
// applied, kept identical on purpose: Bar 3 requires them to survive the move,
// so they move by value and not by rewrite.
export const OBJECTIVE_TEMPLATES = [
  {
    key: 'balanced',
    en: 'Balanced',
    he: 'מאוזן',
    descEn: 'Revenue-leaning, viewer-protective',
    descHe: 'נוטה-להכנסה אך שומר על הצופים',
    values: { revenue_weight: 60, risk_lambda: 0, min_retention_floor: 0.72 },
  },
  {
    key: 'revenue',
    en: 'Revenue priority',
    he: 'מקסום הכנסה',
    descEn: 'Maximize revenue to the guardrails',
    descHe: 'ממקסם הכנסה עד גבול הרגולציה',
    values: { revenue_weight: 85, risk_lambda: 0, min_retention_floor: 0.70 },
  },
  {
    key: 'retention',
    en: 'Retention guardrail',
    he: 'הגנת שימור',
    descEn: 'Fewer breaks, higher floor',
    descHe: 'פחות ברייקים, רצפת צפייה גבוהה',
    values: { revenue_weight: 35, risk_lambda: 0, min_retention_floor: 0.78 },
  },
  {
    key: 'conservative',
    en: 'Conservative',
    he: 'זהיר באי-ודאות',
    descEn: 'Reports at the worst plausible retention cost',
    descHe: 'מדווח לפי עלות השימור הסבירה הגרועה ביותר',
    values: { revenue_weight: 60, risk_lambda: 1, min_retention_floor: 0.74 },
  },
];

// The two engine focuses. Named by what they do to the plan, never by the
// engine's field name.
export const OBJECTIVE_FOCUS = [
  {
    key: 'blend',
    en: 'Balanced, the default',
    he: 'מאוזן, ברירת המחדל',
    descEn: 'The engine balances gross revenue against keeping viewers, using the weight above.',
    descHe: 'המנוע מאזן בין הכנסות ברוטו לשמירה על הצופים, לפי המשקל שלמעלה.',
  },
  {
    key: 'revenue_net',
    en: 'Net focused',
    he: 'ממוקד נטו',
    descEn: 'The engine drops breaks whose revenue is below their retention cost: fewer breaks, lower gross, higher net.',
    descHe: 'המנוע מוותר על ברייקים שההכנסה שלהם נמוכה מעלות השימור שלהם: פחות ברייקים, ברוטו נמוך יותר, נטו גבוה יותר.',
  },
];

export function templateMatches(template, settings) {
  if (!settings) return false;
  return Object.entries(template.values).every(([field, value]) => {
    const current = Number(settings[field]);
    return Number.isFinite(current) && Math.abs(current - Number(value)) < 1e-9;
  });
}

// The lever names, in the operator's words. The engine's own field names never
// reach a label.
export const LEVER_LABELS = {
  revenue_weight: { en: 'Revenue and retention balance', he: 'איזון הכנסה מול צפייה' },
  retention_floor: { en: 'Minimum retention floor', he: 'רצפת צפייה מינימלית' },
  min_retention_floor: { en: 'Minimum retention floor', he: 'רצפת צפייה מינימלית' },
  max_breaks_per_hour: { en: 'Breaks per hour, at most', he: 'ברייקים לשעה, לכל היותר' },
  risk_lambda: { en: 'Uncertainty caution', he: 'זהירות מול אי-ודאות' },
  objective_mode: { en: 'Engine focus', he: 'מיקוד המנוע' },
};

export function leverLabel(field, locale) {
  const entry = LEVER_LABELS[field];
  if (!entry) return field;
  return locale === 'he' ? entry.he : entry.en;
}

// One lever's value in the words the surface uses for it, so a change can be
// read as "72% becomes 80%" rather than as two raw numbers. The engine focus is
// a key and reads as its own name; everything else is a number and carries its
// own unit.
export function leverValueText(field, value, locale) {
  if (field === 'objective_mode') {
    const mode = OBJECTIVE_FOCUS.find((item) => item.key === String(value));
    if (!mode) return String(value ?? '');
    return locale === 'he' ? mode.he : mode.en;
  }
  const number = Number(value);
  if (!Number.isFinite(number)) return '';
  if (field === 'min_retention_floor' || field === 'retention_floor') return `${Math.round(number * 100)}%`;
  if (field === 'risk_lambda') return String(Math.round(number * 100));
  return String(Math.round(number));
}

// The five levers a comparison leg ran under, as the five fields the saved
// objective keeps them in. One field is named differently on the two sides and
// nothing else is translated, so the values the card's money was computed on are
// the values the objective receives.
export const ADOPT_FIELDS = [
  ['revenue_weight', 'revenue_weight'],
  ['retention_floor', 'min_retention_floor'],
  ['max_breaks_per_hour', 'max_breaks_per_hour'],
  ['risk_lambda', 'risk_lambda'],
  ['objective_mode', 'objective_mode'],
];

export function objectiveFromLevers(levers) {
  if (!levers || typeof levers !== 'object') return null;
  const out = {};
  for (const [from, to] of ADOPT_FIELDS) {
    const value = levers[from];
    // A partial adoption would leave the objective disagreeing with the card
    // that offered it, so anything short of all five is refused outright.
    if (value === null || value === undefined || value === '') return null;
    out[to] = value;
  }
  return out;
}

// Why a board has no broadcast day to draw, in both languages. The server sends
// a code beside its own English prose, exactly as the plan-version store does.
const BOARD_REASONS = {
  date_not_in_programme_source: {
    en: 'The programme source carries no programme on your channel on that broadcast day, so there is no board to draw for it.',
    he: 'מקור התוכניות אינו כולל אף תוכנית בערוץ שלכם ביום השידור הזה, ולכן אין לוח להציג עבורו.',
  },
  no_programme_in_source: {
    en: 'The programme source carries no programme on your channel, so there is no board to draw.',
    he: 'מקור התוכניות אינו כולל אף תוכנית בערוץ שלכם, ולכן אין לוח להציג.',
  },
  unreadable_date: {
    en: 'That broadcast day could not be read as a date.',
    he: 'לא ניתן היה לקרוא את יום השידור הזה כתאריך.',
  },
};

export function boardReason(board, locale) {
  if (!board) return null;
  const entry = BOARD_REASONS[String(board.reason_code || '')];
  if (entry) return locale === 'he' ? entry.he : entry.en;
  return board.reason || null;
}

// The vocabulary this destination speaks, resolved once so a component reads a
// word rather than a key.
export function planWords(locale) {
  return {
    run: word('action.run_weekly_plan', locale),
    runShort: word('action.run_plan', locale),
    preview: word('action.preview', locale),
    publish: word('action.publish', locale),
    planVersion: word('object.plan_version', locale),
    weeklyPlan: word('object.weekly_plan', locale),
    modelVersion: word('object.model_version', locale),
    expectedRevenue: word('concept.expected_revenue', locale),
    retentionCost: word('concept.retention_cost', locale),
    retentionFloor: word('concept.retention_floor', locale),
    balance: word('concept.revenue_balance', locale),
    caution: word('concept.caution', locale),
    supply: word('concept.supply', locale),
    breaks: word('object.breaks', locale),
    planCurrent: word('state.plan_current', locale),
    planOutOfDate: word('state.plan_out_of_date', locale),
    place: word('place.plan.week', locale),
  };
}

// The refusals the plan-version store can return, in both languages. The server
// sends a code beside its own English prose so a person reads their language and
// a log keeps the exact string; a code with no entry here falls back to the
// prose rather than to silence.
const DIFF_REASONS = {
  first_version: {
    en: 'This is the first frozen plan, so there is nothing before it to compare against.',
    he: 'זו התוכנית המוקפאת הראשונה, ולכן אין לפניה גרסה להשוות אליה.',
  },
  unknown_version: {
    en: 'That plan version is not in the store any more.',
    he: 'גרסת התוכנית הזאת כבר אינה במאגר.',
  },
  no_frozen_file: {
    en: 'That plan version has no frozen file, so it cannot be compared or restored.',
    he: 'לגרסת התוכנית הזאת אין קובץ מוקפא, ולכן לא ניתן להשוות אליה או לחזור אליה.',
  },
};

export function diffReason(payload, locale) {
  if (!payload) return null;
  const entry = DIFF_REASONS[String(payload.reason_code || '')];
  if (entry) return locale === 'he' ? entry.he : entry.en;
  return payload.reason || null;
}

// The scope a money figure was summed on, printed beside the figure and never
// in a tooltip. Returns null when the payload does not state one, which is the
// case a caller must render as an absence rather than fill in.
// A Hebrew channel name inside an English sentence, or a Latin one inside a
// Hebrew sentence, reorders the punctuation around it unless it is isolated.
// These are the Unicode isolate characters, which work inside a plain string
// where a <bdi> element cannot go.
const ISOLATE_START = '⁨';
const ISOLATE_END = '⁩';

export function isolate(value) {
  const text = String(value ?? '').trim();
  return text ? `${ISOLATE_START}${text}${ISOLATE_END}` : '';
}

export function scopeLine(note, locale) {
  if (!note || typeof note !== 'object') return null;
  if (note.scoped === false) {
    return locale === 'he'
      ? 'לא הוגדר ערוץ מפעיל, ולכן המספרים כאן הם של כל הערוצים במקור הנתונים'
      : 'No operator channel is configured, so these figures cover every channel in the source';
  }
  const channel = isolate(note.scope_channel);
  if (!channel) return null;
  const rows = Number(note.rows_out);
  if (!Number.isFinite(rows)) {
    return locale === 'he' ? `הערוץ שלכם: ${channel}` : `Your channel: ${channel}`;
  }
  return locale === 'he'
    ? `הערוץ שלכם, ${channel}, ${rows.toLocaleString('he-IL')} שורות תוכנית`
    : `Your channel, ${channel}, ${rows.toLocaleString('en-US')} plan rows`;
}

// The saved plan is also a file, and that file is not this board.
//
// The export carries every channel the source holds, so it is not offered from a
// surface scoped to the operator: a download that hands over three rivals' plans
// is the competitor boundary broken by a button. The board names what the file
// is and where it lives instead, with both counts read from the payload's own
// scope note rather than from a constant, so the sentence cannot outlive the
// figure it describes.
export function exportScopeNote(note, locale) {
  if (!note || typeof note !== 'object') return null;
  const rowsIn = Number(note.rows_in);
  if (!Number.isFinite(rowsIn)) return null;
  const rowsOut = Number(note.rows_out);
  const inText = rowsIn.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US');
  if (note.scoped === false || !Number.isFinite(rowsOut) || rowsOut >= rowsIn) {
    return locale === 'he'
      ? `קובץ התוכנית המלא נמצא במסך המקורות. יש בו ${inText} שורות מכל הערוצים שבמקור הנתונים, ולא רק שלכם.`
      : `The full plan file is on Sources. It holds ${inText} rows from every channel in the source, not only yours.`;
  }
  const outText = rowsOut.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US');
  return locale === 'he'
    ? `קובץ התוכנית המלא נמצא במסך המקורות. יש בו ${inText} שורות מכל הערוצים שבמקור הנתונים, ומתוכן ${outText} שלכם.`
    : `The full plan file is on Sources. It holds ${inText} rows from every channel in the source, of which ${outText} are yours.`;
}
