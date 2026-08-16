import { BROADCAST_ZONE, formatClock, formatDayWithWeekday, formatStamp, isWeekendDay, isoDay, todayIso } from '../shell/dates';

// Every word History renders, in both languages, in one place.
//
// The API sends no prose: an entry carries a kind, an action code, an actor and
// numbers, and this file is where each of those becomes a sentence. That is
// what keeps the payload free of the training lexicon and what makes a
// vocabulary change one edit instead of a sweep.
//
// The nine addressable places are named by the frozen src/vocabulary.js, so the
// destination's own name is not spelled twice in the product.

import { word } from '../vocabulary.js';

// The words a stored row of the eight restorable stores is read in live in
// history-row-words.js and are re-exported here, so this file stays the one
// import a surface needs for a word. They sit in a file of their own only
// because this one is at the 450-line cap.
export {
  ROW_CHANNEL_SCOPE, ROW_EFFECTS, ROW_EVENT_TYPES, ROW_KINDS, ROW_MODES, ROW_MORE,
  ROW_PARTS, ROW_SOURCES, ROW_UNNAMED, ROW_UNREADABLE, ROW_WEEKDAY_INDEX,
} from './history-row-words.js';

export function historyPlace(locale) {
  return word('place.history', locale === 'he' ? 'he' : 'en');
}

// The Israeli week: it starts on Sunday and ends on Saturday, and the weekend
// is Friday and Saturday only. Indexed by getDay(), which is 0 for Sunday.
export const WEEKDAYS = [
  ['Sunday', 'ראשון'],
  ['Monday', 'שני'],
  ['Tuesday', 'שלישי'],
  ['Wednesday', 'רביעי'],
  ['Thursday', 'חמישי'],
  ['Friday', 'שישי'],
  ['Saturday', 'שבת'],
];

export const WEEKEND_DAYS = [5, 6];

// Takes the ISO day itself, not a Date built from it. The Date form read the
// weekday in the machine's local zone, which files a Friday broadcast day under
// Thursday for a reader west of here.
export const isWeekend = isWeekendDay;

// The six kinds a timeline entry can be. A closed vocabulary, so a reader can
// learn the whole set once and a critic can enumerate it.
//
// preview is a kind of its own because the recorder records every mutating verb
// and a large share of them save nothing: measured on the running instance, 57
// of 345 recorded requests were the day board scoring a placement that had not
// been saved. A row that says a change happened when nothing was saved is wrong
// about the one thing this destination exists to answer.
export const KIND_LABELS = {
  change: ['Change', 'שינוי'],
  run: ['Run', 'הרצה'],
  restore_point: ['Restore point', 'נקודת שחזור'],
  restore: ['Restore', 'שחזור'],
  sign_in: ['Account', 'חשבון'],
  preview: ['Preview', 'תצוגה מקדימה'],
};

// What a kind means, on the control that filters by it. The change hint names
// both halves because the tab counts attempts: a write that the wall refused is
// a change entry and it changed nothing, and the row says which it was.
export const KIND_HINTS = {
  change: ['Something was saved, or the attempt was refused', 'משהו נשמר, או שהניסיון נדחה'],
  run: ['The plan was run', 'התוכנית הורצה'],
  restore_point: ['A point you can go back to', 'נקודה שאפשר לחזור אליה'],
  restore: ['Something was put back', 'משהו הוחזר'],
  sign_in: ['Somebody signed in or out', 'מישהו נכנס או יצא'],
  preview: ['An answer was computed and nothing was saved', 'חושבה תשובה ולא נשמר דבר'],
};

// A mutating request, in the operator's own words. The API sends the code; an
// unknown code falls back to the raw method and path, exactly as the settings
// activity log has always rendered one.
export const ACTION_LABELS = {
  settings_change: ['Settings saved', 'הגדרות נשמרו'],
  pricing_change: ['Rate card saved', 'מחירון נשמר'],
  plan_run: ['Weekly plan run', 'הרצת הלוח השבועי'],
  plan_publish: ['Plan version published', 'גרסת תוכנית הופצה'],
  plan_restore: ['Plan version put back', 'גרסת תוכנית הוחזרה'],
  target_change: ['Target saved', 'יעד נשמר'],
  restriction_change: ['Restriction saved', 'הגבלה נשמרה'],
  override_change: ['Pin saved', 'נעיצה נשמרה'],
  placement_change: ['Break moved', 'ברייק הוזז'],
  gold_change: ['Gold break saved', 'ברייק זהב נשמר'],
  break_change: ['Break saved', 'ברייק נשמר'],
  guardrail_change: ['Regulatory limit saved', 'מגבלת רגולציה נשמרה'],
  model_activation_change: ['Audience model activation saved', 'מצב ההפעלה של מודל הקהל נשמר'],
  channel_change: ['Operator channel saved', 'ערוץ המפעיל נשמר'],
  client_change: ['Client record saved', 'רשומת לקוח נשמרה'],
  client_onboarding: ['Client onboarded', 'לקוח נקלט'],
  campaign_change: ['Campaign saved', 'קמפיין נשמר'],
  calendar_change: ['Calendar event saved', 'אירוע ביומן נשמר'],
  source_upload: ['Source file uploaded', 'קובץ מקור הועלה'],
  restore_point_saved: ['Restore point saved', 'נקודת שחזור נשמרה'],
  restore_point_renamed: ['Restore point renamed', 'שם נקודת שחזור שונה'],
  restore: ['Restore applied', 'שחזור בוצע'],
  assistant_action: ['Human decision on a Mabat proposal recorded', 'החלטה אנושית על הצעה של מבט נרשמה'],
  assistant_undo: ['Mabat change undone', 'שינוי של מבט בוטל'],
  assistant_upload: ['File given to Mabat', 'קובץ נמסר למבט'],
  conversation_change: ['Mabat conversation changed', 'שיחה עם מבט שונתה'],
  job_change: ['Job changed', 'תפקיד שונה'],
  password_change: ['Password changed', 'סיסמה שונתה'],
  account_change: ['Account changed', 'חשבון שונה'],
  decision: ['Decision recorded', 'החלטה נרשמה'],
  model_training: ['Model trained', 'המודל אומן'],
  model_version: ['Model version recorded', 'גרסת מודל נרשמה'],
  model_decision: ['Model release decision recorded', 'החלטת הפצת מודל נרשמה'],
  candidate_measure: ['Candidate model measured', 'מודל מועמד נמדד'],
  preview: ['Plan preview', 'תצוגה מקדימה של התוכנית'],
  placement_preview: ['Placement scored', 'מיקום נבדק'],
  price_preview: ['Price change previewed', 'שינוי מחיר בתצוגה מקדימה'],
  price_test: ['Price tested', 'בדיקת מחיר'],
  restriction_preview: ['Restriction previewed', 'הגבלה בתצוגה מקדימה'],
  source_check: ['Source file checked', 'קובץ מקור נבדק'],
  assistant_context: ['Mabat read the page', 'מבט קרא את המסך'],
  assistant_ask: ['Question to Mabat', 'שאלה למבט'],
  other: ['Change', 'שינוי'],
};

// Where an action's own surface lives, so a row is never a dead end. The hash
// is the shell's own route, so every one of these opens a real destination.
// Four kinds of act have no door of their own in this wave and are named here
// as absent rather than pointed at a page that does not answer for them: the
// training acts, which live behind the company shell, and the account acts.
export const ACTION_DOORS = {
  settings_change: 'Settings',
  pricing_change: 'Pricing',
  price_test: 'Pricing',
  price_preview: 'Pricing',
  plan_run: 'Schedule',
  plan_publish: 'Optimizer',
  plan_restore: 'Optimizer',
  target_change: 'Overview',
  restriction_change: 'Settings',
  restriction_preview: 'Settings',
  override_change: 'Overrides',
  placement_change: 'Schedule',
  placement_preview: 'Schedule',
  gold_change: 'Overrides',
  break_change: 'Break Library',
  guardrail_change: 'Settings',
  model_activation_change: 'Settings',
  channel_change: 'Settings',
  client_change: 'Advertisers',
  client_onboarding: 'Advertisers',
  campaign_change: 'Campaigns',
  calendar_change: 'Calendar',
  source_upload: 'Data',
  source_check: 'Data',
  restore_point_saved: 'Versions',
  restore_point_renamed: 'Versions',
  restore: 'Versions',
  assistant_action: 'Assistant',
  assistant_undo: 'Assistant',
  assistant_upload: 'Assistant',
  assistant_ask: 'Assistant',
  assistant_context: 'Assistant',
  conversation_change: 'Assistant',
  decision: 'Overview',
  preview: 'Optimizer',
};

export const SOURCE_LABELS = {
  manual_edit: ['Saved on a surface', 'נשמר במסך'],
  assistant_apply: ['Applied through Mabat', 'אושר דרך מבט'],
  manual_snapshot: ['Saved by hand', 'נשמר ידנית'],
  pre_restore: ['Saved before a restore', 'נשמר לפני שחזור'],
};

export const FILE_LABELS = {
  settings: ['Settings', 'הגדרות'],
  constraints: ['Restrictions', 'הגבלות'],
  overrides: ['Pins', 'נעיצות'],
  advertisers: ['Advertisers', 'מפרסמים'],
  conditions: ['Advertiser conditions', 'תנאי מפרסמים'],
  events: ['Calendar events', 'אירועי יומן'],
  agencies: ['Agencies', 'סוכנויות'],
  agency_links: ['Agency links', 'שיוכי סוכנות'],
  agency_conditions: ['Agency conditions', 'תנאי סוכנות'],
};

// Why a restore point cannot be put back. Each names the cause and what is
// still true, because a refusal without a reason is just a disabled button.
export const RESTORE_BLOCKS = {
  foreign_store: [
    'Recorded against a different store, so restoring it would write files this deployment never produced. It stays readable as history.',
    'נרשמה מול מאגר אחר, ולכן שחזור שלה יכתוב קבצים שההתקנה הזו מעולם לא יצרה. היא נשארת קריאה כהיסטוריה.',
  ],
  missing_snapshot: [
    'The saved copy of one of its files is gone, so it cannot be put back. It stays readable as history.',
    'העותק השמור של אחד הקבצים שלה חסר, ולכן אי אפשר להחזיר אותה. היא נשארת קריאה כהיסטוריה.',
  ],
  no_files: [
    'It covers no file this product knows how to put back.',
    'היא לא מכסה אף קובץ שהמוצר יודע להחזיר.',
  ],
};

export const SIGN_IN_LABELS = {
  login: ['Signed in', 'כניסה למערכת'],
  login_failed: ['Sign-in refused', 'כניסה נדחתה'],
  logout: ['Signed out', 'יציאה מהמערכת'],
};

// Three actors are not people and the recorder stores each as a bare token.
// Rendering the token would put an internal word where a name belongs, and it
// would also hide the honest fact behind it: on a deployment with no login wall
// there is no identity to record, and that is worth reading rather than
// decoding. The row carries the short word and the opened entry carries the
// sentence, which is the same shape the kind filter already uses.
export const ACTOR_LABELS = {
  'auth-disabled': ['No sign-in', 'ללא כניסה'],
  anonymous: ['Not signed in', 'ללא הזדהות'],
  engine: ['The engine', 'המנוע'],
};

export const ACTOR_HINTS = {
  'auth-disabled': [
    'This deployment has no sign-in wall, so no identity was recorded for this act.',
    'בהתקנה הזו אין חומת כניסה, ולכן לא נרשמה זהות לפעולה הזו.',
  ],
  anonymous: [
    'Sign-in is required here and this request carried no live session.',
    'נדרשת כניסה, והבקשה הזו לא נשאה חיבור פעיל.',
  ],
  engine: [
    'Recorded by the engine itself, not by a person.',
    'נרשם על ידי המנוע עצמו, לא על ידי אדם.',
  ],
};

export function actorLabel(actor, locale) {
  return pair(ACTOR_LABELS, actor, locale) || String(actor || '');
}

export function actorHint(actor, locale) {
  return pair(ACTOR_HINTS, actor, locale);
}

// A path with a record id inside it is unreadable on a row and it is the one
// place this surface could put an engine key in front of a person: a break act
// arrives as /api/breaks/2024-11-01|<channel>|000~1/placement. The row prints
// the shape of the act and the detail prints the path in full, so nothing is
// hidden and nothing has to be decoded at a glance.
const ID_LIKE = /^[0-9a-f]{8,}$|[|~]|^\d{4}-\d{2}-\d{2}|^\d+$/;

export function pathStem(path) {
  const text = String(path || '');
  if (!text.startsWith('/api/')) return text;
  const parts = text.split('/').filter(Boolean);
  const kept = parts.map((part) => (ID_LIKE.test(part) ? '…' : part));
  return `/${kept.join('/')}`;
}

// The recorded figures a run reports. The unit decides how each one is read, so
// a percentage is never printed as shekels and a count is never rounded.
export const RUN_FIELDS = [
  ['projected_revenue', 'Expected revenue', 'הכנסה צפויה', 'money'],
  ['total_breaks', 'Breaks', 'ברייקים', 'count'],
  ['total_ad_seconds', 'Ad seconds', 'שניות פרסום', 'count'],
  ['average_retention', 'Average retention', 'שימור ממוצע', 'percent'],
  ['objective', 'Objective', 'ערך המטרה', 'ratio'],
  ['segment_count', 'Programme segments', 'רצועות שידור', 'count'],
];

// What was in force when a run happened, in the operator's words. Fifteen keys,
// measured across all 527 records in the run log and present in every one of
// them, so this is a closed set rather than a sample. An unrecognised key falls
// back to its own name, which is honest and visibly unfinished rather than
// silently dropped.
//
// The words are the product's own: "caution level" is what the settings surface
// already calls risk_lambda, and reusing it means the same number is not read
// under two names in one product.
export const FORCE_LABELS = {
  gold_breaks_max_per_day: ['Gold breaks allowed per day', 'ברייקי זהב מותרים ביום', 'count'],
  max_ad_seconds_per_hour: ['Ad seconds allowed per hour', 'שניות פרסום מותרות בשעה', 'count'],
  max_breaks_per_hour: ['Breaks allowed per hour', 'ברייקים מותרים בשעה', 'count'],
  max_daily_ad_seconds: ['Ad seconds allowed per day', 'שניות פרסום מותרות ביום', 'count'],
  min_break_spacing_seconds: ['Least gap between breaks, in seconds', 'מרווח מזערי בין ברייקים, בשניות', 'count'],
  min_retention_floor: ['Retention floor', 'רצפת שימור', 'fraction'],
  protected_max_ad_seconds_per_hour: ['Ad seconds allowed per hour in protected programmes', 'שניות פרסום מותרות בשעה בתוכניות מוגנות', 'count'],
  protected_program_types: ['Protected programme types', 'סוגי תוכניות מוגנות', 'list'],
  default_break_length_seconds: ['Break length, in seconds', 'אורך ברייק, בשניות', 'count'],
  default_max_breaks: ['Breaks per programme, at most', 'ברייקים לתוכנית, לכל היותר', 'count'],
  first_break_multiplier: ['How much more the first break costs in retention', 'כמה יותר עולה הברייק הראשון בשימור', 'ratio'],
  retention_baseline: ['Retention baseline', 'בסיס השימור', 'ratio'],
  retention_impact_per_break: ['Retention given up per break', 'שימור שנגרע לכל ברייק', 'ratio'],
  revenue_weight: ['Weight on revenue against retention', 'המשקל של ההכנסה מול השימור', 'ratio'],
  risk_lambda: ['Caution level', 'רמת זהירות', 'ratio'],
  // A record rather than a number, and the only one on this readout. It arrives
  // as {window, day_fraction}, both null when no cap is configured, which is the
  // shipped default. Without this row it fell through to String(key) beside an
  // [object Object], printing a raw engine key next to nothing readable, one
  // line above nine guardrails rendered correctly in Hebrew.
  airtime_caps: ['Airtime caps', 'תקרות זמן פרסום', 'record'],
};

// The members of a recorded RECORD, kept out of FORCE_LABELS on purpose. That
// map is pinned by a test to exactly the keys the run log records at the top
// level, and it should stay that way: a nested member is not a recorded
// guardrail, and adding one there would quietly widen what the guard asserts.
const RECORD_MEMBER_LABELS = {
  window: ['Cap over a window of hours', 'תקרה על חלון שעות'],
  day_fraction: ['Cap as a share of the day', 'תקרה כחלק מהיממה'],
};

export function recordMemberLabel(key, locale) {
  const found = RECORD_MEMBER_LABELS[key];
  if (!found) return String(key || '');
  return locale === 'he' ? found[1] : found[0];
}

export function forceLabel(key, locale) {
  const found = FORCE_LABELS[key];
  if (!found) return String(key || '');
  return locale === 'he' ? found[1] : found[0];
}

export function forceUnit(key) {
  const found = FORCE_LABELS[key];
  return found ? found[2] : 'raw';
}

export const VIA_LABELS = {
  assistant: ['Mabat', 'מבט'],
  engine: ['Engine', 'מנוע'],
  dashboard: ['', ''],
};

// The keys this surface answers to, taught on the row that performs each one.
export const KEY_HINTS = [
  ['J', 'Next', 'הבא'],
  ['K', 'Previous', 'הקודם'],
  ['Enter', 'Open', 'פתיחה'],
  ['Esc', 'Close', 'סגירה'],
  ['/', 'Search', 'חיפוש'],
];

export function pair(table, key, locale) {
  const found = table[key];
  if (!found) return '';
  return locale === 'he' ? found[1] : found[0];
}

// The declared zone lives with the dates, not with the words. Re-exported here
// because this file was where the rest of History reached for it.
export { BROADCAST_ZONE };

export function dayHeading(iso, locale) {
  return formatDayWithWeekday(iso, locale) || iso;
}

export const clockLabel = (iso) => formatClock(iso);
export const stampLabel = (iso) => formatStamp(iso);
export { isoDay, todayIso };

// An entry has an address, so it can be linked to, reloaded onto and handed
// over. Mabat hands back a restore point and its "see it in the history" control
// has to land on that point rather than on the top of a list of six thousand
// rows, and a person who finds the row that explains a number has to be able to
// send it to somebody.
//
// The address is a query parameter rather than part of the hash because the
// shell's own router resolves the hash by exact match against its seventeen
// entries, so anything appended to it would route to the wrong page. The search
// string survives a hash change untouched, which is what makes this possible
// without reaching into a frozen file.
export const ADDRESS_PARAM = 'entry';

export function addressOf(entry) {
  return entry && entry.id ? String(entry.id) : '';
}

export function readAddress() {
  if (typeof window === 'undefined') return '';
  return new URLSearchParams(window.location.search).get(ADDRESS_PARAM) || '';
}

export function writeAddress(id) {
  if (typeof window === 'undefined' || !window.history) return;
  const params = new URLSearchParams(window.location.search);
  if (id) params.set(ADDRESS_PARAM, id);
  else params.delete(ADDRESS_PARAM);
  const search = params.toString();
  const next = `${window.location.pathname}${search ? `?${search}` : ''}${window.location.hash}`;
  window.history.replaceState(null, '', next);
}
