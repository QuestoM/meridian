// The words a stored row is read in, in both languages.
//
// A restore preview shows rows out of eight stores, and a row is a record with
// the engine's own column names on it. The diff endpoint sends that record
// verbatim and says nothing about what it is, which is what keeps the payload
// free of either language. So the surface names it, and it names it with the
// words the store's own destination already uses: the restriction effects are
// the ones the restriction builder offers, the condition effects and modes are
// the ones a client's conditions are edited with, the pin kinds are the ones
// the day board writes, and the event types are the ones the calendar manages.
// One vocabulary, chosen by the surface rather than by the store.
//
// history-labels.js re-exports every table below, so the destination still has
// one import for its words. They sit in a file of their own only because that
// one is at 388 lines against the 450-line cap.
//
// Two vocabularies are read from where they already live rather than copied:
// the daypart names from the shell's own daypartLabel, and the programme genres
// from the frozen vocabulary.js.

// What a restriction row does, and what an advertiser or agency condition does.
// The six restriction effects are the engine's own set in
// kairos/optimize/constraints_store.py; the four condition effects are the set
// in kairos/optimize/_rule_models.py. forbid is in both and reads the same.
export const ROW_EFFECTS = {
  fix_offset: ['Fix offset', 'היסט קבוע'],
  offset_window: ['Offset window', 'חלון היסט'],
  pin_count: ['Pin count', 'מספר ברייקים קבוע'],
  duration_range: ['Duration range', 'טווח אורך'],
  gold: ['Gold break', 'ברייק זהב'],
  forbid: ['Forbid', 'איסור'],
  premium: ['Coefficient', 'מקדם'],
  require: ['Require', 'חובה'],
  pressure: ['Placement preference', 'העדפת שיבוץ'],
};

// What a pin row decides. The four segment kinds and the two spot kinds are the
// engine's own set in kairos/optimize/overrides.py.
export const ROW_KINDS = {
  pin: ['Pin', 'נעיצה'],
  force: ['Forced break count', 'קיבוע מספר ברייקים'],
  forbid: ['No breaks here', 'מניעת ברייקים כאן'],
  gold: ['Gold break', 'ברייק זהב'],
  lock: ['Locked spot', 'תשדיר נעוץ'],
  move: ['Moved spot', 'תשדיר מוזז'],
};

// How a condition's value is read.
export const ROW_MODES = {
  multiplier: ['Multiplier', 'מכפיל'],
  percent: ['Percent', 'אחוז'],
  cpp_absolute: ['CPP absolute', 'נקודה מוחלטת'],
  cpp_add: ['CPP add', 'תוספת לנקודה'],
  cpp_discount: ['CPP discount', 'הנחה מהנקודה'],
  premium_discount: ['Surcharge discount', 'הנחה על תוספת המחיר'],
};

export const ROW_EVENT_TYPES = {
  holiday: ['Holiday', 'חג'],
  war: ['War', 'מלחמה'],
  special: ['Special event', 'אירוע מיוחד'],
  sport: ['Sport', 'ספורט'],
  other: ['Other', 'אחר'],
};

// The engine's daypart keys. The same five pairs the shell's own daypartLabel
// carries, copied rather than imported because that module reads the build
// environment at load and so cannot be executed by a test. An unknown key
// passes through verbatim there and here.
export const ROW_DAYPARTS = {
  morning: ['Morning', 'בוקר'],
  noon: ['Noon', 'צהריים'],
  evening: ['Evening', 'ערב'],
  prime: ['Prime time', 'פריים טיים'],
  night: ['Night', 'לילה'],
  unclassified: ['Unclassified', 'ללא סיווג'],
};

// Where a stored row came from.
export const ROW_SOURCES = {
  observed: ['Seen in the data', 'נצפה בנתונים'],
  manual: ['Entered by hand', 'הוזן ידנית'],
  synthetic: ['Demo seed', 'נתוני הדגמה'],
  recommendation: ['From a recommendation', 'מהמלצה'],
};

// The predicate's weekday tokens, as an index into the Sunday-first week.
export const ROW_WEEKDAY_INDEX = {
  Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6,
};

// What each part of an identity is. A part is printed only when the row carries
// it, so a name here is never a promise that a value exists.
export const ROW_PARTS = {
  advertiser: ['Advertiser', 'מפרסם'],
  agency: ['Agency', 'סוכנות'],
  applies_to: ['Applies to', 'חל על'],
  asked_by: ['Asked by', 'נדרש על ידי'],
  breaks: ['Breaks', 'ברייקים'],
  effect: ['Effect', 'השפעה'],
  expires: ['Stops applying', 'יפסיק לחול'],
  id: ['Id', 'מזהה'],
  intensity: ['Intensity', 'עוצמה'],
  length: ['Length in seconds', 'אורך בשניות'],
  note: ['Note', 'הערה'],
  offset: ['Offset in seconds', 'היסט בשניות'],
  positions: ['Positions', 'מיקומים'],
  price: ['Price multiplier', 'מכפיל מחיר'],
  programme: ['Programme', 'תוכנית'],
  reason: ['Reason', 'סיבה'],
  scope: ['Scope', 'היקף'],
  seen_on: ['First seen', 'נצפה לראשונה'],
  source: ['Source', 'מקור'],
  starts: ['Starts applying', 'מתחיל לחול'],
  status: ['Status', 'מצב'],
  type: ['Type', 'סוג'],
  when: ['When', 'מתי'],
  worth: ['Value', 'ערך'],
};

// A row that names itself in no way its own store recognises. Each says what
// the row is and what it is missing, because a blank chip and a chip that says
// nothing is missing are the same lie in different type.
export const ROW_UNNAMED = {
  constraints: ['A restriction with no note and no effect on it', 'הגבלה בלי הערה ובלי השפעה'],
  overrides: ['A pin with no kind recorded', 'נעיצה בלי סוג רשום'],
  conditions: ['A condition with no advertiser on it', 'תנאי בלי מפרסם'],
  agency_conditions: ['A condition with no agency on it', 'תנאי בלי סוכנות'],
  events: ['An event with no name', 'אירוע בלי שם'],
  agencies: ['An agency with no name', 'סוכנות בלי שם'],
  agency_links: ['A link with no client on it', 'שיוך בלי לקוח'],
};

// A predicate that will not parse. Unknown, which is not the same as absent.
export const ROW_UNREADABLE = ['Recorded in a shape this surface cannot read', 'נרשם במבנה שהמסך הזה לא יודע לקרוא'];

// The one legacy scope shape that can carry a channel name. This surface cannot
// tell the operator's own channel from a rival's, so it names the shape and
// withholds the value rather than printing a name it cannot vouch for.
export const ROW_CHANNEL_SCOPE = ['Scoped to a channel, which this preview does not name', 'בהיקף של ערוץ, והמסך הזה לא נוקב בשמו'];

// The rest of a predicate this surface will not put into words: a condition on
// a field outside the six, an operator other than is, or a nested group. It is
// counted rather than guessed at.
export const ROW_MORE = ['more conditions on it', 'תנאים נוספים עליה'];

export function rowWord(table, key, locale) {
  const found = table[String(key ?? '').trim().toLowerCase()];
  if (!found) return '';
  return locale === 'he' ? found[1] : found[0];
}

export function rowPhrase(phrase, locale) {
  return locale === 'he' ? phrase[1] : phrase[0];
}
