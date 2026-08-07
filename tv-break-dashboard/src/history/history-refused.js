// What a recorded act actually did, and the words for the half that did nothing.
//
// The measured defect, taken by a blind critic on 2026-08-02 and reproduced on
// this repository's own recorder on 2026-08-03. The request recorder has stored
// the status the server answered with on every line since the log existed, and
// nothing on this destination read it: the act was derived from the method and
// the path alone, so a write the wall refused carried the sentence of one that
// happened. Four consecutive rows read "the regulatory limit was saved" at the
// same minute, two of them refused at 403, the only difference a small red
// number at the far end. 680 of the 2,264 change entries on the record (30.0
// percent) answered 400 or more; measured on this repository's own store at
// 00:33 on 2026-08-04 it is 811 of 3,263 recorded requests (24.9 percent), of
// which 528 are 403.
//
// So every act has two halves on this surface: what was attempted, which is the
// action code, and whether it landed, which is the outcome. A refused act keeps
// its place on the list, because an attempt somebody made is exactly what a
// person reading this destination needs to see, and it keeps none of the words
// that say it happened.
//
// Tri-state, never two: applied, refused, and the outcome nobody recorded. The
// store holds no line of the third kind today, which is why it is carried.
//
// This is plain JavaScript so the rule can be executed by a test rather than
// grepped, which is the pattern history-runs.js established here. The extension
// on the import is explicit for the same reason: node resolves no other way.
import { ACTION_LABELS, pair } from './history-labels.js';
import { SCOPE_SELF } from './history-since.js';

export const APPLIED = 'applied';
export const REFUSED = 'refused';
export const OUTCOME_UNKNOWN = 'unknown';

// The line the server draws. Kept here as well as on the endpoint so a surface
// reading an older payload still tells the two apart rather than assuming the
// happy one, which is the assumption this whole module exists to remove.
//
// A 5xx is on neither side of it. The server failed rather than declined, and a
// failure can land after a write has begun, so calling it a refusal would be the
// same certainty pointed the other way.
const REFUSED_FROM = 400;
const SERVER_ERROR_FROM = 500;

// What one entry did. The payload says so directly; a payload that does not is
// read from the status code beside it, and a record with neither is unknown.
//
// Both shapes this product stores are accepted: the timeline nests the fields
// under facts and the settings activity log carries them at the top level, and
// naming that here is what lets both surfaces read one rule.
export function outcomeOf(entry) {
  const facts = (entry && entry.facts) || entry || {};
  const stated = String(facts.outcome || '');
  if (stated === APPLIED || stated === REFUSED || stated === OUTCOME_UNKNOWN) return stated;
  const status = Number(facts.status);
  if (!Number.isFinite(status) || status < 100 || status >= SERVER_ERROR_FROM) return OUTCOME_UNKNOWN;
  return status >= REFUSED_FROM ? REFUSED : APPLIED;
}

// The same act, refused. One pair per action code in ACTION_LABELS, so a code a
// route can produce can never fall back to the word for the act that happened.
// The verb is the one the product already uses for a refusal on its own screens:
// a save that was refused is a save that did not happen.
export const REFUSED_LABELS = {
  settings_change: ['Settings save refused', 'שמירת ההגדרות נדחתה'],
  pricing_change: ['Rate card save refused', 'שמירת המחירון נדחתה'],
  plan_run: ['Weekly plan run refused', 'הרצת הלוח השבועי נדחתה'],
  plan_publish: ['Plan version publish refused', 'הפצת גרסת התוכנית נדחתה'],
  plan_restore: ['Plan version restore refused', 'החזרת גרסת התוכנית נדחתה'],
  target_change: ['Target save refused', 'שמירת היעד נדחתה'],
  restriction_change: ['Restriction save refused', 'שמירת ההגבלה נדחתה'],
  override_change: ['Pin save refused', 'שמירת הנעיצה נדחתה'],
  placement_change: ['Break move refused', 'הזזת הברייק נדחתה'],
  gold_change: ['Gold break save refused', 'שמירת ברייק הזהב נדחתה'],
  break_change: ['Break save refused', 'שמירת הברייק נדחתה'],
  guardrail_change: ['Regulatory limit save refused', 'שמירת מגבלת רגולציה נדחתה'],
  model_activation_change: ['Audience model switch change refused', 'שינוי מתג מודל הקהל נדחה'],
  channel_change: ['Operator channel save refused', 'שמירת ערוץ המפעיל נדחתה'],
  client_change: ['Client record save refused', 'שמירת רשומת הלקוח נדחתה'],
  client_onboarding: ['Client onboarding refused', 'קליטת הלקוח נדחתה'],
  campaign_change: ['Campaign save refused', 'שמירת הקמפיין נדחתה'],
  calendar_change: ['Calendar event save refused', 'שמירת האירוע ביומן נדחתה'],
  source_upload: ['Source file upload refused', 'העלאת קובץ המקור נדחתה'],
  restore_point_saved: ['Restore point save refused', 'שמירת נקודת השחזור נדחתה'],
  restore_point_renamed: ['Restore point rename refused', 'שינוי שם נקודת השחזור נדחה'],
  restore: ['Restore refused', 'השחזור נדחה'],
  assistant_action: ['Kai proposal decision refused', 'הכרעת ההצעה של קאי נדחתה'],
  assistant_undo: ['Kai undo refused', 'ביטול השינוי של קאי נדחה'],
  assistant_upload: ['File to Kai refused', 'מסירת הקובץ לקאי נדחתה'],
  conversation_change: ['Kai conversation change refused', 'שינוי השיחה עם קאי נדחה'],
  job_change: ['Job change refused', 'שינוי התפקיד נדחה'],
  password_change: ['Password change refused', 'שינוי הסיסמה נדחה'],
  account_change: ['Account change refused', 'שינוי החשבון נדחה'],
  decision: ['Decision record refused', 'רישום ההחלטה נדחה'],
  model_training: ['Model training refused', 'אימון המודל נדחה'],
  model_version: ['Model version record refused', 'רישום גרסת המודל נדחה'],
  model_decision: ['Ship decision record refused', 'רישום הכרעת השילוח נדחה'],
  candidate_measure: ['Candidate measurement refused', 'מדידת המועמד נדחתה'],
  preview: ['Plan preview refused', 'התצוגה המקדימה של התוכנית נדחתה'],
  placement_preview: ['Placement scoring refused', 'בדיקת המיקום נדחתה'],
  price_preview: ['Price change preview refused', 'התצוגה המקדימה של שינוי המחיר נדחתה'],
  price_test: ['Price test refused', 'בדיקת המחיר נדחתה'],
  restriction_preview: ['Restriction preview refused', 'התצוגה המקדימה של ההגבלה נדחתה'],
  source_check: ['Source file check refused', 'בדיקת קובץ המקור נדחתה'],
  assistant_context: ['Kai page read refused', 'קריאת המסך על ידי קאי נדחתה'],
  assistant_ask: ['Question to Kai refused', 'השאלה לקאי נדחתה'],
  other: ['Change refused', 'השינוי נדחה'],
};

// The short word beside the row, for the two outcomes that are not the happy
// one. Colour is never the only signal on this surface, so the state is a word.
export const OUTCOME_WORDS = {
  refused: ['Refused', 'נדחתה'],
  unknown: ['Result unknown', 'התוצאה לא ידועה'],
};

// What the opened entry says about it, in the place the preview note already
// occupies, because a reader who opened a row is asking what it means.
export const OUTCOME_NOTES = {
  refused: [
    'This request was refused, so nothing was saved and there is nothing here to put back. The result beside it is the answer the server gave.',
    'הבקשה נדחתה, ולכן לא נשמר דבר ואין כאן מה להחזיר. התוצאה שלצידה היא התשובה שהשרת החזיר.',
  ],
  unknown: [
    'The recorded result does not say whether anything was saved, so this product claims neither. A request the server failed on can leave part of a write behind.',
    'התוצאה שנרשמה לא אומרת אם נשמר משהו, ולכן המוצר לא קובע כאן דבר. בקשה שהשרת נכשל בה יכולה להשאיר אחריה חלק מהכתיבה.',
  ],
};

// The door out of a row, worded for what actually happened. A refused act gets
// none: "open the surface this changed" for a request that changed nothing is
// the same sentence as the row that started this, one line lower.
const DOOR_WORDS = {
  applied: ['Open the surface this changed', 'פתחו את המסך שהשתנה'],
  preview: ['Open the surface this was computed for', 'פתחו את המסך שעבורו זה חושב'],
  unknown: ['Open the surface this act was on', 'פתחו את המסך שהפעולה נגעה בו'],
};

// The act, in the operator's own words, told by what it did. An unknown code
// falls back to the same "Change" it always did, and to the refusal of it.
export function actLabel(action, outcome, locale) {
  const table = outcome === REFUSED ? REFUSED_LABELS : ACTION_LABELS;
  return pair(table, action, locale) || pair(table, 'other', locale);
}

export function outcomeWord(outcome, locale) {
  return pair(OUTCOME_WORDS, outcome, locale);
}

export function outcomeNote(outcome, locale) {
  return pair(OUTCOME_NOTES, outcome, locale);
}

// The label on the door, or an empty string when this act opens none.
export function doorLabel(outcome, preview, locale) {
  if (outcome === REFUSED) return '';
  if (preview) return pair(DOOR_WORDS, 'preview', locale);
  return pair(DOOR_WORDS, outcome === APPLIED ? 'applied' : 'unknown', locale);
}

function count(value, locale) {
  return Number(value || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB');
}

// What the figure beside the Change filter is, so the tally on that control is
// never read as a count of things that happened. Singular is written out,
// because "1 of these were refused" is the kind of sentence a reader stops on.
export function refusedTabLine(refused) {
  if (Number(refused) === 1) {
    return ['One of these was refused and changed nothing.', 'אחת מהן נדחתה ולא שינתה דבר.'];
  }
  return [
    `${count(refused, 'en')} of these were refused and changed nothing.`,
    `נדחו ${count(refused, 'he')} מהן ולא שינו דבר.`,
  ];
}

// The sentence beside the attestation, which is the figure a compliance owner
// reads first. It is printed whenever anything was refused, so what happened and
// what was attempted are both on the strip and are never one number.
//
// This figure is tallied over the "change" kind alone (`outcome_counts` in
// `history_api_timeline.py` skips every other kind), and the activity list it
// draws from is the one the activity-scope filter already narrows to the
// caller's own account when `scope` is "self" (`history_api._assemble`, ahead
// of the merge). So a self-scoped refused count is never a mix the way the
// count sentence's is: it is only ever this account's own refused attempts.
// The sentence still has to say so, because a reader of the strip cannot see
// which rule produced the number, only the number itself, and the same
// unqualified sentence was measured printing "160" for an admin and "2" for a
// self-scoped reader on the identical store in the same minute with no word
// telling the two apart. `sinceCountLine` and `sinceEmptyLine` already take
// this argument; this is the third and last sentence on the strip that did not.
export function refusedSinceLine(refused, scope) {
  if (scope === SCOPE_SELF) {
    if (Number(refused) === 1) {
      return [
        'One of your own attempts was refused and changed nothing.',
        'ניסיון אחד משלכם נדחה ולא שינה דבר.',
      ];
    }
    return [
      `${count(refused, 'en')} of your own attempts were refused and changed nothing.`,
      `נדחו ${count(refused, 'he')} מהניסיונות שלכם ולא שינו דבר.`,
    ];
  }
  if (Number(refused) === 1) {
    return ['One attempt was refused and changed nothing.', 'ניסיון אחד נדחה ולא שינה דבר.'];
  }
  return [
    `${count(refused, 'en')} attempts were refused and changed nothing.`,
    `נדחו ${count(refused, 'he')} ניסיונות שלא שינו דבר.`,
  ];
}
