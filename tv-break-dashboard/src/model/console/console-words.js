// The model console's own words, in both languages, one concept per key.
//
// The canonical product terms come from the frozen vocabulary module and are
// re-exported through `word` rather than repeated here, so a term this console
// shares with the rest of the product cannot drift. Only the words that exist
// nowhere else in the product are defined below.

import { word } from '../../vocabulary.js';

export { word };

const WORDS = {
  'console.title': { en: 'Model governance', he: 'ממשל מודל' },
  'console.marker': { en: 'Company staff only', he: 'לצוות החברה בלבד' },
  'console.back': { en: 'Return to operations', he: 'חזרה למערכת התפעולית' },
  'console.subtitle': {
    en: 'Operational status, evidence, training runs and release decisions for the audience model.',
    he: 'מצב תפעולי, ראיות, הרצות אימון והחלטות הפצה עבור מודל הקהל.',
  },

  'section.gates': { en: 'Gates', he: 'שערים' },
  'section.coverage': { en: 'Coverage', he: 'כיסוי' },
  'section.drift': { en: 'Drift', he: 'סחיפה' },
  'section.candidates': { en: 'Candidates', he: 'מועמדים' },
  'section.training': { en: 'Training', he: 'אימון' },
  'section.versions': { en: 'Model versions', he: 'גרסאות מודל' },
  'section.provenance': { en: 'Lineage', he: 'ייחוס ומקורות' },

  'header.version': { en: 'Model version', he: 'גרסת מודל' },
  'header.trained': { en: 'trained at', he: 'אומנה ב' },
  'header.recorded': { en: 'Recorded', he: 'רשומה' },
  'header.not_recorded': { en: 'Not recorded yet', he: 'טרם נרשמה' },
  'header.record': { en: 'Record this version', he: 'רישום הגרסה הזו' },
  'header.identity': { en: 'Identity', he: 'זהות' },
  'header.activation': { en: 'Audience model in runs', he: 'מודל הקהל בהרצות' },
  'header.activation_on': { en: 'On', he: 'דלוק' },
  'header.activation_off': { en: 'Off', he: 'כבוי' },
  'header.activation_no_artifact': { en: 'On, nothing trained', he: 'דלוק, אין מודל מאומן' },
  'header.control_on_rules': { en: 'Open the activation control in Governance', he: 'פתיחת בקרת ההפעלה בממשל' },
  'header.gates_measured_at': { en: 'Gate verdicts measured on', he: 'הכרעות השערים נמדדו בתאריך' },
  'header.who_may_train': { en: 'Who may run a rebuild', he: 'מי רשאי להריץ אימון מחדש' },
  'header.can_edit_yes': {
    en: 'This account may start a run and record a decision.',
    he: 'לחשבון הזה יש הרשאה להריץ אימון ולתעד הכרעה.',
  },

  'gates.of': { en: 'of', he: 'מתוך' },
  'gates.filter': { en: 'Filter gates', he: 'סינון שערים' },
  'gates.all': { en: 'All', he: 'הכול' },
  'gates.basis': { en: 'Basis', he: 'בסיס' },
  'gates.measured': { en: 'measured', he: 'נמדד' },
  'gates.bar': { en: 'bar', he: 'רף' },
  'gates.observations': { en: 'observations', he: 'תצפיות' },
  'gates.folds': { en: 'folds', he: 'קיפולים' },
  'gates.bar_from': { en: 'Bar read from', he: 'הרף נקרא מ' },
  'gates.show_record': { en: 'Show the record behind this', he: 'הצגת הרישום שמאחורי זה' },
  'gates.hide_record': { en: 'Hide the record', he: 'הסתרת הרישום' },
  'gates.no_figure': { en: 'no figure: the gate could not run', he: 'אין מספר: השער לא יכול היה לרוץ' },
  'gates.retention': { en: 'Predicted retention', he: 'שימור חזוי' },
  'gates.audience': { en: 'Expected rating', he: 'רייטינג צפוי' },
  'gates.layers': { en: 'Operator-controlled layers', he: 'שכבות בשליטת המפעיל' },
  'gates.layer_on': { en: 'On', he: 'דלוקה' },
  'gates.layer_off': { en: 'Off', he: 'כבויה' },
  'gates.unblock': { en: 'Requirement to change this status', he: 'דרישה לשינוי המצב' },
  'gates.none_in_state': { en: 'No gate is in this state right now.', he: 'אף שער אינו במצב הזה כרגע.' },
  'gates.none_recorded': { en: 'No gate is recorded at all.', he: 'לא נרשם אף שער.' },
  'gates.show_all': { en: 'Show all gates', he: 'הצגת כל השערים' },

  'coverage.window': { en: 'Training window', he: 'חלון האימון' },
  'coverage.days': { en: 'days', he: 'ימים' },
  'coverage.breaks': { en: 'measured breaks', he: 'ברייקים נמדדים' },
  'coverage.cells': { en: 'cells', he: 'תאים' },
  'coverage.per_cell': { en: 'observations per cell, smallest to largest', he: 'תצפיות לתא, מהקטן לגדול' },
  'coverage.ratio': { en: 'Contrast ratio', he: 'יחס ניגודיות' },
  'coverage.retention': { en: 'Predicted retention: the cells', he: 'שימור חזוי: התאים' },
  'coverage.audience': { en: 'Expected rating: the base', he: 'רייטינג צפוי: הבסיס' },
  'coverage.levels': { en: 'levels', he: 'רמות' },
  'coverage.blocked': { en: 'Data requirements not met', he: 'דרישות הנתונים אינן מתקיימות' },
  'coverage.condition': { en: 'Requirement to remove the block', he: 'דרישה להסרת החסימה' },
  'coverage.earliest': { en: 'Earliest', he: 'המועד המוקדם ביותר' },
  'coverage.earliest_unknown': { en: 'No date can be computed for this one', he: 'לא ניתן לחשב תאריך לזה' },
  'coverage.channels_counted': { en: 'channels in the base', he: 'ערוצים בבסיס' },
  'coverage.counted': { en: 'What was counted', he: 'מה נספר' },
  'coverage.evidence.days_in_window': { en: 'days in the window', he: 'ימים בחלון' },
  'coverage.evidence.event_free_days_in_window': { en: 'days with no active event', he: 'ימים ללא אירוע פעיל' },
  'coverage.evidence.seasons_in_window': { en: 'seasons in the window', he: 'עונות בחלון' },
  'coverage.supply': { en: 'Who supplies it', he: 'מי מספק את זה' },
  'coverage.supply_store': {
    en: 'The event store the operator owns. Adding an event there, or ending one, is what ends this block.',
    he: 'מאגר האירועים של המפעיל. הוספת אירוע שם, או סיום אירוע קיים, היא מה שמסיים את החסימה.',
  },
  'coverage.supply_store_open': { en: 'Open the event calendar in Rules', he: 'פתיחת לוח האירועים בכללים' },
  'coverage.supply_time': {
    en: 'Nobody can supply this one. Only history that covers the condition above ends the block.',
    he: 'אף אחד אינו יכול לספק את זה. רק היסטוריה שמכסה את התנאי שלמעלה תסיים את החסימה.',
  },
  'coverage.supply_unknown': {
    en: 'This screen cannot say who supplies this source, so it does not guess.',
    he: 'המסך הזה אינו יכול לומר מי מספק את המקור הזה, ולכן הוא אינו מנחש.',
  },
  'coverage.between_within': { en: 'between cells over within cells', he: 'בין התאים חלקי בתוך התא' },

  'drift.per_week': { en: 'Drift per week', he: 'סחיפה לשבוע' },
  'drift.threshold': { en: 'Binding threshold', he: 'סף מחייב' },
  'drift.binding': { en: 'Above the threshold', he: 'מעל הסף' },
  'drift.stable': { en: 'Within the threshold', he: 'בתוך הסף' },
  'drift.unknown': { en: 'Not determined', he: 'לא נקבע' },
  'drift.weekly': { en: 'Weekly mean level', he: 'רמה שבועית ממוצעת' },
  'drift.week': { en: 'Week', he: 'שבוע' },
  'drift.criterion': { en: 'The rule behind this verdict', he: 'הכלל שמאחורי ההכרעה' },
  'drift.series': { en: 'Across model versions', he: 'לאורך גרסאות מודל' },

  'candidates.title': { en: 'Candidate models', he: 'מודלים מועמדים' },
  'candidates.subject': { en: 'Scope of model change', he: 'היקף השינוי במודל' },
  'candidates.gate_deltas': { en: 'Gate differences from the released model', he: 'הבדלי שערים מהמודל המופץ' },
  'candidates.no_gate_deltas': { en: 'No gate decides differently', he: 'אף שער אינו מכריע אחרת' },
  'candidates.key_absent': { en: 'the artifact does not record this', he: 'הקובץ אינו רושם את זה' },
  'candidates.value_null': { en: 'recorded with no value', he: 'נרשם ללא ערך' },
  'candidates.cells_moved': { en: 'coefficient cells changed', he: 'תאי מקדם שהשתנו' },
  'candidates.largest_move': { en: 'largest coefficient delta', he: 'השינוי הגדול ביותר במקדם' },
  'candidates.money': { en: 'Projected revenue impact', he: 'השפעה חזויה על ההכנסה' },
  'candidates.not_measured': { en: 'Not measured yet', he: 'טרם נמדד' },
  'candidates.measure': { en: 'Measure it', he: 'למדוד' },
  'candidates.measuring': { en: 'Computing the plan comparison', he: 'מחשב את השוואת התוכנית' },
  'candidates.stale': { en: 'An input changed since this was measured', he: 'קלט השתנה מאז המדידה' },
  'candidates.remeasure': { en: 'Measure it again', he: 'למדוד שוב' },
  'candidates.watching': { en: 'Updating until the measurement completes.', he: 'הנתונים מתעדכנים עד להשלמת המדידה.' },
  'candidates.watch_lost': { en: 'The measurement could not be read again, so this screen stopped following it.', he: 'לא ניתן היה לקרוא שוב את המדידה, ולכן המסך הפסיק לעקוב אחריה.' },
  'candidates.watch_again': { en: 'Read it again', he: 'קריאה חוזרת' },
  'candidates.scope_owned': { en: 'operator channel', he: 'ערוץ המפעיל' },
  'candidates.scope_plan': { en: 'whole plan', he: 'כל התוכנית' },
  'candidates.rows': { en: 'rows', he: 'שורות' },
  'candidates.decide': { en: 'Record a release decision', he: 'רישום החלטת הפצה' },
  'candidates.decide_again': { en: 'Record a new release decision', he: 'רישום החלטת הפצה חדשה' },
  'candidates.past_runs': { en: 'past measurements took', he: 'מדידות קודמות ארכו' },
  'candidates.seconds': { en: 'seconds', he: 'שניות' },
  'candidates.channels': { en: 'channels', he: 'ערוצים' },
  'candidates.measured_at': { en: 'measured at', he: 'נמדד ב' },
  'candidates.engine': { en: 'engine', he: 'מנוע' },
  'candidates.side_shipped': { en: 'released', he: 'מופץ' },
  'candidates.side_candidate': { en: 'candidate', he: 'מועמד' },
  'candidates.held_out': { en: 'The held-out figures behind those gates', he: 'המספרים מחוץ למדגם שמאחורי אותם שערים' },
  'candidates.no_held_out_moves': { en: 'No held-out figure differs between the two artifacts.', he: 'אין הבדל במספרים מחוץ למדגם בין שני הקבצים.' },
  'candidates.candidate_records_nothing': { en: 'The candidate records nothing for this gate, so the two cannot be compared on it.', he: 'המועמד אינו רושם דבר עבור השער הזה, ולכן לא ניתן להשוות ביניהם עליו.' },
  'candidates.shipped_records_nothing': { en: 'The released artifact records nothing for this gate, so the two cannot be compared on it.', he: 'הקובץ המופץ אינו רושם דבר עבור השער הזה, ולכן לא ניתן להשוות ביניהם עליו.' },
  'candidates.no_sentence': { en: 'no sentence recorded', he: 'לא נרשם משפט' },

  'candidates.verdict': { en: 'The verdict on this candidate', he: 'ההכרעה על המועמד הזה' },
  'candidates.verdict_reading': { en: 'Reading the verdict', he: 'קורא את ההכרעה' },
  'candidates.no_verdict': { en: 'No verdict has been recorded about this candidate against the model version in force,', he: 'טרם נרשמה הכרעה על המועמד הזה מול גרסת המודל שבתוקף,' },
  'candidates.verdict_unreadable': { en: 'The verdict could not be read, so none is shown rather than a guess.', he: 'לא ניתן היה לקרוא את ההכרעה, ולכן לא מוצגת אחת במקום ניחוש.' },
  'candidates.verdict_on': { en: 'Recorded on', he: 'נרשמה על' },
  'candidates.verdict_not_measured': { en: 'Recorded with no measurement: this candidate\'s projected revenue impact was never measured.', he: 'נרשמה ללא מדידה: השפעת המועמד על ההכנסה החזויה מעולם לא נמדדה.' },
  'candidates.verdict_stale': { en: 'An input changed after that measurement, so the figure used for the verdict is superseded.', he: 'קלט השתנה לאחר המדידה, ולכן המספר ששימש לקבלת ההכרעה אינו עדכני.' },
  'candidates.verdict_record': { en: 'Show the verdict record behind this', he: 'הצגת רישום ההכרעה שמאחורי זה' },
  'candidates.adoption_recorded': { en: 'Recorded, not adopted', he: 'נרשמה, לא הוטמעה' },

  'training.title': { en: 'Start a training run', he: 'הפעלת הרצת אימון' },
  'training.log': { en: 'Runs started from this console', he: 'הרצות שהופעלו מהקונסולה הזו' },
  'training.start': { en: 'Start training run', he: 'הפעלת הרצת אימון' },
  'training.running': { en: 'Training run in progress', he: 'הרצת האימון מתבצעת' },
  'training.overrides': { en: 'Gate settings for this run', he: 'הגדרות שערים להרצה הזו' },
  'training.auto': { en: 'Let the gate decide', he: 'שהשער יכריע' },
  'training.force_on': { en: 'Force on', he: 'לכפות דלוק' },
  'training.force_off': { en: 'Force off', he: 'לכפות כבוי' },
  'training.writes': { en: 'Run output targets', he: 'יעדי הכתיבה של ההרצה' },
  'training.no_runs': { en: 'No training run has been started from this console yet.', he: 'טרם הופעלה הרצת אימון מהקונסולה הזו.' },
  'training.would_change': { en: 'Output scope', he: 'היקף הפלט' },
  'training.took': { en: 'took', he: 'ארכה' },
  'training.failed': { en: 'Failed', he: 'נכשלה' },
  'training.done': { en: 'Done', he: 'הסתיימה' },
  'training.watching': { en: 'Run status updates until completion.', he: 'מצב ההרצה מתעדכן עד להשלמתה.' },
  'training.watch_lost': { en: 'The run could not be read again, so this screen stopped following it.', he: 'לא ניתן היה לקרוא שוב את ההרצה, ולכן המסך הפסיק לעקוב אחריה.' },
  'training.watch_again': { en: 'Read it again', he: 'קריאה חוזרת' },

  'versions.title': { en: 'Model versions and human decisions', he: 'גרסאות מודל והחלטות אנושיות' },
  'versions.recorded': { en: 'Recorded versions', he: 'גרסאות רשומות' },
  'versions.decisions': { en: 'Decisions', he: 'הכרעות' },
  'versions.none': { en: 'No decision has been recorded yet.', he: 'טרם נרשמה הכרעה.' },
  'versions.ship': { en: 'Approve for release', he: 'אישור להפצה' },
  'versions.no_ship': { en: 'Reject for release', he: 'דחייה להפצה' },
  'versions.reason': { en: 'Decision rationale', he: 'נימוק ההחלטה' },
  'versions.note': { en: 'Release note for the operator side', he: 'הערת גרסה לצד המפעיל' },
  'versions.note_rule': {
    en: 'Plain language. It may not carry a gate verdict, a p-value or a coefficient.',
    he: 'שפה פשוטה. היא אינה יכולה לשאת הכרעת שער, ערך מובהקות או מקדם.',
  },
  'versions.record': { en: 'Record the decision', he: 'רישום ההכרעה' },
  'versions.cancel': { en: 'Cancel', he: 'ביטול' },
  'versions.shipped': { en: 'Approved for release', he: 'אושרה להפצה' },
  'versions.not_shipped': { en: 'Rejected for release', he: 'נדחתה להפצה' },
  'versions.escalated': { en: 'Adoption escalated, not performed', he: 'ההטמעה הועברה לאישור, לא בוצעה' },
  'versions.by': { en: 'by', he: 'על ידי' },
  'versions.subject_current': { en: 'the released model', he: 'המודל המופץ' },
  'versions.subject_candidate': { en: 'candidate', he: 'מועמד' },

  'provenance.title': { en: 'Model lineage', he: 'ייחוס המודל' },
  'provenance.artifacts': { en: 'Artifacts and recorded input sources', he: 'קובצי מודל ומקורות קלט מתועדים' },
  'provenance.read_from': { en: 'Read from', he: 'נקרא מתוך' },
  'provenance.no_fingerprints': { en: 'This artifact records no input fingerprints, so what it was fitted on cannot be checked from the file.', he: 'הקובץ הזה אינו רושם טביעות אצבע של קלטים, ולכן לא ניתן לבדוק ממנו על מה הוא הותאם.' },
  'provenance.no_artifacts': { en: 'No trained artifact is on disk.', he: 'אין בדיסק קובץ מאומן.' },
  'provenance.digest': { en: 'Content digest', he: 'טביעת אצבע של התוכן' },
  'provenance.trained_at': { en: 'Trained at', he: 'אומן ב' },
  'provenance.method': { en: 'Method and seeds', he: 'שיטה וזרעים' },
  'provenance.pooling': { en: 'Pooling', he: 'איחוד' },
  'provenance.interval': { en: 'Intervals', he: 'רווחי ביטחון' },
  'provenance.detrend': { en: 'Detrend baseline', he: 'בסיס ניכוי מגמה' },
  'provenance.window': { en: 'Before and after window, minutes', he: 'חלון לפני ואחרי, דקות' },
  'provenance.audience_base': { en: 'Expected rating base', he: 'בסיס הרייטינג הצפוי' },
  'provenance.interval_seed': { en: 'Interval seed', he: 'זרע הרווחים' },
  'provenance.bootstrap': { en: 'Bootstrap draws', he: 'דגימות בוטסטרפ' },
  'provenance.placebo_seed': { en: 'Placebo seed', he: 'זרע הפלצבו' },
  'provenance.flags': { en: 'Gate override flags', he: 'דגלי עקיפת שערים' },
  'provenance.flag_recorded': { en: 'Recorded in the artifact', he: 'נרשם בקובץ' },
  'provenance.flag_not_recorded': { en: 'Not recorded in the artifact', he: 'אינו נרשם בקובץ' },
  'provenance.actor': { en: 'Who ran it', he: 'מי הריץ' },
  'provenance.commands': { en: 'Commands capable of replacing the released artifact', he: 'פקודות שיכולות להחליף את קובץ המודל המופץ' },
  'provenance.commands_sub': { en: 'This console runs neither of these. Its own training runs write into the releases store instead.', he: 'הקונסולה הזו אינה מריצה אף אחת מהן. הרצות האימון שלה כותבות למאגר הגרסאות במקום.' },

  'state.loading': { en: 'Loading model records', he: 'טוען את רשומות המודל' },
  'state.refused': { en: 'This view is company staff only.', he: 'התצוגה הזו שמורה לצוות החברה.' },
  'state.unreachable': { en: 'The model service did not respond. No data is shown.', he: 'שירות המודל לא השיב. לא מוצגים נתונים.' },
  'state.retry': { en: 'Try again', he: 'לנסות שוב' },
  'state.shortcuts': { en: 'Shortcuts', he: 'קיצורים' },
};

// The server's own words for why an account may not act here, carried through
// rather than re-invented, so the sentence a person reads before the click and
// the one a 403 would carry cannot drift. English is a mirror of the one
// detail the wall can actually send a company account that already passed the
// affiliation gate: a viewer's role refusal. An unrecognised string still
// renders, in its own words, rather than nothing.
const CAN_EDIT_REASONS_EN = new Map([
  ['לחשבון צפייה אין הרשאת עריכה', 'A viewing account has no edit permission.'],
]);

export function canEditReason(reason, locale) {
  const text = String(reason || '').trim();
  if (!text || locale === 'he') return text;
  return CAN_EDIT_REASONS_EN.get(text) || text;
}

export function t(key, locale = 'he') {
  const entry = WORDS[key];
  if (!entry) return '';
  return locale === 'en' ? entry.en : entry.he;
}

// A bilingual pair the API already carries as two fields, read in one call, so
// a payload that only has one of them still renders rather than blanking.
export function pick(payload, base, locale = 'he') {
  if (!payload) return '';
  const value = locale === 'en' ? payload[`${base}_en`] : payload[`${base}_he`];
  return value || payload[`${base}_en`] || payload[`${base}_he`] || '';
}
