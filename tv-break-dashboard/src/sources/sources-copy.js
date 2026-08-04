// Every word the Sources destination renders, in both languages, in one place.
//
// The state vocabulary is closed and it is the server's: six words, and the
// server decides which one an input is in. A surface that invents a seventh
// word or re-derives the state from three booleans is how two screens end up
// disagreeing about whether the engine is reading a file.

export const VIEWS = ['inputs', 'files', 'downloads'];

export const VIEW_LABELS = {
  inputs: { en: "Today's inputs", he: 'הקלטים של היום' },
  files: { en: 'Source files', he: 'קבצי מקור' },
  downloads: { en: 'Reports', he: 'דוחות' },
};

// The rail entry each view belongs to, so the hash the shell reads stays in
// step with the view the content shows and the rail lights the right entry.
export const VIEW_HASH = {
  inputs: 'Data',
  files: 'Data',
  downloads: 'Reports',
};

// Six words, and no two of them are the same word. Shadowed and not read are
// different news: one means another file of the same kind is read instead, and
// the other means nothing reads this kind at all, so they never share a label.
// Empty is a third piece of news again: the engine does read this file and it
// carries nothing, which no amount of green would say.
export const STATE_LABELS = {
  in_use: { en: 'In use', he: 'בשימוש' },
  shadowed: { en: 'Another file is read', he: 'נקרא קובץ אחר במקום' },
  not_read: { en: 'Nothing reads it', he: 'שום דבר לא קורא אותו' },
  empty: { en: 'Read, and it has no rows', he: 'נקרא, ואין בו שורות' },
  invalid: { en: 'Needs a fix', he: 'דורש תיקון' },
  missing: { en: 'No file yet', he: 'אין קובץ עדיין' },
};

// The tone each state carries. Only a state the engine really reads, from a
// file that really carries rows, is green.
export const STATE_TONE = {
  in_use: 'ok',
  shadowed: 'warn',
  not_read: 'warn',
  empty: 'warn',
  invalid: 'bad',
  missing: 'muted',
};

export const FILTER_ORDER = ['all', 'in_use', 'shadowed', 'not_read', 'empty', 'invalid', 'missing'];

export const FILTER_LABELS = {
  all: { en: 'All inputs', he: 'כל הקלטים' },
  ...STATE_LABELS,
};

export const CADENCE_LABELS = {
  weekly: { en: 'Weekly', he: 'שבועי' },
  daily: { en: 'Daily', he: 'יומי' },
  reference: { en: 'Reference', he: 'רפרנס' },
  config: { en: 'Configuration', he: 'תצורה' },
};

// What each cadence means, kept from the grouped upload page it replaced, so a
// first-day reader still learns why one file arrives every morning and another
// one twice a year.
export const CADENCE_NOTES = {
  weekly: { en: 'The channel programme lineup, refreshed at the start of each week.', he: 'לוח התוכניות של הערוץ, מתעדכן בתחילת כל שבוע.' },
  daily: { en: "The day's booked ads, loaded each morning for the next broadcast day.", he: 'הפרסומות שהוזמנו ליום, נטענות בכל בוקר ליום השידור הבא.' },
  reference: { en: 'Historical ratings the model is measured on. Refreshed occasionally, not every day.', he: 'נתוני רייטינג היסטוריים שעליהם נמדד המודל. מתרעננים מדי פעם, לא מדי יום.' },
  config: { en: 'Not channel data: advertiser terms, the rate card, and campaign flights with delivery goals.', he: 'לא נתוני ערוץ: תנאי מפרסמים, מחירון וקמפיינים עם יעדי אספקה.' },
};

// What a finding is about when it is about no single column: the whole file,
// its header row, or the table the loader parsed it into. The server sends a
// code and never a word, because the word is read in two languages, and a
// finding that IS about a column sends the column's own name and no code.
export const SCOPE_LABELS = {
  file: { en: 'The whole file', he: 'הקובץ כולו' },
  header: { en: 'The header row', he: 'שורת הכותרת' },
  frame: { en: 'The table', he: 'הטבלה' },
};

export const ROLE_LABELS = {
  input: { en: 'Input', he: 'קלט' },
  plan: { en: 'Plan', he: 'תוכנית' },
  model: { en: 'Model', he: 'מודל' },
};

export const MODEL_STATE_LABELS = {
  fresh: { en: 'Current', he: 'עדכנית' },
  stale: { en: 'Sources have moved on', he: 'המקורות השתנו' },
  unknown: { en: 'Cannot be checked', he: 'לא ניתן לבדוק' },
};

// The basis labels the server sends with every report, kept here as a fallback
// only. The payload's own words are used whenever it carries them.
export const BASIS_LABELS = {
  period: { en: 'Period', he: 'תקופה' },
  scope: { en: 'Scope', he: 'היקף' },
  built_from: { en: 'Built from', he: 'נבנה מתוך' },
  updated: { en: 'Source updated', he: 'המקור עודכן' },
};

export const TEXT = {
  destination: { en: 'Sources', he: 'מקורות' },
  destinationBody: { en: 'Every input a run reads, what the engine is actually reading right now, and the reports built from it.', he: 'כל קלט שהרצה קוראת, מה שהמנוע קורא בפועל כרגע, והדוחות שנבנים ממנו.' },
  modelVersion: { en: 'Model version', he: 'גרסת מודל' },
  modelMeasuredOn: { en: 'Measured on', he: 'נמדדה על' },
  modelChanged: { en: 'Changed since', he: 'השתנו מאז' },
  modelUnavailable: { en: 'No model version is on disk, so the plan rests on the declared assumption instead.', he: 'אין גרסת מודל על הדיסק, ולכן התוכנית נשענת על ההנחה המוצהרת במקום.' },
  rows: { en: 'Rows', he: 'שורות' },
  columns: { en: 'Columns', he: 'עמודות' },
  size: { en: 'Size', he: 'גודל' },
  updated: { en: 'Updated', he: 'עודכן' },
  engineReads: { en: 'The engine reads', he: 'המנוע קורא' },
  engineReadsNone: { en: 'Nothing, no file of this kind is on disk', he: 'שום דבר, אין קובץ מסוג זה על הדיסק' },
  fromTheFile: { en: 'Read from the file', he: 'נקרא מהקובץ' },
  chooseFile: { en: 'Choose a file', he: 'בחרו קובץ' },
  checking: { en: 'Checking the file', he: 'בודק את הקובץ' },
  uploading: { en: 'Uploading', he: 'מעלה' },
  accepted: { en: 'The file passed every check', he: 'הקובץ עבר את כל הבדיקות' },
  acceptedNotRead: { en: 'The file passed every check, and nothing will read it', he: 'הקובץ עבר את כל הבדיקות, ושום דבר לא יקרא אותו' },
  acceptedNoRows: { en: 'The file passed every check, and it carries no rows', he: 'הקובץ עבר את כל הבדיקות, ואין בו אף שורה' },
  acceptedWarned: { en: 'The file was not refused, and a check on it came back with a warning', he: 'הקובץ לא נדחה, ובדיקה שנעשתה עליו החזירה אזהרה' },
  acceptedRead: { en: 'The file was not refused, and it carries a value that can be read two ways', he: 'הקובץ לא נדחה, ויש בו ערך שאפשר לקרוא בשתי צורות' },
  savesTo: { en: 'Will be stored as', he: 'יישמר בשם' },
  refused: { en: 'The file was refused and nothing was replaced', he: 'הקובץ נדחה ושום דבר לא הוחלף' },
  commit: { en: 'Upload this file', he: 'העלו את הקובץ הזה' },
  discard: { en: 'Choose a different file', he: 'בחרו קובץ אחר' },
  showRows: { en: 'Show the rows', he: 'הצגת השורות' },
  rowsTitle: { en: 'Rows in this file', he: 'שורות בקובץ הזה' },
  rowsShown: { en: 'Shown', he: 'מוצגות' },
  rowsOwned: { en: 'Your channel', he: 'הערוץ שלכם' },
  rowsTotal: { en: 'In the file', he: 'בקובץ' },
  rowsExcluded: { en: 'Rows on other channels are not shown.', he: 'שורות של ערוצים אחרים אינן מוצגות.' },
  columnsHidden: { en: 'Columns for other channels are not shown.', he: 'עמודות של ערוצים אחרים אינן מוצגות.' },
  close: { en: 'Close', he: 'סגירה' },
  readOnly: { en: 'A viewer account can read every state here and change none of it.', he: 'חשבון צפייה יכול לקרוא כאן כל מצב ולא לשנות אף אחד מהם.' },
  offline: { en: 'The Kairos API is unavailable, so the state of the inputs cannot be read.', he: 'ה־API של Kairos לא זמין, ולכן לא ניתן לקרוא את מצב הקלטים.' },
  loading: { en: 'Reading the state of every input', he: 'קורא את מצב כל הקלטים' },
  none: { en: 'No input is in this state.', he: 'אין קלט במצב הזה.' },
  fileRole: { en: 'Role', he: 'תפקיד' },
  fileState: { en: 'Read by the engine', he: 'נקרא על ידי המנוע' },
  filePath: { en: 'File', he: 'קובץ' },
  fileYes: { en: 'Yes', he: 'כן' },
  fileNo: { en: 'No', he: 'לא' },
  filesTitle: { en: 'Every file this product reads or writes', he: 'כל קובץ שהמוצר קורא או כותב' },
  filesBody: { en: 'Present on disk and read by the engine are two different facts, so each file carries both and the reason when nothing reads it.', he: 'קיים על הדיסק ונקרא על ידי המנוע הן שתי עובדות שונות, ולכן כל קובץ נושא את שתיהן ואת הסיבה כששום דבר לא קורא אותו.' },
  filesMissing: { en: 'Not on disk', he: 'לא על הדיסק' },
  downloadAll: { en: 'Download every ready report', he: 'הורדת כל דוח מוכן' },
  downloadsBody: { en: 'Each report is built from the current data when you download it, and the row count beside it is the exact number of rows the file will carry.', he: 'כל דוח נבנה מהנתונים הנוכחיים ברגע ההורדה, ומספר השורות שלצידו הוא בדיוק מספר השורות שהקובץ יישא.' },
  reportRows: { en: 'rows in this download', he: 'שורות בהורדה הזו' },
  reportEmpty: { en: 'This report has no rows yet, so there is nothing to download.', he: 'לדוח הזה אין עדיין שורות, ולכן אין מה להוריד.' },
  sourcePackage: { en: 'Files behind these reports', he: 'הקבצים שמאחורי הדוחות' },
  readOfPresent: { en: 'read by the engine, of', he: 'נקראים על ידי המנוע, מתוך' },
  present: { en: 'present on disk', he: 'קיימים על הדיסק' },
  findingColumn: { en: 'Column', he: 'עמודה' },
  findingMessage: { en: 'What is wrong', he: 'מה לא תקין' },
  findingRow: { en: 'Row', he: 'שורה' },
  findingRows: { en: 'Rows', he: 'שורות' },
  findingRowsMore: { en: 'more', he: 'נוספות' },
  findingRowsNote: { en: 'Counted from the first data row of the file you chose. The header is not row 1.', he: 'נספרות משורת הנתונים הראשונה בקובץ שבחרתם. הכותרת אינה שורה 1.' },
  lastChecked: { en: 'Last checked', he: 'נבדק לאחרונה' },
  noCheck: { en: 'No file of this kind has been checked yet.', he: 'עדיין לא נבדק קובץ מסוג זה.' },
  storedUnread: { en: 'Also on disk here, and the engine does not read it', he: 'נמצא כאן על הדיסק, והמנוע אינו קורא אותו' },
  storedUnreadMore: { en: 'more files stored here that the engine does not read', he: 'קבצים נוספים שנשמרו כאן והמנוע אינו קורא' },
  fields: { en: 'Facts on the card', he: 'עובדות על הכרטיס' },
  fieldsHint: { en: 'Choose which facts print on every card.', he: 'בחרו אילו עובדות יודפסו על כל כרטיס.' },
  whatIsChecked: { en: 'What the check runs, and what it cannot answer', he: 'מה הבדיקה מריצה, ומה אינה יכולה לענות' },
  checksRequired: { en: 'Required columns', he: 'עמודות נדרשות' },
  checksLoader: { en: 'Parsed with', he: 'נקרא באמצעות' },
  checksContract: { en: 'Checked against', he: 'נבדק מול' },
  checksCannot: { en: 'This check cannot answer', he: 'הבדיקה הזו אינה יכולה לענות' },
  path: { en: 'Path', he: 'נתיב' },
  cadence: { en: 'Arrives', he: 'מגיע' },
  consequenceField: { en: 'An upload here', he: 'העלאה כאן' },
  remedyField: { en: 'What to do', he: 'מה לעשות' },
  openReportRows: { en: 'Open the rows behind this number', he: 'פתחו את השורות שמאחורי המספר הזה' },
  reportRowsTitle: { en: 'Rows in this download', he: 'שורות בהורדה הזו' },
  reportRowsSource: { en: 'Read from', he: 'נקרא מתוך' },
  previous: { en: 'Previous', he: 'הקודם' },
  next: { en: 'Next', he: 'הבא' },
  position: { en: 'of', he: 'מתוך' },
  openRowsForFile: { en: 'Open the rows in this file', he: 'פתחו את השורות בקובץ הזה' },
};

export function text(key, locale) {
  const entry = TEXT[key];
  if (!entry) return '';
  return locale === 'he' ? entry.he : entry.en;
}

export function label(table, key, locale) {
  const entry = table[key];
  if (!entry) return String(key || '');
  return locale === 'he' ? entry.he : entry.en;
}

// A record the server sent as {code, en, he}: use its own words, never a local
// copy, so the sentence a person reads is the one the server computed.
export function serverText(record, locale) {
  if (!record || typeof record !== 'object') return '';
  return String((locale === 'he' ? record.he : record.en) || '');
}
