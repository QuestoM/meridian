// Every authored string this board renders, in both languages, in one table.
//
// The same rule the terminal half of this piece follows: a pair that lives in a
// table cannot drift into one language, because a key added without its partner
// is visible in one look at one file, and a test walks this table and asserts
// both halves of every entry.
//
// Nothing here names the act. The verdict on a candidate is recorded at a
// terminal by the people who own the model, and this screen is a reading of
// what was measured, so no string here is a command, a path into that act or an
// instruction to perform one.

export const WORDS = {
  'board.title': { en: 'Candidate board', he: 'לוח המועמדים' },
  'board.sub': { en: 'Every artifact scored again on one identical set of breaks, so a difference between two rows is a difference between two models.', he: 'כל הקבצים נמדדו מחדש על אותה קבוצת ברייקים בדיוק, ולכן הפרש בין שתי שורות הוא הפרש בין שני מודלים.' },
  'board.read_only': { en: 'This screen records nothing. It is the reading the verdict rests on.', he: 'המסך הזה אינו רושם דבר. הוא הקריאה שההכרעה נשענת עליה.' },

  'state.title': { en: 'Is this comparison about the artifacts on the server now', he: 'האם ההשוואה הזו עוסקת בקבצים שעל השרת עכשיו' },
  'state.current': { en: 'Current', he: 'עדכנית' },
  'state.stale': { en: 'Stale', he: 'לא עדכנית' },
  'state.unknown': { en: 'Unknown', he: 'לא ידוע' },
  'state.current_reason': { en: 'Every artifact the server is serving carries the digest this comparison was measured on.', he: 'כל קובץ שהשרת מגיש נושא את טביעת האצבע שההשוואה הזו נמדדה עליה.' },
  'state.stale_reason': { en: 'The server is serving a different artifact from the one measured here, so the figures below are about a file that has since changed.', he: 'השרת מגיש קובץ אחר מזה שנמדד כאן, ולכן המספרים למטה עוסקים בקובץ שהשתנה מאז.' },
  'state.unknown_reason': { en: 'The candidate route did not answer, so whether these figures are about the artifacts on disk now cannot be checked from here.', he: 'המסלול של המועמדים לא השיב, ולכן לא ניתן לבדוק מכאן אם המספרים האלה עוסקים בקבצים שעל הדיסק עכשיו.' },
  'state.moved': { en: 'What moved', he: 'מה זז' },
  'state.checking': { en: 'Checking against the live route', he: 'נבדק מול המסלול החי' },
  'state.measured_at': { en: 'Measured', he: 'נמדד' },
  'state.published_at': { en: 'Published', he: 'פורסם' },

  'limit.title': { en: 'The limit of this evaluation', he: 'מגבלת המדידה הזו' },
  'limit.lifted': { en: 'What would lift it', he: 'מה ירים אותה' },

  'table.title': { en: 'Artifacts, closest to the measured effects first', he: 'קבצים, הקרוב ביותר לאפקטים הנמדדים תחילה' },
  'table.artifact': { en: 'Artifact', he: 'קובץ' },
  'table.rmse': { en: 'Error', he: 'שגיאה' },
  'table.delta': { en: 'Against shipped', he: 'מול המשודר' },
  'table.statistic': { en: 'Statistic', he: 'סטטיסטי' },
  'table.verdict': { en: 'Measured verdict', he: 'הכרעת המדידה' },
  'table.cells': { en: 'Coefficients', he: 'מקדמים' },
  'table.money': { en: 'Money on the operator channel', he: 'כסף על ערוץ המפעיל' },
  'table.recorded': { en: 'Verdict on record', he: 'הכרעה רשומה' },
  'table.shipped_row': { en: 'Shipped, live', he: 'המשודר, החי' },
  'table.version': { en: 'Model version', he: 'גרסת המודל' },
  'table.pick': { en: 'Open the evidence for this artifact', he: 'פתיחת הראיות לקובץ הזה' },

  'verdict.identical': { en: 'Identical', he: 'זהה' },
  'verdict.better': { en: 'Better', he: 'טוב יותר' },
  'verdict.worse': { en: 'Worse', he: 'גרוע יותר' },
  'verdict.not_distinguishable': { en: 'No difference', he: 'ללא הבדל' },
  'verdict.unknown': { en: 'Not scored', he: 'לא נמדד' },

  'decision.shipped': { en: 'Shipped', he: 'הושקה' },
  'decision.not_shipped': { en: 'Not shipped', he: 'לא הושקה' },
  'decision.none': { en: 'None', he: 'אין' },
  'decision.by': { en: 'by', he: 'על ידי' },
  'decision.before_comparison': { en: 'Taken before this comparison existed, so it rests on each artifact reading its own held-out figures, which come from different splits and are not comparable.', he: 'התקבלה לפני שההשוואה הזו התקיימה, ולכן היא נשענת על המספרים שכל קובץ מדווח על עצמו, שמגיעים מפיצולים שונים ואינם ברי השוואה.' },
  'decision.on_comparison': { en: 'Taken on this common-basis comparison.', he: 'התקבלה על בסיס ההשוואה המשותפת הזו.' },

  'money.measured': { en: 'Measured and current', he: 'נמדד ועדכני' },
  'money.stale': { en: 'Stale', he: 'לא עדכני' },
  'money.not_measured': { en: 'Not measured', he: 'לא נמדד' },
  'money.last_known': { en: 'Last measured', he: 'נמדד לאחרונה' },
  'money.rows': { en: 'plan rows', he: 'שורות תוכנית' },
  'money.whole_plan': { en: 'Whole plan, every channel the optimizer schedules', he: 'התוכנית כולה, כל הערוצים שהמנוע מתזמן' },
  'money.basis': { en: 'Basis', he: 'בסיס' },

  'detail.title': { en: 'The evidence for', he: 'הראיות עבור' },
  'detail.rule': { en: 'The rule that decided it', he: 'הכלל שהכריע' },
  'detail.breaks_moved': { en: 'Breaks it moves closer, and further', he: 'ברייקים שהוא מקרב, ושהוא מרחיק' },
  'detail.coefficients': { en: 'What its coefficients change', he: 'מה המקדמים שלו משנים' },
  'detail.largest': { en: 'Largest move', he: 'התזוזה הגדולה ביותר' },
  'detail.cell': { en: 'Cell', he: 'תא' },
  'detail.shipped': { en: 'Shipped', he: 'משודר' },
  'detail.candidate': { en: 'Candidate', he: 'מועמד' },
  'detail.delta': { en: 'Move', he: 'תזוזה' },
  'detail.breaks': { en: 'Breaks', he: 'ברייקים' },
  'detail.bought': { en: 'Squared error moved', he: 'שגיאה ריבועית שזזה' },
  'detail.share': { en: 'Share', he: 'חלק' },
  'detail.top_of': { en: 'The ranked head of the cells. The rest are not shown here.', he: 'ראש הדירוג של התאים. השאר אינם מוצגים כאן.' },
  'detail.identity': { en: 'The file this row was measured on', he: 'הקובץ שהשורה הזו נמדדה עליו' },
  'detail.fitted_on': { en: 'Fitted on', he: 'הותאם על' },
  'detail.duplicate': { en: 'Predicts the same value as', he: 'חוזה את אותו ערך כמו' },
  'detail.pick_one': { en: 'Pick an artifact above to read what it was decided on.', he: 'יש לבחור קובץ למעלה כדי לקרוא על מה הוא הוכרע.' },

  'baselines.title': { en: 'Baselines, out of sample, no artifact involved', he: 'בסיסי השוואה, מחוץ למדגם, בלי שום קובץ' },
  'finding.title': { en: 'Standing finding, and no candidate here answers it', he: 'ממצא עומד, ואף מועמד כאן אינו עונה עליו' },
  'finding.owner': { en: 'Whose decision this is', he: 'של מי ההחלטה הזו' },
  'finding.none_address': { en: 'None of the candidates changes the set of cells at all, so every one of them is a choice made inside this structure.', he: 'אף אחד מהמועמדים אינו משנה את קבוצת התאים כלל, ולכן כל אחד מהם הוא בחירה בתוך המבנה הזה.' },

  'evaluation.breaks': { en: 'breaks', he: 'ברייקים' },
  'evaluation.cells': { en: 'cells', he: 'תאים' },
  'evaluation.folds': { en: 'temporal folds', he: 'קיפולים בזמן' },
  'evaluation.metric': { en: 'Metric', he: 'מדד' },
  'evaluation.spread': { en: 'Spread of the target itself', he: 'פיזור היעד עצמו' },
  'evaluation.bar': { en: 'Bar', he: 'רף' },
  'evaluation.dispersion': { en: 'Fold dispersion', he: 'פיזור הקיפולים' },
  'evaluation.duplicates': { en: 'Predict the same value for every break', he: 'חוזים את אותו ערך בכל ברייק' },
};

export function t(key, locale) {
  const entry = WORDS[key];
  if (!entry) return key;
  return locale === 'en' ? entry.en : entry.he;
}

// The half of a payload pair this locale reads. The payloads carry both halves
// under `<name>_en` and `<name>_he`, so a screen never translates a measured
// sentence and never invents one that the measurement did not produce.
export function pick(record, name, locale) {
  if (!record) return '';
  return String(record[`${name}_${locale === 'en' ? 'en' : 'he'}`] || '');
}
