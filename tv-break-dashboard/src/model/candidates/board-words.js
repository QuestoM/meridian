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
  'board.title': { en: 'Candidate model comparison', he: 'השוואת מודלים מועמדים' },
  'board.sub': { en: 'Every artifact scored again on one identical set of breaks, so a difference between two rows is a difference between two models.', he: 'כל הקבצים נמדדו מחדש על אותה קבוצת ברייקים בדיוק, ולכן הפרש בין שתי שורות הוא הפרש בין שני מודלים.' },
  'board.read_only': { en: 'Read-only. Release decisions are recorded separately.', he: 'לקריאה בלבד. החלטות הפצה נרשמות בנפרד.' },

  'state.title': { en: 'Artifact currency check', he: 'בדיקת עדכניות הקבצים' },
  'state.current': { en: 'Current', he: 'עדכנית' },
  'state.stale': { en: 'Stale', he: 'לא עדכנית' },
  'state.unknown': { en: 'Unknown', he: 'לא ידוע' },
  'state.current_reason': { en: 'Every artifact the server is serving carries the digest this comparison was measured on.', he: 'כל קובץ שהשרת מגיש נושא את טביעת האצבע שההשוואה הזו נמדדה עליה.' },
  'state.stale_reason': { en: 'The server is serving a different artifact from the one measured here, so the figures below are about a file that has since changed.', he: 'השרת מגיש קובץ אחר מזה שנמדד כאן, ולכן המספרים למטה עוסקים בקובץ שהשתנה מאז.' },
  'state.unknown_reason': { en: 'The candidate route did not answer, so whether these figures are about the artifacts on disk now cannot be checked from here.', he: 'המסלול של המועמדים לא השיב, ולכן לא ניתן לבדוק מכאן אם המספרים האלה עוסקים בקבצים שעל הדיסק עכשיו.' },
  'state.moved': { en: 'Changed artifacts', he: 'קבצים שהשתנו' },
  'state.checking': { en: 'Checking against the live route', he: 'נבדק מול המסלול החי' },
  'state.measured_at': { en: 'Measured', he: 'נמדד' },
  'state.published_at': { en: 'Published', he: 'פורסם' },

  'limit.title': { en: 'Evaluation limitation', he: 'מגבלת ההערכה' },
  'limit.lifted': { en: 'Required evidence', he: 'ראיות נדרשות' },
  'limit.rows': { en: 'The rows this is true of', he: 'השורות שזה נכון לגביהן' },

  'basis.mark': { en: 'Fitted on fewer breaks than it is scored on', he: 'אומן על פחות ברייקים מאלה שהוא נמדד עליהם' },
  'basis.unknown_mark': { en: 'Records no fit basis', he: 'אינו רושם בסיס אימון' },
  'basis.title': { en: 'What it was fitted on', he: 'על מה הוא אומן' },
  'basis.of': { en: 'of', he: 'מתוך' },
  'basis.never_fitted': { en: 'of them were never in its own fit', he: 'מהם מעולם לא היו באימון שלו' },

  'self.title': { en: 'Producer recommendation', he: 'המלצת יוצר הקובץ' },
  'self.basis': { en: "A self-test is the artifact's own split under its own fit, so it is readable about that artifact alone and is never comparable with another row here.", he: 'בדיקה עצמית היא הפיצול של הקובץ עצמו תחת האימון של עצמו, ולכן היא ניתנת לקריאה על אותו קובץ בלבד ולעולם אינה בת השוואה לשורה אחרת כאן.' },
  'self.advised_against': { en: 'Advised against', he: 'הומלץ שלא' },
  'self.recommended': { en: 'Recommended', he: 'הומלץ' },
  'self.recorded_without_a_verdict': { en: 'No recommendation', he: 'ללא המלצה' },
  'self.words': { en: 'Recorded rationale', he: 'נימוק מתועד' },
  'self.n_test': { en: 'Taken on', he: 'נלקחה על' },
  'self.breaks_own': { en: 'breaks in its evaluation set', he: 'ברייקים בקבוצת ההערכה שלו' },

  'table.title': { en: 'Artifacts ranked by measured-effect error', he: 'קבצים בדירוג לפי שגיאת האפקט הנמדד' },
  'table.artifact': { en: 'Artifact', he: 'קובץ' },
  'table.rmse': { en: 'Error', he: 'שגיאה' },
  'table.delta': { en: 'Against released', he: 'מול המודל המופץ' },
  'table.statistic': { en: 'Statistic', he: 'סטטיסטי' },
  'table.verdict': { en: 'Measured verdict', he: 'הכרעת המדידה' },
  'table.cells': { en: 'Coefficients', he: 'מקדמים' },
  'table.money': { en: 'Projected revenue impact', he: 'השפעה חזויה על ההכנסה' },
  'table.recorded': { en: 'Verdict on record', he: 'הכרעה רשומה' },
  'table.shipped_row': { en: 'Released and active', he: 'מופץ ופעיל' },
  'table.version': { en: 'Model version', he: 'גרסת המודל' },
  'table.pick': { en: 'Open the evidence for this artifact', he: 'פתיחת הראיות לקובץ הזה' },

  'verdict.identical': { en: 'Identical', he: 'זהה' },
  'verdict.better': { en: 'Better', he: 'טוב יותר' },
  'verdict.worse': { en: 'Worse', he: 'גרוע יותר' },
  'verdict.not_distinguishable': { en: 'No difference', he: 'ללא הבדל' },
  'verdict.unknown': { en: 'Not scored', he: 'לא נמדד' },

  'decision.shipped': { en: 'Approved for release', he: 'אושרה להפצה' },
  'decision.not_shipped': { en: 'Rejected for release', he: 'נדחתה להפצה' },
  'decision.none': { en: 'None', he: 'אין' },
  'decision.by': { en: 'by', he: 'על ידי' },
  'decision.before_comparison': { en: 'Taken before this comparison existed, so it rests on each artifact reading its own held-out figures, which come from different splits and are not comparable.', he: 'התקבלה לפני שההשוואה הזו התקיימה, ולכן היא נשענת על המספרים שכל קובץ מדווח על עצמו, שמגיעים מפיצולים שונים ואינם ברי השוואה.' },
  'decision.on_comparison': { en: 'Taken on this common-basis comparison.', he: 'התקבלה על בסיס ההשוואה המשותפת הזו.' },
  'decision.mark_legend': { en: 'against a verdict means', he: 'ליד הכרעה פירושו' },

  // Whether any of this has actually replaced the live model. The column beside
  // it says "Shipped" of a recorded verdict, which is a decision and not an act,
  // and a reader who stops at that word can leave believing the candidate is
  // live. The terminal states this and the screen did not.
  'adopted.title': { en: 'Active artifact', he: 'קובץ פעיל' },
  'adopted.none': { en: 'No candidate adoption is recorded. The active artifact remains the one produced by the training process; a recorded verdict does not replace it.', he: 'לא נרשמה הטמעה של מודל מועמד. הקובץ הפעיל נשאר זה שהופק בתהליך האימון; הכרעה מתועדת אינה מחליפה אותו.' },

  'table.count': { en: 'candidate artifacts compared with the released artifact', he: 'קבצים מועמדים בהשוואה מול הקובץ המופץ' },
  // How the table is worked, in ink. Four of the eight columns sort and the only
  // thing saying which was a bare circle glyph with no legend, and the up and
  // down keys moved the selection with nothing on the screen saying so.
  'table.how_to_read': { en: 'A column heading that is a button sorts the table, and its mark shows the direction. Up and down move the selection once the table has focus.', he: 'כותרת עמודה שהיא לחצן ממיינת את הטבלה, והסימן שלה מראה את הכיוון. מקשי מעלה ומטה מזיזים את הבחירה לאחר שהטבלה בפוקוס.' },
  // A count of verdicts, because one artifact here was refused twice and the
  // column printed one word for it. The terminal states the count and the screen
  // did not, so a steward reading only the screen could not see the second one.
  'decision.count': { en: 'verdicts on record', he: 'הכרעות רשומות' },

  'money.measured': { en: 'Measured and current', he: 'נמדד ועדכני' },
  'money.stale': { en: 'Stale', he: 'לא עדכני' },
  'money.not_measured': { en: 'Not measured', he: 'לא נמדד' },
  'money.last_known': { en: 'Last measured', he: 'נמדד לאחרונה' },
  'money.rows': { en: 'plan rows', he: 'שורות תוכנית' },
  'money.whole_plan': { en: 'Whole plan, every channel the optimizer schedules', he: 'התוכנית כולה, כל הערוצים שהמנוע מתזמן' },
  'money.basis': { en: 'Basis', he: 'בסיס' },

  'detail.title': { en: 'The evidence for', he: 'הראיות עבור' },
  'detail.rule': { en: 'The rule that decided it', he: 'הכלל שהכריע' },
  'detail.breaks_moved': { en: 'Break-level error changes', he: 'שינויי שגיאה ברמת הברייק' },
  'detail.coefficients': { en: 'Coefficient changes', he: 'שינויים במקדמים' },
  'detail.largest': { en: 'Largest delta', he: 'השינוי הגדול ביותר' },
  'detail.cell': { en: 'Cell', he: 'תא' },
  'detail.shipped': { en: 'Released', he: 'מופץ' },
  'detail.candidate': { en: 'Candidate', he: 'מועמד' },
  'detail.delta': { en: 'Delta', he: 'שינוי' },
  'detail.breaks': { en: 'Breaks', he: 'ברייקים' },
  'detail.bought': { en: 'Squared-error change', he: 'שינוי בשגיאה הריבועית' },
  'detail.share': { en: 'Share', he: 'חלק' },
  'detail.top_of': { en: 'The ranked head of the cells. The rest are not shown here.', he: 'ראש הדירוג של התאים. השאר אינם מוצגים כאן.' },
  'detail.identity': { en: 'The file this row was measured on', he: 'הקובץ שהשורה הזו נמדדה עליו' },
  'detail.bytes': { en: 'bytes', he: 'בייטים' },
  // When the artifact itself was produced, which the payload has carried on
  // every row since the board was built and no surface rendered. It is the fact
  // that dates the shelf: measured on this tree, all five candidates were
  // produced on 05/07/2026 and the live artifact on 29/07/2026, so every
  // candidate here predates the model it is being compared against.
  'detail.built': { en: 'Produced', he: 'הופק' },
  'detail.fitted_on': { en: 'Fitted on', he: 'הותאם על' },
  'detail.duplicate': { en: 'Predicts the same value as', he: 'חוזה את אותו ערך כמו' },
  'detail.pick_one': { en: 'Select an artifact to review its evidence.', he: 'יש לבחור קובץ כדי לעיין בראיות שלו.' },

  // The gate block. JS-19 reads the gates before it reads the money, and until
  // this round they were reachable only from the terminal's last command.
  //
  // Five of these are the model console's own strings, verbatim, from
  // console-words.js lines 115, 116, 142, 144 and 145. The console names these
  // exact things on its own candidate screen, this board names them beside it,
  // and two surfaces naming one thing two ways is a divergence a steward walks
  // into inside one session. The console's file is frozen, so this row adopts
  // the console's words rather than inventing a second set.
  'gates.title': { en: 'Gate differences from the released model', he: 'הבדלי שערים מהמודל המופץ' },
  'gates.none': { en: 'No gate decides differently', he: 'אף שער אינו מכריע אחרת' },
  'gates.held_out': { en: 'The held-out figures behind those gates', he: 'המספרים מחוץ למדגם שמאחורי אותם שערים' },
  'gates.candidate_absent': { en: 'The candidate records nothing for this gate, so the two cannot be compared on it.', he: 'המועמד אינו רושם דבר עבור השער הזה, ולכן לא ניתן להשוות ביניהם עליו.' },
  'gates.shipped_absent': { en: 'The released artifact records nothing for this gate, so the two cannot be compared on it.', he: 'הקובץ המופץ אינו רושם דבר עבור השער הזה, ולכן לא ניתן להשוות ביניהם עליו.' },
  'gates.key': { en: 'Gate', he: 'שער' },
  'gates.block': { en: 'Held-out block', he: 'בלוק מחוץ למדגם' },
  'gates.absent_short': { en: 'Not carried', he: 'אינו נישא' },
  'gates.comparable': { en: 'Same amount on both sides', he: 'אותה כמות בשני הצדדים' },
  'gates.not_comparable': { en: 'Not comparable', he: 'אינו בר השוואה' },

  // What each artifact was built for and what data it was built from. The
  // purpose is a stored value in the producer's own words and these are only its
  // captions: the sentence itself rides on the payload and is never authored
  // here, and an artifact that records none says so rather than showing nothing.
  'origin.title': { en: 'Built for', he: 'נבנה בשביל' },
  'origin.none': { en: 'No purpose recorded', he: 'לא נרשם ייעוד' },
  'origin.sources': { en: 'What it was fitted from', he: 'ממה הוא אומן' },
  'origin.file': { en: 'Source file', he: 'קובץ מקור' },
  'origin.digest': { en: 'Digest it recorded', he: 'טביעת האצבע שרשם' },
  'origin.on_disk': { en: 'On disk now', he: 'על הדיסק עכשיו' },
  'origin.matches': { en: 'Same bytes', he: 'אותם בייטים' },
  'origin.differs': { en: 'Different bytes', he: 'בייטים אחרים' },
  'origin.missing': { en: 'Not on disk', he: 'אינו על הדיסק' },
  'origin.recipe': { en: 'Rebuilding it', he: 'בנייה מחדש שלו' },

  // What a bar is a share of, said wherever one is drawn. A bar whose
  // denominator is not on the screen is the visual form of a figure nobody
  // measured, so each of the two legends names the figure it divides by.
  'meter.movement': { en: 'The bar is the coefficient delta divided by the same row\'s fold dispersion. A full bar means the delta equals that dispersion.', he: 'הפס הוא שינוי המקדם חלקי פיזור הקיפולים של אותה שורה. פס מלא פירושו שהשינוי שווה לפיזור.' },
  'meter.spread': { en: 'The bar is the released model\'s error divided by the spread of the effect it predicts.', he: 'הפס הוא שגיאת המודל המופץ חלקי פיזור האפקט שהוא חוזה.' },
  'evaluation.error_share': { en: 'Error as a share of that spread', he: 'השגיאה כחלק מאותו פיזור' },

  // Every verdict ever recorded on an artifact, rather than the newest one.
  // The column on the shelf prints one word and a count, and on this tree one
  // artifact carries two verdicts that hold the same word for two different
  // stated reasons, which a count cannot tell apart from a repeat.
  //
  // The steward's own sentence is not here and is not on the payload either. It
  // is unbounded text written at a terminal, the console renders it from the
  // store, and a second copy in a bundled file is a second source that can
  // disagree with the first.
  'history.title': { en: 'Verdicts on record, newest first', he: 'הכרעות רשומות, החדשה תחילה' },
  'history.when': { en: 'Recorded', he: 'נרשמה' },
  'history.who': { en: 'By', he: 'על ידי' },
  'history.what': { en: 'Verdict', he: 'הכרעה' },
  'history.standing': { en: 'In force', he: 'בתוקף' },
  'history.superseded': { en: 'Superseded by a later verdict', he: 'הוחלפה בהכרעה מאוחרת יותר' },
  'history.version': { en: 'Against model version', he: 'מול גרסת מודל' },
  // Not a blank and not a zero. A version name is a stored string and its
  // absence is a fact about the record, so the surface says so in words.
  'history.version_none': { en: 'No version name recorded', he: 'לא נרשם שם גרסה' },
  'history.basis': { en: 'What it rests on', he: 'על מה היא נשענת' },
  'history.on_comparison': { en: 'This comparison', he: 'ההשוואה הזו' },
  'history.before_comparison': { en: 'Before it existed', he: 'לפני שהתקיימה' },
  'history.other_version': { en: 'Taken against a model version that is not the one in force', he: 'התקבלה מול גרסת מודל שאינה זו שבתוקף' },
  'history.note': { en: 'Carries a release note for the operator side', he: 'נושאת הערת גרסה לצד המפעיל' },
  'history.hidden': { en: 'verdicts a column showing only the newest does not show', he: 'הכרעות שעמודה המציגה רק את החדשה אינה מציגה' },
  'history.none': { en: 'No verdict has been recorded on this artifact.', he: 'לא נרשמה שום הכרעה על הקובץ הזה.' },

  // What is on record about the live artifact itself. A decision record may be
  // about the shipped model rather than about a candidate, and every read this
  // board made of the log filtered those out, so the shelf showed five verdicts
  // and said nothing about a standing verdict on the artifact all five are
  // measured against.
  'live.title': { en: 'Decision records for the released model', he: 'רישומי החלטה עבור המודל המופץ' },
  'live.log': { en: 'records in the decision log', he: 'רשומות ביומן ההכרעות' },
  'live.on_the_shelf': { en: 'on the candidate artifacts below', he: 'על הקבצים המועמדים שלמטה' },
  'live.on_the_live_model': { en: 'on the released model', he: 'על המודל המופץ' },
  'live.off_the_shelf': { en: 'on artifacts outside the current candidate list', he: 'על קבצים שאינם ברשימת המועמדים הנוכחית' },

  'baselines.title': { en: 'Baselines, out of sample, no artifact involved', he: 'בסיסי השוואה, מחוץ למדגם, בלי שום קובץ' },
  'finding.title': { en: 'Unresolved finding', he: 'ממצא לא פתור' },
  'finding.owner': { en: 'Decision owner', he: 'בעל ההחלטה' },
  'finding.none_address': { en: 'None of the candidates changes the set of cells at all, so every one of them is a choice made inside this structure.', he: 'אף אחד מהמועמדים אינו משנה את קבוצת התאים כלל, ולכן כל אחד מהם הוא בחירה בתוך המבנה הזה.' },

  'evaluation.breaks': { en: 'breaks', he: 'ברייקים' },
  'evaluation.cells': { en: 'cells', he: 'תאים' },
  // The window is two calendar days on the payload now, and it renders through
  // shell/dates formatSpan like every other span in this product. It arrived
  // here pre-joined, "2024-11-01 to 2024-11-30", an English preposition inside
  // a machine date, printed into a Hebrew line.
  'evaluation.window': { en: 'Measured over', he: 'נמדד על פני' },
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
