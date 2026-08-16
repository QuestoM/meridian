import React, { useRef } from 'react';
import { Button } from '../../studio/actions';
import { formatNumber, pageText } from '../../shell/format';
import { Code } from '../../shell/bidi';
import { Dialog } from '../../studio/modal';

export default function PlanConsequenceDialog({
  review,
  locale,
  scopeText,
  dirty,
  versionName,
  runAllowed,
  onCancel,
  onConfirm,
}) {
  const cancelRef = useRef(null);
  return (
    <Dialog
      open={Boolean(review)}
      onClose={onCancel}
      title={review?.kind === 'run'
        ? pageText(locale, 'Review the weekly rewrite', 'בדיקת כתיבת השבוע מחדש')
        : review?.kind === 'restore'
          ? pageText(locale, 'Review the rollback', 'בדיקת החזרה לגרסה')
          : pageText(locale, 'Review the freeze', 'בדיקת הקפאת התוכנית')}
      description={review?.kind === 'run'
        ? pageText(locale, 'Nothing is written until you confirm this scope and consequence.', 'דבר לא נכתב עד לאישור ההיקף וההשפעה.')
        : pageText(locale, 'Confirm the named plan and what this action changes.', 'אשרו את התוכנית הנקובה ואת מה שהפעולה משנה.')}
      closeLabel={pageText(locale, 'Close review', 'סגירת הבדיקה')}
      initialFocusRef={cancelRef}
      dismissOnBackdrop={false}
      className="plan-consequence-dialog"
      footer={(
        <>
          <Button ref={cancelRef} type="button" variant="outlined" onClick={onCancel}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </Button>
          <Button type="button" variant="contained" disabled={review?.kind === 'run' && !runAllowed} onClick={onConfirm}>
            {review?.kind === 'run'
              ? pageText(locale, 'Rewrite the weekly plan', 'כתיבת התוכנית השבועית מחדש')
              : review?.kind === 'restore'
                ? pageText(locale, 'Roll back the live plan', 'החזרת התוכנית החיה')
                : pageText(locale, 'Freeze this named plan', 'הקפאת התוכנית בשם זה')}
          </Button>
        </>
      )}
    >
      <dl className="plan-consequence-ledger">
        <div>
          <dt>{pageText(locale, 'Scope', 'היקף')}</dt>
          <dd>{scopeText || pageText(locale, 'The saved weekly plan scope reported by the server', 'היקף התוכנית השבועית השמורה כפי שדווח מהשרת')}</dd>
        </div>
        {review?.kind === 'run' ? (
          <>
            <div>
              <dt>{pageText(locale, 'Input', 'קלט')}</dt>
              <dd>{dirty
                ? pageText(locale, 'The saved objective. Unsaved objective changes on screen are not included.', 'המטרה השמורה. שינויים שלא נשמרו במטרה שעל המסך לא ייכללו.')
                : pageText(locale, 'The saved objective shown on step 1', 'המטרה השמורה שמוצגת בשלב 1')}</dd>
            </div>
            <div>
              <dt>{pageText(locale, 'Placement source', 'מקור השיבוץ')}</dt>
              <dd>
                {review.inventoryMode === 'identity'
                  ? pageText(locale, 'Verified no-inventory mode; the optional inventory file is absent.', 'אומת מצב ללא שקלול מלאי; קובץ המלאי האופציונלי חסר.')
                  : pageText(locale, `${formatNumber(review.inventorySlots, locale)} usable placement slots`, `${formatNumber(review.inventorySlots, locale)} משבצות שיבוץ שמישות`)}
                {' · '}<Code>{review.inventoryPath}</Code>
              </dd>
            </div>
            <div>
              <dt>{pageText(locale, 'Consequence', 'השפעה')}</dt>
              <dd>{pageText(locale, 'The weekly plan on disk is rewritten. Every screen that reads the live plan moves to the new result. Frozen versions are kept.', 'התוכנית השבועית שעל הדיסק נכתבת מחדש. כל מסך שקורא את התוכנית החיה עובר לתוצאה החדשה. גרסאות מוקפאות נשמרות.')}</dd>
            </div>
          </>
        ) : review?.kind === 'restore' ? (
          <>
            <div>
              <dt>{pageText(locale, 'Version', 'גרסה')}</dt>
              <dd><bdi>{review.version?.name || review.version?.version_id}</bdi></dd>
            </div>
            <div>
              <dt>{pageText(locale, 'Consequence', 'השפעה')}</dt>
              <dd>{pageText(locale, 'The live plan is replaced byte for byte by this version. The plan currently on disk is frozen first, so the rollback remains reversible.', 'התוכנית החיה מוחלפת בית־אחר־בית בגרסה הזאת. התוכנית שנמצאת כעת על הדיסק מוקפאת קודם, כך שהחזרה נשארת הפיכה.')}</dd>
            </div>
          </>
        ) : (
          <>
            <div>
              <dt>{pageText(locale, 'Version name', 'שם הגרסה')}</dt>
              <dd><bdi>{versionName || pageText(locale, 'Unnamed', 'ללא שם')}</bdi></dd>
            </div>
            <div>
              <dt>{pageText(locale, 'Consequence', 'השפעה')}</dt>
              <dd>{pageText(locale, 'A byte-for-byte version of the live plan is created under this name. The live plan itself does not move.', 'נוצרת גרסה בית־אחר־בית של התוכנית החיה בשם הזה. התוכנית החיה עצמה אינה משתנה.')}</dd>
            </div>
          </>
        )}
      </dl>
    </Dialog>
  );
}
