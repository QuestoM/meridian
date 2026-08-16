import React, { useRef } from 'react';
import { Button } from '../../studio/actions';
import { Code } from '../../shell/bidi';
import { formatNumber, pageText } from '../../shell/format';
import { Dialog } from '../../studio/modal';
import './day-run-safety.css';

function scopeText(review) {
  return (review?.scope || [])
    .map((entry) => [entry.channel, entry.day].filter(Boolean).join(' / '))
    .filter(Boolean)
    .join(', ');
}

export default function DayRunReviewDialog({ review, locale, busy, onCancel, onConfirm }) {
  const cancelRef = useRef(null);
  const inventory = review?.inputs?.inventory;
  return (
    <Dialog
      open={Boolean(review)}
      onClose={onCancel}
      title={pageText(locale, 'Review the channel-day rewrite', 'בדיקת כתיבת יום־הערוץ מחדש')}
      description={pageText(locale, 'Nothing is written until the saved inputs, scope and consequence are confirmed.', 'דבר לא נכתב עד לאישור הקלטים השמורים, ההיקף וההשפעה.')}
      closeLabel={pageText(locale, 'Close day run review', 'סגירת בדיקת הרצת היום')}
      initialFocusRef={cancelRef}
      dismissOnBackdrop={false}
      footer={(
        <>
          <Button ref={cancelRef} type="button" variant="outlined" onClick={onCancel}>{pageText(locale, 'Cancel', 'ביטול')}</Button>
          <Button type="button" variant="contained" disabled={busy} onClick={onConfirm}>{busy ? pageText(locale, 'Checking inputs', 'בודק קלטים') : pageText(locale, 'Rewrite this channel-day', 'כתיבת יום־הערוץ מחדש')}</Button>
        </>
      )}
    >
      <dl className="day-run-review-ledger">
        <div><dt>{pageText(locale, 'Saved settings', 'הגדרות שמורות')}</dt><dd>{pageText(locale, 'Verified before this review opened', 'אומתו לפני פתיחת הבדיקה')}</dd></div>
        <div><dt>{pageText(locale, 'Scope', 'היקף')}</dt><dd><Code>{scopeText(review)}</Code></dd></div>
        <div>
          <dt>{pageText(locale, 'Placement source', 'מקור השיבוץ')}</dt>
          <dd>{inventory?.mode === 'identity'
            ? pageText(locale, 'Verified no-inventory mode; the optional inventory file is absent.', 'אומת מצב ללא שקלול מלאי; קובץ המלאי האופציונלי חסר.')
            : pageText(locale, `${formatNumber(inventory?.slots, locale)} usable placement slots`, `${formatNumber(inventory?.slots, locale)} משבצות שיבוץ שמישות`)}{' '}<Code>{inventory?.path}</Code></dd>
        </div>
        <div><dt>{pageText(locale, 'Consequence', 'השפעה')}</dt><dd>{pageText(locale, 'If the run succeeds, this channel-day in the saved plan is replaced. Break counts, identifiers and placements may change.', 'אם ההרצה תצליח, יום־הערוץ בתוכנית השמורה יוחלף. מספרי ברייקים, מזהים ומיקומים עשויים להשתנות.')}</dd></div>
      </dl>
    </Dialog>
  );
}
