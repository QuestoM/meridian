import React, { useRef } from 'react';
import { Button } from '../../studio/actions';
import { Code, Figure, Name } from '../../shell/bidi';
import { pageText } from '../../shell/format';
import { Dialog } from '../../studio/modal';
import { humanOffset, secondsToClock } from './schedule-editor-format';

export default function ScheduleEditorPinReviewDialog({ review, locale, busy, onCancel, onConfirm }) {
  const cancelRef = useRef(null);
  return (
    <Dialog
      open={Boolean(review)}
      onClose={onCancel}
      title={pageText(locale, 'Review the saved placement', 'בדיקת שמירת הנעיצה')}
      description={pageText(locale, 'Confirm the exact break, airing, time and plan consequence before writing.', 'אשרו את הברייק, השידור, הזמן וההשפעה על התוכנית לפני הכתיבה.')}
      closeLabel={pageText(locale, 'Close placement review', 'סגירת בדיקת הנעיצה')}
      initialFocusRef={cancelRef}
      dismissOnBackdrop={false}
      footer={(
        <>
          <Button ref={cancelRef} type="button" variant="outlined" onClick={onCancel}>{pageText(locale, 'Cancel', 'ביטול')}</Button>
          <Button type="button" variant="contained" disabled={busy} onClick={onConfirm}>{pageText(locale, 'Save this placement', 'שמירת הנעיצה הזאת')}</Button>
        </>
      )}
    >
      <dl className="day-removal-ledger">
        <div><dt>{pageText(locale, 'Break', 'ברייק')}</dt><dd><Name>{review?.item?.program_title}</Name> · <Code>{review?.target?.item?.break_id}</Code></dd></div>
        <div><dt>{pageText(locale, 'Scope', 'היקף')}</dt><dd>{review?.scope}</dd></div>
        <div><dt>{pageText(locale, 'Time', 'זמן')}</dt><dd><Figure>{secondsToClock(review?.startSec || 0)}</Figure> · {humanOffset(review?.offsetSeconds || 0, locale)} · <Figure>{Math.round(review?.durationSec || 0)}s</Figure></dd></div>
        <div><dt>{pageText(locale, 'Consequence', 'השפעה')}</dt><dd>{pageText(locale, 'A placement restriction is written for this airing and the engine re-plans its channel-day. Other break identifiers and placements may change.', 'נכתבת מגבלת מיקום לשידור הזה והמנוע מתכנן מחדש את יום־הערוץ שלו. מזהים ומיקומים של ברייקים אחרים עשויים להשתנות.')}</dd></div>
      </dl>
    </Dialog>
  );
}
