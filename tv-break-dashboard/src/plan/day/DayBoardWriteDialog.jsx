import React, { useRef } from 'react';
import { Button } from '../../studio/actions';
import { Code, Name } from '../../shell/bidi';
import { pageText } from '../../shell/format';
import { Dialog } from '../../studio/modal';

export default function DayBoardWriteDialog({ review, locale, busy, onCancel, onConfirm }) {
  const cancelRef = useRef(null);
  const gold = review?.kind === 'gold';
  const removingGold = gold && review?.live?.isGold;
  return (
    <Dialog
      open={Boolean(review)}
      onClose={onCancel}
      title={gold ? pageText(locale, 'Review the gold decision', 'בדיקת החלטת הזהב') : pageText(locale, 'Review the placement save', 'בדיקת שמירת המיקומים')}
      description={pageText(locale, 'Confirm the exact object, scope and consequence before the day is re-planned.', 'אשרו את האובייקט, ההיקף וההשפעה לפני תכנון היום מחדש.')}
      closeLabel={pageText(locale, 'Close write review', 'סגירת בדיקת הכתיבה')}
      initialFocusRef={cancelRef}
      dismissOnBackdrop={false}
      footer={(
        <>
          <Button ref={cancelRef} type="button" variant="outlined" onClick={onCancel}>{pageText(locale, 'Cancel', 'ביטול')}</Button>
          <Button type="button" variant="contained" disabled={busy} onClick={onConfirm}>{gold ? (removingGold ? pageText(locale, 'Remove gold and re-plan', 'הסרת הזהב ותכנון מחדש') : pageText(locale, 'Mark gold and re-plan', 'סימון כזהב ותכנון מחדש')) : pageText(locale, 'Save placements and re-plan', 'שמירת המיקומים ותכנון מחדש')}</Button>
        </>
      )}
    >
      <dl className="day-removal-ledger">
        <div><dt>{pageText(locale, 'Object', 'אובייקט')}</dt><dd>{gold ? <><Name>{review?.item?.programme}</Name> · <Code>{review?.item?.break_id}</Code></> : pageText(locale, `${review?.count || 0} edited break placements`, `${review?.count || 0} מיקומי ברייק שנערכו`)}</dd></div>
        <div><dt>{pageText(locale, 'Scope', 'היקף')}</dt><dd><Code>{review?.scope}</Code></dd></div>
        {!gold && <div><dt>{pageText(locale, 'Breaks', 'ברייקים')}</dt><dd><Code>{(review?.breakIds || []).join(', ')}</Code></dd></div>}
        <div><dt>{pageText(locale, 'Consequence', 'השפעה')}</dt><dd>{gold ? (removingGold ? pageText(locale, 'The gold decision is deleted and the engine re-plans this channel-day. Break counts, identifiers and placements may change.', 'החלטת הזהב נמחקת והמנוע מתכנן מחדש את יום־הערוץ. מספרי ברייקים, מזהים ומיקומים עשויים להשתנות.') : pageText(locale, 'A gold decision is written for this programme and the engine re-plans this channel-day. Break counts, identifiers and placements may change.', 'החלטת זהב נכתבת לתוכנית והמנוע מתכנן מחדש את יום־הערוץ. מספרי ברייקים, מזהים ומיקומים עשויים להשתנות.')) : pageText(locale, 'One placement restriction is written per edited break and the engine re-plans the whole channel-day. Other break identifiers and placements may change.', 'נכתבת מגבלת מיקום אחת לכל ברייק שנערך והמנוע מתכנן מחדש את כל יום־הערוץ. מזהים ומיקומים של ברייקים אחרים עשויים להשתנות.')}</dd></div>
      </dl>
    </Dialog>
  );
}
