import React, { useRef } from 'react';
import { Button } from '../../studio/actions';
import { Trash2 } from 'lucide-react';
import { pageText } from '../../shell/surface-helpers';
import { Dialog } from '../../studio/modal';
import { Code } from '../../shell/bidi';
import { kindLabel } from './override-console-lib';
import DayRunReviewDialog from './DayRunReviewDialog';

export default function OverrideDecisionDialogs({
  locale,
  pendingDeleteOverride,
  dayRun,
  onCancelDelete,
  onConfirmDelete,
  onCancelDayRun,
  onConfirmDayRun,
}) {
  const cancelDeleteRef = useRef(null);
  const absent = pageText(locale, 'Not recorded', 'לא תועד');
  return (
    <>
      <Dialog
        open={Boolean(pendingDeleteOverride)}
        onClose={onCancelDelete}
        title={pageText(locale, 'Confirm override removal', 'אישור הסרת עקיפה')}
        description={pageText(locale, 'Review the stored record and the consequence before deleting it.', 'בדקו את הרשומה השמורה ואת ההשפעה לפני המחיקה.')}
        closeLabel={pageText(locale, 'Close removal review', 'סגירת בדיקת ההסרה')}
        initialFocusRef={cancelDeleteRef}
        dismissOnBackdrop={false}
        footer={(
          <>
            <Button ref={cancelDeleteRef} type="button" variant="outlined" onClick={onCancelDelete}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
            <Button type="button" className="is-danger" variant="contained" onClick={onConfirmDelete}>
              <Trash2 size={16} aria-hidden="true" />
              {pageText(locale, 'Remove override', 'הסרת העקיפה')}
            </Button>
          </>
        )}
      >
        <dl className="oc-delete-ledger">
          <div><dt>{pageText(locale, 'Stored record', 'רשומה שמורה')}</dt><dd><bdi>{pendingDeleteOverride?.anchor_title || pendingDeleteOverride?.target_id || absent}</bdi></dd></div>
          <div><dt>{pageText(locale, 'Decision', 'החלטה')}</dt><dd>{pendingDeleteOverride ? kindLabel(pendingDeleteOverride.kind, locale) : absent}</dd></div>
          <div><dt>{pageText(locale, 'Scope', 'היקף')}</dt><dd><Code>{pendingDeleteOverride?.target_id || absent}</Code></dd></div>
          <div><dt>{pageText(locale, 'Consequence', 'השפעה')}</dt><dd>{pageText(locale, 'This override record is deleted immediately. The next plan run will no longer enforce it. The saved plan does not change until a run is confirmed.', 'רשומת העקיפה נמחקת מיד. ההרצה הבאה לא תאכוף אותה. התוכנית השמורה אינה משתנה עד לאישור הרצה.')}</dd></div>
        </dl>
      </Dialog>
      <DayRunReviewDialog
        review={dayRun.review}
        locale={locale}
        busy={dayRun.safety.status === 'checking' || dayRun.jobState === 'running'}
        onCancel={onCancelDayRun}
        onConfirm={onConfirmDayRun}
      />
    </>
  );
}
