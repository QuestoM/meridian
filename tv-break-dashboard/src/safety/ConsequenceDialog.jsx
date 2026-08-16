import React from 'react';
import { Button } from '../studio/actions';
import { Dialog } from '../studio/modal';
import { pageText } from '../shell/format';
import './consequence-dialog.css';

// A consequence review is deliberately narrower than a generic confirmation.
// It names the stored object, the exact write boundary and the downstream
// effect before the dangerous action becomes available. The canonical native
// Dialog owns focus containment, Escape cancellation and focus return.
export function focusAfterDialogClose(targetRef) {
  if (typeof window === 'undefined') return;
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      const target = targetRef?.current;
      if (target instanceof HTMLElement && target.isConnected) {
        target.focus({ preventScroll: true });
      }
    });
  });
}

export default function ConsequenceDialog({
  open,
  locale,
  title,
  description,
  object,
  scope,
  consequence,
  recovery,
  confirmLabel,
  workingLabel,
  busy = false,
  onCancel,
  onConfirm,
}) {
  const cancelRef = React.useRef(null);
  const cancel = () => {
    if (!busy) onCancel?.();
  };

  return (
    <Dialog
      className="consequence-dialog"
      size="narrow"
      open={Boolean(open)}
      title={title}
      description={description}
      closeLabel={pageText(locale, 'Cancel and close', 'ביטול וסגירה')}
      initialFocusRef={cancelRef}
      dismissOnBackdrop={!busy}
      onClose={cancel}
      footer={(
        <>
          <Button ref={cancelRef} variant="outlined" disabled={busy} onClick={cancel}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </Button>
          <Button
            variant="contained"
            color="error"
            loading={busy}
            loadingIndicator={workingLabel || pageText(locale, 'Applying the change', 'מבצע את השינוי')}
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </>
      )}
    >
      <dl className="consequence-review">
        <div className="consequence-review__row">
          <dt>{pageText(locale, 'Object', 'אובייקט')}</dt>
          <dd>{object}</dd>
        </div>
        <div className="consequence-review__row">
          <dt>{pageText(locale, 'Scope', 'היקף')}</dt>
          <dd>{scope}</dd>
        </div>
        <div className="consequence-review__row">
          <dt>{pageText(locale, 'Consequence', 'תוצאה')}</dt>
          <dd>{consequence}</dd>
        </div>
        {recovery ? (
          <div className="consequence-review__row">
            <dt>{pageText(locale, 'Recovery', 'שחזור')}</dt>
            <dd>{recovery}</dd>
          </div>
        ) : null}
      </dl>
    </Dialog>
  );
}
