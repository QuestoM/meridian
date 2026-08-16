import React from 'react';
import { IconButton } from '../studio/actions';
import { X } from 'lucide-react';

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

function cx(...values) {
  return values.flat().filter(Boolean).join(' ');
}

export function focusFirstWithin(container) {
  if (!container) return false;
  const target = Array.from(container.querySelectorAll(FOCUSABLE_SELECTOR)).find((element) => {
    const style = window.getComputedStyle(element);
    return style.visibility !== 'hidden' && style.display !== 'none' && !element.closest('[inert]');
  });
  target?.focus({ preventScroll: true });
  return Boolean(target);
}

export function useFocusReturn(active) {
  const previousFocus = React.useRef(null);
  const wasActive = React.useRef(false);
  const restore = React.useCallback((target) => {
    if (!(target instanceof HTMLElement) || !target.isConnected) return;
    // Native <dialog>.close() performs its own focus bookkeeping after React's
    // effect. Restore on the second painted frame so that bookkeeping has
    // settled; do not cancel this work when a conditionally rendered dialog
    // unmounts in the same commit. Never steal focus from a newer modal.
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        if (!target.isConnected) return;
        const openDialog = document.querySelector('dialog[open]');
        if (openDialog && !openDialog.contains(target)) return;
        target.focus({ preventScroll: true });
      });
    });
  }, []);
  React.useEffect(() => {
    if (active && !wasActive.current) {
      previousFocus.current = document.activeElement;
      wasActive.current = true;
      return undefined;
    }
    if (!active && wasActive.current) {
      const target = previousFocus.current;
      wasActive.current = false;
      previousFocus.current = null;
      restore(target);
    }
    return undefined;
  }, [active, restore]);
  React.useEffect(() => () => {
    const target = previousFocus.current;
    if (wasActive.current) restore(target);
    wasActive.current = false;
    previousFocus.current = null;
  }, [restore]);
}

function useNativeDialog(dialogRef, open, initialFocusRef) {
  useFocusReturn(open);
  React.useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return undefined;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
    if (!open) return undefined;
    const frame = window.requestAnimationFrame(() => {
      if (initialFocusRef?.current) initialFocusRef.current.focus({ preventScroll: true });
      else focusFirstWithin(dialog);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [dialogRef, initialFocusRef, open]);
}

function ModalFrame({ kind, open, onClose, title, description, closeLabel = 'Close', initialFocusRef, dismissOnBackdrop = true, footer, placement = 'end', size = 'standard', className = '', children }) {
  const dialogRef = React.useRef(null);
  const titleId = React.useId();
  const descriptionId = React.useId();
  useNativeDialog(dialogRef, open, initialFocusRef);
  return (
    <dialog
      ref={dialogRef}
      className={cx('studio-modal', `studio-${kind}`, className)}
      data-placement={kind === 'sheet' ? placement : undefined}
      data-size={size}
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={description ? descriptionId : undefined}
      onCancel={(event) => { event.preventDefault(); onClose?.('escape'); }}
      onClick={(event) => {
        if (dismissOnBackdrop && event.target === event.currentTarget) onClose?.('backdrop');
      }}
    >
      <div className="studio-modal__frame">
        <header className="studio-modal__header">
          <div>
            <h2 id={titleId}>{title}</h2>
            {description ? <p id={descriptionId}>{description}</p> : null}
          </div>
          <IconButton className="studio-modal__close" type="button" aria-label={closeLabel} onClick={() => onClose?.('close-button')}>
            <X size={20} strokeWidth={1.75} aria-hidden="true" />
          </IconButton>
        </header>
        <div className="studio-modal__body">{children}</div>
        {footer ? <footer className="studio-modal__footer">{footer}</footer> : null}
      </div>
    </dialog>
  );
}

export function Sheet(props) {
  return <ModalFrame kind="sheet" {...props} />;
}

export function Dialog(props) {
  return <ModalFrame kind="dialog" {...props} />;
}
