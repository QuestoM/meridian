import React, { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { MabatIcon } from '../shell/kairos-icons';
import AssistantPanel from './AssistantPanel';
import { Pressable } from '../studio/dom-controls';
import './kai-shortcuts';
import './assistant-console.css';
import './kai-conversation-head.css';
import './studio-ledger-kai.css';

// The docked assistant column: a real layout sibling of the workspace inside
// the shell flex row, never an overlay, so opening it shrinks the content
// beside it. It sits at the shell's inline-end (opposite the navigation rail,
// which owns inline-start), stays mounted across view switches so the
// conversation continues until a new one is started from the rail, and its
// width is drag-resizable within a clamp and remembered per browser.
//
// The kai-shortcuts import is deliberate and load-bearing rather than
// decorative: the shell imports this file at module scope, so importing the
// shortcut module here is what makes Cmd J work from a screen where the dock is
// closed and no component of Mabat's is mounted. The conversation-head stylesheet
// rides the same fact: it corrects a head that both this dock and the full page
// render, and importing it here puts it on every screen without adding a line to
// two files that are already at the size cap.

const WIDTH_KEY = 'kairos.assistant.dockWidth';
const MIN_WIDTH = 320;
const MAX_WIDTH = 640;
const DEFAULT_WIDTH = 420;
const KEY_STEP = 24;

function clampWidth(value) {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(value)));
}

function savedWidth() {
  try {
    const stored = Number(window.localStorage.getItem(WIDTH_KEY));
    return Number.isFinite(stored) && stored > 0 ? clampWidth(stored) : DEFAULT_WIDTH;
  } catch {
    return DEFAULT_WIDTH;
  }
}

function persistWidth(value) {
  try {
    window.localStorage.setItem(WIDTH_KEY, String(value));
  } catch {
    // localStorage may be unavailable (private mode); the session width still works.
  }
}

export default function AssistantDock({ locale, notify, onClose }) {
  const [width, setWidth] = useState(savedWidth);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef(null);
  const dockRef = useRef(null);
  const returnFocusRef = useRef(null);
  const isRtl = locale === 'he';

  useEffect(() => {
    returnFocusRef.current = document.activeElement;
    const frame = window.requestAnimationFrame(() => dockRef.current?.focus({ preventScroll: true }));
    return () => {
      window.cancelAnimationFrame(frame);
      const target = returnFocusRef.current;
      if (target instanceof HTMLElement && target.isConnected) target.focus({ preventScroll: true });
    };
  }, []);

  // Pointer drag on the grip. The dock sits at inline-end: in RTL it renders
  // leftmost with the grip on its right edge, so moving the pointer toward
  // larger clientX widens it; in LTR the geometry mirrors and the sign flips.
  function onGripPointerDown(event) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStart.current = { x: event.clientX, width };
    setDragging(true);
  }

  function onGripPointerMove(event) {
    if (!dragging || !dragStart.current) return;
    const delta = event.clientX - dragStart.current.x;
    setWidth(clampWidth(dragStart.current.width + (isRtl ? delta : -delta)));
  }

  function onGripPointerUp() {
    if (!dragging) return;
    setDragging(false);
    dragStart.current = null;
    setWidth((current) => {
      persistWidth(current);
      return current;
    });
  }

  // Keyboard resize on the same separator: the arrow pointing away from the
  // content widens the dock, in both directions.
  function onGripKeyDown(event) {
    const grow = isRtl ? 'ArrowRight' : 'ArrowLeft';
    const shrink = isRtl ? 'ArrowLeft' : 'ArrowRight';
    if (event.key !== grow && event.key !== shrink) return;
    event.preventDefault();
    setWidth((current) => {
      const next = clampWidth(current + (event.key === grow ? KEY_STEP : -KEY_STEP));
      persistWidth(next);
      return next;
    });
  }

  return (
    <aside ref={dockRef} tabIndex={-1} className={dragging ? 'asst-dock dragging' : 'asst-dock'} style={{ width }} aria-labelledby="assistant-dock-title">
      <span className="studio-visually-hidden" role="status" aria-live="polite">
        {pageText(locale, 'Mabat assistant panel opened', 'חלונית העוזר מבט נפתחה')}
      </span>
      <div
        className="asst-dock-grip"
        role="separator"
        aria-orientation="vertical"
        aria-label={pageText(locale, 'Resize the assistant panel', 'שינוי רוחב חלונית העוזר')}
        aria-valuenow={width}
        aria-valuemin={MIN_WIDTH}
        aria-valuemax={MAX_WIDTH}
        tabIndex={0}
        onPointerDown={onGripPointerDown}
        onPointerMove={onGripPointerMove}
        onPointerUp={onGripPointerUp}
        onPointerCancel={onGripPointerUp}
        onKeyDown={onGripKeyDown}
      />
      <header className="asst-dock-head">
        <div className="asst-dock-title">
          <span className="asst-dock-mark" aria-hidden="true"><MabatIcon size={17} /></span>
          <div>
            <strong id="assistant-dock-title">{pageText(locale, 'Mabat', 'מבט')}</strong>
            <small>{pageText(locale, 'The Kairos operations assistant', 'העוזר התפעולי של קיירוס')}</small>
          </div>
        </div>
        <Pressable type="button" className="asst-dock-close" onClick={onClose} aria-label={pageText(locale, 'Close the assistant panel', 'סגירת חלונית העוזר')}>
          <X size={15} />
        </Pressable>
      </header>
      <div className="asst-dock-body">
        <AssistantPanel locale={locale} notify={notify} dock />
      </div>
    </aside>
  );
}
