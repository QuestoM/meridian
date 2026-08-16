// Mabat's keyboard entry, available from every surface.
//
// The reference agent opens on the record with Cmd J and acts where the work
// already is, so Mabat does the same. The dock's open state belongs to the shell,
// which is frozen, and the shell publishes exactly one way in: an #Assistant
// hash opens the dock over whatever page is showing. This module drives that
// published seam and puts the address bar back exactly as it was, so the
// shortcut never navigates and never leaves a URL that would reload elsewhere.
//
// Registration is a module side effect because the dock is only mounted while
// it is open, so a component-scoped listener could close the dock but never
// open it. The shell imports AssistantDock at module scope, AssistantDock
// imports this, so the listener exists from the first paint. It is idempotent:
// a second import binds nothing.

export const OPEN_HASH = 'Assistant';
export const FOCUS_EVENT = 'kai:focus-composer';
export const FOCUS_PENDING = '__kairosKaiFocusPending';
const REGISTERED = '__kairosKaiShortcuts';

export function isDockOpen() {
  return typeof document !== 'undefined' && Boolean(document.querySelector('.asst-dock'));
}

export function focusComposer() {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(FOCUS_EVENT));
}

// Open the dock through the shell's own hash route, then restore the address
// bar. replaceState does not fire hashchange, so the event is dispatched
// explicitly and the restore is silent.
export function openDock() {
  if (typeof window === 'undefined') return;
  const previous = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  try {
    window.history.replaceState(null, '', `#${OPEN_HASH}`);
    const HashChange = window.HashChangeEvent;
    window.dispatchEvent(typeof HashChange === 'function' ? new HashChange('hashchange') : new Event('hashchange'));
  } catch {
    window.location.hash = OPEN_HASH;
    return;
  }
  try {
    window.history.replaceState(null, '', previous);
  } catch {
    // A restricted history is harmless here: the dock is already open and the
    // only cost is an address bar reading #Assistant.
  }
}

export function handleShortcut(event) {
  if (!event || event.key !== 'j' && event.key !== 'J') return false;
  if (!(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey) return false;
  event.preventDefault();
  if (isDockOpen()) {
    focusComposer();
    return true;
  }
  // The panel is not mounted yet, so an event dispatched now would have no
  // listener. A flag the panel reads on mount is the deterministic half; the
  // event stays for the already-open case.
  window[FOCUS_PENDING] = true;
  openDock();
  return true;
}

export function registerShortcuts() {
  if (typeof window === 'undefined' || window[REGISTERED]) return;
  window[REGISTERED] = true;
  window.addEventListener('keydown', handleShortcut);
}

registerShortcuts();
