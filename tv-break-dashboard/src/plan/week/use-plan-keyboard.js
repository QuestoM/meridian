import { useEffect, useRef } from 'react';

// Keyboard control for Plan, the week.
//
// Linear's rule, and the one worth taking: a shortcut never fires while somebody
// is typing, chords are two plain keys rather than a modifier stack, and every
// binding is also reachable from the palette so nobody has to memorise one to
// work. Measured before this piece, the product had no keyboard control at all.
//
// A chord lead is held for a second and a half, exactly long enough to finish a
// two-key move and short enough that a stray lead never eats the next
// keystroke. The leads are read off the command list itself rather than named
// here, so a command that prints `U then B` in the palette is a command whose
// `u` starts a chord: the list stays the single source of both halves. A single
// key command may not use a letter that leads a chord, and none does.

const CHORD_MS = 1500;

export function chordLeads(commands) {
  const leads = new Set();
  (commands || []).forEach((command) => {
    const shortcut = command.shortcut || [];
    if (shortcut.length === 2 && shortcut[0] !== 'mod') leads.add(String(shortcut[0]).toLowerCase());
  });
  return leads;
}

export function isTypingTarget(target) {
  if (!target) return false;
  const tag = String(target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
  return Boolean(target.isContentEditable);
}

export function usePlanKeyboard({ commands, enabled = true }) {
  const chord = useRef({ lead: null, at: 0 });
  const latest = useRef(commands);
  latest.current = commands;

  useEffect(() => {
    if (!enabled) return undefined;

    function onKeyDown(event) {
      if (event.altKey) return;
      const mod = event.metaKey || event.ctrlKey;
      const key = String(event.key || '').toLowerCase();
      const typing = isTypingTarget(event.target);

      // The palette opens from anywhere, including from inside a text field,
      // because that is the one binding a person reaches for when lost.
      if (mod && key === 'k') {
        event.preventDefault();
        latest.current.find((command) => command.id === 'palette')?.run();
        return;
      }
      if (typing || event.shiftKey) return;

      const now = Date.now();
      const lead = chord.current.lead && now - chord.current.at < CHORD_MS ? chord.current.lead : null;
      if (!mod && !lead && chordLeads(latest.current).has(key)) {
        chord.current = { lead: key, at: now };
        return;
      }

      const match = latest.current.find((command) => {
        const shortcut = command.shortcut || [];
        if (shortcut.length === 2 && shortcut[0] === 'mod') {
          return mod && shortcut[1].toLowerCase() === key;
        }
        if (shortcut.length === 2) {
          return lead === shortcut[0].toLowerCase() && shortcut[1].toLowerCase() === key;
        }
        if (shortcut.length === 1) {
          return !mod && !lead && shortcut[0].toLowerCase() === key;
        }
        return false;
      });
      if (!match || match.disabled) return;
      event.preventDefault();
      chord.current = { lead: null, at: 0 };
      match.run();
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [enabled]);
}

export default usePlanKeyboard;
