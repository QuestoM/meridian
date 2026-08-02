import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { fetchSession } from '../session.js';
import ModelConsole from './console/ModelConsole';

// The context switcher and the console's own root, mounted from this tree.
//
// **Why this is a bridge and not the final wiring, stated here so nobody has to
// guess.** Section 4.7 of the specification puts the switcher in the account
// menu, which lives in `src/shell/side-rail.jsx`, and the console would be a
// case in `src/shell/workspace-router.jsx`. Both files froze at the close of
// wave zero and belong to no piece, so wiring them is an escalation rather than
// a task. Every other wave-one destination inherited an existing route; this
// one is the only destination that does not exist today, so it is the only one
// with nowhere to be mounted from.
//
// What this module does instead is the smallest thing that is still honest: it
// owns its own DOM node, mounts its own React root over the operator shell, and
// renders the switcher exactly as section 4.7 specifies it, naming the
// destination and never the act. Replacing it later is a two-line change in the
// shell and a deletion here.
//
// The affiliation rule is unchanged by the bridge. The switcher renders only
// when `GET /api/auth/me` reports a company account. For a channel account
// there is no control, no disabled state, no tooltip and no route: the hash
// resolves to the operator's own page with no message, so nothing tells a
// channel account the other side exists. The server refuses every route on the
// surface as well, so this is a second line and not the only one.

const CONSOLE_HASH = 'Model';
const NODE_ID = 'kairos-model-console-root';

// The operator destination that carries the audience model's activation switch.
// Section 4.1 makes throwing that switch a run-side act and section 8.2 hands
// the store to P5, so this console mirrors the state and owns no control. The
// mirror still has to be able to REACH the switch, which is what this address
// is for. It is the hash the frozen `src/shell/nav.js` resolves to the Rules
// page, and a test pins that resolution so the name cannot rot into a word.
const RULES_HASH = 'Settings';

// The element the application mounts into, from the frozen `index.html` and
// `src/index.jsx`. It is where the sign-in transition happens, and watching it
// is how this bridge learns that the session changed under it.
const APP_ROOT_ID = 'root';

function appRoot() {
  if (typeof document === 'undefined') return null;
  return document.getElementById(APP_ROOT_ID) || document.body;
}

// The two attributes that put the operator shell out of reach while the console
// is over it, and the strict lookup they are applied to.
//
// `appRoot` falls back to the body because a missing `#root` still has to be
// watched for the auth transition. Nothing may be muted on that fallback: the
// console mounts into the body itself, so muting the body would mute the
// console with it. This lookup returns the shell or nothing.
const SHELL_MUTED = [['inert', ''], ['aria-hidden', 'true']];

function shellRoot() {
  if (typeof document === 'undefined') return null;
  const node = document.getElementById(APP_ROOT_ID);
  if (!node || node.contains(document.getElementById(NODE_ID))) return null;
  return node;
}

function hashIsConsole() {
  if (typeof window === 'undefined') return false;
  return decodeURIComponent(String(window.location.hash || '').replace(/^#/, '')) === CONSOLE_HASH;
}

// The address the page was loaded on, read at module evaluation, which is
// before any component renders. Measured on the running app: the operator shell
// rewrites the hash to its own landing view during boot, so by the time this
// component first renders the hash is already gone and a reload on `#Model`
// landed on the operator's page instead. Reading it once, here, makes the
// console an address a person can bookmark and reload into.
const BOOT_HASH_IS_CONSOLE = hashIsConsole();

// The operator shell writes the chosen locale onto its own root element, so the
// console reads it there rather than fetching the settings a second time.
function currentLocale() {
  if (typeof document === 'undefined') return 'he';
  const shell = document.querySelector('.kairos-shell');
  const value = shell && shell.getAttribute('lang');
  return value === 'en' ? 'en' : 'he';
}

function openConsole() {
  if (typeof window === 'undefined') return;
  window.location.hash = CONSOLE_HASH;
}

function closeConsole() {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  url.hash = '';
  window.history.replaceState(null, '', `${url.pathname}${url.search}`);
  window.dispatchEvent(new HashChangeEvent('hashchange'));
}

function ConsoleBridge() {
  const [company, setCompany] = useState(null);
  const [open, setOpen] = useState(BOOT_HASH_IS_CONSOLE);
  const [locale, setLocale] = useState(currentLocale());
  const alive = useRef(true);
  const asking = useRef(false);
  const askAgain = useRef(false);

  // Asking who this is, once per reason to think the answer changed.
  //
  // An ask that arrives while one is in flight is queued rather than dropped.
  // The in-flight request was sent BEFORE whatever transition triggered this
  // call, so its answer is the stale one and must not be the last word.
  const askSession = useCallback(function ask() {
    if (asking.current) {
      askAgain.current = true;
      return;
    }
    asking.current = true;
    askAgain.current = false;
    fetchSession().then(({ ok, session }) => {
      asking.current = false;
      if (!alive.current) return;
      setCompany(Boolean(ok && session && session.isCompany));
      setLocale(currentLocale());
      if (askAgain.current) ask();
    });
  }, []);

  useEffect(() => {
    alive.current = true;
    askSession();
    return () => { alive.current = false; };
  }, [askSession]);

  // **The session is asked again when the application's own auth state moves.**
  //
  // Measured on the running app before this existed: the bridge asked once, on
  // a React root that is created at boot and never remounts, so an account that
  // signed in through the form after the bridge had already been answered 401
  // never got the switcher at all. Not late, not disabled, not explained:
  // absent, for the whole session, until a manual page reload. The destination's
  // only door disappeared on the most common way into the product.
  //
  // The signal is the application root's own child list. The shell returns the
  // waiting card, the sign-in card or the forced password card INSTEAD of the
  // workspace (`src/shell/TVBreakDashboard.jsx`, `if (authScreen) return
  // authScreen`), and mounts into `#root` (`src/index.jsx`), so every auth
  // transition in either direction is exactly one childList mutation there,
  // and an ordinary re-render inside the workspace is not one at all. That
  // makes this free where a poll would not be, and a test pins the two frozen
  // lines it rests on so the premise cannot rot in silence.
  //
  // The chosen language is watched in the same effect and for the same reason.
  // The shell writes it onto the workspace root as an attribute, and it writes
  // it AFTER that root first appears, because the language comes from the
  // settings read and the settings read has not landed yet. Measured on the
  // running app before this: an English shell got a Hebrew switcher, and kept
  // it, because the answer to "who is this" arrived before the answer to "in
  // which language". A second, narrow observer follows the attribute, attached
  // to whichever workspace root is current.
  useEffect(() => {
    const host = appRoot();
    if (!host || typeof MutationObserver !== 'function') return undefined;
    let langWatch = null;
    const watchLocale = () => {
      if (langWatch) langWatch.disconnect();
      langWatch = null;
      setLocale(currentLocale());
      const shell = document.querySelector('.kairos-shell');
      if (!shell) return;
      langWatch = new MutationObserver(() => setLocale(currentLocale()));
      langWatch.observe(shell, { attributes: true, attributeFilter: ['lang'] });
    };
    const observer = new MutationObserver(() => {
      askSession();
      watchLocale();
    });
    observer.observe(host, { childList: true });
    watchLocale();
    return () => {
      observer.disconnect();
      if (langWatch) langWatch.disconnect();
    };
  }, [askSession]);

  // Signing in happens in another tab often enough to matter, and that tab
  // shares this one's cookie, so coming back is an auth transition too. Asked
  // only while the answer is still not company, so a settled console never
  // pays for it.
  useEffect(() => {
    if (company === true) return undefined;
    const onReturn = () => {
      if (!document.hidden) askSession();
    };
    window.addEventListener('focus', onReturn);
    document.addEventListener('visibilitychange', onReturn);
    return () => {
      window.removeEventListener('focus', onReturn);
      document.removeEventListener('visibilitychange', onReturn);
    };
  }, [askSession, company]);

  // The hash only ever OPENS the console; it never closes it. Measured on the
  // running app: the operator shell rewrites the hash to its own landing view a
  // moment after boot, which under a hash-derived open state dismissed the
  // console a second after it appeared. A different shell is left by its own
  // control, so "Back to the channel" is the only thing that closes this one.
  useEffect(() => {
    function onHash() {
      setLocale(currentLocale());
      if (hashIsConsole()) setOpen(true);
    }
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  // While the console is on screen the address bar says so. The shell's boot
  // rewrite happens after this component's first render, so on a reload into
  // `#Model` the hash has to be put back. It is replaced without dispatching a
  // hashchange, because the shell is underneath and must not be made to
  // re-render for an address it does not own.
  useEffect(() => {
    if (!open || hashIsConsole()) return;
    const url = new URL(window.location.href);
    url.hash = CONSOLE_HASH;
    window.history.replaceState(null, '', url.toString());
  }, [open, company]);

  // **The operator shell is put out of reach while the console covers it.**
  //
  // A different shell that leaves the one underneath it live is a lie to anyone
  // who does not use a mouse. Measured on the live DOM before this: 61 focusable
  // controls outside the console root, two `main` landmarks, and the first
  // focusable element in the whole document was the shell's own overview
  // control, painted over by this overlay, so the first Tab put a keyboard
  // steward on an operator control he could not see and could not leave.
  //
  // `inert` is the one attribute that takes a whole subtree out of the tab
  // order, out of hit testing and out of find-in-page at once; `aria-hidden`
  // takes it out of the accessibility tree, which is the half an engine that
  // does not map the two together would otherwise leave exposed. Both go on
  // together and both come off together, and an attribute the shell had set for
  // itself is restored rather than deleted.
  useEffect(() => {
    if (!open || company !== true) return undefined;
    const shell = shellRoot();
    if (!shell) return undefined;
    const previous = SHELL_MUTED.map(([name]) => [name, shell.getAttribute(name)]);
    SHELL_MUTED.forEach(([name, value]) => shell.setAttribute(name, value));
    return () => {
      previous.forEach(([name, value]) => {
        if (value === null) shell.removeAttribute(name);
        else shell.setAttribute(name, value);
      });
    };
  }, [open, company]);

  const back = useCallback(() => {
    closeConsole();
    setOpen(false);
  }, []);

  // The second way out, and the only one the console's own words promise.
  //
  // The header states that the activation switch lives on Rules and the console
  // deliberately does not carry it. Measured on the live DOM before this
  // existed: the entire header held one control, "Back to the channel", and the
  // sentence naming Rules was a plain note, so a steward who read it had to go
  // find the page himself. Section 3.6 calls a name that reaches nothing a dead
  // end. Leaving is the same two steps back takes, with the address written
  // rather than cleared, so the shell's own hashchange lands the reader there.
  const toRules = useCallback(() => {
    closeConsole();
    setOpen(false);
    window.location.hash = RULES_HASH;
  }, []);

  if (company !== true) return null;
  if (open) {
    return <ModelConsole locale={locale} onBack={back} onOpenRules={toRules} />;
  }
  return (
    <button type="button" className="mc-switcher" onClick={openConsole}>
      {locale === 'en' ? 'Model console (company)' : 'קונסולת המודל (חברה)'}
    </button>
  );
}

let mounted = false;

export function mountModelConsole() {
  if (mounted || typeof document === 'undefined') return;
  mounted = true;
  const attach = () => {
    if (document.getElementById(NODE_ID)) return;
    const node = document.createElement('div');
    node.id = NODE_ID;
    document.body.appendChild(node);
    createRoot(node).render(<ConsoleBridge />);
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach, { once: true });
  } else {
    attach();
  }
}

export default ConsoleBridge;
