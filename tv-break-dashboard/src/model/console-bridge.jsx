import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Button } from '../studio/actions';
import { fetchSession } from '../session.js';
import ModelConsole from './console/ModelConsole';

// Deprecated compatibility bridge for the standalone Model-console harness.
// The live application does not import this file: ModelConsole now renders as
// one lazy workspace inside the canonical shell. Keeping the exported mount is
// necessary for downstream integrators and regression harnesses that still
// embed the console directly.
const CONSOLE_HASH = 'Model';
const NODE_ID = 'kairos-model-console-root';
const RULES_HASH = 'Settings';
const EVENTS_HASH = 'Calendar';
const APP_ROOT_ID = 'root';

function appRoot() {
  if (typeof document === 'undefined') return null;
  return document.getElementById(APP_ROOT_ID) || document.body;
}

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

const BOOT_HASH_IS_CONSOLE = hashIsConsole();

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

  useEffect(() => {
    function onHash() {
      setLocale(currentLocale());
      if (hashIsConsole()) setOpen(true);
    }
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => {
    if (!open || hashIsConsole()) return;
    const url = new URL(window.location.href);
    url.hash = CONSOLE_HASH;
    window.history.replaceState(null, '', url.toString());
  }, [open, company]);

  useLayoutEffect(() => {
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

  const toRules = useCallback(() => {
    closeConsole();
    setOpen(false);
    window.location.hash = RULES_HASH;
  }, []);

  const toEvents = useCallback(() => {
    closeConsole();
    setOpen(false);
    window.location.hash = EVENTS_HASH;
  }, []);

  if (company !== true) return null;
  if (open) {
    return (
      <ModelConsole
        locale={locale}
        onBack={back}
        onOpenRules={toRules}
        onOpenEvents={toEvents}
      />
    );
  }
  return (
    <Button type="button" className="mc-switcher" onClick={openConsole}>
      {locale === 'en' ? 'Model console (company)' : 'קונסולת המודל (חברה)'}
    </Button>
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
