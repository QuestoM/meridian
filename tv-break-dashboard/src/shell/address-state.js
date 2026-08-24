// Every place the operator can BE has an address, and Back walks them.
//
// The shell already had the whole machine: the hash is the view, params are
// owned per domain (nav.js PARAM_DOMAIN), every history write announces
// kairos:addresschange, and a traversal (Back/Forward) REMOUNTS the workspace
// so components re-read their state from the URL. What was missing is the last
// mile, and it was missing in two ways at once: most feature states lived in
// useState with no param at all, and the few in-place writes used
// replaceState - which OVERWRITES the history entry, so Back skipped the very
// states an operator had just walked through and landed somewhere else
// entirely. That is the exact complaint this module closes.
//
// THE RULE, owner-stated 2026-08-24: every discrete place - page, tab, selected
// advertiser, selected campaign, open drawer - is a pushState, so Back returns
// to it. Continuous input (a filter being typed, a slider mid-drag) is
// replaceState, so history holds places rather than keystrokes. Hover and
// focus are nobody's address and never touch the URL.
//
// Usage inside a component:
//   const [view, setView] = useAddressParam('pacingView', 'board');
//   ... setView('ledger')            // pushState -> a Back entry
//   ... setView(value, {push:false}) // replaceState -> same entry updated
// The param must be registered in nav.js PARAM_DOMAIN (and VALID_VALUES when
// it is an enum) so cross-domain navigation cleans it and normalization
// validates it - an unregistered param survives into foreign domains and
// resurrects stale state there.

import { useCallback, useEffect, useState } from 'react';

export function readAddressParam(name, fallback = '') {
  if (typeof window === 'undefined') return fallback;
  const value = new URLSearchParams(window.location.search).get(name);
  return value === null || value === '' ? fallback : value;
}

export function writeAddressParam(name, value, { push = true } = {}) {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  const current = url.searchParams.get(name) || '';
  const next = value === null || value === undefined ? '' : String(value);
  if (current === next) return;
  if (next === '') url.searchParams.delete(name);
  else url.searchParams.set(name, next);
  const address = `${url.pathname}${url.search}${url.hash}`;
  const state = { ...(window.history.state || {}), kairos: true };
  if (push) window.history.pushState(state, '', address);
  else window.history.replaceState(state, '', address);
}

// One hook, three guarantees: initialized from the address, every set writes
// the address, and an address change made by anyone else (Back, a drawer, the
// assistant) flows back in without a remount.
export function useAddressParam(name, fallback = '', { push = true } = {}) {
  const [value, setValue] = useState(() => readAddressParam(name, fallback));

  useEffect(() => {
    const sync = () => setValue(readAddressParam(name, fallback));
    window.addEventListener('kairos:addresschange', sync);
    window.addEventListener('popstate', sync);
    return () => {
      window.removeEventListener('kairos:addresschange', sync);
      window.removeEventListener('popstate', sync);
    };
    // fallback is a literal at every call site; name never changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  const set = useCallback((next, options = {}) => {
    const resolved = next === null || next === undefined ? '' : String(next);
    writeAddressParam(name, resolved === fallback ? '' : resolved, {
      push: options.push !== undefined ? options.push : push,
    });
    setValue(resolved === '' ? fallback : resolved);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name, fallback, push]);

  return [value, set];
}
