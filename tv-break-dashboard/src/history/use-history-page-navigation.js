import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { addressQuery } from './history-address';
import { KIND_LABELS, readAddress } from './history-labels';

const HISTORY_KIND_PARAM = 'historyKind';

function kindFromLocation() {
  if (typeof window === 'undefined') return '';
  const params = new URLSearchParams(window.location.search);
  const fromEntry = addressQuery(readAddress()).kind;
  const requested = fromEntry || params.get(HISTORY_KIND_PARAM);
  return requested === '' || Object.prototype.hasOwnProperty.call(KIND_LABELS, requested) ? requested : '';
}

export function useHistoryKindNavigation({ locale, requested, setBefore, setSelectedId }) {
  const [kind, setKind] = useState(kindFromLocation);
  const kindTabsRef = useRef([]);

  useEffect(() => {
    function syncFromAddress() {
      const address = readAddress();
      requested.current = address;
      setSelectedId(address);
      setKind(kindFromLocation());
      setBefore('');
    }
    window.addEventListener('popstate', syncFromAddress);
    return () => window.removeEventListener('popstate', syncFromAddress);
  }, [requested, setBefore, setSelectedId]);

  const chooseKind = useCallback((next) => {
    if (next !== '' && !Object.prototype.hasOwnProperty.call(KIND_LABELS, next)) return;
    setBefore('');
    setKind(next);
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    if (next) params.set(HISTORY_KIND_PARAM, next);
    else params.delete(HISTORY_KIND_PARAM);
    params.delete('entry');
    requested.current = '';
    setSelectedId('');
    const search = params.toString();
    window.history.pushState({ workspace: 'history', kind: next }, '', `${window.location.pathname}${search ? `?${search}` : ''}${window.location.hash}`);
  }, [requested, setBefore, setSelectedId]);

  const kindOptions = useMemo(() => ['', ...Object.keys(KIND_LABELS)], []);

  function onKindTabKeyDown(event, index) {
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = kindOptions.length - 1;
    else if (event.key === 'ArrowRight') next = (index + (locale === 'he' ? -1 : 1) + kindOptions.length) % kindOptions.length;
    else if (event.key === 'ArrowLeft') next = (index + (locale === 'he' ? 1 : -1) + kindOptions.length) % kindOptions.length;
    else return;
    event.preventDefault();
    chooseKind(kindOptions[next]);
    kindTabsRef.current[next]?.focus();
  }

  return { chooseKind, kind, kindTabsRef, onKindTabKeyDown, setKind };
}
