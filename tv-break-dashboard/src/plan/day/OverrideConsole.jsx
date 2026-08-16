import React, { useEffect, useRef, useState } from 'react';
import { Button } from '../../studio/actions';
import DayPage from './DayPage';
import OverrideDecisions from './OverrideDecisions';
import { pageText } from '../../shell/format';
import './broadcast-workspace.css';

const TABS = ['day', 'decisions'];

// Plan, at day zoom. The scheduler's door, and the decisions already taken behind it.
//
// This module used to be the Overrides console: a page of levers you visited in
// order to do something to a break that lived somewhere else. A pin is not a
// console you visit, it is something you do to a break, so the board comes
// first and the decisions already taken sit under it as the record of what the
// board is currently obeying.
//
// The console itself moved to OverrideDecisions.jsx unchanged, so everything it
// could do it can still do, including the with-and-without effect preview and
// the verbatim list of the overrides the optimizer refused. The 450-line cap is
// why that was a move rather than an addition here.
function OverrideConsole({ copy, locale, notify, onGlobalRefresh, prefill, onPrefillConsumed, refreshKey }) {
  const [tab, setTab] = useState(() => {
    if (prefill) return 'decisions';
    const addressed = typeof window === 'undefined' ? null : new URLSearchParams(window.location.search).get('broadcast');
    return TABS.includes(addressed) ? addressed : 'day';
  });
  const tabRefs = useRef([]);

  function go(next, { history = true, focus = false } = {}) {
    if (!TABS.includes(next)) return;
    setTab(next);
    if (typeof window !== 'undefined' && history) {
      const url = new URL(window.location.href);
      url.searchParams.set('broadcast', next);
      window.history.pushState({ ...(window.history.state || {}), broadcast: next }, '', `${url.pathname}${url.search}${url.hash}`);
    }
    if (focus) window.setTimeout(() => tabRefs.current[TABS.indexOf(next)]?.focus(), 0);
  }

  useEffect(() => {
    function restore() {
      const addressed = new URLSearchParams(window.location.search).get('broadcast');
      setTab(TABS.includes(addressed) ? addressed : 'day');
    }
    window.addEventListener('popstate', restore);
    return () => window.removeEventListener('popstate', restore);
  }, []);

  useEffect(() => {
    if (prefill) go('decisions');
  }, [prefill]);

  function moveTab(event) {
    const current = TABS.indexOf(tab);
    let next = current;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = TABS.length - 1;
    else if (event.key === 'ArrowRight') next = (current + (locale === 'he' ? -1 : 1) + TABS.length) % TABS.length;
    else if (event.key === 'ArrowLeft') next = (current + (locale === 'he' ? 1 : -1) + TABS.length) % TABS.length;
    else return;
    event.preventDefault();
    go(TABS[next], { focus: true });
  }

  return (
    <div className="broadcast-workspace">
      <nav className="broadcast-local-nav" role="tablist" aria-label={pageText(locale, 'Broadcast workspace', 'מרחב השידור')} onKeyDown={moveTab}>
        {TABS.map((id, index) => {
          const active = id === tab;
          return (
            <Button
              key={id}
              ref={(node) => { tabRefs.current[index] = node; }}
              id={`broadcast-tab-${id}`}
              type="button"
              variant="text"
              disableRipple
              role="tab"
              tabIndex={active ? 0 : -1}
              aria-selected={active}
              aria-controls={`broadcast-panel-${id}`}
              onClick={() => go(id)}
            >
              {id === 'day' ? pageText(locale, 'Day timeline', 'ציר היום') : pageText(locale, 'Manual decisions', 'החלטות ידניות')}
            </Button>
          );
        })}
      </nav>
      <div id={`broadcast-panel-${tab}`} role="tabpanel" aria-labelledby={`broadcast-tab-${tab}`} tabIndex={0}>
        {tab === 'day' ? (
          <DayPage
            locale={locale}
            notify={notify}
            onGlobalRefresh={onGlobalRefresh}
            refreshKey={refreshKey}
          />
        ) : (
          <OverrideDecisions
            copy={copy}
            locale={locale}
            notify={notify}
            onGlobalRefresh={onGlobalRefresh}
            prefill={prefill}
            onPrefillConsumed={onPrefillConsumed}
          />
        )}
      </div>
    </div>
  );
}

export default OverrideConsole;
export { OverrideDecisions };
