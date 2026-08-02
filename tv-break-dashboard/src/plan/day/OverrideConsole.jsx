import React from 'react';
import DayPage from './DayPage';
import OverrideDecisions from './OverrideDecisions';

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
  return (
    <>
      <DayPage
        locale={locale}
        notify={notify}
        onGlobalRefresh={onGlobalRefresh}
        refreshKey={refreshKey}
      />
      <OverrideDecisions
        copy={copy}
        locale={locale}
        notify={notify}
        onGlobalRefresh={onGlobalRefresh}
        prefill={prefill}
        onPrefillConsumed={onPrefillConsumed}
      />
    </>
  );
}

export default OverrideConsole;
export { OverrideDecisions };
